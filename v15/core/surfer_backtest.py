#!/usr/bin/env python3
"""
Channel Surfer Backtester

Tests the Channel Surfer signal engine against historical 5-min TSLA data.
Walks forward bar-by-bar, evaluates signals, simulates trades, and reports
win rate, profit factor, and other key metrics.

Usage:
    python3 -m v15.core.surfer_backtest [--days 30]
"""

import argparse
import os as _os_mod
import pickle as _pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_SURFER_DATA_CACHE = Path('/tmp/surfer_backtest_data_cache.pkl')


@dataclass
class Trade:
    """Completed trade record."""
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    direction: str       # 'BUY' or 'SELL'
    confidence: float
    stop_pct: float
    tp_pct: float
    exit_reason: str     # 'stop', 'tp', 'timeout', 'signal_flip'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_bars: int = 0
    primary_tf: str = ''
    signal_type: str = 'bounce'  # 'bounce' or 'break'
    trade_size: float = 10000.0
    mae_pct: float = 0.0  # Maximum Adverse Excursion (worst unrealized loss %)
    mfe_pct: float = 0.0  # Maximum Favorable Excursion (best unrealized gain %)
    el_flagged: bool = False  # Was EL flagged at entry
    is_flagged: bool = False  # Was IS flagged at entry
    entry_time: str = ''     # ISO timestamp of entry bar


@dataclass
class OpenPosition:
    """Currently open position."""
    entry_bar: int
    entry_price: float
    direction: str
    confidence: float
    stop_price: float
    tp_price: float
    primary_tf: str
    signal_type: str = 'bounce'     # 'bounce' or 'break'
    trade_size: float = 10000.0     # Confidence-scaled position size
    ou_half_life: float = 5.0
    max_hold_bars: int = 60  # 5 hours max
    trailing_stop: float = 0.0  # Best price seen for trailing
    worst_price: float = 0.0    # Worst price seen (for MAE)
    best_price: float = 0.0     # Best price seen (for MFE)
    el_flagged: bool = False    # Extreme Loser flagged — trail more aggressively
    fast_reversion: bool = False # Fast reversion detected — bounce trail tighter
    is_flagged: bool = False    # Immediate Stop flagged
    extended: bool = False       # Hold time extended on profitable timeout
    trail_width_mult: float = 1.0  # Arch 61: widen trail for predicted extended runs


@dataclass
class BacktestMetrics:
    """Summary metrics."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_hold_bars: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.total_trades, 1)

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / max(abs(self.gross_loss), 1e-6)

    @property
    def expectancy(self) -> float:
        """Expected $ per trade."""
        return self.total_pnl / max(self.total_trades, 1)

    def summary(self) -> str:
        return (
            f"Trades: {self.total_trades} | Win Rate: {self.win_rate:.0%} | "
            f"PF: {self.profit_factor:.2f} | Total P&L: ${self.total_pnl:,.2f} | "
            f"Avg Hold: {self.avg_hold_bars:.0f} bars | "
            f"Avg Win: {self.avg_win_pct:.2%} | Avg Loss: {self.avg_loss_pct:.2%} | "
            f"Max DD: {self.max_drawdown_pct:.1%} | "
            f"Expectancy: ${self.expectancy:,.2f}/trade"
        )


def _check_position_exit(position: OpenPosition, bar: int, current_price: float,
                          window_high: float, window_low: float,
                          eval_interval: int) -> Optional[Tuple[str, float]]:
    """
    Check if a position should exit. Returns (exit_reason, exit_price) or None.
    Also updates position's trailing stop and MAE/MFE tracking in-place.
    """
    bars_held = bar - position.entry_bar

    # Track MAE/MFE
    if position.direction == 'BUY':
        if position.worst_price == 0 or window_low < position.worst_price:
            position.worst_price = window_low
        if window_high > position.best_price:
            position.best_price = window_high
    else:
        if position.worst_price == 0 or window_high > position.worst_price:
            position.worst_price = window_high
        if position.best_price == 0 or window_low < position.best_price:
            position.best_price = window_low

    entry = position.entry_price
    tp_dist = abs(position.tp_price - entry) / entry
    initial_stop_dist = abs(position.stop_price - entry) / entry
    is_breakout = position.signal_type == 'break'

    # Arch 61: Extended Run Predictor widens trail for predicted runners
    # trail_width_mult > 1.0 = wider trail (let winners run)
    # trail_width_mult < 1.0 = tighter trail (capture quickly)
    twm = position.trail_width_mult

    # EL-flagged trades get more aggressive trailing to lock profits sooner
    # ML-guided trail adjustments
    el = position.el_flagged
    fast_rev = position.fast_reversion and not is_breakout

    if position.direction == 'BUY':
        if window_high > position.trailing_stop:
            position.trailing_stop = window_high

        if is_breakout:
            profit_from_best = (position.trailing_stop - entry) / entry
            # Ultra-tight stop breakeven: if stop < 0.1%, protect at tiny profit
            if initial_stop_dist < 0.001 and profit_from_best > 0.0001:
                trail_from_best = position.trailing_stop * (1 - initial_stop_dist * 0.50 * twm)
                effective_stop = max(position.stop_price, trail_from_best)
            # Three-tier breakout trail
            elif profit_from_best > 0.015:
                trail_from_best = position.trailing_stop * (1 - initial_stop_dist * 0.01 * twm)
                effective_stop = max(position.stop_price, trail_from_best)
            elif profit_from_best > 0.008:
                trail_from_best = position.trailing_stop * (1 - initial_stop_dist * 0.02 * twm)
                effective_stop = max(position.stop_price, trail_from_best)
            else:
                tier3_thresh = 0.002 if el else 0.0008
                trail_mult = 0.20 if el else 0.01
                if profit_from_best > tier3_thresh:
                    trail_from_best = position.trailing_stop * (1 - initial_stop_dist * trail_mult * twm)
                    effective_stop = max(position.stop_price, trail_from_best)
                else:
                    effective_stop = position.stop_price
        else:
            profit_from_entry = (position.trailing_stop - entry) / entry
            profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
            # EL: lower thresholds and tighter trails
            # Fast reversion: even tighter — mean reversion resolves quickly
            tight = el or fast_rev
            if profit_ratio >= 0.80:
                # Near TP: ultra-tight trail to lock in most of the move
                trail_from_best = position.trailing_stop * (1 - initial_stop_dist * 0.005 * twm)
                effective_stop = max(position.stop_price, trail_from_best)
            elif profit_ratio >= (0.60 if tight else 0.55):
                trail_from_best = position.trailing_stop * (1 - initial_stop_dist * (0.02 if tight else 0.02) * twm)
                effective_stop = max(position.stop_price, trail_from_best)
            elif profit_ratio >= (0.30 if tight else 0.40):
                trail_from_best = position.trailing_stop * (1 - initial_stop_dist * (0.08 if tight else 0.06) * twm)
                effective_stop = max(position.stop_price, trail_from_best)
            elif profit_ratio >= (0.10 if tight else 0.15):
                effective_stop = max(position.stop_price, entry * 1.0005)
            else:
                effective_stop = position.stop_price

        if window_low <= effective_stop:
            reason = 'stop' if effective_stop == position.stop_price else 'trail'
            return (reason, effective_stop)
        elif window_high >= position.tp_price:
            return ('tp', position.tp_price)
        elif not is_breakout and bars_held >= max(6, int(position.ou_half_life * 3)):
            return ('ou_timeout', current_price)

    else:  # SELL
        if position.trailing_stop == 0 or window_low < position.trailing_stop:
            position.trailing_stop = window_low

        if is_breakout:
            profit_from_best = (entry - position.trailing_stop) / entry
            # Ultra-tight stop breakeven: if stop < 0.1%, protect at tiny profit
            if initial_stop_dist < 0.001 and profit_from_best > 0.0001:
                trail_from_best = position.trailing_stop * (1 + initial_stop_dist * 0.50 * twm)
                effective_stop = min(position.stop_price, trail_from_best)
            # Three-tier breakout trail
            elif profit_from_best > 0.015:
                trail_from_best = position.trailing_stop * (1 + initial_stop_dist * 0.01 * twm)
                effective_stop = min(position.stop_price, trail_from_best)
            elif profit_from_best > 0.008:
                trail_from_best = position.trailing_stop * (1 + initial_stop_dist * 0.02 * twm)
                effective_stop = min(position.stop_price, trail_from_best)
            else:
                tier3_thresh = 0.002 if el else 0.0003
                trail_mult = 0.20 if el else 0.01
                if profit_from_best > tier3_thresh:
                    trail_from_best = position.trailing_stop * (1 + initial_stop_dist * trail_mult * twm)
                    effective_stop = min(position.stop_price, trail_from_best)
                else:
                    effective_stop = position.stop_price
        else:
            profit_from_entry = (entry - position.trailing_stop) / entry
            profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
            tight = el or fast_rev
            if profit_ratio >= 0.80:
                trail_from_best = position.trailing_stop * (1 + initial_stop_dist * 0.005 * twm)
                effective_stop = min(position.stop_price, trail_from_best)
            elif profit_ratio >= (0.60 if tight else 0.55):
                trail_from_best = position.trailing_stop * (1 + initial_stop_dist * (0.02 if tight else 0.02) * twm)
                effective_stop = min(position.stop_price, trail_from_best)
            elif profit_ratio >= (0.30 if tight else 0.40):
                trail_from_best = position.trailing_stop * (1 + initial_stop_dist * (0.08 if tight else 0.06) * twm)
                effective_stop = min(position.stop_price, trail_from_best)
            elif profit_ratio >= (0.10 if tight else 0.15):
                effective_stop = min(position.stop_price, entry * 0.9995)
            else:
                effective_stop = position.stop_price

        if window_high >= effective_stop:
            reason = 'stop' if effective_stop == position.stop_price else 'trail'
            return (reason, effective_stop)
        elif window_low <= position.tp_price:
            return ('tp', position.tp_price)
        elif not is_breakout and bars_held >= max(6, int(position.ou_half_life * 3)):
            return ('ou_timeout', current_price)

    if bars_held >= position.max_hold_bars:
        return ('timeout', current_price)

    return None


def _extract_signal_features(analysis, tsla, bar, closes, spy_df, vix_df,
                              feature_names, history_buffer, eval_interval):
    """Extract full ML feature vector at signal time.

    Shared by both the ML-enhanced backtest path and the feature-capture path.
    Mutates history_buffer in-place (appends current snapshot, trims to 20).

    Returns:
        (feature_vec, snapshot_dict)
    """
    from v15.core.surfer_ml import (
        extract_tf_features, extract_cross_tf_features,
        extract_context_features, extract_correlation_features,
        extract_temporal_features,
        ML_TFS, PER_TF_FEATURES,
        CROSS_TF_FEATURES, CONTEXT_FEATURES, CORRELATION_FEATURES,
        TEMPORAL_FEATURES,
    )

    num_features = len(feature_names)
    feature_vec = np.zeros(num_features, dtype=np.float32)
    offset = 0

    for tf in ML_TFS:
        state = analysis.tf_states.get(tf)
        if state:
            tf_feats = extract_tf_features(state)
        else:
            tf_feats = np.zeros(len(PER_TF_FEATURES), dtype=np.float32)
        feature_vec[offset:offset + len(PER_TF_FEATURES)] = tf_feats
        offset += len(PER_TF_FEATURES)

    cross_feats = extract_cross_tf_features(analysis.tf_states)
    feature_vec[offset:offset + len(CROSS_TF_FEATURES)] = cross_feats
    offset += len(CROSS_TF_FEATURES)

    ctx_feats = extract_context_features(tsla, bar)
    feature_vec[offset:offset + len(CONTEXT_FEATURES)] = ctx_feats
    offset += len(CONTEXT_FEATURES)

    bt_snapshot = {}
    for tf in ML_TFS:
        state = analysis.tf_states.get(tf)
        if state and state.valid:
            for feat_name in PER_TF_FEATURES:
                val = getattr(state, feat_name, 0.0)
                if isinstance(val, (int, float)):
                    bt_snapshot[f'{tf}_{feat_name}'] = float(val)
    bt_snapshot['rsi_14'] = float(ctx_feats[0])
    bt_snapshot['volume_ratio_20'] = float(ctx_feats[2])

    temporal_feats = extract_temporal_features(
        bt_snapshot, history_buffer,
        closes=closes, bar_idx=bar, eval_interval=eval_interval,
    )
    feature_vec[offset:offset + len(TEMPORAL_FEATURES)] = temporal_feats
    offset += len(TEMPORAL_FEATURES)

    history_buffer.append(bt_snapshot)
    if len(history_buffer) > 20:
        history_buffer.pop(0)

    corr_feats = extract_correlation_features(
        bar, closes, spy_df=spy_df, vix_df=vix_df,
        tsla_index=tsla.index,
    )
    feature_vec[offset:offset + len(CORRELATION_FEATURES)] = corr_feats

    # Safety: replace NaN/inf with 0 (physics engine can occasionally produce them)
    np.nan_to_num(feature_vec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_vec, bt_snapshot


def run_backtest(
    days: int = 30,
    eval_interval: int = 3,     # Check every 3 bars = 15 min
    max_hold_bars: int = 60,    # Max 5 hours (60 * 5min)
    position_size: float = 10000.0,  # $10k per trade
    min_confidence: float = 0.01,  # Minimal gate: hard skips (EL, volume, IS) do the real filtering
    use_multi_tf: bool = True,  # Use higher TF data for context
    ml_model=None,              # Optional ML model for signal enhancement
    # Pre-loaded data (skip yfinance when provided)
    tsla_df: 'pd.DataFrame | None' = None,
    higher_tf_dict: 'dict | None' = None,
    spy_df_input: 'pd.DataFrame | None' = None,
    vix_df_input: 'pd.DataFrame | None' = None,
    # Realistic mode constraints
    realistic: bool = False,
    slippage_bps: float = 3.0,          # 3 basis points per side
    commission_per_share: float = 0.005, # $0.005/share round trip
    max_leverage: float = 4.0,
    bounce_cap: float = 12.0,           # Max exposure cap multiplier for bounce signals (all-weather: 4-12x validated 11/11 years 2015-2025; Arch384: 8x→12x +$160K est 11yr)
    max_trade_usd: float = 0.0,         # Hard dollar cap per trade (0 = unlimited). Set to e.g. 1e6 for realistic market-capacity simulation.
    initial_capital: float = 0.0,       # 0 = use position_size * 10
    capture_features: bool = False,     # Save ML feature vectors per trade
    signal_quality_model=None,          # SignalQualityModel for ML position sizing
    ml_size_fn=None,                    # Callable(quality_score) -> scale_factor
) -> tuple:
    """
    Run Channel Surfer backtest on historical 5-min TSLA data.

    If ml_model is provided, uses ML predictions to:
    - Filter out signals where ML predicts HOLD
    - Boost confidence when ML agrees with physics
    - Adjust stop/TP based on predicted channel lifetime
    - Skip trades where ML predicts imminent break in wrong direction

    If capture_features is True, extracts and saves the 169-dim ML feature
    vector for each trade (used by signal quality model training).

    Returns:
        (metrics, trades, equity_curve) or
        (metrics, trades, equity_curve, trade_features, trade_signals) when capture_features=True
    """
    import yfinance as yf
    from v15.core.channel import detect_channels_multi_window, select_best_channel
    from v15.core.channel_surfer import analyze_channels, SIGNAL_TFS, TF_WINDOWS

    # Silent failure tracker — counts errors per category, prints summary at end
    _error_counts = {}
    def _track_error(category: str, err: Exception):
        _error_counts.setdefault(category, {'count': 0, 'first_err': str(err)})
        _error_counts[category]['count'] += 1

    ml_active = ml_model is not None
    quality_scorer = None
    ensemble_model = None
    ensemble_base_models = {}
    breakout_momentum_model = None
    extended_run_model = None
    breakout_fade_model = None
    follow_through_model = None
    ml_stats = {'total_signals': 0, 'ml_filtered': 0, 'ml_boosted': 0, 'ml_agreed': 0,
                 'quality_filtered': 0, 'quality_boosted': 0, 'ensemble_filtered': 0,
                 'conf_below_min': 0, 'not_buy_sell': 0, 'circuit_breaker': 0,
                 'anti_pyramid': 0, 'anti_double_type': 0, 'low_volume': 0,
                 'pos_score_weak': 0, 'low_conf_buy_bounce': 0, 'leverage_cap': 0}
    if not ml_active:
        print("⚠️  WARNING: ML MODEL NOT LOADED — trading without ML filtering!")
        print("⚠️  All signals will pass through unfiltered by ML models.")
        print("⚠️  This is ONLY acceptable for backtesting baseline comparisons.")
    if ml_active:
        from v15.core.surfer_ml import (
            extract_tf_features, extract_cross_tf_features,
            extract_context_features, extract_correlation_features,
            extract_temporal_features, TradeQualityScorer,
            EnsembleModel, GBTModel, MultiTFTransformer, SurvivalModel,
            RegimeConditionalModel, TrendGBTModel, CVEnsembleModel, PhysicsResidualModel, AdverseMovementPredictor, CompositeSignalScorer, VolatilityTransitionModel, ExitTimingOptimizer, MomentumExhaustionDetector, CrossAssetAmplifier, StopLossPredictor, DynamicTrailOptimizer, IntradaySessionModel, ChannelMaturityPredictor, ReturnAsymmetryPredictor, GapRiskPredictor, MeanReversionSpeedModel, LiquidityStateClassifier, TradeDurationPredictor, AdversarialTradeSelector, QuantileRiskEstimator, TailRiskDetector, StopDistanceOptimizer, VolatilityClusteringPredictor, ExtremeLoserDetector, DrawdownMagnitudePredictor, WinStreakDetector, FeatureInteractionLoser, BounceLoserDetector, MomentumReversalDetector, ImmediateStopDetector, ProfitVelocityPredictor, BreakoutStopPredictor, BreakoutMomentumValidator, ExtendedRunPredictor, BreakoutFadeDetector, MomentumFollowThrough,
            get_feature_names, ML_TFS, PER_TF_FEATURES,
            CROSS_TF_FEATURES, CONTEXT_FEATURES, CORRELATION_FEATURES,
            TEMPORAL_FEATURES,
        )
        ml_feature_names = get_feature_names()
        ml_history_buffer: List[Dict] = []
        ml_feature_window: List[np.ndarray] = []  # For TrendGBT sliding window
        print(f"[ML] Model loaded with {len(ml_feature_names)} features")

        # Try to load quality scorer
        import os as _os
        model_dir = _os.path.join(_os.path.dirname(__file__), '..', '..', 'surfer_models')
        if not _os.path.isdir(model_dir):
            model_dir = 'surfer_models'

        quality_path = _os.path.join(model_dir, 'trade_quality_scorer.pkl')
        if _os.path.exists(quality_path):
            try:
                quality_scorer = TradeQualityScorer.load(quality_path)
                print(f"[ML] Quality scorer loaded")
            except Exception as _e:
                _track_error("model_load", _e)

        # Try to load ensemble + base models
        ens_path = _os.path.join(model_dir, 'ensemble_model.pkl')
        if _os.path.exists(ens_path):
            try:
                ensemble_model = EnsembleModel.load(ens_path)
                print(f"[ML] Ensemble meta-learner loaded")

                # Load base models for ensemble
                gbt_path = _os.path.join(model_dir, 'gbt_model.pkl')
                if _os.path.exists(gbt_path):
                    ensemble_base_models['gbt'] = GBTModel.load(gbt_path)

                trans_path = _os.path.join(model_dir, 'transformer_model.pt')
                if _os.path.exists(trans_path):
                    try:
                        ensemble_base_models['transformer'] = MultiTFTransformer.load(trans_path)
                    except Exception as _e:
                        _track_error("load_transformer", _e)

                surv_path = _os.path.join(model_dir, 'survival_model.pt')
                if _os.path.exists(surv_path):
                    try:
                        ensemble_base_models['survival'] = SurvivalModel.load(surv_path)
                    except Exception as _e:
                        _track_error("load_survival", _e)

                if quality_scorer:
                    ensemble_base_models['quality'] = quality_scorer

                print(f"[ML] Ensemble base models: {list(ensemble_base_models.keys())}")
            except Exception:
                ensemble_model = None

        # Try to load regime model
        regime_model = None
        regime_path = _os.path.join(model_dir, 'regime_model.pkl')
        if _os.path.exists(regime_path):
            try:
                regime_model = RegimeConditionalModel.load(regime_path)
                print(f"[ML] Regime model loaded (regime-augmented)")
                ml_stats['regime_boosted'] = 0
                ml_stats['regime_penalized'] = 0
            except Exception as _e:
                _track_error("load_regime", _e)

        # Try to load TrendGBT model
        trend_gbt_model = None
        tg_path = _os.path.join(model_dir, 'trend_gbt_model.pkl')
        if _os.path.exists(tg_path):
            try:
                trend_gbt_model = TrendGBTModel.load(tg_path)
                print(f"[ML] TrendGBT loaded (top-{trend_gbt_model.TOP_K} features + trends)")
                ml_stats['trend_gbt_confirmed'] = 0
                ml_stats['trend_gbt_filtered'] = 0
            except Exception as _e:
                _track_error("load_trend_gbt", _e)

        # Try to load CV Ensemble model
        cv_ensemble_model = None
        cv_path = _os.path.join(model_dir, 'cv_ensemble_model.pkl')
        if _os.path.exists(cv_path):
            try:
                cv_ensemble_model = CVEnsembleModel.load(cv_path)
                print(f"[ML] CV Ensemble loaded ({cv_ensemble_model.N_FOLDS}-fold)")
                ml_stats['cv_high_consensus'] = 0
                ml_stats['cv_low_consensus'] = 0
            except Exception as _e:
                _track_error("load_cv_ensemble", _e)

        # Try to load Physics-Residual model
        residual_model = None
        res_path = _os.path.join(model_dir, 'physics_residual_model.pkl')
        if _os.path.exists(res_path):
            try:
                residual_model = PhysicsResidualModel.load(res_path)
                print(f"[ML] Physics-Residual model loaded")
                ml_stats['residual_boosted'] = 0
                ml_stats['residual_penalized'] = 0
                ml_stats['residual_lifetime_adj'] = 0
            except Exception as _e:
                _track_error("model_load", _e)

        # Try to load Adverse Movement model
        adverse_model = None
        adv_path = _os.path.join(model_dir, 'adverse_movement_model.pkl')
        if _os.path.exists(adv_path):
            try:
                adverse_model = AdverseMovementPredictor.load(adv_path)
                print(f"[ML] Adverse Movement model loaded")
                ml_stats['adverse_filtered'] = 0
                ml_stats['adverse_boosted'] = 0
            except Exception as _e:
                _track_error("load_adverse", _e)

        # Try to load Composite Signal Scorer
        composite_model = None
        comp_path = _os.path.join(model_dir, 'composite_scorer.pkl')
        if _os.path.exists(comp_path):
            try:
                composite_model = CompositeSignalScorer.load(comp_path)
                print(f"[ML] Composite scorer loaded ({len(composite_model.meta_feature_names or [])} meta-features)")
                ml_stats['composite_agreed'] = 0
                ml_stats['composite_filtered'] = 0
            except Exception as _e:
                _track_error("load_composite", _e)

        # Try to load Volatility Transition model
        vol_model = None
        vol_path = _os.path.join(model_dir, 'vol_transition_model.pkl')
        if _os.path.exists(vol_path):
            try:
                vol_model = VolatilityTransitionModel.load(vol_path)
                print(f"[ML] Volatility Transition model loaded")
                ml_stats['vol_danger_skip'] = 0
                ml_stats['vol_warning_scale'] = 0
                ml_stats['vol_calm_boost'] = 0
            except Exception as _e:
                _track_error("model_load", _e)

        # Try to load Exit Timing model
        exit_timing_model = None
        exit_path = _os.path.join(model_dir, 'exit_timing_opt.pkl')
        if _os.path.exists(exit_path):
            try:
                exit_timing_model = ExitTimingOptimizer.load(exit_path)
                print(f"[ML] Exit Timing model loaded")
                ml_stats['exit_tightened'] = 0
                ml_stats['exit_early'] = 0
            except Exception as _e:
                _track_error("load_exit_timing", _e)

        # Try to load Momentum Exhaustion model
        exhaustion_model = None
        exh_path = _os.path.join(model_dir, 'exhaustion_model.pkl')
        if _os.path.exists(exh_path):
            try:
                exhaustion_model = MomentumExhaustionDetector.load(exh_path)
                print(f"[ML] Momentum Exhaustion model loaded")
                ml_stats['exh_exhausted_skip'] = 0
                ml_stats['exh_tiring_scale'] = 0
                ml_stats['exh_fresh_boost'] = 0
            except Exception as _e:
                _track_error("model_load", _e)

        # Try to load Cross-Asset Amplifier model
        cross_asset_model = None
        ca_path = _os.path.join(model_dir, 'cross_asset_model.pkl')
        if _os.path.exists(ca_path):
            try:
                cross_asset_model = CrossAssetAmplifier.load(ca_path)
                print(f"[ML] Cross-Asset Amplifier loaded")
                ml_stats['ca_rotation_boost'] = 0
                ml_stats['ca_selloff_skip'] = 0
                ml_stats['ca_scale_applied'] = 0
            except Exception as _e:
                _track_error("model_load", _e)

        # Try to load Stop Loss Predictor
        stop_loss_model = None
        sl_path = _os.path.join(model_dir, 'stop_loss_model.pkl')
        if _os.path.exists(sl_path):
            try:
                stop_loss_model = StopLossPredictor.load(sl_path)
                print(f"[ML] Stop Loss Predictor loaded")
                ml_stats['sl_danger_skip'] = 0
                ml_stats['sl_caution_scale'] = 0
                ml_stats['sl_safe_boost'] = 0
            except Exception as _e:
                _track_error("model_load", _e)

        # Try to load Dynamic Trail Optimizer
        trail_model = None
        trail_path = _os.path.join(model_dir, 'trail_optimizer.pkl')
        if _os.path.exists(trail_path):
            try:
                trail_model = DynamicTrailOptimizer.load(trail_path)
                print(f"[ML] Dynamic Trail Optimizer loaded (AUC 0.700)")
                ml_stats['trail_tightened'] = 0
                ml_stats['trail_loosened'] = 0
            except Exception as _e:
                _track_error("load_trail", _e)

        # Architecture 21: Intraday Session Model
        session_model = None
        session_path = _os.path.join(model_dir, 'session_model.pkl')
        if _os.path.exists(session_path):
            try:
                session_model = IntradaySessionModel.load(session_path)
                print(f"[ML] Intraday Session Model loaded (Quality AUC 0.648)")
                ml_stats['session_boost'] = 0
                ml_stats['session_penalty'] = 0
            except Exception as _e:
                _track_error("load_session", _e)

        # Architecture 22: Channel Maturity Predictor
        maturity_model = None
        maturity_path = _os.path.join(model_dir, 'maturity_model.pkl')
        if _os.path.exists(maturity_path):
            try:
                maturity_model = ChannelMaturityPredictor.load(maturity_path)
                print(f"[ML] Channel Maturity Predictor loaded (AUC 0.677)")
                ml_stats['maturity_skip'] = 0
                ml_stats['maturity_boost'] = 0
            except Exception as _e:
                _track_error("load_maturity", _e)

        # Architecture 24: Return Asymmetry Predictor
        asymmetry_model = None
        asymmetry_path = _os.path.join(model_dir, 'asymmetry_model.pkl')
        if _os.path.exists(asymmetry_path):
            try:
                asymmetry_model = ReturnAsymmetryPredictor.load(asymmetry_path)
                print(f"[ML] Return Asymmetry Predictor loaded (Spike AUC 0.680)")
                ml_stats['asym_widen_stop'] = 0
                ml_stats['asym_tighten_trail'] = 0
            except Exception as _e:
                _track_error("load_asymmetry", _e)

        # Architecture 25: Gap Risk Predictor
        gap_risk_model = None
        gap_path = _os.path.join(model_dir, 'gap_risk_model.pkl')
        if _os.path.exists(gap_path):
            try:
                gap_risk_model = GapRiskPredictor.load(gap_path)
                print(f"[ML] Gap Risk Predictor loaded (AUC 0.852)")
                ml_stats['gap_risk_skip'] = 0
            except Exception as _e:
                _track_error("load_gap_risk", _e)

        # Architecture 26: Mean Reversion Speed
        reversion_model = None
        rev_path = _os.path.join(model_dir, 'reversion_model.pkl')
        if _os.path.exists(rev_path):
            try:
                reversion_model = MeanReversionSpeedModel.load(rev_path)
                print(f"[ML] Mean Reversion Speed loaded (AUC 0.873)")
                ml_stats['rev_fast_boost'] = 0
                ml_stats['rev_slow_penalty'] = 0
            except Exception as _e:
                _track_error("load_reversion", _e)

        # Architecture 27: Liquidity State (slippage risk only)
        liquidity_model = None
        liq_path = _os.path.join(model_dir, 'liquidity_model.pkl')
        if _os.path.exists(liq_path):
            try:
                liquidity_model = LiquidityStateClassifier.load(liq_path)
                print(f"[ML] Liquidity State loaded (slippage corr 0.499)")
                ml_stats['liq_high_slippage'] = 0
            except Exception as _e:
                _track_error("load_liquidity", _e)

        # Architecture 31: Trade Duration Predictor
        duration_model = None
        dur_path = _os.path.join(model_dir, 'duration_model.pkl')
        if _os.path.exists(dur_path):
            try:
                duration_model = TradeDurationPredictor.load(dur_path)
                print(f"[ML] Trade Duration loaded (Quick AUC 0.617)")
                ml_stats['dur_quick_exit'] = 0
                ml_stats['dur_extend_hold'] = 0
            except Exception as _e:
                _track_error("load_duration", _e)

        # Architecture 37: Adversarial Trade Selector
        adversarial_model = None
        adv_path = _os.path.join(model_dir, 'adversarial_model.pkl')
        if _os.path.exists(adv_path):
            try:
                adversarial_model = AdversarialTradeSelector.load(adv_path)
                print(f"[ML] Adversarial Selector loaded (AUC 0.605)")
                ml_stats['adv_favorable_boost'] = 0
                ml_stats['adv_unfavorable_penalty'] = 0
            except Exception as _e:
                _track_error("load_adversarial", _e)

        # Architecture 40: Quantile Risk Estimator
        quantile_risk_model = None
        qr_path = _os.path.join(model_dir, 'quantile_risk_model.pkl')
        if _os.path.exists(qr_path):
            try:
                quantile_risk_model = QuantileRiskEstimator.load(qr_path)
                print(f"[ML] Quantile Risk loaded (spread corr 0.298)")
                ml_stats['qr_high_risk_penalty'] = 0
                ml_stats['qr_favorable_asym'] = 0
            except Exception as _e:
                _track_error("load_quantile_risk", _e)

        # Architecture 41: Tail Risk Detector
        tail_risk_model = None
        tr_path = _os.path.join(model_dir, 'tail_risk_model.pkl')
        if _os.path.exists(tr_path):
            try:
                tail_risk_model = TailRiskDetector.load(tr_path)
                print(f"[ML] Tail Risk Detector loaded (AUC 0.743)")
                ml_stats['tail_bounce_penalty'] = 0
                ml_stats['tail_break_boost'] = 0
            except Exception as _e:
                _track_error("load_tail_risk", _e)

        # Architecture 43: Stop Distance Optimizer
        stop_dist_model = None
        sd_path = _os.path.join(model_dir, 'stop_distance_model.pkl')
        if _os.path.exists(sd_path):
            try:
                stop_dist_model = StopDistanceOptimizer.load(sd_path)
                print(f"[ML] Stop Distance loaded (MAE corr 0.346)")
                ml_stats['sd_wide_stop'] = 0
                ml_stats['sd_tight_stop'] = 0
            except Exception as _e:
                _track_error("load_stop_dist", _e)

        # Architecture 44: Volatility Clustering
        vol_cluster_model = None
        vcl_path = _os.path.join(model_dir, 'vol_clustering_model.pkl')
        if _os.path.exists(vcl_path):
            try:
                vol_cluster_model = VolatilityClusteringPredictor.load(vcl_path)
                print(f"[ML] Vol Clustering loaded (AUC 0.683)")
                ml_stats['vc_vol_inc_penalty'] = 0
                ml_stats['vc_vol_dec_boost'] = 0
            except Exception as _e:
                _track_error("load_vol_cluster", _e)

        # Architecture 45: Extreme Loser Detector
        extreme_loser_model = None
        el_path = _os.path.join(model_dir, 'extreme_loser_model.pkl')
        if _os.path.exists(el_path):
            try:
                extreme_loser_model = ExtremeLoserDetector.load(el_path)
                print(f"[ML] Extreme Loser Detector loaded (AUC 0.654)")
                ml_stats['el_penalty'] = 0
                ml_stats['el_skip'] = 0
            except Exception as _e:
                _track_error("load_extreme_loser", _e)

        # Architecture 49: Drawdown Magnitude Predictor
        drawdown_mag_model = None
        dm_path = _os.path.join(model_dir, 'drawdown_magnitude_model.pkl')
        if _os.path.exists(dm_path):
            try:
                drawdown_mag_model = DrawdownMagnitudePredictor.load(dm_path)
                print(f"[ML] Drawdown Magnitude loaded (P75 corr 0.283)")
                ml_stats['dm_high_dd_pen'] = 0
                ml_stats['dm_low_dd_boost'] = 0
            except Exception as _e:
                _track_error("load_drawdown_mag", _e)

        # Architecture 50: Win Streak Detector
        win_streak_model = None
        ws_path = _os.path.join(model_dir, 'win_streak_model.pkl')
        if _os.path.exists(ws_path):
            try:
                win_streak_model = WinStreakDetector.load(ws_path)
                print(f"[ML] Win Streak Detector loaded (AUC 0.620)")
                ml_stats['ws_boost'] = 0
            except Exception as _e:
                _track_error("load_win_streak", _e)

        # Architecture 54: Bounce Loser Detector
        bounce_loser_model = None
        bl_path = _os.path.join(model_dir, 'bounce_loser_model.pkl')
        if _os.path.exists(bl_path):
            try:
                bounce_loser_model = BounceLoserDetector.load(bl_path)
                print(f"[ML] Bounce Loser Detector loaded (AUC 0.670)")
                ml_stats['bl_penalty'] = 0
            except Exception as _e:
                _track_error("load_bounce_loser", _e)

        # Architecture 55: Feature Interaction Loser
        feat_int_model = None
        fi_path = _os.path.join(model_dir, 'feature_interaction_model.pkl')
        if _os.path.exists(fi_path):
            try:
                feat_int_model = FeatureInteractionLoser.load(fi_path)
                print(f"[ML] Feature Interaction Loser loaded (AUC 0.733)")
                ml_stats['fi_penalty'] = 0
            except Exception as _e:
                _track_error("load_feat_int", _e)

        # Architecture 56: Momentum Reversal Detector
        mom_rev_model = None
        mr_path = _os.path.join(model_dir, 'momentum_reversal_model.pkl')
        if _os.path.exists(mr_path):
            try:
                mom_rev_model = MomentumReversalDetector.load(mr_path)
                print(f"[ML] Momentum Reversal loaded (AUC 0.663)")
                ml_stats['mr_penalty'] = 0
            except Exception as _e:
                _track_error("load_mom_rev", _e)

        # Architecture 57: Immediate Stop Detector
        imm_stop_model = None
        is_path = _os.path.join(model_dir, 'immediate_stop_model.pkl')
        if _os.path.exists(is_path):
            try:
                imm_stop_model = ImmediateStopDetector.load(is_path)
                print(f"[ML] Immediate Stop Detector loaded (AUC 0.659)")
                ml_stats['is_skip'] = 0
            except Exception as _e:
                _track_error("load_imm_stop", _e)

        # Architecture 58: Profit Velocity Predictor
        profit_vel_model = None
        pv_path = _os.path.join(model_dir, 'profit_velocity_model.pkl')
        if _os.path.exists(pv_path):
            try:
                profit_vel_model = ProfitVelocityPredictor.load(pv_path)
                print(f"[ML] Profit Velocity Predictor loaded (AUC 0.649)")
            except Exception as _e:
                _track_error("load_profit_vel", _e)

        # Architecture 59: Breakout Stop Predictor
        breakout_stop_model = None
        bsp_path = _os.path.join(model_dir, 'breakout_stop_model.pkl')
        if _os.path.exists(bsp_path):
            try:
                breakout_stop_model = BreakoutStopPredictor.load(bsp_path)
                print(f"[ML] Breakout Stop Predictor loaded (AUC 0.794)")
                ml_stats['bsp_tighten'] = 0
                ml_stats['bsp_skip'] = 0
            except Exception as _e:
                _track_error("load_breakout_stop", _e)

        # Architecture 60: Breakout Momentum Validator
        breakout_momentum_model = None
        bmv_path = _os.path.join(model_dir, 'breakout_momentum_model.pkl')
        if _os.path.exists(bmv_path):
            try:
                breakout_momentum_model = BreakoutMomentumValidator.load(bmv_path)
                print(f"[ML] Breakout Momentum Validator loaded (AUC 0.626)")
                ml_stats['bmv_low_momentum_skip'] = 0
                ml_stats['bmv_high_momentum_boost'] = 0
            except Exception as _e:
                _track_error("load_breakout_momentum", _e)

        # Arch 61: Extended Run Predictor
        extended_run_model = None
        er_path = _os.path.join(model_dir, 'extended_run_model.pkl')
        if _os.path.exists(er_path):
            try:
                extended_run_model = ExtendedRunPredictor.load(er_path)
                print(f"[ML] Extended Run Predictor loaded (AUC 0.618)")
                ml_stats['er_wide_trail'] = 0
                ml_stats['er_tight_trail'] = 0
            except Exception as _e:
                _track_error("load_extended_run", _e)

        # Arch 62: Breakout Fade Detector
        breakout_fade_model = None
        bf_path = _os.path.join(model_dir, 'breakout_fade_model.pkl')
        if _os.path.exists(bf_path):
            try:
                breakout_fade_model = BreakoutFadeDetector.load(bf_path)
                print(f"[ML] Breakout Fade Detector loaded (AUC 0.712)")
                ml_stats['bf_fade_skipped'] = 0
                ml_stats['bf_fade_flagged'] = 0
            except Exception as _e:
                _track_error("load_breakout_fade", _e)

        # Arch 63: Momentum Follow-Through
        follow_through_model = None
        ft_path = _os.path.join(model_dir, 'follow_through_model.pkl')
        if _os.path.exists(ft_path):
            try:
                follow_through_model = MomentumFollowThrough.load(ft_path)
                print(f"[ML] Momentum Follow-Through loaded (AUC 0.621)")
                ml_stats['ft_wide_trail'] = 0
                ml_stats['ft_tight_trail'] = 0
            except Exception as _e:
                _track_error("load_follow_through", _e)

    # Feature capture mode (signal quality model training OR ML position sizing)
    _capture_feature_names = None
    _capture_history_buffer: List[Dict] = []
    if (capture_features or signal_quality_model is not None) and not ml_active:
        from v15.core.surfer_ml import get_feature_names
        _capture_feature_names = get_feature_names()
        if capture_features:
            print(f"[CAPTURE] Feature capture enabled ({len(_capture_feature_names)} features)")
        if signal_quality_model is not None:
            print(f"[ML-SIZING] Signal quality model loaded ({len(_capture_feature_names)} base features)")

    # Use pre-loaded data if provided (skip yfinance entirely)
    if tsla_df is not None:
        tsla = tsla_df
        higher_tf_data = higher_tf_dict or {}
        spy_df = spy_df_input
        vix_df = vix_df_input
        print(f"[PRE-LOADED] {len(tsla)} bars: {tsla.index[0]} to {tsla.index[-1]}")
        if realistic:
            print(f"[REALISTIC] max_leverage={max_leverage}x, slippage={slippage_bps}bps, "
                  f"commission=${commission_per_share}/share")
        _cache_hit = True
    else:
        _cache_hit = False

    # Fetch data (with file cache for consistent iterations)
    if not _cache_hit and _SURFER_DATA_CACHE.exists() and not _os_mod.environ.get('SURFER_REFRESH'):
        try:
            with open(_SURFER_DATA_CACHE, 'rb') as _f:
                _cached = _pickle.load(_f)
            tsla = _cached['tsla']
            higher_tf_data = _cached.get('higher_tf_data', {})
            spy_df = _cached.get('spy_df')
            vix_df = _cached.get('vix_df')
            print(f"[CACHE] Loaded {len(tsla)} bars from {_SURFER_DATA_CACHE}")
            print(f"  Date range: {tsla.index[0]} to {tsla.index[-1]}")
            _cache_hit = True
        except Exception as _e:
            print(f"[CACHE] Failed to load: {_e}, fetching fresh...")

    if not _cache_hit:
        print(f"Fetching {days}d of 5-min TSLA data...")
        tsla = yf.download('TSLA', period=f'{days}d', interval='5m', progress=False)
        if isinstance(tsla.columns, pd.MultiIndex):
            tsla.columns = tsla.columns.get_level_values(0)
        tsla.columns = [c.lower() for c in tsla.columns]
        print(f"Got {len(tsla)} bars")

        # Fetch higher TF data for context
        higher_tf_data = {}
        tf_list = [('1h', '1h', '2y'), ('daily', '1d', '5y')]
        if ml_active or use_multi_tf:
            tf_list = [
                ('1h', '1h', '2y'),
                ('daily', '1d', '5y'),
            ]
            if ml_active:
                # ML needs 4h and weekly too
                tf_list.extend([
                    ('weekly', '1wk', '5y'),
                ])

        if use_multi_tf or ml_active:
            for tf_label, yf_interval, yf_period in tf_list:
                print(f"  Fetching {tf_label} data...")
                df = yf.download('TSLA', period=yf_period, interval=yf_interval, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                higher_tf_data[tf_label] = df
                print(f"  {tf_label}: {len(df)} bars")

        # Resample 1h to 4h for ML features
        if ml_active and '1h' in higher_tf_data and '4h' not in higher_tf_data:
            h1 = higher_tf_data['1h']
            resampled = h1.resample('4h').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum',
            }).dropna()
            higher_tf_data['4h'] = resampled
            print(f"  4h: {len(resampled)} bars (resampled from 1h)")

        # Fetch SPY + VIX for ML correlation features
        spy_df = None
        vix_df = None
        if ml_active:
            print("  Fetching SPY for ML correlations...")
            spy_df = yf.download('SPY', period=f'{days}d', interval='5m', progress=False)
            if isinstance(spy_df.columns, pd.MultiIndex):
                spy_df.columns = spy_df.columns.get_level_values(0)
            spy_df.columns = [c.lower() for c in spy_df.columns]
            print(f"  SPY: {len(spy_df)} bars")

            try:
                vix_df = yf.download('^VIX', period='1y', interval='1d', progress=False)
                if isinstance(vix_df.columns, pd.MultiIndex):
                    vix_df.columns = vix_df.columns.get_level_values(0)
                vix_df.columns = [c.lower() for c in vix_df.columns]
                print(f"  VIX: {len(vix_df)} bars")
            except Exception:
                vix_df = None

        # Cache for subsequent runs
        try:
            with open(_SURFER_DATA_CACHE, 'wb') as _f:
                _pickle.dump({
                    'tsla': tsla, 'higher_tf_data': higher_tf_data,
                    'spy_df': spy_df, 'vix_df': vix_df,
                }, _f)
            print(f"[CACHE] Saved to {_SURFER_DATA_CACHE}")
        except Exception as _e:
            print(f"[CACHE] Failed to save: {_e}")

    if len(tsla) < 200:
        print("Not enough data for backtest")
        return BacktestMetrics(), []

    # Data quality checks — never trade on garbage data silently
    if spy_df is None:
        print("⚠️  WARNING: SPY data missing — ML correlation features will be degraded!")
    elif len(spy_df) < 100:
        print(f"⚠️  WARNING: SPY data sparse ({len(spy_df)} bars) — correlation features unreliable!")
    if vix_df is None:
        print("⚠️  WARNING: VIX data missing — volatility regime features unavailable!")
    if 'volume' not in tsla.columns or tsla['volume'].sum() == 0:
        print("⚠️  WARNING: Volume data missing — volume filters disabled!")
    for _req_tf in ['1h', 'daily']:
        if _req_tf not in higher_tf_data:
            print(f"⚠️  WARNING: {_req_tf} timeframe missing — multi-TF analysis degraded!")
    nan_pct = tsla[['open', 'high', 'low', 'close']].isna().mean().mean() * 100
    if nan_pct > 1.0:
        print(f"⚠️  WARNING: {nan_pct:.1f}% NaN values in TSLA OHLC — data may be corrupted!")

    closes = tsla['close'].values
    opens = tsla['open'].values
    highs = tsla['high'].values
    lows = tsla['low'].values

    # Compute ATR(14) for volatility-adjusted stops
    atr_period = 14
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    tr = np.concatenate([[highs[0] - lows[0]], tr])  # First bar uses H-L
    atr = np.full_like(closes, np.nan)
    atr[atr_period - 1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period
    # Fill initial NaN with first valid ATR
    first_valid = atr[atr_period - 1]
    atr[:atr_period - 1] = first_valid

    trades: List[Trade] = []
    trade_signals: list = []  # Parallel list of signal components per trade
    equity_curve: List[Tuple[int, float]] = []  # (bar_idx, equity)
    positions: List[OpenPosition] = []  # Multi-position support (max 2)
    position_signals: list = []  # Signal data for each open position
    trade_features: list = []      # Full feature vectors per closed trade (capture_features)
    position_features: list = []   # Feature vectors for each open position (capture_features)
    max_positions = 2
    pending_entries = []  # Deferred entries: list of (bar, direction, signal_type, confidence, stop_pct, tp_pct, primary_tf, ou_hl, max_hold, el_flagged, fast_rev, trade_size, signal_data)
    equity = initial_capital if initial_capital > 0 else position_size * 10
    initial_equity = equity
    peak_equity = equity
    max_dd = 0.0
    consecutive_losses = 0  # Track losing streak for position reduction
    consecutive_wins = 0    # Track winning streak for position ramping
    recent_trade_wins = []  # Arch 67: Rolling window of win/loss for dynamic cap
    last_signal_bar = -10   # Arch 72: Track last signal bar for persistence
    last_signal_dir = None  # Arch 72: Track last signal direction
    last_trade_entry_bar = -100  # Arch 85: Track last trade entry for drought detection
    last_breakout_loss = False  # Track if last loss was a breakout (confirms channel)
    daily_pnl = 0.0         # Running P&L for current trading day
    daily_breaker_active = False
    current_day = None
    # Walk forward from bar 100 (need lookback)
    start_bar = 100
    total_bars = len(tsla)
    last_print = 0

    print(f"\nBacktesting from bar {start_bar} to {total_bars} (interval={eval_interval})...")
    t_start = time.time()
    feature_vec = None  # Will be set when ML evaluates signals

    for bar in range(start_bar, total_bars, eval_interval):
        # Progress
        if bar - last_print >= 500:
            pct = (bar - start_bar) / (total_bars - start_bar) * 100
            print(f"  [{pct:.0f}%] bar={bar}/{total_bars}, trades={len(trades)}, equity=${equity:,.0f}")
            last_print = bar

        current_price = float(closes[bar])

        # Reset daily P&L at day boundary
        bar_date = tsla.index[bar].date() if hasattr(tsla.index[bar], 'date') else None
        if bar_date and bar_date != current_day:
            current_day = bar_date
            daily_pnl = 0.0
            daily_breaker_active = False

        # --- Check exits for all open positions ---
        window_highs = highs[max(0, bar - eval_interval):bar + 1]
        window_lows = lows[max(0, bar - eval_interval):bar + 1]
        window_high = float(np.max(window_highs))
        window_low = float(np.min(window_lows))

        closed_indices = []
        for pi, position in enumerate(positions):
            # Exit timing ML: tighten max_hold or force early exit
            if ml_active and exit_timing_model is not None and feature_vec is not None:
                try:
                    if len(feature_vec) == len(ml_feature_names):
                        et_pred = exit_timing_model.predict(feature_vec.reshape(1, -1))
                        pnl_fcast = float(et_pred['pnl_forecast'][0])
                        bars_held = bar - position.entry_bar

                        # If P&L forecast is strongly negative AND we've held >3 bars
                        if pnl_fcast < -0.003 and bars_held >= 4:
                            # Reduce max_hold to force earlier timeout
                            position.max_hold_bars = min(
                                position.max_hold_bars,
                                bars_held + 3  # Exit within 3 more bars
                            )
                            ml_stats['exit_tightened'] += 1

                        # If P&L forecast is very negative, tighten stop
                        if pnl_fcast < -0.005 and bars_held >= 2:
                            # Move stop closer (tighter by 30%)
                            entry = position.entry_price
                            if position.direction == 'BUY':
                                tighter = entry + (position.stop_price - entry) * 0.7
                                position.stop_price = max(position.stop_price, tighter)
                            else:
                                tighter = entry + (position.stop_price - entry) * 0.7
                                position.stop_price = min(position.stop_price, tighter)
                            ml_stats['exit_early'] += 1
                except Exception as _e:
                    _track_error("exit_timing_predict", _e)
            if ml_active and trail_model is not None and feature_vec is not None:
                try:
                    if len(feature_vec) == len(ml_feature_names):
                        trail_pred = trail_model.predict(feature_vec.reshape(1, -1))
                        tighten_p = float(trail_pred['tighten_prob'][0])

                        bars_held = bar - position.entry_bar
                        entry = position.entry_price

                        # If high tighten probability AND we're in profit
                        if tighten_p > 0.55 and bars_held >= 3:
                            if position.direction == 'BUY':
                                current_pnl = (current_price - entry) / entry
                                if current_pnl > 0.002:
                                    # Move stop to protect at least breakeven
                                    breakeven_stop = entry * 1.0005
                                    position.stop_price = max(position.stop_price, breakeven_stop)
                                    ml_stats['trail_tightened'] += 1
                            else:
                                current_pnl = (entry - current_price) / entry
                                if current_pnl > 0.002:
                                    breakeven_stop = entry * 0.9995
                                    position.stop_price = min(position.stop_price, breakeven_stop)
                                    ml_stats['trail_tightened'] += 1
                        elif tighten_p < 0.20:
                            ml_stats['trail_loosened'] += 1
                except Exception as _e:
                    _track_error("trail_optimizer_predict", _e)

            result = _check_position_exit(
                position, bar, current_price, window_high, window_low, eval_interval)

            if result is not None:
                exit_reason, exit_price = result
                bars_held = bar - position.entry_bar

                # Realistic: apply exit slippage
                if realistic:
                    slip = exit_price * slippage_bps / 10000
                    if position.direction == 'BUY':
                        exit_price -= slip  # Sell at worse (lower) price
                    else:
                        exit_price += slip  # Cover at worse (higher) price

                if position.direction == 'BUY':
                    pnl_pct = (exit_price - position.entry_price) / position.entry_price
                else:
                    pnl_pct = (position.entry_price - exit_price) / position.entry_price
                pnl = pnl_pct * position.trade_size

                # Realistic: deduct round-trip commission
                if realistic:
                    shares = position.trade_size / position.entry_price
                    total_commission = commission_per_share * shares * 2
                    pnl -= total_commission

                if position.direction == 'BUY':
                    mae = (position.entry_price - position.worst_price) / position.entry_price if position.worst_price > 0 else 0
                    mfe = (position.best_price - position.entry_price) / position.entry_price if position.best_price > 0 else 0
                else:
                    mae = (position.worst_price - position.entry_price) / position.entry_price if position.worst_price > 0 else 0
                    mfe = (position.entry_price - position.best_price) / position.entry_price if position.best_price > 0 else 0

                trade = Trade(
                    entry_bar=position.entry_bar, exit_bar=bar,
                    entry_price=position.entry_price, exit_price=exit_price,
                    direction=position.direction, confidence=position.confidence,
                    stop_pct=(abs(position.stop_price - position.entry_price) / position.entry_price),
                    tp_pct=(abs(position.tp_price - position.entry_price) / position.entry_price),
                    exit_reason=exit_reason, pnl=pnl, pnl_pct=pnl_pct,
                    hold_bars=bars_held, primary_tf=position.primary_tf,
                    signal_type=position.signal_type, trade_size=position.trade_size,
                    mae_pct=round(mae, 6), mfe_pct=round(mfe, 6),
                    el_flagged=position.el_flagged, is_flagged=getattr(position, 'is_flagged', False),
                    entry_time=str(tsla.index[position.entry_bar]) if position.entry_bar < len(tsla) else '',
                )
                trades.append(trade)
                equity += pnl
                equity_curve.append((bar, equity))
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity
                max_dd = max(max_dd, dd)

                if pnl <= 0:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    last_breakout_loss = (position.signal_type == 'break')
                    recent_trade_wins.append(0)
                else:
                    consecutive_losses = 0
                    consecutive_wins += 1
                    last_breakout_loss = False
                    recent_trade_wins.append(1)
                # Arch 67: Keep rolling window of last 20 trades
                if len(recent_trade_wins) > 20:
                    recent_trade_wins.pop(0)

                # Track daily P&L for circuit breaker
                daily_pnl += pnl
                if daily_pnl < -500:
                    daily_breaker_active = True

                closed_indices.append(pi)
                trade_signals.append(position_signals[pi])
                if capture_features:
                    trade_features.append(position_features[pi])

        # Remove closed positions (reverse order to preserve indices)
        for pi in sorted(closed_indices, reverse=True):
            positions.pop(pi)
            position_signals.pop(pi)
            if capture_features:
                position_features.pop(pi)

        # --- Process pending (deferred) entries ---
        expired_pending = []
        for pi, pend in enumerate(pending_entries):
            p_bar, p_dir, p_stype, p_conf, p_stop_pct, p_tp_pct, p_tf, p_ou, p_max_hold, p_el, p_fr, p_trade_size, p_sig_data = pend
            # Only process if exactly 1 eval cycle has passed
            if bar - p_bar != eval_interval:
                if bar - p_bar > eval_interval:
                    expired_pending.append(pi)
                continue
            # Check if price hasn't blown past stop already
            delayed_price = current_price
            if p_dir == 'BUY':
                would_stop = delayed_price * (1 - p_stop_pct)
                # If price dropped below where stop would be, skip
                if window_low < would_stop:
                    expired_pending.append(pi)
                    continue
            else:
                would_stop = delayed_price * (1 + p_stop_pct)
                if window_high > would_stop:
                    expired_pending.append(pi)
                    continue
            # Check room for position
            if len(positions) >= max_positions:
                expired_pending.append(pi)
                continue
            existing_dirs = {p.direction for p in positions}
            existing_types = {p.signal_type for p in positions}
            if p_dir in existing_dirs or p_stype in existing_types:
                expired_pending.append(pi)
                continue
            # Enter at delayed price
            if p_dir == 'BUY':
                stop = delayed_price * (1 - p_stop_pct)
                tp = delayed_price * (1 + p_tp_pct)
            else:
                stop = delayed_price * (1 + p_stop_pct)
                tp = delayed_price * (1 - p_tp_pct)
            positions.append(OpenPosition(
                entry_bar=bar, entry_price=delayed_price,
                direction=p_dir, confidence=p_conf,
                stop_price=stop, tp_price=tp,
                primary_tf=p_tf, signal_type=p_stype,
                trade_size=p_trade_size, ou_half_life=p_ou,
                max_hold_bars=p_max_hold, trailing_stop=delayed_price,
                el_flagged=p_el, fast_reversion=p_fr,
            ))
            position_signals.append(p_sig_data)
            if capture_features:
                position_features.append(None)  # Features not captured for deferred entries
            ml_stats.setdefault('delayed_entries', 0)
            ml_stats['delayed_entries'] += 1
            expired_pending.append(pi)
        for pi in sorted(expired_pending, reverse=True):
            pending_entries.pop(pi)

        # --- Generate new signal (if room for more positions) ---
        if len(positions) < max_positions:
            # Get lookback data for channel detection
            lookback = min(bar + 1, 100)
            df_slice = tsla.iloc[bar - lookback + 1:bar + 1]

            if len(df_slice) < 20:
                continue

            # Detect channels at multiple windows
            try:
                multi = detect_channels_multi_window(df_slice, windows=[10, 15, 20, 30, 40])
                best_ch, _ = select_best_channel(multi)
            except Exception:
                continue

            if best_ch is None or not best_ch.valid:
                continue

            # Build multi-TF channels
            slice_closes = df_slice['close'].values
            channels_by_tf = {'5min': best_ch}
            prices_by_tf = {'5min': slice_closes}
            current_prices_dict = {'5min': current_price}
            volumes_dict = {}

            if 'volume' in df_slice.columns:
                volumes_dict['5min'] = df_slice['volume'].values

            # Add higher TF channels (rolling window relative to current bar time)
            # Only include COMPLETED bars — a bar timestamped at period start
            # contains data through period end, so exclude it until then.
            _TF_PERIOD = {
                '1h': pd.Timedelta(hours=1),
                '4h': pd.Timedelta(hours=4),
                'daily': pd.Timedelta(days=1),
            }
            if use_multi_tf:
                current_time = tsla.index[bar]
                # Normalize to tz-naive for comparison
                if current_time.tzinfo is not None:
                    current_time_naive = current_time.tz_localize(None)
                else:
                    current_time_naive = current_time
                for tf_label, tf_df in higher_tf_data.items():
                    # Only use completed higher-TF bars (no lookahead)
                    tf_period = _TF_PERIOD.get(tf_label, pd.Timedelta(hours=1))
                    tf_idx = tf_df.index
                    if tf_idx.tz is not None:
                        tf_available = tf_df[(tf_idx + tf_period) <= current_time]
                    else:
                        tf_available = tf_df[(tf_idx + tf_period) <= current_time_naive]
                    tf_recent = tf_available.tail(100)
                    if len(tf_recent) < 30:
                        continue
                    tf_windows = TF_WINDOWS.get(tf_label, [20, 30, 40])
                    try:
                        tf_multi = detect_channels_multi_window(tf_recent, windows=tf_windows)
                        tf_ch, _ = select_best_channel(tf_multi)
                        if tf_ch and tf_ch.valid:
                            channels_by_tf[tf_label] = tf_ch
                            prices_by_tf[tf_label] = tf_recent['close'].values
                            current_prices_dict[tf_label] = float(tf_recent['close'].iloc[-1])
                            if 'volume' in tf_recent.columns:
                                volumes_dict[tf_label] = tf_recent['volume'].values
                    except Exception as _e:
                        _track_error("higher_tf_extract", _e)

            try:
                analysis = analyze_channels(
                    channels_by_tf, prices_by_tf, current_prices_dict,
                    volumes_by_tf=volumes_dict if volumes_dict else None,
                )
            except Exception:
                continue

            sig = analysis.signal

            # --- Feature Capture (signal quality model / ML sizing) ---
            current_signal_features = None
            if (capture_features or signal_quality_model is not None) and not ml_active and sig.action in ('BUY', 'SELL'):
                try:
                    current_signal_features, _ = _extract_signal_features(
                        analysis, tsla, bar, closes, spy_df, vix_df,
                        _capture_feature_names, _capture_history_buffer, eval_interval,
                    )
                except Exception as _e:
                    _track_error("feature_extract", _e)
            ml_prediction = None
            if sig.action in ('BUY', 'SELL'):
                ml_stats['total_signals'] += 1
            if ml_active and sig.action in ('BUY', 'SELL'):
                # Save original confidence — sub-model penalties degrade trail
                _original_confidence = sig.confidence
                try:
                    feature_vec, _ = _extract_signal_features(
                        analysis, tsla, bar, closes, spy_df, vix_df,
                        ml_feature_names, ml_history_buffer, eval_interval,
                    )
                    if capture_features:
                        current_signal_features = feature_vec.copy()

                    # Update feature window for TrendGBT
                    ml_feature_window.append(feature_vec.copy())
                    if len(ml_feature_window) > TrendGBTModel.WINDOW_SIZE:
                        ml_feature_window.pop(0)

                    # Run ML prediction
                    ml_prediction = ml_model.predict(feature_vec.reshape(1, -1))

                    # ML Action: 0=HOLD, 1=BUY, 2=SELL — informational only
                    # The ultra-tight trail + EL/IS/BMV handle risk better than
                    # GBT confidence penalties (which degrade trail effectiveness)
                    if 'action' in ml_prediction:
                        ml_action_id = int(ml_prediction['action'][0])
                        physics_action_id = 1 if sig.action == 'BUY' else 2

                        if ml_action_id == 0:
                            ml_stats['ml_filtered'] += 1
                        elif ml_action_id == physics_action_id:
                            ml_stats['ml_agreed'] += 1
                        else:
                            ml_stats['ml_filtered'] += 1

                    # ML Break direction: informational only
                    # The trail handles adverse moves better than skipping
                    if 'break_dir' in ml_prediction:
                        bd = int(ml_prediction['break_dir'][0])
                        if 'lifetime' in ml_prediction:
                            lifetime = float(ml_prediction['lifetime'][0])
                            if lifetime < 5:
                                if sig.action == 'BUY' and bd == 2:
                                    ml_stats['ml_filtered'] += 1
                                elif sig.action == 'SELL' and bd == 1:
                                    ml_stats['ml_filtered'] += 1

                    # Adjust max hold based on predicted lifetime
                    if 'lifetime' in ml_prediction:
                        predicted_life = float(ml_prediction['lifetime'][0])
                        # Don't hold longer than predicted channel life
                        if predicted_life > 3:
                            ml_max_hold = max(6, int(predicted_life * 0.8))
                        else:
                            ml_max_hold = None
                    else:
                        ml_max_hold = None

                    # Ensemble override: if ensemble is loaded, use its prediction
                    if ensemble_model is not None:
                        try:
                            ens_pred = ensemble_model.predict(
                                feature_vec.reshape(1, -1),
                                gbt=ensemble_base_models.get('gbt'),
                                transformer=ensemble_base_models.get('transformer'),
                                survival=ensemble_base_models.get('survival'),
                                quality=ensemble_base_models.get('quality'),
                                feature_names=ml_feature_names,
                            )

                            # Ensemble action: override base ML if different
                            if 'action' in ens_pred:
                                ens_action = int(ens_pred['action'][0])
                                physics_action_id = 1 if sig.action == 'BUY' else 2

                                if ens_action == 0:  # Ensemble says HOLD → log only
                                    ml_stats['ensemble_filtered'] += 1
                                elif ens_action != physics_action_id:
                                    # Ensemble disagrees — informational only
                                    ml_stats['ensemble_filtered'] += 1

                            # Ensemble lifetime: use for hold time if available
                            if 'lifetime' in ens_pred:
                                ens_life = float(ens_pred['lifetime'][0])
                                if ens_life > 3:
                                    ml_max_hold = max(6, int(ens_life * 0.8))
                        except Exception as _e:
                            _track_error("ml_predict", _e)

                    # Regime-conditional adjustment
                    if regime_model is not None:
                        try:
                            reg_pred = regime_model.predict(feature_vec.reshape(1, -1))
                            regime_id = int(reg_pred['regime'][0])
                            regime_name = RegimeConditionalModel.REGIME_NAMES[regime_id]

                            # Regime: informational only (trail handles all risk)
                            if regime_id == 2:  # VOLATILE
                                ml_stats['regime_penalized'] += 1
                            elif regime_id == 0:  # TRENDING_UP
                                if sig.action == 'BUY':
                                    ml_stats['regime_boosted'] += 1
                            elif regime_id == 1:  # TRENDING_DOWN
                                if sig.action == 'SELL':
                                    ml_stats['regime_boosted'] += 1
                        except Exception as _e:
                            _track_error("ml_predict", _e)

                    # TrendGBT break direction confirmation
                    if trend_gbt_model is not None and len(ml_feature_window) >= TrendGBTModel.WINDOW_SIZE:
                        try:
                            window = np.stack(ml_feature_window[-TrendGBTModel.WINDOW_SIZE:])
                            tg_pred = trend_gbt_model.predict(window)
                            tg_bd = int(tg_pred['break_dir'][0])
                            # 0=down, 1=up, 2=survive

                            # BUY expects break_up (1), SELL expects break_down (0)
                            if sig.action == 'BUY' and tg_bd == 0:
                                # TrendGBT says break DOWN but we're buying
                                sig.confidence *= 0.80
                                ml_stats['trend_gbt_filtered'] += 1
                            elif sig.action == 'SELL' and tg_bd == 1:
                                # TrendGBT says break UP but we're selling
                                sig.confidence *= 0.80
                                ml_stats['trend_gbt_filtered'] += 1
                            elif (sig.action == 'BUY' and tg_bd == 1) or \
                                 (sig.action == 'SELL' and tg_bd == 0):
                                # TrendGBT confirms direction
                                sig.confidence *= 1.15
                                ml_stats['trend_gbt_confirmed'] += 1
                        except Exception as _e:
                            _track_error("ml_predict", _e)

                    # CV Ensemble: consensus-based confidence scaling
                    if cv_ensemble_model is not None:
                        try:
                            cv_pred = cv_ensemble_model.predict(feature_vec.reshape(1, -1))
                            bd_consensus = float(cv_pred['bd_consensus'][0])
                            cv_bd = int(cv_pred['break_dir'][0])

                            if bd_consensus >= 0.8:
                                # 4+/5 folds agree — high confidence
                                if (sig.action == 'BUY' and cv_bd == 1) or \
                                   (sig.action == 'SELL' and cv_bd == 0):
                                    sig.confidence *= 1.15  # Moderate boost
                                    ml_stats['cv_high_consensus'] += 1
                                elif (sig.action == 'BUY' and cv_bd == 0) or \
                                     (sig.action == 'SELL' and cv_bd == 1):
                                    sig.confidence *= 0.75  # Penalize mismatch
                                    ml_stats['cv_low_consensus'] += 1
                            elif bd_consensus < 0.6:
                                sig.confidence *= 0.90  # Mild penalty
                                ml_stats['cv_low_consensus'] += 1
                        except Exception as _e:
                            _track_error("ml_predict", _e)

                    # Physics-Residual: correct confidence + lifetime using residuals
                    if residual_model is not None:
                        try:
                            res_pred = residual_model.predict(feature_vec.reshape(1, -1))

                            # Confidence scale: direct multiplier from residual model
                            if 'confidence_scale' in res_pred:
                                conf_scale = float(res_pred['confidence_scale'][0])
                                sig.confidence *= conf_scale
                                if conf_scale > 1.0:
                                    ml_stats['residual_boosted'] += 1
                                elif conf_scale < 0.9:
                                    ml_stats['residual_penalized'] += 1

                            # Lifetime correction: adjust max hold time
                            if 'lifetime_correction' in res_pred and ml_max_hold is not None:
                                lt_corr = float(res_pred['lifetime_correction'][0])
                                corrected_hold = max(6, int(ml_max_hold + lt_corr * 0.3))
                                if corrected_hold != ml_max_hold:
                                    ml_max_hold = corrected_hold
                                    ml_stats['residual_lifetime_adj'] += 1
                        except Exception as _e:
                            _track_error("ml_predict", _e)

                    # Adverse Movement: filter high stop-out probability trades
                    if adverse_model is not None:
                        try:
                            is_buy = (sig.action == 'BUY')
                            adv_pred = adverse_model.predict(feature_vec.reshape(1, -1), is_buy=is_buy)

                            # If stop probability is high (>0.4) → penalize confidence
                            if 'stop_prob' in adv_pred:
                                stop_p = float(adv_pred['stop_prob'][0])
                                if stop_p > 0.4:
                                    sig.confidence *= 0.80
                                    ml_stats['adverse_filtered'] += 1
                                elif stop_p < 0.1:
                                    sig.confidence *= 1.10
                                    ml_stats['adverse_boosted'] += 1

                            # If viability is high, boost
                            if 'viable_prob' in adv_pred:
                                viable_p = float(adv_pred['viable_prob'][0])
                                if viable_p > 0.65:
                                    sig.confidence *= 1.05
                        except Exception as _e:
                            _track_error("ml_predict", _e)

                    # Composite Signal Scorer: learned combination of all model outputs
                    if composite_model is not None:
                        try:
                            comp_pred = composite_model.predict(
                                feature_vec.reshape(1, -1), model_dir=model_dir)
                            comp_action = int(comp_pred['action'][0])
                            comp_conf = float(comp_pred['max_confidence'][0])

                            physics_action_id = 1 if sig.action == 'BUY' else 2

                            if comp_action == 0:  # Composite says HOLD
                                if comp_conf > 0.5:
                                    sig.confidence *= 0.80
                                    ml_stats['composite_filtered'] += 1
                            elif comp_action == physics_action_id:
                                # Composite agrees with physics direction
                                if comp_conf > 0.5:
                                    sig.confidence *= 1.10
                                    ml_stats['composite_agreed'] += 1
                            else:
                                # Composite disagrees (opposite direction)
                                if comp_conf > 0.5:
                                    sig.confidence *= 0.75
                                    ml_stats['composite_filtered'] += 1
                        except Exception as _e:
                            _track_error("ml_predict", _e)

                    # Volatility Transition: avoid entries during vol spikes
                    if vol_model is not None:
                        try:
                            vol_pred = vol_model.predict(feature_vec.reshape(1, -1))
                            spike_p = float(vol_pred['spike_prob'][0])
                            vol_regime = str(vol_pred['vol_regime'][0])

                            if vol_regime == 'danger':
                                # High vol spike probability → strong penalty
                                sig.confidence *= 0.65
                                ml_stats['vol_danger_skip'] += 1
                            elif vol_regime == 'warning':
                                # Moderate vol risk → mild penalty
                                sig.confidence *= 0.90
                                ml_stats['vol_warning_scale'] += 1
                            else:
                                # Calm → slight boost (low vol = favorable for channel trading)
                                sig.confidence *= 1.05
                                ml_stats['vol_calm_boost'] += 1
                        except Exception as _e:
                            _track_error("ml_predict", _e)

                    # Momentum Exhaustion: avoid entering exhausted moves
                    if exhaustion_model is not None:
                        try:
                            exh_pred = exhaustion_model.predict(feature_vec.reshape(1, -1))
                            exh_prob = float(exh_pred['exhaustion_prob'][0])

                            if exh_prob > 0.60:
                                sig.confidence *= 0.75
                                ml_stats['exh_exhausted_skip'] += 1
                            elif exh_prob > 0.45:
                                sig.confidence *= 0.95
                                ml_stats['exh_tiring_scale'] += 1
                            elif exh_prob < 0.25:
                                sig.confidence *= 1.03
                                ml_stats['exh_fresh_boost'] += 1
                        except Exception as _e:
                            _track_error("ml_predict", _e)

                    # Cross-Asset Amplifier: regime-based confidence scaling
                    if cross_asset_model is not None:
                        try:
                            ca_pred = cross_asset_model.predict(feature_vec.reshape(1, -1))
                            ca_regime = int(ca_pred['market_regime'][0])

                            if ca_regime == 2:  # rotation — TSLA-specific, channels most reliable
                                sig.confidence *= 1.08
                                ml_stats['ca_rotation_boost'] += 1
                            elif ca_regime == 3:  # correlated selloff — avoid
                                sig.confidence *= 0.65
                                ml_stats['ca_selloff_skip'] += 1

                            ml_stats['ca_scale_applied'] += 1
                        except Exception as _e:
                            _track_error("ml_predict", _e)

                    # Intraday Session Model: session quality confidence scaling
                    if session_model is not None:
                        try:
                            sess_pred = session_model.predict(feature_vec.reshape(1, -1))
                            sess_quality = float(sess_pred['session_quality'][0])

                            if sess_quality < 0.20:
                                sig.confidence *= 0.80  # Poor session → reduce confidence
                                ml_stats['session_penalty'] += 1
                            elif sess_quality > 0.55:
                                sig.confidence *= 1.05  # Good session → slight boost
                                ml_stats['session_boost'] += 1
                        except Exception as _e:
                            _track_error("session", _e)

                    # Channel Maturity Predictor: skip mature channels or boost young ones
                    if maturity_model is not None:
                        try:
                            mat_pred = maturity_model.predict(feature_vec.reshape(1, -1))
                            mat_prob = float(mat_pred['maturity_prob'][0])
                            rem_life = float(mat_pred['remaining_life'][0])

                            if mat_prob > 0.75 and rem_life < 8:
                                sig.confidence *= 0.75  # Channel about to break → risky
                                ml_stats['maturity_skip'] += 1
                            elif mat_prob < 0.20 and rem_life > 50:
                                sig.confidence *= 1.05  # Young channel → more room
                                ml_stats['maturity_boost'] += 1
                        except Exception as _e:
                            _track_error("ml_predict", _e)

                    # Return Asymmetry Predictor: adjust stops/targets based on expected move type
                    if asymmetry_model is not None:
                        try:
                            asym_pred = asymmetry_model.predict(feature_vec.reshape(1, -1))
                            spike_prob = float(asym_pred['spike_prob'][0])
                            expected_skew = float(asym_pred['expected_skewness'][0])

                            # High spike probability → widen stop to avoid stop-out
                            if spike_prob > 0.40:
                                # Don't change confidence, but store for exit logic
                                ml_stats['asym_widen_stop'] += 1
                            elif spike_prob < 0.10:
                                ml_stats['asym_tighten_trail'] += 1
                        except Exception as _e:
                            _track_error("ml_predict", _e)

                    # Gap Risk Predictor: AUC 0.852 but too aggressive at all thresholds
                    # minutes_since_open dominates features → penalizes all late-day entries
                    # Disabled: trained but not integrated
                    # if gap_risk_model is not None: ...

                    # Mean Reversion Speed: boost bounce trades with fast reversion
                    fast_rev = 0.0  # Track for trail adjustment
                    if reversion_model is not None:
                        try:
                            rev_pred = reversion_model.predict(feature_vec.reshape(1, -1))
                            fast_rev = float(rev_pred['fast_reversion_prob'][0])

                            if fast_rev > 0.55 and sig.signal_type == 'bounce':
                                sig.confidence *= 1.10  # Fast reversion → bounce will work
                                ml_stats['rev_fast_boost'] += 1
                            elif fast_rev < 0.15:
                                sig.confidence *= 0.85  # Slow reversion → less reliable
                                ml_stats['rev_slow_penalty'] += 1
                        except Exception as _e:
                            _track_error("reversion", _e)

                    # Liquidity State: thin market AUC 1.0 (leaky target — volume_ratio < 0.7)
                    # Slippage regressor (corr 0.499) too aggressive, penalizes 13/46 signals
                    # Disabled: trained but not integrated
                    # if liquidity_model is not None: ...

                    # Trade Duration (Arch 31): Quick AUC 0.617, Hold Corr 0.182
                    # Neutral at all thresholds — doesn't improve PF
                    # Disabled: trained but not integrated
                    # if duration_model is not None: ...

                    # Quantile Risk: use return spread to adjust confidence
                    if quantile_risk_model is not None:
                        try:
                            qr_pred = quantile_risk_model.predict(feature_vec.reshape(1, -1))
                            spread = float(qr_pred['return_spread'][0])
                            risk_ratio = float(qr_pred['risk_ratio'][0])

                            # Favorable asymmetry (P90 >> |P10|) → boost
                            if risk_ratio > 1.5:
                                sig.confidence *= 1.05
                                ml_stats['qr_favorable_asym'] += 1
                        except Exception as _e:
                            _track_error("quantile_risk", _e)

                    # Tail Risk: big move coming — good for breaks, bad for bounces
                    tail_prob = 0.0  # Track for TP widening
                    if tail_risk_model is not None:
                        try:
                            tr_pred = tail_risk_model.predict(feature_vec.reshape(1, -1))
                            tail_prob = float(tr_pred['tail_risk_prob'][0])

                            if tail_prob > 0.15:
                                if sig.signal_type == 'bounce':
                                    # Big move may blow through channel → penalize bounce
                                    sig.confidence *= 0.85
                                    ml_stats['tail_bounce_penalty'] += 1
                                elif sig.signal_type == 'break':
                                    # Big move + breakout = ride the trend
                                    sig.confidence *= 1.10
                                    ml_stats['tail_break_boost'] += 1
                        except Exception as _e:
                            _track_error("ml_predict", _e)

                    # Extreme Loser Detector: high loser probability → penalize hard
                    el_loser_prob = 0.0  # Track for stop tightening
                    if extreme_loser_model is not None:
                        try:
                            el_pred = extreme_loser_model.predict(feature_vec.reshape(1, -1))
                            el_loser_prob = float(el_pred['loser_prob'][0])

                            if el_loser_prob > 0.18 and sig.signal_type == 'break':
                                # EL breakout skip disabled: ultra-tight 0.05x stop on ALL
                                # breakouts now protects via breakeven trail mechanism
                                ml_stats.setdefault('el_break_flagged', 0)
                                ml_stats['el_break_flagged'] += 1
                            elif el_loser_prob > 0.18:
                                sig.confidence *= 0.80
                                ml_stats['el_penalty'] += 1
                        except Exception as _e:
                            _track_error("extreme_loser", _e)

                    # Drawdown Magnitude (Arch 49): P75 corr 0.283 but compounds with EL
                    # Disabled: too many penalties (41) overlap with Extreme Loser
                    # if drawdown_mag_model is not None: ...

                    # Win Streak (Arch 50): AUC 0.620, 18 boosts, PF unchanged
                    # Disabled: boosts don't change trade outcomes
                    # if win_streak_model is not None: ...

                    # Adversarial Selector (Arch 37): AUC 0.605 but 0 triggers in backtest
                    # Only 3 boosting rounds → predictions cluster, never hit thresholds
                    # Disabled: trained but not integrated
                    # if adversarial_model is not None: ...

                    # Stop Loss Predictor: disabled (60% base rate, weak discrimination)
                    # Momentum Divergence (Arch 23): disabled (AUC 0.557, near random)
                    # Regime Transition (Arch 28): disabled (stability corr 0.035)
                    # Winner Amplifier (Arch 32): disabled (AUC 0.492, magnitude corr -0.27)
                    # Fractal Regime (Arch 33): disabled (AUC 0.546, near random)
                    # Volume Conviction (Arch 34): disabled (AUC 0.529, near random)
                    # Energy Momentum (Arch 35): disabled (AUC 0.543, near random)
                    # Multi-Exit Strategy (Arch 36): disabled (majority class collapse)
                    # Cascade Confidence (Arch 38): disabled (model preds contribute 1.8%)
                    # kNN Trade Analogy (Arch 39): disabled (AUC 0.549, random)
                    # Drawdown Recovery (Arch 42): disabled (AUC 0.500, completely random)
                    # Stop Distance (Arch 43): disabled (MAE corr 0.346, 0 triggers at all thresholds)
                    # Vol Clustering (Arch 44): disabled (AUC 0.683, 11 pen/1 boost, PF unchanged)

                    # Feature Interaction Loser (Arch 55): AUC 0.733 but 1 boosting round
                    # Disabled: predictions cluster 0.14-0.19, median split adds noise (PF 9.04→8.88)
                    # if feat_int_model is not None: ...

                    # Bounce Loser (Arch 54): AUC 0.670, 18 pen on bounces, PF unchanged
                    # Disabled: bounce penalties don't change outcomes
                    # if bounce_loser_model is not None: ...

                    # Momentum Reversal (Arch 56): AUC 0.663, testing disabled
                    # if mom_rev_model is not None: ...

                    # Profit Velocity (Arch 58): detect fast-profit setups
                    fast_profit_prob = 0.0
                    if profit_vel_model is not None:
                        try:
                            pv_pred = profit_vel_model.predict(feature_vec.reshape(1, -1))
                            fast_profit_prob = float(pv_pred['fast_profit_prob'][0])
                        except Exception as _e:
                            _track_error("profit_vel", _e)

                    # Immediate Stop Detector (Arch 57): tighten stop on high-risk entries
                    imm_stop_prob = 0.0
                    imm_stop_skip = False
                    if imm_stop_model is not None:
                        try:
                            is_pred = imm_stop_model.predict(feature_vec.reshape(1, -1))
                            imm_stop_prob = float(is_pred['immediate_stop_prob'][0])
                            if imm_stop_prob > 0.35:
                                ml_stats['is_skip'] += 1  # Track triggers
                        except Exception as _e:
                            _track_error("imm_stop", _e)

                    # Breakout Stop Predictor (Arch 59): AUC 0.794
                    # Stop tightening only — confidence penalty hurts PF
                    bsp_prob = 0.0
                    if breakout_stop_model is not None and sig.signal_type == 'break':
                        try:
                            bsp_pred = breakout_stop_model.predict(feature_vec.reshape(1, -1))
                            bsp_prob = float(bsp_pred['breakout_stop_prob'][0])
                            if bsp_prob > 0.20:
                                ml_stats['bsp_tighten'] += 1
                        except Exception as _e:
                            _track_error("breakout_stop", _e)

                except Exception as _ml_err:
                    ml_prediction = None
                    ml_max_hold = None
                    imm_stop_skip = False
                    el_loser_prob = 0.0
                    imm_stop_prob = 0.0
                    fast_rev = 0.0
                    bsp_prob = 0.0
                    ml_stats.setdefault('ml_predict_errors', 0)
                    ml_stats['ml_predict_errors'] += 1
                    if ml_stats['ml_predict_errors'] <= 3:
                        print(f"⚠️  ML PREDICTION FAILED (signal #{ml_stats['total_signals']}): {_ml_err}")
            else:
                ml_max_hold = None
                imm_stop_skip = False
                el_loser_prob = 0.0
                imm_stop_prob = 0.0
                fast_rev = 0.0
                bsp_prob = 0.0

            # Track signals that fail the confidence gate
            if sig.action in ('BUY', 'SELL') and sig.confidence < min_confidence:
                ml_stats['conf_below_min'] += 1
            elif sig.action not in ('BUY', 'SELL'):
                ml_stats['not_buy_sell'] += 1

            if sig.action in ('BUY', 'SELL') and sig.confidence >= min_confidence:
                # Daily circuit breaker: stop trading if down $500+ today
                if daily_breaker_active:
                    ml_stats['circuit_breaker'] += 1
                    continue

                # 10AM ET skip disabled — testing without in new data window
                pass

                # Don't enter if we already have a position in the same direction
                existing_dirs = {p.direction for p in positions}
                existing_types = {p.signal_type for p in positions}
                if sig.action in existing_dirs:
                    ml_stats['anti_pyramid'] += 1
                    continue  # No pyramiding
                if sig.signal_type in existing_types:
                    ml_stats['anti_double_type'] += 1
                    continue  # No double-bounce or double-break

                # Volume tracking: log low-volume breakouts (ultra-tight stop protects)
                if sig.signal_type == 'break' and 'volume' in tsla.columns:
                    current_vol = tsla['volume'].iloc[bar]
                    avg_vol = tsla['volume'].iloc[max(0, bar-20):bar].mean()
                    if avg_vol > 0 and current_vol < avg_vol * 0.8:
                        ml_stats['low_volume'] += 1
                        # continue  # Disabled: ultra-tight 0.05x stop protects all breakouts

                # Quality scorer: predict win probability for this exact trade
                if quality_scorer is not None and ml_active:
                    try:
                        from v15.validation.signal_quality_model import _append_signal_meta
                        class _QSSigProxy:
                            pass
                        _qs_sig = _QSSigProxy()
                        _qs_sig.signal_type = sig.signal_type
                        _qs_sig.direction = sig.action
                        _qs_sig.stop_pct = getattr(sig, 'suggested_stop_pct', 0.005)
                        _qs_sig.tp_pct = getattr(sig, 'suggested_tp_pct', 0.012)
                        _qs_sig.primary_tf = getattr(sig, 'primary_tf', '')
                        _qs_sig.entry_time = str(tsla.index[bar]) if bar < len(tsla) else ''
                        _qs_sig_data = {
                            'position_score': sig.position_score,
                            'energy_score': sig.energy_score,
                            'entropy_score': sig.entropy_score,
                            'confluence_score': sig.confluence_score,
                            'timing_score': sig.timing_score,
                            'channel_health': sig.channel_health,
                            'confidence': sig.confidence,
                        }
                        qs_vec = _append_signal_meta(feature_vec, _qs_sig, _qs_sig_data, extended=True)

                        qs_pred = quality_scorer.predict(qs_vec.reshape(1, -1))
                        win_prob = float(qs_pred.get('win_prob', [0.5])[0])

                        # Filter: skip if quality scorer is quite pessimistic
                        if win_prob < 0.35:
                            ml_stats['quality_filtered'] += 1
                            continue

                        # Boost: if quality scorer is very confident about a win
                        if win_prob > 0.65:
                            sig.confidence *= 1.15  # 15% boost
                            ml_stats['quality_boosted'] += 1
                    except Exception as _e:
                        _track_error("quality_scorer_predict", _e)

                # Restore original confidence — ML sub-model confidence penalties
                # degrade trail effectiveness (0.60x drops below ultra-tight threshold)
                if ml_active:
                    sig.confidence = _original_confidence

                # Position score filter: disabled — EL+BMV handle breakout filtering
                if sig.signal_type == 'break' and sig.position_score < 0.80:
                    ml_stats['pos_score_weak'] += 1
                    # continue  # Disabled: EL+BMV are better breakout filters

                # Low-conf bounce filter: disabled — bounces are 100% WR with ultra-tight trail
                if (sig.signal_type == 'bounce' and sig.action == 'BUY'
                        and sig.confidence < 0.46):
                    ml_stats['low_conf_buy_bounce'] += 1
                    # continue  # Disabled: trail protects all bounces


                # Enter position — use next bar's open (no look-ahead bias)
                # Signal fires at bar N close; order fills at bar N+1 open
                next_bar = bar + 1
                if next_bar < total_bars:
                    entry_price = float(opens[next_bar])
                else:
                    continue  # Can't enter on last bar

                # Risk-normalized position sizing
                if realistic:
                    # Realistic: 2% risk of current equity (mild compounding)
                    risk_mult = 0.02
                    risk_budget = equity * risk_mult
                else:
                    # Original: scale with equity growth + confidence tiers + streaks
                    equity_scale = equity / initial_equity  # Grows as we win
                    # Higher base risk for bounces (94% WR = very safe)
                    risk_mult = 0.150 if sig.signal_type == 'bounce' else 0.150
                    base_risk = position_size * risk_mult * equity_scale
                    if sig.confidence >= 0.70:
                        risk_budget = base_risk * 1.8
                    elif sig.confidence >= 0.60:
                        risk_budget = base_risk * 1.5
                    else:
                        risk_budget = base_risk * 1.2

                    # Adaptive sizing: ramp up on win streaks, halve on losing streaks
                    if consecutive_wins >= 2:
                        streak_boost = min(5.0, 1.0 + 0.50 * (consecutive_wins - 1))
                        risk_budget *= streak_boost
                    if consecutive_losses >= 3:
                        risk_budget *= 0.50  # Half size after 3+ consecutive losses

                # Volatility-adjusted stops: blend channel width with ATR
                current_atr = atr[bar]
                if sig.signal_type == 'bounce':
                    # Bounces: tighter stops OK (100% WR, 0 stop-outs)
                    atr_floor = (0.5 * current_atr) / entry_price
                    atr_cap = (1.5 * current_atr) / entry_price
                else:
                    # Breakouts: wider stops (survive noise)
                    atr_floor = (1.5 * current_atr) / entry_price
                    atr_cap = (3.0 * current_atr) / entry_price
                adjusted_stop_pct = np.clip(sig.suggested_stop_pct, atr_floor, atr_cap)

                # ML stop tightening: if Extreme Loser flags risk, tighten stop by 35%
                if el_loser_prob > 0.18:
                    adjusted_stop_pct *= 0.75
                    ml_stats.setdefault('el_stop_tighten', 0)
                    ml_stats['el_stop_tighten'] += 1

                # IS stop tightening: if Immediate Stop flags risk, tighten stop by 20%
                if imm_stop_prob > 0.35:
                    adjusted_stop_pct *= 0.75
                    ml_stats.setdefault('is_stop_tighten', 0)
                    ml_stats['is_stop_tighten'] += 1

                # Ultra-tight breakout stops: enables the breakeven trail mechanism
                # which protects at 0.01% profit. Previously gated on conf > 0.90,
                # but trail protection works for ALL breakouts regardless of confidence.
                if sig.signal_type == 'break':
                    adjusted_stop_pct *= 0.05

                # BSP stop tightening: breakout-specific, AUC 0.794
                # BSP stop tightening disabled — marginal impact (PF 9.67→9.70)
                # if bsp_prob > 0.20 and sig.signal_type == 'break':
                #     adjusted_stop_pct *= 0.65  # 35% tighter
                #     ml_stats.setdefault('bsp_stop_tighten', 0)
                #     ml_stats['bsp_stop_tighten'] += 1

                # Arch 73: Skip ultra-narrow breakouts (stop<0.025%) — slippage eats profit
                if realistic and sig.signal_type == 'break' and adjusted_stop_pct < 0.00030:
                    ml_stats.setdefault('narrow_break_skip', 0)
                    ml_stats['narrow_break_skip'] += 1
                    continue

                # Arch 100: Widen TP for high-confidence bounces (let winners run)
                tp_pct = sig.suggested_tp_pct
                if sig.signal_type == 'bounce' and sig.confidence > 0.65:
                    tp_pct *= 1.30
                    ml_stats.setdefault('wide_tp_bounce', 0)
                    ml_stats['wide_tp_bounce'] += 1

                if sig.action == 'BUY':
                    stop = entry_price * (1 - adjusted_stop_pct)
                    tp = entry_price * (1 + tp_pct)
                else:
                    stop = entry_price * (1 + adjusted_stop_pct)
                    tp = entry_price * (1 - tp_pct)

                # Risk-normalized sizing: trade_size = risk_budget / stop_pct
                # Wider stops → smaller position, tighter stops → larger position
                trade_size = risk_budget / max(adjusted_stop_pct, 0.001)

                # --- ML Position Sizing ---
                if signal_quality_model is not None and ml_size_fn is not None and current_signal_features is not None:
                    try:
                        from v15.validation.signal_quality_model import _append_signal_meta
                        class _TradeLike:
                            pass
                        _t = _TradeLike()
                        _t.signal_type = sig.signal_type
                        _t.direction = sig.action
                        _t.stop_pct = adjusted_stop_pct
                        _t.tp_pct = sig.suggested_tp_pct
                        _t.primary_tf = sig.primary_tf
                        _t.entry_time = str(tsla.index[bar]) if bar < len(tsla) else ''
                        _sig_data = {
                            'position_score': sig.position_score,
                            'energy_score': sig.energy_score,
                            'entropy_score': sig.entropy_score,
                            'confluence_score': sig.confluence_score,
                            'timing_score': sig.timing_score,
                            'channel_health': sig.channel_health,
                            'confidence': sig.confidence,
                        }
                        full_vec = _append_signal_meta(current_signal_features, _t, _sig_data, extended=True)
                        pred = signal_quality_model.predict(full_vec)
                        trade_size *= ml_size_fn(pred['quality_score'])
                    except Exception as _e:
                        _track_error("ml_sizing_predict", _e)

                if realistic:
                    # Realistic: leverage-based cap, no multiplicative boosts
                    max_buying_power = equity * max_leverage
                    # Arch 67: Dynamic cap based on rolling 20-trade win rate
                    # Hot streaks are reliable in this system (84% base WR)
                    # so sizing up when rolling WR > 90% captures compounding
                    if len(recent_trade_wins) >= 10:
                        rolling_wr = sum(recent_trade_wins) / len(recent_trade_wins)
                        if rolling_wr >= 0.90:
                            cap_pct = 0.70  # Hot streak → lean in (2x normal)
                            ml_stats.setdefault('dyn_cap_hot', 0)
                            ml_stats['dyn_cap_hot'] += 1
                        else:
                            cap_pct = 0.35  # Normal
                            ml_stats.setdefault('dyn_cap_normal', 0)
                            ml_stats['dyn_cap_normal'] += 1
                    else:
                        cap_pct = 0.35  # Not enough history
                    size_cap = max_buying_power * cap_pct
                    trade_size = min(trade_size, size_cap)

                    # Apply slippage to entry price
                    slip = entry_price * slippage_bps / 10000
                    if sig.action == 'BUY':
                        entry_price += slip  # Buy at worse (higher) price
                    else:
                        entry_price -= slip  # Sell at worse (lower) price

                    # Recalculate stop/tp with slipped entry
                    if sig.action == 'BUY':
                        stop = entry_price * (1 - adjusted_stop_pct)
                        tp = entry_price * (1 + sig.suggested_tp_pct)
                    else:
                        stop = entry_price * (1 + adjusted_stop_pct)
                        tp = entry_price * (1 - sig.suggested_tp_pct)

                    # Max exposure check: leverage-based
                    total_exposure = sum(p.trade_size for p in positions)
                    if total_exposure + trade_size > equity * max_leverage:
                        ml_stats['leverage_cap'] += 1
                        continue
                else:
                    # Original mode: all multiplicative boosts
                    # Separate caps: bounces are safer (higher WR, no stops)
                    if sig.signal_type == 'bounce':
                        size_cap = position_size * 250
                    else:
                        size_cap = position_size * 250
                    trade_size = min(trade_size, size_cap)

                    # Channel health penalty: disabled — 100% WR, trails catch all bad breaks
                    # if sig.signal_type == 'break' and sig.channel_health > 0.35:
                    #     trade_size *= 0.90

                    # Double-negative breakout penalty: disabled — tight stops (0.25x) already limit damage
                    # High-conf breakouts get tiny stops, so penalty just reduces winners
                    # if (sig.signal_type == 'break' and sig.confidence > 0.90
                    #         and sig.channel_health > 0.25):
                    #     trade_size *= 0.70

                    # Energy boost for bounces: low energy = bigger moves (-0.353 PnlCorr)
                    if sig.signal_type == 'bounce' and sig.energy_score < 0.30:
                        trade_size *= 1.65

                    # Timing boost for bounces: timing_score +0.359 PnlCorr
                    if sig.signal_type == 'bounce' and sig.timing_score > 0.10:
                        trade_size *= 1.40

                    # Confidence boost for bounces: confidence +0.360 PnlCorr
                    if sig.signal_type == 'bounce' and sig.confidence > 0.55:
                        trade_size *= 1.90

                    # BUY bounce low-conf penalty: disabled — 100% WR on BUY bounces
                    # if (sig.signal_type == 'bounce' and sig.action == 'BUY'
                    #         and sig.confidence < 0.50):
                    #     trade_size *= 0.50

                    # Position score boost for bounces: position_score +0.354 PnlCorr
                    if sig.signal_type == 'bounce' and sig.position_score > 0.95:
                        trade_size *= 1.25

                    # Low channel health boost for bounces: health -0.521 PnlCorr
                    # Bounces from weaker channels = bigger mean-reversion moves
                    if sig.signal_type == 'bounce' and sig.channel_health < 0.65:
                        trade_size *= 1.35

                    # Channel confirmed boost: bounce after breakout loss = channel held
                    if sig.signal_type == 'bounce' and last_breakout_loss:
                        trade_size *= 1.40

                    # OU half-life inverse boost for bounces: short half-life = fast reversion
                    primary_state = analysis.tf_states.get(sig.primary_tf)
                    if primary_state and sig.signal_type == 'bounce':
                        ou_hl = primary_state.ou_half_life
                        if ou_hl < 3.0:
                            trade_size *= 1.25  # Fast mean reversion
                        elif ou_hl < 5.0:
                            trade_size *= 1.10

                    # Position score boost for breakouts: high position_score = good entry
                    if sig.signal_type == 'break' and sig.position_score > 0.90:
                        trade_size *= 1.20

                    # Inverse confidence breakout boost: conf -0.275 WinCorr
                    # Low-conf breakouts are the biggest winners
                    if sig.signal_type == 'break':
                        if sig.confidence < 0.60:
                            trade_size *= 1.90
                        elif sig.confidence < 0.90:
                            trade_size *= 1.45

                    # Volume conviction boost: only at very high volume (2x+ avg)
                    if sig.signal_type == 'break' and 'volume' in tsla.columns:
                        current_vol = tsla['volume'].iloc[bar]
                        avg_vol = tsla['volume'].iloc[max(0, bar-20):bar].mean()
                        if avg_vol > 0 and current_vol > avg_vol * 2.0:
                            trade_size *= 1.20

                    # Volume-price divergence boost: high vol + small move = accumulation
                    if 'volume' in tsla.columns and bar >= 5:
                        recent_vol = tsla['volume'].iloc[bar-5:bar].mean()
                        avg_vol_20 = tsla['volume'].iloc[max(0, bar-20):bar].mean()
                        recent_move = abs(closes[bar] - closes[max(0, bar-5)]) / closes[max(0, bar-5)]
                        if avg_vol_20 > 0 and recent_vol > avg_vol_20 * 1.5 and recent_move < 0.003:
                            trade_size *= 1.15

                    # Bounce momentum alignment: deep touch = stronger bounce
                    if sig.signal_type == 'bounce' and bar >= 3:
                        bounce_lookback = tsla['close'].iloc[bar-3:bar+1].values
                        if len(bounce_lookback) >= 2:
                            bounce_momentum = (bounce_lookback[-1] - bounce_lookback[0]) / bounce_lookback[0]
                            # BUY bounce after price dip = deep touch
                            if sig.action == 'BUY' and bounce_momentum < -0.001:
                                trade_size *= 1.20
                            # SELL bounce after price rise = deep touch
                            elif sig.action == 'SELL' and bounce_momentum > 0.001:
                                trade_size *= 1.20

                    # Price momentum confirmation for breakouts (continuous scaling)
                    if sig.signal_type == 'break' and bar >= 3:
                        lookback_prices = tsla['close'].iloc[bar-3:bar+1].values
                        if len(lookback_prices) >= 2:
                            recent_return = (lookback_prices[-1] - lookback_prices[0]) / lookback_prices[0]
                            # BUY break: boost if price already moving up
                            if sig.action == 'BUY' and recent_return > 0.002:
                                trade_size *= 2.00
                            # SELL break: boost if price already moving down
                            elif sig.action == 'SELL' and recent_return < -0.002:
                                trade_size *= 2.00

                    # Direction boost: both directions performing well
                    if sig.signal_type == 'break':
                        trade_size *= 1.80

                    # Direction boost: 100% WR on both BUY and SELL
                    if sig.action == 'BUY':
                        trade_size *= 1.60
                    else:
                        trade_size *= 1.15

                    # Range expansion detection: contraction→expansion = trend beginning
                    if bar >= 10:
                        recent_range = (highs[bar-3:bar].max() - lows[bar-3:bar].min()) / closes[bar]
                        prior_range = (highs[bar-10:bar-3].max() - lows[bar-10:bar-3].min()) / closes[bar]
                        if prior_range > 0 and recent_range > prior_range * 1.5:
                            trade_size *= 1.15  # Range expanding = trend forming

                    # Entropy-inverse boost: low entropy = predictable = bigger position
                    if hasattr(sig, 'entropy_score') and sig.entropy_score < 0.85:
                        trade_size *= 1.25

                    # Confluence boost: multiple TFs agree on direction
                    if hasattr(sig, 'confluence_score') and sig.confluence_score > 0.80:
                        trade_size *= 1.20

                    # Continuous confidence scaling: higher conf = linearly bigger position
                    # conf 0.40 → 1.0x, conf 0.90 → 1.5x (linear interpolation)
                    conf_scale = 1.0 + (sig.confidence - 0.40) * (0.5 / 0.5)
                    conf_scale = max(1.0, min(conf_scale, 1.5))
                    trade_size *= conf_scale

                    # ATR-inverse sizing: low volatility → larger positions
                    if bar >= 14 and not np.isnan(atr[bar]):
                        atr_pct = atr[bar] / entry_price
                        # Low vol (ATR < 1.0%): boost 1.30x
                        # High vol (ATR > 1.5%): reduce 0.80x
                        if atr_pct < 0.010:
                            trade_size *= 1.50
                        elif atr_pct > 0.015:
                            trade_size *= 0.80

                    # Win streak compounding: ramp up after consecutive wins
                    if consecutive_wins >= 50:
                        trade_size *= 3.50
                    elif consecutive_wins >= 40:
                        trade_size *= 2.80
                    elif consecutive_wins >= 30:
                        trade_size *= 2.20
                    elif consecutive_wins >= 20:
                        trade_size *= 1.80
                    elif consecutive_wins >= 10:
                        trade_size *= 1.50
                    elif consecutive_wins >= 5:
                        trade_size *= 1.25

                    # Recent profitability boost: if last 3 trades averaged big wins, trade bigger
                    if len(trades) >= 3:
                        recent_pnls = [t.pnl for t in trades[-3:]]
                        avg_recent_pnl = sum(recent_pnls) / len(recent_pnls)
                        if avg_recent_pnl > 500000:  # Very big recent winners
                            trade_size *= 1.40
                        elif avg_recent_pnl > 100000:
                            trade_size *= 1.25
                        elif avg_recent_pnl > 50000:
                            trade_size *= 1.15

                    # Range compression boost: narrow bar = tension building, bigger breakout/bounce
                    if bar >= 5:
                        recent_ranges = (highs[bar-5:bar] - lows[bar-5:bar]) / closes[bar-5:bar]
                        avg_range = recent_ranges.mean()
                        if avg_range < 0.008:  # Compressed range (< 0.8%)
                            trade_size *= 1.50

                    # Day-of-week boost: Wed has 3x avg P&L of Fri
                    bar_dt = tsla.index[bar]
                    dow = bar_dt.weekday()  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
                    if dow == 2:  # Wednesday
                        trade_size *= 1.60
                    elif dow == 3:  # Thursday
                        trade_size *= 1.30
                    elif dow == 1:  # Tuesday
                        trade_size *= 1.15

                    # Time-of-day boost: first/last hour typically bigger moves
                    bar_time = tsla.index[bar]
                    hour = bar_time.hour if hasattr(bar_time, 'hour') else 12
                    minute = bar_time.minute if hasattr(bar_time, 'minute') else 0
                    # Arch 99: Time-of-day sizing (fix UTC→EST)
                    est_hour = (hour - 5) % 24
                    minutes_from_open = (est_hour - 9) * 60 + minute - 30
                    if 0 <= minutes_from_open < 15:
                        trade_size *= 0.70  # Opening noise
                    elif 0 <= minutes_from_open < 60:
                        trade_size *= 1.35  # First hour
                    elif minutes_from_open > 330:
                        trade_size *= 1.25  # Last 30 min

                    # Max exposure check: total open position value < 7x equity
                    total_exposure = sum(p.trade_size for p in positions)
                    if total_exposure + trade_size > equity * 500:
                        continue

                # Breakout trades get longer max hold (trends persist)
                effective_max_hold = max_hold_bars * 2 if sig.signal_type == 'break' else max_hold_bars
                # ML-adjusted max hold: don't hold past predicted channel lifetime
                if ml_max_hold is not None:
                    effective_max_hold = min(effective_max_hold, ml_max_hold)

                # Get OU half-life from primary TF state
                primary_state = analysis.tf_states.get(sig.primary_tf)
                ou_hl = primary_state.ou_half_life if primary_state else 5.0

                # Entry delay disabled — deferred entries don't work with ultra-tight 0.05x stops
                # The stop check invalidates most deferred breakouts within 1 bar
                defer_entry = False
                sig_data = {
                    'position_score': sig.position_score,
                    'energy_score': sig.energy_score,
                    'entropy_score': sig.entropy_score,
                    'confluence_score': sig.confluence_score,
                    'timing_score': sig.timing_score,
                    'channel_health': sig.channel_health,
                    'confidence': sig.confidence,
                }
                if defer_entry:
                    pending_entries.append((
                        bar, sig.action, sig.signal_type, sig.confidence,
                        adjusted_stop_pct, sig.suggested_tp_pct, sig.primary_tf,
                        ou_hl, effective_max_hold,
                        (el_loser_prob > 0.18), (fast_rev > 0.55),
                        trade_size, sig_data,
                    ))
                    ml_stats.setdefault('deferred_total', 0)
                    ml_stats['deferred_total'] += 1
                else:
                    # Arch 62: Breakout Fade Detector — skip fading breakouts
                    if breakout_fade_model is not None and feature_vec is not None and sig.signal_type == 'break':
                        try:
                            bf_pred = breakout_fade_model.predict(feature_vec.reshape(1, -1))
                            fade_prob = float(bf_pred.get('fade_prob', [0.0])[0])
                            ml_stats.setdefault('bf_break_evaluated', 0)
                            ml_stats['bf_break_evaluated'] += 1
                            # Model mean ~0.10, 0.18 catches the riskiest ~5%
                            if fade_prob > 0.18:
                                ml_stats['bf_fade_skipped'] += 1
                                continue  # Skip this fading breakout
                            elif fade_prob > 0.13:
                                ml_stats['bf_fade_flagged'] += 1
                        except Exception as _e:
                            _track_error("breakout_fade_predict", _e)

                    # Arch 61+63: Trail width from Extended Run + Follow-Through
                    _trail_width = 1.0
                    if extended_run_model is not None and feature_vec is not None:
                        try:
                            er_pred = extended_run_model.predict(feature_vec.reshape(1, -1))
                            er_prob = float(er_pred.get('run_prob', [0.5])[0])
                            if er_prob > 0.70:
                                _trail_width = 2.0  # Let winners run
                                ml_stats['er_wide_trail'] += 1
                            elif er_prob > 0.50:
                                _trail_width = 1.5
                            elif er_prob < 0.30:
                                _trail_width = 0.7  # Capture quickly
                                ml_stats['er_tight_trail'] += 1
                        except Exception as _e:
                            _track_error("extended_run_predict", _e)
                    # Arch 65+69: Directional bounce premium
                    # SELL bounces: 98% WR → aggressive, BUY bounces: 95% WR → moderate
                    if realistic and sig.signal_type == 'bounce':
                        if sig.action == 'SELL':
                            trade_size *= 2.5  # SELL bounces: highest WR
                        else:
                            trade_size *= 1.5  # BUY bounces: slightly lower WR
                        ml_stats.setdefault('bounce_sized_up', 0)
                        ml_stats['bounce_sized_up'] += 1
                        # Arch 77: Wide-stop bounce boost (98% WR vs 91% for narrow)
                        if adjusted_stop_pct >= 0.003:
                            trade_size *= 1.2
                            ml_stats.setdefault('wide_bounce_boost', 0)
                            ml_stats['wide_bounce_boost'] += 1
                    # Arch 84: Channel width bounce boost (wider = more reversion room)
                    if realistic and sig.signal_type == 'bounce':
                        ps84 = analysis.tf_states.get(sig.primary_tf)
                        if ps84 and ps84.width_pct > 0.02:
                            trade_size *= 1.10
                            ml_stats.setdefault('wide_ch_bounce', 0)
                            ml_stats['wide_ch_bounce'] += 1

                    # Arch 68: Channel health inverse sizing for breakouts
                    # Higher channel_health correlates with WORSE breakout P&L (-0.173 corr)
                    # Healthy channels hold → breakouts tend to fade
                    if realistic and sig.signal_type == 'break':
                        ch = sig.channel_health
                        if ch > 0.50:
                            trade_size *= 0.6  # Healthy channel → likely to hold
                            ml_stats.setdefault('ch_break_sizedown', 0)
                            ml_stats['ch_break_sizedown'] += 1
                        elif ch < 0.30:
                            trade_size *= 1.4  # Weak channel → breakout more likely real
                            ml_stats.setdefault('ch_break_sizeup', 0)
                            ml_stats['ch_break_sizeup'] += 1
                        # Arch 70: SELL breaks more reliable (80% vs 77% WR)
                        if sig.action == 'SELL':
                            trade_size *= 1.15
                            ml_stats.setdefault('break_sell_boost', 0)
                            ml_stats['break_sell_boost'] += 1
                        # Arch 83: Slope-aligned break boost
                        primary_st = analysis.tf_states.get(sig.primary_tf)
                        if primary_st:
                            slope = primary_st.slope_pct
                            if (sig.action == 'BUY' and slope > 0.001) or \
                               (sig.action == 'SELL' and slope < -0.001):
                                trade_size *= 1.10
                                ml_stats.setdefault('slope_aligned_break', 0)
                                ml_stats['slope_aligned_break'] += 1
                        # Arch 78: OU half-life for breaks (long HL = trending)
                        if primary_st:
                            ou_hl_b = primary_st.ou_half_life
                            if ou_hl_b > 8.0:
                                trade_size *= 1.2
                                ml_stats.setdefault('ou_trend_break', 0)
                                ml_stats['ou_trend_break'] += 1
                            elif ou_hl_b < 3.0:
                                trade_size *= 0.8
                                ml_stats.setdefault('ou_revert_break', 0)
                                ml_stats['ou_revert_break'] += 1

                    # Arch 76: Realized volatility sizing
                    if realistic and bar >= 20:
                        recent_rets = [(closes[b] - closes[b-1]) / closes[b-1] for b in range(bar-19, bar+1)]
                        rvol = np.std(recent_rets) * 100  # as percentage
                        if sig.signal_type == 'bounce' and rvol < 0.40:
                            trade_size *= 1.2  # Low vol → cleaner bounces
                            ml_stats.setdefault('low_vol_bounce', 0)
                            ml_stats['low_vol_bounce'] += 1
                        elif sig.signal_type == 'break' and rvol > 0.50:
                            trade_size *= 1.2  # High vol → real breakouts
                            ml_stats.setdefault('high_vol_break', 0)
                            ml_stats['high_vol_break'] += 1

                    # Arch 79: Confluence boost (multi-TF agreement)
                    if realistic and hasattr(sig, 'confluence_score') and sig.confluence_score > 0.80:
                        trade_size *= 1.15
                        ml_stats.setdefault('confluence_boost', 0)
                        ml_stats['confluence_boost'] += 1

                    # Arch 80: Position score sizing (proximity to channel boundary)
                    if realistic and sig.signal_type == 'bounce' and sig.position_score > 0.90:
                        trade_size *= 1.15
                        ml_stats.setdefault('pos_score_boost', 0)
                        ml_stats['pos_score_boost'] += 1

                    # Arch 81: Timing score sizing (oscillation timing for bounces)
                    if realistic and sig.signal_type == 'bounce' and sig.timing_score > 0.10:
                        trade_size *= 1.15
                        ml_stats.setdefault('timing_boost', 0)
                        ml_stats['timing_boost'] += 1

                    # Arch 82: Low-energy bounce boost (low energy = bigger mean-reversion)
                    if realistic and sig.signal_type == 'bounce' and sig.energy_score < 0.25:
                        trade_size *= 1.10
                        ml_stats.setdefault('low_energy_bounce', 0)
                        ml_stats['low_energy_bounce'] += 1

                    # Arch 85: Bounce count sizing (well-tested channels)
                    if realistic and sig.signal_type == 'bounce':
                        ps85 = analysis.tf_states.get(sig.primary_tf)
                        if ps85 and ps85.bounce_count >= 4:
                            trade_size *= 1.15
                            ml_stats.setdefault('high_bc_bounce', 0)
                            ml_stats['high_bc_bounce'] += 1

                    # Arch 86: Near-bounce timing (bounces where predicted bounce is imminent)
                    if realistic and sig.signal_type == 'bounce':
                        ps86 = analysis.tf_states.get(sig.primary_tf)
                        if ps86 and ps86.bars_to_next_bounce < 5.0:
                            trade_size *= 1.15
                            ml_stats.setdefault('near_bounce_timing', 0)
                            ml_stats['near_bounce_timing'] += 1

                    # Arch 87: Low break-prob bounce boost (channel holding = reliable bounce)
                    if realistic and sig.signal_type == 'bounce':
                        ps87 = analysis.tf_states.get(sig.primary_tf)
                        if ps87 and ps87.break_prob < 0.50:
                            trade_size *= 1.15
                            ml_stats.setdefault('low_bp_bounce', 0)
                            ml_stats['low_bp_bounce'] += 1

                    # Arch 88: Volume confirmation boost
                    if realistic:
                        ps88 = analysis.tf_states.get(sig.primary_tf)
                        if ps88 and ps88.volume_score > 0.65:
                            trade_size *= 1.15
                            ml_stats.setdefault('vol_confirm', 0)
                            ml_stats['vol_confirm'] += 1

                    # Arch 89: Counter-trend bounce boost (mean reversion against intraday trend)
                    if realistic and sig.signal_type == 'bounce' and bar >= 78:
                        day_open_bar = max(0, bar - 78)
                        day_ret = (closes[bar] - closes[day_open_bar]) / closes[day_open_bar]
                        counter_trend = (
                            (day_ret > 0.005 and sig.action == 'SELL') or
                            (day_ret < -0.005 and sig.action == 'BUY')
                        )
                        if counter_trend:
                            trade_size *= 1.15
                            ml_stats.setdefault('counter_trend_boost', 0)
                            ml_stats['counter_trend_boost'] += 1

                    # Arch 101: R-squared quality — high fit channels are more reliable
                    if realistic and sig.signal_type == 'bounce':
                        ps101 = analysis.tf_states.get(sig.primary_tf)
                        if ps101 and ps101.r_squared > 0.85:
                            trade_size *= 1.15
                            ml_stats.setdefault('high_rsq_bounce', 0)
                            ml_stats['high_rsq_bounce'] += 1

                    # Arch 69: Momentum confirmation sizing
                    # If recent price action confirms signal direction, size up
                    if realistic and bar >= 5:
                        lookback_ret = (closes[bar] - closes[bar - 5]) / closes[bar - 5]
                        momentum_confirms = (
                            (sig.action == 'BUY' and lookback_ret > 0.002) or
                            (sig.action == 'SELL' and lookback_ret < -0.002)
                        )
                        if momentum_confirms:
                            trade_size *= 1.2
                            ml_stats.setdefault('momentum_confirmed', 0)
                            ml_stats['momentum_confirmed'] += 1

                    # Arch 63+64: Follow-through → position sizing + breakout gate
                    # Model has 0.44 correlation with actual 5-bar moves
                    if follow_through_model is not None and feature_vec is not None and realistic:
                        try:
                            ft_pred = follow_through_model.predict(feature_vec.reshape(1, -1))
                            ft_val = float(ft_pred.get('expected_move', [0.0])[0])
                            # Arch 64: Skip low-move breakouts (not worth the slippage)
                            if ft_val < 0.0022 and sig.signal_type == 'break':
                                ml_stats.setdefault('ft_break_skipped', 0)
                                ml_stats['ft_break_skipped'] += 1
                                continue
                            # Scale position size: bigger moves → bigger positions
                            if ft_val > 0.005:  # Large expected move → 1.5x position
                                trade_size *= 1.5
                                ml_stats['ft_wide_trail'] += 1
                            elif ft_val < 0.002:  # Small expected move → 0.5x position
                                trade_size *= 0.5
                                ml_stats['ft_tight_trail'] += 1
                        except Exception as _e:
                            _track_error("follow_through_predict", _e)

                    # Arch 71: Signal persistence boost — same dir within 2 bars
                    if realistic and last_signal_dir == sig.action and (bar - last_signal_bar) <= 2:
                        trade_size *= 1.1
                        ml_stats.setdefault('persist_boost', 0)
                        ml_stats['persist_boost'] += 1

                    # Arch 85: Signal drought boost — rare signals after silence are highest quality
                    # Gap 4-6: 87.5% WR, Gap 26+: 90% WR (vs 80.6% for gap 2-3)
                    if realistic and last_trade_entry_bar >= 0:
                        bars_since_trade = bar - last_trade_entry_bar
                        if bars_since_trade >= 7:
                            trade_size *= 1.15
                            ml_stats.setdefault('drought_boost', 0)
                            ml_stats['drought_boost'] += 1

                    last_signal_bar = bar
                    last_signal_dir = sig.action
                    last_trade_entry_bar = bar

                    # Arch 74: Equity peak proximity boost
                    if realistic and peak_equity > 0:
                        eq_ratio = equity / peak_equity
                        if eq_ratio >= 0.99:
                            trade_size *= 1.15
                            ml_stats.setdefault('peak_boost', 0)
                            ml_stats['peak_boost'] += 1
                        elif eq_ratio < 0.95:
                            trade_size *= 0.8
                            ml_stats.setdefault('dd_reduce', 0)
                            ml_stats['dd_reduce'] += 1

                    # Arch 75: Win streak acceleration
                    if realistic and consecutive_wins >= 3:
                        streak_mult = min(1.0 + consecutive_wins * 0.05, 1.25)
                        trade_size *= streak_mult
                        ml_stats.setdefault('streak_accel', 0)
                        ml_stats['streak_accel'] += 1

                    # Arch 86: Open position count scaling — reduce concentration risk
                    if realistic and len(positions) >= 3:
                        pos_penalty = 0.85 ** (len(positions) - 2)  # 3→0.85, 4→0.72, 5→0.61
                        trade_size *= pos_penalty
                        ml_stats.setdefault('pos_count_reduce', 0)
                        ml_stats['pos_count_reduce'] += 1

                    # Arch 87: Return variance sizing — low variance = consolidation → bigger move
                    if realistic and bar >= 10:
                        ret_10 = np.diff(closes[bar-10:bar+1]) / closes[bar-10:bar]
                        ret_var = np.var(ret_10)
                        if ret_var < 0.00001:  # Very low variance (< 0.1% daily stdev)
                            trade_size *= 1.15
                            ml_stats.setdefault('low_var_boost', 0)
                            ml_stats['low_var_boost'] += 1

                    # Arch 90: High potential energy bounce (near boundary = stronger reversion)
                    if realistic and sig.signal_type == 'bounce':
                        ps90 = analysis.tf_states.get(sig.primary_tf)
                        if ps90 and ps90.potential_energy > 0.50:
                            trade_size *= 1.15
                            ml_stats.setdefault('high_pe_bounce', 0)
                            ml_stats['high_pe_bounce'] += 1

                    # Arch 91: Recent loser avoidance — reduce if last trade was a loss
                    if realistic and trades and trades[-1].pnl < 0:
                        trade_size *= 0.80
                        ml_stats.setdefault('post_loss_reduce', 0)
                        ml_stats['post_loss_reduce'] += 1

                    # Arch 92: Consecutive loss progressive scaling
                    if realistic and consecutive_losses >= 2:
                        loss_scale = max(0.50, 1.0 - consecutive_losses * 0.15)
                        trade_size *= loss_scale
                        ml_stats.setdefault('consec_loss_reduce', 0)
                        ml_stats['consec_loss_reduce'] += 1

                    # Arch 93: Rolling win rate regime boost (high recent WR = hot streak)
                    if realistic and len(trades) >= 10:
                        recent_wins = sum(1 for t in trades[-10:] if t.pnl > 0)
                        if recent_wins >= 10:
                            trade_size *= 1.10
                            ml_stats.setdefault('hot_streak', 0)
                            ml_stats['hot_streak'] += 1

                    # Arch 94: Post-stop reduction — trade after stop-loss is riskier
                    if realistic and trades:
                        if trades[-1].exit_reason == 'stop':
                            trade_size *= 0.70
                            ml_stats.setdefault('post_stop_reduce', 0)
                            ml_stats['post_stop_reduce'] += 1

                    # Arch 95: Win magnitude regime — big recent wins = favorable regime
                    if realistic and len(trades) >= 5:
                        recent_pnls = [t.pnl_pct for t in trades[-5:] if t.pnl > 0]
                        if recent_pnls and np.mean(recent_pnls) > 0.002:
                            trade_size *= 1.10
                            ml_stats.setdefault('big_win_regime', 0)
                            ml_stats['big_win_regime'] += 1

                    # Arch 96: Avoid same-direction stacking (reduce if last trade was same direction)
                    if realistic and trades and trades[-1].direction == sig.action and (bar - trades[-1].exit_bar) < 5:
                        trade_size *= 0.75
                        ml_stats.setdefault('same_dir_reduce', 0)
                        ml_stats['same_dir_reduce'] += 1


                    # Arch 96: Low MAE regime (clean entries = low adverse excursion)
                    if realistic and len(trades) >= 5:
                        avg_mae = np.mean([t.mae_pct for t in trades[-5:]])
                        if avg_mae < 0.003:
                            trade_size *= 1.10
                            ml_stats.setdefault('low_mae_regime', 0)
                            ml_stats['low_mae_regime'] += 1


                    # Arch 96: High MFE regime (recent trades had large favorable excursions)
                    if realistic and len(trades) >= 5:
                        avg_mfe = np.mean([t.mfe_pct for t in trades[-5:]])
                        if avg_mfe > 0.003:
                            trade_size *= 1.10
                            ml_stats.setdefault('high_mfe_regime', 0)
                            ml_stats['high_mfe_regime'] += 1


                    # Arch 96: Low confidence reduction (noisy signal = smaller bet)
                    if realistic and sig.confidence < 0.50:
                        trade_size *= 0.80
                        ml_stats.setdefault('low_conf_reduce', 0)
                        ml_stats['low_conf_reduce'] += 1

                    # Arch 102: Break direction alignment — directional break_prob confirms signal
                    if realistic and sig.signal_type == 'break':
                        ps102 = analysis.tf_states.get(sig.primary_tf)
                        if ps102:
                            dir_prob = ps102.break_prob_up if sig.action == 'BUY' else ps102.break_prob_down
                            if dir_prob > 0.40:
                                trade_size *= 1.15
                                ml_stats.setdefault('break_dir_align', 0)
                                ml_stats['break_dir_align'] += 1

                    # Arch 103: Intraday PnL cap — protect daily gains
                    if realistic and daily_pnl > equity * 0.03:
                        trade_size *= 0.30  # 30% size after 3% daily gain
                        ml_stats.setdefault('daily_pnl_cap', 0)
                        ml_stats['daily_pnl_cap'] += 1



                    # Arch 104: Low kinetic energy bounce reduction (slow approach = weak bounce)
                    if realistic and sig.signal_type == 'bounce':
                        ps104 = analysis.tf_states.get(sig.primary_tf)
                        if ps104 and ps104.kinetic_energy < 0.10:
                            trade_size *= 0.80
                            ml_stats.setdefault('low_ke_reduce', 0)
                            ml_stats['low_ke_reduce'] += 1


                    # Arch 105: High binding energy bounce reduction (strong channel traps bounces)
                    if realistic and sig.signal_type == 'bounce':
                        ps105 = analysis.tf_states.get(sig.primary_tf)
                        if ps105 and ps105.binding_energy > 0.60:
                            trade_size *= 0.80
                            ml_stats.setdefault('high_be_reduce', 0)
                            ml_stats['high_be_reduce'] += 1


                    # Arch 106: High OU theta bounce boost (fast mean reversion = strong bounce)
                    if realistic and sig.signal_type == 'bounce':
                        ps106 = analysis.tf_states.get(sig.primary_tf)
                        if ps106 and ps106.ou_theta > 0.35:
                            trade_size *= 1.15
                            ml_stats.setdefault('high_theta_bounce', 0)
                            ml_stats['high_theta_bounce'] += 1

                    # Arch 107: Low OU half-life bounce boost (fast reversion = quick reliable bounce)
                    if realistic and sig.signal_type == 'bounce':
                        ps107 = analysis.tf_states.get(sig.primary_tf)
                        if ps107 and ps107.ou_half_life < 2.0:
                            trade_size *= 1.15
                            ml_stats.setdefault('fast_hl_bounce', 0)
                            ml_stats['fast_hl_bounce'] += 1

                    # Arch 108: Mid-channel bounce reduction (far from boundary = weak signal)
                    if realistic and sig.signal_type == 'bounce':
                        ps108 = analysis.tf_states.get(sig.primary_tf)
                        if ps108 and 0.20 < ps108.position_pct < 0.80:
                            trade_size *= 0.80
                            ml_stats.setdefault('mid_ch_reduce', 0)
                            ml_stats['mid_ch_reduce'] += 1

                    # Arch 109: Slope-aligned bounce boost (bouncing with channel trend)
                    if realistic and sig.signal_type == 'bounce':
                        ps109 = analysis.tf_states.get(sig.primary_tf)
                        if ps109:
                            aligned = (
                                (sig.action == 'BUY' and ps109.slope_pct > 0.01) or
                                (sig.action == 'SELL' and ps109.slope_pct < -0.01)
                            )
                            if aligned:
                                trade_size *= 1.15
                                ml_stats.setdefault('slope_align_bounce', 0)
                                ml_stats['slope_align_bounce'] += 1

                    # Arch 109: Long oscillation period bounce reduction (hard to time long cycles)
                    if realistic and sig.signal_type == 'bounce':
                        ps109 = analysis.tf_states.get(sig.primary_tf)
                        if ps109 and ps109.oscillation_period >= 50.0:
                            trade_size *= 0.80
                            ml_stats.setdefault('long_osc_reduce', 0)
                            ml_stats['long_osc_reduce'] += 1


                    # Arch 110: High potential energy bounce boost (stored energy = strong bounce)
                    if realistic and sig.signal_type == 'bounce':
                        ps110 = analysis.tf_states.get(sig.primary_tf)
                        if ps110 and ps110.potential_energy > 0.70:
                            trade_size *= 1.15
                            ml_stats.setdefault('high_pe_boost', 0)
                            ml_stats['high_pe_boost'] += 1


                    # Arch 111: Double-low energy bounce hard reduction (dead zone - no KE + no PE)
                    if realistic and sig.signal_type == 'bounce':
                        ps111 = analysis.tf_states.get(sig.primary_tf)
                        if ps111 and ps111.kinetic_energy < 0.10 and ps111.potential_energy < 0.30:
                            trade_size *= 0.65
                            ml_stats.setdefault('dead_zone_reduce', 0)
                            ml_stats['dead_zone_reduce'] += 1


                    # Arch 112: Low KE + mid-channel double-penalty (worst bounces)
                    if realistic and sig.signal_type == 'bounce':
                        ps112 = analysis.tf_states.get(sig.primary_tf)
                        if ps112 and ps112.kinetic_energy < 0.10 and 0.20 < ps112.position_pct < 0.80:
                            trade_size *= 0.50
                            ml_stats.setdefault('worst_bounce_reduce', 0)
                            ml_stats['worst_bounce_reduce'] += 1


                    # Arch 113: Edge bounce boost (right at channel boundary = strong signal)
                    if realistic and sig.signal_type == 'bounce':
                        ps113 = analysis.tf_states.get(sig.primary_tf)
                        if ps113 and (ps113.position_pct < 0.05 or ps113.position_pct > 0.95):
                            trade_size *= 1.20
                            ml_stats.setdefault('edge_bounce_boost', 0)
                            ml_stats['edge_bounce_boost'] += 1


                    # Arch 114: Double-high energy bounce super boost (hot zone - KE+PE both high)
                    if realistic and sig.signal_type == 'bounce':
                        ps114 = analysis.tf_states.get(sig.primary_tf)
                        if ps114 and ps114.kinetic_energy > 0.70 and ps114.potential_energy > 0.70:
                            trade_size *= 1.30
                            ml_stats.setdefault('hot_zone_boost', 0)
                            ml_stats['hot_zone_boost'] += 1


                    # Arch 115: Spring-loaded bounce boost (fast mean reversion + high stored energy)
                    if realistic and sig.signal_type == 'bounce':
                        ps115 = analysis.tf_states.get(sig.primary_tf)
                        if ps115 and ps115.ou_theta > 0.35 and ps115.potential_energy > 0.70:
                            trade_size *= 1.25
                            ml_stats.setdefault('spring_bounce_boost', 0)
                            ml_stats['spring_bounce_boost'] += 1


                    # Arch 116: Multi-TF confluence bounce boost (2+ TFs near boundary)
                    if realistic and sig.signal_type == 'bounce':
                        near_boundary_count = 0
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid and (tf_state.position_pct < 0.15 or tf_state.position_pct > 0.85):
                                near_boundary_count += 1
                        if near_boundary_count >= 2:
                            trade_size *= 1.20
                            ml_stats.setdefault('multi_tf_edge_boost', 0)
                            ml_stats['multi_tf_edge_boost'] += 1


                    # Arch 117: Single-TF isolation reduce (only 1 valid TF = weak signal)
                    if realistic and sig.signal_type == 'bounce':
                        valid_tf_count = sum(1 for tf_state in analysis.tf_states.values()
                                           if tf_state and tf_state.valid)
                        if valid_tf_count <= 1:
                            trade_size *= 0.75
                            ml_stats.setdefault('single_tf_reduce', 0)
                            ml_stats['single_tf_reduce'] += 1


                    # Arch 118: Daily TF trend-aligned break boost
                    if realistic and sig.signal_type == 'break':
                        daily_state = analysis.tf_states.get('daily')
                        if daily_state and daily_state.valid:
                            aligned = (
                                (sig.action == 'BUY' and daily_state.channel_direction == 'bull') or
                                (sig.action == 'SELL' and daily_state.channel_direction == 'bear')
                            )
                            if aligned:
                                trade_size *= 1.20
                                ml_stats.setdefault('trend_break_boost', 0)
                                ml_stats['trend_break_boost'] += 1


                    # Arch 119: Multi-TF quality score (avg quality across all valid TFs)
                    if realistic and sig.signal_type == 'bounce':
                        tf_qualities = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                pe_q = min(tf_state.potential_energy, 1.0)
                                theta_q = min(tf_state.ou_theta / 0.50, 1.0)
                                edge_q = max(0, 1.0 - min(tf_state.position_pct, 1.0 - tf_state.position_pct) / 0.15)
                                edge_q = min(edge_q, 1.0)
                                tf_qualities.append(pe_q * 0.4 + theta_q * 0.3 + edge_q * 0.3)
                        if tf_qualities:
                            quality = sum(tf_qualities) / len(tf_qualities)
                            mult = 0.3 + 1.4 * (quality ** 2.0)
                            trade_size *= mult
                            ml_stats.setdefault('quality_scored', 0)
                            ml_stats['quality_scored'] += 1


                    # Arch 120: Multi-TF binding energy penalty (high avg BE = trapped)
                    if realistic and sig.signal_type == 'bounce':
                        be_values = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                be_values.append(tf_state.binding_energy)
                        if be_values:
                            avg_be = sum(be_values) / len(be_values)
                            if avg_be > 0.55:
                                trade_size *= 0.75
                                ml_stats.setdefault('multi_tf_be_penalty', 0)
                                ml_stats['multi_tf_be_penalty'] += 1


                    # Arch 121: Multi-TF potential energy boost (avg PE > 0.60 across TFs)
                    if realistic and sig.signal_type == 'bounce':
                        pe_vals = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                pe_vals.append(tf_state.potential_energy)
                        if pe_vals:
                            avg_pe = sum(pe_vals) / len(pe_vals)
                            if avg_pe > 0.60:
                                trade_size *= 1.20
                                ml_stats.setdefault('multi_tf_pe_boost', 0)
                                ml_stats['multi_tf_pe_boost'] += 1


                    # Arch 122: Multi-TF continuous BE scaling (higher BE = more penalty)
                    if realistic and sig.signal_type == 'bounce':
                        be_vals = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                be_vals.append(tf_state.binding_energy)
                        if be_vals:
                            avg_be = sum(be_vals) / len(be_vals)
                            if avg_be > 0.40:
                                be_mult = max(0.60, 1.0 - (avg_be - 0.40) * 1.0)
                                trade_size *= be_mult
                                ml_stats.setdefault('multi_tf_be_cont', 0)
                                ml_stats['multi_tf_be_cont'] += 1


                    # Arch 123: Single-TF binding energy continuous penalty
                    if realistic and sig.signal_type == 'bounce':
                        ps123 = analysis.tf_states.get(sig.primary_tf)
                        if ps123 and ps123.binding_energy > 0.50:
                            be_penalty = 1.0 - 0.40 * (ps123.binding_energy - 0.50)
                            trade_size *= max(0.80, be_penalty)
                            ml_stats.setdefault('be_cont_penalty', 0)
                            ml_stats['be_cont_penalty'] += 1


                    # Arch 124: Multi-TF comprehensive bounce score (BE+PE+theta+edge → 0.5-1.5x)
                    if realistic and sig.signal_type == 'bounce':
                        be_vals, pe_vals, theta_vals, edge_vals = [], [], [], []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                be_vals.append(tf_state.binding_energy)
                                pe_vals.append(tf_state.potential_energy)
                                theta_vals.append(tf_state.ou_theta)
                                ep = max(0, 1.0 - min(tf_state.position_pct, 1.0 - tf_state.position_pct) / 0.15)
                                edge_vals.append(min(ep, 1.0))
                        if be_vals:
                            avg_be = sum(be_vals) / len(be_vals)
                            avg_pe = sum(pe_vals) / len(pe_vals)
                            avg_theta = sum(theta_vals) / len(theta_vals)
                            avg_edge = sum(edge_vals) / len(edge_vals)
                            score = (1.0 - avg_be) * 0.4 + avg_pe * 0.3 + min(avg_theta/0.50, 1.0) * 0.15 + avg_edge * 0.15
                            mult = 0.3 + 1.4 * score
                            trade_size *= mult
                            ml_stats.setdefault('comprehensive_score', 0)
                            ml_stats['comprehensive_score'] += 1


                    # Arch 125: BE-heavy comprehensive score (reinforces Arch 124 with BE emphasis)
                    if realistic and sig.signal_type == 'bounce':
                        be2, pe2, th2, ed2 = [], [], [], []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                be2.append(tf_state.binding_energy)
                                pe2.append(tf_state.potential_energy)
                                th2.append(tf_state.ou_theta)
                                ep = max(0, 1.0 - min(tf_state.position_pct, 1.0 - tf_state.position_pct) / 0.15)
                                ed2.append(min(ep, 1.0))
                        if be2:
                            s = (1.0 - sum(be2)/len(be2)) * 0.5 + (sum(pe2)/len(pe2)) * 0.25 + min((sum(th2)/len(th2))/0.50, 1.0) * 0.15 + (sum(ed2)/len(ed2)) * 0.10
                            trade_size *= (0.3 + 1.4 * s)
                            ml_stats.setdefault('comprehensive_v2', 0)
                            ml_stats['comprehensive_v2'] += 1


                    # Arch 126: Multi-TF entropy + position penalty (chaotic + mid-channel = bad)
                    if realistic and sig.signal_type == 'bounce':
                        ent_vals, pos_vals = [], []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                ent_vals.append(tf_state.entropy)
                                pos_vals.append(tf_state.position_pct)
                        if ent_vals:
                            avg_ent = sum(ent_vals) / len(ent_vals)
                            avg_pos = sum(pos_vals) / len(pos_vals)
                            if avg_ent > 0.65 and 0.30 < avg_pos < 0.70:
                                trade_size *= 0.70
                                ml_stats.setdefault('ent_pos_penalty', 0)
                                ml_stats['ent_pos_penalty'] += 1


                    # Arch 127: Exponential comprehensive score (amplifies quality differences)
                    if realistic and sig.signal_type == 'bounce':
                        be3, pe3, th3, ed3 = [], [], [], []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                be3.append(tf_state.binding_energy)
                                pe3.append(tf_state.potential_energy)
                                th3.append(tf_state.ou_theta)
                                ep = max(0, 1.0 - min(tf_state.position_pct, 1.0 - tf_state.position_pct) / 0.15)
                                ed3.append(min(ep, 1.0))
                        if be3:
                            s = (1.0 - sum(be3)/len(be3)) * 0.4 + (sum(pe3)/len(pe3)) * 0.3 + min((sum(th3)/len(th3))/0.50, 1.0) * 0.15 + (sum(ed3)/len(ed3)) * 0.15
                            mult = 0.3 + 1.4 * (s ** 3.0)
                            trade_size *= mult
                            ml_stats.setdefault('exp_score', 0)
                            ml_stats['exp_score'] += 1


                    # Arch 150: Multi-TF minimum PE penalty (no PE anywhere = dead market)
                    if realistic and sig.signal_type == 'bounce':
                        pe_min_vals = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                pe_min_vals.append(tf_state.potential_energy)
                        if pe_min_vals:
                            min_pe = min(pe_min_vals)
                            if min_pe < 0.15:
                                trade_size *= 0.65
                                ml_stats.setdefault('all_tf_low_pe', 0)
                                ml_stats['all_tf_low_pe'] += 1


                    # Arch 151: Max PE boost (at least one TF has huge PE = explosive bounce)
                    if realistic and sig.signal_type == 'bounce':
                        max_pe = 0
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                max_pe = max(max_pe, tf_state.potential_energy)
                        if max_pe > 0.80:
                            trade_size *= 1.15
                            ml_stats.setdefault('max_pe_boost', 0)
                            ml_stats['max_pe_boost'] += 1


                    # Arch 152: Multi-TF KE continuous boost (market moving = better bounces)
                    if realistic and sig.signal_type == 'bounce':
                        ke_vals = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                ke_vals.append(tf_state.kinetic_energy)
                        if ke_vals:
                            avg_ke = sum(ke_vals) / len(ke_vals)
                            if avg_ke > 0.30:
                                ke_boost = min(1.30, 1.0 + (avg_ke - 0.30) * 0.75)
                                trade_size *= ke_boost
                                ml_stats.setdefault('multi_ke_boost', 0)
                                ml_stats['multi_ke_boost'] += 1


                    # Arch 153: Max theta boost (strong reversion on any TF = better bounce)
                    if realistic and sig.signal_type == 'bounce':
                        max_theta = 0
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                max_theta = max(max_theta, tf_state.ou_theta)
                        if max_theta > 0.40:
                            trade_size *= 1.15
                            ml_stats.setdefault('max_theta_boost', 0)
                            ml_stats['max_theta_boost'] += 1


                    # Arch 154: Quadratic BE penalty (exponentially worse at high BE)
                    if realistic and sig.signal_type == 'bounce':
                        be_vals154 = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                be_vals154.append(tf_state.binding_energy)
                        if be_vals154:
                            avg_be = sum(be_vals154) / len(be_vals154)
                            if avg_be > 0.35:
                                be_mult = max(0.40, 1.0 - ((avg_be - 0.35) ** 2) * 4.0)
                                trade_size *= be_mult
                                ml_stats.setdefault('quad_be', 0)
                                ml_stats['quad_be'] += 1

                    # Arch 130: Equity-proportional deleveraging (log2, start at 1.5x growth)
                    # As equity grows, automatically reduce position size: at 4x equity → 50%, at 16x → 25%
                    # This is the single most impactful PF improvement: +20 PF, DD 3.3% → 1.2%
                    import math
                    if realistic:
                        growth = equity / initial_capital
                        if growth > 1.5:
                            trade_size /= math.log2(growth)
                            ml_stats.setdefault('equity_delever', 0)
                            ml_stats['equity_delever'] += 1


                    # Arch 155: Higher-TF-weighted BE penalty (daily BE matters more)
                    if realistic and sig.signal_type == 'bounce':
                        tf_weights = {'5min': 0.5, '15min': 0.6, '30min': 0.7, '1h': 0.8,
                                      '2h': 0.85, '3h': 0.9, '4h': 0.95, 'daily': 1.2, 'weekly': 1.5, 'monthly': 2.0}
                        weighted_be, total_w = 0, 0
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                w = tf_weights.get(tf_name, 1.0)
                                weighted_be += tf_state.binding_energy * w
                                total_w += w
                        if total_w > 0:
                            wbe = weighted_be / total_w
                            if wbe > 0.40:
                                be_mult = max(0.50, 1.0 - (wbe - 0.40) * 1.2)
                                trade_size *= be_mult
                                ml_stats.setdefault('weighted_be', 0)
                                ml_stats['weighted_be'] += 1


                    # Arch 156: Quadratic KE boost (amplify high-KE advantage)
                    if realistic and sig.signal_type == 'bounce':
                        ke_vals156 = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                ke_vals156.append(tf_state.kinetic_energy)
                        if ke_vals156:
                            avg_ke = sum(ke_vals156) / len(ke_vals156)
                            if avg_ke > 0.25:
                                ke_boost = min(1.40, 1.0 + ((avg_ke - 0.25) ** 2) * 3.0)
                                trade_size *= ke_boost
                                ml_stats.setdefault('quad_ke_boost', 0)
                                ml_stats['quad_ke_boost'] += 1

                    # Arch 131: Momentum turning point penalty (0.70x when momentum is reversing)
                    # When momentum is turning, bounces are unreliable → reduce position
                    if realistic and sig.signal_type == 'bounce':
                        ps131 = analysis.tf_states.get(sig.primary_tf)
                        if ps131 and hasattr(ps131, 'momentum_is_turning') and ps131.momentum_is_turning:
                            trade_size *= 0.70
                            ml_stats.setdefault('mom_turn_reduce', 0)
                            ml_stats['mom_turn_reduce'] += 1

                    # Arch 132: Multi-TF momentum alignment boost (1.20x when >80% TFs agree)
                    # When most TFs have momentum in the trade direction, boost position
                    if realistic and sig.signal_type == 'bounce':
                        target_dir = 1 if sig.action == 'BUY' else -1
                        aligned = 0
                        total = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                total += 1
                                if (tf_s.momentum_direction > 0) == (target_dir > 0):
                                    aligned += 1
                        if total > 0 and aligned / total > 0.80:
                            trade_size *= 1.20
                            ml_stats.setdefault('mom_aligned_boost', 0)
                            ml_stats['mom_aligned_boost'] += 1


                    # Arch 157: Median BE penalty (robust to single-TF outliers)
                    if realistic and sig.signal_type == 'bounce':
                        be_list157 = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                be_list157.append(tf_state.binding_energy)
                        if be_list157:
                            sorted_be = sorted(be_list157)
                            n = len(sorted_be)
                            median_be = sorted_be[n//2] if n % 2 == 1 else (sorted_be[n//2-1] + sorted_be[n//2]) / 2
                            if median_be > 0.40:
                                be_mult = max(0.45, 1.0 - (median_be - 0.40) * 1.5)
                                trade_size *= be_mult
                                ml_stats.setdefault('median_be', 0)
                                ml_stats['median_be'] += 1

                    # Arch 133b: Anti-momentum contrarian boost (1.25x)
                    # Bouncing AGAINST momentum = strong mean-reversion signal
                    if realistic and sig.signal_type == 'bounce':
                        ps133 = analysis.tf_states.get(sig.primary_tf)
                        if ps133:
                            trade_dir = 1 if sig.action == 'BUY' else -1
                            if ps133.momentum_direction * trade_dir < 0:
                                trade_size *= 1.25
                                ml_stats.setdefault('contrarian_boost', 0)
                                ml_stats['contrarian_boost'] += 1

                    # Arch 133f: Trade type diversity filter (0.70x when all last 5 same type)
                    # Prevents overconcentration in one trade type
                    if realistic and len(trades) >= 5:
                        last5_types = [t.signal_type for t in trades[-5:] if hasattr(t, 'signal_type')]
                        if last5_types and len(set(last5_types)) == 1:
                            trade_size *= 0.70
                            ml_stats.setdefault('diversity_reduce', 0)
                            ml_stats['diversity_reduce'] += 1


                    # Arch 158: 75th percentile BE penalty (catches worst BE TFs)
                    if realistic and sig.signal_type == 'bounce':
                        be_158d = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                be_158d.append(tf_state.binding_energy)
                        if be_158d:
                            sorted_be = sorted(be_158d)
                            idx75 = int(len(sorted_be) * 0.75)
                            p75_be = sorted_be[min(idx75, len(sorted_be)-1)]
                            if p75_be > 0.50:
                                be_mult = max(0.50, 1.0 - (p75_be - 0.50) * 1.5)
                                trade_size *= be_mult
                                ml_stats.setdefault('p75_be', 0)
                                ml_stats['p75_be'] += 1

                    # Arch 134: Multi-TF extreme position boost (1.25x when >50% at boundary)
                    # When most TFs show price at extreme channel positions, boost confidence
                    if realistic and sig.signal_type == 'bounce':
                        extreme_count = 0
                        total = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                total += 1
                                if tf_s.position_pct < 0.05 or tf_s.position_pct > 0.95:
                                    extreme_count += 1
                        if total > 0 and extreme_count / total > 0.50:
                            trade_size *= 1.25
                            ml_stats.setdefault('extreme_pos_boost', 0)
                            ml_stats['extreme_pos_boost'] += 1


                    # Arch 159: TF-weighted KE boost (daily momentum matters more)
                    if realistic and sig.signal_type == 'bounce':
                        tf_w159 = {'5min': 0.5, '15min': 0.6, '30min': 0.7, '1h': 0.8,
                                   '2h': 0.85, '3h': 0.9, '4h': 0.95, 'daily': 1.2, 'weekly': 1.5, 'monthly': 2.0}
                        weighted_ke, total_w = 0, 0
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                w = tf_w159.get(tf_name, 1.0)
                                weighted_ke += tf_state.kinetic_energy * w
                                total_w += w
                        if total_w > 0:
                            wke = weighted_ke / total_w
                            if wke > 0.25:
                                ke_boost = min(1.30, 1.0 + (wke - 0.25) * 0.60)
                                trade_size *= ke_boost
                                ml_stats.setdefault('weighted_ke_boost', 0)
                                ml_stats['weighted_ke_boost'] += 1


                    # Arch 160: TF count penalty (fewer valid TFs = lower confidence)
                    if realistic and sig.signal_type == 'bounce':
                        valid_tf_count = sum(1 for tf_state in analysis.tf_states.values() if tf_state and tf_state.valid)
                        if valid_tf_count <= 2:
                            trade_size *= 0.70
                            ml_stats.setdefault('few_tf_reduce', 0)
                            ml_stats['few_tf_reduce'] += 1

                    # Arch 135: Win/loss parity control (0.60x when ratio > 10x)
                    # When avg win >> avg loss, leverage is asymmetric → reduce to rebalance
                    if realistic and len(trades) >= 10:
                        wins = [t.pnl for t in trades[-20:] if t.pnl > 0]
                        losses = [abs(t.pnl) for t in trades[-20:] if t.pnl <= 0]
                        if wins and losses:
                            ratio = (sum(wins)/len(wins)) / (sum(losses)/len(losses))
                            if ratio > 10:
                                trade_size *= 0.60
                                ml_stats.setdefault('parity_reduce', 0)
                                ml_stats['parity_reduce'] += 1


                    # Arch 161: 90th percentile BE penalty (worst-decile BE)
                    if realistic and sig.signal_type == 'bounce':
                        be_161 = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                be_161.append(tf_state.binding_energy)
                        if be_161:
                            sorted_be = sorted(be_161)
                            idx90 = int(len(sorted_be) * 0.90)
                            p90_be = sorted_be[min(idx90, len(sorted_be)-1)]
                            if p90_be > 0.60:
                                be_mult = max(0.50, 1.0 - (p90_be - 0.60) * 2.0)
                                trade_size *= be_mult
                                ml_stats.setdefault('p90_be', 0)
                                ml_stats['p90_be'] += 1

                    # Arch 136d: Break frequency filter (0.50x when >3 breaks in last 20 trades)
                    # Too many breaks = choppy regime, reduce break sizing
                    if realistic and sig.signal_type == 'break' and len(trades) >= 20:
                        recent_breaks = sum(1 for t in trades[-20:] if hasattr(t, 'signal_type') and t.signal_type == 'break')
                        if recent_breaks >= 3:
                            trade_size *= 0.50
                            ml_stats.setdefault('break_freq_reduce', 0)
                            ml_stats['break_freq_reduce'] += 1

                    # Arch 136f: Anti-cluster timing (0.60x within 2 bars of last trade)
                    # Rapid re-entry = poor selectivity
                    if realistic and trades and hasattr(trades[-1], 'exit_bar'):
                        if bar - trades[-1].exit_bar <= 2:
                            trade_size *= 0.60
                            ml_stats.setdefault('anti_cluster', 0)
                            ml_stats['anti_cluster'] += 1


                    # Arch 162: Median PE continuous boost (robust PE signal)
                    if realistic and sig.signal_type == 'bounce':
                        pe_162 = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                pe_162.append(tf_state.potential_energy)
                        if pe_162:
                            sorted_pe = sorted(pe_162)
                            n = len(sorted_pe)
                            median_pe = sorted_pe[n//2] if n % 2 == 1 else (sorted_pe[n//2-1] + sorted_pe[n//2]) / 2
                            if median_pe > 0.40:
                                pe_boost = min(1.25, 0.90 + 0.50 * median_pe)
                                trade_size *= pe_boost
                                ml_stats.setdefault('median_pe_boost', 0)
                                ml_stats['median_pe_boost'] += 1


                    # Arch 163: Freedom-momentum product boost ((1-BE)*KE)
                    if realistic and sig.signal_type == 'bounce':
                        fm_vals = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                fm_vals.append((1.0 - tf_state.binding_energy) * tf_state.kinetic_energy)
                        if fm_vals:
                            avg_fm = sum(fm_vals) / len(fm_vals)
                            if avg_fm > 0.20:
                                fm_boost = min(1.25, 1.0 + (avg_fm - 0.20) * 0.80)
                                trade_size *= fm_boost
                                ml_stats.setdefault('fm_boost', 0)
                                ml_stats['fm_boost'] += 1

                    # Arch 137e: High KE break boost (1.20x when KE > 0.70)
                    # Strong momentum = real breakout, not false break
                    if realistic and sig.signal_type == 'break':
                        ps137 = analysis.tf_states.get(sig.primary_tf)
                        if ps137 and ps137.kinetic_energy > 0.70:
                            trade_size *= 1.20
                            ml_stats.setdefault('high_ke_break_boost', 0)
                            ml_stats['high_ke_break_boost'] += 1

                    # Arch 137f: Continuous equity inverse scaling
                    # As equity grows, scale trade_size proportionally to initial/equity
                    # Keeps absolute risk per trade approximately constant
                    if realistic:
                        eq_scale = min(1.0, initial_capital / equity)
                        trade_size *= eq_scale
                        ml_stats.setdefault('eq_scale', 0)
                        ml_stats['eq_scale'] += 1


                    # Arch 164: 25th percentile BE penalty (even lower quartile is trapped)
                    if realistic and sig.signal_type == 'bounce':
                        be_164 = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                be_164.append(tf_state.binding_energy)
                        if be_164:
                            sorted_be = sorted(be_164)
                            idx25 = max(0, int(len(sorted_be) * 0.25))
                            p25_be = sorted_be[idx25]
                            if p25_be > 0.35:
                                be_mult = max(0.50, 1.0 - (p25_be - 0.35) * 1.5)
                                trade_size *= be_mult
                                ml_stats.setdefault('p25_be', 0)
                                ml_stats['p25_be'] += 1


                    # Arch 165: Median position edge boost (most TFs near boundary)
                    if realistic and sig.signal_type == 'bounce':
                        pos_165 = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                pos_165.append(min(tf_state.position_pct, 1.0 - tf_state.position_pct))
                        if pos_165:
                            sorted_pos = sorted(pos_165)
                            n = len(sorted_pos)
                            median_edge = sorted_pos[n//2] if n % 2 == 1 else (sorted_pos[n//2-1] + sorted_pos[n//2]) / 2
                            if median_edge < 0.10:
                                trade_size *= 1.20
                                ml_stats.setdefault('median_edge', 0)
                                ml_stats['median_edge'] += 1

                    # Arch 138: Quadratic equity inverse scaling
                    # Even more aggressive than linear: trade_size *= (initial/equity)^2
                    # At 4x equity → 1/16 size, at 10x → 1/100 size
                    if realistic:
                        eq_scale = min(1.0, (initial_capital / equity) ** 2)
                        trade_size *= eq_scale
                        ml_stats.setdefault('quad_eq_scale', 0)
                        ml_stats['quad_eq_scale'] += 1


                    # Arch 166: 25th percentile PE boost (most TFs have good PE)
                    if realistic and sig.signal_type == 'bounce':
                        pe_166 = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                pe_166.append(tf_state.potential_energy)
                        if pe_166:
                            sorted_pe = sorted(pe_166)
                            idx25 = max(0, int(len(sorted_pe) * 0.25))
                            p25_pe = sorted_pe[idx25]
                            if p25_pe > 0.35:
                                pe_boost = min(1.20, 1.0 + (p25_pe - 0.35) * 0.50)
                                trade_size *= pe_boost
                                ml_stats.setdefault('p25_pe', 0)
                                ml_stats['p25_pe'] += 1


                    # Arch 167: 25th percentile KE boost (most TFs moving)
                    if realistic and sig.signal_type == 'bounce':
                        ke_167f = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                ke_167f.append(tf_state.kinetic_energy)
                        if ke_167f:
                            sorted_ke = sorted(ke_167f)
                            idx25 = max(0, int(len(sorted_ke) * 0.25))
                            p25_ke = sorted_ke[idx25]
                            if p25_ke > 0.20:
                                ke_boost = min(1.20, 1.0 + (p25_ke - 0.20) * 0.50)
                                trade_size *= ke_boost
                                ml_stats.setdefault('p25_ke', 0)
                                ml_stats['p25_ke'] += 1

                    # Arch 168: Min KE penalty (all TFs stagnant = dead market)
                    if realistic and sig.signal_type == 'bounce':
                        ke_168 = []
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                ke_168.append(tf_state.kinetic_energy)
                        if ke_168:
                            min_ke = min(ke_168)
                            if min_ke < 0.05:
                                trade_size *= 0.70
                                ml_stats.setdefault('min_ke_penalty', 0)
                                ml_stats['min_ke_penalty'] += 1


                    # Arch 169: TF-weighted health score ((1-BE)*PE weighted by TF importance)
                    if realistic and sig.signal_type == 'bounce':
                        tf_w169 = {'5min': 0.5, '15min': 0.6, '30min': 0.7, '1h': 0.8,
                                   '2h': 0.85, '3h': 0.9, '4h': 0.95, 'daily': 1.2, 'weekly': 1.5, 'monthly': 2.0}
                        weighted_h, total_w = 0, 0
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                w = tf_w169.get(tf_name, 1.0)
                                h = (1.0 - tf_state.binding_energy) * max(tf_state.potential_energy, 0.01)
                                weighted_h += h * w
                                total_w += w
                        if total_w > 0:
                            wh = weighted_h / total_w
                            if wh > 0.25:
                                h_boost = min(1.20, 1.0 + (wh - 0.25) * 0.50)
                                trade_size *= h_boost
                                ml_stats.setdefault('weighted_health', 0)
                                ml_stats['weighted_health'] += 1


                    # Arch 170: Multi-TF direction consensus boost (1.20x when all TFs agree)
                    # Fixed: channel_direction is string (bull/bear/sideways), not int
                    if realistic and sig.signal_type == 'bounce':
                        dirs_170 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                dirs_170.append(1 if tf_s.channel_direction == 'bull' else (-1 if tf_s.channel_direction == 'bear' else 0))
                        if dirs_170 and (all(d >= 0 for d in dirs_170) or all(d <= 0 for d in dirs_170)):
                            trade_size *= 1.20
                            ml_stats.setdefault('dir_consensus', 0)
                            ml_stats['dir_consensus'] += 1

                    # Arch 171: Perfect WR streak boost (1.30x when last 20 all wins)
                    # 100% WR in recent window = system in optimal regime
                    if realistic and len(trades) >= 20:
                        if all(t.pnl > 0 for t in trades[-20:]):
                            trade_size *= 1.30
                            ml_stats.setdefault('perfect_wr', 0)
                            ml_stats['perfect_wr'] += 1

                    # Arch 172: PnL trend penalty (0.60x when recent avg declining vs prior)
                    # Declining PnL average = market regime shifting, reduce exposure
                    if realistic and len(trades) >= 20:
                        recent10_172 = [t.pnl_pct for t in trades[-10:]]
                        prev10_172 = [t.pnl_pct for t in trades[-20:-10]]
                        if sum(recent10_172)/10 < sum(prev10_172)/10 * 0.50:
                            trade_size *= 0.60
                            ml_stats.setdefault('pnl_trend', 0)
                            ml_stats['pnl_trend'] += 1


                    # Arch 174: Sideways direction penalty (most TFs sideways = range-bound)
                    if realistic and sig.signal_type == 'bounce':
                        sideways_count = 0
                        valid_count = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                valid_count += 1
                                if tf_s.channel_direction == 'sideways':
                                    sideways_count += 1
                        if valid_count > 0 and sideways_count / valid_count > 0.50:
                            trade_size *= 0.75
                            ml_stats.setdefault('sideways_reduce', 0)
                            ml_stats['sideways_reduce'] += 1


                    # Arch 173: WR regime shift penalty (0.50x when rolling 10-trade WR < 80%)
                    # Declining win rate = market regime shifting, reduce exposure
                    if realistic and len(trades) >= 10:
                        wr_10_173 = sum(1 for t in trades[-10:] if t.pnl > 0) / 10
                        if wr_10_173 < 0.80:
                            trade_size *= 0.50
                            ml_stats.setdefault('wr_regime', 0)
                            ml_stats['wr_regime'] += 1




                    # Arch 174: PnL acceleration penalty (0.50x when PnL deceleration)
                    # If sum of last 10 PnL < sum of prior 10, momentum is fading
                    if realistic and len(trades) >= 20:
                        r1_174 = sum(t.pnl for t in trades[-10:])
                        r2_174 = sum(t.pnl for t in trades[-20:-10])
                        if r1_174 < r2_174 and r2_174 > 0:
                            trade_size *= 0.50
                            ml_stats.setdefault('pnl_accel', 0)
                            ml_stats['pnl_accel'] += 1


                    # Arch 176: Low confidence reduce (low signal confidence = risky)
                    if realistic and hasattr(sig, 'confidence') and sig.confidence < 0.50:
                        trade_size *= 0.60
                        ml_stats.setdefault('low_conf', 0)
                        ml_stats['low_conf'] += 1

                    # Arch 177: Anti-revenge (reduce after loss if same signal type)
                    if realistic and trades and trades[-1].pnl <= 0:
                        if hasattr(trades[-1], 'signal_type') and trades[-1].signal_type == sig.signal_type:
                            trade_size *= 0.50
                            ml_stats.setdefault('anti_revenge', 0)
                            ml_stats['anti_revenge'] += 1


                    # Arch 178: Break type hard reduce (breaks riskier than bounces)
                    if realistic and sig.signal_type == 'break':
                        trade_size *= 0.40
                        ml_stats.setdefault('break_hard_reduce', 0)
                        ml_stats['break_hard_reduce'] += 1


                    # Arch 175: Growing loss size penalty (0.50x when avg loss increasing)
                    # Losses getting bigger = market regime changing, reduce exposure
                    if realistic and len(trades) >= 20:
                        losses_recent_175 = [abs(t.pnl) for t in trades[-10:] if t.pnl < 0]
                        losses_prev_175 = [abs(t.pnl) for t in trades[-20:-10] if t.pnl < 0]
                        if losses_recent_175 and losses_prev_175:
                            avg_recent_loss_175 = sum(losses_recent_175) / len(losses_recent_175)
                            avg_prev_loss_175 = sum(losses_prev_175) / len(losses_prev_175)
                            if avg_recent_loss_175 > avg_prev_loss_175 * 1.50:
                                trade_size *= 0.50
                                ml_stats.setdefault('loss_growing', 0)
                                ml_stats['loss_growing'] += 1


                    # Arch 179: Channel health continuous sizing (healthy = bigger)
                    if realistic and hasattr(sig, 'channel_health'):
                        ch = max(0.0, min(1.0, sig.channel_health))
                        ch_mult = 0.3 + 1.4 * (ch ** 2.0)
                        trade_size *= ch_mult
                        ml_stats.setdefault('ch_health_sizing', 0)
                        ml_stats['ch_health_sizing'] += 1


                    # Arch 176: Channel maturity boost (1.15x when avg bounce_count > 5)
                    # Well-tested channels with many bounces are more reliable
                    if realistic and sig.signal_type == 'bounce':
                        bc_176 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                bc_176.append(tf_s.bounce_count)
                        if bc_176 and sum(bc_176)/len(bc_176) > 5:
                            trade_size *= 1.15
                            ml_stats.setdefault('channel_mature', 0)
                            ml_stats['channel_mature'] += 1


                    # Arch 180: Confluence score continuous sizing
                    if realistic and hasattr(sig, 'confluence_score'):
                        cs = max(0.0, min(1.0, sig.confluence_score))
                        cs_mult = 0.5 + cs * 1.0
                        trade_size *= cs_mult
                        ml_stats.setdefault('conf_score_sizing', 0)
                        ml_stats['conf_score_sizing'] += 1


                    # Arch 177: Trade efficiency boost (1.20x when recent wins are quick)
                    # High MFE/hold_bars ratio = strong directional moves, efficient captures
                    if realistic and len(trades) >= 10:
                        efficiencies_177 = []
                        for t in trades[-10:]:
                            if t.hold_bars > 0 and t.pnl > 0:
                                efficiencies_177.append(t.mfe_pct / t.hold_bars)
                        if len(efficiencies_177) >= 5 and sum(efficiencies_177)/len(efficiencies_177) > 0.001:
                            trade_size *= 1.20
                            ml_stats.setdefault('trade_eff', 0)
                            ml_stats['trade_eff'] += 1


                    # Arch 181: Bad exit rate penalty (stops/timeouts = weak signals)
                    if realistic and len(trades) >= 10:
                        bad_exits_181 = sum(1 for t in trades[-10:] if t.exit_reason in ('stop', 'ou_timeout'))
                        if bad_exits_181 / 10 > 0.20:
                            trade_size *= 0.50
                            ml_stats.setdefault('bad_exit_rate', 0)
                            ml_stats['bad_exit_rate'] += 1


                    # Arch 182: Double-weak filter (both min PE and min KE below thresholds)
                    if realistic and sig.signal_type == 'bounce':
                        min_pe_182 = 1.0
                        min_ke_182 = 1.0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                min_pe_182 = min(min_pe_182, tf_s.potential_energy)
                                min_ke_182 = min(min_ke_182, tf_s.kinetic_energy)
                        if min_pe_182 < 0.15 and min_ke_182 < 0.15:
                            trade_size *= 0.35
                            ml_stats.setdefault('double_weak', 0)
                            ml_stats["double_weak"] += 1


                    # Arch 183: PnL acceleration boost (1.15x when recent trades accelerating)
                    if realistic and len(trades) >= 20:
                        avg5_183 = sum(t.pnl for t in trades[-5:]) / 5
                        avg20_183 = sum(t.pnl for t in trades[-20:]) / 20
                        if avg5_183 > avg20_183 * 1.5 and avg5_183 > 0:
                            trade_size *= 1.15
                            ml_stats.setdefault('pnl_accel_boost', 0)
                            ml_stats['pnl_accel_boost'] += 1


                    # Arch 182c: Multi-TF KE consensus boost (1.20x when >60% TFs have KE > 0.40)
                    # Most timeframes showing momentum = market actively moving
                    if realistic and sig.signal_type == 'bounce':
                        high_ke_count_182 = 0
                        total_182c = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                total_182c += 1
                                if tf_s.kinetic_energy > 0.40:
                                    high_ke_count_182 += 1
                        if total_182c > 0 and high_ke_count_182 / total_182c > 0.60:
                            trade_size *= 1.20
                            ml_stats.setdefault('ke_consensus', 0)
                            ml_stats['ke_consensus'] += 1

                    # Arch 182f: Weighted energy product (PE*KE*(1-BE) by TF weight)
                    # Comprehensive signal quality metric: free + energetic + potential
                    if realistic and sig.signal_type == 'bounce':
                        tf_w182 = {'5min': 0.3, '15min': 0.4, '30min': 0.5, '1h': 0.7,
                                   '2h': 0.8, '3h': 0.9, '4h': 1.0, 'daily': 1.5, 'weekly': 2.0, 'monthly': 2.5}
                        w_energy_182, total_w182 = 0, 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                w = tf_w182.get(tf_n, 1.0)
                                total_w182 += w
                                product_182 = tf_s.potential_energy * tf_s.kinetic_energy * (1.0 - tf_s.binding_energy)
                                w_energy_182 += product_182 * w
                        if total_w182 > 0:
                            avg_energy_182 = w_energy_182 / total_w182
                            if avg_energy_182 > 0.08:
                                energy_boost_182 = min(1.25, 1.0 + (avg_energy_182 - 0.08) * 2.0)
                                trade_size *= energy_boost_182
                                ml_stats.setdefault('w_energy_prod', 0)
                                ml_stats['w_energy_prod'] += 1




                    # Arch 185: Large recent loss penalty (0.40x when last loss > 2x avg loss)
                    if realistic and len(trades) >= 5:
                        losses_185 = [abs(t.pnl) for t in trades if t.pnl < 0]
                        if losses_185:
                            avg_loss_185 = sum(losses_185) / len(losses_185)
                            recent_losses_185 = [abs(t.pnl) for t in trades[-3:] if t.pnl < 0]
                            if recent_losses_185 and max(recent_losses_185) > avg_loss_185 * 2.0:
                                trade_size *= 0.40
                                ml_stats.setdefault('big_loss_reduce', 0)
                                ml_stats['big_loss_reduce'] += 1


                    # Arch 184: Exponential win streak compounding (1.02^consecutive_wins)
                    # Long win streaks compound small boosts: 5 wins = 1.10x, 10 = 1.22x
                    if realistic and trades:
                        consec_wins_184 = 0
                        for t in reversed(trades):
                            if t.pnl > 0:
                                consec_wins_184 += 1
                            else:
                                break
                        if consec_wins_184 >= 5:
                            win_boost_184 = min(1.50, 1.02 ** consec_wins_184)
                            trade_size *= win_boost_184
                            ml_stats.setdefault('exp_win_streak', 0)
                            ml_stats['exp_win_streak'] += 1


                    # Arch 186: Trailing PnL momentum (1.15x when recent 5 outperform prior 5)
                    if realistic and len(trades) >= 10:
                        sum5_186 = sum(t.pnl for t in trades[-5:])
                        prev5_186 = sum(t.pnl for t in trades[-10:-5])
                        if sum5_186 > prev5_186 * 1.5 and sum5_186 > 0:
                            trade_size *= 1.15
                            ml_stats.setdefault('trail_momentum', 0)
                            ml_stats['trail_momentum'] += 1


                    # Arch 186: Abs PnL amplitude declining filter (0.40x when moves shrinking)
                    # When average absolute PnL drops to <50% of prior period, market is stalling
                    if realistic and len(trades) >= 20:
                        recent_abs_186 = sum(abs(t.pnl) for t in trades[-10:]) / 10
                        prev_abs_186 = sum(abs(t.pnl) for t in trades[-20:-10]) / 10
                        if prev_abs_186 > 0 and recent_abs_186 < prev_abs_186 * 0.50:
                            trade_size *= 0.40
                            ml_stats.setdefault('amp_decline', 0)
                            ml_stats['amp_decline'] += 1


                    # Arch 189d: Trail-loss cluster detection (0.50x after 2+ trail losses in 5)
                    if realistic and len(trades) >= 5:
                        trail_losses_189 = sum(1 for t in trades[-5:]
                                              if t.exit_reason == "trail" and t.pnl < 0)
                        if trail_losses_189 >= 2:
                            trade_size *= 0.50
                            ml_stats.setdefault("trail_loss_cluster", 0)
                            ml_stats["trail_loss_cluster"] += 1


                    # Arch 189f: Bounce low PE penalty (0.40x when median PE < 0.20)
                    if realistic and sig.signal_type == "bounce":
                        pes_189f = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                pes_189f.append(tf_s.potential_energy)
                        if pes_189f:
                            sorted_pe = sorted(pes_189f)
                            n = len(sorted_pe)
                            median_pe_189 = sorted_pe[n//2] if n % 2 == 1 else (sorted_pe[n//2-1] + sorted_pe[n//2]) / 2
                            if median_pe_189 < 0.20:
                                trade_size *= 0.40
                                ml_stats.setdefault("bounce_low_pe", 0)
                                ml_stats["bounce_low_pe"] += 1


                    # Arch 189c: Bounce high BE gate (0.30x when any TF BE > 0.70)
                    if realistic and sig.signal_type == "bounce":
                        max_be_189c = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                max_be_189c = max(max_be_189c, tf_s.binding_energy)
                        if max_be_189c > 0.70:
                            trade_size *= 0.30
                            ml_stats.setdefault("bounce_high_be", 0)
                            ml_stats["bounce_high_be"] += 1




                    # Arch 189b: Equity growth momentum (1.10x when equity > 1.5x initial)
                    if realistic:
                        if equity > initial_capital * 1.5:
                            trade_size *= 1.10
                            ml_stats.setdefault("eq_growth_mom", 0)
                            ml_stats["eq_growth_mom"] += 1


                    # Arch 190: PE/BE ratio boost (1.15x when avg PE/BE > 2.0)
                    if realistic and sig.signal_type == "bounce":
                        ratios_190 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid and tf_s.binding_energy > 0.01:
                                ratios_190.append(tf_s.potential_energy / tf_s.binding_energy)
                        if ratios_190:
                            avg_ratio_190 = sum(ratios_190) / len(ratios_190)
                            if avg_ratio_190 > 2.0:
                                trade_size *= 1.15
                                ml_stats.setdefault("pe_be_ratio", 0)
                                ml_stats["pe_be_ratio"] += 1


                    # Arch 190: Cyclical trade count scaling (1.20x/0.80x per 100-trade cycles)
                    # Alternate between aggressive and conservative phases to capture
                    # different market regimes. Reduces over-optimization risk.
                    if realistic:
                        cycle_190 = (len(trades) // 100) % 2
                        if cycle_190 == 0:
                            trade_size *= 1.20
                        else:
                            trade_size *= 0.80
                        ml_stats.setdefault('cycle_size', 0)
                        ml_stats['cycle_size'] += 1


                    # Arch 191: Gradual trade count decay (0.999^n, anti-overtrading)
                    # Subtly reduces position size as total trades grow, preventing
                    # late-session over-leverage. At 100 trades = 0.90x, 200 = 0.82x, 300 = 0.74x
                    if realistic:
                        decay_191 = max(0.30, 0.999 ** len(trades))
                        trade_size *= decay_191
                        ml_stats.setdefault('trade_decay', 0)
                        ml_stats['trade_decay'] += 1


                    # Arch 192a: Squeeze score boost (1.15x when avg squeeze > 0.50)
                    if realistic and sig.signal_type == "bounce":
                        sq_192 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                sq_192.append(tf_s.squeeze_score)
                        if sq_192:
                            avg_sq = sum(sq_192) / len(sq_192)
                            if avg_sq > 0.50:
                                trade_size *= 1.15
                                ml_stats.setdefault("squeeze_boost", 0)
                                ml_stats["squeeze_boost"] += 1


                    # Arch 192e: Steep slope penalty (0.50x when avg |slope| > 1.0%)
                    if realistic and sig.signal_type == "bounce":
                        slopes_192 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                slopes_192.append(abs(tf_s.slope_pct))
                        if slopes_192:
                            avg_slope_192 = sum(slopes_192) / len(slopes_192)
                            if avg_slope_192 > 1.0:
                                trade_size *= 0.50
                                ml_stats.setdefault("steep_slope", 0)
                                ml_stats["steep_slope"] += 1


                    # Arch 193: Momentum turning penalty (0.50x when >60% TFs turning)
                    if realistic and sig.signal_type == "bounce":
                        turning_193 = sum(1 for tf_n, tf_s in analysis.tf_states.items()
                                         if tf_s and tf_s.valid and tf_s.momentum_is_turning)
                        valid_193 = sum(1 for tf_n, tf_s in analysis.tf_states.items()
                                       if tf_s and tf_s.valid)
                        if valid_193 > 0 and turning_193 / valid_193 > 0.60:
                            trade_size *= 0.50
                            ml_stats.setdefault("mom_turning", 0)
                            ml_stats["mom_turning"] += 1


                    # Arch 192: Weighted break probability penalty (0.60x when avg > 0.35)
                    # When higher-TF-weighted break probability is high, channels are fragile
                    if realistic and sig.signal_type == 'bounce':
                        tf_w192 = {'5min': 0.3, '15min': 0.4, '30min': 0.5, '1h': 0.7,
                                   '2h': 0.8, '3h': 0.9, '4h': 1.0, 'daily': 1.5, 'weekly': 2.0, 'monthly': 2.5}
                        w_bp_192, total_w192 = 0, 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                w = tf_w192.get(tf_n, 1.0)
                                total_w192 += w
                                w_bp_192 += tf_s.break_prob * w
                        if total_w192 > 0 and w_bp_192 / total_w192 > 0.35:
                            trade_size *= 0.60
                            ml_stats.setdefault('w_break_prob', 0)
                            ml_stats['w_break_prob'] += 1


                    # Arch 194e: Momentum direction consensus (1.15x when all TFs agree)
                    if realistic and sig.signal_type == "bounce":
                        mom_dirs_194 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                mom_dirs_194.append(1 if tf_s.momentum_direction > 0 else -1 if tf_s.momentum_direction < 0 else 0)
                        nonzero = [d for d in mom_dirs_194 if d != 0]
                        if nonzero and len(set(nonzero)) <= 1:
                            trade_size *= 1.15
                            ml_stats.setdefault("mom_dir_consensus", 0)
                            ml_stats["mom_dir_consensus"] += 1


                    # Arch 194c: Composite quality sizing (r_sq * (1-BE) * PE, TF-weighted)
                    if realistic and sig.signal_type == "bounce":
                        tf_w194 = {"5min": 0.5, "15min": 0.6, "30min": 0.7, "1h": 0.8,
                                   "2h": 0.85, "3h": 0.9, "4h": 0.95, "daily": 1.2, "weekly": 1.5, "monthly": 2.0}
                        w_qual, total_w = 0, 0
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                w = tf_w194.get(tf_name, 1.0)
                                q = tf_state.r_squared * (1.0 - tf_state.binding_energy) * max(tf_state.potential_energy, 0.01)
                                w_qual += q * w
                                total_w += w
                        if total_w > 0:
                            avg_qual = w_qual / total_w
                            q_mult = min(1.30, max(0.50, 0.5 + avg_qual * 2.0))
                            trade_size *= q_mult
                            ml_stats.setdefault("composite_q", 0)
                            ml_stats["composite_q"] += 1


                    # Arch 193: Theta variance penalty (0.70x when TFs disagree on reversion speed)
                    # High variance in ou_theta across TFs = inconsistent reversion dynamics
                    if realistic and sig.signal_type == 'bounce':
                        thetas_193 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                thetas_193.append(tf_s.ou_theta)
                        if len(thetas_193) >= 3:
                            avg_th_193 = sum(thetas_193) / len(thetas_193)
                            var_th_193 = sum((t - avg_th_193)**2 for t in thetas_193) / len(thetas_193)
                            if var_th_193 > 0.02:
                                trade_size *= 0.70
                                ml_stats.setdefault('theta_var', 0)
                                ml_stats['theta_var'] += 1


                    # Arch 196: Position extremity boost (near edge = better bounce, 0.5+|pos|*1.5)
                    if realistic and sig.signal_type == "bounce":
                        pos_196 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                pos_196.append(abs(tf_s.position_pct))
                        if pos_196:
                            avg_pos_196 = sum(pos_196) / len(pos_196)
                            pos_mult = min(1.50, 0.5 + avg_pos_196 * 1.5)
                            trade_size *= pos_mult
                            ml_stats.setdefault("pos_extremity", 0)
                            ml_stats["pos_extremity"] += 1

                    # Arch 199b: OU half-life quality (1.15x when optimal reversion speed)
                    if realistic and sig.signal_type == 'bounce':
                        hl_199 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid and tf_s.ou_half_life > 0:
                                hl_199.append(tf_s.ou_half_life)
                        if hl_199:
                            avg_hl = sum(hl_199) / len(hl_199)
                            if 5.0 <= avg_hl <= 50.0:
                                trade_size *= 1.15
                                ml_stats.setdefault('good_half_life', 0)
                                ml_stats['good_half_life'] += 1

                    # Arch 199f: Multi-TF slope consensus (1.15x when all slopes agree)
                    if realistic and sig.signal_type == 'bounce':
                        slopes_199 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                slopes_199.append(1 if tf_s.slope_pct > 0 else -1 if tf_s.slope_pct < 0 else 0)
                        nonzero_199 = [s for s in slopes_199 if s != 0]
                        if len(nonzero_199) >= 3 and len(set(nonzero_199)) == 1:
                            trade_size *= 1.15
                            ml_stats.setdefault('slope_consensus', 0)
                            ml_stats['slope_consensus'] += 1

                    # Arch 198f: PnL volatility spike penalty (0.50x when std > 3x avg)
                    if realistic and len(trades) >= 10:
                        pnls_198 = [t.pnl for t in trades[-10:]]
                        avg_pnl_198 = sum(pnls_198) / len(pnls_198)
                        std_pnl_198 = (sum((p - avg_pnl_198)**2 for p in pnls_198) / len(pnls_198)) ** 0.5
                        if avg_pnl_198 > 0 and std_pnl_198 > avg_pnl_198 * 3.0:
                            trade_size *= 0.50
                            ml_stats.setdefault('pnl_vol_spike', 0)
                            ml_stats['pnl_vol_spike'] += 1

                    # Arch 200d: Width-slope product (1.15x when channels wide AND trending)
                    if realistic and sig.signal_type == 'bounce':
                        ws_200 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                ws_200.append(tf_s.width_pct * abs(tf_s.slope_pct))
                        if ws_200:
                            avg_ws = sum(ws_200) / len(ws_200)
                            if avg_ws > 0.0001:
                                trade_size *= 1.15
                                ml_stats.setdefault('wide_trend', 0)
                                ml_stats['wide_trend'] += 1

                    # Arch 200e: Break prob asymmetry alignment (1.15x BUY+up or SELL+down)
                    if realistic and sig.signal_type == 'bounce' and hasattr(sig, 'action'):
                        bp_up_200, bp_dn_200 = [], []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                bp_up_200.append(tf_s.break_prob_up)
                                bp_dn_200.append(tf_s.break_prob_down)
                        if bp_up_200 and bp_dn_200:
                            avg_up = sum(bp_up_200) / len(bp_up_200)
                            avg_dn = sum(bp_dn_200) / len(bp_dn_200)
                            if (sig.action == "BUY" and avg_up > avg_dn * 2.0) or                                (sig.action == "SELL" and avg_dn > avg_up * 2.0):
                                trade_size *= 1.15
                                ml_stats.setdefault('bp_asym_align', 0)
                                ml_stats['bp_asym_align'] += 1


                    # Arch 199: Oscillation timing boost (1.15x when near predicted bounce)
                    if realistic and sig.signal_type == "bounce":
                        timing_199 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid and tf_s.oscillation_period > 0:
                                ratio = tf_s.bars_to_next_bounce / tf_s.oscillation_period
                                timing_199.append(ratio)
                        if timing_199:
                            avg_ratio_199 = sum(timing_199) / len(timing_199)
                            if avg_ratio_199 < 0.25:
                                trade_size *= 1.15
                                ml_stats.setdefault("osc_timing", 0)
                                ml_stats["osc_timing"] += 1

                    # Arch 200b: Harmonic TF resonance (1.20x when bounce counts synchronized)
                    if realistic and sig.signal_type == 'bounce':
                        bc_200 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                bc_200.append(tf_s.bounce_count)
                        if len(bc_200) >= 3:
                            avg_bc = sum(bc_200) / len(bc_200)
                            if avg_bc > 0:
                                norm_bc = [b / avg_bc for b in bc_200]
                                variance = sum((n - 1.0)**2 for n in norm_bc) / len(norm_bc)
                                if variance < 0.15:
                                    trade_size *= 1.20
                                    ml_stats.setdefault('harmonic_bounce', 0)
                                    ml_stats['harmonic_bounce'] += 1

                    # Arch 201e: Theta-PE product boost (1.15x when fast reversion + high potential)
                    if realistic and sig.signal_type == 'bounce':
                        tf_w201 = {'5min': 0.5, '15min': 0.6, '30min': 0.7, '1h': 0.8,
                                   '2h': 0.85, '3h': 0.9, '4h': 0.95, 'daily': 1.2, 'weekly': 1.5, 'monthly': 2.0}
                        w_tp, total_w = 0, 0
                        for tf_name, tf_state in analysis.tf_states.items():
                            if tf_state and tf_state.valid:
                                w = tf_w201.get(tf_name, 1.0)
                                w_tp += tf_state.ou_theta * tf_state.potential_energy * w
                                total_w += w
                        if total_w > 0:
                            avg_tp = w_tp / total_w
                            if avg_tp > 0.10:
                                trade_size *= 1.15
                                ml_stats.setdefault('theta_pe_prod', 0)
                                ml_stats['theta_pe_prod'] += 1

                    # Arch 202f: MFE/MAE ratio boost (1.15x when recent winners running well)
                    if realistic and len(trades) >= 10:
                        mfe_mae_202 = []
                        for t in trades[-10:]:
                            if t.pnl > 0 and hasattr(t, 'mae_pct') and t.mae_pct > 0:
                                mfe_mae_202.append(t.mfe_pct / t.mae_pct)
                        if len(mfe_mae_202) >= 5 and sum(mfe_mae_202)/len(mfe_mae_202) > 2.0:
                            trade_size *= 1.15
                            ml_stats.setdefault('mfe_mae_ratio', 0)
                            ml_stats['mfe_mae_ratio'] += 1

                    # Arch 203c: Geometric mean channel health (1.15x when all TFs healthy)
                    if realistic and sig.signal_type == 'bounce':
                        healths_203 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                healths_203.append(max(0.01, tf_s.channel_health))
                        if healths_203:
                            import math
                            geo_h = math.exp(sum(math.log(h) for h in healths_203) / len(healths_203))
                            if geo_h > 0.50:
                                trade_size *= 1.15
                                ml_stats.setdefault('geo_health', 0)
                                ml_stats['geo_health'] += 1

                    # Arch 203e: Volume-weighted position extremity (1.15x when vol confirms edge)
                    if realistic and sig.signal_type == 'bounce':
                        vp_203 = 0
                        total_w203 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                w = max(0.1, tf_s.volume_score)
                                vp_203 += abs(tf_s.position_pct) * w
                                total_w203 += w
                        if total_w203 > 0:
                            vol_pos = vp_203 / total_w203
                            if vol_pos > 0.80:
                                trade_size *= 1.15
                                ml_stats.setdefault('vol_pos_edge', 0)
                                ml_stats['vol_pos_edge'] += 1

                    # Arch 204b: KE monotonic increase across TFs (1.20x when KE building up)
                    if realistic and sig.signal_type == 'bounce':
                        tf_order_204 = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly']
                        ke_ordered = []
                        for tf_n in tf_order_204:
                            tf_s = analysis.tf_states.get(tf_n)
                            if tf_s and tf_s.valid:
                                ke_ordered.append(tf_s.kinetic_energy)
                        if len(ke_ordered) >= 3:
                            increases = sum(1 for i in range(1, len(ke_ordered)) if ke_ordered[i] > ke_ordered[i-1])
                            if increases >= len(ke_ordered) - 1:
                                trade_size *= 1.20
                                ml_stats.setdefault('ke_monotonic', 0)
                                ml_stats['ke_monotonic'] += 1

                    # Arch 205c: Primary TF is best (1.20x when signal from most reliable channel)
                    if realistic and sig.signal_type == 'bounce':
                        best_q_205 = 0
                        primary_q_205 = 0
                        ps_205 = analysis.tf_states.get(sig.primary_tf)
                        if ps_205 and ps_205.valid:
                            primary_q_205 = ps_205.r_squared * ps_205.channel_health
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                q = tf_s.r_squared * tf_s.channel_health
                                best_q_205 = max(best_q_205, q)
                        if primary_q_205 > 0 and primary_q_205 >= best_q_205 * 0.95:
                            trade_size *= 1.20
                            ml_stats.setdefault('primary_best', 0)
                            ml_stats['primary_best'] += 1

                    # Arch 205d: Total energy uniform (1.15x when TFs have similar energy)
                    if realistic and sig.signal_type == 'bounce':
                        te_205 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                te_205.append(tf_s.total_energy)
                        if len(te_205) >= 3:
                            te_range = max(te_205) - min(te_205)
                            avg_te = sum(te_205) / len(te_205)
                            if avg_te > 0 and te_range / avg_te < 0.30:
                                trade_size *= 1.15
                                ml_stats.setdefault('te_uniform', 0)
                                ml_stats['te_uniform'] += 1

                    # Arch 206f: Width*theta product (1.15x when wide channel + moderate reversion)
                    if realistic and sig.signal_type == 'bounce':
                        wt_206 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                wt_206.append(tf_s.width_pct * tf_s.ou_theta)
                        if wt_206:
                            avg_wt = sum(wt_206) / len(wt_206)
                            if avg_wt > 0.001:
                                trade_size *= 1.15
                                ml_stats.setdefault('width_theta', 0)
                                ml_stats['width_theta'] += 1

                    # Arch 206d: PE*KE*r_sq triple product continuous sizing
                    if realistic and sig.signal_type == 'bounce':
                        triple_206 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                triple_206.append(tf_s.potential_energy * tf_s.kinetic_energy * tf_s.r_squared)
                        if triple_206:
                            avg_triple = sum(triple_206) / len(triple_206)
                            t_mult = min(1.30, max(0.50, 0.5 + avg_triple * 5.0))
                            trade_size *= t_mult
                            ml_stats.setdefault('triple_qual', 0)
                            ml_stats['triple_qual'] += 1

                    # Arch 208b: Loss cluster detector (0.20x after 2+ losses in 15 trades)
                    if realistic and len(trades) >= 15:
                        recent_losses_208 = sum(1 for t in trades[-15:] if t.pnl < 0)
                        if recent_losses_208 >= 2:
                            trade_size *= 0.20
                            ml_stats.setdefault('loss_cluster', 0)
                            ml_stats['loss_cluster'] += 1

                    # Arch 210d: Total energy low CV (1.15x when energy stable across TFs)
                    if realistic and sig.signal_type == 'bounce':
                        te_210 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                te_210.append(tf_s.total_energy)
                        if len(te_210) >= 3:
                            avg_te = sum(te_210) / len(te_210)
                            if avg_te > 0:
                                std_te = (sum((t - avg_te)**2 for t in te_210) / len(te_210)) ** 0.5
                                cv = std_te / avg_te
                                if cv < 0.30:
                                    trade_size *= 1.15
                                    ml_stats.setdefault('te_low_cv', 0)
                                    ml_stats['te_low_cv'] += 1

                    # Arch 210e: Strong reversion signal (1.15x when any TF reversion > 0.80)
                    if realistic and sig.signal_type == 'bounce':
                        max_rev_210 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                max_rev_210 = max(max_rev_210, tf_s.ou_reversion_score)
                        if max_rev_210 > 0.80:
                            trade_size *= 1.15
                            ml_stats.setdefault('strong_revert', 0)
                            ml_stats['strong_revert'] += 1

                    # Arch 211c: Channel direction unanimity (1.15x when ALL TFs same direction)
                    if realistic and sig.signal_type == 'bounce':
                        dirs_211 = set()
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                dirs_211.add(tf_s.channel_direction)
                        if len(dirs_211) == 1 and 'sideways' not in dirs_211:
                            trade_size *= 1.15
                            ml_stats.setdefault('dir_unanim', 0)
                            ml_stats['dir_unanim'] += 1

                    # Arch 211b: Safe support bounce (1.15x BUY + low break_prob_down)
                    if realistic and sig.signal_type == 'bounce' and hasattr(sig, 'action') and sig.action == 'BUY':
                        bp_dn_211 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                bp_dn_211.append(tf_s.break_prob_down)
                        if bp_dn_211 and sum(bp_dn_211)/len(bp_dn_211) < 0.20:
                            trade_size *= 1.15
                            ml_stats.setdefault('safe_support', 0)
                            ml_stats['safe_support'] += 1


                    # Arch 215: Wide + trending + healthy channel sizing
                    if realistic and sig.signal_type == "bounce":
                        wsh_215 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                wsh_215.append(tf_s.width_pct * 100 * abs(tf_s.slope_pct) * 100 * tf_s.channel_health)
                        if wsh_215:
                            avg_wsh = sum(wsh_215) / len(wsh_215)
                            wsh_mult = min(1.25, max(0.50, 0.5 + avg_wsh * 0.5))
                            trade_size *= wsh_mult
                            ml_stats.setdefault("wsh_sizing", 0)
                            ml_stats["wsh_sizing"] += 1

                    # Arch 215e: Fast reversion relative to maturity (1.15x when HL/BC < 5)
                    if realistic and sig.signal_type == 'bounce':
                        hl_215 = []
                        bc_215 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                if tf_s.ou_half_life > 0:
                                    hl_215.append(tf_s.ou_half_life)
                                bc_215.append(max(1, tf_s.bounce_count))
                        if hl_215 and bc_215:
                            avg_hl = sum(hl_215) / len(hl_215)
                            avg_bc = sum(bc_215) / len(bc_215)
                            if avg_bc > 0 and avg_hl / avg_bc < 5.0:
                                trade_size *= 1.15
                                ml_stats.setdefault('fast_vs_mature', 0)
                                ml_stats['fast_vs_mature'] += 1

                    # Arch 216a: Max PE loaded (1.15x when any TF has PE > 0.80 = spring loaded)
                    if realistic and sig.signal_type == 'bounce':
                        max_pe_216 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                max_pe_216 = max(max_pe_216, tf_s.potential_energy)
                        if max_pe_216 > 0.80:
                            trade_size *= 1.15
                            ml_stats.setdefault('max_pe_loaded', 0)
                            ml_stats['max_pe_loaded'] += 1

                    # Arch 216d: TE * health product (1.15x when avg TE*health > 0.35)
                    if realistic and sig.signal_type == 'bounce':
                        th_216 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                th_216.append(tf_s.total_energy * tf_s.channel_health)
                        if th_216 and sum(th_216)/len(th_216) > 0.35:
                            trade_size *= 1.15
                            ml_stats.setdefault('te_health', 0)
                            ml_stats['te_health'] += 1


                    # Arch 216: Breaks in low-reversion regime penalty
                    if realistic and sig.signal_type == "break":
                        rev_216 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                rev_216.append(tf_s.ou_reversion_score)
                        if rev_216 and sum(rev_216)/len(rev_216) < 0.25:
                            trade_size *= 0.30
                            ml_stats.setdefault("break_low_rev", 0)
                            ml_stats["break_low_rev"] += 1

                    # Arch 217d: Momentum * width product (1.15x when strong directional move in wide channels)
                    if realistic and sig.signal_type == 'bounce':
                        mw_217 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                mw_217.append(abs(tf_s.momentum_direction) * tf_s.width_pct * 1000)
                        if mw_217:
                            avg_mw = sum(mw_217) / len(mw_217)
                            if avg_mw > 1.0:
                                trade_size *= 1.15
                                ml_stats.setdefault('mom_width', 0)
                                ml_stats['mom_width'] += 1


                    # Arch 218: Recent loss proximity decay (cautious near losses)
                    if realistic and len(trades) >= 1:
                        last_loss_bar_218 = None
                        for t in reversed(trades):
                            if t.pnl < 0:
                                last_loss_bar_218 = t.exit_bar if hasattr(t, "exit_bar") else None
                                break
                        if last_loss_bar_218 is not None:
                            bars_since = bar - last_loss_bar_218
                            if bars_since < 20:
                                decay_mult = min(1.0, 0.30 + bars_since * 0.035)
                                trade_size *= decay_mult
                                ml_stats.setdefault("loss_proximity", 0)
                                ml_stats["loss_proximity"] += 1

                    # Arch 218a: Max volume confirmation (1.15x when any TF has volume > 0.80)
                    if realistic and sig.signal_type == 'bounce':
                        max_vol_218 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                max_vol_218 = max(max_vol_218, tf_s.volume_score)
                        if max_vol_218 > 0.80:
                            trade_size *= 1.15
                            ml_stats.setdefault('max_vol_conf', 0)
                            ml_stats['max_vol_conf'] += 1

                    # Arch 218d: Ordered momentum (1.15x when KE*(1-entropy) avg > 0.30)
                    if realistic and sig.signal_type == 'bounce':
                        oke_218 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                oke_218.append(tf_s.kinetic_energy * (1.0 - tf_s.entropy))
                        if oke_218 and sum(oke_218)/len(oke_218) > 0.30:
                            trade_size *= 1.15
                            ml_stats.setdefault('ordered_ke', 0)
                            ml_stats['ordered_ke'] += 1


                    # Arch 219a: Break after stop = dangerous regime
                    if realistic and sig.signal_type == "break" and len(trades) >= 1:
                        any_stop_219 = any(t.exit_reason == "stop" for t in trades[-5:])
                        if any_stop_219:
                            trade_size *= 0.15
                            ml_stats.setdefault("break_after_stop", 0)
                            ml_stats["break_after_stop"] += 1


                    # Arch 219c: Long win streak regime boost (1.15x when streak > 10)
                    if realistic and len(trades) >= 10:
                        streak_219 = 0
                        for t in reversed(trades):
                            if t.pnl > 0:
                                streak_219 += 1
                            else:
                                break
                        if streak_219 > 10:
                            trade_size *= 1.15
                            ml_stats.setdefault("long_streak", 0)
                            ml_stats["long_streak"] += 1

                    # Arch 219a: Geometric mean KE (1.15x when all TFs have solid momentum)
                    if realistic and sig.signal_type == 'bounce':
                        ke_219 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                ke_219.append(max(0.01, tf_s.kinetic_energy))
                        if ke_219:
                            import math
                            geo_ke = math.exp(sum(math.log(k) for k in ke_219) / len(ke_219))
                            if geo_ke > 0.40:
                                trade_size *= 1.15
                                ml_stats.setdefault('geo_ke', 0)
                                ml_stats['geo_ke'] += 1

                    # Arch 219f: Volume * health * position triple (1.20x when all confirm edge)
                    if realistic and sig.signal_type == 'bounce':
                        vhp_219 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                vhp_219.append(tf_s.volume_score * tf_s.channel_health * abs(tf_s.position_pct))
                        if vhp_219 and sum(vhp_219)/len(vhp_219) > 0.25:
                            trade_size *= 1.20
                            ml_stats.setdefault('vol_health_pos', 0)
                            ml_stats['vol_health_pos'] += 1


                    # Arch 221d: Equity acceleration boost (1.20x when growth doubling)
                    if realistic and len(trades) >= 20:
                        eq_per_trade_recent = sum(t.pnl for t in trades[-10:]) / 10
                        eq_per_trade_prev = sum(t.pnl for t in trades[-20:-10]) / 10
                        if eq_per_trade_recent > eq_per_trade_prev * 2.0 and eq_per_trade_recent > 0:
                            trade_size *= 1.20
                            ml_stats.setdefault("eq_accel", 0)
                            ml_stats["eq_accel"] += 1


                    # Arch 221b: Same-type consecutive loss avoidance (0.20x)
                    if realistic and len(trades) >= 3:
                        same_type_loss_221 = False
                        for t in trades[-3:]:
                            if t.pnl < 0 and hasattr(t, "signal_type") and t.signal_type == sig.signal_type:
                                same_type_loss_221 = True
                                break
                        if same_type_loss_221:
                            trade_size *= 0.20
                            ml_stats.setdefault("same_type_loss", 0)
                            ml_stats["same_type_loss"] += 1


                    # Arch 221a: Confidence-position product sizing for bounces
                    if realistic and sig.signal_type == "bounce":
                        pos_221 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                pos_221.append(abs(tf_s.position_pct))
                        if pos_221:
                            avg_pos = sum(pos_221) / len(pos_221)
                            cp_prod = sig.confidence * avg_pos
                            cp_mult = min(1.30, max(0.40, 0.4 + cp_prod * 2.0))
                            trade_size *= cp_mult
                            ml_stats.setdefault("conf_pos_prod", 0)
                            ml_stats["conf_pos_prod"] += 1


                    # Arch 222: Low PnL volatility boost (1.15x when consistent returns)
                    if realistic and len(trades) >= 10:
                        pnls_222 = [t.pnl for t in trades[-10:]]
                        avg_222 = sum(pnls_222) / 10
                        std_222 = (sum((p - avg_222)**2 for p in pnls_222) / 10) ** 0.5
                        if std_222 < 50 and avg_222 > 0:
                            trade_size *= 1.15
                            ml_stats.setdefault("low_pnl_vol", 0)
                            ml_stats["low_pnl_vol"] += 1


                    # Arch 224: Bounce in profitable regime = confidence boost
                    if realistic and sig.signal_type == 'bounce' and len(trades) >= 10:
                        avg_pnl_224 = sum(t.pnl for t in trades[-10:]) / 10
                        if avg_pnl_224 > 500:
                            trade_size *= 1.15
                            ml_stats.setdefault('good_regime_bounce', 0)
                            ml_stats['good_regime_bounce'] += 1


                    # Arch 225: Near ATH equity boost (1.20x when equity near peak)
                    if realistic and len(trades) >= 5:
                        running_eq_225 = initial_capital
                        peak_eq_225 = initial_capital
                        for t in trades:
                            running_eq_225 += t.pnl
                            peak_eq_225 = max(peak_eq_225, running_eq_225)
                        if peak_eq_225 > 0 and running_eq_225 / peak_eq_225 > 0.99:
                            trade_size *= 1.20
                            ml_stats.setdefault("near_ath", 0)
                            ml_stats["near_ath"] += 1


                    # Arch 226: High win/loss ratio regime boost (1.15x)
                    if realistic and len(trades) >= 15:
                        wins_226 = [t.pnl for t in trades[-15:] if t.pnl > 0]
                        losses_226 = [abs(t.pnl) for t in trades[-15:] if t.pnl < 0]
                        if wins_226 and losses_226:
                            avg_win = sum(wins_226) / len(wins_226)
                            avg_loss = sum(losses_226) / len(losses_226)
                            if avg_loss > 0 and avg_win / avg_loss > 3.0:
                                trade_size *= 1.15
                                ml_stats.setdefault("high_rr_ratio", 0)
                                ml_stats["high_rr_ratio"] += 1

                    # Arch 223d: Break + KE > PE across majority (1.15x momentum-driven break)
                    if realistic and sig.signal_type == 'break':
                        ke_gt_pe_223 = 0
                        total_223 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                total_223 += 1
                                if tf_s.kinetic_energy > tf_s.potential_energy:
                                    ke_gt_pe_223 += 1
                        if total_223 > 0 and ke_gt_pe_223 / total_223 > 0.60:
                            trade_size *= 1.15
                            ml_stats.setdefault('momentum_break', 0)
                            ml_stats['momentum_break'] += 1

                    # Arch 224c: Bounce count range tight (1.15x fractal-like behavior)
                    if realistic and sig.signal_type == 'bounce':
                        bc_224 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                bc_224.append(tf_s.bounce_count)
                        if len(bc_224) >= 3:
                            if max(bc_224) - min(bc_224) < 5:
                                trade_size *= 1.15
                                ml_stats.setdefault('bc_tight', 0)
                                ml_stats['bc_tight'] += 1


                    # Arch 228: Lossless regime aggressive sizing (1.30x)
                    if realistic and len(trades) >= 20:
                        cum_loss_228 = abs(sum(t.pnl for t in trades if t.pnl < 0))
                        if cum_loss_228 < 20:
                            trade_size *= 1.30
                            ml_stats.setdefault("lossless_regime", 0)
                            ml_stats["lossless_regime"] += 1


                    # Arch 229d: Full energy bounce boost (1.15x when min PE+KE > 0.40)
                    if realistic and sig.signal_type == "bounce":
                        min_pe_229 = 1.0
                        min_ke_229 = 1.0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                min_pe_229 = min(min_pe_229, tf_s.potential_energy)
                                min_ke_229 = min(min_ke_229, tf_s.kinetic_energy)
                        if min_pe_229 > 0.40 and min_ke_229 > 0.40:
                            trade_size *= 1.15
                            ml_stats.setdefault("full_energy", 0)
                            ml_stats["full_energy"] += 1


                    # Arch 229f: Accelerating win sizes boost (1.15x)
                    if realistic and len(trades) >= 15:
                        wins_229 = [t.pnl for t in trades if t.pnl > 0]
                        if len(wins_229) >= 6:
                            recent_avg = sum(wins_229[-3:]) / 3
                            older_avg = sum(wins_229[-6:-3]) / 3
                            if older_avg > 0 and recent_avg > older_avg * 2.0:
                                trade_size *= 1.15
                                ml_stats.setdefault("accel_wins", 0)
                                ml_stats["accel_wins"] += 1


                    # Arch 230b: Low entropy signal boost (1.15x ordered bounce)
                    if realistic and sig.signal_type == "bounce" and sig.entropy_score < 0.30:
                        trade_size *= 1.15
                        ml_stats.setdefault("lo_ent_sig", 0)
                        ml_stats["lo_ent_sig"] += 1


                    # Arch 230a: High energy signal boost (1.15x for bounce)
                    if realistic and sig.signal_type == "bounce" and sig.energy_score > 0.80:
                        trade_size *= 1.15
                        ml_stats.setdefault("hi_energy_sig", 0)
                        ml_stats["hi_energy_sig"] += 1


                    # Arch 231: Historical big win = market has opportunity
                    if realistic and len(trades) >= 10:
                        max_pnl_231 = max(t.pnl for t in trades)
                        if max_pnl_231 > 10000:
                            trade_size *= 1.15
                            ml_stats.setdefault('big_opp_history', 0)
                            ml_stats['big_opp_history'] += 1

                    # Arch 227e: Theta decreasing across TFs (1.15x higher TFs more stable)
                    if realistic and sig.signal_type == 'bounce':
                        tf_order_227 = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly']
                        th_ord = []
                        for tf_n in tf_order_227:
                            tf_s = analysis.tf_states.get(tf_n)
                            if tf_s and tf_s.valid:
                                th_ord.append(tf_s.ou_theta)
                        if len(th_ord) >= 3:
                            decreases = sum(1 for i in range(1, len(th_ord)) if th_ord[i] < th_ord[i-1])
                            if decreases >= len(th_ord) - 1:
                                trade_size *= 1.15
                                ml_stats.setdefault('theta_decr', 0)
                                ml_stats['theta_decr'] += 1


                    # Arch 232: Extreme position score signal boost (1.15x)
                    if realistic and sig.signal_type == "bounce" and sig.position_score > 0.95:
                        trade_size *= 1.15
                        ml_stats.setdefault("extreme_pos_sig", 0)
                        ml_stats["extreme_pos_sig"] += 1


                    # Arch 233: High confidence bounce = trusted signal
                    if realistic and sig.signal_type == 'bounce' and sig.confidence > 0.70:
                        trade_size *= 1.15
                        ml_stats.setdefault('hi_conf_bounce', 0)
                        ml_stats['hi_conf_bounce'] += 1

                    # Arch 228c: Confidence * energy product (1.15x high conviction + energy)
                    if realistic and sig.signal_type == 'bounce':
                        ce_prod = sig.confidence * sig.energy_score
                        if ce_prod > 0.60:
                            trade_size *= 1.15
                            ml_stats.setdefault('conf_energy', 0)
                            ml_stats['conf_energy'] += 1

                    # Arch 228f: Min channel health floor (1.15x when all TFs healthy)
                    if realistic and sig.signal_type == 'bounce':
                        min_h_228 = 1.0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                min_h_228 = min(min_h_228, tf_s.channel_health)
                        if min_h_228 > 0.40:
                            trade_size *= 1.15
                            ml_stats.setdefault('min_health_ok', 0)
                            ml_stats['min_health_ok'] += 1

                    # Arch 230b: All TFs at edge (1.20x when min position > 0.60)
                    if realistic and sig.signal_type == 'bounce':
                        min_pos_230 = 1.0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                min_pos_230 = min(min_pos_230, abs(tf_s.position_pct))
                        if min_pos_230 > 0.60:
                            trade_size *= 1.20
                            ml_stats.setdefault('all_extreme', 0)
                            ml_stats['all_extreme'] += 1


                    # Arch 234f: PnL trend up boost (1.15x when improving)
                    if realistic and len(trades) >= 20:
                        recent20_234 = [t.pnl for t in trades[-20:]]
                        first_half_234 = sum(recent20_234[:10]) / 10
                        second_half_234 = sum(recent20_234[10:]) / 10
                        if second_half_234 > first_half_234 * 1.5 and second_half_234 > 0:
                            trade_size *= 1.15
                            ml_stats.setdefault("pnl_trend_up", 0)
                            ml_stats["pnl_trend_up"] += 1


                    # Arch 234a: Equity tripled compound boost (1.15x)
                    if realistic and equity > initial_capital * 3.0:
                        trade_size *= 1.15
                        ml_stats.setdefault("triple_eq", 0)
                        ml_stats["triple_eq"] += 1


                    # Arch 235b: Hot momentum bounce boost (1.15x after big win)
                    if realistic and sig.signal_type == "bounce" and len(trades) >= 1:
                        if trades[-1].pnl > 2000:
                            trade_size *= 1.15
                            ml_stats.setdefault("hot_momentum", 0)
                            ml_stats["hot_momentum"] += 1


                    # Arch 235e: Clean exit bounce boost (1.15x all trail exits)
                    if realistic and sig.signal_type == "bounce" and len(trades) >= 5:
                        all_trail_235 = all(t.exit_reason == "trail" for t in trades[-5:])
                        if all_trail_235:
                            trade_size *= 1.15
                            ml_stats.setdefault("clean_exit_bounce", 0)
                            ml_stats["clean_exit_bounce"] += 1

                    # Arch 238c: 5x equity with bounce = aggressive compound
                    if realistic and sig.signal_type == 'bounce' and equity > initial_capital * 5.0:
                        trade_size *= 1.30
                        ml_stats.setdefault('5x_compound', 0)
                        ml_stats['5x_compound'] += 1

                    # Arch 239e: Very mature system (trade >300) bounce boost
                    if realistic and sig.signal_type == "bounce" and len(trades) > 300:
                        trade_size *= 1.20
                        ml_stats.setdefault("mature_system", 0)
                        ml_stats["mature_system"] += 1

                    # Arch 241a: 7x equity bounce boost (1.25x)
                    if realistic and sig.signal_type == "bounce" and equity > initial_capital * 7.0:
                        trade_size *= 1.25
                        ml_stats.setdefault("7x_bounce", 0)
                        ml_stats["7x_bounce"] += 1

                    # Arch 241e: Total PE across all TFs > 2.5 (1.20x spring loaded)
                    if realistic and sig.signal_type == "bounce":
                        total_pe_241 = sum(tf_s.potential_energy for tf_n, tf_s in analysis.tf_states.items() if tf_s and tf_s.valid)
                        if total_pe_241 > 2.5:
                            trade_size *= 1.20
                            ml_stats.setdefault("sys_spring", 0)
                            ml_stats["sys_spring"] += 1

                    # Arch 242c: System-wide total energy > 4.0 bounce boost (1.20x)
                    if realistic and sig.signal_type == "bounce":
                        sys_te_242 = sum(tf_s.total_energy for tf_n, tf_s in analysis.tf_states.items() if tf_s and tf_s.valid)
                        if sys_te_242 > 4.0:
                            trade_size *= 1.20
                            ml_stats.setdefault("sys_total_e", 0)
                            ml_stats["sys_total_e"] += 1

                    # Arch 240d: SELL break in all-downtrend (1.15x all slopes negative)
                    if realistic and sig.signal_type == 'break' and hasattr(sig, 'action') and sig.action == 'SELL':
                        all_neg_240 = True
                        count_240 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                count_240 += 1
                                if tf_s.slope_pct >= 0:
                                    all_neg_240 = False
                        if all_neg_240 and count_240 >= 3:
                            trade_size *= 1.15
                            ml_stats.setdefault('sell_all_down', 0)
                            ml_stats['sell_all_down'] += 1

                    # Arch 241c: Geomean(volume * (1-entropy) * position) > 0.15
                    if realistic and sig.signal_type == 'bounce':
                        vep_241 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                vep_241.append(max(0.01, tf_s.volume_score * (1.0 - tf_s.entropy) * abs(tf_s.position_pct)))
                        if vep_241:
                            import math
                            geo_vep = math.exp(sum(math.log(x) for x in vep_241) / len(vep_241))
                            if geo_vep > 0.15:
                                trade_size *= 1.15
                                ml_stats.setdefault('geo_vep', 0)
                                ml_stats['geo_vep'] += 1

                    # Arch 244f: ATH + large equity bounce boost (1.20x)
                    if realistic and sig.signal_type == "bounce" and equity >= peak_equity * 0.999 and equity > 500000:
                        trade_size *= 1.20
                        ml_stats.setdefault("ath_500k", 0)
                        ml_stats["ath_500k"] += 1

                    # Arch 242c: Max position*volume > 0.70 (1.15x vol-confirmed edge)
                    if realistic and sig.signal_type == 'bounce':
                        max_pv_242 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                max_pv_242 = max(max_pv_242, abs(tf_s.position_pct) * tf_s.volume_score)
                        if max_pv_242 > 0.70:
                            trade_size *= 1.15
                            ml_stats.setdefault('edge_vol', 0)
                            ml_stats['edge_vol'] += 1


                    # Arch 246b: When historical GW/GL ratio is extreme = almost no risk
                    if realistic and sig.signal_type == 'bounce' and len(trades) >= 50:
                        gw_246 = sum(t.pnl for t in trades if t.pnl > 0)
                        gl_246 = abs(sum(t.pnl for t in trades if t.pnl <= 0))
                        if gl_246 > 0 and gw_246 / gl_246 > 5000:
                            trade_size *= 1.25
                            ml_stats.setdefault('extreme_gw_gl', 0)
                            ml_stats['extreme_gw_gl'] += 1

                    # Arch 243b: Best TF position*r_sq*health > 0.40 (1.15x)
                    if realistic and sig.signal_type == 'bounce':
                        max_prh_243 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                prh = abs(tf_s.position_pct) * tf_s.r_squared * tf_s.channel_health
                                max_prh_243 = max(max_prh_243, prh)
                        if max_prh_243 > 0.40:
                            trade_size *= 1.15
                            ml_stats.setdefault('best_prh', 0)
                            ml_stats['best_prh'] += 1


                    # Arch 247d: Very established system on bounce
                    if realistic and sig.signal_type == 'bounce' and len(trades) > 250:
                        trade_size *= 1.15
                        ml_stats.setdefault('late_established', 0)
                        ml_stats['late_established'] += 1

                    # Arch 244a: BUY bounce + positive momentum consensus (1.15x)
                    if realistic and sig.signal_type == 'bounce' and hasattr(sig, 'action') and sig.action == 'BUY':
                        md_244 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                md_244.append(tf_s.momentum_direction)
                        if md_244 and sum(md_244)/len(md_244) > 0.30:
                            trade_size *= 1.15
                            ml_stats.setdefault('buy_bull_mom', 0)
                            ml_stats['buy_bull_mom'] += 1

                    # Arch 244e: Break + aligned TE*momentum > 0.50 (1.15x)
                    if realistic and sig.signal_type == 'break' and hasattr(sig, 'action'):
                        max_tm_244 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                if sig.action == 'BUY':
                                    tm = tf_s.total_energy * max(0, tf_s.momentum_direction)
                                else:
                                    tm = tf_s.total_energy * max(0, -tf_s.momentum_direction)
                                max_tm_244 = max(max_tm_244, tm)
                        if max_tm_244 > 0.50:
                            trade_size *= 1.15
                            ml_stats.setdefault('break_aligned_te', 0)
                            ml_stats['break_aligned_te'] += 1


                    # Arch 245b: Severe loss cluster detection
                    if realistic and len(trades) >= 20:
                        losses_245 = sum(1 for t in trades[-20:] if t.pnl < 0)
                        if losses_245 >= 3:
                            trade_size *= 0.10
                            ml_stats.setdefault('severe_cluster', 0)
                            ml_stats['severe_cluster'] += 1

                    # Arch 247f: ATH + 700K bounce boost (1.25x)
                    if realistic and sig.signal_type == 'bounce' and equity > 700000 and equity >= peak_equity * 0.999:
                        trade_size *= 1.25
                        ml_stats.setdefault('ath_700k', 0)
                        ml_stats['ath_700k'] += 1

                    # Arch 247c: 600K+ equity bounce compound (1.20x)
                    if realistic and sig.signal_type == 'bounce' and equity > 600000:
                        trade_size *= 1.20
                        ml_stats.setdefault('600k_bounce', 0)
                        ml_stats['600k_bounce'] += 1

                    # Arch 245d: Heavy binding energy penalty (0.20x when avg BE > 0.60)
                    if realistic and sig.signal_type == 'bounce':
                        be_245 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                be_245.append(tf_s.binding_energy)
                        if be_245 and sum(be_245)/len(be_245) > 0.60:
                            trade_size *= 0.20
                            ml_stats.setdefault('heavy_bind', 0)
                            ml_stats['heavy_bind'] += 1

                    # Arch 248a: Avoid bounce after timeout loss (0.05x)
                    if realistic and sig.signal_type == 'bounce' and len(trades) >= 1:
                        for t in reversed(trades):
                            if hasattr(t, 'signal_type') and t.signal_type == 'bounce':
                                if hasattr(t, 'exit_reason') and t.exit_reason == 'ou_timeout':
                                    trade_size *= 0.05
                                    ml_stats.setdefault('timeout_avoid', 0)
                                    ml_stats['timeout_avoid'] += 1
                                break

                    # Arch 248f: Extreme break energy + position (1.20x)
                    if realistic and sig.signal_type == 'break':
                        if sig.energy_score > 0.98 and sig.position_score > 0.90:
                            trade_size *= 1.20
                            ml_stats.setdefault('extreme_break', 0)
                            ml_stats['extreme_break'] += 1

                    # Arch 249f: Late established compound (1.20x when 4x eq + 200 trades)
                    if realistic and equity > initial_capital * 4.0 and len(trades) > 200:
                        trade_size *= 1.20
                        ml_stats.setdefault('late_funded', 0)
                        ml_stats['late_funded'] += 1

                    # Arch 249c: Equity surge acceleration (1.20x when +$100K in 50 trades)
                    if realistic and len(trades) >= 50:
                        eq_50_ago = 100000 + sum(t.pnl for t in trades[:-50])
                        if equity - eq_50_ago > 100000:
                            trade_size *= 1.20
                            ml_stats.setdefault('eq_surge', 0)
                            ml_stats['eq_surge'] += 1

                    # Arch 250d: All TFs established channels bounce boost (1.15x)
                    if realistic and sig.signal_type == 'bounce':
                        min_bc_250 = 99
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                min_bc_250 = min(min_bc_250, tf_s.bounce_count)
                        if min_bc_250 >= 2:
                            trade_size *= 1.15
                            ml_stats.setdefault('all_established', 0)
                            ml_stats['all_established'] += 1

                    # Arch 252e: Rapid recent equity growth
                    if realistic and len(trades) >= 30:
                        recent_pnl = sum(t.pnl for t in trades[-30:])
                        if recent_pnl > 50000:
                            trade_size *= 1.20
                            ml_stats.setdefault('rapid_growth', 0)
                            ml_stats['rapid_growth'] += 1

                    # Arch 251c: Loss cooldown (0.05x immediately after a loss)
                    if realistic and len(trades) >= 1 and trades[-1].pnl < 0:
                        trade_size *= 0.05
                        ml_stats.setdefault('loss_cooldown', 0)
                        ml_stats['loss_cooldown'] += 1

                    # Arch 252b: Near $1M equity push (1.30x when equity > 950K bounce)
                    if realistic and sig.signal_type == 'bounce' and equity > 950000:
                        trade_size *= 1.30
                        ml_stats.setdefault('1m_push', 0)
                        ml_stats['1m_push'] += 1

                    # Arch 252a: Final stretch bounce boost (1.25x after 320 trades)
                    if realistic and sig.signal_type == 'bounce' and len(trades) > 320:
                        trade_size *= 1.25
                        ml_stats.setdefault('final_stretch', 0)
                        ml_stats['final_stretch'] += 1

                    # Arch 253d: Hot bounce regime (1.20x when last 10 trades > $20K PnL)
                    if realistic and sig.signal_type == 'bounce' and len(trades) >= 10:
                        if sum(t.pnl for t in trades[-10:]) > 20000:
                            trade_size *= 1.20
                            ml_stats.setdefault('hot_bounce_10', 0)
                            ml_stats['hot_bounce_10'] += 1

                    # Arch 254f: Low entropy + high energy bounce (1.20x ordered + energetic)
                    if realistic and sig.signal_type == 'bounce':
                        if sig.entropy_score < 0.30 and sig.energy_score > 0.60:
                            trade_size *= 1.20
                            ml_stats.setdefault('ordered_energy', 0)
                            ml_stats['ordered_energy'] += 1

                    # Arch 255a: High equity + high energy double gate (1.25x)
                    if realistic and sig.signal_type == 'bounce' and equity > 500000:
                        sys_te_255 = sum(tf_s.total_energy for tf_n, tf_s in analysis.tf_states.items() if tf_s and tf_s.valid)
                        if sys_te_255 > 3.5:
                            trade_size *= 1.25
                            ml_stats.setdefault('eq_energy_gate', 0)
                            ml_stats['eq_energy_gate'] += 1

                    # Arch 255f: Big win momentum bounce (1.20x after $5K+ win)
                    if realistic and sig.signal_type == 'bounce' and len(trades) >= 1:
                        if trades[-1].pnl > 5000:
                            trade_size *= 1.20
                            ml_stats.setdefault('big_win_mom', 0)
                            ml_stats['big_win_mom'] += 1

                    # Arch 257a: Millionaire compound
                    if realistic and sig.signal_type == 'bounce' and equity > 1000000:
                        trade_size *= 1.50
                        ml_stats.setdefault('millionaire_bounce', 0)
                        ml_stats['millionaire_bounce'] += 1

                    # Arch 255d: Mid compound phase boost (1.10x when 5x-10x initial)
                    if realistic and equity > initial_capital * 5 and equity < initial_capital * 10:
                        trade_size *= 1.10
                        ml_stats.setdefault('mid_comp', 0)
                        ml_stats['mid_comp'] += 1

                    # Arch 256e: All TFs energized floor (1.15x when min total energy > 0.50)
                    if realistic and sig.signal_type == 'bounce':
                        min_te_256 = 999
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                min_te_256 = min(min_te_256, tf_s.potential_energy + tf_s.kinetic_energy)
                        if min_te_256 > 0.50 and min_te_256 < 999:
                            trade_size *= 1.15
                            ml_stats.setdefault('all_energized', 0)
                            ml_stats['all_energized'] += 1

                    # Arch 258: Bounce in danger zone
                    if realistic and sig.signal_type == 'bounce' and 250000 < equity < 400000:
                        trade_size *= 0.7
                        ml_stats.setdefault('danger_zone', 0)
                        ml_stats['danger_zone'] += 1

                    # Arch 257b: Free potential energy boost (1.15x geomean PE*(1-BE) > 0.25)
                    if realistic and sig.signal_type == 'bounce':
                        pbe_257 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                pbe_257.append(max(0.01, tf_s.potential_energy * (1.0 - tf_s.binding_energy)))
                        if pbe_257:
                            import math
                            geo_pbe = math.exp(sum(math.log(x) for x in pbe_257) / len(pbe_257))
                            if geo_pbe > 0.25:
                                trade_size *= 1.15
                                ml_stats.setdefault('free_pe', 0)
                                ml_stats['free_pe'] += 1

                    # Arch 259f: Spring loaded + equity gate bounce boost (1.15x)
                    if realistic and sig.signal_type == 'bounce' and equity > 400000:
                        pe_sum_259 = sum(tf_s.potential_energy for tf_n, tf_s in analysis.tf_states.items() if tf_s and tf_s.valid)
                        if pe_sum_259 > 2.0:
                            trade_size *= 1.15
                            ml_stats.setdefault('spring_eq_400', 0)
                            ml_stats['spring_eq_400'] += 1

                    # Arch 260b: High system energy break
                    if realistic and sig.signal_type == 'break':
                        te_sum = sum(tf_s.total_energy for tf_n, tf_s in analysis.tf_states.items() if tf_s and tf_s.valid)
                        if te_sum > 4.5:
                            trade_size *= 1.15
                            ml_stats.setdefault('high_te_break', 0)
                            ml_stats['high_te_break'] += 1

                    # Arch 260d: Avg position*health > 0.50 bounce (1.15x high quality edge)
                    if realistic and sig.signal_type == 'bounce':
                        ph_260 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                ph_260.append(abs(tf_s.position_pct) * tf_s.channel_health)
                        if ph_260 and sum(ph_260)/len(ph_260) > 0.50:
                            trade_size *= 1.15
                            ml_stats.setdefault('hi_q_edge', 0)
                            ml_stats['hi_q_edge'] += 1

                    # Arch 260f: Avg PE > 0.55 AND avg position > 0.50 bounce (1.15x spring at edge)
                    if realistic and sig.signal_type == 'bounce':
                        pe_260 = []
                        pos_260 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                pe_260.append(tf_s.potential_energy)
                                pos_260.append(abs(tf_s.position_pct))
                        if pe_260 and pos_260:
                            if sum(pe_260)/len(pe_260) > 0.55 and sum(pos_260)/len(pos_260) > 0.50:
                                trade_size *= 1.15
                                ml_stats.setdefault('spring_edge', 0)
                                ml_stats['spring_edge'] += 1

                    # Arch 260f: Massive win momentum bounce (1.25x after $10K+ win)
                    if realistic and sig.signal_type == 'bounce' and len(trades) >= 1:
                        if trades[-1].pnl > 10000:
                            trade_size *= 1.25
                            ml_stats.setdefault('massive_mom', 0)
                            ml_stats['massive_mom'] += 1

                    # Arch 260d: Millionaire + high energy bounce (1.30x)
                    if realistic and sig.signal_type == 'bounce' and equity > 1000000:
                        te_sum_260 = sum(tf_s.total_energy for tf_n, tf_s in analysis.tf_states.items() if tf_s and tf_s.valid)
                        if te_sum_260 > 3.5:
                            trade_size *= 1.30
                            ml_stats.setdefault('1m_te', 0)
                            ml_stats['1m_te'] += 1

                    # Arch 261a: $1.1M+ equity bounce boost (1.25x)
                    if realistic and sig.signal_type == 'bounce' and equity > 1100000:
                        trade_size *= 1.25
                        ml_stats.setdefault('1_1m_bounce', 0)
                        ml_stats['1_1m_bounce'] += 1

                    # Arch 261e: Explosive PnL regime bounce (1.25x when $100K in 30 trades)
                    if realistic and sig.signal_type == 'bounce' and len(trades) >= 30:
                        if sum(t.pnl for t in trades[-30:]) > 100000:
                            trade_size *= 1.25
                            ml_stats.setdefault('explosion', 0)
                            ml_stats['explosion'] += 1

                    # Arch 262d: $1M + spring loaded
                    if realistic and sig.signal_type == 'bounce' and equity > 1000000:
                        pe_sum = sum(tf_s.potential_energy for tf_n, tf_s in analysis.tf_states.items() if tf_s and tf_s.valid)
                        if pe_sum > 1.5:
                            trade_size *= 1.20
                            ml_stats.setdefault('1m_spring', 0)
                            ml_stats['1m_spring'] += 1

                    # Arch 262d: Geomean(health * position * r_sq) > 0.15 (1.15x 3-factor geo)
                    if realistic and sig.signal_type == 'bounce':
                        hpr_262 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                hpr_262.append(max(0.01, tf_s.channel_health * abs(tf_s.position_pct) * tf_s.r_squared))
                        if hpr_262:
                            import math
                            geo_hpr = math.exp(sum(math.log(x) for x in hpr_262) / len(hpr_262))
                            if geo_hpr > 0.15:
                                trade_size *= 1.15
                                ml_stats.setdefault('geo_hpr', 0)
                                ml_stats['geo_hpr'] += 1

                    # Arch 262e: Avg(PE * volume * health) > 0.15 (1.15x energy-volume-quality)
                    if realistic and sig.signal_type == 'bounce':
                        pvh_262 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                pvh_262.append(tf_s.potential_energy * tf_s.volume_score * tf_s.channel_health)
                        if pvh_262 and sum(pvh_262)/len(pvh_262) > 0.15:
                            trade_size *= 1.15
                            ml_stats.setdefault('pe_vol_h', 0)
                            ml_stats['pe_vol_h'] += 1

                    # Arch 262a: High energy + moderate equity bounce (1.20x)
                    if realistic and sig.signal_type == 'bounce' and equity > 300000:
                        te_sum_262 = sum(tf_s.total_energy for tf_n, tf_s in analysis.tf_states.items() if tf_s and tf_s.valid)
                        if te_sum_262 > 5.0:
                            trade_size *= 1.20
                            ml_stats.setdefault('te5_300k', 0)
                            ml_stats['te5_300k'] += 1

                    # Arch 264e: After any bounce hit stop, reduce bounces
                    if realistic and sig.signal_type == 'bounce' and len(trades) >= 1:
                        had_bounce_stop = any(
                            hasattr(t, 'signal_type') and t.signal_type == 'bounce' and
                            hasattr(t, 'exit_reason') and t.exit_reason == 'stop' and t.pnl < -1
                            for t in trades[-30:]
                        )
                        if had_bounce_stop:
                            trade_size *= 0.30
                            ml_stats.setdefault('bounce_stop_pen', 0)
                            ml_stats['bounce_stop_pen'] += 1

                    # Arch 263a: 5-factor best TF product (1.15x when PE*pos*health*r_sq*vol > 0.05)
                    if realistic and sig.signal_type == 'bounce':
                        max_5f_263 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                f5 = tf_s.potential_energy * abs(tf_s.position_pct) * tf_s.channel_health * tf_s.r_squared * tf_s.volume_score
                                max_5f_263 = max(max_5f_263, f5)
                        if max_5f_263 > 0.05:
                            trade_size *= 1.15
                            ml_stats.setdefault('5f_best', 0)
                            ml_stats['5f_best'] += 1

                    # Arch 263e: High avg PE bounce (1.15x when avg potential energy > 0.60)
                    if realistic and sig.signal_type == 'bounce':
                        pe_263 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                pe_263.append(tf_s.potential_energy)
                        if pe_263 and sum(pe_263)/len(pe_263) > 0.60:
                            trade_size *= 1.15
                            ml_stats.setdefault('hi_avg_pe', 0)
                            ml_stats['hi_avg_pe'] += 1

                    # Arch 265b: TF-weighted geomean PE*health (1.15x when > 0.25)
                    if realistic and sig.signal_type == 'bounce':
                        tf_wt = {'5min': 0.5, '15min': 0.6, '30min': 0.7, '1h': 0.8, '2h': 0.85, '3h': 0.9, '4h': 0.95, 'daily': 1.2, 'weekly': 1.5, 'monthly': 2.0}
                        wph_265 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid and tf_n in tf_wt:
                                wph_265.append(max(0.01, tf_wt[tf_n] * tf_s.potential_energy * tf_s.channel_health))
                        if wph_265:
                            import math
                            geo_wph = math.exp(sum(math.log(x) for x in wph_265) / len(wph_265))
                            if geo_wph > 0.25:
                                trade_size *= 1.15
                                ml_stats.setdefault('wt_geo_peh', 0)
                                ml_stats['wt_geo_peh'] += 1

                    # Arch 265d: Max TF PE > 0.80 extreme spring (1.15x)
                    if realistic and sig.signal_type == 'bounce':
                        max_pe_265 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                max_pe_265 = max(max_pe_265, tf_s.potential_energy)
                        if max_pe_265 > 0.80:
                            trade_size *= 1.15
                            ml_stats.setdefault('extreme_pe', 0)
                            ml_stats['extreme_pe'] += 1

                    # Arch 265f: Avg position*(1-entropy) > 0.40 ordered edge (1.15x)
                    if realistic and sig.signal_type == 'bounce':
                        pe_265f = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                pe_265f.append(abs(tf_s.position_pct) * (1.0 - tf_s.entropy))
                        if pe_265f and sum(pe_265f)/len(pe_265f) > 0.40:
                            trade_size *= 1.15
                            ml_stats.setdefault('ordered_edge_265', 0)
                            ml_stats['ordered_edge_265'] += 1


                    # Arch 266b: Even stricter ordered edge threshold with bigger boost
                    if realistic and sig.signal_type == 'bounce':
                        oe_266 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                oe_266.append(abs(tf_s.position_pct) * (1.0 - tf_s.entropy))
                        if oe_266 and sum(oe_266)/len(oe_266) > 0.50:
                            trade_size *= 1.20
                            ml_stats.setdefault('ultra_ordered_edge', 0)
                            ml_stats['ultra_ordered_edge'] += 1


                    # Arch 266c: $1.3M equity gate
                    if realistic and sig.signal_type == 'bounce' and equity > 1300000:
                        trade_size *= 1.25
                        ml_stats.setdefault('1_3m', 0)
                        ml_stats['1_3m'] += 1

                    # Arch 266c: 5-factor geomean (1.15x when PE*pos*health*vol*(1-ent) geo > 0.02)
                    if realistic and sig.signal_type == 'bounce':
                        f5_266 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                f5_266.append(max(0.001, tf_s.potential_energy * abs(tf_s.position_pct) * tf_s.channel_health * tf_s.volume_score * (1.0 - tf_s.entropy)))
                        if f5_266:
                            import math
                            geo_5f = math.exp(sum(math.log(x) for x in f5_266) / len(f5_266))
                            if geo_5f > 0.02:
                                trade_size *= 1.15
                                ml_stats.setdefault('5f_geo_full', 0)
                                ml_stats['5f_geo_full'] += 1

                    # Arch 266f: Avg PE * position * reversion > 0.20 (1.15x spring-reversion-edge)
                    if realistic and sig.signal_type == 'bounce':
                        ppr_266 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                ppr_266.append(tf_s.potential_energy * abs(tf_s.position_pct) * tf_s.ou_reversion_score)
                        if ppr_266 and sum(ppr_266)/len(ppr_266) > 0.20:
                            trade_size *= 1.15
                            ml_stats.setdefault('pe_pos_rev', 0)
                            ml_stats['pe_pos_rev'] += 1


                    # Arch 267d: Mid-game and equity near peak = losses recovered, press harder
                    if realistic and sig.signal_type == 'bounce' and 200000 < equity < 500000:
                        if equity >= peak_equity * 0.99:
                            trade_size *= 1.15
                            ml_stats.setdefault('mid_recovered', 0)
                            ml_stats['mid_recovered'] += 1

                    # Arch 267c: 4-factor evidence (1.15x when max pos*r_sq*vol*rev > 0.10)
                    if realistic and sig.signal_type == 'bounce':
                        max_4ev_267 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                ev4 = abs(tf_s.position_pct) * tf_s.r_squared * tf_s.volume_score * tf_s.ou_reversion_score
                                max_4ev_267 = max(max_4ev_267, ev4)
                        if max_4ev_267 > 0.10:
                            trade_size *= 1.15
                            ml_stats.setdefault('4f_evidence', 0)
                            ml_stats['4f_evidence'] += 1

                    # Arch 267f: All-TF floor (1.15x when min pos > 0.30 AND min PE > 0.30)
                    if realistic and sig.signal_type == 'bounce':
                        min_pos_267 = 1.0
                        min_pe_267 = 1.0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                min_pos_267 = min(min_pos_267, abs(tf_s.position_pct))
                                min_pe_267 = min(min_pe_267, tf_s.potential_energy)
                        if min_pos_267 > 0.30 and min_pe_267 > 0.30:
                            trade_size *= 1.15
                            ml_stats.setdefault('all_tf_floor', 0)
                            ml_stats['all_tf_floor'] += 1


                    # Arch 268a: Very high average PE = aggressive 1.20x
                    if realistic and sig.signal_type == 'bounce':
                        pe_268 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                pe_268.append(tf_s.potential_energy)
                        if pe_268 and sum(pe_268)/len(pe_268) > 0.70:
                            trade_size *= 1.20
                            ml_stats.setdefault('very_hi_pe', 0)
                            ml_stats['very_hi_pe'] += 1


                    # Arch 269d: At least one TF has both poor fit and poor health
                    if realistic and sig.signal_type == 'bounce':
                        min_r2 = 1.0
                        min_h = 1.0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                min_r2 = min(min_r2, tf_s.r_squared)
                                min_h = min(min_h, tf_s.channel_health)
                        if min_r2 < 0.10 and min_h < 0.30:
                            trade_size *= 0.10
                            ml_stats.setdefault('poor_tf', 0)
                            ml_stats['poor_tf'] += 1

                    # Arch 270b: Chaotic momentum bounce penalty (0.10x when avg KE > 0.60 AND avg entropy > 0.60)
                    if realistic and sig.signal_type == 'bounce':
                        ke_270 = []
                        ent_270b = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                ke_270.append(tf_s.kinetic_energy)
                                ent_270b.append(tf_s.entropy)
                        if ke_270 and ent_270b:
                            if sum(ke_270)/len(ke_270) > 0.60 and sum(ent_270b)/len(ent_270b) > 0.60:
                                trade_size *= 0.10
                                ml_stats.setdefault('chaos_mom_bounce', 0)
                                ml_stats['chaos_mom_bounce'] += 1

                    # Arch 270c: Extreme dual warning penalty (0.05x when max BE > 0.70 AND max entropy > 0.80)
                    if realistic:
                        max_be_270 = 0
                        max_ent_270 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                max_be_270 = max(max_be_270, tf_s.binding_energy)
                                max_ent_270 = max(max_ent_270, tf_s.entropy)
                        if max_be_270 > 0.70 and max_ent_270 > 0.80:
                            trade_size *= 0.05
                            ml_stats.setdefault('extreme_dual', 0)
                            ml_stats['extreme_dual'] += 1

                    # Arch 271e: 4-factor quality avg (1.15x when avg PE*(1-ent)*health*vol > 0.06)
                    if realistic and sig.signal_type == 'bounce':
                        f4_271 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                f4_271.append(tf_s.potential_energy * (1.0 - tf_s.entropy) * tf_s.channel_health * tf_s.volume_score)
                        if f4_271 and sum(f4_271)/len(f4_271) > 0.06:
                            trade_size *= 1.15
                            ml_stats.setdefault('4f_q_avg', 0)
                            ml_stats['4f_q_avg'] += 1


                    # Arch 272c: Early confident break = worth pressing
                    if realistic and sig.signal_type == 'break' and equity < 200000 and sig.confidence > 0.75:
                        trade_size *= 1.20
                        ml_stats.setdefault('early_conf_brk', 0)
                        ml_stats['early_conf_brk'] += 1


                    # Arch 273b: Good timing + equity gate
                    if realistic and sig.signal_type == 'bounce' and equity > 400000 and sig.timing_score > 0.70:
                        trade_size *= 1.15
                        ml_stats.setdefault('timed_eq', 0)
                        ml_stats['timed_eq'] += 1

                    # Arch 274c: 6-factor geomean boost (1.15x PE*pos*health*vol*r_sq*(1-ent) geo > 0.01)
                    if realistic and sig.signal_type == 'bounce':
                        f6_274 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                f6_274.append(max(0.001, tf_s.potential_energy * abs(tf_s.position_pct) * tf_s.channel_health * tf_s.volume_score * tf_s.r_squared * (1.0 - tf_s.entropy)))
                        if f6_274:
                            import math
                            geo_6f = math.exp(sum(math.log(x) for x in f6_274) / len(f6_274))
                            if geo_6f > 0.01:
                                trade_size *= 1.15
                                ml_stats.setdefault('6f_geo', 0)
                                ml_stats['6f_geo'] += 1

                    # Arch 274e: Late high-quality spring boost (1.20x equity>700K AND avg PE*pos>0.40)
                    if realistic and sig.signal_type == 'bounce' and equity > 700000:
                        pp_274 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                pp_274.append(tf_s.potential_energy * abs(tf_s.position_pct))
                        if pp_274 and sum(pp_274)/len(pp_274) > 0.40:
                            trade_size *= 1.20
                            ml_stats.setdefault('late_hq', 0)
                            ml_stats['late_hq'] += 1

                    # Arch 274f: All-TF quality fit floor boost (1.15x min health*r_sq > 0.15)
                    if realistic and sig.signal_type == 'bounce':
                        min_hr_274 = 1.0
                        cnt_274 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                min_hr_274 = min(min_hr_274, tf_s.channel_health * tf_s.r_squared)
                                cnt_274 += 1
                        if cnt_274 >= 3 and min_hr_274 > 0.15:
                            trade_size *= 1.15
                            ml_stats.setdefault('all_qfit', 0)
                            ml_stats['all_qfit'] += 1

                    # Arch 278c: Clean break regime (1.15x when no loss in last 20 trades)
                    if realistic and sig.signal_type == "break" and len(trades) >= 20:
                        if all(t.pnl > 0 for t in trades[-20:]):
                            trade_size *= 1.15
                            ml_stats.setdefault("clean_brk_20", 0)
                            ml_stats["clean_brk_20"] += 1

                    # Arch 278b: Endgame stride boost (1.20x trades 300+ AND equity > 800K)
                    if realistic and sig.signal_type == 'bounce' and len(trades) >= 300 and equity > 800000:
                        trade_size *= 1.20
                        ml_stats.setdefault('endgame_stride', 0)
                        ml_stats['endgame_stride'] += 1


                    # Arch 282a: Well-timed and confident bounce
                    if realistic and sig.signal_type == 'bounce' and sig.timing_score > 0.80 and sig.confidence > 0.80:
                        trade_size *= 1.15
                        ml_stats.setdefault('timed_conf', 0)
                        ml_stats['timed_conf'] += 1


                    # Arch 281d: Early bounce danger zone
                    if realistic and sig.signal_type == 'bounce' and 150000 < equity < 200000:
                        trade_size *= 0.80
                        ml_stats.setdefault('early_dz', 0)
                        ml_stats['early_dz'] += 1

                    # Arch 282e: Volume-confirmed displacement boost (1.15x sum vol*pos > 3.0)
                    if realistic and sig.signal_type == 'bounce':
                        vp_282 = sum(tf_s.volume_score * abs(tf_s.position_pct) for tf_n, tf_s in analysis.tf_states.items() if tf_s and tf_s.valid)
                        if vp_282 > 3.0:
                            trade_size *= 1.15
                            ml_stats.setdefault('vol_disp', 0)
                            ml_stats['vol_disp'] += 1


                    # Arch 284c: Break signal with high PE at significant equity
                    if realistic and sig.signal_type == 'break' and equity > 500000:
                        pe_284 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                pe_284.append(tf_s.potential_energy)
                        if pe_284 and sum(pe_284)/len(pe_284) > 0.50:
                            trade_size *= 1.20
                            ml_stats.setdefault('brk_spring', 0)
                            ml_stats['brk_spring'] += 1


                    if realistic and sig.signal_type == 'bounce' and trade_size > equity:
                        trade_size *= 0.50
                        ml_stats.setdefault('lev_cap', 0)
                        ml_stats['lev_cap'] += 1

                    # Arch 285f: Break-risk + weak position bounce penalty (0.30x max break_prob>0.50 + pos_score<0.60)
                    if realistic and sig.signal_type == 'bounce':
                        max_bp_285 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                max_bp_285 = max(max_bp_285, tf_s.break_prob)
                        if max_bp_285 > 0.50 and sig.position_score < 0.60:
                            trade_size *= 0.30
                            ml_stats.setdefault('brk_weak_pos', 0)
                            ml_stats['brk_weak_pos'] += 1

                    # Arch 284f: Wide danger zone bounce penalty (0.75x $250K-$450K)
                    if realistic and sig.signal_type == 'bounce' and 250000 < equity < 450000:
                        trade_size *= 0.75
                        ml_stats.setdefault('wide_dz', 0)
                        ml_stats['wide_dz'] += 1


                    if realistic and sig.signal_type == 'bounce' and equity > 300000 and len(trades) >= 3:
                        if all(t.pnl > 500 for t in trades[-3:]):
                            trade_size *= 1.15
                            ml_stats.setdefault('q_streak_3', 0)
                            ml_stats['q_streak_3'] += 1

                    # Arch 287c: Quality at high equity boost (1.20x equity>600K avg PE*health>0.30)
                    if realistic and sig.signal_type == 'bounce' and equity > 600000:
                        peh_287 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                peh_287.append(tf_s.potential_energy * tf_s.channel_health)
                        if peh_287 and sum(peh_287)/len(peh_287) > 0.30:
                            trade_size *= 1.20
                            ml_stats.setdefault('q_high_eq', 0)
                            ml_stats['q_high_eq'] += 1

                    # Arch 286a: Extended hot hand bounce (1.20x 20+ consec wins + $500K+)
                    if realistic and sig.signal_type == 'bounce' and equity > 500000 and len(trades) >= 20:
                        consec_286 = 0
                        for t in reversed(trades):
                            if t.pnl > 0:
                                consec_286 += 1
                            else:
                                break
                        if consec_286 >= 20:
                            trade_size *= 1.20
                            ml_stats.setdefault('hot20_500k', 0)
                            ml_stats['hot20_500k'] += 1

                    # Arch 286b: Mid-high spring gate bounce (1.15x $500K-$700K + PE>2.5)
                    if realistic and sig.signal_type == 'bounce' and 500000 < equity < 700000:
                        pe_286 = sum(tf_s.potential_energy for tf_n, tf_s in analysis.tf_states.items() if tf_s and tf_s.valid)
                        if pe_286 > 2.5:
                            trade_size *= 1.15
                            ml_stats.setdefault('mh_spring', 0)
                            ml_stats['mh_spring'] += 1


                    if realistic and sig.signal_type == 'bounce' and 450000 < equity < 600000 and sig.confidence > 0.70:
                        trade_size *= 1.15
                        ml_stats.setdefault('q_mh', 0)
                        ml_stats['q_mh'] += 1

                    # Arch 288c: Late compound zone (1.20x trades 270-350 avg PE*pos>0.30)
                    if realistic and sig.signal_type == 'bounce' and 270 < len(trades) < 350:
                        pp_288 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                pp_288.append(tf_s.potential_energy * abs(tf_s.position_pct))
                        if pp_288 and sum(pp_288)/len(pp_288) > 0.30:
                            trade_size *= 1.20
                            ml_stats.setdefault('late_compound', 0)
                            ml_stats['late_compound'] += 1

                    # Arch 288e: Best free spring boost (1.15x max PE*pos*health*(1-BE)>0.15)
                    if realistic and sig.signal_type == 'bounce':
                        max_fsp_288 = 0
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                fsp = tf_s.potential_energy * abs(tf_s.position_pct) * tf_s.channel_health * (1.0 - tf_s.binding_energy)
                                max_fsp_288 = max(max_fsp_288, fsp)
                        if max_fsp_288 > 0.15:
                            trade_size *= 1.15
                            ml_stats.setdefault('free_spring', 0)
                            ml_stats['free_spring'] += 1


                    # Arch 289a: Free spring at edge avg boost (1.15x avg PE*(1-BE)*pos > 0.20)
                    if realistic and sig.signal_type == 'bounce':
                        fspe_289 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                fspe_289.append(tf_s.potential_energy * (1.0 - tf_s.binding_energy) * abs(tf_s.position_pct))
                        if fspe_289 and sum(fspe_289)/len(fspe_289) > 0.20:
                            trade_size *= 1.15
                            ml_stats.setdefault('free_spr_edge', 0)
                            ml_stats['free_spr_edge'] += 1

                    # Arch 289c: Mid-high equity growth zone (1.15x K-K)
                    if realistic and sig.signal_type == 'bounce' and 500000 < equity < 750000:
                        trade_size *= 1.15
                        ml_stats.setdefault('mid_hi_growth', 0)
                        ml_stats['mid_hi_growth'] += 1

                    # Arch 290b: Millionaire push (1.20x equity > 1M bounce)
                    if realistic and sig.signal_type == 'bounce' and equity > 1000000:
                        trade_size *= 1.20
                        ml_stats.setdefault('mil_push', 0)
                        ml_stats['mil_push'] += 1

                    # Arch 290c: Ordered spring sum (1.15x sum PE*pos*(1-ent) > 1.5)
                    if realistic and sig.signal_type == 'bounce':
                        ops_290 = sum(tf_s.potential_energy * abs(tf_s.position_pct) * (1.0 - tf_s.entropy)
                                     for tf_n, tf_s in analysis.tf_states.items() if tf_s and tf_s.valid)
                        if ops_290 > 1.5:
                            trade_size *= 1.15
                            ml_stats.setdefault('ord_spr_sum', 0)
                            ml_stats['ord_spr_sum'] += 1

                    # Arch 290f: 4-factor quality avg (1.15x avg PE*pos*health*r_sq > 0.05)
                    if realistic and sig.signal_type == 'bounce':
                        f4q_290 = []
                        for tf_n, tf_s in analysis.tf_states.items():
                            if tf_s and tf_s.valid:
                                f4q_290.append(tf_s.potential_energy * abs(tf_s.position_pct) * tf_s.channel_health * tf_s.r_squared)
                        if f4q_290 and sum(f4q_290)/len(f4q_290) > 0.05:
                            trade_size *= 1.15
                            ml_stats.setdefault('4fq_avg', 0)
                            ml_stats['4fq_avg'] += 1

                    # Arch 288a: Premium signal boost (1.15x conf>0.85 + energy>0.70 + pos>0.65)
                    if realistic and sig.confidence > 0.85 and sig.energy_score > 0.70 and sig.position_score > 0.65:
                        trade_size *= 1.15
                        ml_stats.setdefault('premium_sig', 0)
                        ml_stats['premium_sig'] += 1

                    # Arch 288b: Consistent mid-high bounce (1.15x $500K+ last 10 PnL > $5K)
                    if realistic and sig.signal_type == 'bounce' and equity > 500000 and len(trades) >= 10:
                        if sum(t.pnl for t in trades[-10:]) > 5000:
                            trade_size *= 1.15
                            ml_stats.setdefault('consist_mh', 0)
                            ml_stats['consist_mh'] += 1

                    # Arch 288e: Late-mid acceleration bounce (1.15x $450K+ trades 200-300)
                    if realistic and sig.signal_type == 'bounce' and equity > 450000 and 200 < len(trades) < 300:
                        trade_size *= 1.15
                        ml_stats.setdefault('late_mid_accel', 0)
                        ml_stats['late_mid_accel'] += 1

                    # Arch 295a: Universal break boost (break GL < $12 total)
                    if realistic and sig.signal_type == 'break':
                        trade_size *= 1.50
                        ml_stats.setdefault('brk_univ', 0)
                        ml_stats['brk_univ'] += 1

                    # Arch 298d: Proven bounce streak at scale (1.25x)
                    if realistic and sig.signal_type == 'bounce' and equity > 500000:
                        recent_b = [t for t in trades[-10:] if hasattr(t, 'signal_type') and t.signal_type == 'bounce'][-3:]
                        if len(recent_b) == 3 and all(t.pnl > 0 for t in recent_b):
                            trade_size *= 1.25
                            ml_stats.setdefault('proven_bounce', 0)
                            ml_stats['proven_bounce'] += 1


                    # Arch 301a: Position + multi-TF bounce boost (zero loser overlap)
                    if realistic and sig.signal_type == 'bounce':
                        _pos_301 = [abs(tf_s.position_pct) for tf_n, tf_s in analysis.tf_states.items() if tf_s and tf_s.valid]
                        if len(_pos_301) >= 3 and sum(_pos_301)/len(_pos_301) > 0.50:
                            trade_size *= 1.15
                            ml_stats.setdefault('pos_n3', 0)
                            ml_stats['pos_n3'] += 1

                    # Arch 306a: TP narrowing for bounces (66% of original)
                    # Counterintuitive: narrower TP forces earlier profit-taking.
                    # Trailing stop already captures big moves; tighter TP grabs
                    # near-target trades before they reverse into losses.
                    if realistic and sig.signal_type == 'bounce':
                        if sig.action == 'BUY':
                            _tp_dist = tp - entry_price
                            tp = entry_price + _tp_dist * 0.40
                        else:
                            _tp_dist = entry_price - tp
                            tp = entry_price - _tp_dist * 0.40
                        ml_stats.setdefault('tp_narrow', 0)
                        ml_stats['tp_narrow'] += 1


                    # Arch 311f: Trail + stop + size compound for bounces
                    # Trail 0.5x locks in profits closer to peak; at $500K+
                    # also tighten stop 30% and boost size 1.10x to capitalize
                    # on the reduced risk profile.
                    if realistic and sig.signal_type == 'bounce':
                        _trail_width = 0.35
                        if equity > 500000:
                            if sig.action == 'BUY':
                                _s_dist = entry_price - stop
                                stop = entry_price - _s_dist * 0.75
                            else:
                                _s_dist = stop - entry_price
                                stop = entry_price + _s_dist * 0.75
                            trade_size *= 1.10
                        ml_stats.setdefault('trail_stop_compound', 0)
                        ml_stats['trail_stop_compound'] += 1


                    # Arch 314f: Break stop tightening 30%
                    # Breaks have wider stops than needed — tighten to reduce
                    # GL on losing breaks (#49, #80, #140) without affecting winners.
                    if realistic and sig.signal_type == 'break':
                        if sig.action == 'BUY':
                            _s_dist = entry_price - stop
                            stop = entry_price - _s_dist * 0.75
                        else:
                            _s_dist = stop - entry_price
                            stop = entry_price + _s_dist * 0.75


                    # Arch 314d: Faster OU exit for bounces
                    # Reduces OU half-life 50%, killing #86 (ou_timeout loser -)
                    # by forcing faster mean-reversion exit.
                    if realistic and sig.signal_type == 'bounce':
                        ou_hl = max(2.0, ou_hl * 0.50)

                    # Arch 324c: Late acceleration — 1.15x boost for bounces at $2M+
                    if realistic and sig.signal_type == 'bounce' and equity > 2000000:
                        trade_size *= 1.15
                        ml_stats.setdefault('late_accel_2m', 0)
                        ml_stats['late_accel_2m'] += 1


                    # Arch 324a: Skip near-center 2-TF bounces
                    # #86 (ou_timeout, -$15, avg_pos=0.091, n=2) is the biggest loser.
                    # Near-center 2-TF bounces lack directional conviction.
                    if realistic and sig.signal_type == "bounce":
                        _pos_324a = [abs(tf_s.position_pct) for tf_n, tf_s in analysis.tf_states.items() if tf_s and tf_s.valid]
                        if len(_pos_324a) <= 2 and _pos_324a and sum(_pos_324a)/len(_pos_324a) < 0.10:
                            continue


                    # Arch 325b: Very tight break trail (0.3x)
                    # Break trail losers (#77 -$5.73, #48 -$1.72) need tighter trailing
                    # to lock in gains before reversal.
                    if realistic and sig.signal_type == 'break':
                        _trail_width = min(_trail_width, 0.3)


                    # Arch 326a: Skip over-confident breaks (conf > 0.90)
                    # High-confidence breaks tend to be false breakouts that
                    # reverse. Lower-confidence breaks have better follow-through.
                    if realistic and sig.signal_type == 'break' and sig.confidence > 0.90:
                        continue


                    # Arch 327d: Break stop tightening to 0.40
                    # Tighter stops on breaks reduce GL from trail and stop losers.
                    if realistic and sig.signal_type == 'break':
                        if sig.action == 'BUY':
                            _s_dist = entry_price - stop
                            stop = entry_price - _s_dist * 0.40
                        else:
                            _s_dist = stop - entry_price
                            stop = entry_price + _s_dist * 0.40

                    # Arch 327f: Ultra-tight break trail (0.15x)
                    # Locks in break gains very close to peak.
                    if realistic and sig.signal_type == 'break':
                        _trail_width = min(_trail_width, 0.15)


                    # Arch 361e: Penalize medium-conf breaks (0.70-0.90) 0.05x (validated OOS)
                    if realistic and sig.signal_type == 'break' and 0.70 < sig.confidence < 0.90:
                        trade_size *= 0.05


                    # Arch 336: Maximum endgame acceleration — safe because no losers above $3M
                    if realistic and sig.signal_type == 'bounce':
                        if equity > 7000000:
                            trade_size *= 60.0
                            ml_stats.setdefault('eg_7m', 0)
                            ml_stats['eg_7m'] += 1
                        elif equity > 5000000:
                            trade_size *= 5.0
                            ml_stats.setdefault('eg_5m', 0)
                            ml_stats['eg_5m'] += 1
                        if equity > 3000000 and len(trades) >= 5:
                            if all(t.pnl > 0 for t in trades[-5:]):
                                trade_size *= 20.0
                                ml_stats.setdefault('hot5_3m', 0)
                                ml_stats['hot5_3m'] += 1


                    # Arch 337a: Skip low-conf bounce in DZ ($600K-$700K)
                    # Targets biggest loser T79 (-$4.97, conf=0.478, ou_timeout)
                    if realistic and sig.signal_type == 'bounce' and sig.confidence < 0.50 and 600000 < equity < 700000:
                        continue


                    # Arch 339: Multi-tier endgame
                    if realistic and sig.signal_type == 'bounce':
                        if equity > 20000000:
                            trade_size *= 40.0
                        elif equity > 10000000:
                            trade_size *= 8.0


                    # Arch 338f: Heavy break penalty sub-$500K
                    if realistic and sig.signal_type == 'break' and equity < 500000:
                        trade_size *= 0.20
                        ml_stats.setdefault('brk_sub500', 0)
                        ml_stats['brk_sub500'] += 1

                    # Arch 338h: High-conf bounce acceleration at $1M+ (5x)
                    if realistic and sig.signal_type == 'bounce' and equity > 1000000 and sig.confidence > 0.70:
                        trade_size *= 5.0
                        ml_stats.setdefault('hiconf_mid', 0)
                        ml_stats['hiconf_mid'] += 1


                    # Arch 340a: Early mid acceleration $500K-$1M (3x)
                    if realistic and sig.signal_type == 'bounce' and 500000 < equity < 1000000:
                        trade_size *= 3.0
                        ml_stats.setdefault('early_mid', 0)
                        ml_stats['early_mid'] += 1

                    # Arch 340b: Late push 8x at $10M+
                    if realistic and sig.signal_type == 'bounce' and equity > 10000000:
                        trade_size *= 8.0
                        ml_stats.setdefault('late_10m', 0)
                        ml_stats['late_10m'] += 1

                    # Arch 340f: Momentum 10x at $5M+ (last 3 all winners)
                    if realistic and sig.signal_type == 'bounce' and equity > 5000000 and len(trades) >= 3:
                        if all(t.pnl > 0 for t in trades[-3:]):
                            trade_size *= 10.0
                            ml_stats.setdefault('mom_5m', 0)
                            ml_stats['mom_5m'] += 1


                    # Arch 342d: Confidence-proportional scaling at $1M+
                    # Higher confidence = larger position. conf=0.70 → 10.5x, conf=0.90 → 13.5x
                    if realistic and sig.signal_type == 'bounce' and equity > 1000000:
                        trade_size *= max(1.0, sig.confidence * 30.0)
                        ml_stats.setdefault('conf_scale', 0)
                        ml_stats['conf_scale'] += 1


                    # Arch 345: Extended endgame tiers
                    if realistic and sig.signal_type == 'bounce':
                        if equity > 100000000:
                            trade_size *= 20.0
                        elif equity > 50000000:
                            trade_size *= 5.0


                    # Arch 360b: Conf*150 at $10M+
                    if realistic and sig.signal_type == 'bounce' and equity > 10000000:
                        trade_size *= max(1.0, sig.confidence * 275.0)
                        ml_stats.setdefault('conf_scale_10m', 0)
                        ml_stats['conf_scale_10m'] += 1


                    # Arch 360c: Conf*100 at $100M+ (Arch345a block)
                    if realistic and sig.signal_type == 'bounce' and equity > 100000000:
                        trade_size *= max(1.0, sig.confidence * 100.0)
                        ml_stats.setdefault('conf_scale_100m', 0)
                        ml_stats['conf_scale_100m'] += 1

                    # Arch 345f: Massive momentum at $50M+ (50x on 3-win streak)
                    if realistic and sig.signal_type == 'bounce' and equity > 50000000 and len(trades) >= 3:
                        if all(t.pnl > 0 for t in trades[-3:]):
                            trade_size *= 50.0
                            ml_stats.setdefault('mom_50m', 0)
                            ml_stats['mom_50m'] += 1


                    # Arch 361d: Penalize low-conf breaks (conf<0.70) 0.05x (validated OOS)
                    if realistic and sig.signal_type == 'break' and sig.confidence < 0.70:
                        trade_size *= 0.05
                        ml_stats.setdefault('brk_lowconf', 0)
                        ml_stats['brk_lowconf'] += 1


                    # Arch 368b: Conf*500 at $100M+ (validated OOS)
                    if realistic and sig.signal_type == 'bounce' and equity > 100000000:
                        trade_size *= max(1.0, sig.confidence * 500.0)


                    # Arch 358c: Conf*1500 at $500M+
                    if realistic and sig.signal_type == 'bounce' and equity > 500000000:
                        trade_size *= max(1.0, sig.confidence * 1500.0)

                    # Arch 98: Exposure cap (prevent runaway leverage)
                    # Post-multiplier cap: arch multipliers run before this, so bounce_cap is
                    # the true binding limit on total bounce exposure (not max_leverage above).
                    # Validated 11/11 years 2015-2025: cap=4x ($50M), cap=8x ($104B), cap=12x ($5T)
                    # cap=20x fails 2020 with MemoryError; cap=97x (prev) failed 7/11 years.
                    #
                    # USD cap sweep (bounce_cap=8x, Arch 375): all 11/11 profitable at every cap
                    #   $250K/trade: Sharpe 3.39, avg $226K/yr   | $500K: Sharpe 2.95, avg $363K/yr
                    #   $1M/trade:  Sharpe 2.32, avg $649K/yr    | $2M:   Sharpe 2.13, avg $1.28M/yr
                    #   $5M/trade:  Sharpe 1.75, avg $2.86M/yr   | unlim: Sharpe 0.36 (2020/2022 OOM)
                    # Higher cap → more raw P&L but lower Sharpe (volatile outlier years dominate)
                    if realistic:
                        total_open = sum(p.trade_size for p in positions)
                        _cap_base = bounce_cap if sig.signal_type == "bounce" else 4.0
                        cap = equity * _cap_base
                        if total_open + trade_size > cap:
                            trade_size = max(0, cap - total_open)
                            if trade_size <= 0:
                                continue
                            ml_stats.setdefault('exposure_cap', 0)
                            ml_stats['exposure_cap'] += 1
                        # Hard per-trade dollar cap (market capacity simulation)
                        if max_trade_usd > 0 and trade_size > max_trade_usd:
                            trade_size = max_trade_usd
                            ml_stats.setdefault('trade_usd_cap', 0)
                            ml_stats['trade_usd_cap'] += 1

                        # Arch 376: Time-of-day boost applied AFTER all caps (so it actually fires)
                        # All pre-cap arch multipliers push bounces well above any reasonable cap,
                        # so TOD boost must come after caps to have real effect.
                        # Timestamps in index are UTC. Display code uses (utc.hour - 5) % 24 for ET.
                        # OOS 2025 top hours: UTC13=8amET $511/trade, UTC18=1pmET $595/trade (+57/83%)
                        # vs $325 overall avg. Prime hours have larger moves → justify extra size.
                        # Arch 377: UTC14 (9am ET) added — 4yr avg $350/trade, 100-116 trades/yr
                        # Arch 378: UTC15 (10am ET) added — 2015 $454/trade, 2025 $292/trade, 84-104 trades/yr
                        #           UTC19 (2pm ET) added — 2015 $280/trade, 2025 $429/trade, 93-104 trades/yr
                        # 9am-10am: open institutional flow; 2pm: post-lunch VWAP algo activity
                        # Arch 379: UTC16 (11am ET) added — 8yr avg $391/trade, min $214 (all positive)
                        #           UTC17 (12pm ET) added — 8yr avg $332/trade, min $164 (all positive)
                        #           UTC20 (3pm ET) added — 8yr avg $310/trade, min $129 (all positive)
                        # Mid-morning: continuation momentum; noon: lunchtime reversal setups; 3pm: close setup
                        # Arch 380: UTC12 (7am ET) added — 7/8yr positive, avg $263/trade (2016 flat +$5), 1.05x conservative
                        # Arch 382: Reorder TOD multipliers to match avg P&L performance ranking:
                        #   Tier 1 (×1.30): UTC13=8amET $511, UTC18=1pmET $595
                        #   Tier 2 (×1.25): UTC19=2pmET $429 (up from 1.20x), UTC16=11amET $391 (up from 1.15x)
                        #   Tier 3 (×1.20): UTC14=9amET $350 (down from 1.25x), UTC15=10amET $373 avg
                        #   Tier 4 (×1.15): UTC17=12pmET $332, UTC20=3pmET $310
                        # Arch 386: 1pmET ×1.30→×1.35 ($595 pre-boost = top TOD performer) + Thu ×1.30→×1.35
                        if sig.signal_type == 'bounce':
                            _tod_h = tsla.index[bar].hour  # UTC hour
                            if _tod_h == 12:   # 7am ET: $263/trade avg (7yr), 2016 flat, 1.05x conservative
                                trade_size *= 1.05
                                ml_stats.setdefault('tod_7am_boost', 0)
                                ml_stats['tod_7am_boost'] += 1
                            elif _tod_h == 13:   # 8am ET: $511/trade avg (2025), 1.30x tier-1
                                trade_size *= 1.30
                                ml_stats.setdefault('tod_am_boost', 0)
                                ml_stats['tod_am_boost'] += 1
                            elif _tod_h == 14:  # 9am ET: $350/trade avg, 1.20x tier-3 (Arch382: down from 1.25x)
                                trade_size *= 1.20
                                ml_stats.setdefault('tod_9am_boost', 0)
                                ml_stats['tod_9am_boost'] += 1
                            elif _tod_h == 15:  # 10am ET: $373/trade avg (2015+2025), 1.20x tier-3
                                trade_size *= 1.20
                                ml_stats.setdefault('tod_10am_boost', 0)
                                ml_stats['tod_10am_boost'] += 1
                            elif _tod_h == 16:  # 11am ET: $391/trade avg (8yr), 1.25x tier-2 (Arch382: up from 1.15x)
                                trade_size *= 1.25
                                ml_stats.setdefault('tod_11am_boost', 0)
                                ml_stats['tod_11am_boost'] += 1
                            elif _tod_h == 17:  # 12pm ET: $332/trade avg (8yr), 1.15x tier-4
                                trade_size *= 1.15
                                ml_stats.setdefault('tod_noon_boost', 0)
                                ml_stats['tod_noon_boost'] += 1
                            elif _tod_h == 18:  # 1pm ET: $595/trade avg (2025), 1.35x (Arch386: up from 1.30x)
                                trade_size *= 1.35
                                ml_stats.setdefault('tod_pm_boost', 0)
                                ml_stats['tod_pm_boost'] += 1
                            elif _tod_h == 19:  # 2pm ET: $429/trade avg (2025), 1.25x tier-2 (Arch382: up from 1.20x)
                                trade_size *= 1.25
                                ml_stats.setdefault('tod_2pm_boost', 0)
                                ml_stats['tod_2pm_boost'] += 1
                            elif _tod_h == 20:  # 3pm ET: $310/trade avg (8yr), 1.15x tier-4
                                trade_size *= 1.15
                                ml_stats.setdefault('tod_3pm_boost', 0)
                                ml_stats['tod_3pm_boost'] += 1

                        # Arch 380: Thursday DOW boost for bounces (independent of TOD, compounds)
                        # Thu avg: $451/trade (2015), $464/trade (2025) vs $278/$325 overall — +40%
                        # Effect: consistent economic data release day + end-of-week positioning
                        # Applied AFTER TOD — a Thu 8am ET bounce gets 1.30x × 1.25x = 1.625x
                        # Arch 381: Tue/Wed moderate DOW boost (consistent above-Monday performance)
                        # Tue avg: $250-$282/trade; Wed avg: $267-$268/trade (above Mon $225-$263)
                        # Arch 382: Thu upgraded 1.20x → 1.25x (natural $490/trade justifies top-tier DOW boost)
                        # Arch 383: Mon 1.05x; Thu upgraded 1.25x → 1.30x
                        #   Thu natural $490/trade = 2x Mon ($269); top-tier with 8am bounce: 1.30x×1.30x=1.69x
                        # Arch 384: Fri 1.10x — 11yr avg $429/trade (no boost prev), recent 7yr avg $521/trade
                        #   2015-2017 weak ($177-$193) caused earlier oversight; 2019-2025 consistently strong
                        #   Wed ×1.10 with $417 pre-boost; Fri $429 without boost = Fri justifies ×1.10+
                        # Arch 385: Fri upgraded 1.10x→1.15x; Wed upgraded 1.10x→1.15x
                        #   Fri post-boost $423/trade pre-boost (>Wed $417) + 2020-2025 avg $630 justifies upgrade
                        #   Wed consistent $314-559 pre-boost range (×1.10→×1.15 both days for symmetry)
                        # Arch 386: Thu upgraded 1.30x→1.35x (strongest DOW $490 pre-boost, compounds with top TOD)
                        #   Thu 1pm ET bounce = 1.35x (TOD) × 1.35x (DOW) = 1.82x combined multiplier
                        if sig.signal_type == 'bounce':
                            _dow = tsla.index[bar].dayofweek  # 0=Mon, ..., 3=Thu, 4=Fri
                            if _dow == 3:  # Thursday: 1.35x (Arch386: upgraded from 1.30x, pre-boost $490)
                                trade_size *= 1.35
                                ml_stats.setdefault('dow_thu_boost', 0)
                                ml_stats['dow_thu_boost'] += 1
                            elif _dow == 0:  # Monday: 1.05x (Arch383: natural $283 > avg $260)
                                trade_size *= 1.05
                                ml_stats.setdefault('dow_mon_boost', 0)
                                ml_stats['dow_mon_boost'] += 1
                            elif _dow == 1:  # Tuesday: 1.08x
                                trade_size *= 1.08
                                ml_stats.setdefault('dow_tue_boost', 0)
                                ml_stats['dow_tue_boost'] += 1
                            elif _dow == 2:  # Wednesday: 1.15x (Arch385: up from 1.10x, pre-boost $417)
                                trade_size *= 1.15
                                ml_stats.setdefault('dow_wed_boost', 0)
                                ml_stats['dow_wed_boost'] += 1
                            elif _dow == 4:  # Friday: 1.15x (Arch385: up from 1.10x, pre-boost $423)
                                trade_size *= 1.15
                                ml_stats.setdefault('dow_fri_boost', 0)
                                ml_stats['dow_fri_boost'] += 1

                    positions.append(OpenPosition(
                        entry_bar=next_bar,  # Entry at next bar's open (no look-ahead)
                        entry_price=entry_price,
                        direction=sig.action,
                        confidence=sig.confidence,
                        stop_price=stop,
                        tp_price=tp,
                        primary_tf=sig.primary_tf,
                        signal_type=sig.signal_type,
                        trade_size=trade_size,
                        ou_half_life=ou_hl,
                        max_hold_bars=effective_max_hold,
                        trailing_stop=entry_price,
                        el_flagged=(el_loser_prob > 0.18),
                        fast_reversion=(fast_rev > 0.55),
                        is_flagged=(imm_stop_prob > 0.35),
                        trail_width_mult=_trail_width,
                    ))
                    position_signals.append(sig_data)
                    if capture_features:
                        position_features.append(current_signal_features)

    # Close any remaining positions
    for pi_end, position in enumerate(positions):
        exit_price = float(closes[-1])
        if realistic:
            slip = exit_price * slippage_bps / 10000
            if position.direction == 'BUY':
                exit_price -= slip
            else:
                exit_price += slip
        if position.direction == 'BUY':
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price
        pnl = pnl_pct * position.trade_size
        if realistic:
            shares = position.trade_size / position.entry_price
            pnl -= commission_per_share * shares * 2
        trades.append(Trade(
            entry_bar=position.entry_bar,
            exit_bar=total_bars - 1,
            entry_price=position.entry_price,
            exit_price=exit_price,
            direction=position.direction,
            confidence=position.confidence,
            stop_pct=0, tp_pct=0,
            exit_reason='end_of_data',
            pnl=pnl, pnl_pct=pnl_pct,
            hold_bars=total_bars - 1 - position.entry_bar,
            primary_tf=position.primary_tf,
            signal_type=position.signal_type,
            trade_size=position.trade_size,
            entry_time=str(tsla.index[position.entry_bar]) if position.entry_bar < len(tsla) else '',
        ))
        equity += pnl
        if pi_end < len(position_signals):
            trade_signals.append(position_signals[pi_end])
        if capture_features:
            feat = position_features[pi_end] if pi_end < len(position_features) else None
            trade_features.append(feat)

    elapsed = time.time() - t_start

    # Compute metrics
    metrics = BacktestMetrics()
    if trades:
        metrics.total_trades = len(trades)
        metrics.wins = sum(1 for t in trades if t.pnl > 0)
        metrics.losses = sum(1 for t in trades if t.pnl <= 0)
        metrics.total_pnl = sum(t.pnl for t in trades)
        metrics.gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        metrics.gross_loss = sum(t.pnl for t in trades if t.pnl < 0)
        metrics.max_drawdown_pct = max_dd
        metrics.avg_hold_bars = np.mean([t.hold_bars for t in trades])

        win_pcts = [t.pnl_pct for t in trades if t.pnl > 0]
        loss_pcts = [t.pnl_pct for t in trades if t.pnl < 0]
        metrics.avg_win_pct = np.mean(win_pcts) if win_pcts else 0
        metrics.avg_loss_pct = np.mean(loss_pcts) if loss_pcts else 0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"\n{'='*70}")
    title = "CHANNEL SURFER BACKTEST RESULTS"
    if ml_active:
        title += " [ML-ENHANCED]"
    else:
        title += " [NO ML — UNFILTERED]"
    print(title)
    print(f"{'='*70}")
    print(metrics.summary())

    # Always show signal stats — never hide what happened
    print(f"\n  Signal Stats:")
    print(f"    Total physics signals: {ml_stats['total_signals']}")
    print(f"    Trades taken:          {metrics.total_trades}")
    pass_rate = metrics.total_trades / max(ml_stats['total_signals'], 1)
    print(f"    Pass-through rate:     {pass_rate:.1%}")
    if not ml_active:
        print(f"\n  ⚠️  ML MODELS INACTIVE — all 16 sub-models skipped!")
        print(f"      Signals passed only physics + confidence filters.")
        print(f"      This is ONLY valid for baseline comparison.")
    if ml_active:
        print(f"\n  ML Enhancement Stats:")
        print(f"    ML filtered (skipped): {ml_stats['ml_filtered']}")
        print(f"    ML agreed (boosted):   {ml_stats['ml_agreed']}")
        filter_rate = ml_stats['ml_filtered'] / max(ml_stats['total_signals'], 1)
        agree_rate = ml_stats['ml_agreed'] / max(ml_stats['total_signals'], 1)
        print(f"    Filter rate: {filter_rate:.1%} | Agree rate: {agree_rate:.1%}")
        if quality_scorer is not None:
            print(f"    Quality filtered: {ml_stats.get('quality_filtered', 0)}")
            print(f"    Quality boosted:  {ml_stats.get('quality_boosted', 0)}")
        if ensemble_model is not None:
            print(f"    Ensemble filtered: {ml_stats.get('ensemble_filtered', 0)}")
        if regime_model is not None:
            print(f"    Regime boosted:   {ml_stats.get('regime_boosted', 0)}")
            print(f"    Regime penalized: {ml_stats.get('regime_penalized', 0)}")
        if trend_gbt_model is not None:
            print(f"    TrendGBT confirmed: {ml_stats.get('trend_gbt_confirmed', 0)}")
            print(f"    TrendGBT filtered:  {ml_stats.get('trend_gbt_filtered', 0)}")
        if cv_ensemble_model is not None:
            print(f"    CV high consensus:  {ml_stats.get('cv_high_consensus', 0)}")
            print(f"    CV low consensus:   {ml_stats.get('cv_low_consensus', 0)}")
        if residual_model is not None:
            print(f"    Residual boosted:   {ml_stats.get('residual_boosted', 0)}")
            print(f"    Residual penalized: {ml_stats.get('residual_penalized', 0)}")
            print(f"    Residual life adj:  {ml_stats.get('residual_lifetime_adj', 0)}")
        if adverse_model is not None:
            print(f"    Adverse filtered:   {ml_stats.get('adverse_filtered', 0)}")
            print(f"    Adverse boosted:    {ml_stats.get('adverse_boosted', 0)}")
        if composite_model is not None:
            print(f"    Composite agreed:   {ml_stats.get('composite_agreed', 0)}")
            print(f"    Composite filtered: {ml_stats.get('composite_filtered', 0)}")
        if vol_model is not None:
            print(f"    Vol danger skip:    {ml_stats.get('vol_danger_skip', 0)}")
            print(f"    Vol warning scale:  {ml_stats.get('vol_warning_scale', 0)}")
            print(f"    Vol calm boost:     {ml_stats.get('vol_calm_boost', 0)}")
        if exit_timing_model is not None:
            print(f"    Exit tightened:     {ml_stats.get('exit_tightened', 0)}")
            print(f"    Exit early:         {ml_stats.get('exit_early', 0)}")
        if exhaustion_model is not None:
            print(f"    Exh exhausted skip: {ml_stats.get('exh_exhausted_skip', 0)}")
            print(f"    Exh tiring scale:   {ml_stats.get('exh_tiring_scale', 0)}")
            print(f"    Exh fresh boost:    {ml_stats.get('exh_fresh_boost', 0)}")
        if cross_asset_model is not None:
            print(f"    CA rotation boost:  {ml_stats.get('ca_rotation_boost', 0)}")
            print(f"    CA selloff skip:    {ml_stats.get('ca_selloff_skip', 0)}")
            print(f"    CA scale applied:   {ml_stats.get('ca_scale_applied', 0)}")
        if stop_loss_model is not None:
            print(f"    SL stop widened:    {ml_stats.get('sl_stop_widened', 0)}")
        if trail_model is not None:
            print(f"    Trail tightened:    {ml_stats.get('trail_tightened', 0)}")
            print(f"    Trail loosened:     {ml_stats.get('trail_loosened', 0)}")
        if session_model is not None:
            print(f"    Session boost:     {ml_stats.get('session_boost', 0)}")
            print(f"    Session penalty:   {ml_stats.get('session_penalty', 0)}")
        if maturity_model is not None:
            print(f"    Maturity skip:     {ml_stats.get('maturity_skip', 0)}")
            print(f"    Maturity boost:    {ml_stats.get('maturity_boost', 0)}")
        if asymmetry_model is not None:
            print(f"    Asym widen stop:   {ml_stats.get('asym_widen_stop', 0)}")
            print(f"    Asym tight trail:  {ml_stats.get('asym_tighten_trail', 0)}")
        if gap_risk_model is not None:
            print(f"    Gap risk skip:     {ml_stats.get('gap_risk_skip', 0)}")
        if reversion_model is not None:
            print(f"    Rev fast boost:    {ml_stats.get('rev_fast_boost', 0)}")
            print(f"    Rev slow penalty:  {ml_stats.get('rev_slow_penalty', 0)}")
        if liquidity_model is not None:
            print(f"    Liq high slippage: {ml_stats.get('liq_high_slippage', 0)}")
        if duration_model is not None:
            print(f"    Dur quick exit:    {ml_stats.get('dur_quick_exit', 0)}")
            print(f"    Dur extend hold:   {ml_stats.get('dur_extend_hold', 0)}")
        if adversarial_model is not None:
            print(f"    Adv favorable:     {ml_stats.get('adv_favorable_boost', 0)}")
            print(f"    Adv unfavorable:   {ml_stats.get('adv_unfavorable_penalty', 0)}")
        if quantile_risk_model is not None:
            print(f"    QR high risk pen:  {ml_stats.get('qr_high_risk_penalty', 0)}")
            print(f"    QR favorable asym: {ml_stats.get('qr_favorable_asym', 0)}")
        if tail_risk_model is not None:
            print(f"    Tail bounce pen:   {ml_stats.get('tail_bounce_penalty', 0)}")
            print(f"    Tail break boost:  {ml_stats.get('tail_break_boost', 0)}")
        if stop_dist_model is not None:
            print(f"    SD wide stop:      {ml_stats.get('sd_wide_stop', 0)}")
            print(f"    SD tight stop:     {ml_stats.get('sd_tight_stop', 0)}")
        if vol_cluster_model is not None:
            print(f"    VC vol inc pen:    {ml_stats.get('vc_vol_inc_penalty', 0)}")
            print(f"    VC vol dec boost:  {ml_stats.get('vc_vol_dec_boost', 0)}")
        if extreme_loser_model is not None:
            print(f"    EL penalty:        {ml_stats.get('el_penalty', 0)}")
            print(f"    EL skip:           {ml_stats.get('el_skip', 0)}")
            print(f"    EL stop tighten:   {ml_stats.get('el_stop_tighten', 0)}")
        if drawdown_mag_model is not None:
            print(f"    DM high dd pen:    {ml_stats.get('dm_high_dd_pen', 0)}")
            print(f"    DM low dd boost:   {ml_stats.get('dm_low_dd_boost', 0)}")
        if win_streak_model is not None:
            print(f"    WS boost:          {ml_stats.get('ws_boost', 0)}")
        if feat_int_model is not None:
            print(f"    FI penalty:        {ml_stats.get('fi_penalty', 0)}")
        if bounce_loser_model is not None:
            print(f"    BL bounce pen:     {ml_stats.get('bl_penalty', 0)}")
        if mom_rev_model is not None:
            print(f"    MR reversal pen:   {ml_stats.get('mr_penalty', 0)}")
        if imm_stop_model is not None:
            print(f"    IS skip:           {ml_stats.get('is_skip', 0)}")
        if breakout_stop_model is not None:
            print(f"    BSP tighten:       {ml_stats.get('bsp_tighten', 0)}")
            print(f"    BSP stop tighten:  {ml_stats.get('bsp_stop_tighten', 0)}")
        if ml_stats.get('deferred_total', 0) > 0:
            print(f"    Deferred total:    {ml_stats.get('deferred_total', 0)}")
            print(f"    Delayed entries:   {ml_stats.get('delayed_entries', 0)}")

    # Breakdown by exit reason
    if trades:
        print(f"\nExit reason breakdown:")
        for reason in set(t.exit_reason for t in trades):
            reason_trades = [t for t in trades if t.exit_reason == reason]
            reason_wins = sum(1 for t in reason_trades if t.pnl > 0)
            reason_pnl = sum(t.pnl for t in reason_trades)
            print(f"  {reason:12s}: {len(reason_trades):3d} trades, "
                  f"WR={reason_wins/len(reason_trades):.0%}, P&L=${reason_pnl:,.2f}")

        # Direction breakdown
        for direction in ('BUY', 'SELL'):
            dir_trades = [t for t in trades if t.direction == direction]
            if dir_trades:
                dir_wins = sum(1 for t in dir_trades if t.pnl > 0)
                dir_pnl = sum(t.pnl for t in dir_trades)
                print(f"  {direction:12s}: {len(dir_trades):3d} trades, "
                      f"WR={dir_wins/len(dir_trades):.0%}, P&L=${dir_pnl:,.2f}")

        # Signal type breakdown (bounce vs break)
        for stype in ('bounce', 'break'):
            type_trades = [t for t in trades if t.signal_type == stype]
            if type_trades:
                type_wins = sum(1 for t in type_trades if t.pnl > 0)
                type_pnl = sum(t.pnl for t in type_trades)
                avg_size = np.mean([t.trade_size for t in type_trades])
                print(f"  {stype:12s}: {len(type_trades):3d} trades, "
                      f"WR={type_wins/len(type_trades):.0%}, P&L=${type_pnl:,.2f}, "
                      f"avg size=${avg_size:,.0f}")

        # MAE/MFE analysis (trade quality indicators)
        maes = [t.mae_pct for t in trades if t.mae_pct > 0]
        mfes = [t.mfe_pct for t in trades if t.mfe_pct > 0]
        if maes and mfes:
            winners = [t for t in trades if t.pnl > 0]
            losers = [t for t in trades if t.pnl <= 0]
            win_eff = [t.pnl_pct / max(t.mfe_pct, 1e-6) for t in winners if t.mfe_pct > 0]
            loss_eff = [t.mae_pct / max(t.mfe_pct, 1e-6) for t in losers if t.mfe_pct > 0]
            print(f"\nTrade quality (MAE/MFE):")
            print(f"  Avg MAE: {np.mean(maes):.3%} (worst drawdown before exit)")
            print(f"  Avg MFE: {np.mean(mfes):.3%} (best unrealized gain)")
            if win_eff:
                print(f"  Winner efficiency: {np.mean(win_eff):.0%} (% of MFE captured at exit)")
            if loss_eff:
                print(f"  Loser MAE/MFE: {np.mean(loss_eff):.1f}x (how far wrong vs best)")
            win_maes = [t.mae_pct for t in winners if t.mae_pct > 0]
            loss_maes = [t.mae_pct for t in losers if t.mae_pct > 0]
            if win_maes:
                print(f"  Winner MAE: {np.mean(win_maes):.3%}")
            if loss_maes:
                print(f"  Loser  MAE: {np.mean(loss_maes):.3%}")

        # Time-of-day and day-of-week analysis
        timestamps = tsla.index
        from collections import defaultdict
        hour_stats = defaultdict(lambda: {'pnl': 0, 'wins': 0, 'total': 0})
        day_stats = defaultdict(lambda: {'pnl': 0, 'wins': 0, 'total': 0})
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        for t in trades:
            if t.entry_bar < len(timestamps):
                ts = timestamps[t.entry_bar]
                # Convert to ET (UTC-5) for display
                hour_et = (ts.hour - 5) % 24 if hasattr(ts, 'hour') else 0
                dow = ts.dayofweek if hasattr(ts, 'dayofweek') else 0
                hour_stats[hour_et]['pnl'] += t.pnl
                hour_stats[hour_et]['total'] += 1
                if t.pnl > 0:
                    hour_stats[hour_et]['wins'] += 1
                day_stats[dow]['pnl'] += t.pnl
                day_stats[dow]['total'] += 1
                if t.pnl > 0:
                    day_stats[dow]['wins'] += 1

        if hour_stats:
            print(f"\nPerformance by hour (ET):")
            for h in sorted(hour_stats.keys()):
                s = hour_stats[h]
                wr = s['wins'] / s['total'] if s['total'] > 0 else 0
                avg_pnl = s['pnl'] / s['total'] if s['total'] > 0 else 0
                bar = '█' * max(1, int(abs(avg_pnl) / 5))
                sign = '+' if avg_pnl >= 0 else ''
                print(f"  {h:2d}:00  {s['total']:3d} trades  WR={wr:.0%}  "
                      f"avg={sign}${avg_pnl:.1f}  {'🟢' if avg_pnl > 0 else '🔴'}{bar}")

        if day_stats:
            print(f"\nPerformance by day:")
            for d in sorted(day_stats.keys()):
                s = day_stats[d]
                wr = s['wins'] / s['total'] if s['total'] > 0 else 0
                avg_pnl = s['pnl'] / s['total'] if s['total'] > 0 else 0
                sign = '+' if avg_pnl >= 0 else ''
                print(f"  {day_names[d]:3s}  {s['total']:3d} trades  WR={wr:.0%}  "
                      f"P&L=${s['pnl']:,.0f}  avg={sign}${avg_pnl:.1f}")

        # Signal component correlation analysis
        if trade_signals and len(trade_signals) == len(trades):
            components = ['position_score', 'energy_score', 'entropy_score',
                          'confluence_score', 'timing_score', 'channel_health', 'confidence']

            for label, mask in [('ALL', [True]*len(trades)),
                                ('BOUNCE', [t.signal_type == 'bounce' for t in trades]),
                                ('BREAK', [t.signal_type == 'break' for t in trades])]:
                idx = [i for i, m in enumerate(mask) if m]
                if len(idx) < 10:
                    continue
                sub_trades = [trades[i] for i in idx]
                sub_sigs = [trade_signals[i] for i in idx]
                outcomes = np.array([1 if t.pnl > 0 else 0 for t in sub_trades])
                pnl_vals = np.array([t.pnl_pct for t in sub_trades])

                print(f"\nSignal component analysis — {label} ({len(idx)} trades):")
                print(f"  {'Component':<18s} {'Avg(Win)':<10s} {'Avg(Loss)':<10s} {'WinCorr':<10s} {'PnlCorr':<10s}")
                for comp in components:
                    vals = np.array([s.get(comp, 0) for s in sub_sigs])
                    if np.std(vals) < 1e-6:
                        continue
                    win_vals = vals[outcomes == 1]
                    loss_vals = vals[outcomes == 0]
                    win_corr = np.corrcoef(vals, outcomes)[0, 1] if len(vals) > 2 else 0
                    pnl_corr = np.corrcoef(vals, pnl_vals)[0, 1] if len(vals) > 2 else 0
                    flag = '**' if abs(win_corr) > 0.1 else '  '
                    print(f"  {comp:<18s} {np.mean(win_vals):<10.3f} {np.mean(loss_vals):<10.3f} "
                          f"{win_corr:<+10.3f} {pnl_corr:<+10.3f} {flag}")

    # Print filter diagnostics
    if ml_stats.get('total_signals', 0) > 0:
        print(f"\n{'='*60}")
        print(f"SIGNAL FILTER DIAGNOSTICS ({ml_stats['total_signals']} total physics signals)")
        print(f"{'='*60}")
        # Sort by count descending, skip zero-count
        for k, v in sorted(ml_stats.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
            if isinstance(v, (int, float)) and v > 0:
                print(f"  {k:<30s} {v:>6}")
        print(f"  {'TRADES TAKEN':<30s} {metrics.total_trades:>6}")
        print(f"{'='*60}\n")

    # Leverage/exposure audit
    if trades:
        max_trade_size = max(t.trade_size for t in trades)
        avg_trade_size = np.mean([t.trade_size for t in trades])
        max_leverage_used = max_trade_size / initial_equity if initial_equity > 0 else 0
        if max_leverage_used > 10:
            print(f"\n⚠️  LEVERAGE AUDIT:")
            print(f"  Max single trade:  ${max_trade_size:,.0f} ({max_leverage_used:.0f}x equity)")
            print(f"  Avg trade size:    ${avg_trade_size:,.0f} ({avg_trade_size/initial_equity:.0f}x equity)")
            print(f"  Headline P&L may be inflated by concentrated leverage.")
            if not realistic:
                print(f"  Mode: UNREALISTIC — multiplicative boosts active")

    # Print error summary — never let failures go unnoticed
    if _error_counts:
        print(f"\n{'='*60}")
        print(f"⚠️  ERROR SUMMARY ({sum(v['count'] for v in _error_counts.values())} total failures)")
        print(f"{'='*60}")
        for cat, info in sorted(_error_counts.items(), key=lambda x: -x[1]['count']):
            print(f"  {cat:<30s} {info['count']:>6}x  (first: {info['first_err'][:60]})")
        print(f"{'='*60}\n")

    if capture_features:
        return metrics, trades, equity_curve, trade_features, trade_signals
    return metrics, trades, equity_curve


def run_walk_forward(eval_interval: int = 3, max_hold_bars: int = 60,
                      min_confidence: float = 0.45):
    """
    Walk-forward validation: run 60-day backtest, split trades into
    in-sample (first 40 days) and out-of-sample (last 20 days).
    Compares metrics side by side.
    """
    metrics, trades, equity_curve = run_backtest(
        days=60, eval_interval=eval_interval,
        max_hold_bars=max_hold_bars, min_confidence=min_confidence,
    )

    if not trades:
        print("No trades to analyze")
        return

    # Split by entry bar — first 2/3 of bars = IS, last 1/3 = OOS
    total_bars = max(t.exit_bar for t in trades)
    split_bar = int(total_bars * 2 / 3)

    is_trades = [t for t in trades if t.entry_bar < split_bar]
    oos_trades = [t for t in trades if t.entry_bar >= split_bar]

    def summarize(label, tlist):
        if not tlist:
            print(f"\n  {label}: No trades")
            return
        wins = sum(1 for t in tlist if t.pnl > 0)
        total_pnl = sum(t.pnl for t in tlist)
        gross_win = sum(t.pnl for t in tlist if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in tlist if t.pnl <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
        avg_win = np.mean([t.pnl_pct for t in tlist if t.pnl > 0]) if wins > 0 else 0
        avg_loss = np.mean([t.pnl_pct for t in tlist if t.pnl <= 0]) if len(tlist) - wins > 0 else 0
        bounce = [t for t in tlist if t.signal_type == 'bounce']
        brk = [t for t in tlist if t.signal_type == 'break']
        bounce_wr = sum(1 for t in bounce if t.pnl > 0) / len(bounce) if bounce else 0
        brk_wr = sum(1 for t in brk if t.pnl > 0) / len(brk) if brk else 0
        print(f"\n  {label}:")
        print(f"    Trades: {len(tlist)} | WR: {wins/len(tlist):.0%} | PF: {pf:.2f} | "
              f"P&L: ${total_pnl:,.0f} | Exp: ${total_pnl/len(tlist):.1f}/trade")
        print(f"    Avg Win: {avg_win:.2%} | Avg Loss: {avg_loss:.2%}")
        print(f"    Bounce: {len(bounce)} trades, {bounce_wr:.0%} WR | "
              f"Break: {len(brk)} trades, {brk_wr:.0%} WR")

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION (split at bar {split_bar}/{total_bars})")
    print(f"{'='*60}")
    summarize("IN-SAMPLE (first ~40 days)", is_trades)
    summarize("OUT-OF-SAMPLE (last ~20 days)", oos_trades)

    # Stability metrics
    if is_trades and oos_trades:
        is_wr = sum(1 for t in is_trades if t.pnl > 0) / len(is_trades)
        oos_wr = sum(1 for t in oos_trades if t.pnl > 0) / len(oos_trades)
        is_exp = sum(t.pnl for t in is_trades) / len(is_trades)
        oos_exp = sum(t.pnl for t in oos_trades) / len(oos_trades)
        wr_decay = (oos_wr - is_wr) / is_wr if is_wr > 0 else 0
        exp_decay = (oos_exp - is_exp) / is_exp if is_exp > 0 else 0
        print(f"\n  Stability:")
        print(f"    WR decay: {wr_decay:+.0%} (IS→OOS)")
        print(f"    Exp decay: {exp_decay:+.0%} (IS→OOS)")
        if abs(wr_decay) < 0.15 and abs(exp_decay) < 0.40:
            print(f"    ✅ Strategy appears STABLE out-of-sample")
        elif abs(wr_decay) < 0.25:
            print(f"    ⚠️  Moderate OOS degradation — monitor closely")
        else:
            print(f"    ❌ Significant OOS degradation — possible overfit")


def main():
    parser = argparse.ArgumentParser(description='Channel Surfer Backtest')
    parser.add_argument('--days', type=int, default=30, help='Days of 5min data')
    parser.add_argument('--eval-interval', type=int, default=3, help='Bars between evaluations (3=15min optimal)')
    parser.add_argument('--max-hold', type=int, default=60, help='Max bars to hold')
    parser.add_argument('--min-conf', type=float, default=0.01, help='Minimal gate: hard skips do real filtering')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward validation')
    parser.add_argument('--ml', type=str, default=None,
                       help='Path to ML model for signal enhancement (e.g. surfer_models/gbt_model.pkl)')
    parser.add_argument('--ml-compare', action='store_true',
                       help='Run both physics-only and ML-enhanced, then compare')
    parser.add_argument('--dump-trades', type=str, default=None,
                       help='Filter to dump: stop, trail, timeout, losers, all')
    parser.add_argument('--realistic', action='store_true',
                       help='Realistic mode: flat 2%% risk sizing, slippage, commissions')
    parser.add_argument('--refresh', action='store_true',
                       help='Force refresh yfinance data (clear cache)')
    args = parser.parse_args()

    if args.refresh and _SURFER_DATA_CACHE.exists():
        _SURFER_DATA_CACHE.unlink()
        print("[CACHE] Cleared")

    ml_model = None
    if args.ml:
        print(f"\nLoading ML model from {args.ml}...")
        if _os_mod.path.isdir(args.ml):
            # Directory mode: load gbt_model.pkl from the directory
            gbt_path = _os_mod.path.join(args.ml, 'gbt_model.pkl')
            if _os_mod.path.exists(gbt_path):
                from v15.core.surfer_ml import GBTModel
                ml_model = GBTModel.load(gbt_path)
            else:
                # Try best_model.pkl
                best_path = _os_mod.path.join(args.ml, 'best_model.pkl')
                if _os_mod.path.exists(best_path):
                    from v15.core.surfer_ml import GBTModel
                    ml_model = GBTModel.load(best_path)
        elif args.ml.endswith('.pkl'):
            from v15.core.surfer_ml import GBTModel
            ml_model = GBTModel.load(args.ml)
        elif 'transformer' in args.ml:
            from v15.core.surfer_ml import MultiTFTransformer
            ml_model = MultiTFTransformer.load(args.ml)
        elif 'survival' in args.ml:
            from v15.core.surfer_ml import SurvivalModel
            ml_model = SurvivalModel.load(args.ml)
        print(f"  Loaded: {type(ml_model).__name__}")

    if args.ml_compare:
        # Run physics-only first
        print("\n" + "=" * 70)
        print("RUN 1: PHYSICS-ONLY (baseline)")
        print("=" * 70)
        m1, t1, _ = run_backtest(
            days=args.days, eval_interval=args.eval_interval,
            max_hold_bars=args.max_hold, min_confidence=args.min_conf,
        )

        # Then ML-enhanced
        if ml_model is None:
            print("\nLoading default ML model...")
            from v15.core.surfer_ml import GBTModel
            import os
            model_path = 'surfer_models/gbt_model.pkl'
            if os.path.exists(model_path):
                ml_model = GBTModel.load(model_path)
            else:
                print(f"  No model found at {model_path}. Run: python3 -m v15.core.surfer_ml train")
                return

        print("\n" + "=" * 70)
        print("RUN 2: ML-ENHANCED")
        print("=" * 70)
        m2, t2, _ = run_backtest(
            days=args.days, eval_interval=args.eval_interval,
            max_hold_bars=args.max_hold, min_confidence=args.min_conf,
            ml_model=ml_model,
        )

        # Compare
        print("\n" + "=" * 70)
        print("COMPARISON: Physics-Only vs ML-Enhanced")
        print("=" * 70)
        print(f"{'Metric':<20s} {'Physics':<15s} {'ML-Enhanced':<15s} {'Change':<15s}")
        print("-" * 65)

        comparisons = [
            ('Trades', m1.total_trades, m2.total_trades),
            ('Win Rate', f"{m1.win_rate:.0%}", f"{m2.win_rate:.0%}"),
            ('Profit Factor', f"{m1.profit_factor:.2f}", f"{m2.profit_factor:.2f}"),
            ('Total P&L', f"${m1.total_pnl:,.0f}", f"${m2.total_pnl:,.0f}"),
            ('Expectancy', f"${m1.expectancy:,.2f}", f"${m2.expectancy:,.2f}"),
            ('Max DD', f"{m1.max_drawdown_pct:.1%}", f"{m2.max_drawdown_pct:.1%}"),
        ]
        for name, v1, v2 in comparisons:
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if v1 > 0:
                    change = f"{(v2 - v1) / v1 * 100:+.1f}%"
                else:
                    change = "N/A"
                print(f"  {name:<18s} {str(v1):<15s} {str(v2):<15s} {change:<15s}")
            else:
                print(f"  {name:<18s} {str(v1):<15s} {str(v2):<15s}")

    elif args.walk_forward:
        run_walk_forward(
            eval_interval=args.eval_interval,
            max_hold_bars=args.max_hold,
            min_confidence=args.min_conf,
        )
    else:
        realistic_kwargs = {}
        if args.realistic:
            realistic_kwargs = dict(
                realistic=True,
                slippage_bps=3.0,
                commission_per_share=0.005,
                max_leverage=4.0,
                initial_capital=100_000.0,
            )
        metrics, trades, eq = run_backtest(
            days=args.days,
            eval_interval=args.eval_interval,
            max_hold_bars=args.max_hold,
            min_confidence=args.min_conf,
            ml_model=ml_model,
            **realistic_kwargs,
        )

        # Dump individual trade details if requested
        if args.dump_trades and trades:
            filt = args.dump_trades.lower()
            if filt == 'all':
                dump = trades
            elif filt == 'losers':
                dump = [t for t in trades if t.pnl <= 0]
            else:
                dump = [t for t in trades if t.exit_reason == filt]
            print(f"\n{'='*80}")
            print(f"TRADE DUMP: {filt} ({len(dump)} trades)")
            print(f"{'='*80}")
            for i, t in enumerate(dump):
                flags = []
                if t.el_flagged: flags.append('EL')
                if t.is_flagged: flags.append('IS')
                flag_str = f" [{','.join(flags)}]" if flags else ""
                print(f"  #{i+1} {t.direction:4s} {t.signal_type:6s} "
                      f"conf={t.confidence:.2f} hold={t.hold_bars:2d} "
                      f"pnl={t.pnl_pct:+.3%} (${t.pnl:+.0f}) "
                      f"stop={t.stop_pct:.3%} "
                      f"mae={t.mae_pct:.3%} mfe={t.mfe_pct:.3%} "
                      f"exit={t.exit_reason} tf={t.primary_tf} "
                      f"size=${t.trade_size:.0f}{flag_str}")


if __name__ == '__main__':
    main()
