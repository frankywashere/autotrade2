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

    # EL-flagged trades get more aggressive trailing to lock profits sooner
    # ML-guided trail adjustments
    el = position.el_flagged
    fast_rev = position.fast_reversion and not is_breakout

    if position.direction == 'BUY':
        if window_high > position.trailing_stop:
            position.trailing_stop = window_high

        if is_breakout:
            profit_from_best = (position.trailing_stop - entry) / entry
            # Three-tier breakout trail (with progressive tightening)
            if profit_from_best > 0.015:
                trail_from_best = position.trailing_stop * (1 - initial_stop_dist * 0.06)
                effective_stop = max(position.stop_price, trail_from_best)
            elif profit_from_best > 0.008:
                trail_from_best = position.trailing_stop * (1 - initial_stop_dist * 0.10)
                effective_stop = max(position.stop_price, trail_from_best)
            elif profit_from_best > (0.002 if el else 0.0035):
                trail_mult = 0.20 if el else 0.25
                trail_from_best = position.trailing_stop * (1 - initial_stop_dist * trail_mult)
                effective_stop = max(position.stop_price, trail_from_best)
            else:
                effective_stop = position.stop_price
        else:
            profit_from_entry = (position.trailing_stop - entry) / entry
            profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
            # EL: lower thresholds and tighter trails
            # Fast reversion: even tighter — mean reversion resolves quickly
            tight = el or fast_rev
            if profit_ratio >= (0.60 if tight else 0.70):
                trail_from_best = position.trailing_stop * (1 - initial_stop_dist * (0.10 if tight else 0.12))
                effective_stop = max(position.stop_price, trail_from_best)
            elif profit_ratio >= (0.30 if tight else 0.40):
                trail_from_best = position.trailing_stop * (1 - initial_stop_dist * (0.25 if tight else 0.30))
                effective_stop = max(position.stop_price, trail_from_best)
            elif profit_ratio >= (0.15 if tight else 0.20):
                effective_stop = max(position.stop_price, entry * 0.999)
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
            # Three-tier breakout trail
            if profit_from_best > 0.015:
                trail_from_best = position.trailing_stop * (1 + initial_stop_dist * 0.06)
                effective_stop = min(position.stop_price, trail_from_best)
            elif profit_from_best > 0.008:
                trail_from_best = position.trailing_stop * (1 + initial_stop_dist * 0.10)
                effective_stop = min(position.stop_price, trail_from_best)
            elif profit_from_best > (0.002 if el else 0.0035):
                trail_mult = 0.20 if el else 0.25
                trail_from_best = position.trailing_stop * (1 + initial_stop_dist * trail_mult)
                effective_stop = min(position.stop_price, trail_from_best)
            else:
                effective_stop = position.stop_price
        else:
            profit_from_entry = (entry - position.trailing_stop) / entry
            profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
            tight = el or fast_rev
            if profit_ratio >= (0.60 if tight else 0.70):
                trail_from_best = position.trailing_stop * (1 + initial_stop_dist * (0.10 if tight else 0.12))
                effective_stop = min(position.stop_price, trail_from_best)
            elif profit_ratio >= (0.30 if tight else 0.40):
                trail_from_best = position.trailing_stop * (1 + initial_stop_dist * (0.25 if tight else 0.30))
                effective_stop = min(position.stop_price, trail_from_best)
            elif profit_ratio >= (0.15 if tight else 0.20):
                effective_stop = min(position.stop_price, entry * 1.001)
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


def run_backtest(
    days: int = 30,
    eval_interval: int = 3,     # Check every 3 bars = 15 min
    max_hold_bars: int = 60,    # Max 5 hours (60 * 5min)
    position_size: float = 10000.0,  # $10k per trade
    min_confidence: float = 0.45,
    use_multi_tf: bool = True,  # Use higher TF data for context
    ml_model=None,              # Optional ML model for signal enhancement
) -> tuple:
    """
    Run Channel Surfer backtest on historical 5-min TSLA data.

    If ml_model is provided, uses ML predictions to:
    - Filter out signals where ML predicts HOLD
    - Boost confidence when ML agrees with physics
    - Adjust stop/TP based on predicted channel lifetime
    - Skip trades where ML predicts imminent break in wrong direction

    Returns:
        (metrics, trades) tuple
    """
    import yfinance as yf
    from v15.core.channel import detect_channels_multi_window, select_best_channel
    from v15.core.channel_surfer import analyze_channels, SIGNAL_TFS, TF_WINDOWS

    ml_active = ml_model is not None
    quality_scorer = None
    ensemble_model = None
    ensemble_base_models = {}
    if ml_active:
        from v15.core.surfer_ml import (
            extract_tf_features, extract_cross_tf_features,
            extract_context_features, extract_correlation_features,
            extract_temporal_features, TradeQualityScorer,
            EnsembleModel, GBTModel, MultiTFTransformer, SurvivalModel,
            RegimeConditionalModel, TrendGBTModel, CVEnsembleModel, PhysicsResidualModel, AdverseMovementPredictor, CompositeSignalScorer, VolatilityTransitionModel, ExitTimingOptimizer, MomentumExhaustionDetector, CrossAssetAmplifier, StopLossPredictor, DynamicTrailOptimizer, IntradaySessionModel, ChannelMaturityPredictor, ReturnAsymmetryPredictor, GapRiskPredictor, MeanReversionSpeedModel, LiquidityStateClassifier, TradeDurationPredictor, AdversarialTradeSelector, QuantileRiskEstimator, TailRiskDetector, StopDistanceOptimizer, VolatilityClusteringPredictor, ExtremeLoserDetector, DrawdownMagnitudePredictor, WinStreakDetector, FeatureInteractionLoser, BounceLoserDetector, MomentumReversalDetector, ImmediateStopDetector, ProfitVelocityPredictor, BreakoutStopPredictor,
            get_feature_names, ML_TFS, PER_TF_FEATURES,
            CROSS_TF_FEATURES, CONTEXT_FEATURES, CORRELATION_FEATURES,
            TEMPORAL_FEATURES,
        )
        ml_feature_names = get_feature_names()
        ml_history_buffer: List[Dict] = []
        ml_feature_window: List[np.ndarray] = []  # For TrendGBT sliding window
        ml_stats = {'total_signals': 0, 'ml_filtered': 0, 'ml_boosted': 0, 'ml_agreed': 0,
                     'quality_filtered': 0, 'quality_boosted': 0, 'ensemble_filtered': 0}
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
            except Exception:
                pass

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
                    except Exception:
                        pass

                surv_path = _os.path.join(model_dir, 'survival_model.pt')
                if _os.path.exists(surv_path):
                    try:
                        ensemble_base_models['survival'] = SurvivalModel.load(surv_path)
                    except Exception:
                        pass

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
            except Exception:
                pass

        # Try to load TrendGBT model
        trend_gbt_model = None
        tg_path = _os.path.join(model_dir, 'trend_gbt_model.pkl')
        if _os.path.exists(tg_path):
            try:
                trend_gbt_model = TrendGBTModel.load(tg_path)
                print(f"[ML] TrendGBT loaded (top-{trend_gbt_model.TOP_K} features + trends)")
                ml_stats['trend_gbt_confirmed'] = 0
                ml_stats['trend_gbt_filtered'] = 0
            except Exception:
                pass

        # Try to load CV Ensemble model
        cv_ensemble_model = None
        cv_path = _os.path.join(model_dir, 'cv_ensemble_model.pkl')
        if _os.path.exists(cv_path):
            try:
                cv_ensemble_model = CVEnsembleModel.load(cv_path)
                print(f"[ML] CV Ensemble loaded ({cv_ensemble_model.N_FOLDS}-fold)")
                ml_stats['cv_high_consensus'] = 0
                ml_stats['cv_low_consensus'] = 0
            except Exception:
                pass

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
            except Exception:
                pass

        # Try to load Adverse Movement model
        adverse_model = None
        adv_path = _os.path.join(model_dir, 'adverse_movement_model.pkl')
        if _os.path.exists(adv_path):
            try:
                adverse_model = AdverseMovementPredictor.load(adv_path)
                print(f"[ML] Adverse Movement model loaded")
                ml_stats['adverse_filtered'] = 0
                ml_stats['adverse_boosted'] = 0
            except Exception:
                pass

        # Try to load Composite Signal Scorer
        composite_model = None
        comp_path = _os.path.join(model_dir, 'composite_scorer.pkl')
        if _os.path.exists(comp_path):
            try:
                composite_model = CompositeSignalScorer.load(comp_path)
                print(f"[ML] Composite scorer loaded ({len(composite_model.meta_feature_names or [])} meta-features)")
                ml_stats['composite_agreed'] = 0
                ml_stats['composite_filtered'] = 0
            except Exception:
                pass

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
            except Exception:
                pass

        # Try to load Exit Timing model
        exit_timing_model = None
        exit_path = _os.path.join(model_dir, 'exit_timing_opt.pkl')
        if _os.path.exists(exit_path):
            try:
                exit_timing_model = ExitTimingOptimizer.load(exit_path)
                print(f"[ML] Exit Timing model loaded")
                ml_stats['exit_tightened'] = 0
                ml_stats['exit_early'] = 0
            except Exception:
                pass

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
            except Exception:
                pass

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
            except Exception:
                pass

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
            except Exception:
                pass

        # Try to load Dynamic Trail Optimizer
        trail_model = None
        trail_path = _os.path.join(model_dir, 'trail_optimizer.pkl')
        if _os.path.exists(trail_path):
            try:
                trail_model = DynamicTrailOptimizer.load(trail_path)
                print(f"[ML] Dynamic Trail Optimizer loaded (AUC 0.700)")
                ml_stats['trail_tightened'] = 0
                ml_stats['trail_loosened'] = 0
            except Exception:
                pass

        # Architecture 21: Intraday Session Model
        session_model = None
        session_path = _os.path.join(model_dir, 'session_model.pkl')
        if _os.path.exists(session_path):
            try:
                session_model = IntradaySessionModel.load(session_path)
                print(f"[ML] Intraday Session Model loaded (Quality AUC 0.648)")
                ml_stats['session_boost'] = 0
                ml_stats['session_penalty'] = 0
            except Exception:
                pass

        # Architecture 22: Channel Maturity Predictor
        maturity_model = None
        maturity_path = _os.path.join(model_dir, 'maturity_model.pkl')
        if _os.path.exists(maturity_path):
            try:
                maturity_model = ChannelMaturityPredictor.load(maturity_path)
                print(f"[ML] Channel Maturity Predictor loaded (AUC 0.677)")
                ml_stats['maturity_skip'] = 0
                ml_stats['maturity_boost'] = 0
            except Exception:
                pass

        # Architecture 24: Return Asymmetry Predictor
        asymmetry_model = None
        asymmetry_path = _os.path.join(model_dir, 'asymmetry_model.pkl')
        if _os.path.exists(asymmetry_path):
            try:
                asymmetry_model = ReturnAsymmetryPredictor.load(asymmetry_path)
                print(f"[ML] Return Asymmetry Predictor loaded (Spike AUC 0.680)")
                ml_stats['asym_widen_stop'] = 0
                ml_stats['asym_tighten_trail'] = 0
            except Exception:
                pass

        # Architecture 25: Gap Risk Predictor
        gap_risk_model = None
        gap_path = _os.path.join(model_dir, 'gap_risk_model.pkl')
        if _os.path.exists(gap_path):
            try:
                gap_risk_model = GapRiskPredictor.load(gap_path)
                print(f"[ML] Gap Risk Predictor loaded (AUC 0.852)")
                ml_stats['gap_risk_skip'] = 0
            except Exception:
                pass

        # Architecture 26: Mean Reversion Speed
        reversion_model = None
        rev_path = _os.path.join(model_dir, 'reversion_model.pkl')
        if _os.path.exists(rev_path):
            try:
                reversion_model = MeanReversionSpeedModel.load(rev_path)
                print(f"[ML] Mean Reversion Speed loaded (AUC 0.873)")
                ml_stats['rev_fast_boost'] = 0
                ml_stats['rev_slow_penalty'] = 0
            except Exception:
                pass

        # Architecture 27: Liquidity State (slippage risk only)
        liquidity_model = None
        liq_path = _os.path.join(model_dir, 'liquidity_model.pkl')
        if _os.path.exists(liq_path):
            try:
                liquidity_model = LiquidityStateClassifier.load(liq_path)
                print(f"[ML] Liquidity State loaded (slippage corr 0.499)")
                ml_stats['liq_high_slippage'] = 0
            except Exception:
                pass

        # Architecture 31: Trade Duration Predictor
        duration_model = None
        dur_path = _os.path.join(model_dir, 'duration_model.pkl')
        if _os.path.exists(dur_path):
            try:
                duration_model = TradeDurationPredictor.load(dur_path)
                print(f"[ML] Trade Duration loaded (Quick AUC 0.617)")
                ml_stats['dur_quick_exit'] = 0
                ml_stats['dur_extend_hold'] = 0
            except Exception:
                pass

        # Architecture 37: Adversarial Trade Selector
        adversarial_model = None
        adv_path = _os.path.join(model_dir, 'adversarial_model.pkl')
        if _os.path.exists(adv_path):
            try:
                adversarial_model = AdversarialTradeSelector.load(adv_path)
                print(f"[ML] Adversarial Selector loaded (AUC 0.605)")
                ml_stats['adv_favorable_boost'] = 0
                ml_stats['adv_unfavorable_penalty'] = 0
            except Exception:
                pass

        # Architecture 40: Quantile Risk Estimator
        quantile_risk_model = None
        qr_path = _os.path.join(model_dir, 'quantile_risk_model.pkl')
        if _os.path.exists(qr_path):
            try:
                quantile_risk_model = QuantileRiskEstimator.load(qr_path)
                print(f"[ML] Quantile Risk loaded (spread corr 0.298)")
                ml_stats['qr_high_risk_penalty'] = 0
                ml_stats['qr_favorable_asym'] = 0
            except Exception:
                pass

        # Architecture 41: Tail Risk Detector
        tail_risk_model = None
        tr_path = _os.path.join(model_dir, 'tail_risk_model.pkl')
        if _os.path.exists(tr_path):
            try:
                tail_risk_model = TailRiskDetector.load(tr_path)
                print(f"[ML] Tail Risk Detector loaded (AUC 0.743)")
                ml_stats['tail_bounce_penalty'] = 0
                ml_stats['tail_break_boost'] = 0
            except Exception:
                pass

        # Architecture 43: Stop Distance Optimizer
        stop_dist_model = None
        sd_path = _os.path.join(model_dir, 'stop_distance_model.pkl')
        if _os.path.exists(sd_path):
            try:
                stop_dist_model = StopDistanceOptimizer.load(sd_path)
                print(f"[ML] Stop Distance loaded (MAE corr 0.346)")
                ml_stats['sd_wide_stop'] = 0
                ml_stats['sd_tight_stop'] = 0
            except Exception:
                pass

        # Architecture 44: Volatility Clustering
        vol_cluster_model = None
        vcl_path = _os.path.join(model_dir, 'vol_clustering_model.pkl')
        if _os.path.exists(vcl_path):
            try:
                vol_cluster_model = VolatilityClusteringPredictor.load(vcl_path)
                print(f"[ML] Vol Clustering loaded (AUC 0.683)")
                ml_stats['vc_vol_inc_penalty'] = 0
                ml_stats['vc_vol_dec_boost'] = 0
            except Exception:
                pass

        # Architecture 45: Extreme Loser Detector
        extreme_loser_model = None
        el_path = _os.path.join(model_dir, 'extreme_loser_model.pkl')
        if _os.path.exists(el_path):
            try:
                extreme_loser_model = ExtremeLoserDetector.load(el_path)
                print(f"[ML] Extreme Loser Detector loaded (AUC 0.654)")
                ml_stats['el_penalty'] = 0
                ml_stats['el_skip'] = 0
            except Exception:
                pass

        # Architecture 49: Drawdown Magnitude Predictor
        drawdown_mag_model = None
        dm_path = _os.path.join(model_dir, 'drawdown_magnitude_model.pkl')
        if _os.path.exists(dm_path):
            try:
                drawdown_mag_model = DrawdownMagnitudePredictor.load(dm_path)
                print(f"[ML] Drawdown Magnitude loaded (P75 corr 0.283)")
                ml_stats['dm_high_dd_pen'] = 0
                ml_stats['dm_low_dd_boost'] = 0
            except Exception:
                pass

        # Architecture 50: Win Streak Detector
        win_streak_model = None
        ws_path = _os.path.join(model_dir, 'win_streak_model.pkl')
        if _os.path.exists(ws_path):
            try:
                win_streak_model = WinStreakDetector.load(ws_path)
                print(f"[ML] Win Streak Detector loaded (AUC 0.620)")
                ml_stats['ws_boost'] = 0
            except Exception:
                pass

        # Architecture 54: Bounce Loser Detector
        bounce_loser_model = None
        bl_path = _os.path.join(model_dir, 'bounce_loser_model.pkl')
        if _os.path.exists(bl_path):
            try:
                bounce_loser_model = BounceLoserDetector.load(bl_path)
                print(f"[ML] Bounce Loser Detector loaded (AUC 0.670)")
                ml_stats['bl_penalty'] = 0
            except Exception:
                pass

        # Architecture 55: Feature Interaction Loser
        feat_int_model = None
        fi_path = _os.path.join(model_dir, 'feature_interaction_model.pkl')
        if _os.path.exists(fi_path):
            try:
                feat_int_model = FeatureInteractionLoser.load(fi_path)
                print(f"[ML] Feature Interaction Loser loaded (AUC 0.733)")
                ml_stats['fi_penalty'] = 0
            except Exception:
                pass

        # Architecture 56: Momentum Reversal Detector
        mom_rev_model = None
        mr_path = _os.path.join(model_dir, 'momentum_reversal_model.pkl')
        if _os.path.exists(mr_path):
            try:
                mom_rev_model = MomentumReversalDetector.load(mr_path)
                print(f"[ML] Momentum Reversal loaded (AUC 0.663)")
                ml_stats['mr_penalty'] = 0
            except Exception:
                pass

        # Architecture 57: Immediate Stop Detector
        imm_stop_model = None
        is_path = _os.path.join(model_dir, 'immediate_stop_model.pkl')
        if _os.path.exists(is_path):
            try:
                imm_stop_model = ImmediateStopDetector.load(is_path)
                print(f"[ML] Immediate Stop Detector loaded (AUC 0.659)")
                ml_stats['is_skip'] = 0
            except Exception:
                pass

        # Architecture 58: Profit Velocity Predictor
        profit_vel_model = None
        pv_path = _os.path.join(model_dir, 'profit_velocity_model.pkl')
        if _os.path.exists(pv_path):
            try:
                profit_vel_model = ProfitVelocityPredictor.load(pv_path)
                print(f"[ML] Profit Velocity Predictor loaded (AUC 0.649)")
            except Exception:
                pass

        # Architecture 59: Breakout Stop Predictor
        breakout_stop_model = None
        bsp_path = _os.path.join(model_dir, 'breakout_stop_model.pkl')
        if _os.path.exists(bsp_path):
            try:
                breakout_stop_model = BreakoutStopPredictor.load(bsp_path)
                print(f"[ML] Breakout Stop Predictor loaded (AUC 0.794)")
                ml_stats['bsp_tighten'] = 0
                ml_stats['bsp_skip'] = 0
            except Exception:
                pass

    # Fetch data (with file cache for consistent iterations)
    _cache_hit = False
    if _SURFER_DATA_CACHE.exists() and not _os_mod.environ.get('SURFER_REFRESH'):
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

    closes = tsla['close'].values
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
    max_positions = 2
    pending_entries = []  # Deferred entries: list of (bar, direction, signal_type, confidence, stop_pct, tp_pct, primary_tf, ou_hl, max_hold, el_flagged, fast_rev, trade_size, signal_data)
    equity = position_size * 10  # Start with 100k
    initial_equity = equity
    peak_equity = equity
    max_dd = 0.0
    consecutive_losses = 0  # Track losing streak for position reduction
    consecutive_wins = 0    # Track winning streak for position ramping
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
                except Exception:
                    pass

            # Dynamic Trail Optimizer: adjust trail tightness based on ML
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
                except Exception:
                    pass

            result = _check_position_exit(
                position, bar, current_price, window_high, window_low, eval_interval)

            if result is not None:
                exit_reason, exit_price = result
                bars_held = bar - position.entry_bar

                if position.direction == 'BUY':
                    pnl_pct = (exit_price - position.entry_price) / position.entry_price
                else:
                    pnl_pct = (position.entry_price - exit_price) / position.entry_price
                pnl = pnl_pct * position.trade_size

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
                else:
                    consecutive_losses = 0
                    consecutive_wins += 1

                # Track daily P&L for circuit breaker
                daily_pnl += pnl
                if daily_pnl < -500:
                    daily_breaker_active = True

                closed_indices.append(pi)
                trade_signals.append(position_signals[pi])

        # Remove closed positions (reverse order to preserve indices)
        for pi in sorted(closed_indices, reverse=True):
            positions.pop(pi)
            position_signals.pop(pi)

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
            if use_multi_tf:
                current_time = tsla.index[bar]
                # Normalize to tz-naive for comparison
                if current_time.tzinfo is not None:
                    current_time_naive = current_time.tz_localize(None)
                else:
                    current_time_naive = current_time
                for tf_label, tf_df in higher_tf_data.items():
                    # Only use higher-TF data available at current time (no lookahead)
                    tf_idx = tf_df.index
                    if tf_idx.tz is not None:
                        tf_available = tf_df[tf_idx <= current_time]
                    else:
                        tf_available = tf_df[tf_idx <= current_time_naive]
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
                    except Exception:
                        pass

            try:
                analysis = analyze_channels(
                    channels_by_tf, prices_by_tf, current_prices_dict,
                    volumes_by_tf=volumes_dict if volumes_dict else None,
                )
            except Exception:
                continue

            sig = analysis.signal

            # --- ML Enhancement ---
            ml_prediction = None
            if ml_active and sig.action in ('BUY', 'SELL'):
                ml_stats['total_signals'] += 1
                try:
                    # Extract features
                    num_features = len(ml_feature_names)
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

                    # Build snapshot for temporal features
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
                        bt_snapshot, ml_history_buffer,
                        closes=closes, bar_idx=bar, eval_interval=3,
                    )
                    feature_vec[offset:offset + len(TEMPORAL_FEATURES)] = temporal_feats
                    offset += len(TEMPORAL_FEATURES)

                    ml_history_buffer.append(bt_snapshot)
                    if len(ml_history_buffer) > 20:
                        ml_history_buffer.pop(0)

                    corr_feats = extract_correlation_features(
                        bar, closes, spy_df=spy_df, vix_df=vix_df,
                        tsla_index=tsla.index,
                    )
                    feature_vec[offset:offset + len(CORRELATION_FEATURES)] = corr_feats

                    # Update feature window for TrendGBT
                    ml_feature_window.append(feature_vec.copy())
                    if len(ml_feature_window) > TrendGBTModel.WINDOW_SIZE:
                        ml_feature_window.pop(0)

                    # Run ML prediction
                    ml_prediction = ml_model.predict(feature_vec.reshape(1, -1))

                    # ML Action: 0=HOLD, 1=BUY, 2=SELL
                    if 'action' in ml_prediction:
                        ml_action_id = int(ml_prediction['action'][0])
                        physics_action_id = 1 if sig.action == 'BUY' else 2

                        # If ML says HOLD → filter the signal
                        if ml_action_id == 0:
                            ml_stats['ml_filtered'] += 1
                            continue

                        # If ML agrees with physics → boost confidence
                        if ml_action_id == physics_action_id:
                            sig.confidence *= 1.25  # 25% boost
                            ml_stats['ml_agreed'] += 1

                        # If ML disagrees (says opposite direction) → reduce confidence
                        elif ml_action_id != 0 and ml_action_id != physics_action_id:
                            sig.confidence *= 0.60  # 40% penalty
                            ml_stats['ml_filtered'] += 1

                    # ML Break direction: if imminent break against our position, skip
                    if 'break_dir' in ml_prediction:
                        bd = int(ml_prediction['break_dir'][0])
                        if 'lifetime' in ml_prediction:
                            lifetime = float(ml_prediction['lifetime'][0])
                            # If channel breaks in < 5 bars in wrong direction, skip
                            if lifetime < 5:
                                if sig.action == 'BUY' and bd == 2:  # Break down
                                    ml_stats['ml_filtered'] += 1
                                    continue
                                elif sig.action == 'SELL' and bd == 1:  # Break up
                                    ml_stats['ml_filtered'] += 1
                                    continue

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

                                if ens_action == 0:  # Ensemble says HOLD
                                    ml_stats['ensemble_filtered'] += 1
                                    continue
                                elif ens_action != physics_action_id:
                                    # Ensemble disagrees with physics → penalize
                                    sig.confidence *= 0.70
                                    ml_stats['ensemble_filtered'] += 1

                            # Ensemble lifetime: use for hold time if available
                            if 'lifetime' in ens_pred:
                                ens_life = float(ens_pred['lifetime'][0])
                                if ens_life > 3:
                                    ml_max_hold = max(6, int(ens_life * 0.8))
                        except Exception:
                            pass

                    # Regime-conditional adjustment
                    if regime_model is not None:
                        try:
                            reg_pred = regime_model.predict(feature_vec.reshape(1, -1))
                            regime_id = int(reg_pred['regime'][0])
                            regime_name = RegimeConditionalModel.REGIME_NAMES[regime_id]

                            # Use regime to adjust confidence
                            if regime_id == 2:  # VOLATILE
                                # In volatile regimes, require higher confidence
                                if sig.confidence < 0.55:
                                    ml_stats['regime_penalized'] += 1
                                    continue
                                # Tighten hold time in volatile
                                if ml_max_hold and ml_max_hold > 12:
                                    ml_max_hold = 12
                            elif regime_id == 0:  # TRENDING_UP
                                if sig.action == 'BUY':
                                    sig.confidence *= 1.10  # Trend-aligned boost
                                    ml_stats['regime_boosted'] += 1
                            elif regime_id == 1:  # TRENDING_DOWN
                                if sig.action == 'SELL':
                                    sig.confidence *= 1.10  # Trend-aligned boost
                                    ml_stats['regime_boosted'] += 1
                        except Exception:
                            pass

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
                        except Exception:
                            pass

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
                        except Exception:
                            pass

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
                        except Exception:
                            pass

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
                        except Exception:
                            pass

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
                        except Exception:
                            pass

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
                        except Exception:
                            pass

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
                        except Exception:
                            pass

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
                        except Exception:
                            pass

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
                        except Exception:
                            pass

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
                        except Exception:
                            pass

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
                        except Exception:
                            pass

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
                        except Exception:
                            pass

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
                        except Exception:
                            pass

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
                        except Exception:
                            pass

                    # Extreme Loser Detector: high loser probability → penalize hard
                    el_loser_prob = 0.0  # Track for stop tightening
                    if extreme_loser_model is not None:
                        try:
                            el_pred = extreme_loser_model.predict(feature_vec.reshape(1, -1))
                            el_loser_prob = float(el_pred['loser_prob'][0])

                            if el_loser_prob > 0.18:
                                sig.confidence *= 0.80
                                ml_stats['el_penalty'] += 1
                        except Exception:
                            pass

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
                        except Exception:
                            pass

                    # Immediate Stop Detector (Arch 57): tighten stop on high-risk entries
                    imm_stop_prob = 0.0
                    imm_stop_skip = False
                    if imm_stop_model is not None:
                        try:
                            is_pred = imm_stop_model.predict(feature_vec.reshape(1, -1))
                            imm_stop_prob = float(is_pred['immediate_stop_prob'][0])
                            if imm_stop_prob > 0.35:
                                ml_stats['is_skip'] += 1  # Track triggers
                        except Exception:
                            pass

                    # Breakout Stop Predictor (Arch 59): AUC 0.794
                    # Stop tightening only — confidence penalty hurts PF
                    bsp_prob = 0.0
                    if breakout_stop_model is not None and sig.signal_type == 'break':
                        try:
                            bsp_pred = breakout_stop_model.predict(feature_vec.reshape(1, -1))
                            bsp_prob = float(bsp_pred['breakout_stop_prob'][0])
                            if bsp_prob > 0.20:
                                ml_stats['bsp_tighten'] += 1
                        except Exception:
                            pass

                except Exception:
                    ml_prediction = None
                    ml_max_hold = None
                    imm_stop_skip = False
                    el_loser_prob = 0.0
                    imm_stop_prob = 0.0
                    fast_rev = 0.0
                    bsp_prob = 0.0
            else:
                ml_max_hold = None
                imm_stop_skip = False
                el_loser_prob = 0.0
                imm_stop_prob = 0.0
                fast_rev = 0.0
                bsp_prob = 0.0

            if sig.action in ('BUY', 'SELL') and sig.confidence >= min_confidence:
                # Daily circuit breaker: stop trading if down $500+ today
                if daily_breaker_active:
                    continue

                # 10AM ET skip disabled — testing without in new data window
                pass

                # Don't enter if we already have a position in the same direction
                existing_dirs = {p.direction for p in positions}
                existing_types = {p.signal_type for p in positions}
                if sig.action in existing_dirs:
                    continue  # No pyramiding
                if sig.signal_type in existing_types:
                    continue  # No double-bounce or double-break

                # Volume confirmation: skip breakouts on thin volume
                if sig.signal_type == 'break' and 'volume' in tsla.columns:
                    current_vol = tsla['volume'].iloc[bar]
                    avg_vol = tsla['volume'].iloc[max(0, bar-20):bar].mean()
                    if avg_vol > 0 and current_vol < avg_vol * 0.8:
                        continue  # Below-average volume → weak breakout

                # Quality scorer: predict win probability for this exact trade
                if quality_scorer is not None and ml_active:
                    try:
                        base_feats = feature_vec[:len(ml_feature_names)]
                        qs_features = quality_scorer.feature_names or []
                        qs_vec = np.zeros(len(qs_features), dtype=np.float32)
                        qs_vec[:len(ml_feature_names)] = base_feats
                        n_base = len(ml_feature_names)
                        qs_vec[n_base + 0] = sig.confidence
                        qs_vec[n_base + 1] = 1.0 if sig.signal_type == 'bounce' else 0.0
                        qs_vec[n_base + 2] = 1.0 if sig.action == 'BUY' else 0.0
                        qs_vec[n_base + 3] = sig.stop_pct if hasattr(sig, 'stop_pct') else 0.005
                        qs_vec[n_base + 4] = sig.tp_pct if hasattr(sig, 'tp_pct') else 0.012
                        qs_vec[n_base + 5] = 1.0

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
                    except Exception:
                        pass  # Quality scorer failure doesn't block trade

                # Position score filter: skip breakouts with weak position
                if sig.signal_type == 'break' and sig.position_score < 0.90:
                    continue


                # Enter position
                entry_price = current_price

                # Risk-normalized position sizing: scale with equity growth
                equity_scale = equity / initial_equity  # Grows as we win
                # Higher base risk for bounces (100% WR = zero stop risk)
                risk_mult = 0.025 if sig.signal_type == 'bounce' else 0.018
                base_risk = position_size * risk_mult * equity_scale
                if sig.confidence >= 0.70:
                    risk_budget = base_risk * 1.3
                elif sig.confidence >= 0.60:
                    risk_budget = base_risk * 1.15
                else:
                    risk_budget = base_risk

                # Adaptive sizing: ramp up on win streaks, halve on losing streaks
                if consecutive_wins >= 2:
                    streak_boost = min(5.0, 1.0 + 0.50 * (consecutive_wins - 1))
                    risk_budget *= streak_boost
                if consecutive_losses >= 3:
                    risk_budget *= 0.50  # Half size after 3+ consecutive losses

                # Volatility-adjusted stops: blend channel width with ATR
                # Floor at 1.5*ATR (survive noise), cap at 2.5*ATR (don't overexpose)
                current_atr = atr[bar]
                atr_floor = (1.5 * current_atr) / entry_price
                atr_mult = 3.0 if sig.signal_type == 'break' else 2.5
                atr_cap = (atr_mult * current_atr) / entry_price
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

                # High-conf breakout tightening: high-conf breakouts have
                # inverse reliability — tighter stops limit damage
                if sig.signal_type == 'break' and sig.confidence > 0.90:
                    adjusted_stop_pct *= 0.65

                # BSP stop tightening: breakout-specific, AUC 0.794
                # BSP stop tightening disabled — marginal impact (PF 9.67→9.70)
                # if bsp_prob > 0.20 and sig.signal_type == 'break':
                #     adjusted_stop_pct *= 0.65  # 35% tighter
                #     ml_stats.setdefault('bsp_stop_tighten', 0)
                #     ml_stats['bsp_stop_tighten'] += 1

                if sig.action == 'BUY':
                    stop = entry_price * (1 - adjusted_stop_pct)
                    tp = entry_price * (1 + sig.suggested_tp_pct)
                else:
                    stop = entry_price * (1 + adjusted_stop_pct)
                    tp = entry_price * (1 - sig.suggested_tp_pct)

                # Risk-normalized sizing: trade_size = risk_budget / stop_pct
                # Wider stops → smaller position, tighter stops → larger position
                trade_size = risk_budget / max(adjusted_stop_pct, 0.001)
                # Higher cap for bounces (0% stop rate, 87%+ WR)
                size_cap = position_size * (12 if sig.signal_type == 'bounce' else 3)
                trade_size = min(trade_size, size_cap)

                # Channel health penalty: high health breakouts are less decisive
                if sig.signal_type == 'break' and sig.channel_health > 0.35:
                    trade_size *= 0.60

                # Energy boost for bounces: higher energy → bigger P&L (+0.390 PnlCorr)
                if sig.signal_type == 'bounce' and sig.energy_score > 0.40:
                    trade_size *= 1.30

                # Timing boost for bounces: timing_score +0.359 PnlCorr
                if sig.signal_type == 'bounce' and sig.timing_score > 0.30:
                    trade_size *= 1.20

                # Confidence boost for bounces: confidence +0.360 PnlCorr
                if sig.signal_type == 'bounce' and sig.confidence > 0.55:
                    trade_size *= 1.15

                # Position score boost for bounces: position_score +0.354 PnlCorr
                if sig.signal_type == 'bounce' and sig.position_score > 0.95:
                    trade_size *= 1.15

                # Low channel health boost for bounces: health -0.521 PnlCorr
                # Bounces from weaker channels = bigger mean-reversion moves
                if sig.signal_type == 'bounce' and sig.channel_health < 0.65:
                    trade_size *= 1.25

                # High confluence + low confidence breakout boost
                # Confluence (+0.196 WinCorr) and conf is inverse (-0.275 WinCorr)
                if (sig.signal_type == 'break' and sig.confluence_score > 0.90
                        and sig.confidence < 0.80):
                    trade_size *= 1.25

                # Volume conviction boost: only at very high volume (2x+ avg)
                if sig.signal_type == 'break' and 'volume' in tsla.columns:
                    current_vol = tsla['volume'].iloc[bar]
                    avg_vol = tsla['volume'].iloc[max(0, bar-20):bar].mean()
                    if avg_vol > 0 and current_vol > avg_vol * 2.0:
                        trade_size *= 1.20

                # Max exposure check: total open position value < 3x equity
                total_exposure = sum(p.trade_size for p in positions)
                if total_exposure + trade_size > equity * 3:
                    continue

                # Breakout trades get longer max hold (trends persist)
                effective_max_hold = max_hold_bars * 2 if sig.signal_type == 'break' else max_hold_bars
                # ML-adjusted max hold: don't hold past predicted channel lifetime
                if ml_max_hold is not None:
                    effective_max_hold = min(effective_max_hold, ml_max_hold)

                # Get OU half-life from primary TF state
                primary_state = analysis.tf_states.get(sig.primary_tf)
                ou_hl = primary_state.ou_half_life if primary_state else 5.0

                # Entry delay disabled — all deferred trades get skipped (PF 8.49→6.61)
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
                    positions.append(OpenPosition(
                        entry_bar=bar,
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
                    ))
                    position_signals.append(sig_data)

    # Close any remaining positions
    for position in positions:
        exit_price = float(closes[-1])
        if position.direction == 'BUY':
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price
        pnl = pnl_pct * position.trade_size
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
        ))
        equity += pnl

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
    print(title)
    print(f"{'='*70}")
    print(metrics.summary())

    if ml_active:
        print(f"\n  ML Enhancement Stats:")
        print(f"    Total physics signals: {ml_stats['total_signals']}")
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
    parser.add_argument('--eval-interval', type=int, default=6, help='Bars between evaluations')
    parser.add_argument('--max-hold', type=int, default=60, help='Max bars to hold')
    parser.add_argument('--min-conf', type=float, default=0.45, help='Minimum signal confidence')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward validation')
    parser.add_argument('--ml', type=str, default=None,
                       help='Path to ML model for signal enhancement (e.g. surfer_models/gbt_model.pkl)')
    parser.add_argument('--ml-compare', action='store_true',
                       help='Run both physics-only and ML-enhanced, then compare')
    parser.add_argument('--dump-trades', type=str, default=None,
                       help='Filter to dump: stop, trail, timeout, losers, all')
    parser.add_argument('--refresh', action='store_true',
                       help='Force refresh yfinance data (clear cache)')
    args = parser.parse_args()

    if args.refresh and _SURFER_DATA_CACHE.exists():
        _SURFER_DATA_CACHE.unlink()
        print("[CACHE] Cleared")

    ml_model = None
    if args.ml:
        print(f"\nLoading ML model from {args.ml}...")
        if args.ml.endswith('.pkl'):
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
        metrics, trades, eq = run_backtest(
            days=args.days,
            eval_interval=args.eval_interval,
            max_hold_bars=args.max_hold,
            min_confidence=args.min_conf,
            ml_model=ml_model,
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
