"""
Phase B1 initial program: Current surfer-ml entry logic (on_bar + _compute_current_atr).

Defines on_bar() and _compute_current_atr() that the evaluator monkey-patches
onto SurferMLAlgo. This is the starting point -- the LLM evolves from here.

IMPORTANT: These functions use `self` -- they are methods that will be bound
to the SurferMLAlgo instance. Available on `self`:

  self.data — DataProvider with get_bars(tf, time, symbol='TSLA'/'SPY'/'VIX')
  self.config — AlgoConfig with:
    .algo_id ('surfer-ml')
    .eval_interval (3 — check every 3 bars = 15 min)
    .params dict: 'min_confidence', 'stop_pct', 'tp_pct', 'atr_period',
                  'breakout_stop_mult', 'ou_half_life', 'flat_sizing', 'max_hold_bars'
  self._gbt_model — GBT soft gate model (dict of Boosters: {'action': Booster, ...})
  self._el_model — Extreme Loser detector (single Booster)
  self._er_model — Extended Run predictor (single Booster)
  self._fast_rev_model — Fast Reversion detector (single Booster)
  self._el_derive — callable(full_vec) -> subset for EL model
  self._er_derive — callable(full_vec) -> subset for ER model
  self._fr_derive — callable(full_vec) -> subset for fast_rev model
  self._history_buffer — list of past feature vectors (for temporal features)
  self._feature_names — list of feature names (or None)
  self._pos_state — dict[pos_id, dict] for per-position state

Parameters:
  time — pd.Timestamp of current bar
  bar — dict with 'open', 'high', 'low', 'close', 'volume'
  open_positions — list of Position objects with:
    .direction ('long'/'short'), .signal_type ('bounce'/'break'), etc.
  context — TradeContext (passed to build_feature_vector)

Returns:
  List[Signal] where Signal(algo_id, direction, price, confidence,
                            stop_pct, tp_pct, signal_type, metadata)

ML models are FROZEN -- you cannot change the models, only how you use their outputs.
Exit logic is FROZEN -- only entry logic (on_bar + _compute_current_atr) changes.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def on_bar(self, time: pd.Timestamp, bar: dict,
           open_positions: list,
           context=None) -> list:
    """Run Channel Surfer analysis, optionally gate with GBT model.

    Replicates the original surfer_backtest signal generation path:
    - 100-bar 5-min lookback for channel detection
    - Higher TFs: completed bars only, tail(100)
    - Direct analyze_channels() call (not prepare_multi_tf_analysis)
    """
    try:
        from v15.core.channel import detect_channels_multi_window, select_best_channel
        from v15.core.channel_surfer import analyze_channels, TF_WINDOWS
    except ImportError as e:
        logger.error("SurferML.on_bar() FAILED: ImportError: %s", e)
        return []

    from v15.validation.unified_backtester.algo_base import Signal

    # Anti-pyramid: skip if already have position in same direction
    existing_dirs = {p.direction for p in open_positions}
    existing_types = {getattr(p, 'signal_type', '') for p in open_positions}

    # Get 5-min data: last 100 bars (matching original line 1443)
    df5 = self.data.get_bars('5min', time)
    if len(df5) < 20:
        return []
    df_slice = df5.tail(100)

    # Detect channels on 5-min slice (matching original lines 1450-1457)
    try:
        multi = detect_channels_multi_window(df_slice, windows=[10, 15, 20, 30, 40])
        best_ch, _ = select_best_channel(multi)
    except Exception as e:
        logger.error("SurferML channel detection failed: %s", e, exc_info=True)
        return []

    if best_ch is None or not best_ch.valid:
        return []

    # Build multi-TF channel dict (matching original lines 1460-1506)
    slice_closes = df_slice['close'].values
    channels_by_tf = {'5min': best_ch}
    prices_by_tf = {'5min': slice_closes}
    current_prices = {'5min': float(slice_closes[-1])}
    volumes_dict = {}
    if 'volume' in df_slice.columns:
        volumes_dict['5min'] = df_slice['volume'].values

    # Add higher TF channels (matching original lines 1477-1506)
    _TF_PERIOD = {
        '1h': pd.Timedelta(hours=1),
        '4h': pd.Timedelta(hours=4),
        'daily': pd.Timedelta(days=1),
    }
    for tf_label in ('1h', '4h', 'daily'):
        try:
            tf_df = self.data.get_bars(tf_label, time)
        except (ValueError, KeyError):
            continue
        if len(tf_df) == 0:
            continue
        # Only include completed bars
        tf_period = _TF_PERIOD.get(tf_label, pd.Timedelta(hours=1))
        tf_available = tf_df[tf_df.index + tf_period <= time]
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
                current_prices[tf_label] = float(tf_recent['close'].iloc[-1])
                if 'volume' in tf_recent.columns:
                    volumes_dict[tf_label] = tf_recent['volume'].values
        except Exception as e:
            logger.warning("SurferML %s channel detection failed: %s", tf_label, e)
            continue

    try:
        analysis = analyze_channels(
            channels_by_tf, prices_by_tf, current_prices,
            volumes_by_tf=volumes_dict if volumes_dict else None,
        )
    except Exception as e:
        logger.error("SurferML analyze_channels() failed: %s", e, exc_info=True)
        return []

    sig = analysis.signal
    if sig.action not in ('BUY', 'SELL'):
        return []

    conf = sig.confidence
    if conf < self.config.params.get('min_confidence', 0.01):
        return []

    # Anti-pyramid: no same-direction or same-type positions
    direction = 'long' if sig.action == 'BUY' else 'short'
    if direction in existing_dirs:
        return []
    signal_type = sig.signal_type or 'bounce'
    if signal_type in existing_types:
        return []

    # Base stop/TP from signal
    stop_pct = sig.suggested_stop_pct or self.config.params['stop_pct']
    tp_pct = sig.suggested_tp_pct or self.config.params['tp_pct']

    # ATR-adjusted stops (matching surfer_backtest lines 2199-2208)
    atr_val = self._compute_current_atr(time)
    entry_price = bar['close']
    if atr_val > 0 and entry_price > 0:
        if signal_type == 'bounce':
            atr_floor = (0.5 * atr_val) / entry_price
            atr_cap = (1.5 * atr_val) / entry_price
        else:
            atr_floor = (1.5 * atr_val) / entry_price
            atr_cap = (3.0 * atr_val) / entry_price
        stop_pct = np.clip(stop_pct, atr_floor, atr_cap)

    # Ultra-tight breakout stops (surfer_backtest line 2227)
    if signal_type == 'break':
        stop_pct *= self.config.params.get('breakout_stop_mult', 1.00)

    # TP widening for high-confidence bounces (surfer_backtest line 2244)
    if signal_type == 'bounce' and conf > 0.65:
        tp_pct *= 1.30

    # EL/ER predictions (stored in metadata for exit logic)
    el_flagged = False
    twm = 1.0
    fast_rev = False

    # Build ML feature vector if any model is loaded
    feature_vec = None
    if any(m is not None for m in [self._gbt_model, self._el_model,
                                    self._er_model, self._fast_rev_model]):
        try:
            from v15.core.signal_features import build_feature_vector
            closes_arr = df_slice['close'].values
            # Get SPY/VIX data for correlation features
            spy_df = None
            vix_df = None
            try:
                spy_daily = self.data.get_bars('daily', time, symbol='SPY')
                if len(spy_daily) > 0:
                    spy_df = spy_daily
            except Exception as e:
                logger.debug("SPY daily bars not available: %s", e)
            try:
                # Try 1-min VIX first (intraday resolution), fall back to daily
                vix_1m = self.data.get_bars('1min', time, symbol='VIX')
                if len(vix_1m) > 0:
                    vix_df = vix_1m
                else:
                    vix_daily = self.data.get_bars('daily', time, symbol='VIX')
                    if len(vix_daily) > 0:
                        vix_df = vix_daily
            except Exception as e:
                logger.debug("VIX bars not available: %s", e)
            feature_vec, _ = build_feature_vector(
                analysis=analysis,
                bar_data=bar,
                closes=closes_arr,
                spy_df=spy_df,
                vix_df=vix_df,
                tsla_index=df5.index,
                history_buffer=self._history_buffer,
                eval_interval=self.config.eval_interval,
                context=context,
                bars_df=df5,
            )
        except Exception as e:
            raise RuntimeError(
                f"SurferML feature vector build failed: {e}") from e

    # GBT soft gate: scale confidence, don't hard-skip
    # _gbt_model is a dict of Boosters: {'action': Booster, 'lifetime': Booster, ...}
    if self._gbt_model is not None and feature_vec is not None:
        try:
            fv = feature_vec.reshape(1, -1)
            action_model = self._gbt_model.get('action')
            if action_model is not None:
                probs = action_model.predict(fv)  # shape (1, 3): [HOLD, BUY, SELL]
                ml_action = int(probs[0].argmax())
                # 0=HOLD, 1=BUY, 2=SELL
                physics_action = 1 if direction == 'long' else 2
                if ml_action == 0:
                    conf *= 0.80  # ML says HOLD — penalize 20%
                elif ml_action != physics_action:
                    conf *= 0.80  # ML disagrees with direction — penalize 20%
        except Exception as e:
            raise RuntimeError(
                f"SurferML GBT prediction failed: {e}") from e

    # Extended Run predictor — single Booster with derived features
    if self._er_model is not None and feature_vec is not None:
        try:
            fv = self._er_derive(feature_vec) if self._er_derive else feature_vec
            er_prob = float(self._er_model.predict(fv.reshape(1, -1))[0])
            if er_prob > 0.70:
                twm = 2.0  # Let winners run — wider trail
            elif er_prob > 0.55:
                twm = 1.5
        except Exception as e:
            raise RuntimeError(
                f"SurferML ER model prediction failed: {e}") from e

    # Extreme Loser detector — single Booster with derived features
    if self._el_model is not None and feature_vec is not None:
        try:
            fv = self._el_derive(feature_vec) if self._el_derive else feature_vec
            el_loser_prob = float(self._el_model.predict(fv.reshape(1, -1))[0])
            if el_loser_prob > 0.18:
                el_flagged = True
                if signal_type == 'bounce':
                    conf *= 0.80  # Penalize EL bounces
        except Exception as e:
            raise RuntimeError(
                f"SurferML EL model prediction failed: {e}") from e

    # Fast Reversion (Momentum Reversal) detector — single Booster with derived features
    if self._fast_rev_model is not None and feature_vec is not None:
        try:
            fv = self._fr_derive(feature_vec) if self._fr_derive else feature_vec
            fast_rev_prob = float(self._fast_rev_model.predict(fv.reshape(1, -1))[0])
            if fast_rev_prob > 0.55:
                fast_rev = True
        except Exception as e:
            raise RuntimeError(
                f"SurferML fast_rev model prediction failed: {e}") from e

    # Re-check confidence after ML gating
    if conf < self.config.params.get('min_confidence', 0.01):
        return []

    return [Signal(
        algo_id=self.config.algo_id,
        direction=direction,
        price=bar['close'],
        confidence=conf,
        stop_pct=stop_pct,
        tp_pct=tp_pct,
        signal_type=signal_type,
        metadata={
            'el_flagged': el_flagged,
            'trail_width_mult': twm,
            'fast_reversion': fast_rev,
            'ou_half_life': self.config.params.get('ou_half_life', 5.0),
            # Signal bar high/low for seeding first exit window
            # (original includes signal bar in first eval window)
            'signal_bar_high': bar['high'],
            'signal_bar_low': bar['low'],
        },
    )]


def _compute_current_atr(self, time: pd.Timestamp) -> float:
    """Compute current ATR from 5-min bars up to given time."""
    bars = self.data.get_bars('5min', time)
    period = self.config.params.get('atr_period', 14)
    if len(bars) < period + 1:
        return 0.0
    recent = bars.tail(period + 1)
    highs = recent['high'].values
    lows = recent['low'].values
    closes = recent['close'].values
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    # Wilder EMA (matching surfer_backtest.py line 1092)
    atr = float(np.mean(tr[:period])) if len(tr) >= period else float(np.mean(tr))
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period
    return float(atr)
