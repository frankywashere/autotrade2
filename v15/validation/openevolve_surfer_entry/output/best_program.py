# Changes from best (score=2343530):
# 1. GBT strong-agreement boost: +8% conf when GBT prob > 0.55 for our direction (reward high-confidence agreement)
# 2. ER-guided TP tightening: TP *= 0.90 when ER < 0.20 (non-extended runs → take profits faster → better WR)
# 3. Break signal volume confirmation: 0.90x conf penalty if volume < 80% of recent 10-bar avg (reduce false breaks)
# 4. EL soft-zone TP reduction: TP *= 0.90 when EL in 0.10-0.15 range (lock in profits on elevated-risk trades)

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def on_bar(self, time: pd.Timestamp, bar: dict,
           open_positions: list,
           context=None) -> list:
    """Run Channel Surfer analysis, optionally gate with GBT model."""
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

    # Time-of-day filter: skip first 30 min (9:30-10:00 ET) and last 15 min (15:45-16:00 ET)
    try:
        if hasattr(time, 'tzinfo') and time.tzinfo is not None:
            bar_et = time.tz_convert('America/New_York')
        else:
            bar_et = pd.Timestamp(time).tz_localize('UTC').tz_convert('America/New_York')
        bar_mins = bar_et.hour * 60 + bar_et.minute
        # 10:00 ET = 600 min, 15:45 ET = 945 min
        if bar_mins < 600 or bar_mins >= 945:
            return []
    except Exception:
        pass  # If timezone conversion fails, don't block trading

    # Get 5-min data: last 100 bars
    df5 = self.data.get_bars('5min', time)
    if len(df5) < 20:
        return []
    df_slice = df5.tail(100)

    # Detect channels on 5-min slice
    try:
        multi = detect_channels_multi_window(df_slice, windows=[10, 15, 20, 30, 40])
        best_ch, ch_score = select_best_channel(multi)
    except Exception as e:
        logger.error("SurferML channel detection failed: %s", e, exc_info=True)
        return []

    if best_ch is None or not best_ch.valid:
        return []

    # Channel quality gate
    if ch_score is not None and ch_score < 0.25:
        return []

    # Build multi-TF channel dict
    slice_closes = df_slice['close'].values
    channels_by_tf = {'5min': best_ch}
    prices_by_tf = {'5min': slice_closes}
    current_prices = {'5min': float(slice_closes[-1])}
    volumes_dict = {}
    if 'volume' in df_slice.columns:
        volumes_dict['5min'] = df_slice['volume'].values

    # Add higher TF channels
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

    # Break signals require higher confidence floor — raised to 0.06
    if signal_type == 'break' and conf < 0.06:
        return []

    # 1h trend alignment: skip bounce signals against 1h trend (±0.15% threshold)
    if signal_type == 'bounce' and '1h' in prices_by_tf:
        try:
            tf1h_c = prices_by_tf['1h']
            if len(tf1h_c) >= 10:
                mid = len(tf1h_c) // 2
                older_avg = np.mean(tf1h_c[:mid])
                recent_avg = np.mean(tf1h_c[mid:])
                if recent_avg > older_avg * 1.0015:
                    tf1h_trend = 'up'
                elif recent_avg < older_avg * 0.9985:
                    tf1h_trend = 'down'
                else:
                    tf1h_trend = 'neutral'

                if direction == 'long' and tf1h_trend == 'down':
                    return []
                elif direction == 'short' and tf1h_trend == 'up':
                    return []
        except Exception:
            pass

    # 4h trend alignment: skip bounce signals that go against the 4h trend direction
    if signal_type == 'bounce' and '4h' in prices_by_tf:
        try:
            tf4h_c = prices_by_tf['4h']
            if len(tf4h_c) >= 10:
                mid = len(tf4h_c) // 2
                older_avg = np.mean(tf4h_c[:mid])
                recent_avg = np.mean(tf4h_c[mid:])
                if recent_avg > older_avg * 1.001:
                    tf4h_trend = 'up'
                elif recent_avg < older_avg * 0.999:
                    tf4h_trend = 'down'
                else:
                    tf4h_trend = 'neutral'

                if direction == 'long' and tf4h_trend == 'down':
                    return []
                elif direction == 'short' and tf4h_trend == 'up':
                    return []
        except Exception:
            pass

    # Daily trend alignment: also skip bounce signals against daily trend (wider threshold ±0.5%)
    if signal_type == 'bounce' and 'daily' in prices_by_tf:
        try:
            daily_c = prices_by_tf['daily']
            if len(daily_c) >= 10:
                mid = len(daily_c) // 2
                older_avg = np.mean(daily_c[:mid])
                recent_avg = np.mean(daily_c[mid:])
                if recent_avg > older_avg * 1.005:
                    daily_trend = 'up'
                elif recent_avg < older_avg * 0.995:
                    daily_trend = 'down'
                else:
                    daily_trend = 'neutral'

                if direction == 'long' and daily_trend == 'down':
                    return []
                elif direction == 'short' and daily_trend == 'up':
                    return []
        except Exception:
            pass

    # Base stop/TP from signal
    stop_pct = sig.suggested_stop_pct or self.config.params['stop_pct']
    tp_pct = sig.suggested_tp_pct or self.config.params['tp_pct']

    # ATR-adjusted stops
    atr_val = self._compute_current_atr(time)
    entry_price = bar['close']
    if atr_val > 0 and entry_price > 0:
        if signal_type == 'bounce':
            atr_floor = (0.5 * atr_val) / entry_price
            atr_cap = (1.2 * atr_val) / entry_price
        else:
            atr_floor = (1.5 * atr_val) / entry_price
            atr_cap = (3.0 * atr_val) / entry_price
        stop_pct = np.clip(stop_pct, atr_floor, atr_cap)

    # Ultra-tight breakout stops
    if signal_type == 'break':
        stop_pct *= self.config.params.get('breakout_stop_mult', 1.00)

    # TP widening for high-confidence bounces
    if signal_type == 'bounce':
        if conf > 0.65:
            tp_pct *= 1.38
        elif conf > 0.55:
            tp_pct *= 1.15

    # SPY macro alignment: unified filter for both break and bounce signals
    spy_5d_return = None
    try:
        spy_daily = self.data.get_bars('daily', time, symbol='SPY')
        if len(spy_daily) >= 5:
            spy_closes = spy_daily['close'].values[-5:]
            spy_5d_return = (spy_closes[-1] - spy_closes[0]) / spy_closes[0]
    except Exception as e:
        logger.debug("SPY macro filter failed: %s", e)

    if spy_5d_return is not None:
        if signal_type == 'break':
            if direction == 'long' and spy_5d_return < -0.015:
                return []
            if direction == 'short' and spy_5d_return > 0.015:
                return []
        elif signal_type == 'bounce':
            if direction == 'long' and spy_5d_return < -0.02:
                conf *= 0.85
            elif direction == 'short' and spy_5d_return > 0.02:
                conf *= 0.85

    # EL/ER predictions
    el_flagged = False
    twm = 1.0
    fast_rev = False
    el_prob = 0.0
    er_prob = 0.0

    # Build ML feature vector if any model is loaded
    feature_vec = None
    vix_df = None
    if any(m is not None for m in [self._gbt_model, self._el_model,
                                    self._er_model, self._fast_rev_model]):
        try:
            from v15.core.signal_features import build_feature_vector
            closes_arr = df_slice['close'].values
            spy_df = None
            try:
                spy_daily_fv = self.data.get_bars('daily', time, symbol='SPY')
                if len(spy_daily_fv) > 0:
                    spy_df = spy_daily_fv
            except Exception as e:
                logger.debug("SPY daily bars not available: %s", e)
            try:
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

    # VIX regime filter: skip extreme volatility, require higher conf in high-VIX
    if vix_df is not None and len(vix_df) > 0:
        try:
            current_vix = float(vix_df['close'].iloc[-1])
            if current_vix > 50:
                return []  # Extreme volatility: skip all trades
            if current_vix > 35 and signal_type == 'break':
                if conf < 0.12:
                    return []
        except Exception:
            pass

    # GBT soft gate: HOLD penalty 0.75, direction disagreement 0.65
    # NEW: strong-agreement boost +8% when GBT prob > 0.55 for our direction
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
                    conf *= 0.75  # ML says HOLD — 25% penalty
                elif ml_action != physics_action:
                    conf *= 0.65  # ML disagrees direction — 35% penalty
                else:
                    # ML agrees — boost if strongly confident (prob > 0.55 for our direction)
                    dir_prob = float(probs[0][physics_action])
                    if dir_prob > 0.55:
                        conf *= 1.08  # Strong ML agreement boost
        except Exception as e:
            raise RuntimeError(
                f"SurferML GBT prediction failed: {e}") from e

    # Extended Run predictor — with additional super-super-tier for highest ER confidence
    # NEW: ER < 0.20 → tighten TP to capture quicker profits (improve WR on non-extended trades)
    if self._er_model is not None and feature_vec is not None:
        try:
            fv = self._er_derive(feature_vec) if self._er_derive else feature_vec
            er_prob = float(self._er_model.predict(fv.reshape(1, -1))[0])
            if er_prob > 0.80:
                twm = 3.0    # Ultra-extended run — widest trail
                tp_pct *= 1.70  # Most aggressive TP for very highest ER confidence
            elif er_prob > 0.75:
                twm = 2.5    # Super-extended run
                tp_pct *= 1.55
            elif er_prob > 0.70:
                twm = 2.0    # Let winners run — wider trail
                tp_pct *= 1.40
            elif er_prob > 0.65:
                twm = 1.5
                tp_pct *= 1.20
            elif er_prob > 0.55:
                twm = 1.3
            elif er_prob < 0.20:
                # Very low ER: tighten TP to capture profits faster → better WR
                tp_pct *= 0.90
        except Exception as e:
            raise RuntimeError(
                f"SurferML ER model prediction failed: {e}") from e

    # Extreme Loser detector — hard block at 0.30
    # NEW: EL in 0.10-0.15 soft zone → also tighten TP 0.90x (lock in profits on elevated-risk trades)
    if self._el_model is not None and feature_vec is not None:
        try:
            fv = self._el_derive(feature_vec) if self._el_derive else feature_vec
            el_loser_prob = float(self._el_model.predict(fv.reshape(1, -1))[0])
            el_prob = el_loser_prob
            if el_loser_prob > 0.30:
                return []  # Hard block
            if el_loser_prob > 0.15:
                el_flagged = True
                if signal_type == 'bounce':
                    conf *= 0.80
            elif el_loser_prob > 0.10:
                # Soft elevated risk: tighten TP to improve capture rate
                tp_pct *= 0.90
            if el_loser_prob > 0.25:
                stop_pct *= 0.85
        except Exception as e:
            raise RuntimeError(
                f"SurferML EL model prediction failed: {e}") from e

    # Fast Reversion detector — penalizes break signals
    if self._fast_rev_model is not None and feature_vec is not None:
        try:
            fv = self._fr_derive(feature_vec) if self._fr_derive else feature_vec
            fast_rev_prob = float(self._fast_rev_model.predict(fv.reshape(1, -1))[0])
            if fast_rev_prob > 0.55:
                fast_rev = True
                if signal_type == 'break':
                    conf *= 0.80
        except Exception as e:
            raise RuntimeError(
                f"SurferML fast_rev model prediction failed: {e}") from e

    # Break signal volume confirmation: low-volume breaks get confidence penalty
    # NEW: if current bar volume < 80% of recent 10-bar avg, penalize break signals
    if signal_type == 'break' and 'volume' in bar and 'volume' in df_slice.columns:
        try:
            recent_vol_avg = float(np.mean(df_slice['volume'].values[-10:]))
            current_vol = float(bar['volume'])
            if recent_vol_avg > 0 and current_vol < recent_vol_avg * 0.80:
                conf *= 0.90  # Low-volume break — likely false breakout
        except Exception:
            pass

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