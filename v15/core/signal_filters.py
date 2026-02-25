"""Unified signal filter cascade for Channel Surfer.

Three independently toggleable filters that decide WHETHER to trade:
  1. Signal Quality Gate — skip low win_prob trades (uses existing LightGBM model)
  2. Break Predictor — penalize directional mismatches on breakout signals
  3. Swing Regime — boost bounce confidence when S1041 (weekly channel support + fear) fires

Usage:
    from v15.core.signal_filters import SignalFilterCascade

    cascade = SignalFilterCascade(sq_gate_threshold=0.50, break_predictor_enabled=True,
                                  swing_regime_enabled=True)
    cascade.precompute_swing_regime(daily_tsla, daily_spy, daily_vix, weekly_tsla)

    # In backtest loop:
    should_trade, adj_conf, reasons = cascade.evaluate(
        sig, analysis, feature_vec, bar_datetime, higher_tf_data, spy_df, vix_df)
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# S1041 helper functions (extracted from swing_backtest.py)
# ---------------------------------------------------------------------------

def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute RSI from a close price array, returning the last value."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _macd_histogram(closes_series: pd.Series, i: int,
                    fast: int = 12, slow: int = 26, signal_period: int = 9) -> Optional[float]:
    """Return MACD histogram (MACD line - signal line) at bar i."""
    if i < slow + signal_period:
        return None
    closes = closes_series.iloc[:i + 1].astype(float)
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return float(macd_line.iloc[-1] - signal_line.iloc[-1])


def _atr_components(tsla: pd.DataFrame, i: int):
    """Return (atr_5, atr_20) or None if insufficient data."""
    if i < 20:
        return None
    closes = tsla['close'].iloc[i - 20:i + 1].values.astype(float)
    highs = tsla['high'].iloc[i - 20:i + 1].values.astype(float)
    lows = tsla['low'].iloc[i - 20:i + 1].values.astype(float)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]),
                   np.abs(lows[1:] - closes[:-1])))
    return tr[-5:].mean(), tr.mean()


def _channel_at(df_slice: pd.DataFrame):
    """Detect channel on a bar slice, return None on failure."""
    if len(df_slice) < 10:
        return None
    try:
        from v15.core.channel import detect_channel
        ch = detect_channel(df_slice)
        return ch if (ch and ch.valid) else None
    except Exception:
        return None


def _near_lower(price: float, ch, frac: float = 0.25) -> bool:
    """True if price is within bottom `frac` fraction of channel width."""
    if ch is None:
        return False
    lower = ch.lower_line[-1]
    upper = ch.upper_line[-1]
    w = upper - lower
    if w <= 0:
        return False
    return (price - lower) / w < frac


def _tsla_lagging_spy(tsla, spy, i: int, lookback: int = 20, lag: float = 0.05) -> bool:
    """TSLA has underperformed SPY by at least `lag` over `lookback` days."""
    if i < lookback:
        return False
    tsla_ret = (float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i - lookback])) - 1.0
    spy_ret = (float(spy['close'].iloc[i]) / float(spy['close'].iloc[i - lookback])) - 1.0
    return (spy_ret - tsla_ret) >= lag


def _check_s929_base(i: int, tsla, vix, weekly_tsla, vix_lo: float = 18) -> bool:
    """S929 base: VIX range + ATR compressed + weekly channel lower 25%."""
    if i < 20:
        return False
    # VIX check
    vix_now = float(vix['close'].iloc[i])
    if not (vix_lo <= vix_now <= 50):
        return False
    # ATR compression
    c = _atr_components(tsla, i)
    if c is None:
        return False
    atr_5, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return False
    # Weekly channel lower 25% — OR over 20/30/40/50 windows
    if weekly_tsla is None or len(weekly_tsla) < 50:
        return False
    daily_date = tsla.index[i]
    wk_idx = weekly_tsla.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 20:
        return False
    close_w = float(weekly_tsla['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(weekly_tsla.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return True
    return False


def _check_s993(i: int, tsla, spy, vix, weekly_tsla) -> bool:
    """S993: S929 base (VIX 18-50) + one of: lag, 3d-selloff, SPY-up-20d, SPY-up-5d."""
    if not _check_s929_base(i, tsla, vix, weekly_tsla, vix_lo=18):
        return False
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return True
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return True
    if i >= 20:
        spy_now = float(spy['close'].iloc[i])
        spy_20d = float(spy['close'].iloc[i - 20])
        if spy_20d > 0 and spy_now > spy_20d:
            return True
    if i >= 5:
        spy_now = float(spy['close'].iloc[i])
        spy_5d = float(spy['close'].iloc[i - 5])
        if spy_5d > 0 and spy_now > spy_5d:
            return True
    return False


def _check_s1034(i: int, tsla, spy, vix, weekly_tsla) -> bool:
    """S1034: S929 base with VIX 15-50 + (3d-selloff OR MACD<0 OR RSI<40)."""
    if not _check_s929_base(i, tsla, vix, weekly_tsla, vix_lo=15):
        return False
    # MACD histogram < 0
    hist = _macd_histogram(tsla['close'], i)
    if hist is not None and hist < 0:
        return True
    # RSI < 40
    closes = tsla['close'].values[:i + 1].astype(float)
    if _rsi(closes, 14) < 40:
        return True
    # 3d selloff
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return True
    return False


def _check_s1041(i: int, tsla, spy, vix, weekly_tsla) -> bool:
    """S1041 = S993 OR S1034 — the definitive swing champion."""
    return (_check_s993(i, tsla, spy, vix, weekly_tsla) or
            _check_s1034(i, tsla, spy, vix, weekly_tsla))


# ---------------------------------------------------------------------------
# Main filter cascade
# ---------------------------------------------------------------------------

class SignalFilterCascade:
    """Combines SQ gate + break predictor + swing regime into one filter.

    Each filter can be toggled independently. The evaluate() method returns
    (should_trade, adjusted_confidence, reasons) where reasons is a list
    of human-readable strings for logging/debugging.
    """

    def __init__(
        self,
        sq_gate_threshold: float = 0.0,       # 0 = disabled, 0.50 = skip <50% win_prob
        break_predictor_enabled: bool = False,
        swing_regime_enabled: bool = False,
        swing_boost: float = 1.2,             # Confidence multiplier when S1041 active
        break_penalty: float = 0.5,           # Confidence multiplier when break dir mismatch
    ):
        self.sq_gate_threshold = sq_gate_threshold
        self.break_predictor_enabled = break_predictor_enabled
        self.swing_regime_enabled = swing_regime_enabled
        self.swing_boost = swing_boost
        self.break_penalty = break_penalty

        # SQ model (lazy-loaded)
        self._sq_model = None
        self._sq_model_loaded = False

        # Swing regime precomputed status {date_str: bool}
        self._swing_status: Dict[str, bool] = {}

        # Per-evaluation log — records every filter decision for replay/audit
        # Each entry: {bar_datetime, action, signal_type, conf_in, conf_out, rejected, reasons}
        self.eval_log: List[dict] = []

        # Stats tracking
        self.stats = {
            'sq_rejected': 0,
            'sq_passed': 0,
            'break_penalized': 0,
            'break_passed': 0,
            'swing_boosted': 0,
            'swing_penalty': 0,
            'total_evaluated': 0,
            'total_rejected': 0,
        }

    def _load_sq_model(self):
        """Load signal quality model (once)."""
        if self._sq_model_loaded:
            return
        self._sq_model_loaded = True
        base_dir = Path(__file__).parent.parent / 'validation'
        for name in ('signal_quality_model_c10_arch2.pkl',
                      'signal_quality_model_tuned.pkl',
                      'signal_quality_model.pkl'):
            path = base_dir / name
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        self._sq_model = pickle.load(f)
                    print(f"[FILTER] Loaded SQ model: {path.name}")
                    return
                except Exception as e:
                    print(f"[FILTER] Failed to load {path.name}: {e}")
        print("[FILTER] WARNING: No signal quality model found — SQ gate disabled")

    def precompute_swing_regime(self, daily_tsla, daily_spy, daily_vix, weekly_tsla):
        """Precompute S1041 status for each trading day.

        Call this BEFORE the backtest loop with daily-frequency DataFrames.
        All DataFrames should be date-aligned (same positional index = same date).

        Args:
            daily_tsla: Daily OHLCV TSLA DataFrame
            daily_spy:  Daily OHLCV SPY DataFrame
            daily_vix:  Daily VIX DataFrame (needs 'close' column)
            weekly_tsla: Weekly OHLCV TSLA DataFrame
        """
        if not self.swing_regime_enabled:
            return

        if daily_tsla is None or daily_spy is None or daily_vix is None or weekly_tsla is None:
            print("[FILTER] WARNING: Missing data for swing regime — disabled")
            self.swing_regime_enabled = False
            return

        # Align all DataFrames to common dates
        def _to_date_index(df):
            idx = df.index
            if hasattr(idx, 'tz') and idx.tz is not None:
                idx = idx.tz_convert(None)
            return idx.normalize()

        tsla_dates = _to_date_index(daily_tsla)
        spy_dates = _to_date_index(daily_spy)
        vix_dates = _to_date_index(daily_vix)
        common = tsla_dates.intersection(spy_dates).intersection(vix_dates)

        if len(common) < 100:
            print(f"[FILTER] WARNING: Only {len(common)} common dates — swing regime disabled")
            self.swing_regime_enabled = False
            return

        # Reindex to common dates
        tsla_a = daily_tsla.loc[daily_tsla.index.isin(common) |
                                 _to_date_index(daily_tsla).isin(common)].copy()
        spy_a = daily_spy.loc[daily_spy.index.isin(common) |
                               _to_date_index(daily_spy).isin(common)].copy()
        vix_a = daily_vix.loc[daily_vix.index.isin(common) |
                               _to_date_index(daily_vix).isin(common)].copy()

        # Ensure same length by taking min
        min_len = min(len(tsla_a), len(spy_a), len(vix_a))
        tsla_a = tsla_a.iloc[:min_len]
        spy_a = spy_a.iloc[:min_len]
        vix_a = vix_a.iloc[:min_len]

        n = len(tsla_a)
        count = 0
        for i in range(50, n):
            dt = tsla_a.index[i]
            date_str = str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10]
            try:
                active = _check_s1041(i, tsla_a, spy_a, vix_a, weekly_tsla)
            except Exception:
                active = False
            self._swing_status[date_str] = active
            if active:
                count += 1

        print(f"[FILTER] Swing regime: {count} S1041 days out of {n} "
              f"({count / max(n, 1) * 100:.1f}%)")

    def evaluate(
        self,
        sig,                    # Signal object (.action, .confidence, .signal_type, etc.)
        analysis,               # ChannelAnalysis
        feature_vec,            # Base ~177 feature vector (or None)
        bar_datetime,           # datetime of current bar
        higher_tf_data=None,    # dict with 'daily' TSLA etc.
        spy_df=None,            # SPY data (5-min or daily)
        vix_df=None,            # VIX data (daily)
    ) -> Tuple[bool, float, List[str]]:
        """Evaluate all filters for a signal.

        Returns:
            (should_trade, adjusted_confidence, reasons)
        """
        self.stats['total_evaluated'] += 1
        confidence = sig.confidence
        reasons = []

        # --- Filter 1: Signal Quality Gate ---
        if self.sq_gate_threshold > 0 and feature_vec is not None:
            self._load_sq_model()
            if self._sq_model is not None:
                try:
                    from v15.validation.signal_quality_model import _append_signal_meta

                    class _SigProxy:
                        pass
                    proxy = _SigProxy()
                    proxy.signal_type = getattr(sig, 'signal_type', 'bounce')
                    proxy.direction = sig.action
                    proxy.stop_pct = getattr(sig, 'suggested_stop_pct', 0.005)
                    proxy.tp_pct = getattr(sig, 'suggested_tp_pct', 0.012)
                    proxy.primary_tf = getattr(sig, 'primary_tf', '')
                    proxy.entry_time = str(bar_datetime)

                    sig_data = {
                        'position_score': getattr(sig, 'position_score', 0.0),
                        'energy_score': getattr(sig, 'energy_score', 0.0),
                        'entropy_score': getattr(sig, 'entropy_score', 0.0),
                        'confluence_score': getattr(sig, 'confluence_score', 0.0),
                        'timing_score': getattr(sig, 'timing_score', 0.0),
                        'channel_health': getattr(sig, 'channel_health', 0.0),
                        'confidence': sig.confidence,
                    }
                    full_vec = _append_signal_meta(feature_vec, proxy, sig_data, extended=True)

                    pred = self._sq_model.predict(full_vec)
                    win_prob = float(pred.get('win_prob', 0.5))

                    if win_prob < self.sq_gate_threshold:
                        self.stats['sq_rejected'] += 1
                        self.stats['total_rejected'] += 1
                        reasons.append(f"SQ_REJECT({win_prob:.2f}<{self.sq_gate_threshold:.2f})")
                        self.eval_log.append({
                            'bar_datetime': bar_datetime, 'action': sig.action,
                            'signal_type': getattr(sig, 'signal_type', 'bounce'),
                            'conf_in': sig.confidence, 'conf_out': confidence,
                            'rejected': True, 'reasons': list(reasons),
                        })
                        return False, confidence, reasons
                    self.stats['sq_passed'] += 1
                    reasons.append(f"SQ_PASS({win_prob:.2f})")
                except Exception as e:
                    reasons.append(f"SQ_ERROR({e})")

        # --- Filter 2: Break Predictor Direction Check ---
        if self.break_predictor_enabled and getattr(sig, 'signal_type', 'bounce') == 'break':
            try:
                from v15.core.break_predictor import predict_break, extract_break_features

                # Build native_tf_data dict for extract_break_features
                native_tf_data = {}
                if higher_tf_data and 'daily' in higher_tf_data:
                    native_tf_data['TSLA'] = {'daily': higher_tf_data['daily']}

                bp_features = extract_break_features(
                    analysis, native_tf_data,
                    current_spy=spy_df, current_vix=vix_df,
                )

                if bp_features is not None:
                    bp_result = predict_break(bp_features)
                    pred_dir = bp_result['direction']   # 'UP' or 'DOWN'
                    sig_dir = 'UP' if sig.action == 'BUY' else 'DOWN'

                    if pred_dir != sig_dir:
                        confidence *= self.break_penalty
                        self.stats['break_penalized'] += 1
                        reasons.append(f"BP_PENALTY(pred={pred_dir},sig={sig_dir},"
                                       f"conf*={self.break_penalty})")
                    else:
                        self.stats['break_passed'] += 1
                        reasons.append(f"BP_AGREE({pred_dir})")
            except Exception as e:
                reasons.append(f"BP_ERROR({e})")

        # --- Filter 3: Swing Regime Overlay (bounces only) ---
        if self.swing_regime_enabled and getattr(sig, 'signal_type', 'bounce') == 'bounce':
            dt = bar_datetime
            date_str = str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10]
            s1041_active = self._swing_status.get(date_str, False)

            if s1041_active:
                confidence *= self.swing_boost
                self.stats['swing_boosted'] += 1
                reasons.append(f"SWING_BOOST(S1041,conf*={self.swing_boost})")
            else:
                # Mild penalty on bounces when TSLA above 50-day MA (strong bull)
                if higher_tf_data and 'daily' in higher_tf_data:
                    daily = higher_tf_data['daily']
                    if len(daily) >= 50:
                        # Find the current day's daily bar
                        dt_naive = dt.tz_localize(None) if hasattr(dt, 'tz_localize') and dt.tzinfo else dt
                        try:
                            daily_idx = daily.index
                            if hasattr(daily_idx, 'tz') and daily_idx.tz is not None:
                                daily_idx_naive = daily_idx.tz_convert(None)
                            else:
                                daily_idx_naive = daily_idx
                            mask = daily_idx_naive <= dt_naive
                            if mask.any():
                                last_daily = daily.loc[mask].iloc[-1]
                                current_close = float(last_daily['close'])
                                ma50 = float(daily.loc[mask]['close'].iloc[-50:].mean())
                                if current_close > ma50:
                                    confidence *= 0.95
                                    self.stats['swing_penalty'] += 1
                                    reasons.append("SWING_BULL_PENALTY(above_50dMA,conf*=0.95)")
                        except Exception:
                            pass  # Data alignment issue — skip penalty

        self.eval_log.append({
            'bar_datetime': bar_datetime, 'action': sig.action,
            'signal_type': getattr(sig, 'signal_type', 'bounce'),
            'conf_in': sig.confidence, 'conf_out': confidence,
            'rejected': False, 'reasons': list(reasons),
        })
        return True, confidence, reasons

    def summary(self) -> str:
        """Return a summary string of filter statistics."""
        s = self.stats
        lines = [
            f"  Total evaluated: {s['total_evaluated']}",
            f"  Total rejected:  {s['total_rejected']}",
        ]
        if self.sq_gate_threshold > 0:
            lines.append(f"  SQ gate ({self.sq_gate_threshold:.0%}): "
                         f"{s['sq_rejected']} rejected, {s['sq_passed']} passed")
        if self.break_predictor_enabled:
            lines.append(f"  Break predictor: {s['break_penalized']} penalized, "
                         f"{s['break_passed']} agreed")
        if self.swing_regime_enabled:
            lines.append(f"  Swing regime: {s['swing_boosted']} boosted, "
                         f"{s['swing_penalty']} bull-penalized")
        return '\n'.join(lines)
