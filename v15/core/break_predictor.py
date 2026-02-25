"""Channel break direction predictor.

Evolved by OpenEvolve Phase 4 (Feb 2026) from daily TSLA/SPY/VIX channel data.
Training: 330 labeled boundary samples (2015-2024), 80% break base rate.
Performance: break_accuracy=82%, direction_accuracy=96%, combined_score=70.14

Key insight: channels almost always break (80% base rate), so the valuable
prediction is DIRECTION, not whether a break occurs at all.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Evolved predictor (Phase 4 best_program.py — verbatim, OpenEvolve output)
# ---------------------------------------------------------------------------

def predict_break(features: dict) -> dict:
    """Predict channel break direction using evolved heuristics.

    Args:
        features: dict with keys matching Phase 4 training feature set.
    Returns:
        {'will_break': bool, 'direction': 'UP'|'DOWN', 'confidence': float}
    """
    position = features['position']
    rsi = features['rsi']
    vix = features['vix']
    atr_ratio = features['atr_ratio']
    ret_1d = features['ret_1d']
    ret_3d = features['ret_3d']
    ret_5d = features['ret_5d']
    channel_slope = features['channel_slope']
    near_upper = features['near_upper']
    near_lower = features['near_lower']
    bounce_count = features['bounce_count']
    volume_ratio = features['volume_ratio']
    r_squared = features['r_squared']
    macd_hist = features['macd_hist']
    lag_5d = features['lag_5d']
    vix_5d_change = features['vix_5d_change']
    rsi_spy = features['rsi_spy']
    bars_since_touch = features['bars_since_touch']

    # Multi-signal direction score: positive = UP, negative = DOWN
    dir_score = 0.0
    if near_upper:
        dir_score += 2.0
    elif near_lower:
        dir_score -= 2.0
    dir_score += channel_slope * 2.5
    dir_score += (position - 0.5) * 1.5
    dir_score += ret_1d * 8.0 + ret_3d * 5.0 + ret_5d * 2.5
    if rsi > 65:
        dir_score += 0.5
    elif rsi > 55:
        dir_score += 0.2
    elif rsi < 35:
        dir_score -= 0.5
    elif rsi < 45:
        dir_score -= 0.2
    dir_score += 0.35 if macd_hist > 0 else -0.35
    dir_score -= lag_5d * 1.5
    if vix_5d_change > 2.0:
        dir_score -= 0.3
    elif vix_5d_change < -2.0:
        dir_score += 0.2
    dir_score += (rsi - rsi_spy) * 0.02

    will_break = True
    direction = 'UP' if dir_score >= 0 else 'DOWN'
    confidence = 0.55

    if near_upper and ret_3d > 0.03 and rsi > 65:
        will_break = True
        direction = 'UP'
        confidence = 0.6
    elif near_lower and ret_3d < -0.03 and rsi < 35:
        will_break = True
        direction = 'DOWN'
        confidence = 0.6

    if vix > 25 and atr_ratio < 0.7:
        will_break = True
        if channel_slope > 0:
            direction = 'UP'
        else:
            direction = 'DOWN'
        confidence = 0.55

    if bounce_count <= 2 and r_squared < 0.5:
        will_break = True
        confidence = 0.55
        if position > 0.5:
            direction = 'UP'
        else:
            direction = 'DOWN'

    if volume_ratio > 1.5 and (near_upper or near_lower):
        will_break = True
        confidence = min(confidence + 0.1, 0.8)
        if near_upper:
            direction = 'UP'
        else:
            direction = 'DOWN'

    if bounce_count >= 6 and r_squared > 0.8 and vix < 22:
        will_break = False
        confidence = 0.6
    elif bars_since_touch > 15 and abs(position - 0.5) < 0.3 and vix < 20:
        will_break = False
        confidence = 0.55

    return {'will_break': will_break, 'direction': direction, 'confidence': confidence}


# ---------------------------------------------------------------------------
# Feature extraction from live dashboard data
# ---------------------------------------------------------------------------

def _rsi(series: np.ndarray, period: int = 14) -> float:
    """Compute RSI-14 from a price array, returning the last value."""
    if len(series) < period + 1:
        return 50.0
    deltas = np.diff(series)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def extract_break_features(analysis, native_tf_data, current_spy=None, current_vix=None) -> dict | None:
    """Build the Phase 4 feature dict from live dashboard data.

    Tries daily channel state first; falls back to 4h or 1h if daily is invalid.
    Returns None if insufficient data to build features.
    """
    # --- Channel state (prefer daily; fall back to 4h / 1h / weekly) ---
    state = None
    for tf in ('daily', '4h', '1h', 'weekly'):
        s = analysis.tf_states.get(tf) if analysis and analysis.tf_states else None
        if s is not None and s.valid:
            state = s
            break
    if state is None:
        return None

    # --- Daily TSLA OHLCV ---
    daily_tsla = None
    if native_tf_data:
        daily_tsla = native_tf_data.get('TSLA', {}).get('daily')
        if daily_tsla is None or (hasattr(daily_tsla, 'empty') and daily_tsla.empty):
            daily_tsla = native_tf_data.get('TSLA', {}).get('1d')
    if daily_tsla is None or len(daily_tsla) < 30:
        return None

    closes = daily_tsla['close'].values.astype(float)
    highs  = daily_tsla['high'].values.astype(float)
    lows   = daily_tsla['low'].values.astype(float)
    vols   = daily_tsla['volume'].values.astype(float)
    n = len(closes)
    i = n - 1  # current bar index

    # --- Returns ---
    def _ret(bars_back: int) -> float:
        if i < bars_back or closes[i - bars_back] == 0:
            return 0.0
        return float(closes[i] / closes[i - bars_back] - 1.0)

    ret_1d = _ret(1)
    ret_3d = _ret(3)
    ret_5d = _ret(5)

    # --- ATR ratio ---
    if i >= 20:
        tr_arr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]),
                       np.abs(lows[1:] - closes[:-1])),
        )
        atr5  = tr_arr[-5:].mean() if len(tr_arr) >= 5 else 1.0
        atr20 = tr_arr[-20:].mean() if len(tr_arr) >= 20 else 1.0
        atr_ratio = float(atr5 / atr20) if atr20 > 0 else 1.0
    else:
        atr_ratio = 1.0

    # --- Volume ratio ---
    vol_avg20 = vols[max(0, i - 20):i].mean() if i >= 1 else 1.0
    volume_ratio = float(vols[i] / vol_avg20) if vol_avg20 > 0 else 1.0

    # --- MACD histogram ---
    close_s = daily_tsla['close'].astype(float)
    ema12 = close_s.ewm(span=12, adjust=False).mean()
    ema26 = close_s.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    sig_line  = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = float((macd_line - sig_line).iloc[-1])

    # --- TSLA RSI ---
    rsi_tsla = _rsi(closes[-30:])

    # --- SPY features ---
    rsi_spy   = 50.0
    spy_ret5d = 0.0
    daily_spy = None
    if native_tf_data:
        daily_spy = native_tf_data.get('SPY', {}).get('daily')
        if daily_spy is None or (hasattr(daily_spy, 'empty') and daily_spy.empty):
            daily_spy = native_tf_data.get('SPY', {}).get('1d')
    if daily_spy is None and current_spy is not None and len(current_spy) > 20:
        # Resample 5-min SPY to daily
        try:
            daily_spy = current_spy.resample('D').agg(
                {'open': 'first', 'high': 'max', 'low': 'min',
                 'close': 'last', 'volume': 'sum'}
            ).dropna()
        except Exception:
            pass
    if daily_spy is not None and len(daily_spy) >= 20:
        spy_closes = daily_spy['close'].values.astype(float)
        rsi_spy = _rsi(spy_closes[-30:])
        if len(spy_closes) >= 6 and spy_closes[-6] != 0:
            spy_ret5d = float(spy_closes[-1] / spy_closes[-6] - 1.0)

    lag_5d = spy_ret5d - ret_5d  # SPY outperforming TSLA (positive = TSLA lagging)

    # --- VIX ---
    vix_level    = 20.0
    vix_5d_change = 0.0
    if current_vix is not None and len(current_vix) >= 1:
        vix_closes = current_vix['close'].values.astype(float)
        vix_level = float(vix_closes[-1])
        bars_per_day = max(1, len(vix_closes) // max(1, len(daily_tsla)))
        lookback = min(5 * bars_per_day, len(vix_closes) - 1)
        if lookback > 0:
            vix_5d_change = float(vix_closes[-1] - vix_closes[-1 - lookback])

    # --- Channel-derived fields ---
    position     = float(getattr(state, 'position_pct', 0.5))
    channel_slope = float(getattr(state, 'slope_pct', 0.0))   # % per bar, normalized
    width_pct    = float(getattr(state, 'width_pct', 0.05))
    r_squared    = float(getattr(state, 'r_squared', 0.5))
    bounce_count = int(getattr(state, 'bounce_count', 3))
    # bars_to_next_bounce is a forward estimate; use as proxy for bars_since_touch
    bars_since_touch = max(1, int(getattr(state, 'bars_to_next_bounce', 5)))

    near_upper = position > 0.80
    near_lower = position < 0.20

    # Upper/lower bounds: derive from current price and width_pct
    close_now   = float(closes[i])
    channel_half = close_now * width_pct / 2.0
    upper_bound = close_now + channel_half * (1.0 - position)  # rough estimate
    lower_bound = close_now - channel_half * position

    return {
        'position':          position,
        'channel_slope':     channel_slope,
        'channel_width_pct': width_pct,
        'r_squared':         r_squared,
        'bounce_count':      bounce_count,
        'bars_since_touch':  bars_since_touch,
        'rsi':               rsi_tsla,
        'rsi_spy':           rsi_spy,
        'vix':               vix_level,
        'vix_5d_change':     vix_5d_change,
        'atr_ratio':         atr_ratio,
        'volume_ratio':      volume_ratio,
        'ret_1d':            ret_1d,
        'ret_3d':            ret_3d,
        'ret_5d':            ret_5d,
        'spy_ret_5d':        spy_ret5d,
        'lag_5d':            lag_5d,
        'near_upper':        near_upper,
        'near_lower':        near_lower,
        'close':             close_now,
        'upper_bound':       upper_bound,
        'lower_bound':       lower_bound,
        'macd_hist':         macd_hist,
    }
