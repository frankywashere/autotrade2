"""OE Signals_5 — evolved daily bounce signal from OpenEvolve.

Uses TSLA/SPY/VIX daily + TSLA weekly bars to detect long entry opportunities
near lower channel boundaries with ATR compression and multi-condition branches.

Exit logic is handled by the live scanner (combo exponential trail, 3% stop,
10-day max hold, flat $100K sizing).
"""

import numpy as np

from v15.core.channel import detect_channel


def _channel_at(df_slice):
    if len(df_slice) < 10:
        return None
    try:
        ch = detect_channel(df_slice)
        return ch if (ch and ch.valid) else None
    except Exception:
        return None


def _near_lower(price, ch, frac=0.25):
    if ch is None:
        return False
    lower = ch.lower_line[-1]
    upper = ch.upper_line[-1]
    w = upper - lower
    if w <= 0:
        return False
    return (price - lower) / w < frac


def _compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _evolved_signal(i, tsla, spy, vix, tw, rt):
    """Core evolved signal logic. Returns 1 (long) or 0 (no signal)."""
    if i < 35 or tw is None or len(tw) < 50:
        return 0

    closes = tsla['close'].iloc[i - 20:i + 1].values.astype(float)
    highs = tsla['high'].iloc[i - 20:i + 1].values.astype(float)
    lows = tsla['low'].iloc[i - 20:i + 1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    atr_5 = tr[-5:].mean()
    atr_20 = tr.mean()
    if atr_5 >= 0.75 * atr_20:
        return 0

    vix_now = float(vix['close'].iloc[i])

    daily_date = tsla.index[i]
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 50:
        return 0
    close_w = float(tw['close'].iloc[wk_idx])

    in_channel_lower = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if _near_lower(close_w, ch, 0.25):
                in_channel_lower = True
                break

    if not in_channel_lower:
        if 18 <= vix_now <= 50 and i >= 20:
            in_ch30 = False
            for window in (20, 30, 40, 50):
                if wk_idx >= window:
                    ch30 = _channel_at(tw.iloc[wk_idx - window:wk_idx])
                    if _near_lower(close_w, ch30, 0.30):
                        in_ch30 = True
                        break
            if in_ch30:
                t_rsi_c = float(rt.iloc[i]) if not np.isnan(rt.iloc[i]) else 50.0
                t20 = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i - 20]) - 1.0
                s20 = float(spy['close'].iloc[i]) / float(spy['close'].iloc[i - 20]) - 1.0
                if t_rsi_c < 33 and (s20 - t20) >= 0.08:
                    return 1
                if t_rsi_c < 38 and (s20 - t20) >= 0.06:
                    return 1
        return 0

    if 18 <= vix_now <= 50:
        if i >= 20:
            tsla_ret = (float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i - 20])) - 1.0
            spy_ret = (float(spy['close'].iloc[i]) / float(spy['close'].iloc[i - 20])) - 1.0
            if (spy_ret - tsla_ret) >= 0.05:
                return 1
        if i >= 3:
            c_now = float(tsla['close'].iloc[i])
            c_3d = float(tsla['close'].iloc[i - 3])
            if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
                return 1
        if i >= 20:
            spy_now = float(spy['close'].iloc[i])
            spy_20d = float(spy['close'].iloc[i - 20])
            if spy_20d > 0 and spy_now > spy_20d:
                return 1
        if i >= 5:
            spy_now = float(spy['close'].iloc[i])
            spy_5d = float(spy['close'].iloc[i - 5])
            if spy_5d > 0 and spy_now > spy_5d:
                return 1
        if i >= 10:
            c_now_a = float(tsla['close'].iloc[i])
            c_10d = float(tsla['close'].iloc[i - 10])
            if c_10d > 0 and (c_now_a - c_10d) / c_10d < -0.06:
                return 1
        if i >= 20:
            ma_20 = float(tsla['close'].iloc[i - 20:i].astype(float).mean())
            c_now_a6 = float(tsla['close'].iloc[i])
            if ma_20 > 0 and (c_now_a6 - ma_20) / ma_20 < -0.08:
                return 1

    if 15 <= vix_now <= 50:
        if i >= 35:
            close_series = tsla['close'].iloc[:i + 1].astype(float)
            ema_12 = close_series.ewm(span=12, adjust=False).mean()
            ema_26 = close_series.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            if float(macd_line.iloc[-1] - signal_line.iloc[-1]) < 0:
                return 1
        t_rsi = float(rt.iloc[i]) if not np.isnan(rt.iloc[i]) else 50.0
        if t_rsi < 40:
            return 1
        if i >= 3:
            c_now = float(tsla['close'].iloc[i])
            c_3d = float(tsla['close'].iloc[i - 3])
            if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
                return 1
        if i >= 4:
            if all(float(tsla['close'].iloc[j]) < float(tsla['close'].iloc[j - 1])
                   for j in range(i - 3, i + 1)):
                return 1
        if i >= 20:
            vol_now = float(tsla['volume'].iloc[i])
            vol_avg = float(tsla['volume'].iloc[i - 20:i].astype(float).mean())
            o_now = float(tsla['open'].iloc[i])
            c_close = float(tsla['close'].iloc[i])
            if vol_avg > 0 and vol_now > 1.5 * vol_avg and o_now > 0 and (c_close - o_now) / o_now < -0.01:
                return 1
        if i >= 5:
            vix_5h = max(float(vix['close'].iloc[i - k]) for k in range(1, 6))
            if vix_5h >= 25 and vix_now <= vix_5h * 0.90:
                return 1
        if i >= 20:
            bb_closes = tsla['close'].iloc[i - 20:i].values.astype(float)
            bb_mean = bb_closes.mean()
            bb_std = bb_closes.std()
            if bb_std > 0 and float(tsla['close'].iloc[i]) < bb_mean - 2.0 * bb_std:
                return 1

    if 10 <= vix_now < 17:
        t_rsi_c = float(rt.iloc[i]) if not np.isnan(rt.iloc[i]) else 50.0
        if t_rsi_c < 43 and i >= 20:
            t_ret_c = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i - 20]) - 1.0
            s_ret_c = float(spy['close'].iloc[i]) / float(spy['close'].iloc[i - 20]) - 1.0
            _div_c = s_ret_c - t_ret_c
            if 0.04 <= _div_c < 0.12:
                return 1

    if 10 <= vix_now < 15:
        t_rsi_ext = float(rt.iloc[i]) if not np.isnan(rt.iloc[i]) else 50.0
        if t_rsi_ext < 32:
            return 1

    return 0


def check_oe_signal(native_tf_data: dict) -> bool:
    """Check OE Signals_5 on the latest completed daily bar.

    Parameters
    ----------
    native_tf_data : dict
        {symbol: {tf: DataFrame}} as stored in DashboardState.native_tf_data

    Returns True if signal fires (long entry).
    """
    tsla_d = native_tf_data.get('TSLA', {}).get('daily')
    spy_d = native_tf_data.get('SPY', {}).get('daily')
    vix_d = native_tf_data.get('^VIX', {}).get('daily')
    tsla_w = native_tf_data.get('TSLA', {}).get('weekly')

    if tsla_d is None or spy_d is None or vix_d is None or tsla_w is None:
        return False
    if len(tsla_d) < 50 or len(spy_d) < 50 or len(vix_d) < 50 or len(tsla_w) < 50:
        return False

    # Normalize columns to lowercase
    for df in [tsla_d, spy_d, vix_d, tsla_w]:
        if any(c != c.lower() for c in df.columns):
            df.columns = [c.lower() for c in df.columns]

    # Compute TSLA RSI
    tsla_rsi = _compute_rsi(tsla_d['close'], 14)

    # Evaluate on last bar
    i = len(tsla_d) - 1
    return _evolved_signal(i, tsla_d, spy_d, vix_d, tsla_w, tsla_rsi) == 1
