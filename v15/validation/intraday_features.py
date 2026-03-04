"""
Intraday Feature Computation Helpers.

Pure numpy functions extracted from intraday_v14b_janfeb.py for reuse in
forward_sim_v2.py and other backtesters. Each function operates on numpy
arrays and returns numpy arrays of the same length.
"""

import numpy as np
import pandas as pd


def compute_vwap(open_arr, high_arr, low_arr, close_arr, volume_arr, dates):
    """Daily-reset VWAP and distance from VWAP (% of VWAP).

    Args:
        open_arr, high_arr, low_arr, close_arr, volume_arr: float64 arrays
        dates: array of date objects (one per bar, for daily reset)

    Returns:
        (vwap, vwap_dist) — both float64 arrays of same length
    """
    n = len(close_arr)
    vwap = np.full(n, np.nan)
    vwap_dist = np.full(n, np.nan)
    tp = (high_arr + low_arr + close_arr) / 3.0
    cum_tp_vol = cum_vol = 0.0
    prev_date = None
    for i in range(n):
        d = dates[i]
        if d != prev_date:
            cum_tp_vol = cum_vol = 0.0
            prev_date = d
        cum_tp_vol += tp[i] * volume_arr[i]
        cum_vol += volume_arr[i]
        if cum_vol > 0:
            vwap[i] = cum_tp_vol / cum_vol
            vwap_dist[i] = (close_arr[i] - vwap[i]) / vwap[i] * 100.0
    return vwap, vwap_dist


def compute_volume_ratio(volume_arr, lookback=20):
    """Volume ratio vs rolling mean."""
    n = len(volume_arr)
    vr = np.full(n, np.nan)
    rv = pd.Series(volume_arr).rolling(lookback, min_periods=lookback).mean().values
    valid = (rv > 0) & ~np.isnan(rv)
    vr[valid] = volume_arr[valid] / rv[valid]
    return vr


def compute_vwap_slope(vwap_dist, lookback=5):
    """Linear regression slope of VWAP distance over lookback bars."""
    n = len(vwap_dist)
    vs = np.full(n, np.nan)
    for i in range(lookback, n):
        seg = vwap_dist[i - lookback + 1:i + 1]
        if np.any(np.isnan(seg)):
            continue
        x = np.arange(lookback, dtype=np.float64)
        mx, my = x.mean(), seg.mean()
        vs[i] = np.sum((x - mx) * (seg - my)) / np.sum((x - mx) ** 2)
    return vs


def compute_rsi(close_arr, period=14):
    """Wilder RSI."""
    n = len(close_arr)
    rsi = np.full(n, np.nan)
    d = np.diff(close_arr)
    if len(d) < period:
        return rsi
    gain = np.maximum(d, 0)
    loss = np.maximum(-d, 0)
    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    for i in range(period, len(d)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss < 1e-10:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


def compute_rsi_slope(rsi_arr, lookback=5):
    """Linear regression slope of RSI over lookback bars."""
    n = len(rsi_arr)
    rs = np.full(n, np.nan)
    for i in range(lookback, n):
        seg = rsi_arr[i - lookback + 1:i + 1]
        if np.any(np.isnan(seg)):
            continue
        x = np.arange(lookback, dtype=np.float64)
        mx, my = x.mean(), seg.mean()
        rs[i] = np.sum((x - mx) * (seg - my)) / np.sum((x - mx) ** 2)
    return rs


def compute_spread_pct(high_arr, low_arr, close_arr):
    """(High - Low) / Close × 100."""
    sp = np.full(len(close_arr), np.nan)
    valid = close_arr > 0
    sp[valid] = (high_arr[valid] - low_arr[valid]) / close_arr[valid] * 100.0
    return sp


def compute_gap_pct(close_arr, dates):
    """Opening gap % from previous day close, forward-filled."""
    n = len(close_arr)
    gap = np.full(n, np.nan)
    prev_close = None
    prev_date = None
    for i in range(n):
        d = dates[i]
        if d != prev_date:
            if prev_close is not None and prev_close > 0:
                gap[i] = (close_arr[i] - prev_close) / prev_close * 100.0
            prev_date = d
        prev_close = close_arr[i]
    # Forward-fill
    cur_gap = np.nan
    for i in range(n):
        if not np.isnan(gap[i]):
            cur_gap = gap[i]
        gap[i] = cur_gap
    return gap


def precompute_5min_features(tsla_5min):
    """Precompute all intraday features from 5-min OHLCV DataFrame.

    Returns dict of numpy arrays aligned to tsla_5min index.
    Channel positions (cp5, daily_cp, etc.) are NOT included here —
    they come from the ChannelAnalysis at each bar.
    """
    c = tsla_5min['close'].values.astype(np.float64)
    h = tsla_5min['high'].values.astype(np.float64)
    l = tsla_5min['low'].values.astype(np.float64)
    o = tsla_5min['open'].values.astype(np.float64)
    v = tsla_5min['volume'].values.astype(np.float64)

    # Get dates for daily-reset features
    idx = tsla_5min.index
    if idx.tz is not None:
        dates = np.array([t.tz_convert('US/Eastern').date() if hasattr(t, 'tz_convert') else t.date() for t in idx])
    else:
        dates = np.array([t.date() for t in idx])

    vwap, vwap_dist = compute_vwap(o, h, l, c, v, dates)
    vol_ratio = compute_volume_ratio(v)
    vwap_slope = compute_vwap_slope(vwap_dist)
    rsi = compute_rsi(c)
    rsi_slope = compute_rsi_slope(rsi)
    spread_pct = compute_spread_pct(h, l, c)
    gap_pct = compute_gap_pct(c, dates)

    return {
        'vwap_dist': vwap_dist,
        'vol_ratio': vol_ratio,
        'vwap_slope': vwap_slope,
        'rsi_slope': rsi_slope,
        'spread_pct': spread_pct,
        'gap_pct': gap_pct,
    }
