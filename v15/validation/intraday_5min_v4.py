#!/usr/bin/env python3
"""
Intraday 5-Min TSLA Backtester V4

Key changes from V2:
1. 1min + 3min TFs for microstructure entry timing
2. Intraday VWAP position (mean reversion signal)
3. Adaptive trail: tight bar 1, wider after profit develops
4. Relaxed CrossTF_Div + multiple entry modes (union of signals)
5. Bar pattern detection (reversal bars on 1min/3min)
6. Time-of-day weighting (first/last hour alpha)
7. Microstructure momentum: 1min turn while 5min channel low
"""
import os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy.stats import norm
import datetime as dt

# ================================================================
# Configuration
# ================================================================
CAPITAL = 100_000.0
SLIPPAGE_PCT = 0.0002
COMM_PER_SHARE = 0.005

TRAIN_END = pd.Timestamp('2021-12-31')
TEST_END  = pd.Timestamp('2025-12-31')

MKT_OPEN  = dt.time(9, 30)
MKT_CLOSE = dt.time(16, 0)

# ================================================================
# Data Structures
# ================================================================
@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    confidence: float
    shares: int
    pnl: float
    hold_bars: int
    exit_reason: str
    signal_name: str

# ================================================================
# Data Loading
# ================================================================
def load_1min(path=None):
    if path is None:
        for p in ['data/TSLAMin.txt', r'C:\AI\x14\data\TSLAMin.txt',
                   os.path.expanduser('~/Desktop/Coding/x14/data/TSLAMin.txt')]:
            if os.path.exists(p):
                path = p
                break
    if path is None:
        raise FileNotFoundError("TSLAMin.txt not found")
    print(f"Loading 1-min data from {path}...")
    t0 = time.time()
    df = pd.read_csv(path, sep=';',
                     names=['datetime','open','high','low','close','volume'],
                     parse_dates=['datetime'],
                     date_format='%Y%m%d %H%M%S')
    df = df.set_index('datetime').sort_index()
    times = df.index.time
    mask = (times >= MKT_OPEN) & (times < MKT_CLOSE)
    df = df[mask].copy()
    print(f"  Loaded {len(df):,} bars in {time.time()-t0:.1f}s")
    return df

def resample_ohlcv(df, rule):
    return df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

# ================================================================
# Feature Computation (Vectorized)
# ================================================================
def channel_position(close_arr, window=60):
    """O(n) rolling linreg channel position."""
    n = len(close_arr)
    close = close_arr.astype(np.float64)
    w = window

    sum_x = w * (w - 1) / 2.0
    sum_x2 = (w - 1) * w * (2 * w - 1) / 6.0
    denom = w * sum_x2 - sum_x ** 2

    cs_y = np.cumsum(close)
    sum_y = np.full(n, np.nan)
    sum_y[w-1] = cs_y[w-1]
    if n > w:
        sum_y[w:] = cs_y[w:] - cs_y[:n-w]

    idx = np.arange(n, dtype=np.float64)
    cs_wy = np.cumsum(idx * close)
    sum_xy = np.full(n, np.nan)
    sum_xy[w-1] = cs_wy[w-1]
    if n > w:
        start_idx = np.arange(w, n, dtype=np.float64) - w + 1
        sum_xy[w:] = (cs_wy[w:] - cs_wy[:n-w]) - start_idx * sum_y[w:]

    slope = (w * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / w
    fitted_last = slope * (w - 1) + intercept

    cs_y2 = np.cumsum(close ** 2)
    sum_y2 = np.full(n, np.nan)
    sum_y2[w-1] = cs_y2[w-1]
    if n > w:
        sum_y2[w:] = cs_y2[w:] - cs_y2[:n-w]

    var_y = (sum_y2 - sum_y ** 2 / w) / w
    var_x = denom / (w ** 2)
    var_resid = np.maximum(var_y - slope ** 2 * var_x, 0)
    std_resid = np.sqrt(var_resid)

    upper = fitted_last + 2 * std_resid
    lower = fitted_last - 2 * std_resid
    width = upper - lower

    pos = np.full(n, np.nan)
    valid = (width > 1e-10) & ~np.isnan(width)
    pos[valid] = (close[valid] - lower[valid]) / width[valid]
    pos = np.clip(pos, 0.0, 1.0)

    return pos

def compute_rsi(close_arr, period=14):
    n = len(close_arr)
    rsi = np.full(n, np.nan)
    diff = np.diff(close_arr)
    if len(diff) < period:
        return rsi
    gains = np.maximum(diff, 0.0)
    losses = np.maximum(-diff, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for i in range(period, len(diff)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss < 1e-10:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi

def compute_bvc(close_arr, volume_arr, lookback=20):
    n = len(close_arr)
    dp = np.diff(close_arr)
    dp = np.concatenate([[0], dp])
    sigma = pd.Series(dp).rolling(lookback, min_periods=lookback).std().values
    z = np.zeros(n)
    valid = sigma > 1e-10
    z[valid] = dp[valid] / sigma[valid]
    buy_pct = norm.cdf(z)
    net_flow = volume_arr * (2 * buy_pct - 1)
    net_cumul = pd.Series(net_flow).rolling(lookback, min_periods=lookback).sum().values
    total_vol = pd.Series(volume_arr).rolling(lookback, min_periods=lookback).sum().values
    bvc = np.full(n, np.nan)
    valid_v = total_vol > 0
    bvc[valid_v] = net_cumul[valid_v] / total_vol[valid_v]
    return bvc

def compute_entropy(close_arr, window=30, bins=8):
    n = len(close_arr)
    ent = np.full(n, np.nan)
    log_ret = np.diff(np.log(np.maximum(close_arr, 1e-10)))
    log_ret = np.concatenate([[0], log_ret])
    for i in range(window, n):
        rets = log_ret[i-window:i]
        if rets.std() < 1e-10:
            ent[i] = 0.0
            continue
        hist, _ = np.histogram(rets, bins=bins, density=False)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        ent[i] = -np.sum(probs * np.log2(probs))
    return ent

def compute_momentum_turn(close_arr, lookback=10):
    n = len(close_arr)
    turning = np.zeros(n, dtype=bool)
    for i in range(lookback * 2, n):
        mom = close_arr[i] - close_arr[i - lookback]
        prev_mom = close_arr[i - lookback] - close_arr[i - 2 * lookback]
        if mom < 0 and (mom - prev_mom) > 0:
            turning[i] = True
    return turning

def compute_atr(high_arr, low_arr, close_arr, period=14):
    n = len(close_arr)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high_arr[0] - low_arr[0]
    for i in range(1, n):
        tr[i] = max(high_arr[i] - low_arr[i],
                     abs(high_arr[i] - close_arr[i-1]),
                     abs(low_arr[i] - close_arr[i-1]))
    atr[period-1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def compute_vwap_intraday(open_arr, high_arr, low_arr, close_arr, volume_arr, dates_arr):
    """Compute intraday VWAP (resets each day).
    Returns vwap array and distance from VWAP as pct."""
    n = len(close_arr)
    vwap = np.full(n, np.nan)
    vwap_dist = np.full(n, np.nan)

    typical = (high_arr + low_arr + close_arr) / 3.0
    cum_tv = 0.0
    cum_v = 0.0
    prev_date = None

    for i in range(n):
        d = dates_arr[i]
        if d != prev_date:
            cum_tv = 0.0
            cum_v = 0.0
            prev_date = d
        cum_tv += typical[i] * volume_arr[i]
        cum_v += volume_arr[i]
        if cum_v > 0:
            vwap[i] = cum_tv / cum_v
            vwap_dist[i] = (close_arr[i] - vwap[i]) / vwap[i] * 100.0

    return vwap, vwap_dist

def compute_reversal_bars(open_arr, high_arr, low_arr, close_arr):
    """Detect bullish reversal bar patterns.
    Returns boolean array. Patterns: hammer, bullish engulfing."""
    n = len(close_arr)
    reversal = np.zeros(n, dtype=bool)

    for i in range(1, n):
        body = close_arr[i] - open_arr[i]
        bar_range = high_arr[i] - low_arr[i]
        if bar_range < 1e-10:
            continue

        # Hammer: small body at top, long lower shadow
        lower_shadow = min(open_arr[i], close_arr[i]) - low_arr[i]
        upper_shadow = high_arr[i] - max(open_arr[i], close_arr[i])
        if lower_shadow > 2 * abs(body) and lower_shadow > 2 * upper_shadow and body >= 0:
            reversal[i] = True
            continue

        # Bullish engulfing
        prev_body = close_arr[i-1] - open_arr[i-1]
        if prev_body < 0 and body > 0 and body > abs(prev_body):
            if open_arr[i] <= close_arr[i-1] and close_arr[i] >= open_arr[i-1]:
                reversal[i] = True

    return reversal

def compute_micro_momentum(close_1m, window=5):
    """1-min micro momentum: rate of change over short window.
    Returns array of momentum values."""
    n = len(close_1m)
    mom = np.full(n, np.nan)
    for i in range(window, n):
        if close_1m[i - window] > 0:
            mom[i] = (close_1m[i] - close_1m[i - window]) / close_1m[i - window] * 100.0
    return mom

# ================================================================
# Multi-TF Feature Builder (V4: adds 1min, 3min)
# ================================================================
def build_features(df1m, verbose=True):
    t0 = time.time()
    if verbose: print("Resampling to multiple TFs...")
    tfs = {}
    for rule, name in [('3min', '3m'), ('5min', '5m'), ('15min', '15m'),
                        ('30min', '30m'), ('1h', '1h'), ('4h', '4h')]:
        tfs[name] = resample_ohlcv(df1m, rule)
    tfs['daily'] = resample_ohlcv(df1m, '1D')
    tfs['1m'] = df1m.copy()  # keep 1min

    if verbose:
        for name, df in tfs.items():
            print(f"  {name}: {len(df):,} bars")

    features = {}
    windows = {'1m': 60, '3m': 60, '5m': 60, '15m': 40, '30m': 30,
               '1h': 24, '4h': 20, 'daily': 40}

    for name, df in tfs.items():
        if verbose: print(f"Computing features for {name}...")
        close = df['close'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        open_ = df['open'].values.astype(np.float64)

        feat = pd.DataFrame(index=df.index)
        feat['open'] = open_
        feat['high'] = high
        feat['low'] = low
        feat['close'] = close
        feat['volume'] = volume

        w = windows[name]
        feat['chan_pos'] = channel_position(close, w)

        if name != '1m':  # Skip heavy features on 1min (too many bars)
            feat['rsi'] = compute_rsi(close, 14)
            feat['bvc'] = compute_bvc(close, volume, 20)
            feat['atr'] = compute_atr(high, low, close, 14)

        if name in ('5m', '15m'):
            feat['entropy'] = compute_entropy(close, 30)
            feat['mom_turn'] = compute_momentum_turn(close, 10)

        if name == '5m':
            # VWAP
            dates = np.array([t.date() for t in df.index])
            vwap, vwap_dist = compute_vwap_intraday(open_, high, low, close, volume, dates)
            feat['vwap'] = vwap
            feat['vwap_dist'] = vwap_dist

        if name == '1m':
            # Micro momentum + reversal bars
            feat['micro_mom'] = compute_micro_momentum(close, 5)
            feat['reversal'] = compute_reversal_bars(open_, high, low, close)
            feat['chan_pos'] = channel_position(close, 60)

        if name == '3m':
            feat['reversal'] = compute_reversal_bars(open_, high, low, close)
            feat['rsi'] = compute_rsi(close, 14)
            feat['mom_turn'] = compute_momentum_turn(close, 10)

        features[name] = feat

    if verbose: print(f"Features computed in {time.time()-t0:.1f}s")
    return features, tfs

# ================================================================
# Pre-compute context arrays
# ================================================================
def precompute_daily_map(features, feat5m_dates):
    daily = features['daily']
    d_cp = daily['chan_pos'].values
    d_rsi = daily['rsi'].values
    d_bvc = daily['bvc'].values
    d_dates = np.array([idx.date() if hasattr(idx, 'date') else idx
                        for idx in daily.index])

    n = len(feat5m_dates)
    daily_cp_arr = np.full(n, np.nan)
    daily_rsi_arr = np.full(n, np.nan)
    daily_bvc_arr = np.full(n, np.nan)

    unique_5m_dates = sorted(set(feat5m_dates))
    date_to_daily = {}
    for d in unique_5m_dates:
        best = -1
        for k in range(len(d_dates)):
            if d_dates[k] < d:
                best = k
            elif d_dates[k] >= d:
                break
        if best >= 0:
            date_to_daily[d] = best

    for i in range(n):
        d = feat5m_dates[i]
        if d in date_to_daily:
            k = date_to_daily[d]
            daily_cp_arr[i] = d_cp[k]
            daily_rsi_arr[i] = d_rsi[k]
            daily_bvc_arr[i] = d_bvc[k]

    return daily_cp_arr, daily_rsi_arr, daily_bvc_arr

def precompute_hourly_arrays(features, feat5m_index):
    """Pre-compute multi-TF arrays aligned to 5m index."""
    n = len(feat5m_index)
    arrays = {
        '1h_cp': np.full(n, np.nan),
        '1h_rsi': np.full(n, np.nan),
        '1h_bvc': np.full(n, np.nan),
        '4h_cp': np.full(n, np.nan),
        '15m_cp': np.full(n, np.nan),
        '15m_rsi': np.full(n, np.nan),
        '30m_cp': np.full(n, np.nan),
        '3m_cp': np.full(n, np.nan),
        '3m_rsi': np.full(n, np.nan),
        '3m_turn': np.full(n, 0.0),
        '3m_reversal': np.full(n, 0.0),
    }

    tf_configs = [
        (features['1h'], {'1h_cp': 'chan_pos', '1h_rsi': 'rsi', '1h_bvc': 'bvc'}),
        (features['4h'], {'4h_cp': 'chan_pos'}),
        (features['15m'], {'15m_cp': 'chan_pos', '15m_rsi': 'rsi'}),
        (features['30m'], {'30m_cp': 'chan_pos'}),
        (features['3m'], {'3m_cp': 'chan_pos', '3m_rsi': 'rsi',
                           '3m_turn': 'mom_turn', '3m_reversal': 'reversal'}),
    ]

    for tf_feat, col_map in tf_configs:
        tf_idx = tf_feat.index.values
        col_arrays = {}
        for arr_key, feat_col in col_map.items():
            col_arrays[arr_key] = tf_feat[feat_col].values

        j = 0
        for i in range(n):
            t = feat5m_index[i]
            while j < len(tf_idx) - 1 and tf_idx[j + 1] <= t:
                j += 1
            if j < len(tf_idx) and tf_idx[j] <= t:
                for arr_key in col_map:
                    arrays[arr_key][i] = col_arrays[arr_key][j]

    return arrays

def precompute_1m_context(features, feat5m_index):
    """Pre-compute 1min microstructure aligned to 5m bars.
    For each 5m bar, get the latest 1min micro_mom, reversal, chan_pos."""
    f1m = features['1m']
    n5 = len(feat5m_index)
    arrays = {
        '1m_micro_mom': np.full(n5, np.nan),
        '1m_reversal': np.full(n5, 0.0),
        '1m_cp': np.full(n5, np.nan),
    }

    idx_1m = f1m.index.values
    mm_vals = f1m['micro_mom'].values
    rev_vals = f1m['reversal'].values.astype(float)
    cp_vals = f1m['chan_pos'].values

    j = 0
    for i in range(n5):
        t = feat5m_index[i]
        while j < len(idx_1m) - 1 and idx_1m[j + 1] <= t:
            j += 1
        if j < len(idx_1m) and idx_1m[j] <= t:
            arrays['1m_micro_mom'][i] = mm_vals[j]
            arrays['1m_cp'][i] = cp_vals[j]
            # Check if any 1m reversal in last 5 bars (= within this 5m bar)
            start = max(0, j - 4)
            if np.any(rev_vals[start:j+1] > 0):
                arrays['1m_reversal'][i] = 1.0

    return arrays

# ================================================================
# Signal Functions (V4: LONG-only)
# ================================================================

def signal_cross_tf_divergence_v4(i, f5m, ctx, params):
    """Cross-TF divergence V4: relaxed thresholds + VWAP + microstructure.
    Higher TFs bullish, 5min lagging -> catch-up trade."""
    cp5 = f5m['chan_pos'].iloc[i]
    if np.isnan(cp5):
        return None

    d_cp = ctx['daily_cp']
    h_cp = ctx['1h_cp'][i]
    h4_cp = ctx['4h_cp'][i]

    if np.isnan(d_cp) or np.isnan(h_cp) or np.isnan(h4_cp):
        return None

    higher_avg = d_cp * 0.35 + h4_cp * 0.35 + h_cp * 0.30
    divergence = higher_avg - cp5

    div_thresh = params.get('div_thresh', 0.35)
    if divergence < div_thresh:
        return None
    if cp5 > params.get('f5_thresh', 0.30):
        return None

    # V4: Require momentum turn on ANY of: 5m, 3m, or 1m reversal
    turn_5m = f5m.get('mom_turn')
    has_turn_5m = turn_5m is not None and turn_5m.iloc[i]
    has_turn_3m = ctx['3m_turn'][i] > 0.5
    has_reversal_1m = ctx['1m_reversal'][i] > 0.5
    has_reversal_3m = ctx['3m_reversal'][i] > 0.5

    if not (has_turn_5m or has_turn_3m or has_reversal_1m or has_reversal_3m):
        return None

    # V4: VWAP boost - below VWAP is better for mean reversion
    vwap_boost = 0.0
    vwap_dist = f5m['vwap_dist'].iloc[i] if 'vwap_dist' in f5m.columns else np.nan
    if not np.isnan(vwap_dist) and vwap_dist < 0:
        vwap_boost = min(abs(vwap_dist) * 0.02, 0.10)

    # V4: Micro momentum boost - 1m momentum turning up
    micro_boost = 0.0
    mm = ctx['1m_micro_mom'][i]
    if not np.isnan(mm) and mm > 0:
        micro_boost = min(mm * 0.5, 0.05)

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)

    conf = 0.55 + 0.25 * min(divergence, 0.7) + 0.10 * (1.0 - cp5) + vwap_boost + micro_boost
    conf = min(conf, 0.95)

    return ('LONG', conf, stop, tp)


def signal_mtf_bounce_v4(i, f5m, ctx, params):
    """Multi-TF bounce V4: relaxed + microstructure entry trigger."""
    cp5 = f5m['chan_pos'].iloc[i]
    rsi5 = f5m['rsi'].iloc[i]

    if np.isnan(cp5) or np.isnan(rsi5):
        return None

    d_cp = ctx['daily_cp']
    h_cp = ctx['1h_cp'][i]
    h4_cp = ctx['4h_cp'][i]
    cp15 = ctx['15m_cp'][i]
    cp30 = ctx['30m_cp'][i]
    cp3 = ctx['3m_cp'][i]

    if np.isnan(d_cp) or np.isnan(h_cp):
        return None

    # Count TFs near bottom
    tfs_near = 0
    if not np.isnan(d_cp) and d_cp < params.get('d_thresh', 0.35):
        tfs_near += 1
    if not np.isnan(h4_cp) and h4_cp < params.get('h4_thresh', 0.30):
        tfs_near += 1
    if not np.isnan(h_cp) and h_cp < params.get('h1_thresh', 0.30):
        tfs_near += 1
    if not np.isnan(cp15) and cp15 < params.get('f15_thresh', 0.25):
        tfs_near += 1
    if not np.isnan(cp30) and cp30 < params.get('f30_thresh', 0.30):
        tfs_near += 1

    min_tfs = params.get('min_tfs', 3)
    if tfs_near < min_tfs:
        return None

    if cp5 > params.get('f5_thresh', 0.20):
        return None
    if rsi5 > params.get('rsi_thresh', 35):
        return None

    # V4: Accept turn on any micro TF
    turn_5m = f5m.get('mom_turn')
    has_turn_5m = turn_5m is not None and turn_5m.iloc[i]
    has_turn_3m = ctx['3m_turn'][i] > 0.5
    has_reversal_1m = ctx['1m_reversal'][i] > 0.5
    has_reversal_3m = ctx['3m_reversal'][i] > 0.5

    if not (has_turn_5m or has_turn_3m or has_reversal_1m or has_reversal_3m):
        return None

    # BVC boost
    bvc5 = f5m['bvc'].iloc[i] if 'bvc' in f5m.columns else np.nan
    bvc_boost = 0.0
    if not np.isnan(bvc5) and bvc5 > 0.1:
        bvc_boost = min(bvc5 * 0.15, 0.10)

    # VWAP boost
    vwap_boost = 0.0
    vwap_dist = f5m['vwap_dist'].iloc[i] if 'vwap_dist' in f5m.columns else np.nan
    if not np.isnan(vwap_dist) and vwap_dist < 0:
        vwap_boost = min(abs(vwap_dist) * 0.02, 0.10)

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)

    conf = 0.55 + 0.10 * tfs_near + 0.10 * (1.0 - cp5) + bvc_boost + vwap_boost
    conf = min(conf, 0.95)

    return ('LONG', conf, stop, tp)


def signal_vwap_reversion(i, f5m, ctx, params):
    """VWAP mean reversion: price far below VWAP with higher TF support.
    Novel V4 signal - pure mean reversion play."""
    cp5 = f5m['chan_pos'].iloc[i]
    vwap_dist = f5m['vwap_dist'].iloc[i] if 'vwap_dist' in f5m.columns else np.nan

    if np.isnan(cp5) or np.isnan(vwap_dist):
        return None

    d_cp = ctx['daily_cp']
    h_cp = ctx['1h_cp'][i]

    if np.isnan(d_cp) or np.isnan(h_cp):
        return None

    # Price must be significantly below VWAP
    vwap_thresh = params.get('vwap_thresh', -0.30)  # at least 0.3% below VWAP
    if vwap_dist > vwap_thresh:
        return None

    # Higher TFs should be neutral or bullish (not crashing)
    if d_cp < params.get('d_min', 0.20):
        return None  # daily too low = major downtrend, skip
    if h_cp < params.get('h1_min', 0.15):
        return None

    # 5m channel should be low
    if cp5 > params.get('f5_thresh', 0.25):
        return None

    # Need micro confirmation
    has_turn_3m = ctx['3m_turn'][i] > 0.5
    has_reversal_1m = ctx['1m_reversal'][i] > 0.5
    turn_5m = f5m.get('mom_turn')
    has_turn_5m = turn_5m is not None and turn_5m.iloc[i]

    if not (has_turn_5m or has_turn_3m or has_reversal_1m):
        return None

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)

    # Confidence based on VWAP distance + channel position
    conf = 0.55 + min(abs(vwap_dist) * 0.05, 0.15) + 0.10 * (1.0 - cp5)
    conf = min(conf, 0.95)

    return ('LONG', conf, stop, tp)


def signal_micro_divergence(i, f5m, ctx, params):
    """Microstructure divergence: 1min/3min momentum turning up while 5min
    channel position is still low. Catches the very earliest reversal."""
    cp5 = f5m['chan_pos'].iloc[i]
    if np.isnan(cp5):
        return None

    cp1m = ctx['1m_cp'][i]
    cp3m = ctx['3m_cp'][i]
    mm = ctx['1m_micro_mom'][i]

    if np.isnan(cp1m) or np.isnan(cp3m) or np.isnan(mm):
        return None

    d_cp = ctx['daily_cp']
    h_cp = ctx['1h_cp'][i]
    if np.isnan(d_cp) or np.isnan(h_cp):
        return None

    # 5m must be low
    if cp5 > params.get('f5_thresh', 0.20):
        return None

    # 1m/3m must be turning UP (higher than 5m = micro divergence)
    micro_avg = cp1m * 0.5 + cp3m * 0.5
    micro_div = micro_avg - cp5
    if micro_div < params.get('micro_div_thresh', 0.15):
        return None

    # 1m momentum must be positive
    if mm < params.get('mm_thresh', 0.02):
        return None

    # Higher TFs should support
    if d_cp < params.get('d_min', 0.25):
        return None
    if h_cp < params.get('h1_min', 0.20):
        return None

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)

    conf = 0.55 + 0.20 * min(micro_div, 0.5) + 0.10 * (1.0 - cp5) + min(mm * 0.3, 0.05)
    conf = min(conf, 0.95)

    return ('LONG', conf, stop, tp)


def signal_tod_power(i, f5m, ctx, params):
    """Time-of-day power: enhanced bounce signals during high-alpha
    intraday periods (first 30min, last 30min = "power hours")."""
    cp5 = f5m['chan_pos'].iloc[i]
    if np.isnan(cp5):
        return None

    bar_time = f5m.index[i]
    tod = bar_time.time()

    # Only in power windows
    in_open_power = dt.time(9, 35) <= tod <= dt.time(10, 5)
    in_close_power = dt.time(15, 0) <= tod <= dt.time(15, 25)
    if not (in_open_power or in_close_power):
        return None

    d_cp = ctx['daily_cp']
    h_cp = ctx['1h_cp'][i]
    if np.isnan(d_cp) or np.isnan(h_cp):
        return None

    if cp5 > params.get('f5_thresh', 0.25):
        return None

    # Need higher TF support
    higher_avg = d_cp * 0.5 + h_cp * 0.5
    if higher_avg < params.get('higher_min', 0.30):
        return None

    # Need micro confirmation
    has_turn_3m = ctx['3m_turn'][i] > 0.5
    has_reversal_1m = ctx['1m_reversal'][i] > 0.5
    turn_5m = f5m.get('mom_turn')
    has_turn_5m = turn_5m is not None and turn_5m.iloc[i]

    if not (has_turn_5m or has_turn_3m or has_reversal_1m):
        return None

    # VWAP check
    vwap_dist = f5m['vwap_dist'].iloc[i] if 'vwap_dist' in f5m.columns else np.nan
    vwap_ok = np.isnan(vwap_dist) or vwap_dist < 0.1  # at or below VWAP

    if not vwap_ok:
        return None

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)

    # Opening power tends to be higher conviction
    tod_boost = 0.05 if in_open_power else 0.0
    conf = 0.55 + 0.15 * (higher_avg - 0.30) + 0.10 * (1.0 - cp5) + tod_boost
    conf = min(conf, 0.95)

    return ('LONG', conf, stop, tp)


def signal_union_v4(i, f5m, ctx, params):
    """Union of all V4 signals: fire if ANY signal fires.
    Uses the highest-confidence signal."""
    best = None
    best_conf = 0.0

    for fn in [signal_cross_tf_divergence_v4, signal_mtf_bounce_v4,
               signal_vwap_reversion, signal_micro_divergence, signal_tod_power]:
        result = fn(i, f5m, ctx, params)
        if result is not None and result[1] > best_conf:
            best = result
            best_conf = result[1]

    return best


# ================================================================
# Trade Simulation V4 (adaptive trail)
# ================================================================
def simulate_v4(feat5m, features, signal_fn, name, params,
                trail_base=0.010, trail_power=4,
                adaptive_trail=False,  # V4: widen trail after profit
                adaptive_bars=3,       # bars to wait before widening
                adaptive_mult=2.0,     # trail multiplier after profit
                max_hold_bars=78, force_eod=True,
                cooldown_bars=6, max_trades_per_day=2,
                _precomputed=None):
    """V4 trade simulation with adaptive trailing stop."""
    if _precomputed is not None:
        daily_arrays, htf_arrays, micro_arrays = _precomputed
    else:
        raise ValueError("Must pass _precomputed to simulate_v4")

    daily_cp_arr, daily_rsi_arr, daily_bvc_arr = daily_arrays

    o_arr = feat5m['open'].values
    h_arr = feat5m['high'].values
    l_arr = feat5m['low'].values
    c_arr = feat5m['close'].values
    times = feat5m.index
    time_of_day = np.array([t.time() for t in times])

    n = len(feat5m)
    trades = []

    in_trade = False
    entry_price = 0.0
    entry_time = None
    confidence = 0.0
    stop_price = 0.0
    tp_price = 0.0
    best_price = 0.0
    hold_bars = 0
    cooldown_remaining = 0
    trades_today = 0
    current_date = None
    pending_signal = None

    ctx = {
        'daily_cp': 0.0,
        '1h_cp': htf_arrays['1h_cp'],
        '4h_cp': htf_arrays['4h_cp'],
        '15m_cp': htf_arrays['15m_cp'],
        '15m_rsi': htf_arrays['15m_rsi'],
        '30m_cp': htf_arrays['30m_cp'],
        '1h_rsi': htf_arrays['1h_rsi'],
        '1h_bvc': htf_arrays['1h_bvc'],
        '3m_cp': htf_arrays['3m_cp'],
        '3m_rsi': htf_arrays['3m_rsi'],
        '3m_turn': htf_arrays['3m_turn'],
        '3m_reversal': htf_arrays['3m_reversal'],
        '1m_micro_mom': micro_arrays['1m_micro_mom'],
        '1m_reversal': micro_arrays['1m_reversal'],
        '1m_cp': micro_arrays['1m_cp'],
    }

    t_entry_start = dt.time(9, 35)
    t_entry_end = dt.time(15, 30)
    t_sig_start = dt.time(9, 40)
    t_sig_end = dt.time(15, 25)
    t_force_exit = dt.time(15, 50)

    for i in range(n):
        bar_time = times[i]
        bar_date = bar_time.date()
        bar_tod = time_of_day[i]
        o, h, l, c = o_arr[i], h_arr[i], l_arr[i], c_arr[i]

        if bar_date != current_date:
            current_date = bar_date
            trades_today = 0

        # Execute pending signal
        if pending_signal is not None and not in_trade:
            sig_conf, sig_stop, sig_tp = pending_signal
            pending_signal = None

            if bar_tod >= t_entry_start and bar_tod <= t_entry_end:
                entry_price = o * (1 + SLIPPAGE_PCT)
                entry_time = bar_time
                confidence = sig_conf
                in_trade = True
                hold_bars = 0
                trades_today += 1
                stop_price = entry_price * (1 - sig_stop)
                tp_price = entry_price * (1 + sig_tp)
                best_price = entry_price

        # Manage position
        if in_trade:
            hold_bars += 1
            exit_price = None
            exit_reason = None

            best_price = max(best_price, h)

            # V4 adaptive trail: tight early, wider after profit develops
            if adaptive_trail and hold_bars >= adaptive_bars:
                pnl_pct = (best_price - entry_price) / entry_price
                if pnl_pct > 0.002:  # at least 0.2% profit
                    trail_pct = trail_base * adaptive_mult * (1.0 - confidence) ** trail_power
                else:
                    trail_pct = trail_base * (1.0 - confidence) ** trail_power
            else:
                trail_pct = trail_base * (1.0 - confidence) ** trail_power

            trail_stop = best_price * (1 - trail_pct)
            if trail_stop > stop_price:
                stop_price = trail_stop

            if l <= stop_price:
                exit_price = max(stop_price, l)
                exit_reason = 'stop' if stop_price < entry_price else 'trail'
            elif h >= tp_price:
                exit_price = tp_price
                exit_reason = 'tp'
            elif hold_bars >= max_hold_bars:
                exit_price = c
                exit_reason = 'timeout'
            elif force_eod and bar_tod >= t_force_exit:
                exit_price = c
                exit_reason = 'eod'

            if exit_price is not None:
                exit_adj = exit_price * (1 - SLIPPAGE_PCT)
                shares = max(1, int(CAPITAL * confidence / entry_price))
                pnl = (exit_adj - entry_price) * shares
                pnl -= COMM_PER_SHARE * shares * 2

                trades.append(Trade(
                    entry_time=entry_time, exit_time=bar_time,
                    direction='LONG', entry_price=entry_price,
                    exit_price=exit_adj, confidence=confidence,
                    shares=shares, pnl=pnl, hold_bars=hold_bars,
                    exit_reason=exit_reason, signal_name=name,
                ))
                in_trade = False
                cooldown_remaining = cooldown_bars

        # New signals
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if in_trade or trades_today >= max_trades_per_day:
            continue
        if bar_tod < t_sig_start or bar_tod > t_sig_end:
            continue

        ctx['daily_cp'] = daily_cp_arr[i]

        result = signal_fn(i, feat5m, ctx, params)
        if result is not None:
            _, conf, stop, tp = result
            pending_signal = (conf, stop, tp)

    return trades

# ================================================================
# Results Display
# ================================================================
def print_summary(trades, label=""):
    if not trades:
        print(f"  {label}: 0 trades")
        return 0, 0, 0
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades)
    bw = max(t.pnl for t in trades)
    wr = wins / n * 100
    avg_hold = np.mean([t.hold_bars for t in trades])
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    r_str = ' '.join(f"{k}={v}" for k, v in sorted(reasons.items()))
    flag = " ***" if wr >= 90 else ""
    print(f"  {label}: {n}t {wins}W/{n-wins}L ({wr:.1f}% WR) "
          f"${pnl:+,.0f} BW=${bw:+,.0f} BL=${bl:+,.0f} "
          f"hold={avg_hold:.1f}b [{r_str}]{flag}")
    return n, wins, pnl

def validate(trades, label):
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    train = [t for t in trades if t.entry_time <= TRAIN_END]
    test = [t for t in trades if TRAIN_END < t.entry_time <= TEST_END]
    oos = [t for t in trades if t.entry_time > TEST_END]
    print_summary(trades, "FULL")
    print_summary(train, "TRAIN")
    print_summary(test, "TEST")
    print_summary(oos, "2026 OOS")
    years = sorted(set(t.entry_time.year for t in trades))
    cum = 0
    for yr in years:
        yt = [t for t in trades if t.entry_time.year == yr]
        if yt:
            w = sum(1 for t in yt if t.pnl > 0)
            p = sum(t.pnl for t in yt)
            bl = min(t.pnl for t in yt)
            cum += p
            wr = w / len(yt) * 100
            f = " ***" if wr >= 90 else ""
            print(f"    {yr}: {len(yt):3d}t {wr:5.1f}% ${p:+8,.0f} BL=${bl:+,.0f} cum=${cum:+,.0f}{f}")

# ================================================================
# Main Experiments
# ================================================================
def main():
    print("=" * 70)
    print("INTRADAY 5-MIN V4: MICROSTRUCTURE + VWAP + ADAPTIVE TRAIL")
    print("=" * 70)

    df1m = load_1min()
    features, tfs = build_features(df1m)
    f5m = features['5m']

    print("Pre-computing context arrays...")
    t0 = time.time()
    bar_dates = np.array([t.date() for t in f5m.index])
    daily_arrays = precompute_daily_map(features, bar_dates)
    htf_arrays = precompute_hourly_arrays(features, f5m.index)
    micro_arrays = precompute_1m_context(features, f5m.index)
    precomp = (daily_arrays, htf_arrays, micro_arrays)
    print(f"  Context pre-computed in {time.time()-t0:.1f}s")

    # ── Experiment 1: Each V4 signal with default params ──
    print("\n" + "=" * 70)
    print("EXP 1: Individual V4 signals (defaults)")
    print("=" * 70)

    default_p = {
        'stop': 0.008, 'tp': 0.020,
        'd_thresh': 0.35, 'h4_thresh': 0.30, 'h1_thresh': 0.30,
        'f15_thresh': 0.25, 'f30_thresh': 0.30, 'f5_thresh': 0.25,
        'rsi_thresh': 35, 'min_tfs': 3,
        'div_thresh': 0.35,
        'vwap_thresh': -0.30, 'd_min': 0.20, 'h1_min': 0.15,
        'micro_div_thresh': 0.15, 'mm_thresh': 0.02,
        'higher_min': 0.30,
    }

    signals = [
        ('CrossTF_Div_v4', signal_cross_tf_divergence_v4),
        ('MTF_Bounce_v4', signal_mtf_bounce_v4),
        ('VWAP_Revert', signal_vwap_reversion),
        ('Micro_Div', signal_micro_divergence),
        ('TOD_Power', signal_tod_power),
        ('Union_v4', signal_union_v4),
    ]

    for sname, sfn in signals:
        trades = simulate_v4(f5m, features, sfn, sname, default_p,
                             trail_base=0.010, trail_power=4,
                             _precomputed=precomp)
        if trades:
            validate(trades, sname)

    # ── Experiment 2: Adaptive trail sweep ──
    print("\n" + "=" * 70)
    print("EXP 2: Adaptive trail sweep (Union_v4)")
    print("=" * 70)

    for tb in [0.005, 0.008, 0.010, 0.015, 0.020]:
        for tp in [2, 4, 6, 8, 12]:
            for adapt in [False, True]:
                for am in ([1.5, 2.0, 3.0, 5.0] if adapt else [1.0]):
                    for ab in ([2, 3, 5] if adapt else [3]):
                        trades = simulate_v4(f5m, features, signal_union_v4,
                                             'union', default_p,
                                             trail_base=tb, trail_power=tp,
                                             adaptive_trail=adapt,
                                             adaptive_mult=am, adaptive_bars=ab,
                                             _precomputed=precomp)
                        if trades and len(trades) >= 5:
                            n = len(trades)
                            w = sum(1 for t in trades if t.pnl > 0)
                            wr = w / n * 100
                            pnl = sum(t.pnl for t in trades)
                            bl = min(t.pnl for t in trades)
                            flag = " ***" if wr >= 90 else ""
                            ad_str = f"adapt={am:.1f}@{ab}b" if adapt else "static"
                            if wr >= 60 or pnl > 50000:
                                print(f"  tb={tb:.3f} tp={tp:2d} {ad_str:15s}: "
                                      f"{n:4d}t {wr:5.1f}% ${pnl:+10,.0f} BL=${bl:+,.0f}{flag}")

    # ── Experiment 3: Signal parameter sweeps ──
    print("\n" + "=" * 70)
    print("EXP 3: CrossTF_Div_v4 threshold sweep")
    print("=" * 70)

    for div in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        for f5t in [0.15, 0.20, 0.25, 0.30, 0.35]:
            p = dict(default_p, div_thresh=div, f5_thresh=f5t)
            for tb in [0.008, 0.010, 0.015]:
                trades = simulate_v4(f5m, features, signal_cross_tf_divergence_v4,
                                     'div4', p, trail_base=tb,
                                     _precomputed=precomp)
                if trades and len(trades) >= 5:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    flag = " ***" if wr >= 90 else ""
                    if wr >= 60:
                        print(f"  div>{div:.2f} 5m<{f5t:.2f} tb={tb:.3f}: "
                              f"{n:4d}t {wr:5.1f}% ${pnl:+10,.0f} BL=${bl:+,.0f}{flag}")

    # ── Experiment 4: VWAP signal sweep ──
    print("\n" + "=" * 70)
    print("EXP 4: VWAP reversion sweep")
    print("=" * 70)

    for vt in [-0.10, -0.20, -0.30, -0.50, -0.80, -1.00]:
        for f5t in [0.15, 0.20, 0.25, 0.30]:
            for dm in [0.15, 0.20, 0.30]:
                p = dict(default_p, vwap_thresh=vt, f5_thresh=f5t, d_min=dm)
                trades = simulate_v4(f5m, features, signal_vwap_reversion,
                                     'vwap', p, trail_base=0.010,
                                     _precomputed=precomp)
                if trades and len(trades) >= 3:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    flag = " ***" if wr >= 90 else ""
                    if wr >= 60 or pnl > 5000:
                        print(f"  vwap<{vt:.2f} 5m<{f5t:.2f} d>{dm:.2f}: "
                              f"{n:4d}t {wr:5.1f}% ${pnl:+8,.0f} BL=${bl:+,.0f}{flag}")

    # ── Experiment 5: Micro divergence sweep ──
    print("\n" + "=" * 70)
    print("EXP 5: Microstructure divergence sweep")
    print("=" * 70)

    for md in [0.05, 0.10, 0.15, 0.20, 0.30]:
        for mm in [0.01, 0.02, 0.05, 0.10]:
            for f5t in [0.15, 0.20, 0.25]:
                p = dict(default_p, micro_div_thresh=md, mm_thresh=mm, f5_thresh=f5t)
                trades = simulate_v4(f5m, features, signal_micro_divergence,
                                     'micro', p, trail_base=0.010,
                                     _precomputed=precomp)
                if trades and len(trades) >= 3:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    flag = " ***" if wr >= 90 else ""
                    if wr >= 60 or pnl > 5000:
                        print(f"  md>{md:.2f} mm>{mm:.2f} 5m<{f5t:.2f}: "
                              f"{n:4d}t {wr:5.1f}% ${pnl:+8,.0f} BL=${bl:+,.0f}{flag}")

    # ── Experiment 6: Cooldown + trades/day sweep ──
    print("\n" + "=" * 70)
    print("EXP 6: Cooldown + max trades/day (Union_v4)")
    print("=" * 70)

    for cd in [0, 3, 6, 12]:
        for mtd in [1, 2, 3]:
            trades = simulate_v4(f5m, features, signal_union_v4,
                                 'union', default_p,
                                 trail_base=0.010, trail_power=4,
                                 cooldown_bars=cd, max_trades_per_day=mtd,
                                 _precomputed=precomp)
            if trades and len(trades) >= 5:
                n = len(trades)
                w = sum(1 for t in trades if t.pnl > 0)
                wr = w / n * 100
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                oos = [t for t in trades if t.entry_time > TEST_END]
                oos_n = len(oos)
                oos_w = sum(1 for t in oos if t.pnl > 0) if oos else 0
                oos_pnl = sum(t.pnl for t in oos) if oos else 0
                flag = " ***" if wr >= 90 else ""
                print(f"  cd={cd:2d} mtd={mtd}: {n:4d}t {wr:5.1f}% ${pnl:+10,.0f} BL=${bl:+,.0f} "
                      f"OOS:{oos_n}t {oos_w}W ${oos_pnl:+,.0f}{flag}")

    # ── Experiment 7: Multi-day hold with adaptive trail ──
    print("\n" + "=" * 70)
    print("EXP 7: Multi-day hold + adaptive trail (Union_v4)")
    print("=" * 70)

    for mh in [78, 156, 390]:
        for tb in [0.010, 0.015, 0.020]:
            for adapt in [False, True]:
                am = 2.5 if adapt else 1.0
                trades = simulate_v4(f5m, features, signal_union_v4,
                                     'union', default_p,
                                     trail_base=tb, trail_power=4,
                                     adaptive_trail=adapt, adaptive_mult=am,
                                     max_hold_bars=mh, force_eod=False,
                                     _precomputed=precomp)
                if trades and len(trades) >= 5:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    ah = np.mean([t.hold_bars for t in trades])
                    flag = " ***" if wr >= 90 else ""
                    ad_s = "adapt" if adapt else "static"
                    if wr >= 55 or pnl > 50000:
                        print(f"  mh={mh:3d} tb={tb:.3f} {ad_s:6s}: "
                              f"{n:4d}t {wr:5.1f}% ${pnl:+10,.0f} BL=${bl:+,.0f} "
                              f"h={ah:.0f}b{flag}")

    # ── Experiment 8: Stop/TP sweep on best signals ──
    print("\n" + "=" * 70)
    print("EXP 8: Stop/TP sweep (CrossTF_Div_v4 + Union_v4)")
    print("=" * 70)

    for sname, sfn in [('CrossTF_Div_v4', signal_cross_tf_divergence_v4),
                        ('Union_v4', signal_union_v4)]:
        print(f"\n--- {sname} ---")
        for stop in [0.003, 0.005, 0.008, 0.012, 0.015, 0.020]:
            for tp in [0.010, 0.015, 0.020, 0.030, 0.040, 0.050]:
                if tp <= stop:
                    continue
                p = dict(default_p, stop=stop, tp=tp)
                trades = simulate_v4(f5m, features, sfn, sname, p,
                                     _precomputed=precomp)
                if trades and len(trades) >= 5:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    flag = " ***" if wr >= 90 else ""
                    if wr >= 60:
                        print(f"  s={stop:.3f} t={tp:.3f}: {n:4d}t {wr:5.1f}% "
                              f"${pnl:+10,.0f} BL=${bl:+,.0f}{flag}")

    print("\n" + "=" * 70)
    print("V4 experiments complete.")
    print("=" * 70)

if __name__ == '__main__':
    main()
