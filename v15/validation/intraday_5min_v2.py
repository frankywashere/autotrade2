#!/usr/bin/env python3
"""
Intraday 5-Min TSLA Backtester V2

Changes from V1:
1. LONG-only (SHORTs were 0% WR across all signals)
2. Wider trailing stop base (0.010 vs 0.004) so trades develop
3. Multiple hold strategies: intraday-only, overnight, multi-day
4. 15min + 30min channel confirmation added
5. Composite "flow score" combining BVC + entropy + volume
6. Higher selectivity thresholds
7. Sweep trail_power and hold period extensively
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
# Data Loading (same as V1)
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
    """Average True Range."""
    n = len(close_arr)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high_arr[0] - low_arr[0]
    for i in range(1, n):
        tr[i] = max(high_arr[i] - low_arr[i],
                     abs(high_arr[i] - close_arr[i-1]),
                     abs(low_arr[i] - close_arr[i-1]))
    # EMA-style ATR
    atr[period-1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

# ================================================================
# Multi-TF Feature Builder
# ================================================================
def build_features(df1m, verbose=True):
    t0 = time.time()
    if verbose: print("Resampling to multiple TFs...")
    tfs = {}
    for rule, name in [('5min', '5m'), ('15min', '15m'), ('30min', '30m'),
                        ('1h', '1h'), ('4h', '4h')]:
        tfs[name] = resample_ohlcv(df1m, rule)
    tfs['daily'] = resample_ohlcv(df1m, '1D')

    if verbose:
        for name, df in tfs.items():
            print(f"  {name}: {len(df):,} bars")

    features = {}
    windows = {'5m': 60, '15m': 40, '30m': 30, '1h': 24, '4h': 20, 'daily': 40}

    for name, df in tfs.items():
        if verbose: print(f"Computing features for {name}...")
        close = df['close'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)

        feat = pd.DataFrame(index=df.index)
        feat['open'] = df['open'].values
        feat['high'] = high
        feat['low'] = low
        feat['close'] = close
        feat['volume'] = volume

        w = windows[name]
        feat['chan_pos'] = channel_position(close, w)
        feat['rsi'] = compute_rsi(close, 14)
        feat['bvc'] = compute_bvc(close, volume, 20)
        feat['atr'] = compute_atr(high, low, close, 14)

        if name in ('5m', '15m'):
            feat['entropy'] = compute_entropy(close, 30)
            feat['mom_turn'] = compute_momentum_turn(close, 10)

        features[name] = feat

    if verbose: print(f"Features computed in {time.time()-t0:.1f}s")
    return features, tfs

# ================================================================
# Context Lookups (avoid lookahead)
# ================================================================
def _latest_before(feat_df, bar_time, col):
    """Get most recent value of col from feat_df strictly before bar_time."""
    prev = feat_df.index[feat_df.index < bar_time]
    if len(prev) == 0:
        return np.nan
    return feat_df.loc[prev[-1], col]

def _latest_at_or_before(feat_df, bar_time, col):
    """Get most recent value of col from feat_df at or before bar_time."""
    prev = feat_df.index[feat_df.index <= bar_time]
    if len(prev) == 0:
        return np.nan
    return feat_df.loc[prev[-1], col]

# ================================================================
# Pre-compute daily context map for speed
# ================================================================
def precompute_daily_map(features, feat5m_dates):
    """Build per-bar daily context array for O(1) lookup.
    Returns array of daily_cp values aligned to 5m bars."""
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

    # Build sorted date -> index mapping
    j = 0
    unique_5m_dates = sorted(set(feat5m_dates))
    date_to_daily = {}
    for d in unique_5m_dates:
        # Find latest daily date < d (previous day, no lookahead)
        best = -1
        for k in range(len(d_dates)):
            if d_dates[k] < d:
                best = k
            elif d_dates[k] >= d:
                break
        if best >= 0:
            date_to_daily[d] = best

    # Map to all 5m bars
    for i in range(n):
        d = feat5m_dates[i]
        if d in date_to_daily:
            k = date_to_daily[d]
            daily_cp_arr[i] = d_cp[k]
            daily_rsi_arr[i] = d_rsi[k]
            daily_bvc_arr[i] = d_bvc[k]

    return daily_cp_arr, daily_rsi_arr, daily_bvc_arr

def precompute_hourly_arrays(features, feat5m_index):
    """Pre-compute hourly channel pos aligned to 5m index for speed."""
    hourly = features['1h']
    h4 = features['4h']
    f15m = features['15m']
    f30m = features['30m']

    n = len(feat5m_index)
    arrays = {
        '1h_cp': np.full(n, np.nan),
        '1h_rsi': np.full(n, np.nan),
        '1h_bvc': np.full(n, np.nan),
        '4h_cp': np.full(n, np.nan),
        '15m_cp': np.full(n, np.nan),
        '15m_rsi': np.full(n, np.nan),
        '30m_cp': np.full(n, np.nan),
    }

    # Forward fill using pre-extracted value arrays
    for tf_key, tf_feat, col_map in [
        ('1h', hourly, {'1h_cp': 'chan_pos', '1h_rsi': 'rsi', '1h_bvc': 'bvc'}),
        ('4h', h4, {'4h_cp': 'chan_pos'}),
        ('15m', f15m, {'15m_cp': 'chan_pos', '15m_rsi': 'rsi'}),
        ('30m', f30m, {'30m_cp': 'chan_pos'}),
    ]:
        tf_idx = tf_feat.index.values  # numpy array for fast comparison
        # Pre-extract column values
        col_arrays = {arr_key: tf_feat[feat_col].values
                      for arr_key, feat_col in col_map.items()}

        j = 0
        for i in range(n):
            t = feat5m_index[i]
            while j < len(tf_idx) - 1 and tf_idx[j + 1] <= t:
                j += 1
            if j < len(tf_idx) and tf_idx[j] <= t:
                for arr_key in col_map:
                    arrays[arr_key][i] = col_arrays[arr_key][j]

    return arrays

# ================================================================
# Signal Functions (LONG-only)
# ================================================================

def signal_multi_tf_bounce(i, f5m, ctx, params):
    """Multi-TF channel bounce: daily+1h+15m+5m all near bottom.
    Most selective signal - requires multi-TF alignment."""
    cp5 = f5m['chan_pos'].iloc[i]
    rsi5 = f5m['rsi'].iloc[i]
    bvc5 = f5m['bvc'].iloc[i]
    turn = f5m.get('mom_turn')
    has_turn = turn is not None and turn.iloc[i]

    if np.isnan(cp5) or np.isnan(rsi5):
        return None

    d_cp = ctx['daily_cp']
    h_cp = ctx['1h_cp'][i]
    cp15 = ctx['15m_cp'][i]
    h4_cp = ctx['4h_cp'][i]

    if np.isnan(d_cp) or np.isnan(h_cp):
        return None

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)

    # Count how many TFs are near bottom
    tfs_near_bottom = 0
    if not np.isnan(d_cp) and d_cp < params.get('d_thresh', 0.35):
        tfs_near_bottom += 1
    if not np.isnan(h4_cp) and h4_cp < params.get('h4_thresh', 0.30):
        tfs_near_bottom += 1
    if not np.isnan(h_cp) and h_cp < params.get('h1_thresh', 0.30):
        tfs_near_bottom += 1
    if not np.isnan(cp15) and cp15 < params.get('f15_thresh', 0.25):
        tfs_near_bottom += 1

    min_tfs = params.get('min_tfs', 3)
    if tfs_near_bottom < min_tfs:
        return None

    # 5min must be at extreme bottom with momentum turning
    if cp5 > params.get('f5_thresh', 0.15):
        return None
    if rsi5 > params.get('rsi_thresh', 30):
        return None
    if not has_turn:
        return None

    # BVC should show buying (optional boost)
    bvc_boost = 0.0
    if not np.isnan(bvc5) and bvc5 > 0.1:
        bvc_boost = min(bvc5 * 0.15, 0.10)

    conf = 0.55 + 0.10 * tfs_near_bottom + 0.10 * (1.0 - cp5) + bvc_boost
    conf = min(conf, 0.95)

    return ('LONG', conf, stop, tp)


def signal_bvc_accumulation(i, f5m, ctx, params):
    """BVC accumulation: strong buying flow at low channel positions.
    Novel signal based on order flow proxy."""
    cp5 = f5m['chan_pos'].iloc[i]
    bvc5 = f5m['bvc'].iloc[i]
    rsi5 = f5m['rsi'].iloc[i]

    if np.isnan(cp5) or np.isnan(bvc5) or np.isnan(rsi5):
        return None

    d_cp = ctx['daily_cp']
    if np.isnan(d_cp):
        return None

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)

    # Strong buying flow at low prices
    bvc_thresh = params.get('bvc_thresh', 0.25)
    if bvc5 < bvc_thresh:
        return None
    if cp5 > params.get('f5_thresh', 0.20):
        return None
    if d_cp > params.get('d_thresh', 0.40):
        return None
    if rsi5 > params.get('rsi_thresh', 35):
        return None

    conf = 0.55 + 0.20 * bvc5 + 0.15 * (1.0 - cp5)
    conf = min(max(conf, 0.45), 0.95)

    return ('LONG', conf, stop, tp)


def signal_entropy_regime(i, f5m, ctx, params):
    """Low-entropy regime bounce: only trade when market is predictable."""
    cp5 = f5m['chan_pos'].iloc[i]
    rsi5 = f5m['rsi'].iloc[i]
    ent = f5m.get('entropy')

    if ent is None or np.isnan(ent.iloc[i]):
        return None
    if np.isnan(cp5) or np.isnan(rsi5):
        return None

    ent_val = ent.iloc[i]

    # Only trade in low-entropy (predictable) regimes
    ent_thresh = params.get('entropy_thresh', 2.0)
    if ent_val > ent_thresh:
        return None

    h_cp = ctx['1h_cp'][i]
    d_cp = ctx['daily_cp']

    if np.isnan(h_cp) or np.isnan(d_cp):
        return None

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)

    # Channel bounce in predictable regime
    if cp5 > params.get('f5_thresh', 0.15):
        return None
    if h_cp > params.get('h1_thresh', 0.35):
        return None
    if rsi5 > params.get('rsi_thresh', 30):
        return None

    turn = f5m.get('mom_turn')
    if turn is not None and not turn.iloc[i]:
        return None

    ent_boost = max(0, (ent_thresh - ent_val) / ent_thresh) * 0.10
    conf = 0.55 + 0.15 * (1.0 - cp5) + 0.10 * (1.0 - d_cp) + ent_boost
    conf = min(conf, 0.95)

    return ('LONG', conf, stop, tp)


def signal_cross_tf_divergence(i, f5m, ctx, params):
    """Cross-TF divergence: higher TFs bullish, 5min lagging -> catch-up."""
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

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)
    div_thresh = params.get('div_thresh', 0.40)

    if divergence < div_thresh:
        return None
    if cp5 > params.get('f5_thresh', 0.25):
        return None

    turn = f5m.get('mom_turn')
    if turn is not None and not turn.iloc[i]:
        return None

    conf = 0.55 + 0.25 * min(divergence, 0.7) + 0.10 * (1.0 - cp5)
    conf = min(conf, 0.95)

    return ('LONG', conf, stop, tp)


def signal_atr_squeeze_bounce(i, f5m, ctx, params):
    """ATR squeeze bounce: low volatility compression + channel bottom.
    When volatility compresses at channel bottom, expansion is often upward."""
    cp5 = f5m['chan_pos'].iloc[i]
    atr5 = f5m['atr'].iloc[i]
    rsi5 = f5m['rsi'].iloc[i]

    if np.isnan(cp5) or np.isnan(atr5) or np.isnan(rsi5):
        return None

    d_cp = ctx['daily_cp']
    if np.isnan(d_cp):
        return None

    # Check if ATR is relatively low (compression)
    # Compare current ATR to rolling average ATR
    atr_series = f5m['atr'].values
    idx = f5m.index.get_loc(f5m.index[i])
    if idx < 60:
        return None

    atr_window = atr_series[idx-60:idx]
    atr_avg = np.nanmean(atr_window)
    if atr_avg < 1e-10:
        return None

    atr_ratio = atr5 / atr_avg  # < 1 means compression

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)
    atr_thresh = params.get('atr_squeeze_thresh', 0.70)

    if atr_ratio > atr_thresh:
        return None
    if cp5 > params.get('f5_thresh', 0.15):
        return None
    if d_cp > params.get('d_thresh', 0.40):
        return None
    if rsi5 > params.get('rsi_thresh', 30):
        return None

    turn = f5m.get('mom_turn')
    if turn is not None and not turn.iloc[i]:
        return None

    conf = 0.60 + 0.15 * (1.0 - atr_ratio) + 0.10 * (1.0 - cp5)
    conf = min(conf, 0.95)

    return ('LONG', conf, stop, tp)


def signal_composite(i, f5m, ctx, params):
    """Composite: require at least N signals to agree."""
    signals = []
    for fn in [signal_multi_tf_bounce, signal_bvc_accumulation,
               signal_entropy_regime, signal_cross_tf_divergence,
               signal_atr_squeeze_bounce]:
        result = fn(i, f5m, ctx, params)
        if result is not None:
            signals.append(result)

    min_agree = params.get('min_agree', 2)
    if len(signals) < min_agree:
        return None

    avg_conf = np.mean([s[1] for s in signals])
    conf = min(avg_conf + 0.05 * (len(signals) - min_agree), 0.95)
    return ('LONG', conf, params.get('stop', 0.008), params.get('tp', 0.020))

# ================================================================
# Trade Simulation (V2 - supports multi-day holds)
# ================================================================
def simulate_v2(feat5m, features, signal_fn, name, params,
                trail_base=0.010, trail_power=4,
                max_hold_bars=78, force_eod=True,
                cooldown_bars=6, max_trades_per_day=1,
                _precomputed=None):
    """V2 trade simulation with configurable hold period and trailing stop.
    Pass _precomputed=(daily_arrays, htf_arrays) to avoid recomputation."""
    # Pre-compute context arrays (reuse if provided)
    if _precomputed is not None:
        daily_cp_arr, daily_rsi_arr, daily_bvc_arr = _precomputed[0]
        htf_arrays = _precomputed[1]
    else:
        bar_dates = np.array([t.date() for t in feat5m.index])
        daily_cp_arr, daily_rsi_arr, daily_bvc_arr = precompute_daily_map(
            features, bar_dates)
        htf_arrays = precompute_hourly_arrays(features, feat5m.index)

    # Pre-extract 5m arrays for fast access
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

    # Build shared context once (htf arrays are read by index)
    ctx = {
        'daily_cp': 0.0,  # will be set per-bar
        '1h_cp': htf_arrays['1h_cp'],
        '4h_cp': htf_arrays['4h_cp'],
        '15m_cp': htf_arrays['15m_cp'],
        '15m_rsi': htf_arrays['15m_rsi'],
        '30m_cp': htf_arrays['30m_cp'],
        '1h_rsi': htf_arrays['1h_rsi'],
        '1h_bvc': htf_arrays['1h_bvc'],
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

        # Set daily context for this bar
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

    # Per-year
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
    print("INTRADAY 5-MIN V2: LONG-ONLY + WIDER TRAIL + MULTI-TF")
    print("=" * 70)

    df1m = load_1min()
    features, tfs = build_features(df1m)
    f5m = features['5m']

    # Pre-compute context arrays ONCE
    print("Pre-computing context arrays...")
    t0 = time.time()
    bar_dates = np.array([t.date() for t in f5m.index])
    daily_arrays = precompute_daily_map(features, bar_dates)
    htf_arrays = precompute_hourly_arrays(features, f5m.index)
    precomp = (daily_arrays, htf_arrays)
    print(f"  Context pre-computed in {time.time()-t0:.1f}s")

    # ── Experiment 1: Each signal with default params ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Individual signals (default params)")
    print("=" * 70)

    default_p = {
        'stop': 0.008, 'tp': 0.020,
        'd_thresh': 0.35, 'h4_thresh': 0.30, 'h1_thresh': 0.30,
        'f15_thresh': 0.25, 'f5_thresh': 0.15, 'rsi_thresh': 30,
        'min_tfs': 3,
        'bvc_thresh': 0.25,
        'entropy_thresh': 2.0,
        'div_thresh': 0.40,
        'atr_squeeze_thresh': 0.70,
        'min_agree': 2,
    }

    signals = [
        ('MTF_Bounce', signal_multi_tf_bounce),
        ('BVC_Accum', signal_bvc_accumulation),
        ('Entropy_Regime', signal_entropy_regime),
        ('CrossTF_Div', signal_cross_tf_divergence),
        ('ATR_Squeeze', signal_atr_squeeze_bounce),
        ('Composite_2', signal_composite),
    ]

    best_signal = None
    best_wr = 0
    best_pnl = 0

    for sname, sfn in signals:
        trades = simulate_v2(f5m, features, sfn, sname, default_p,
                             trail_base=0.010, trail_power=4,
                             _precomputed=precomp)
        if trades:
            validate(trades, sname)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w / n * 100
            pnl = sum(t.pnl for t in trades)
            if wr > best_wr or (wr == best_wr and pnl > best_pnl):
                best_wr = wr
                best_pnl = pnl
                best_signal = sname

    print(f"\nBest signal: {best_signal} ({best_wr:.1f}% WR, ${best_pnl:+,.0f})")

    # ── Experiment 2: Trail parameter sweep ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Trail base x trail power sweep")
    print("=" * 70)

    for sname, sfn in signals:
        print(f"\n--- {sname} ---")
        for tb in [0.005, 0.008, 0.010, 0.015, 0.020, 0.025]:
            for tp in [2, 4, 6, 8, 10, 12]:
                trades = simulate_v2(f5m, features, sfn, sname, default_p,
                                     trail_base=tb, trail_power=tp,
                                     _precomputed=precomp)
                if trades:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    flag = " ***" if wr >= 90 else ""
                    if n >= 5:  # only show meaningful results
                        print(f"  tb={tb:.3f} tp={tp:2d}: {n:4d}t {wr:5.1f}% "
                              f"${pnl:+10,.0f} BL=${bl:+,.0f}{flag}")

    # ── Experiment 3: Stop/TP sweep with best trail settings ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Stop/TP sweep (all signals, trail_base=0.010 tp=4)")
    print("=" * 70)

    for sname, sfn in [('MTF_Bounce', signal_multi_tf_bounce),
                        ('CrossTF_Div', signal_cross_tf_divergence)]:
        print(f"\n--- {sname} ---")
        for stop in [0.003, 0.005, 0.008, 0.010, 0.015, 0.020]:
            for tp in [0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050]:
                if tp <= stop:
                    continue
                p = dict(default_p, stop=stop, tp=tp)
                trades = simulate_v2(f5m, features, sfn, sname, p,
                                     _precomputed=precomp)
                if trades and len(trades) >= 5:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    flag = " ***" if wr >= 90 else ""
                    print(f"  s={stop:.3f} t={tp:.3f}: {n:4d}t {wr:5.1f}% "
                          f"${pnl:+10,.0f} BL=${bl:+,.0f}{flag}")

    # ── Experiment 4: Selectivity sweep (require more TFs) ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Selectivity (MTF_Bounce with varying min_tfs)")
    print("=" * 70)

    for min_tfs in [2, 3, 4]:
        for d_thresh in [0.25, 0.30, 0.35, 0.40]:
            for f5_thresh in [0.10, 0.15, 0.20]:
                p = dict(default_p, min_tfs=min_tfs, d_thresh=d_thresh,
                         f5_thresh=f5_thresh)
                trades = simulate_v2(f5m, features, signal_multi_tf_bounce,
                                     'MTF', p, _precomputed=precomp)
                if trades and len(trades) >= 3:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    flag = " ***" if wr >= 90 else ""
                    if wr >= 60:
                        print(f"  minTF={min_tfs} d<{d_thresh:.2f} 5m<{f5_thresh:.2f}: "
                              f"{n:4d}t {wr:5.1f}% ${pnl:+8,.0f} BL=${bl:+,.0f}{flag}")

    # ── Experiment 5: Multi-day hold (disable EOD exit) ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Multi-day hold (no forced EOD exit)")
    print("=" * 70)

    for sname, sfn in signals[:4]:
        for max_bars in [78, 156, 390, 780]:  # 1d, 2d, 5d, 10d
            for tb in [0.010, 0.015, 0.020]:
                trades = simulate_v2(f5m, features, sfn, sname, default_p,
                                     trail_base=tb, trail_power=4,
                                     max_hold_bars=max_bars, force_eod=False,
                                     _precomputed=precomp)
                if trades and len(trades) >= 3:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    avg_h = np.mean([t.hold_bars for t in trades])
                    flag = " ***" if wr >= 90 else ""
                    if wr >= 50 or pnl > 10000:
                        print(f"  {sname:15s} maxH={max_bars:3d} tb={tb:.3f}: "
                              f"{n:4d}t {wr:5.1f}% ${pnl:+10,.0f} BL=${bl:+,.0f} "
                              f"h={avg_h:.0f}b{flag}")

    # ── Experiment 6: Divergence threshold sweep ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Cross-TF divergence threshold sweep")
    print("=" * 70)

    for div in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        for f5t in [0.15, 0.20, 0.25, 0.30]:
            for tb in [0.010, 0.015, 0.020]:
                p = dict(default_p, div_thresh=div, f5_thresh=f5t)
                trades = simulate_v2(f5m, features, signal_cross_tf_divergence,
                                     'Div', p, trail_base=tb,
                                     _precomputed=precomp)
                if trades and len(trades) >= 3:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    flag = " ***" if wr >= 90 else ""
                    if wr >= 60:
                        print(f"  div>{div:.2f} 5m<{f5t:.2f} tb={tb:.3f}: "
                              f"{n:4d}t {wr:5.1f}% ${pnl:+10,.0f} BL=${bl:+,.0f}{flag}")

    # ── Experiment 7: BVC threshold sweep ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: BVC accumulation threshold sweep")
    print("=" * 70)

    for bvc_t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        for f5t in [0.10, 0.15, 0.20, 0.25]:
            for d_t in [0.30, 0.40, 0.50]:
                p = dict(default_p, bvc_thresh=bvc_t, f5_thresh=f5t, d_thresh=d_t)
                for tb in [0.010, 0.015, 0.020]:
                    trades = simulate_v2(f5m, features, signal_bvc_accumulation,
                                         'BVC', p, trail_base=tb,
                                         _precomputed=precomp)
                    if trades and len(trades) >= 3:
                        n = len(trades)
                        w = sum(1 for t in trades if t.pnl > 0)
                        wr = w / n * 100
                        pnl = sum(t.pnl for t in trades)
                        bl = min(t.pnl for t in trades)
                        flag = " ***" if wr >= 90 else ""
                        if wr >= 70:
                            print(f"  bvc>{bvc_t:.2f} 5m<{f5t:.2f} d<{d_t:.2f} tb={tb:.3f}: "
                                  f"{n:4d}t {wr:5.1f}% ${pnl:+8,.0f} BL=${bl:+,.0f}{flag}")

    print("\nDone.")

if __name__ == '__main__':
    main()
