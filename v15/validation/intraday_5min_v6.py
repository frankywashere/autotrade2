#!/usr/bin/env python3
"""
Intraday 5-Min TSLA Backtester V6

KEY INSIGHT: All trades exit in 1 bar. The system IS a 1-bar bet.
Avg win ~$150, avg loss ~$25. Trail power >= 4 makes trail ~0%.
Phase trail doesn't help because phase 2 never activates.

V6 Strategy: Accept 1-bar reality, maximize edge:
1. ATR-scaled trail_base: wider in volatile periods to capture more bar movement
2. Volume exhaustion signal: declining volume on down-bars = exhaustion bounce
3. Consecutive decline signal: 3+ bars declining channel position = sustained selloff
4. No momentum turn requirement for some signals (more trades)
5. 3 trades/day allowed (tight losses make this safe)
6. New: channel velocity (rate of position change) for timing
7. New: intraday range ratio (current range vs avg range)
"""
import os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List
from scipy.stats import norm
import datetime as dt

CAPITAL = 100_000.0
SLIPPAGE_PCT = 0.0002
COMM_PER_SHARE = 0.005
TRAIN_END = pd.Timestamp('2021-12-31')
TEST_END  = pd.Timestamp('2025-12-31')
MKT_OPEN  = dt.time(9, 30)
MKT_CLOSE = dt.time(16, 0)

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

def channel_position(close_arr, window=60):
    n = len(close_arr)
    close = close_arr.astype(np.float64)
    w = window
    sum_x = w * (w - 1) / 2.0
    sum_x2 = (w - 1) * w * (2 * w - 1) / 6.0
    denom = w * sum_x2 - sum_x ** 2
    cs_y = np.cumsum(close)
    sum_y = np.full(n, np.nan)
    sum_y[w-1] = cs_y[w-1]
    if n > w: sum_y[w:] = cs_y[w:] - cs_y[:n-w]
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
    if n > w: sum_y2[w:] = cs_y2[w:] - cs_y2[:n-w]
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
    if len(diff) < period: return rsi
    gains = np.maximum(diff, 0.0)
    losses = np.maximum(-diff, 0.0)
    ag = gains[:period].mean()
    al = losses[:period].mean()
    for i in range(period, len(diff)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        if al < 1e-10: rsi[i + 1] = 100.0
        else: rsi[i + 1] = 100.0 - 100.0 / (1.0 + ag / al)
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
    nc = pd.Series(net_flow).rolling(lookback, min_periods=lookback).sum().values
    tv = pd.Series(volume_arr).rolling(lookback, min_periods=lookback).sum().values
    bvc = np.full(n, np.nan)
    v = tv > 0
    bvc[v] = nc[v] / tv[v]
    return bvc

def compute_momentum_turn(close_arr, lookback=10):
    n = len(close_arr)
    turning = np.zeros(n, dtype=bool)
    for i in range(lookback * 2, n):
        mom = close_arr[i] - close_arr[i - lookback]
        prev_mom = close_arr[i - lookback] - close_arr[i - 2 * lookback]
        if mom < 0 and (mom - prev_mom) > 0:
            turning[i] = True
    return turning

def compute_vwap_intraday(open_arr, high_arr, low_arr, close_arr, volume_arr, dates_arr):
    n = len(close_arr)
    vwap = np.full(n, np.nan)
    vwap_dist = np.full(n, np.nan)
    typical = (high_arr + low_arr + close_arr) / 3.0
    cum_tv = cum_v = 0.0
    prev_date = None
    for i in range(n):
        d = dates_arr[i]
        if d != prev_date:
            cum_tv = cum_v = 0.0
            prev_date = d
        cum_tv += typical[i] * volume_arr[i]
        cum_v += volume_arr[i]
        if cum_v > 0:
            vwap[i] = cum_tv / cum_v
            vwap_dist[i] = (close_arr[i] - vwap[i]) / vwap[i] * 100.0
    return vwap, vwap_dist

def compute_reversal_bars(open_arr, high_arr, low_arr, close_arr):
    n = len(close_arr)
    reversal = np.zeros(n, dtype=bool)
    for i in range(1, n):
        body = close_arr[i] - open_arr[i]
        bar_range = high_arr[i] - low_arr[i]
        if bar_range < 1e-10: continue
        lower_shadow = min(open_arr[i], close_arr[i]) - low_arr[i]
        upper_shadow = high_arr[i] - max(open_arr[i], close_arr[i])
        if lower_shadow > 2 * abs(body) and lower_shadow > 2 * upper_shadow and body >= 0:
            reversal[i] = True
            continue
        prev_body = close_arr[i-1] - open_arr[i-1]
        if prev_body < 0 and body > 0 and body > abs(prev_body):
            if open_arr[i] <= close_arr[i-1] and close_arr[i] >= open_arr[i-1]:
                reversal[i] = True
    return reversal

def compute_micro_momentum(close_arr, window=5):
    n = len(close_arr)
    mom = np.full(n, np.nan)
    for i in range(window, n):
        if close_arr[i - window] > 0:
            mom[i] = (close_arr[i] - close_arr[i - window]) / close_arr[i - window] * 100.0
    return mom

def compute_channel_velocity(chan_pos_arr, lookback=5):
    """Rate of change of channel position - negative = falling fast."""
    n = len(chan_pos_arr)
    vel = np.full(n, np.nan)
    for i in range(lookback, n):
        if not np.isnan(chan_pos_arr[i]) and not np.isnan(chan_pos_arr[i-lookback]):
            vel[i] = chan_pos_arr[i] - chan_pos_arr[i-lookback]
    return vel

def compute_volume_exhaustion(close_arr, volume_arr, lookback=5):
    """Detect volume exhaustion on down moves: declining volume on consecutive down bars."""
    n = len(close_arr)
    exhaustion = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        # Check if last `lookback` bars are down with declining volume
        all_down = True
        vol_declining = True
        for j in range(i - lookback + 1, i + 1):
            if close_arr[j] >= close_arr[j-1]:
                all_down = False
                break
            if j > i - lookback + 1 and volume_arr[j] >= volume_arr[j-1]:
                vol_declining = False
        if all_down and vol_declining:
            exhaustion[i] = True
    return exhaustion

def compute_consecutive_decline(chan_pos_arr, lookback=3):
    """Count consecutive bars of declining channel position."""
    n = len(chan_pos_arr)
    count = np.zeros(n, dtype=int)
    for i in range(1, n):
        if not np.isnan(chan_pos_arr[i]) and not np.isnan(chan_pos_arr[i-1]):
            if chan_pos_arr[i] < chan_pos_arr[i-1]:
                count[i] = count[i-1] + 1
            else:
                count[i] = 0
    return count

def compute_atr_pct(high_arr, low_arr, close_arr, period=14):
    """ATR as percentage of close price."""
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
    atr_pct = np.full(n, np.nan)
    valid = close_arr > 0
    atr_pct[valid] = atr[valid] / close_arr[valid]
    return atr_pct

# ================================================================
# Build features
# ================================================================
def build_features(df1m, verbose=True):
    t0 = time.time()
    if verbose: print("Resampling to multiple TFs...")
    tfs = {}
    for rule, name in [('3min', '3m'), ('5min', '5m'), ('15min', '15m'),
                        ('30min', '30m'), ('1h', '1h'), ('4h', '4h')]:
        tfs[name] = resample_ohlcv(df1m, rule)
    tfs['daily'] = resample_ohlcv(df1m, '1D')
    tfs['1m'] = df1m.copy()

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
        cp = channel_position(close, w)
        feat['chan_pos'] = cp

        if name != '1m':
            feat['rsi'] = compute_rsi(close, 14)
            feat['bvc'] = compute_bvc(close, volume, 20)

        if name in ('5m', '3m'):
            feat['mom_turn'] = compute_momentum_turn(close, 10)

        if name == '5m':
            dates = np.array([t.date() for t in df.index])
            vwap, vwap_dist = compute_vwap_intraday(open_, high, low, close, volume, dates)
            feat['vwap'] = vwap
            feat['vwap_dist'] = vwap_dist
            feat['chan_vel'] = compute_channel_velocity(cp, 5)
            feat['vol_exhaust'] = compute_volume_exhaustion(close, volume, 3)
            feat['consec_decline'] = compute_consecutive_decline(cp, 1)  # raw count
            feat['atr_pct'] = compute_atr_pct(high, low, close, 14)

        if name in ('1m', '3m'):
            feat['reversal'] = compute_reversal_bars(open_, high, low, close)

        if name == '1m':
            feat['micro_mom'] = compute_micro_momentum(close, 5)

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
    d_dates = np.array([idx.date() if hasattr(idx, 'date') else idx for idx in daily.index])
    n = len(feat5m_dates)
    daily_cp_arr = np.full(n, np.nan)
    daily_rsi_arr = np.full(n, np.nan)
    daily_bvc_arr = np.full(n, np.nan)
    unique_5m_dates = sorted(set(feat5m_dates))
    date_to_daily = {}
    for d in unique_5m_dates:
        best = -1
        for k in range(len(d_dates)):
            if d_dates[k] < d: best = k
            elif d_dates[k] >= d: break
        if best >= 0: date_to_daily[d] = best
    for i in range(n):
        d = feat5m_dates[i]
        if d in date_to_daily:
            k = date_to_daily[d]
            daily_cp_arr[i] = d_cp[k]
            daily_rsi_arr[i] = d_rsi[k]
            daily_bvc_arr[i] = d_bvc[k]
    return daily_cp_arr, daily_rsi_arr, daily_bvc_arr

def precompute_htf_arrays(features, feat5m_index):
    n = len(feat5m_index)
    arrays = {
        '1h_cp': np.full(n, np.nan), '1h_rsi': np.full(n, np.nan),
        '1h_bvc': np.full(n, np.nan), '4h_cp': np.full(n, np.nan),
        '15m_cp': np.full(n, np.nan), '15m_rsi': np.full(n, np.nan),
        '30m_cp': np.full(n, np.nan),
        '3m_cp': np.full(n, np.nan), '3m_rsi': np.full(n, np.nan),
        '3m_turn': np.full(n, 0.0), '3m_reversal': np.full(n, 0.0),
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
        col_arrays = {k: tf_feat[v].values for k, v in col_map.items()}
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
            start = max(0, j - 4)
            if np.any(rev_vals[start:j+1] > 0):
                arrays['1m_reversal'][i] = 1.0
    return arrays

# ================================================================
# Signal Functions V6
# ================================================================

def signal_vwap_v6(i, f5m, ctx, params):
    """VWAP mean reversion V6: relaxed entry requirements."""
    cp5 = f5m['chan_pos'].iloc[i]
    vwap_dist = f5m['vwap_dist'].iloc[i]

    if np.isnan(cp5) or np.isnan(vwap_dist):
        return None

    d_cp = ctx['daily_cp']
    h_cp = ctx['1h_cp'][i]
    if np.isnan(d_cp) or np.isnan(h_cp):
        return None

    if vwap_dist > params.get('vwap_thresh', -0.30):
        return None
    if d_cp < params.get('d_min', 0.20):
        return None
    if h_cp < params.get('h1_min', 0.15):
        return None
    if cp5 > params.get('f5_thresh', 0.25):
        return None

    # V6: Relax turn requirement - accept ANY micro confirmation OR no confirmation
    need_turn = params.get('need_turn', True)
    if need_turn:
        turn_5m = f5m.get('mom_turn')
        has_turn_5m = turn_5m is not None and turn_5m.iloc[i]
        has_turn_3m = ctx['3m_turn'][i] > 0.5
        has_reversal_1m = ctx['1m_reversal'][i] > 0.5
        has_reversal_3m = ctx['3m_reversal'][i] > 0.5
        if not (has_turn_5m or has_turn_3m or has_reversal_1m or has_reversal_3m):
            return None

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)
    conf = 0.55 + min(abs(vwap_dist) * 0.05, 0.15) + 0.10 * (1.0 - cp5)
    conf = min(conf, 0.95)
    return ('LONG', conf, stop, tp)


def signal_divergence_v6(i, f5m, ctx, params):
    """Cross-TF divergence V6: same core logic."""
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

    if divergence < params.get('div_thresh', 0.35):
        return None
    if cp5 > params.get('f5_thresh', 0.30):
        return None

    need_turn = params.get('need_turn', True)
    if need_turn:
        turn_5m = f5m.get('mom_turn')
        has_turn_5m = turn_5m is not None and turn_5m.iloc[i]
        has_turn_3m = ctx['3m_turn'][i] > 0.5
        has_reversal_1m = ctx['1m_reversal'][i] > 0.5
        has_reversal_3m = ctx['3m_reversal'][i] > 0.5
        if not (has_turn_5m or has_turn_3m or has_reversal_1m or has_reversal_3m):
            return None

    vwap_dist = f5m['vwap_dist'].iloc[i]
    vwap_boost = 0.0
    if not np.isnan(vwap_dist) and vwap_dist < 0:
        vwap_boost = min(abs(vwap_dist) * 0.02, 0.10)

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)
    conf = 0.55 + 0.25 * min(divergence, 0.7) + 0.10 * (1.0 - cp5) + vwap_boost
    conf = min(conf, 0.95)
    return ('LONG', conf, stop, tp)


def signal_exhaustion_bounce(i, f5m, ctx, params):
    """Volume exhaustion bounce: declining volume on consecutive down bars."""
    cp5 = f5m['chan_pos'].iloc[i]
    if np.isnan(cp5):
        return None

    vol_exhaust = f5m['vol_exhaust'].iloc[i]
    if not vol_exhaust:
        return None

    d_cp = ctx['daily_cp']
    h_cp = ctx['1h_cp'][i]
    if np.isnan(d_cp) or np.isnan(h_cp):
        return None

    if cp5 > params.get('f5_thresh', 0.30):
        return None
    if d_cp < params.get('d_min', 0.20):
        return None

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)

    # Exhaustion at channel bottom = high confidence
    conf = 0.60 + 0.15 * (1.0 - cp5) + 0.05 * max(0, d_cp - 0.3)
    conf = min(conf, 0.95)
    return ('LONG', conf, stop, tp)


def signal_consec_decline(i, f5m, ctx, params):
    """Consecutive decline bounce: N+ bars of declining channel position."""
    cp5 = f5m['chan_pos'].iloc[i]
    if np.isnan(cp5):
        return None

    consec = f5m['consec_decline'].iloc[i]
    min_bars = params.get('consec_min', 5)
    if consec < min_bars:
        return None

    d_cp = ctx['daily_cp']
    h_cp = ctx['1h_cp'][i]
    if np.isnan(d_cp) or np.isnan(h_cp):
        return None

    if cp5 > params.get('f5_thresh', 0.25):
        return None
    if d_cp < params.get('d_min', 0.20):
        return None

    # Need some sign of turning
    chan_vel = f5m['chan_vel'].iloc[i]
    if not np.isnan(chan_vel) and chan_vel > params.get('vel_thresh', -0.05):
        # Velocity becoming less negative = deceleration
        pass
    else:
        return None

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)
    conf = 0.55 + 0.05 * min(consec - min_bars, 5) + 0.10 * (1.0 - cp5)
    conf = min(conf, 0.95)
    return ('LONG', conf, stop, tp)


def signal_channel_velocity_bounce(i, f5m, ctx, params):
    """Channel velocity bounce: rapid channel descent then deceleration.
    Catches rubber-band snapback after sharp drops."""
    cp5 = f5m['chan_pos'].iloc[i]
    chan_vel = f5m['chan_vel'].iloc[i]

    if np.isnan(cp5) or np.isnan(chan_vel):
        return None

    d_cp = ctx['daily_cp']
    if np.isnan(d_cp):
        return None

    if cp5 > params.get('f5_thresh', 0.25):
        return None
    if d_cp < params.get('d_min', 0.20):
        return None

    # Velocity should be negative (declining) but decelerating
    # Check previous velocity
    if i < 5:
        return None
    prev_vel = f5m['chan_vel'].iloc[i-1] if i > 0 else np.nan
    if np.isnan(prev_vel):
        return None

    # Deceleration: current vel > previous vel (less negative)
    accel = chan_vel - prev_vel
    if accel < params.get('accel_thresh', 0.02):
        return None
    if chan_vel > params.get('vel_max', 0.0):
        return None  # already turned positive
    if prev_vel > params.get('prev_vel_min', -0.05):
        return None  # wasn't dropping fast enough

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.020)
    conf = 0.55 + 0.15 * min(accel, 0.3) + 0.10 * (1.0 - cp5)
    conf = min(conf, 0.95)
    return ('LONG', conf, stop, tp)


def signal_mega_union(i, f5m, ctx, params):
    """Union of all V6 signals."""
    best = None
    best_conf = 0.0
    for fn in [signal_vwap_v6, signal_divergence_v6, signal_exhaustion_bounce,
               signal_consec_decline, signal_channel_velocity_bounce]:
        result = fn(i, f5m, ctx, params)
        if result is not None and result[1] > best_conf:
            best = result
            best_conf = result[1]
    return best


# ================================================================
# Trade Simulation V6 (ATR-scaled trail base)
# ================================================================
def simulate_v6(feat5m, features, signal_fn, name, params,
                trail_base=0.010, trail_power=4,
                atr_scale_trail=False,  # scale trail_base by ATR
                max_hold_bars=78, force_eod=True,
                cooldown_bars=6, max_trades_per_day=2,
                _precomputed=None):
    daily_arrays, htf_arrays, micro_arrays = _precomputed
    daily_cp_arr = daily_arrays[0]

    o_arr = feat5m['open'].values
    h_arr = feat5m['high'].values
    l_arr = feat5m['low'].values
    c_arr = feat5m['close'].values
    atr_pct_arr = feat5m['atr_pct'].values if 'atr_pct' in feat5m.columns else None
    times = feat5m.index
    time_of_day = np.array([t.time() for t in times])

    n = len(feat5m)
    trades = []
    in_trade = False
    entry_price = entry_time = None
    confidence = stop_price = tp_price = best_price = 0.0
    hold_bars = cooldown_remaining = trades_today = 0
    current_date = None
    pending_signal = None

    ctx = {
        'daily_cp': 0.0,
        '1h_cp': htf_arrays['1h_cp'], '4h_cp': htf_arrays['4h_cp'],
        '15m_cp': htf_arrays['15m_cp'], '15m_rsi': htf_arrays['15m_rsi'],
        '30m_cp': htf_arrays['30m_cp'],
        '1h_rsi': htf_arrays['1h_rsi'], '1h_bvc': htf_arrays['1h_bvc'],
        '3m_cp': htf_arrays['3m_cp'], '3m_rsi': htf_arrays['3m_rsi'],
        '3m_turn': htf_arrays['3m_turn'], '3m_reversal': htf_arrays['3m_reversal'],
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

        if in_trade:
            hold_bars += 1
            exit_price = None
            exit_reason = None
            best_price = max(best_price, h)

            # ATR-scaled trail base
            tb = trail_base
            if atr_scale_trail and atr_pct_arr is not None and not np.isnan(atr_pct_arr[i]):
                # Scale trail base proportional to current ATR
                # Normalize: avg ATR ~0.015, so scale = atr/0.015
                atr_scale = max(0.3, min(3.0, atr_pct_arr[i] / 0.015))
                tb = trail_base * atr_scale

            trail_pct = tb * (1.0 - confidence) ** trail_power
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
    ah = np.mean([t.hold_bars for t in trades])
    aw = np.mean([t.pnl for t in trades if t.pnl > 0]) if wins else 0
    al = np.mean([t.pnl for t in trades if t.pnl <= 0]) if n - wins else 0
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    r_str = ' '.join(f"{k}={v}" for k, v in sorted(reasons.items()))
    flag = " ***" if wr >= 90 else ""
    print(f"  {label}: {n}t {wins}W/{n-wins}L ({wr:.1f}% WR) "
          f"${pnl:+,.0f} BW=${bw:+,.0f} BL=${bl:+,.0f} "
          f"aW=${aw:+,.0f} aL=${al:+,.0f} h={ah:.1f}b [{r_str}]{flag}")
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
# Main
# ================================================================
def main():
    print("=" * 70)
    print("INTRADAY 5-MIN V6: ATR-SCALED TRAIL + NEW SIGNALS")
    print("=" * 70)

    df1m = load_1min()
    features, tfs = build_features(df1m)
    f5m = features['5m']

    print("Pre-computing context arrays...")
    t0 = time.time()
    bar_dates = np.array([t.date() for t in f5m.index])
    daily_arrays = precompute_daily_map(features, bar_dates)
    htf_arrays = precompute_htf_arrays(features, f5m.index)
    micro_arrays = precompute_1m_context(features, f5m.index)
    precomp = (daily_arrays, htf_arrays, micro_arrays)
    print(f"  Context pre-computed in {time.time()-t0:.1f}s")

    default_p = {
        'stop': 0.008, 'tp': 0.020,
        'd_min': 0.20, 'h1_min': 0.15,
        'f5_thresh': 0.25, 'div_thresh': 0.35,
        'vwap_thresh': -0.30,
        'need_turn': True,
        'consec_min': 5, 'vel_thresh': -0.05,
        'accel_thresh': 0.02, 'vel_max': 0.0, 'prev_vel_min': -0.05,
    }

    # ── EXP 1: Individual V6 signals ──
    print("\n" + "=" * 70)
    print("EXP 1: Individual V6 signals (defaults)")
    print("=" * 70)

    signals = [
        ('VWAP_v6', signal_vwap_v6),
        ('Div_v6', signal_divergence_v6),
        ('Exhaust', signal_exhaustion_bounce),
        ('ConsecDecl', signal_consec_decline),
        ('ChanVelBounce', signal_channel_velocity_bounce),
        ('MegaUnion', signal_mega_union),
    ]

    for sname, sfn in signals:
        trades = simulate_v6(f5m, features, sfn, sname, default_p,
                             trail_base=0.010, trail_power=4,
                             _precomputed=precomp)
        if trades:
            validate(trades, sname)

    # ── EXP 2: No-turn requirement ──
    print("\n" + "=" * 70)
    print("EXP 2: VWAP/Div WITHOUT momentum turn requirement")
    print("=" * 70)

    for sname, sfn in [('VWAP_noTurn', signal_vwap_v6),
                        ('Div_noTurn', signal_divergence_v6)]:
        p = dict(default_p, need_turn=False)
        trades = simulate_v6(f5m, features, sfn, sname, p,
                             trail_base=0.010, trail_power=4,
                             _precomputed=precomp)
        if trades:
            validate(trades, sname)

    # ── EXP 3: ATR-scaled trail ──
    print("\n" + "=" * 70)
    print("EXP 3: ATR-scaled trail base (MegaUnion)")
    print("=" * 70)

    for atr_sc in [False, True]:
        for tb in [0.005, 0.008, 0.010, 0.015]:
            for tp in [4, 8, 12]:
                label = f"atr={'Y' if atr_sc else 'N'} tb={tb:.3f} tp={tp}"
                trades = simulate_v6(f5m, features, signal_mega_union,
                                     'union', default_p,
                                     trail_base=tb, trail_power=tp,
                                     atr_scale_trail=atr_sc,
                                     _precomputed=precomp)
                if trades and len(trades) >= 5:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    flag = " ***" if wr >= 90 else ""
                    if wr >= 60 or pnl > 400000:
                        print(f"  {label}: {n:4d}t {wr:5.1f}% "
                              f"${pnl:+10,.0f} BL=${bl:+,.0f}{flag}")

    # ── EXP 4: Cooldown + max trades/day ──
    print("\n" + "=" * 70)
    print("EXP 4: Cooldown + trades/day sweep (MegaUnion)")
    print("=" * 70)

    for cd in [0, 2, 3, 6, 12]:
        for mtd in [1, 2, 3, 5]:
            trades = simulate_v6(f5m, features, signal_mega_union,
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

    # ── EXP 5: VWAP parameter mega-sweep ──
    print("\n" + "=" * 70)
    print("EXP 5: VWAP parameter sweep")
    print("=" * 70)

    for vt in [-0.05, -0.10, -0.15, -0.20, -0.30, -0.50]:
        for f5t in [0.15, 0.20, 0.25, 0.30, 0.40]:
            for dm in [0.10, 0.15, 0.20, 0.30]:
                for hm in [0.10, 0.15, 0.20]:
                    for nt in [True, False]:
                        p = dict(default_p, vwap_thresh=vt, f5_thresh=f5t,
                                 d_min=dm, h1_min=hm, need_turn=nt)
                        trades = simulate_v6(f5m, features, signal_vwap_v6,
                                             'vwap', p,
                                             trail_base=0.010, trail_power=4,
                                             _precomputed=precomp)
                        if trades and len(trades) >= 5:
                            n = len(trades)
                            w = sum(1 for t in trades if t.pnl > 0)
                            wr = w / n * 100
                            pnl = sum(t.pnl for t in trades)
                            bl = min(t.pnl for t in trades)
                            flag = " ***" if wr >= 90 else ""
                            nt_s = "turn" if nt else "noTurn"
                            if wr >= 82 or (wr >= 78 and pnl > 350000):
                                print(f"  vwap<{vt:.2f} 5m<{f5t:.2f} d>{dm:.2f} h1>{hm:.2f} {nt_s}: "
                                      f"{n:4d}t {wr:5.1f}% ${pnl:+10,.0f} BL=${bl:+,.0f}{flag}")

    # ── EXP 6: Exhaustion bounce sweep ──
    print("\n" + "=" * 70)
    print("EXP 6: Volume exhaustion bounce sweep")
    print("=" * 70)

    for f5t in [0.15, 0.20, 0.25, 0.30, 0.40]:
        for dm in [0.10, 0.15, 0.20, 0.30]:
            p = dict(default_p, f5_thresh=f5t, d_min=dm)
            trades = simulate_v6(f5m, features, signal_exhaustion_bounce,
                                 'exhaust', p,
                                 trail_base=0.010, trail_power=4,
                                 _precomputed=precomp)
            if trades and len(trades) >= 3:
                n = len(trades)
                w = sum(1 for t in trades if t.pnl > 0)
                wr = w / n * 100
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                flag = " ***" if wr >= 90 else ""
                if wr >= 60 or pnl > 10000:
                    print(f"  5m<{f5t:.2f} d>{dm:.2f}: {n:4d}t {wr:5.1f}% "
                          f"${pnl:+8,.0f} BL=${bl:+,.0f}{flag}")

    # ── EXP 7: Consecutive decline sweep ──
    print("\n" + "=" * 70)
    print("EXP 7: Consecutive decline bounce sweep")
    print("=" * 70)

    for cm in [3, 4, 5, 7, 10]:
        for f5t in [0.15, 0.20, 0.25, 0.30]:
            for vt in [-0.10, -0.05, 0.0, 0.05]:
                p = dict(default_p, consec_min=cm, f5_thresh=f5t, vel_thresh=vt)
                trades = simulate_v6(f5m, features, signal_consec_decline,
                                     'consec', p,
                                     trail_base=0.010, trail_power=4,
                                     _precomputed=precomp)
                if trades and len(trades) >= 3:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    flag = " ***" if wr >= 90 else ""
                    if wr >= 60 or pnl > 10000:
                        print(f"  min={cm} 5m<{f5t:.2f} vel>{vt:.2f}: {n:4d}t {wr:5.1f}% "
                              f"${pnl:+8,.0f} BL=${bl:+,.0f}{flag}")

    # ── EXP 8: Channel velocity bounce sweep ──
    print("\n" + "=" * 70)
    print("EXP 8: Channel velocity bounce sweep")
    print("=" * 70)

    for at in [0.01, 0.02, 0.03, 0.05]:
        for vm in [-0.05, 0.0, 0.05]:
            for pvm in [-0.10, -0.05, -0.03]:
                for f5t in [0.20, 0.25, 0.30]:
                    p = dict(default_p, accel_thresh=at, vel_max=vm,
                             prev_vel_min=pvm, f5_thresh=f5t)
                    trades = simulate_v6(f5m, features, signal_channel_velocity_bounce,
                                         'chanvel', p,
                                         trail_base=0.010, trail_power=4,
                                         _precomputed=precomp)
                    if trades and len(trades) >= 3:
                        n = len(trades)
                        w = sum(1 for t in trades if t.pnl > 0)
                        wr = w / n * 100
                        pnl = sum(t.pnl for t in trades)
                        bl = min(t.pnl for t in trades)
                        flag = " ***" if wr >= 90 else ""
                        if wr >= 65 or pnl > 30000:
                            print(f"  accel>{at:.2f} vel<{vm:.2f} pv<{pvm:.2f} 5m<{f5t:.2f}: "
                                  f"{n:4d}t {wr:5.1f}% ${pnl:+8,.0f} BL=${bl:+,.0f}{flag}")

    print("\n" + "=" * 70)
    print("V6 experiments complete.")
    print("=" * 70)

if __name__ == '__main__':
    main()
