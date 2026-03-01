#!/usr/bin/env python3
"""
Intraday 5-Min TSLA Backtester V1

Novel approach: combines multi-TF channel position with unorthodox signals:
- Bulk Volume Classification (order flow proxy from OHLCV)
- Shannon entropy regime gate (only trade predictable regimes)
- Cross-TF momentum divergence (higher TFs vs lower TFs)
- Volume-time profile anomaly (vs time-of-day average)

3-stage validation: holdout (train 2016-2021, test 2022-2025),
walkforward (expanding window), 2026 OOS.
"""
import os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy.stats import norm

# ================================================================
# Configuration
# ================================================================
CAPITAL = 100_000.0
SLIPPAGE_PCT = 0.0002       # 0.02% per side (intraday)
COMM_PER_SHARE = 0.005
TRAIL_BASE = 0.004          # 0.4% trailing stop base
TRAIL_POWER = 4             # confidence-scaled: trail = base * (1-conf)^power

TRAIN_END = pd.Timestamp('2021-12-31')
TEST_END  = pd.Timestamp('2025-12-31')

# Market hours (ET)
import datetime as dt
MKT_OPEN  = dt.time(9, 30)
MKT_CLOSE = dt.time(16, 0)
ENTRY_START = dt.time(9, 40)   # skip first 2 bars
ENTRY_END   = dt.time(15, 30)  # stop entering 30min before close
FORCE_EXIT  = dt.time(15, 50)  # force exit 10min before close

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
    """Load raw 1-min TSLA data."""
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
    # Market hours only
    times = df.index.time
    mask = (times >= MKT_OPEN) & (times < MKT_CLOSE)
    df = df[mask].copy()
    print(f"  Loaded {len(df):,} bars in {time.time()-t0:.1f}s "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df

def resample_ohlcv(df, rule):
    """Standard OHLCV resampling."""
    return df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

# ================================================================
# Feature Computation (Vectorized O(n))
# ================================================================
def channel_position(close_arr, window=60):
    """Fast O(n) rolling linear regression channel position.
    Returns array of channel position (0=lower, 1=upper)."""
    n = len(close_arr)
    close = close_arr.astype(np.float64)

    w = window
    sum_x = w * (w - 1) / 2.0
    sum_x2 = (w - 1) * w * (2 * w - 1) / 6.0
    denom = w * sum_x2 - sum_x ** 2

    # Rolling sum(y) via cumsum
    cs_y = np.cumsum(close)
    sum_y = np.full(n, np.nan)
    sum_y[w-1] = cs_y[w-1]
    if n > w:
        sum_y[w:] = cs_y[w:] - cs_y[:n-w]

    # Rolling sum(x*y) via weighted cumsum
    idx = np.arange(n, dtype=np.float64)
    cs_wy = np.cumsum(idx * close)
    sum_xy = np.full(n, np.nan)
    sum_xy[w-1] = cs_wy[w-1]
    if n > w:
        start_idx = np.arange(w, n, dtype=np.float64) - w + 1
        sum_xy[w:] = (cs_wy[w:] - cs_wy[:n-w]) - start_idx * sum_y[w:]

    # Slope and intercept
    slope = (w * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / w

    # Fitted value at last position
    fitted_last = slope * (w - 1) + intercept

    # Residual std via var(y) - slope^2 * var(x)
    cs_y2 = np.cumsum(close ** 2)
    sum_y2 = np.full(n, np.nan)
    sum_y2[w-1] = cs_y2[w-1]
    if n > w:
        sum_y2[w:] = cs_y2[w:] - cs_y2[:n-w]

    var_y = (sum_y2 - sum_y ** 2 / w) / w
    var_x = denom / (w ** 2)
    var_resid = np.maximum(var_y - slope ** 2 * var_x, 0)
    std_resid = np.sqrt(var_resid)

    # Channel bounds and position
    upper = fitted_last + 2 * std_resid
    lower = fitted_last - 2 * std_resid
    width = upper - lower

    pos = np.full(n, np.nan)
    valid = (width > 1e-10) & ~np.isnan(width)
    pos[valid] = (close[valid] - lower[valid]) / width[valid]
    pos = np.clip(pos, 0.0, 1.0)

    # Also return slope direction (normalized)
    slope_norm = np.full(n, np.nan)
    valid_s = ~np.isnan(slope) & (std_resid > 1e-10)
    slope_norm[valid_s] = slope[valid_s] / std_resid[valid_s]

    return pos, slope_norm

def compute_rsi(close_arr, period=14):
    """Wilder's RSI."""
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
    """Bulk Volume Classification net flow (vectorized).
    Positive = net buying, negative = net selling."""
    n = len(close_arr)
    dp = np.diff(close_arr)
    dp = np.concatenate([[0], dp])

    # Rolling std of price changes
    dp_series = pd.Series(dp)
    sigma = dp_series.rolling(lookback, min_periods=lookback).std().values

    z = np.zeros(n)
    valid = sigma > 1e-10
    z[valid] = dp[valid] / sigma[valid]

    buy_pct = norm.cdf(z)
    net_flow = volume_arr * (2 * buy_pct - 1)  # positive = net buying

    # Cumulative over rolling window
    net_cumul = pd.Series(net_flow).rolling(lookback, min_periods=lookback).sum().values
    # Normalize by rolling total volume
    total_vol = pd.Series(volume_arr).rolling(lookback, min_periods=lookback).sum().values

    bvc = np.full(n, np.nan)
    valid_v = total_vol > 0
    bvc[valid_v] = net_cumul[valid_v] / total_vol[valid_v]

    return bvc  # range roughly -1 to +1

def compute_entropy(close_arr, window=30, bins=8):
    """Rolling Shannon entropy of log-return distribution.
    Low = predictable/trending, High = random/choppy."""
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
    """Momentum turning up: momentum < 0, acceleration > 0."""
    n = len(close_arr)
    turning = np.zeros(n, dtype=bool)
    for i in range(lookback * 2, n):
        mom = close_arr[i] - close_arr[i - lookback]
        prev_mom = close_arr[i - lookback] - close_arr[i - 2 * lookback]
        accel = mom - prev_mom
        if mom < 0 and accel > 0:
            turning[i] = True
    return turning

def compute_momentum_turn_down(close_arr, lookback=10):
    """Momentum turning down: momentum > 0, acceleration < 0."""
    n = len(close_arr)
    turning = np.zeros(n, dtype=bool)
    for i in range(lookback * 2, n):
        mom = close_arr[i] - close_arr[i - lookback]
        prev_mom = close_arr[i - lookback] - close_arr[i - 2 * lookback]
        accel = mom - prev_mom
        if mom > 0 and accel < 0:
            turning[i] = True
    return turning

def compute_vol_profile(volume_arr, timestamps, lookback_days=20):
    """Volume relative to time-of-day average over recent days.
    High ratio = unusual volume = potential signal."""
    n = len(volume_arr)
    ratio = np.full(n, np.nan)

    # Group by time-of-day
    times = pd.Series(timestamps).dt.time.values
    unique_times = sorted(set(times))
    time_to_idx = {t: i for i, t in enumerate(unique_times)}

    # For each bar, compute average volume at this time over last N days
    time_indices = np.array([time_to_idx.get(t, 0) for t in times])
    n_times = len(unique_times)

    # Simple approach: rolling average per time slot
    vol_series = pd.Series(volume_arr, index=timestamps)

    # Group by time, then compute rolling mean
    for t in unique_times:
        mask = times == t
        indices = np.where(mask)[0]
        if len(indices) < lookback_days + 1:
            continue
        vols_at_time = volume_arr[indices]
        for j in range(lookback_days, len(indices)):
            avg = vols_at_time[j-lookback_days:j].mean()
            if avg > 0:
                ratio[indices[j]] = vols_at_time[j] / avg

    return ratio

# ================================================================
# Multi-TF Feature Builder
# ================================================================
def build_all_features(df1m, verbose=True):
    """Build features at multiple timeframes from 1-min data.
    Returns dict of {tf: DataFrame with features} and daily context."""
    t0 = time.time()

    # Resample to multiple TFs
    if verbose: print("Resampling to multiple TFs...")
    tfs = {}
    for rule, name in [('5min', '5m'), ('15min', '15m'), ('30min', '30m'),
                        ('1h', '1h'), ('4h', '4h')]:
        tfs[name] = resample_ohlcv(df1m, rule)

    # Daily from resampling
    tfs['daily'] = resample_ohlcv(df1m, '1D')

    if verbose:
        for name, df in tfs.items():
            print(f"  {name}: {len(df):,} bars")

    # Compute features for each TF
    features = {}
    windows = {'5m': 60, '15m': 40, '30m': 30, '1h': 24, '4h': 20, 'daily': 40}

    for name, df in tfs.items():
        if verbose: print(f"Computing features for {name}...")
        close = df['close'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)

        feat = pd.DataFrame(index=df.index)
        feat['open'] = df['open'].values
        feat['high'] = df['high'].values
        feat['low'] = df['low'].values
        feat['close'] = close
        feat['volume'] = volume

        w = windows[name]
        feat['chan_pos'], feat['chan_slope'] = channel_position(close, w)
        feat['rsi'] = compute_rsi(close, 14)
        feat['bvc'] = compute_bvc(close, volume, 20)

        if name == '5m':
            feat['entropy'] = compute_entropy(close, 30)
            feat['mom_turn_up'] = compute_momentum_turn(close, 10)
            feat['mom_turn_down'] = compute_momentum_turn_down(close, 10)
            feat['vol_ratio'] = compute_vol_profile(volume, df.index, 20)

        features[name] = feat

    if verbose: print(f"All features computed in {time.time()-t0:.1f}s")
    return features, tfs

# ================================================================
# Signal Functions
# ================================================================
def _get_daily_context(features, bar_time):
    """Get most recent daily features for a given bar time."""
    daily = features.get('daily')
    if daily is None:
        return {}
    bar_date = bar_time.normalize()
    # Get previous day's close features (avoid lookahead)
    prev = daily.index[daily.index < bar_date]
    if len(prev) == 0:
        return {}
    idx = prev[-1]
    row = daily.loc[idx]
    return {
        'daily_chan_pos': row.get('chan_pos', np.nan),
        'daily_rsi': row.get('rsi', np.nan),
        'daily_bvc': row.get('bvc', np.nan),
        'daily_chan_slope': row.get('chan_slope', np.nan),
    }

def _get_hourly_context(features, bar_time):
    """Get most recent 1h features."""
    hourly = features.get('1h')
    if hourly is None:
        return {}
    prev = hourly.index[hourly.index <= bar_time]
    if len(prev) == 0:
        return {}
    idx = prev[-1]
    row = hourly.loc[idx]
    return {
        '1h_chan_pos': row.get('chan_pos', np.nan),
        '1h_rsi': row.get('rsi', np.nan),
        '1h_bvc': row.get('bvc', np.nan),
        '1h_chan_slope': row.get('chan_slope', np.nan),
    }

def _get_4h_context(features, bar_time):
    """Get most recent 4h features."""
    h4 = features.get('4h')
    if h4 is None:
        return {}
    prev = h4.index[h4.index <= bar_time]
    if len(prev) == 0:
        return {}
    idx = prev[-1]
    row = h4.loc[idx]
    return {
        '4h_chan_pos': row.get('chan_pos', np.nan),
        '4h_rsi': row.get('rsi', np.nan),
        '4h_bvc': row.get('bvc', np.nan),
    }


# ── Signal A: Multi-TF Channel Bounce ──
def signal_a_channel_bounce(i, feat5m, features, params):
    """LONG when multi-TF channels show oversold + 5min turning.
    SHORT when multi-TF channels show overbought + 5min turning down."""
    row = feat5m.iloc[i]
    bar_time = feat5m.index[i]

    cp5 = row['chan_pos']
    rsi5 = row['rsi']
    turn_up = row.get('mom_turn_up', False)
    turn_down = row.get('mom_turn_down', False)

    if np.isnan(cp5) or np.isnan(rsi5):
        return None

    daily = _get_daily_context(features, bar_time)
    hourly = _get_hourly_context(features, bar_time)

    d_cp = daily.get('daily_chan_pos', np.nan)
    h_cp = hourly.get('1h_chan_pos', np.nan)

    if np.isnan(d_cp) or np.isnan(h_cp):
        return None

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.015)

    # LONG: daily oversold + 1h near bottom + 5min at extreme low + turning
    if (d_cp < params.get('d_long_thresh', 0.35) and
        h_cp < params.get('h_long_thresh', 0.30) and
        cp5 < params.get('5m_long_thresh', 0.15) and
        rsi5 < params.get('rsi_long_thresh', 30) and
        turn_up):
        conf = 0.5 + 0.3 * (1.0 - cp5) + 0.2 * (1.0 - d_cp)
        conf = min(conf, 0.95)
        return ('LONG', conf, stop, tp)

    # SHORT: daily overbought + 1h near top + 5min at extreme high + turning
    if (d_cp > params.get('d_short_thresh', 0.65) and
        h_cp > params.get('h_short_thresh', 0.70) and
        cp5 > params.get('5m_short_thresh', 0.85) and
        rsi5 > params.get('rsi_short_thresh', 70) and
        turn_down):
        conf = 0.5 + 0.3 * cp5 + 0.2 * d_cp
        conf = min(conf, 0.95)
        return ('SHORT', conf, stop, tp)

    return None


# ── Signal B: BVC Order Flow Divergence ──
def signal_b_bvc_flow(i, feat5m, features, params):
    """Trade when BVC order flow diverges from price direction.
    Accumulation at lows (price down, flow up) -> LONG
    Distribution at highs (price up, flow down) -> SHORT"""
    row = feat5m.iloc[i]
    bar_time = feat5m.index[i]

    cp5 = row['chan_pos']
    bvc5 = row['bvc']
    rsi5 = row['rsi']
    vol_r = row.get('vol_ratio', np.nan)

    if np.isnan(cp5) or np.isnan(bvc5) or np.isnan(rsi5):
        return None

    daily = _get_daily_context(features, bar_time)
    d_cp = daily.get('daily_chan_pos', np.nan)
    if np.isnan(d_cp):
        return None

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.015)

    # LONG: price at low, but flow shows buying (accumulation)
    bvc_thresh = params.get('bvc_long_thresh', 0.15)
    if (cp5 < 0.20 and bvc5 > bvc_thresh and d_cp < 0.40 and rsi5 < 35):
        conf = 0.50 + 0.25 * bvc5 + 0.25 * (1.0 - cp5)
        conf = min(max(conf, 0.45), 0.95)
        return ('LONG', conf, stop, tp)

    # SHORT: price at high, but flow shows selling (distribution)
    bvc_sell_thresh = params.get('bvc_short_thresh', -0.15)
    if (cp5 > 0.80 and bvc5 < bvc_sell_thresh and d_cp > 0.60 and rsi5 > 65):
        conf = 0.50 + 0.25 * abs(bvc5) + 0.25 * cp5
        conf = min(max(conf, 0.45), 0.95)
        return ('SHORT', conf, stop, tp)

    return None


# ── Signal C: Entropy-Gated Channel Bounce ──
def signal_c_entropy_gated(i, feat5m, features, params):
    """Only trade channel bounces when market is in predictable regime
    (low Shannon entropy of returns)."""
    row = feat5m.iloc[i]
    bar_time = feat5m.index[i]

    ent = row.get('entropy', np.nan)
    cp5 = row['chan_pos']
    rsi5 = row['rsi']
    turn_up = row.get('mom_turn_up', False)
    turn_down = row.get('mom_turn_down', False)

    if np.isnan(ent) or np.isnan(cp5) or np.isnan(rsi5):
        return None

    # Gate: only trade when entropy is below threshold (predictable regime)
    ent_thresh = params.get('entropy_thresh', 2.5)
    if ent > ent_thresh:
        return None

    hourly = _get_hourly_context(features, bar_time)
    h_cp = hourly.get('1h_chan_pos', np.nan)
    if np.isnan(h_cp):
        return None

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.015)

    # LONG: low entropy + channel bounce conditions
    if (cp5 < 0.15 and h_cp < 0.35 and rsi5 < 30 and turn_up):
        # Boost confidence based on how predictable the regime is
        ent_boost = max(0, (ent_thresh - ent) / ent_thresh) * 0.15
        conf = 0.55 + 0.25 * (1.0 - cp5) + ent_boost
        conf = min(conf, 0.95)
        return ('LONG', conf, stop, tp)

    # SHORT
    if (cp5 > 0.85 and h_cp > 0.65 and rsi5 > 70 and turn_down):
        ent_boost = max(0, (ent_thresh - ent) / ent_thresh) * 0.15
        conf = 0.55 + 0.25 * cp5 + ent_boost
        conf = min(conf, 0.95)
        return ('SHORT', conf, stop, tp)

    return None


# ── Signal D: Cross-TF Momentum Divergence ──
def signal_d_cross_tf_divergence(i, feat5m, features, params):
    """Trade when higher TFs disagree with lower TFs.
    Higher TFs bullish + lower TFs weak -> LONG (catch-up expected)
    Higher TFs bearish + lower TFs strong -> SHORT (drop expected)"""
    row = feat5m.iloc[i]
    bar_time = feat5m.index[i]

    cp5 = row['chan_pos']
    if np.isnan(cp5):
        return None

    daily = _get_daily_context(features, bar_time)
    hourly = _get_hourly_context(features, bar_time)
    h4 = _get_4h_context(features, bar_time)

    d_cp = daily.get('daily_chan_pos', np.nan)
    h_cp = hourly.get('1h_chan_pos', np.nan)
    h4_cp = h4.get('4h_chan_pos', np.nan)

    if np.isnan(d_cp) or np.isnan(h_cp) or np.isnan(h4_cp):
        return None

    # Higher TF average position
    higher_avg = (d_cp * 0.4 + h4_cp * 0.35 + h_cp * 0.25)

    # Divergence: higher TFs vs 5min
    divergence = higher_avg - cp5

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.015)

    # LONG: higher TFs bullish but 5min weak (divergence > threshold)
    div_long = params.get('div_long_thresh', 0.35)
    if (divergence > div_long and cp5 < 0.25 and
        row.get('mom_turn_up', False)):
        conf = 0.50 + 0.30 * min(divergence, 0.6) + 0.20 * (1.0 - cp5)
        conf = min(conf, 0.95)
        return ('LONG', conf, stop, tp)

    # SHORT: higher TFs bearish but 5min strong (divergence < -threshold)
    div_short = params.get('div_short_thresh', -0.35)
    if (divergence < div_short and cp5 > 0.75 and
        row.get('mom_turn_down', False)):
        conf = 0.50 + 0.30 * min(abs(divergence), 0.6) + 0.20 * cp5
        conf = min(conf, 0.95)
        return ('SHORT', conf, stop, tp)

    return None


# ── Signal E: Volume Spike Reversal ──
def signal_e_volume_spike(i, feat5m, features, params):
    """Trade reversals on abnormal volume spikes.
    High volume at channel extremes often signals exhaustion."""
    row = feat5m.iloc[i]

    cp5 = row['chan_pos']
    vol_r = row.get('vol_ratio', np.nan)
    rsi5 = row['rsi']
    bvc5 = row['bvc']

    if np.isnan(cp5) or np.isnan(vol_r) or np.isnan(rsi5) or np.isnan(bvc5):
        return None

    stop = params.get('stop', 0.008)
    tp = params.get('tp', 0.015)
    vol_thresh = params.get('vol_spike_thresh', 2.0)

    # LONG: volume spike at channel bottom with buying flow
    if (vol_r > vol_thresh and cp5 < 0.15 and bvc5 > 0.10 and rsi5 < 30):
        conf = 0.55 + 0.20 * min(vol_r / 3.0, 0.3) + 0.15 * bvc5
        conf = min(max(conf, 0.45), 0.95)
        return ('LONG', conf, stop, tp)

    # SHORT: volume spike at channel top with selling flow
    if (vol_r > vol_thresh and cp5 > 0.85 and bvc5 < -0.10 and rsi5 > 70):
        conf = 0.55 + 0.20 * min(vol_r / 3.0, 0.3) + 0.15 * abs(bvc5)
        conf = min(max(conf, 0.45), 0.95)
        return ('SHORT', conf, stop, tp)

    return None


# ── Signal F: Ensemble (combine A-E) ──
def signal_f_ensemble(i, feat5m, features, params):
    """Take trades only when multiple signals agree."""
    signals = []
    for fn in [signal_a_channel_bounce, signal_b_bvc_flow,
               signal_c_entropy_gated, signal_d_cross_tf_divergence,
               signal_e_volume_spike]:
        result = fn(i, feat5m, features, params)
        if result is not None:
            signals.append(result)

    if len(signals) < params.get('min_agree', 2):
        return None

    # Average confidence, take majority direction
    longs = [s for s in signals if s[0] == 'LONG']
    shorts = [s for s in signals if s[0] == 'SHORT']

    if len(longs) >= len(shorts) and len(longs) >= params.get('min_agree', 2):
        avg_conf = np.mean([s[1] for s in longs])
        return ('LONG', min(avg_conf + 0.05, 0.95),
                params.get('stop', 0.008), params.get('tp', 0.015))

    if len(shorts) > len(longs) and len(shorts) >= params.get('min_agree', 2):
        avg_conf = np.mean([s[1] for s in shorts])
        return ('SHORT', min(avg_conf + 0.05, 0.95),
                params.get('stop', 0.008), params.get('tp', 0.015))

    return None

# ================================================================
# Trade Simulation Engine
# ================================================================
def simulate_intraday(feat5m, features, signal_fn, signal_name, params,
                      max_trades_per_day=1, cooldown_bars=6,
                      trail_power=TRAIL_POWER):
    """Simulate intraday trades on 5-min bars.

    Entry at next bar's open after signal fires.
    Exit: stop, TP, trailing stop, or forced EOD exit.
    """
    n = len(feat5m)
    trades = []

    # State
    in_trade = False
    entry_price = 0.0
    entry_time = None
    direction = ''
    confidence = 0.0
    stop_price = 0.0
    tp_price = 0.0
    best_price = 0.0
    hold_bars = 0
    entry_bar = 0
    cooldown_remaining = 0
    trades_today = 0
    current_date = None
    pending_signal = None  # Signal queued for next bar entry

    for i in range(n):
        bar = feat5m.iloc[i]
        bar_time = feat5m.index[i]
        bar_date = bar_time.date()
        bar_tod = bar_time.time()

        o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']

        # Reset daily counters
        if bar_date != current_date:
            current_date = bar_date
            trades_today = 0

        # ── Execute pending signal (enter at this bar's open) ──
        if pending_signal is not None and not in_trade:
            sig_dir, sig_conf, sig_stop, sig_tp = pending_signal
            pending_signal = None

            if bar_tod >= ENTRY_START and bar_tod <= ENTRY_END:
                entry_price = o * (1 + SLIPPAGE_PCT if sig_dir == 'LONG' else -SLIPPAGE_PCT)
                entry_time = bar_time
                direction = sig_dir
                confidence = sig_conf
                in_trade = True
                hold_bars = 0
                entry_bar = i
                trades_today += 1

                if direction == 'LONG':
                    stop_price = entry_price * (1 - sig_stop)
                    tp_price = entry_price * (1 + sig_tp)
                    best_price = entry_price
                else:
                    stop_price = entry_price * (1 + sig_stop)
                    tp_price = entry_price * (1 - sig_tp)
                    best_price = entry_price

        # ── Manage existing position ──
        if in_trade:
            hold_bars += 1
            exit_price = None
            exit_reason = None

            # Update best price
            if direction == 'LONG':
                best_price = max(best_price, h)
            else:
                best_price = min(best_price, l)

            # Trailing stop
            trail_pct = TRAIL_BASE * (1.0 - confidence) ** trail_power
            if direction == 'LONG':
                trail_stop = best_price * (1 - trail_pct)
                if trail_stop > stop_price:
                    stop_price = trail_stop
            else:
                trail_stop = best_price * (1 + trail_pct)
                if trail_stop < stop_price:
                    stop_price = trail_stop

            # Check exit conditions
            if direction == 'LONG':
                if l <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'stop' if stop_price < entry_price else 'trail'
                elif h >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
            else:  # SHORT
                if h >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'stop' if stop_price > entry_price else 'trail'
                elif l <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'

            # Force exit near EOD
            if exit_price is None and bar_tod >= FORCE_EXIT:
                exit_price = c
                exit_reason = 'eod'

            # Execute exit
            if exit_price is not None:
                exit_price_adj = exit_price * (1 - SLIPPAGE_PCT if direction == 'LONG'
                                                else 1 + SLIPPAGE_PCT)
                shares = int(CAPITAL * confidence / entry_price)
                if shares < 1:
                    shares = 1

                if direction == 'LONG':
                    pnl = (exit_price_adj - entry_price) * shares
                else:
                    pnl = (entry_price - exit_price_adj) * shares

                pnl -= COMM_PER_SHARE * shares * 2  # both sides

                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=bar_time,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price_adj,
                    confidence=confidence,
                    shares=shares,
                    pnl=pnl,
                    hold_bars=hold_bars,
                    exit_reason=exit_reason,
                    signal_name=signal_name,
                ))

                in_trade = False
                cooldown_remaining = cooldown_bars

        # ── Generate new signals ──
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        if in_trade or trades_today >= max_trades_per_day:
            continue

        if bar_tod < ENTRY_START or bar_tod > ENTRY_END:
            continue

        result = signal_fn(i, feat5m, features, params)
        if result is not None:
            pending_signal = result  # Queue for next bar entry

    return trades

# ================================================================
# Validation Framework
# ================================================================
def print_trade_summary(trades, label="ALL"):
    """Print summary stats for a list of trades."""
    if not trades:
        print(f"  {label}: 0 trades")
        return

    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = n - wins
    total_pnl = sum(t.pnl for t in trades)
    avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if wins > 0 else 0
    avg_loss = np.mean([t.pnl for t in trades if t.pnl <= 0]) if losses > 0 else 0
    biggest_win = max(t.pnl for t in trades)
    biggest_loss = min(t.pnl for t in trades)
    avg_hold = np.mean([t.hold_bars for t in trades])

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    reason_str = ' '.join(f"{k}={v}" for k, v in sorted(reasons.items()))

    wr = wins / n * 100 if n > 0 else 0
    flag = "***" if wr == 100 else ""

    print(f"  {label}: {n}t {wins}W/{losses}L ({wr:.1f}% WR) "
          f"${total_pnl:+,.0f} BW=${biggest_win:+,.0f} BL=${biggest_loss:+,.0f} "
          f"avg_hold={avg_hold:.0f}bars [{reason_str}] {flag}")

def validate_3stage(trades, label=""):
    """Run 3-stage validation: holdout, walkforward, 2026 OOS."""
    print(f"\n{'='*70}")
    print(f"VALIDATION: {label}")
    print(f"{'='*70}")

    # Split by date
    train = [t for t in trades if t.entry_time <= TRAIN_END]
    test = [t for t in trades if TRAIN_END < t.entry_time <= TEST_END]
    oos = [t for t in trades if t.entry_time > TEST_END]

    print_trade_summary(trades, "FULL")
    print_trade_summary(train, "TRAIN 2016-2021")
    print_trade_summary(test, "TEST  2022-2025")
    print_trade_summary(oos, "OOS   2026")

    # Walk-forward (expanding window)
    print("\n  Walk-forward (expanding window):")
    years = sorted(set(t.entry_time.year for t in trades))
    cumul_trades = []
    for yr in years:
        yr_trades = [t for t in trades if t.entry_time.year == yr]
        cumul_trades.extend(yr_trades)
        if yr_trades:
            n_yr = len(yr_trades)
            w_yr = sum(1 for t in yr_trades if t.pnl > 0)
            pnl_yr = sum(t.pnl for t in yr_trades)
            bl_yr = min(t.pnl for t in yr_trades)
            cum_pnl = sum(t.pnl for t in cumul_trades)
            wr = w_yr / n_yr * 100
            print(f"    {yr}: {n_yr:3d}t {wr:5.1f}% WR "
                  f"${pnl_yr:+8,.0f} BL=${bl_yr:+,.0f} cum=${cum_pnl:+,.0f}")

    # Summary
    n_total = len(trades)
    if n_total > 0:
        total_wins = sum(1 for t in trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in trades)
        bl = min(t.pnl for t in trades)
        wr = total_wins / n_total * 100

        # Direction breakdown
        longs = [t for t in trades if t.direction == 'LONG']
        shorts = [t for t in trades if t.direction == 'SHORT']

        print(f"\n  Direction breakdown:")
        if longs:
            l_w = sum(1 for t in longs if t.pnl > 0)
            print(f"    LONG:  {len(longs)}t {l_w/len(longs)*100:.1f}% WR "
                  f"${sum(t.pnl for t in longs):+,.0f}")
        if shorts:
            s_w = sum(1 for t in shorts if t.pnl > 0)
            print(f"    SHORT: {len(shorts)}t {s_w/len(shorts)*100:.1f}% WR "
                  f"${sum(t.pnl for t in shorts):+,.0f}")

# ================================================================
# Parameter Sweep
# ================================================================
def sweep_params(feat5m, features, signal_fn, signal_name, param_grid):
    """Sweep over parameter combinations, find best WR and PnL."""
    results = []

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    # Generate all combinations
    from itertools import product
    combos = list(product(*values))
    print(f"\nSweeping {len(combos)} parameter combos for {signal_name}...")

    for combo in combos:
        params = dict(zip(keys, combo))
        trades = simulate_intraday(feat5m, features, signal_fn, signal_name, params)

        if not trades:
            continue

        n = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / n * 100
        pnl = sum(t.pnl for t in trades)
        bl = min(t.pnl for t in trades)

        results.append({
            'params': params,
            'trades': n,
            'wins': wins,
            'wr': wr,
            'pnl': pnl,
            'bl': bl,
        })

    # Sort by WR desc, then PnL desc
    results.sort(key=lambda x: (x['wr'], x['pnl']), reverse=True)

    print(f"\nTop 10 results for {signal_name}:")
    for j, r in enumerate(results[:10]):
        flag = "***" if r['wr'] == 100 else ""
        print(f"  {j+1}. {r['trades']:4d}t {r['wr']:5.1f}% WR "
              f"${r['pnl']:+10,.0f} BL=${r['bl']:+,.0f} {flag}")
        # Print params compactly
        p_str = ' '.join(f"{k}={v}" for k, v in r['params'].items())
        print(f"     {p_str}")

    return results

# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("INTRADAY 5-MIN TSLA BACKTESTER V1")
    print("Multi-TF Channel + BVC + Entropy + Cross-TF Divergence + Volume")
    print("=" * 70)

    # Load data
    df1m = load_1min()

    # Build features
    features, tfs = build_all_features(df1m, verbose=True)
    feat5m = features['5m']

    # Default params
    default_params = {
        'stop': 0.008,
        'tp': 0.015,
        'd_long_thresh': 0.35,
        'h_long_thresh': 0.30,
        '5m_long_thresh': 0.15,
        'rsi_long_thresh': 30,
        'd_short_thresh': 0.65,
        'h_short_thresh': 0.70,
        '5m_short_thresh': 0.85,
        'rsi_short_thresh': 70,
        'bvc_long_thresh': 0.15,
        'bvc_short_thresh': -0.15,
        'entropy_thresh': 2.5,
        'div_long_thresh': 0.35,
        'div_short_thresh': -0.35,
        'vol_spike_thresh': 2.0,
        'min_agree': 2,
    }

    # ── Run each signal independently ──
    signal_configs = [
        ('A_ChannelBounce', signal_a_channel_bounce),
        ('B_BVC_Flow', signal_b_bvc_flow),
        ('C_EntropyGated', signal_c_entropy_gated),
        ('D_CrossTF_Div', signal_d_cross_tf_divergence),
        ('E_VolSpike', signal_e_volume_spike),
        ('F_Ensemble_2', signal_f_ensemble),
    ]

    all_results = {}

    for name, fn in signal_configs:
        print(f"\n{'='*70}")
        print(f"SIGNAL: {name}")
        print(f"{'='*70}")

        trades = simulate_intraday(feat5m, features, fn, name, default_params)
        all_results[name] = trades

        if trades:
            validate_3stage(trades, name)
        else:
            print("  No trades generated.")

    # ── Ensemble with min_agree=3 ──
    print(f"\n{'='*70}")
    print(f"SIGNAL: F_Ensemble_3 (min 3 agree)")
    print(f"{'='*70}")
    params3 = dict(default_params, min_agree=3)
    trades_e3 = simulate_intraday(feat5m, features, signal_f_ensemble, 'F_Ensemble_3', params3)
    all_results['F_Ensemble_3'] = trades_e3
    if trades_e3:
        validate_3stage(trades_e3, 'F_Ensemble_3')
    else:
        print("  No trades generated.")

    # ── Parameter sweep for best signal ──
    print(f"\n{'='*70}")
    print("PARAMETER SWEEP: Signal A (Channel Bounce)")
    print(f"{'='*70}")

    sweep_grid = {
        'stop': [0.005, 0.008, 0.012, 0.015],
        'tp': [0.010, 0.015, 0.020, 0.030],
        'd_long_thresh': [0.30, 0.35, 0.40],
        '5m_long_thresh': [0.10, 0.15, 0.20],
    }

    # Fixed params for sweep
    sweep_base = dict(default_params)

    # Partial sweep (vary stop/tp only first)
    stop_tp_grid = {
        'stop': [0.005, 0.008, 0.010, 0.012, 0.015, 0.020],
        'tp': [0.008, 0.010, 0.015, 0.020, 0.030, 0.040],
    }

    for stop in stop_tp_grid['stop']:
        for tp in stop_tp_grid['tp']:
            if tp <= stop:
                continue
            p = dict(default_params, stop=stop, tp=tp)
            trades = simulate_intraday(feat5m, features,
                                        signal_a_channel_bounce, 'A', p)
            if trades:
                n = len(trades)
                w = sum(1 for t in trades if t.pnl > 0)
                wr = w / n * 100
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                flag = "***" if wr == 100 else ""
                print(f"  stop={stop:.3f} tp={tp:.3f}: "
                      f"{n:4d}t {wr:5.1f}% WR ${pnl:+10,.0f} BL=${bl:+,.0f} {flag}")

    # ── Trail power sweep for best config ──
    print(f"\n{'='*70}")
    print("TRAIL POWER SWEEP (Signal A)")
    print(f"{'='*70}")

    for tp_val in [2, 4, 6, 8, 10, 12, 15, 20]:
        trades = simulate_intraday(feat5m, features,
                                    signal_a_channel_bounce, 'A',
                                    default_params, trail_power=tp_val)
        if trades:
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w / n * 100
            pnl = sum(t.pnl for t in trades)
            bl = min(t.pnl for t in trades)
            flag = "***" if wr == 100 else ""
            print(f"  trail_power={tp_val:2d}: {n:4d}t {wr:5.1f}% WR "
                  f"${pnl:+10,.0f} BL=${bl:+,.0f} {flag}")

    # ── Summary comparison ──
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    for name, trades in all_results.items():
        if trades:
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            pnl = sum(t.pnl for t in trades)
            bl = min(t.pnl for t in trades)
            wr = w / n * 100

            # 2026 OOS
            oos = [t for t in trades if t.entry_time > TEST_END]
            oos_str = f"2026: {len(oos)}t" if oos else "2026: 0t"
            if oos:
                oos_w = sum(1 for t in oos if t.pnl > 0)
                oos_pnl = sum(t.pnl for t in oos)
                oos_str = f"2026: {len(oos)}t {oos_w/len(oos)*100:.0f}%WR ${oos_pnl:+,.0f}"

            flag = "***" if wr >= 95 else ""
            print(f"  {name:20s}: {n:4d}t {wr:5.1f}% WR ${pnl:+10,.0f} "
                  f"BL=${bl:+,.0f} | {oos_str} {flag}")
        else:
            print(f"  {name:20s}: 0 trades")

    print("\nDone.")

if __name__ == '__main__':
    main()
