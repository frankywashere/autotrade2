#!/usr/bin/env python3
"""
Intraday 5-Min TSLA Backtester V3

Key changes from V2:
1. DEFERRED TRAILING STOP: Trail only activates after trade is up by min_profit%
2. ATR-BASED STOPS: Dynamic stop distance based on 5-min ATR
3. MINIMUM HOLD: Must hold at least N bars before exit allowed (except hard stop)
4. TIME-OF-DAY FILTER: Identify best trading hours
5. COMBINED SIGNALS: Require 2+ independent signals to agree
6. HOLD OVERNIGHT: Option to hold positions through close
"""
import os, sys, time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
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

# ================================================================
# Data Loading + Features (reuse from V2)
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
    print(f"Loading {path}...")
    t0 = time.time()
    df = pd.read_csv(path, sep=';',
                     names=['datetime','open','high','low','close','volume'],
                     parse_dates=['datetime'], date_format='%Y%m%d %H%M%S')
    df = df.set_index('datetime').sort_index()
    mask = (df.index.time >= MKT_OPEN) & (df.index.time < MKT_CLOSE)
    df = df[mask].copy()
    print(f"  {len(df):,} bars in {time.time()-t0:.1f}s")
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
        rsi[i + 1] = 100.0 if al < 1e-10 else 100.0 - 100.0 / (1.0 + ag / al)
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
    vv = total_vol > 0
    bvc[vv] = net_cumul[vv] / total_vol[vv]
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

# ================================================================
# Feature Builder
# ================================================================
def build_features(df1m, verbose=True):
    t0 = time.time()
    if verbose: print("Resampling...")
    tfs = {}
    for rule, name in [('5min', '5m'), ('15min', '15m'), ('30min', '30m'),
                        ('1h', '1h'), ('4h', '4h')]:
        tfs[name] = resample_ohlcv(df1m, rule)
    tfs['daily'] = resample_ohlcv(df1m, '1D')
    if verbose:
        for nm, df in tfs.items():
            print(f"  {nm}: {len(df):,}")

    features = {}
    windows = {'5m': 60, '15m': 40, '30m': 30, '1h': 24, '4h': 20, 'daily': 40}
    for nm, df in tfs.items():
        if verbose: print(f"Features {nm}...")
        c = df['close'].values.astype(np.float64)
        v = df['volume'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        lo = df['low'].values.astype(np.float64)
        feat = pd.DataFrame(index=df.index)
        feat['open'] = df['open'].values
        feat['high'] = h
        feat['low'] = lo
        feat['close'] = c
        feat['volume'] = v
        feat['chan_pos'] = channel_position(c, windows[nm])
        feat['rsi'] = compute_rsi(c, 14)
        feat['bvc'] = compute_bvc(c, v, 20)
        feat['atr'] = compute_atr(h, lo, c, 14)
        if nm in ('5m', '15m'):
            feat['entropy'] = compute_entropy(c, 30)
            feat['mom_turn'] = compute_momentum_turn(c, 10)
        features[nm] = feat
    if verbose: print(f"Done in {time.time()-t0:.1f}s")
    return features, tfs

# ================================================================
# Pre-compute Context Arrays
# ================================================================
def precompute_context(features, f5m_index):
    """Build all context arrays aligned to 5m index."""
    t0 = time.time()
    n = len(f5m_index)

    # Daily: use PREVIOUS day (no lookahead)
    daily = features['daily']
    d_cp = daily['chan_pos'].values
    d_rsi = daily['rsi'].values
    d_bvc = daily['bvc'].values
    d_dates = np.array([idx.date() for idx in daily.index])
    f5m_dates = np.array([t.date() for t in f5m_index])

    daily_cp = np.full(n, np.nan)
    daily_rsi = np.full(n, np.nan)

    # Build date -> prev daily index mapping
    unique_dates = sorted(set(f5m_dates))
    date_to_prev_idx = {}
    for d in unique_dates:
        best = -1
        for k in range(len(d_dates)):
            if d_dates[k] < d:
                best = k
        date_to_prev_idx[d] = best

    for i in range(n):
        k = date_to_prev_idx.get(f5m_dates[i], -1)
        if k >= 0:
            daily_cp[i] = d_cp[k]
            daily_rsi[i] = d_rsi[k]

    # Higher TFs: forward-fill onto 5m index
    ctx = {
        'daily_cp': daily_cp,
        'daily_rsi': daily_rsi,
    }

    for tf_name, cols in [('1h', ['chan_pos', 'rsi', 'bvc']),
                           ('4h', ['chan_pos']),
                           ('15m', ['chan_pos', 'rsi']),
                           ('30m', ['chan_pos'])]:
        tf_feat = features[tf_name]
        tf_idx = tf_feat.index.values
        col_vals = {c: tf_feat[c].values for c in cols}
        arrays = {f'{tf_name}_{c}': np.full(n, np.nan) for c in cols}

        j = 0
        for i in range(n):
            t = f5m_index[i]
            while j < len(tf_idx) - 1 and tf_idx[j + 1] <= t:
                j += 1
            if j < len(tf_idx) and tf_idx[j] <= t:
                for c in cols:
                    arrays[f'{tf_name}_{c}'][i] = col_vals[c][j]
        j = 0  # reset for next TF
        ctx.update(arrays)

    print(f"  Context computed in {time.time()-t0:.1f}s")
    return ctx

# ================================================================
# Signal Functions
# ================================================================
def signal_cross_tf_div(i, f5m_cp, f5m_rsi, f5m_turn, ctx, params):
    """Cross-TF divergence: higher TFs bullish, 5min lagging."""
    cp5 = f5m_cp[i]
    if np.isnan(cp5): return None

    d_cp = ctx['daily_cp'][i]
    h_cp = ctx['1h_chan_pos'][i]
    h4_cp = ctx['4h_chan_pos'][i]

    if np.isnan(d_cp) or np.isnan(h_cp) or np.isnan(h4_cp):
        return None

    higher_avg = d_cp * 0.35 + h4_cp * 0.35 + h_cp * 0.30
    div = higher_avg - cp5

    div_thresh = params.get('div_thresh', 0.40)
    if div < div_thresh: return None
    if cp5 > params.get('f5_thresh', 0.25): return None
    if not f5m_turn[i]: return None

    conf = 0.55 + 0.25 * min(div, 0.7) + 0.10 * (1.0 - cp5)
    return min(conf, 0.95)


def signal_mtf_bounce(i, f5m_cp, f5m_rsi, f5m_turn, ctx, params):
    """Multi-TF channel bounce: multiple TFs near bottom."""
    cp5 = f5m_cp[i]
    rsi5 = f5m_rsi[i]
    if np.isnan(cp5) or np.isnan(rsi5): return None

    d_cp = ctx['daily_cp'][i]
    h_cp = ctx['1h_chan_pos'][i]
    h4_cp = ctx['4h_chan_pos'][i]
    cp15 = ctx['15m_chan_pos'][i]

    if np.isnan(d_cp) or np.isnan(h_cp): return None

    tfs_nb = 0
    if not np.isnan(d_cp) and d_cp < params.get('d_thresh', 0.35): tfs_nb += 1
    if not np.isnan(h4_cp) and h4_cp < 0.30: tfs_nb += 1
    if not np.isnan(h_cp) and h_cp < 0.30: tfs_nb += 1
    if not np.isnan(cp15) and cp15 < 0.25: tfs_nb += 1

    if tfs_nb < params.get('min_tfs', 3): return None
    if cp5 > params.get('f5_thresh', 0.15): return None
    if rsi5 > params.get('rsi_thresh', 30): return None
    if not f5m_turn[i]: return None

    conf = 0.55 + 0.10 * tfs_nb + 0.10 * (1.0 - cp5)
    return min(conf, 0.95)


def signal_entropy_bounce(i, f5m_cp, f5m_rsi, f5m_turn, f5m_ent, ctx, params):
    """Low-entropy regime bounce."""
    cp5 = f5m_cp[i]
    rsi5 = f5m_rsi[i]
    if np.isnan(cp5) or np.isnan(rsi5): return None
    if f5m_ent is None or np.isnan(f5m_ent[i]): return None

    ent = f5m_ent[i]
    if ent > params.get('ent_thresh', 2.0): return None

    h_cp = ctx['1h_chan_pos'][i]
    if np.isnan(h_cp): return None

    if cp5 > 0.15: return None
    if h_cp > 0.35: return None
    if rsi5 > 30: return None
    if not f5m_turn[i]: return None

    ent_boost = max(0, (2.0 - ent) / 2.0) * 0.10
    conf = 0.55 + 0.15 * (1.0 - cp5) + ent_boost
    return min(conf, 0.95)

# ================================================================
# V3 Trade Simulation - Deferred Trail + ATR Stops
# ================================================================
def simulate_v3(f5m, features, ctx, signal_mode, params):
    """V3 simulator with deferred trailing stop and ATR-based exits.

    signal_mode: 'div', 'mtf', 'ent', 'combined'
    params dict keys:
        stop_atr_mult: stop = ATR * mult (default 2.0)
        tp_atr_mult: TP = ATR * mult (default 4.0)
        trail_atr_mult: trail = ATR * mult (default 1.5)
        min_profit_pct: trail activates only after this % profit (default 0.003)
        min_hold_bars: minimum bars before trail/TP can trigger (default 1)
        max_hold_bars: max bars in trade (default 78 = 1 day)
        force_eod: force close at EOD (default True)
        div_thresh, f5_thresh, d_thresh, min_tfs, rsi_thresh, ent_thresh
    """
    # Pre-extract arrays
    o_arr = f5m['open'].values
    h_arr = f5m['high'].values
    l_arr = f5m['low'].values
    c_arr = f5m['close'].values
    atr_arr = f5m['atr'].values
    cp_arr = f5m['chan_pos'].values
    rsi_arr = f5m['rsi'].values
    turn_arr = f5m.get('mom_turn')
    if turn_arr is not None:
        turn_arr = turn_arr.values
    else:
        turn_arr = np.zeros(len(f5m), dtype=bool)
    ent_arr = f5m.get('entropy')
    if ent_arr is not None:
        ent_arr = ent_arr.values
    bvc_arr = f5m['bvc'].values

    times = f5m.index
    tod = np.array([t.time() for t in times])
    n = len(f5m)

    # Params
    stop_mult = params.get('stop_atr_mult', 2.0)
    tp_mult = params.get('tp_atr_mult', 4.0)
    trail_mult = params.get('trail_atr_mult', 1.5)
    min_profit = params.get('min_profit_pct', 0.003)
    min_hold = params.get('min_hold_bars', 1)
    max_hold = params.get('max_hold_bars', 78)
    force_eod = params.get('force_eod', True)
    cooldown = params.get('cooldown_bars', 6)
    max_tpd = params.get('max_trades_per_day', 1)

    trades = []
    in_trade = False
    entry_price = 0.0
    entry_time = None
    confidence = 0.0
    stop_price = 0.0
    tp_price = 0.0
    best_price = 0.0
    hold_bars = 0
    cooldown_rem = 0
    trades_today = 0
    current_date = None
    pending = None

    t_entry_start = dt.time(9, 40)
    t_entry_end = dt.time(15, 25)
    t_force = dt.time(15, 50)

    for i in range(n):
        bar_time = times[i]
        bar_date = bar_time.date()
        bar_tod = tod[i]
        o, h, l, c = o_arr[i], h_arr[i], l_arr[i], c_arr[i]
        atr = atr_arr[i]

        if bar_date != current_date:
            current_date = bar_date
            trades_today = 0

        # Execute pending entry
        if pending is not None and not in_trade:
            p_conf, p_atr = pending
            pending = None
            if bar_tod >= dt.time(9, 35) and bar_tod <= dt.time(15, 30):
                entry_price = o * (1 + SLIPPAGE_PCT)
                entry_time = bar_time
                confidence = p_conf
                in_trade = True
                hold_bars = 0
                trades_today += 1
                # ATR-based stops
                stop_dist = p_atr * stop_mult
                tp_dist = p_atr * tp_mult
                stop_price = entry_price - stop_dist
                tp_price = entry_price + tp_dist
                best_price = entry_price

        # Manage position
        if in_trade:
            hold_bars += 1
            exit_price = None
            exit_reason = None

            # Hard stop always active
            if l <= stop_price:
                exit_price = max(stop_price, l)
                exit_reason = 'stop'
            else:
                best_price = max(best_price, h)

                # TP check (after min_hold)
                if hold_bars >= min_hold and h >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'

                # Deferred trailing stop (only after min_profit)
                elif hold_bars >= min_hold:
                    profit_pct = (best_price - entry_price) / entry_price
                    if profit_pct >= min_profit:
                        # Trail based on ATR at entry
                        trail_dist = atr * trail_mult
                        trail_stop = best_price - trail_dist
                        if trail_stop > stop_price:
                            stop_price = trail_stop
                        if l <= stop_price:
                            exit_price = max(stop_price, l)
                            exit_reason = 'trail'

                # Timeout
                if exit_price is None and hold_bars >= max_hold:
                    exit_price = c
                    exit_reason = 'timeout'

                # EOD exit
                if exit_price is None and force_eod and bar_tod >= t_force:
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
                    exit_reason=exit_reason, signal_name=signal_mode,
                ))
                in_trade = False
                cooldown_rem = cooldown

        # New signals
        if cooldown_rem > 0:
            cooldown_rem -= 1
            continue
        if in_trade or trades_today >= max_tpd:
            continue
        if bar_tod < t_entry_start or bar_tod > t_entry_end:
            continue
        if np.isnan(atr):
            continue

        # Generate signals based on mode
        conf = None
        if signal_mode == 'div':
            conf = signal_cross_tf_div(i, cp_arr, rsi_arr, turn_arr, ctx, params)
        elif signal_mode == 'mtf':
            conf = signal_mtf_bounce(i, cp_arr, rsi_arr, turn_arr, ctx, params)
        elif signal_mode == 'ent':
            conf = signal_entropy_bounce(i, cp_arr, rsi_arr, turn_arr, ent_arr, ctx, params)
        elif signal_mode == 'combined':
            # Require at least 2 signals
            sigs = []
            c1 = signal_cross_tf_div(i, cp_arr, rsi_arr, turn_arr, ctx, params)
            if c1 is not None: sigs.append(c1)
            c2 = signal_mtf_bounce(i, cp_arr, rsi_arr, turn_arr, ctx, params)
            if c2 is not None: sigs.append(c2)
            c3 = signal_entropy_bounce(i, cp_arr, rsi_arr, turn_arr, ent_arr, ctx, params)
            if c3 is not None: sigs.append(c3)
            min_agree = params.get('min_agree', 2)
            if len(sigs) >= min_agree:
                conf = np.mean(sigs) + 0.05 * (len(sigs) - min_agree)
                conf = min(conf, 0.95)

        if conf is not None:
            pending = (conf, atr)

    return trades

# ================================================================
# Results Display
# ================================================================
def show(trades, label=""):
    if not trades:
        print(f"  {label}: 0 trades")
        return
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades)
    bw = max(t.pnl for t in trades)
    wr = w / n * 100
    ah = np.mean([t.hold_bars for t in trades])
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    r_str = ' '.join(f"{k}={v}" for k, v in sorted(reasons.items()))
    flag = " ***" if wr >= 90 else (" **" if wr >= 80 else "")
    print(f"  {label}: {n}t {w}W/{n-w}L ({wr:.1f}%) ${pnl:+,.0f} "
          f"BW=${bw:+,.0f} BL=${bl:+,.0f} h={ah:.1f}b [{r_str}]{flag}")

def full_validate(trades, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    train = [t for t in trades if t.entry_time <= TRAIN_END]
    test = [t for t in trades if TRAIN_END < t.entry_time <= TEST_END]
    oos = [t for t in trades if t.entry_time > TEST_END]
    show(trades, "FULL")
    show(train, "TRAIN")
    show(test, "TEST")
    show(oos, "2026")
    years = sorted(set(t.entry_time.year for t in trades))
    cum = 0
    for yr in years:
        yt = [t for t in trades if t.entry_time.year == yr]
        if yt:
            w = sum(1 for t in yt if t.pnl > 0)
            p = sum(t.pnl for t in yt)
            bl = min(t.pnl for t in yt)
            cum += p
            f = " ***" if w/len(yt)*100 >= 90 else ""
            print(f"    {yr}: {len(yt):3d}t {w/len(yt)*100:5.1f}% ${p:+8,.0f} "
                  f"BL=${bl:+,.0f} cum=${cum:+,.0f}{f}")

# ================================================================
# Main Experiments
# ================================================================
def main():
    print("=" * 70)
    print("INTRADAY V3: DEFERRED TRAIL + ATR STOPS + COMBINED SIGNALS")
    print("=" * 70)

    df1m = load_1min()
    features, tfs = build_features(df1m)
    f5m = features['5m']
    print("Pre-computing context...")
    ctx = precompute_context(features, f5m.index)

    base_p = {
        'stop_atr_mult': 2.0,
        'tp_atr_mult': 4.0,
        'trail_atr_mult': 1.5,
        'min_profit_pct': 0.003,
        'min_hold_bars': 1,
        'max_hold_bars': 78,
        'force_eod': True,
        'cooldown_bars': 6,
        'div_thresh': 0.40,
        'f5_thresh': 0.25,
        'd_thresh': 0.35,
        'min_tfs': 3,
        'rsi_thresh': 30,
        'ent_thresh': 2.0,
        'min_agree': 2,
    }

    # ── Exp 1: Individual signals with default ATR params ──
    print("\n" + "=" * 70)
    print("EXP 1: Individual signals (ATR-based stops)")
    print("=" * 70)

    for mode in ['div', 'mtf', 'ent', 'combined']:
        trades = simulate_v3(f5m, features, ctx, mode, base_p)
        if trades:
            full_validate(trades, f"Signal: {mode}")

    # ── Exp 2: ATR multiplier sweep ──
    print("\n" + "=" * 70)
    print("EXP 2: ATR multiplier sweep (stop x TP)")
    print("=" * 70)

    for mode in ['div', 'mtf', 'combined']:
        print(f"\n--- {mode} ---")
        for sm in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for tm in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
                if tm <= sm: continue
                p = dict(base_p, stop_atr_mult=sm, tp_atr_mult=tm)
                trades = simulate_v3(f5m, features, ctx, mode, p)
                if trades and len(trades) >= 5:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    ah = np.mean([t.hold_bars for t in trades])
                    flag = " ***" if wr >= 90 else (" **" if wr >= 80 else "")
                    print(f"  s={sm:.1f} t={tm:.1f}: {n:4d}t {wr:5.1f}% "
                          f"${pnl:+10,.0f} BL=${bl:+,.0f} h={ah:.1f}b{flag}")

    # ── Exp 3: Min profit for trail activation sweep ──
    print("\n" + "=" * 70)
    print("EXP 3: Min profit for trail activation")
    print("=" * 70)

    for mode in ['div', 'mtf']:
        print(f"\n--- {mode} ---")
        for mp in [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.015]:
            for tm in [1.0, 1.5, 2.0, 3.0]:
                p = dict(base_p, min_profit_pct=mp, trail_atr_mult=tm)
                trades = simulate_v3(f5m, features, ctx, mode, p)
                if trades and len(trades) >= 5:
                    n = len(trades)
                    w = sum(1 for t in trades if t.pnl > 0)
                    wr = w / n * 100
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    ah = np.mean([t.hold_bars for t in trades])
                    flag = " ***" if wr >= 90 else (" **" if wr >= 80 else "")
                    print(f"  mp={mp:.3f} trail={tm:.1f}: {n:4d}t {wr:5.1f}% "
                          f"${pnl:+10,.0f} BL=${bl:+,.0f} h={ah:.1f}b{flag}")

    # ── Exp 4: Min hold bars sweep ──
    print("\n" + "=" * 70)
    print("EXP 4: Minimum hold bars before exit")
    print("=" * 70)

    for mode in ['div', 'mtf']:
        print(f"\n--- {mode} ---")
        for mh in [1, 2, 3, 6, 12, 24]:
            p = dict(base_p, min_hold_bars=mh)
            trades = simulate_v3(f5m, features, ctx, mode, p)
            if trades and len(trades) >= 3:
                n = len(trades)
                w = sum(1 for t in trades if t.pnl > 0)
                wr = w / n * 100
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                ah = np.mean([t.hold_bars for t in trades])
                flag = " ***" if wr >= 90 else (" **" if wr >= 80 else "")
                print(f"  min_hold={mh:2d}: {n:4d}t {wr:5.1f}% "
                      f"${pnl:+10,.0f} BL=${bl:+,.0f} h={ah:.1f}b{flag}")

    # ── Exp 5: Multi-day hold ──
    print("\n" + "=" * 70)
    print("EXP 5: Multi-day hold (disable EOD exit)")
    print("=" * 70)

    for mode in ['div', 'mtf', 'combined']:
        print(f"\n--- {mode} ---")
        for mh_bars in [78, 156, 390, 780]:
            for sm in [1.5, 2.0, 3.0]:
                for tm in [4.0, 6.0, 8.0]:
                    p = dict(base_p, max_hold_bars=mh_bars, force_eod=False,
                             stop_atr_mult=sm, tp_atr_mult=tm)
                    trades = simulate_v3(f5m, features, ctx, mode, p)
                    if trades and len(trades) >= 3:
                        n = len(trades)
                        w = sum(1 for t in trades if t.pnl > 0)
                        wr = w / n * 100
                        pnl = sum(t.pnl for t in trades)
                        bl = min(t.pnl for t in trades)
                        ah = np.mean([t.hold_bars for t in trades])
                        flag = " ***" if wr >= 90 else (" **" if wr >= 80 else "")
                        if wr >= 60 or pnl > 20000:
                            print(f"  maxH={mh_bars:3d} s={sm:.1f} t={tm:.1f}: "
                                  f"{n:4d}t {wr:5.1f}% ${pnl:+10,.0f} "
                                  f"BL=${bl:+,.0f} h={ah:.0f}b{flag}")

    # ── Exp 6: Time-of-day analysis ──
    print("\n" + "=" * 70)
    print("EXP 6: Time-of-day entry analysis (CrossTF_Div)")
    print("=" * 70)

    all_trades = simulate_v3(f5m, features, ctx, 'div', base_p)
    if all_trades:
        # Group by entry hour
        from collections import defaultdict
        by_hour = defaultdict(list)
        for t in all_trades:
            by_hour[t.entry_time.hour].append(t)
        for hr in sorted(by_hour):
            ht = by_hour[hr]
            n = len(ht)
            w = sum(1 for t in ht if t.pnl > 0)
            pnl = sum(t.pnl for t in ht)
            wr = w / n * 100
            flag = " ***" if wr >= 90 else ""
            print(f"  Hour {hr:2d}: {n:4d}t {wr:5.1f}% ${pnl:+8,.0f}{flag}")

    # ── Exp 7: Selectivity sweep for combined signal ──
    print("\n" + "=" * 70)
    print("EXP 7: Combined signal selectivity")
    print("=" * 70)

    for min_agree in [2, 3]:
        for div_t in [0.30, 0.40, 0.50]:
            for f5_t in [0.10, 0.15, 0.20, 0.25]:
                for sm in [1.5, 2.0, 3.0]:
                    p = dict(base_p, min_agree=min_agree, div_thresh=div_t,
                             f5_thresh=f5_t, stop_atr_mult=sm)
                    trades = simulate_v3(f5m, features, ctx, 'combined', p)
                    if trades and len(trades) >= 3:
                        n = len(trades)
                        w = sum(1 for t in trades if t.pnl > 0)
                        wr = w / n * 100
                        pnl = sum(t.pnl for t in trades)
                        bl = min(t.pnl for t in trades)
                        flag = " ***" if wr >= 90 else (" **" if wr >= 80 else "")
                        if wr >= 70:
                            print(f"  agree={min_agree} div>{div_t:.2f} 5m<{f5_t:.2f} "
                                  f"s={sm:.1f}: {n:4d}t {wr:5.1f}% "
                                  f"${pnl:+8,.0f} BL=${bl:+,.0f}{flag}")

    # ── Exp 8: Extreme selectivity for high WR ──
    print("\n" + "=" * 70)
    print("EXP 8: Extreme selectivity (push for >90% WR)")
    print("=" * 70)

    for mode in ['div', 'mtf', 'combined']:
        print(f"\n--- {mode} ---")
        for f5_t in [0.05, 0.08, 0.10, 0.12]:
            for rsi_t in [15, 20, 25]:
                for div_t in [0.45, 0.50, 0.55, 0.60]:
                    for sm in [1.5, 2.0, 3.0]:
                        p = dict(base_p, f5_thresh=f5_t, rsi_thresh=rsi_t,
                                 div_thresh=div_t, stop_atr_mult=sm,
                                 min_tfs=3)
                        trades = simulate_v3(f5m, features, ctx, mode, p)
                        if trades and len(trades) >= 3:
                            n = len(trades)
                            w = sum(1 for t in trades if t.pnl > 0)
                            wr = w / n * 100
                            pnl = sum(t.pnl for t in trades)
                            bl = min(t.pnl for t in trades)
                            flag = " ***" if wr >= 90 else ""
                            if wr >= 80:
                                print(f"  5m<{f5_t:.2f} rsi<{rsi_t} div>{div_t:.2f} "
                                      f"s={sm:.1f}: {n:4d}t {wr:5.1f}% "
                                      f"${pnl:+8,.0f} BL=${bl:+,.0f}{flag}")

    print("\nDone.")

if __name__ == '__main__':
    main()
