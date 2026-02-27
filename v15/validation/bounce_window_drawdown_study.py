#!/usr/bin/env python3
"""
Bounce study v3: Granular channel window sweep + max drawdown proximity analysis.

Two questions:
1. What daily channel window (20-100 bars) gives the cleanest bounce signal?
2. Does proximity to max historical single-day drawdown predict bounces?
   (i.e., if today's drop is in the top 5% of all daily drops AND signals fire,
    is that a strong buy signal?)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.tf_state_backtest import (
    load_all_tfs, compute_daily_states, _norm_cols, _lin_reg_channel,
    _momentum_is_turning_up, TF_WINDOWS,
)
from v15.validation.bounce_timing import _compute_rsi
from v15.data.native_tf import fetch_native_tf


def main():
    start, end = '2016-01-01', '2026-12-31'

    print("Loading data...")
    tf_data = load_all_tfs('data/TSLAMin.txt', start, end)
    daily_df = tf_data['daily']
    weekly_df = tf_data['weekly']

    # SPY RSI
    spy = _norm_cols(fetch_native_tf('SPY', 'daily', start, end))
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_rsi = _compute_rsi(spy['close'], 14)

    # TSLA RSI
    tsla_rsi_d = _compute_rsi(daily_df['close'], 14)
    tsla_rsi_w = _compute_rsi(weekly_df['close'], 14)
    tsla_rsi_w_sma = tsla_rsi_w.rolling(14, min_periods=1).mean()

    # TF states (for w_turning, etc.)
    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates, warmup_bars=260)
    state_by_date = {r['date']: r for r in state_rows}

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: Single-day drawdown analysis
    # ══════════════════════════════════════════════════════════════════════════
    print("\nComputing single-day drawdowns...")

    # Daily return and intraday drawdown metrics
    daily_df = daily_df.copy()
    daily_df['daily_return'] = daily_df['close'].pct_change() * 100  # close-to-close %
    daily_df['intraday_range'] = (daily_df['high'] - daily_df['low']) / daily_df['open'] * 100
    daily_df['open_to_low'] = (daily_df['low'] - daily_df['open']) / daily_df['open'] * 100  # worst intraday drop
    daily_df['open_to_close'] = (daily_df['close'] - daily_df['open']) / daily_df['open'] * 100
    daily_df['prev_close_to_low'] = (daily_df['low'] / daily_df['close'].shift(1) - 1) * 100  # gap + intraday

    # Rolling percentile of today's drop vs trailing history
    LOOKBACK_DAYS = 504  # ~2 years of trading days
    daily_df['drop_pctile'] = np.nan  # percentile of how bad today's drop is (0=worst, 100=best)
    daily_df['drop_rank_2y'] = np.nan  # rank among last 2yr of daily drops

    closes = daily_df['close'].values
    lows = daily_df['low'].values
    opens = daily_df['open'].values
    prev_close_to_low = daily_df['prev_close_to_low'].values

    for i in range(LOOKBACK_DAYS, len(daily_df)):
        # How bad is today's prev_close_to_low vs last 2 years?
        window = prev_close_to_low[i - LOOKBACK_DAYS:i]
        today_val = prev_close_to_low[i]
        if np.isnan(today_val):
            continue
        # Percentile: what % of days had a WORSE (more negative) drop
        pctile = (window < today_val).sum() / len(window) * 100
        daily_df.iloc[i, daily_df.columns.get_loc('drop_pctile')] = pctile
        # Rank: 1 = worst day in 2yr, higher = less extreme
        rank = (window <= today_val).sum()
        daily_df.iloc[i, daily_df.columns.get_loc('drop_rank_2y')] = rank

    # 2-day and 3-day cumulative drop
    daily_df['drop_2d'] = daily_df['daily_return'].rolling(2).sum()
    daily_df['drop_3d'] = daily_df['daily_return'].rolling(3).sum()

    # Max drawdown from recent peak (20-day trailing)
    daily_df['rolling_max_20'] = daily_df['high'].rolling(20, min_periods=1).max()
    daily_df['drawdown_from_peak'] = (daily_df['close'] / daily_df['rolling_max_20'] - 1) * 100

    print(f"  Computed drawdown metrics for {len(daily_df)} days")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: Granular channel window sweep
    # ══════════════════════════════════════════════════════════════════════════
    print("\nRunning granular channel window sweep (20 to 100, step 5)...")

    WINDOWS = list(range(20, 105, 5))  # 20, 25, 30, ..., 100
    THRESHOLD = 0.35
    COOLDOWN = 10

    def measure_forward(date, horizons=[5, 10]):
        if date not in daily_df.index:
            return None
        idx = daily_df.index.get_loc(date)
        if idx + 1 >= len(daily_df):
            return None
        entry_price = daily_df.iloc[idx + 1]['open']
        result = {'entry': entry_price}
        for h in horizons:
            end_idx = idx + 1 + h
            if end_idx >= len(daily_df):
                result[f'fwd_{h}d'] = np.nan
                result[f'dd_{h}d'] = np.nan
                result[f'peak_{h}d'] = np.nan
                continue
            fwd_slice = daily_df.iloc[idx+1:end_idx+1]
            exit_close = fwd_slice.iloc[-1]['close']
            result[f'fwd_{h}d'] = (exit_close / entry_price - 1) * 100
            result[f'dd_{h}d'] = (fwd_slice['low'].min() / entry_price - 1) * 100
            result[f'peak_{h}d'] = (fwd_slice['high'].max() / entry_price - 1) * 100
        return result

    # For each window, compute pos_pct for each date, then find oversold events
    window_results = {}

    for w in WINDOWS:
        events = []
        last_fire = None

        for date in trading_dates:
            if date not in state_by_date:
                continue
            row = state_by_date[date]
            weekly_state = row.get('weekly')
            if not weekly_state:
                continue

            # Compute daily channel with this window
            di = daily_df.index.get_loc(date)
            if di < w + 10:
                continue
            prices = daily_df['close'].iloc[di - w - 10:di + 1].values.astype(float)
            if len(prices) < w:
                continue

            d_pos, r2, width = _lin_reg_channel(prices[-w:])
            w_pos = weekly_state['pos_pct']
            w_turning = weekly_state['is_turning']

            # Also compute daily momentum turning
            d_turning = _momentum_is_turning_up(prices, lookback=min(10, len(prices) - 2))

            if d_pos >= THRESHOLD or w_pos >= THRESHOLD:
                continue

            # Cooldown
            if last_fire and (date - last_fire).days < COOLDOWN:
                continue

            fwd = measure_forward(date)
            if fwd is None:
                continue

            # Get drawdown metrics
            drop_pctile = daily_df.loc[date, 'drop_pctile'] if date in daily_df.index else np.nan
            dd_from_peak = daily_df.loc[date, 'drawdown_from_peak'] if date in daily_df.index else np.nan
            daily_ret = daily_df.loc[date, 'daily_return'] if date in daily_df.index else np.nan
            otl = daily_df.loc[date, 'open_to_low'] if date in daily_df.index else np.nan
            drop_2d = daily_df.loc[date, 'drop_2d'] if date in daily_df.index else np.nan

            # RSI
            td_rsi = float(tsla_rsi_d.loc[date]) if date in tsla_rsi_d.index else np.nan
            tw_rsi_idx = tsla_rsi_w.index.searchsorted(date, side='right') - 1
            tw_rsi = float(tsla_rsi_w.iloc[tw_rsi_idx]) if 0 <= tw_rsi_idx < len(tsla_rsi_w) else np.nan

            events.append({
                'date': date,
                'close': daily_df.loc[date, 'close'],
                'd_pos': d_pos,
                'w_pos': w_pos,
                'r2': r2,
                'd_turning': d_turning,
                'w_turning': w_turning,
                'tsla_rsi_d': td_rsi,
                'tsla_rsi_w': tw_rsi,
                'daily_return': daily_ret,
                'open_to_low': otl,
                'drop_pctile': drop_pctile,
                'dd_from_peak': dd_from_peak,
                'drop_2d': drop_2d,
                **{k: v for k, v in fwd.items() if k != 'entry'},
            })
            last_fire = date

        edf = pd.DataFrame(events) if events else pd.DataFrame()
        n = len(edf)
        if n == 0:
            window_results[w] = {'n': 0}
            continue

        fwd5 = edf['fwd_5d'].dropna()
        fwd10 = edf['fwd_10d'].dropna()
        wr5 = (fwd5 > 0).mean() * 100 if len(fwd5) > 0 else 0
        wr10 = (fwd10 > 0).mean() * 100 if len(fwd10) > 0 else 0

        # Split: at_bottom (<0.15) vs recovering (0.15-0.35)
        bottom = edf[edf['d_pos'] < 0.15]
        recovering = edf[(edf['d_pos'] >= 0.15) & (edf['d_pos'] < 0.35)]

        # With w_turning filter
        wt = edf[edf['w_turning'] == True]

        window_results[w] = {
            'n': n,
            'wr5': wr5,
            'avg5': fwd5.mean(),
            'med5': fwd5.median(),
            'wr10': wr10,
            'avg10': fwd10.mean(),
            'avg_r2': edf['r2'].mean(),
            'n_bottom': len(bottom),
            'wr5_bottom': (bottom['fwd_5d'].dropna() > 0).mean() * 100 if len(bottom) > 0 else 0,
            'avg5_bottom': bottom['fwd_5d'].mean() if len(bottom) > 0 else np.nan,
            'n_recovering': len(recovering),
            'wr5_recovering': (recovering['fwd_5d'].dropna() > 0).mean() * 100 if len(recovering) > 0 else 0,
            'avg5_recovering': recovering['fwd_5d'].mean() if len(recovering) > 0 else np.nan,
            'n_wt': len(wt),
            'wr5_wt': (wt['fwd_5d'].dropna() > 0).mean() * 100 if len(wt) > 0 else 0,
            'avg5_wt': wt['fwd_5d'].mean() if len(wt) > 0 else np.nan,
            'df': edf,
        }

    # ── Print window sweep results ────────────────────────────────────────────
    print(f"\n{'='*130}")
    print("PART 1: CHANNEL WINDOW SWEEP (daily TF, windows 20-100)")
    print(f"Signal: d_pos<0.35 & w_pos<0.35, 10d cooldown, entry at next-day open")
    print(f"{'='*130}")

    print(f"\n  {'Win':>4} {'R²':>5} {'n':>4} {'WR5d':>5} {'avg5d':>7} {'med5d':>7} "
          f"{'n_bot':>5} {'WR_bot':>6} {'avg_bot':>7} "
          f"{'n_rec':>5} {'WR_rec':>6} {'avg_rec':>7} "
          f"{'n_wt':>4} {'WR_wt':>5} {'avg_wt':>7}")
    print(f"  {'─'*120}")

    for w in WINDOWS:
        r = window_results[w]
        if r['n'] == 0:
            print(f"  {w:>4}  {'(no events)':>20}")
            continue
        print(f"  {w:>4} {r['avg_r2']:>5.2f} {r['n']:>4} {r['wr5']:>4.0f}% {r['avg5']:>+6.1f}% {r['med5']:>+6.1f}% "
              f"{r['n_bottom']:>5} {r['wr5_bottom']:>5.0f}% {r['avg5_bottom']:>+6.1f}% "
              f"{r['n_recovering']:>5} {r['wr5_recovering']:>5.0f}% {r['avg5_recovering']:>+6.1f}% "
              f"{r['n_wt']:>4} {r['wr5_wt']:>4.0f}% {r['avg5_wt']:>+6.1f}%")

    # Find optimal windows
    best_overall = max([w for w in WINDOWS if window_results[w]['n'] >= 5],
                       key=lambda w: window_results[w]['avg5'], default=60)
    best_wr = max([w for w in WINDOWS if window_results[w]['n'] >= 5],
                  key=lambda w: window_results[w]['wr5'], default=60)
    best_wt = max([w for w in WINDOWS if window_results[w].get('n_wt', 0) >= 3],
                  key=lambda w: window_results[w]['avg5_wt'], default=60)
    best_recovering = max([w for w in WINDOWS if window_results[w].get('n_recovering', 0) >= 3],
                         key=lambda w: window_results[w]['avg5_recovering'], default=60)

    print(f"\n  OPTIMAL WINDOWS:")
    r = window_results[best_overall]
    print(f"    Best avg 5d return:    W={best_overall}  (n={r['n']}, avg={r['avg5']:+.1f}%, WR={r['wr5']:.0f}%)")
    r = window_results[best_wr]
    print(f"    Best win rate:         W={best_wr}  (n={r['n']}, avg={r['avg5']:+.1f}%, WR={r['wr5']:.0f}%)")
    r = window_results[best_wt]
    print(f"    Best w/ w_turning:     W={best_wt}  (n={r['n_wt']}, avg={r['avg5_wt']:+.1f}%, WR={r['wr5_wt']:.0f}%)")
    r = window_results[best_recovering]
    print(f"    Best recovering zone:  W={best_recovering}  (n={r['n_recovering']}, avg={r['avg5_recovering']:+.1f}%, WR={r['wr5_recovering']:.0f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3: Drawdown proximity analysis (using default W=60 events)
    # ══════════════════════════════════════════════════════════════════════════
    # Use the W=60 event set for drawdown analysis
    edf = window_results[60].get('df', pd.DataFrame())
    if len(edf) == 0:
        print("\nNo events for drawdown analysis.")
        return

    print(f"\n{'='*130}")
    print("PART 2: MAX DRAWDOWN PROXIMITY ANALYSIS")
    print(f"Does the extremity of the day's drop predict bounce quality?")
    print(f"{'='*130}")

    # Show distribution of daily returns on signal days
    print(f"\n  Daily returns on signal days (n={len(edf)}):")
    print(f"    avg={edf['daily_return'].mean():+.1f}%  med={edf['daily_return'].median():+.1f}%  "
          f"worst={edf['daily_return'].min():+.1f}%  best={edf['daily_return'].max():+.1f}%")
    print(f"    avg open-to-low={edf['open_to_low'].mean():+.1f}%  "
          f"worst={edf['open_to_low'].min():+.1f}%")

    # Bucket by drop percentile (how extreme is today's drop vs 2yr history)
    print(f"\n  Drop severity vs 2yr history (drop_pctile: 0=worst day, 100=mildest):")
    for lo, hi, label in [
        (0, 3, 'Top 3% worst days (extreme crash)'),
        (3, 7, 'Top 3-7% (severe drop)'),
        (7, 15, 'Top 7-15% (bad day)'),
        (15, 30, 'Top 15-30% (moderate drop)'),
        (30, 100, 'Mild (not extreme)'),
    ]:
        bucket = edf[(edf['drop_pctile'] >= lo) & (edf['drop_pctile'] < hi)]
        if len(bucket) == 0:
            continue
        fwd5 = bucket['fwd_5d'].dropna()
        wr = (fwd5 > 0).mean() * 100 if len(fwd5) > 0 else 0
        print(f"    {label:<40}:  n={len(bucket):>3}  "
              f"5d avg={fwd5.mean():+.1f}%  WR={wr:.0f}%  "
              f"peak5={bucket['peak_5d'].mean():+.1f}%  dd5={bucket['dd_5d'].mean():+.1f}%")

    # Bucket by daily return magnitude
    print(f"\n  Daily return magnitude on signal day:")
    for lo, hi, label in [
        (-999, -7, 'Crash (< -7%)'),
        (-7, -4, 'Big drop (-7% to -4%)'),
        (-4, -2, 'Moderate drop (-4% to -2%)'),
        (-2, 0, 'Small drop (-2% to 0%)'),
        (0, 999, 'Green day (>0%)'),
    ]:
        bucket = edf[(edf['daily_return'] >= lo) & (edf['daily_return'] < hi)]
        if len(bucket) == 0:
            continue
        fwd5 = bucket['fwd_5d'].dropna()
        wr = (fwd5 > 0).mean() * 100 if len(fwd5) > 0 else 0
        print(f"    {label:<40}:  n={len(bucket):>3}  "
              f"5d avg={fwd5.mean():+.1f}%  WR={wr:.0f}%  "
              f"peak5={bucket['peak_5d'].mean():+.1f}%  dd5={bucket['dd_5d'].mean():+.1f}%")

    # Bucket by open-to-low (intraday worst drop)
    print(f"\n  Intraday drop (open-to-low) on signal day:")
    for lo, hi, label in [
        (-999, -7, 'Intraday crash (> -7%)'),
        (-7, -4, 'Big intraday drop (-7% to -4%)'),
        (-4, -2, 'Moderate intraday (-4% to -2%)'),
        (-2, 0, 'Small intraday (-2% to 0%)'),
    ]:
        bucket = edf[(edf['open_to_low'] >= lo) & (edf['open_to_low'] < hi)]
        if len(bucket) == 0:
            continue
        fwd5 = bucket['fwd_5d'].dropna()
        wr = (fwd5 > 0).mean() * 100 if len(fwd5) > 0 else 0
        print(f"    {label:<40}:  n={len(bucket):>3}  "
              f"5d avg={fwd5.mean():+.1f}%  WR={wr:.0f}%  "
              f"peak5={bucket['peak_5d'].mean():+.1f}%  dd5={bucket['dd_5d'].mean():+.1f}%")

    # Drawdown from 20-day peak
    print(f"\n  Drawdown from 20-day high on signal day:")
    for lo, hi, label in [
        (-999, -20, 'Extreme (> -20%)'),
        (-20, -15, 'Deep (-20% to -15%)'),
        (-15, -10, 'Significant (-15% to -10%)'),
        (-10, -5, 'Moderate (-10% to -5%)'),
        (-5, 0, 'Near peak (-5% to 0%)'),
    ]:
        bucket = edf[(edf['dd_from_peak'] >= lo) & (edf['dd_from_peak'] < hi)]
        if len(bucket) == 0:
            continue
        fwd5 = bucket['fwd_5d'].dropna()
        wr = (fwd5 > 0).mean() * 100 if len(fwd5) > 0 else 0
        print(f"    {label:<40}:  n={len(bucket):>3}  "
              f"5d avg={fwd5.mean():+.1f}%  WR={wr:.0f}%  "
              f"peak5={bucket['peak_5d'].mean():+.1f}%  dd5={bucket['dd_5d'].mean():+.1f}%")

    # 2-day cumulative drop
    print(f"\n  2-day cumulative drop on signal day:")
    for lo, hi, label in [
        (-999, -10, 'Extreme 2d crash (< -10%)'),
        (-10, -5, 'Big 2d drop (-10% to -5%)'),
        (-5, -2, 'Moderate 2d drop (-5% to -2%)'),
        (-2, 999, 'Mild 2d (> -2%)'),
    ]:
        bucket = edf[(edf['drop_2d'] >= lo) & (edf['drop_2d'] < hi)]
        if len(bucket) == 0:
            continue
        fwd5 = bucket['fwd_5d'].dropna()
        wr = (fwd5 > 0).mean() * 100 if len(fwd5) > 0 else 0
        print(f"    {label:<40}:  n={len(bucket):>3}  "
              f"5d avg={fwd5.mean():+.1f}%  WR={wr:.0f}%  "
              f"peak5={bucket['peak_5d'].mean():+.1f}%  dd5={bucket['dd_5d'].mean():+.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 4: Drawdown + signal factor combos
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print("PART 3: DRAWDOWN + SIGNAL FACTOR COMBINATIONS")
    print(f"{'='*130}")

    combos = [
        # Drawdown extremity + momentum turning
        ('Extreme drop pctile<7 + w_turning',
         lambda r: r['drop_pctile'] < 7 and r['w_turning']),
        ('Extreme drop pctile<7 + d_turning',
         lambda r: r['drop_pctile'] < 7 and r['d_turning']),
        ('Extreme drop pctile<7 + any turning',
         lambda r: r['drop_pctile'] < 7 and (r['w_turning'] or r['d_turning'])),
        ('Extreme drop pctile<7 + neither turning',
         lambda r: r['drop_pctile'] < 7 and not r['w_turning'] and not r['d_turning']),

        # Big daily drop + turning
        ('Daily return < -4% + w_turning',
         lambda r: r['daily_return'] < -4 and r['w_turning']),
        ('Daily return < -4% + d_turning',
         lambda r: r['daily_return'] < -4 and r['d_turning']),
        ('Daily return < -4% (no turning req)',
         lambda r: r['daily_return'] < -4),
        ('Daily return < -7% (crash)',
         lambda r: r['daily_return'] < -7),

        # DD from peak + turning
        ('DD from peak > -15% + w_turning',
         lambda r: r['dd_from_peak'] < -15 and r['w_turning']),
        ('DD from peak > -15% + d_turning',
         lambda r: r['dd_from_peak'] < -15 and r['d_turning']),
        ('DD from peak > -10% + w_turning',
         lambda r: r['dd_from_peak'] < -10 and r['w_turning']),

        # 2-day crash
        ('2d drop < -5% + w_turning',
         lambda r: r['drop_2d'] < -5 and r['w_turning']),
        ('2d drop < -5% + rsi_d < 30',
         lambda r: r['drop_2d'] < -5 and r['tsla_rsi_d'] < 30),

        # Extreme drop + RSI sweet spot
        ('Extreme pctile<10 + rsi_w 40-50',
         lambda r: r['drop_pctile'] < 10 and 40 <= r['tsla_rsi_w'] < 50),
        ('Daily < -3% + rsi_w 40-50 + w_turning',
         lambda r: r['daily_return'] < -3 and 40 <= r['tsla_rsi_w'] < 50 and r['w_turning']),
        ('Daily < -3% + rsi_d < 30',
         lambda r: r['daily_return'] < -3 and r['tsla_rsi_d'] < 30),

        # Drawdown from peak + market regime
        ('DD peak > -10% + rsi_w 40-50',
         lambda r: r['dd_from_peak'] < -10 and 40 <= r['tsla_rsi_w'] < 50),
        ('DD peak > -15% + rsi_d < 30',
         lambda r: r['dd_from_peak'] < -15 and r['tsla_rsi_d'] < 30),

        # Kitchen sink: extreme + turning + RSI
        ('BEST: daily<-3% + w_turning + rsi_w 40-50',
         lambda r: r['daily_return'] < -3 and r['w_turning'] and 40 <= r['tsla_rsi_w'] < 50),
        ('BEST2: pctile<10 + w_turning',
         lambda r: r['drop_pctile'] < 10 and r['w_turning']),
        ('BEST3: dd_peak<-10% + w_turning + rsi_w<50',
         lambda r: r['dd_from_peak'] < -10 and r['w_turning'] and r['tsla_rsi_w'] < 50),
    ]

    print(f"\n  {'Combination':<50} {'n':>4} {'WR':>5} {'5d avg':>8} {'5d med':>8} {'peak5':>7} {'dd5':>7}")
    print(f"  {'─'*95}")

    for label, fn in combos:
        try:
            mask = edf.apply(fn, axis=1)
        except Exception:
            mask = pd.Series(False, index=edf.index)
        subset = edf[mask]
        n_s = len(subset)
        if n_s == 0:
            print(f"  {label:<50} {'0':>4}")
            continue
        fwd5 = subset['fwd_5d'].dropna()
        wr = (fwd5 > 0).mean() * 100 if len(fwd5) > 0 else 0
        avg = fwd5.mean()
        med = fwd5.median()
        peak = subset['peak_5d'].dropna().mean()
        dd = subset['dd_5d'].dropna().mean()
        print(f"  {label:<50} {n_s:>4} {wr:>4.0f}% {avg:>+7.1f}% {med:>+7.1f}% "
              f"{peak:>+6.1f}% {dd:>+6.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 5: Correlation of drawdown metrics with forward returns
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print("PART 4: DRAWDOWN FACTOR CORRELATIONS WITH FORWARD RETURNS")
    print(f"{'='*130}")

    dd_factors = ['daily_return', 'open_to_low', 'drop_pctile', 'dd_from_peak',
                  'drop_2d', 'tsla_rsi_d', 'tsla_rsi_w']

    print(f"\n  {'Factor':<20} {'Corr 5d':>10} {'Corr 10d':>10} {'Direction':>15}")
    print(f"  {'─'*60}")

    for f in dd_factors:
        if f not in edf.columns:
            continue
        valid = edf[[f, 'fwd_5d', 'fwd_10d']].dropna()
        if len(valid) < 10:
            continue
        c5 = valid[f].corr(valid['fwd_5d'])
        c10 = valid[f].corr(valid['fwd_10d'])
        marker = ' ***' if abs(c5) > 0.15 else (' **' if abs(c5) > 0.10 else '')
        print(f"  {f:<20} {c5:>+9.3f} {c10:>+9.3f}  {'LOWER=better' if c5 < 0 else 'HIGHER=better'}{marker}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 6: RSI vs Drawdown — are they measuring the same thing?
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print("PART 5: RSI vs DRAWDOWN — DO THEY OVERLAP?")
    print(f"{'='*130}")

    # Correlation between RSI and drawdown metrics
    overlap_factors = [
        ('tsla_rsi_d', 'daily_return'),
        ('tsla_rsi_d', 'dd_from_peak'),
        ('tsla_rsi_d', 'drop_pctile'),
        ('tsla_rsi_d', 'open_to_low'),
        ('tsla_rsi_d', 'drop_2d'),
    ]

    print(f"\n  {'Factor A':<15} {'Factor B':<20} {'Correlation':>12}")
    print(f"  {'─'*55}")

    for a, b in overlap_factors:
        if a not in edf.columns or b not in edf.columns:
            continue
        valid = edf[[a, b]].dropna()
        if len(valid) < 10:
            continue
        corr = valid[a].corr(valid[b])
        overlap = 'OVERLAPPING' if abs(corr) > 0.5 else ('related' if abs(corr) > 0.3 else 'independent')
        print(f"  {a:<15} {b:<20} {corr:>+11.3f}  ({overlap})")

    # Additive value test: RSI alone vs drawdown alone vs both
    print(f"\n  Additive value test (5d forward returns):")

    # RSI < 30 alone
    rsi_only = edf[edf['tsla_rsi_d'] < 30]
    dd_only = edf[edf['dd_from_peak'] < -15]
    both = edf[(edf['tsla_rsi_d'] < 30) & (edf['dd_from_peak'] < -15)]
    neither = edf[(edf['tsla_rsi_d'] >= 30) & (edf['dd_from_peak'] >= -15)]

    for label, subset in [('RSI_d < 30 only', rsi_only),
                          ('DD from peak < -15% only', dd_only),
                          ('Both RSI<30 + DD<-15%', both),
                          ('Neither', neither)]:
        if len(subset) == 0:
            print(f"    {label:<35}:  n=0")
            continue
        fwd5 = subset['fwd_5d'].dropna()
        wr = (fwd5 > 0).mean() * 100 if len(fwd5) > 0 else 0
        print(f"    {label:<35}:  n={len(subset):>3}  "
              f"5d avg={fwd5.mean():+.1f}%  WR={wr:.0f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 7: Individual events with drawdown data
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print("PART 6: INDIVIDUAL EVENTS WITH DRAWDOWN DATA (most recent 25)")
    print(f"{'='*130}")

    recent = edf.tail(25)
    print(f"\n  {'Date':<11} {'Close':>7} {'D_pos':>5} {'W_pos':>5} "
          f"{'DayRet':>7} {'O2L':>6} {'Pctile':>6} {'DD20pk':>7} {'2dDrop':>7} "
          f"{'D_RSI':>5} {'W_trn':>5} "
          f"{'5d%':>6} {'10d%':>7} {'Pk5':>6}")
    print('  ' + '─' * 115)

    for _, e in recent.iterrows():
        date_str = e['date'].strftime('%Y-%m-%d')
        pctile_str = f"{e['drop_pctile']:>5.1f}" if not np.isnan(e['drop_pctile']) else '  n/a'
        print(f"  {date_str:<11} {e['close']:>7.1f} "
              f"{e['d_pos']:>5.2f} {e['w_pos']:>5.2f} "
              f"{e['daily_return']:>+6.1f}% {e['open_to_low']:>+5.1f}% "
              f"{pctile_str}% {e['dd_from_peak']:>+6.1f}% {e['drop_2d']:>+6.1f}% "
              f"{e['tsla_rsi_d']:>5.1f} {'Y' if e['w_turning'] else 'N':>5} "
              f"{e['fwd_5d']:>+5.1f}% {e['fwd_10d']:>+6.1f}% {e['peak_5d']:>+5.1f}%")

    print(f"\n{'='*130}")
    print("STUDY COMPLETE")
    print(f"{'='*130}")


if __name__ == '__main__':
    main()
