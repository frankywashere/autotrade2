#!/usr/bin/env python3
"""
Bounce signal deep study: fire on every oversold condition, measure forward
returns, and identify which confluence factors separate winners from losers.

The goal: understand what makes a good bounce entry vs a bad one.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.tf_state_backtest import (
    load_all_tfs, compute_daily_states, _norm_cols,
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
    spy_rsi_series = _compute_rsi(spy['close'], 14)

    # TSLA daily + weekly RSI
    tsla_rsi_d = _compute_rsi(daily_df['close'], 14)
    tsla_rsi_w = _compute_rsi(weekly_df['close'], 14)
    tsla_rsi_w_sma = tsla_rsi_w.rolling(14, min_periods=1).mean()
    tsla_52w_sma = daily_df['close'].rolling(252, min_periods=60).mean()

    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates, warmup_bars=260)

    def _lookup_weekly(series, date):
        idx = series.index.searchsorted(date, side='right') - 1
        return float(series.iloc[idx]) if 0 <= idx < len(series) else np.nan

    def _lookup_daily(series, date):
        if date in series.index:
            return float(series.loc[date])
        return np.nan

    # ── Measure forward returns ──────────────────────────────────────────────
    def measure_forward(date, horizons=[1, 2, 3, 5, 10]):
        """Return dict of forward returns and max drawdowns for each horizon."""
        if date not in daily_df.index:
            return None
        idx = daily_df.index.get_loc(date)
        ref_close = daily_df.iloc[idx]['close']
        # Entry at next day's open
        if idx + 1 >= len(daily_df):
            return None
        entry_price = daily_df.iloc[idx + 1]['open']

        result = {'ref_close': ref_close, 'entry_open': entry_price}
        for h in horizons:
            end_idx = idx + 1 + h  # h days after entry
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

    # ── Scan for oversold conditions ─────────────────────────────────────────
    # Simple trigger: daily pos_pct < 0.35 AND weekly pos_pct < 0.35
    THRESHOLD = 0.35
    COOLDOWN = 10  # days between signals (hold period)

    events = []
    last_fire_date = None

    for row in state_rows:
        date = row['date']
        d = row.get('daily')
        w = row.get('weekly')
        m = row.get('monthly')
        if not d or not w:
            continue

        d_pos = d['pos_pct']
        w_pos = w['pos_pct']

        if d_pos >= THRESHOLD or w_pos >= THRESHOLD:
            continue

        # Cooldown
        if last_fire_date and (date - last_fire_date).days < COOLDOWN:
            continue

        # Get all factors
        spy_val = np.nan
        idx = spy_rsi_series.index.searchsorted(date)
        if 0 < idx <= len(spy_rsi_series):
            spy_val = float(spy_rsi_series.iloc[idx - 1])

        tw_rsi = _lookup_weekly(tsla_rsi_w, date)
        tw_sma = _lookup_weekly(tsla_rsi_w_sma, date)
        td_rsi = _lookup_daily(tsla_rsi_d, date)

        dist_52w = 0.0
        if date in tsla_52w_sma.index and date in daily_df.index:
            sma_val = tsla_52w_sma.loc[date]
            close_val = daily_df.loc[date, 'close']
            dist_52w = (close_val - sma_val) / sma_val if sma_val > 0 else 0.0

        fwd = measure_forward(date)
        if fwd is None:
            continue

        m_pos = m['pos_pct'] if m else np.nan

        events.append({
            'date': date,
            'close': fwd['ref_close'],
            'entry': fwd['entry_open'],
            'd_pos': d_pos,
            'w_pos': w_pos,
            'm_pos': m_pos,
            'd_turning': d['is_turning'],
            'w_turning': w['is_turning'],
            'd_at_bottom': d.get('at_bottom', False),
            'd_near_bottom': d.get('near_bottom', False),
            'energy_ratio': d.get('energy_ratio', 1.0),
            'spy_rsi': spy_val,
            'tsla_rsi_d': td_rsi,
            'tsla_rsi_w': tw_rsi,
            'tsla_rsi_w_sma': tw_sma,
            'rsi_w_above_sma': tw_rsi > tw_sma if not np.isnan(tw_rsi) and not np.isnan(tw_sma) else False,
            'dist_52w': dist_52w,
            **{k: v for k, v in fwd.items() if k != 'ref_close' and k != 'entry_open'},
        })
        last_fire_date = date

    df = pd.DataFrame(events)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    n = len(df)

    print(f"\n{'='*110}")
    print(f"BOUNCE SIGNAL STUDY: {n} oversold events (d_pos<{THRESHOLD} & w_pos<{THRESHOLD}, {COOLDOWN}d cooldown)")
    print(f"Period: {start} to {end}")
    print(f"Entry: next day's open after signal fires")
    print(f"{'='*110}")

    # ── Individual events ────────────────────────────────────────────────────
    print(f"\n{'Date':<11} {'Close':>7} {'Entry':>7} {'D_pos':>5} {'W_pos':>5} "
          f"{'D_RSI':>5} {'W_RSI':>5} {'W>SMA':>5} {'SPY':>4} {'52w%':>6} "
          f"{'D_trn':>5} {'W_trn':>5} "
          f"{'1d%':>6} {'2d%':>6} {'3d%':>6} {'5d%':>6} {'10d%':>6} {'Peak5':>6} {'DD5':>6}")
    print('─' * 130)

    for _, e in df.iterrows():
        rsi_flag = '✓' if e.get('rsi_w_above_sma') else '✗'
        print(f"{e['date_str']:<11} {e['close']:>7.1f} {e['entry']:>7.1f} "
              f"{e['d_pos']:>5.2f} {e['w_pos']:>5.2f} "
              f"{e['tsla_rsi_d']:>5.1f} {e['tsla_rsi_w']:>5.1f} {rsi_flag:>5} "
              f"{e['spy_rsi']:>4.0f} {e['dist_52w']:>+5.0%} "
              f"{'Y' if e['d_turning'] else 'N':>5} {'Y' if e['w_turning'] else 'N':>5} "
              f"{e['fwd_1d']:>+5.1f}% {e['fwd_2d']:>+5.1f}% {e['fwd_3d']:>+5.1f}% "
              f"{e['fwd_5d']:>+5.1f}% {e['fwd_10d']:>+5.1f}% "
              f"{e['peak_5d']:>+5.1f}% {e['dd_5d']:>+5.1f}%")

    # ── Summary stats ────────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("AGGREGATE STATS")
    print(f"{'='*110}")

    for h in [1, 2, 3, 5, 10]:
        col = f'fwd_{h}d'
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        wr = (valid > 0).mean() * 100
        avg = valid.mean()
        med = valid.median()
        print(f"  {h:>2}d forward:  WR={wr:.0f}%  avg={avg:+.1f}%  median={med:+.1f}%  "
              f"best={valid.max():+.1f}%  worst={valid.min():+.1f}%")

    # ── Winner vs Loser factor comparison ────────────────────────────────────
    print(f"\n{'='*110}")
    print("FACTOR ANALYSIS: Winners (5d > +2%) vs Losers (5d < 0%)")
    print(f"{'='*110}")

    winners = df[df['fwd_5d'] > 2.0]
    losers = df[df['fwd_5d'] < 0.0]
    flat = df[(df['fwd_5d'] >= 0) & (df['fwd_5d'] <= 2.0)]

    print(f"\n  Winners: {len(winners)} ({len(winners)/n*100:.0f}%)  |  "
          f"Losers: {len(losers)} ({len(losers)/n*100:.0f}%)  |  "
          f"Flat: {len(flat)} ({len(flat)/n*100:.0f}%)")

    factors = ['d_pos', 'w_pos', 'tsla_rsi_d', 'tsla_rsi_w', 'spy_rsi', 'dist_52w', 'energy_ratio']
    bool_factors = ['d_turning', 'w_turning', 'rsi_w_above_sma', 'd_at_bottom']

    print(f"\n  {'Factor':<20} {'Winners':>10} {'Losers':>10} {'All':>10}  Signal")
    print(f"  {'─'*75}")

    for f in factors:
        w_avg = winners[f].mean() if len(winners) > 0 else np.nan
        l_avg = losers[f].mean() if len(losers) > 0 else np.nan
        a_avg = df[f].mean()
        diff = w_avg - l_avg if not np.isnan(w_avg) and not np.isnan(l_avg) else 0
        arrow = '<<<' if abs(diff) > 0.1 * abs(a_avg + 0.001) else ''
        print(f"  {f:<20} {w_avg:>10.2f} {l_avg:>10.2f} {a_avg:>10.2f}  {arrow}")

    for f in bool_factors:
        w_pct = winners[f].mean() * 100 if len(winners) > 0 else 0
        l_pct = losers[f].mean() * 100 if len(losers) > 0 else 0
        a_pct = df[f].mean() * 100
        diff = w_pct - l_pct
        arrow = '<<<' if abs(diff) > 15 else ''
        print(f"  {f:<20} {w_pct:>9.0f}% {l_pct:>9.0f}% {a_pct:>9.0f}%  {arrow}")

    # ── RSI regime breakdown ─────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("RSI REGIME BREAKDOWN (5d forward returns)")
    print(f"{'='*110}")

    # Weekly RSI above vs below SMA
    above = df[df['rsi_w_above_sma'] == True]
    below = df[df['rsi_w_above_sma'] == False]

    print(f"\n  TSLA Weekly RSI vs SMA(14):")
    if len(above) > 0:
        print(f"    RSI > SMA (recovering):  n={len(above):>3}  "
              f"5d avg={above['fwd_5d'].mean():+.1f}%  WR={((above['fwd_5d']>0).mean()*100):.0f}%  "
              f"10d avg={above['fwd_10d'].mean():+.1f}%")
    if len(below) > 0:
        print(f"    RSI < SMA (falling):     n={len(below):>3}  "
              f"5d avg={below['fwd_5d'].mean():+.1f}%  WR={((below['fwd_5d']>0).mean()*100):.0f}%  "
              f"10d avg={below['fwd_10d'].mean():+.1f}%")

    # Daily RSI buckets
    print(f"\n  TSLA Daily RSI buckets:")
    for lo, hi, label in [(0, 25, '<25 (extreme)'), (25, 35, '25-35 (oversold)'),
                          (35, 45, '35-45 (weak)'), (45, 100, '>45')]:
        bucket = df[(df['tsla_rsi_d'] >= lo) & (df['tsla_rsi_d'] < hi)]
        if len(bucket) == 0:
            continue
        print(f"    RSI {label:<20}:  n={len(bucket):>3}  "
              f"5d avg={bucket['fwd_5d'].mean():+.1f}%  WR={((bucket['fwd_5d']>0).mean()*100):.0f}%  "
              f"peak={bucket['peak_5d'].mean():+.1f}%  dd={bucket['dd_5d'].mean():+.1f}%")

    # Weekly RSI buckets
    print(f"\n  TSLA Weekly RSI buckets:")
    for lo, hi, label in [(0, 30, '<30 (extreme)'), (30, 40, '30-40'),
                          (40, 50, '40-50'), (50, 100, '>50')]:
        bucket = df[(df['tsla_rsi_w'] >= lo) & (df['tsla_rsi_w'] < hi)]
        if len(bucket) == 0:
            continue
        print(f"    RSI {label:<20}:  n={len(bucket):>3}  "
              f"5d avg={bucket['fwd_5d'].mean():+.1f}%  WR={((bucket['fwd_5d']>0).mean()*100):.0f}%  "
              f"peak={bucket['peak_5d'].mean():+.1f}%  dd={bucket['dd_5d'].mean():+.1f}%")

    # Turning point combinations
    print(f"\n  Momentum turning combinations:")
    for dt, wt, label in [(True, True, 'Both turning'), (True, False, 'Daily only'),
                          (False, True, 'Weekly only'), (False, False, 'Neither')]:
        bucket = df[(df['d_turning'] == dt) & (df['w_turning'] == wt)]
        if len(bucket) == 0:
            continue
        print(f"    {label:<20}:  n={len(bucket):>3}  "
              f"5d avg={bucket['fwd_5d'].mean():+.1f}%  WR={((bucket['fwd_5d']>0).mean()*100):.0f}%  "
              f"peak={bucket['peak_5d'].mean():+.1f}%  dd={bucket['dd_5d'].mean():+.1f}%")

    # Distance from 52w SMA
    print(f"\n  Distance from 52-week SMA:")
    for lo, hi, label in [(-999, -0.20, '<-20% (deep)'), (-0.20, -0.10, '-20% to -10%'),
                          (-0.10, 0.0, '-10% to 0%'), (0.0, 999, 'Above SMA')]:
        bucket = df[(df['dist_52w'] >= lo) & (df['dist_52w'] < hi)]
        if len(bucket) == 0:
            continue
        print(f"    {label:<20}:  n={len(bucket):>3}  "
              f"5d avg={bucket['fwd_5d'].mean():+.1f}%  WR={((bucket['fwd_5d']>0).mean()*100):.0f}%  "
              f"peak={bucket['peak_5d'].mean():+.1f}%  dd={bucket['dd_5d'].mean():+.1f}%")


if __name__ == '__main__':
    main()
