#!/usr/bin/env python3
"""
RSI Bottom Percentile — Forward-Return Analysis

For each TF (5min, 1h, 4h, daily, weekly), computes RSI(14) Wilder-smoothed on
that TF's bars and classifies each trading day into RSI-extreme buckets:

  Percentile thresholds (vs trailing 252-day RSI distribution):
    p10 = RSI at bottom 10th percentile  (extreme oversold)
    p15 = bottom 15th
    p20 = bottom 20th
    p25 = bottom 25th (top quartile of oversold)

  Absolute thresholds:
    abs25 = RSI < 25
    abs30 = RSI < 30
    abs35 = RSI < 35
    abs40 = RSI < 40

Signal groups:
  1. Per-TF single signals  (5 TFs × 8 thresholds = 40)
  2. Multi-TF confluence    (N+ TFs simultaneously at same percentile)
  3. Weekly-anchored combos (weekly extreme + 1h/4h/daily confirmation)

Forward tracking: identical to tf_state_targets.py
  - Hit rates for +3/5/7/10/15/20% profit targets
  - Hit rates for -3/5/7/10/15/20% stop levels
  - Expectancy at 5/10/20/30/45/60 day holds ($100K capital)
  - Race: which hits first (+10% vs -10%)

Usage:
    python3 -m v15.validation.rsi_bottom_targets --tsla data/TSLAMin.txt
    python3 -m v15.validation.rsi_bottom_targets --tsla data/TSLAMin.txt --detail
    python3 -m v15.validation.rsi_bottom_targets --tsla data/TSLAMin.txt --tf daily
    python3 -m v15.validation.rsi_bottom_targets --tsla data/TSLAMin.txt --end-year 2025
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.tf_state_backtest import load_all_tfs, _safe_call
from v15.validation.tf_state_targets import (
    forward_analysis,
    print_result,
    print_summary_table,
    PROFIT_TARGETS,
    STOP_TARGETS,
    HOLD_DURATIONS,
    MAX_FORWARD,
)

# ── RSI computation ───────────────────────────────────────────────────────────

def _wilder_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI via exponential moving average.  Returns array same length as closes."""
    n = len(closes)
    if n < period + 1:
        return np.full(n, np.nan)

    rsi = np.full(n, np.nan)
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed: simple average over first period
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    alpha = 1.0 / period
    for i in range(period, n - 1):
        g = gains[i]
        l = losses[i]
        avg_gain = avg_gain * (1 - alpha) + g * alpha
        avg_loss = avg_loss * (1 - alpha) + l * alpha
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


# ── Per-day RSI state computation ─────────────────────────────────────────────

def compute_rsi_states(
    tf_data: dict,
    trading_dates: pd.DatetimeIndex,
    rsi_period: int = 14,
    percentile_window: int = 252,
    warmup_bars: int = 300,
) -> list:
    """
    For each trading day, compute RSI on each TF and classify
    the current RSI into percentile buckets and absolute buckets.

    Returns list of dicts: {'date': ..., '5min': state_dict, '1h': ..., ...}
    where state_dict = {
        'rsi':      float,   # current RSI value on that TF
        'pct_rank': float,   # percentile rank in [0, 100]; 0 = most extreme oversold
        'p10': bool,         # RSI at bottom 10th percentile of trailing history
        'p15': bool,
        'p20': bool,
        'p25': bool,
        'abs25': bool,       # RSI < 25
        'abs30': bool,
        'abs35': bool,
        'abs40': bool,
    }
    """
    print(f"Pre-computing RSI({rsi_period}) for all TFs...")
    t0 = time.time()

    # Pre-compute full RSI series for each TF
    rsi_series = {}
    for tf, df in tf_data.items():
        if df is None or len(df) < rsi_period + 2:
            continue
        closes = df['close'].values.astype(float)
        rsi_vals = _wilder_rsi(closes, rsi_period)
        rsi_series[tf] = pd.Series(rsi_vals, index=df.index)
        print(f"  {tf}: {len(df):,} bars, RSI range [{np.nanmin(rsi_vals):.1f}, {np.nanmax(rsi_vals):.1f}]")

    print(f"RSI pre-computation done in {time.time()-t0:.0f}s")
    print(f"\nComputing per-day RSI states for {len(trading_dates):,} trading days...")
    t0 = time.time()

    rows = []
    for i, date in enumerate(trading_dates):
        if i < warmup_bars:
            continue
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(trading_dates)} ({time.time()-t0:.0f}s)...")

        day_state = {'date': date}

        for tf in ['5min', '1h', '4h', 'daily', 'weekly']:
            rsi_s = rsi_series.get(tf)
            if rsi_s is None:
                continue

            # Most recent bar on this TF as of this trading date
            loc = rsi_s.index.searchsorted(date + pd.Timedelta(days=1), side='left') - 1
            if loc < percentile_window:
                continue

            current_rsi = float(rsi_s.iloc[loc])
            if np.isnan(current_rsi):
                continue

            # Trailing percentile rank: what % of the last `percentile_window` RSI
            # readings are >= current_rsi? (0 = current is the lowest seen)
            hist = rsi_s.iloc[max(0, loc - percentile_window): loc + 1].values
            hist = hist[~np.isnan(hist)]
            if len(hist) < 50:
                continue

            pct_rank = float(np.mean(hist <= current_rsi) * 100)

            day_state[tf] = {
                'rsi':      current_rsi,
                'pct_rank': pct_rank,
                'p10':      pct_rank <= 10.0,
                'p15':      pct_rank <= 15.0,
                'p20':      pct_rank <= 20.0,
                'p25':      pct_rank <= 25.0,
                'abs25':    current_rsi < 25.0,
                'abs30':    current_rsi < 30.0,
                'abs35':    current_rsi < 35.0,
                'abs40':    current_rsi < 40.0,
            }

        rows.append(day_state)

    print(f"  Done in {time.time()-t0:.0f}s — {len(rows):,} rows")
    return rows


# ── Signal definitions ────────────────────────────────────────────────────────

ALL_TFS    = ['5min', '1h', '4h', 'daily', 'weekly']
PCT_KEYS   = ['p10', 'p15', 'p20', 'p25']
ABS_KEYS   = ['abs25', 'abs30', 'abs35', 'abs40']
PCT_LABELS = {'p10': 'pct10', 'p15': 'pct15', 'p20': 'pct20', 'p25': 'pct25'}
ABS_LABELS = {'abs25': 'RSI<25', 'abs30': 'RSI<30', 'abs35': 'RSI<35', 'abs40': 'RSI<40'}


def _flag(s, tf, key):
    """Helper: is this TF's RSI flag set?"""
    return bool(s.get(tf) and s[tf].get(key, False))


def _count_flag(s, key):
    """Count TFs with this flag set."""
    return sum(1 for tf in ALL_TFS if _flag(s, tf, key))


def build_signals(tf_filter=None):
    """
    Build all signal (name, fn) pairs.

    tf_filter: if set (e.g. 'daily'), only include signals for that TF.
    """
    sigs = []

    # ── Group 1: Per-TF single signals ───────────────────────────────────────
    tfs = [tf_filter] if tf_filter else ALL_TFS

    for tf in tfs:
        for k in PCT_KEYS:
            lbl = PCT_LABELS[k]
            sigs.append((
                f'{tf} RSI {lbl}',
                (lambda s, _tf=tf, _k=k: _flag(s, _tf, _k)),
            ))
        for k in ABS_KEYS:
            lbl = ABS_LABELS[k]
            sigs.append((
                f'{tf} {lbl}',
                (lambda s, _tf=tf, _k=k: _flag(s, _tf, _k)),
            ))

    if tf_filter:
        return sigs

    # ── Group 2: Multi-TF confluence (N+ TFs simultaneously) ─────────────────
    for k in PCT_KEYS:
        lbl = PCT_LABELS[k]
        for n in [3, 4, 5]:
            sigs.append((
                f'{n}+TFs RSI {lbl}',
                (lambda s, _k=k, _n=n: _count_flag(s, _k) >= _n),
            ))

    for k in ABS_KEYS:
        lbl = ABS_LABELS[k]
        for n in [3, 4, 5]:
            sigs.append((
                f'{n}+TFs {lbl}',
                (lambda s, _k=k, _n=n: _count_flag(s, _k) >= _n),
            ))

    # ── Group 3: Weekly-anchored combos ──────────────────────────────────────
    combos = [
        # Weekly extreme + lower TF confirmation
        ('wkly_p10 + 1h_p15',      lambda s: _flag(s,'weekly','p10') and _flag(s,'1h','p15')),
        ('wkly_p10 + 4h_p15',      lambda s: _flag(s,'weekly','p10') and _flag(s,'4h','p15')),
        ('wkly_p10 + daily_p15',   lambda s: _flag(s,'weekly','p10') and _flag(s,'daily','p15')),
        ('wkly_p15 + daily_p15',   lambda s: _flag(s,'weekly','p15') and _flag(s,'daily','p15')),
        ('wkly_p15 + 1h_p20',      lambda s: _flag(s,'weekly','p15') and _flag(s,'1h','p20')),
        ('wkly_p15 + 4h_p20',      lambda s: _flag(s,'weekly','p15') and _flag(s,'4h','p20')),
        ('wkly_p15 + daily_p20',   lambda s: _flag(s,'weekly','p15') and _flag(s,'daily','p20')),
        ('wkly_p20 + daily_p20',   lambda s: _flag(s,'weekly','p20') and _flag(s,'daily','p20')),
        ('wkly_p20 + 1h_p25',      lambda s: _flag(s,'weekly','p20') and _flag(s,'1h','p25')),
        # Weekly + 2 lower TFs
        ('wkly_p10 + daily_p15 + 1h_p20',
         lambda s: (_flag(s,'weekly','p10') and _flag(s,'daily','p15')
                    and _flag(s,'1h','p20'))),
        ('wkly_p15 + daily_p15 + 1h_p20',
         lambda s: (_flag(s,'weekly','p15') and _flag(s,'daily','p15')
                    and _flag(s,'1h','p20'))),
        ('wkly_p15 + daily_p20 + 4h_p25',
         lambda s: (_flag(s,'weekly','p15') and _flag(s,'daily','p20')
                    and _flag(s,'4h','p25'))),
        # Absolute combos
        ('wkly_RSI<35 + daily_RSI<40',
         lambda s: _flag(s,'weekly','abs35') and _flag(s,'daily','abs40')),
        ('wkly_RSI<30 + daily_RSI<35',
         lambda s: _flag(s,'weekly','abs30') and _flag(s,'daily','abs35')),
        ('wkly_RSI<35 + 1h_RSI<35',
         lambda s: _flag(s,'weekly','abs35') and _flag(s,'1h','abs35')),
    ]
    sigs.extend(combos)

    return sigs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='RSI bottom percentile forward-return analysis')
    parser.add_argument('--tsla', type=str, default='data/TSLAMin.txt')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end',   type=str, default='2025-12-31')
    parser.add_argument('--days',  type=int, default=MAX_FORWARD,
                        help=f'Max forward days (default: {MAX_FORWARD})')
    parser.add_argument('--rsi-period', type=int, default=14, dest='rsi_period')
    parser.add_argument('--pct-window', type=int, default=252, dest='pct_window',
                        help='Trailing bars for percentile rank (default: 252 daily equiv)')
    parser.add_argument('--start-year', type=int, default=2015, dest='start_year')
    parser.add_argument('--end-year',   type=int, default=2024, dest='end_year')
    parser.add_argument('--capital', type=float, default=100_000.0)
    parser.add_argument('--tf', type=str, default=None,
                        help='Filter to one TF (5min, 1h, 4h, daily, weekly)')
    parser.add_argument('--detail', action='store_true',
                        help='Print full breakdown for every signal')
    parser.add_argument('--min-n', type=int, default=3, dest='min_n',
                        help='Min firings to show in summary (default: 3)')
    args = parser.parse_args()

    print(f"\n{'='*75}")
    print("RSI BOTTOM PERCENTILE — FORWARD-RETURN ANALYSIS")
    print(f"RSI period: {args.rsi_period}  |  Percentile window: {args.pct_window} bars")
    print(f"IS period: {args.start_year}-{args.end_year}  |  Forward: {args.days}d")
    print(f"Capital: ${args.capital:,.0f}")
    if args.tf:
        print(f"TF filter: {args.tf} only")
    print(f"{'='*75}")

    tf_data = load_all_tfs(args.tsla, args.start, args.end)
    daily_df = tf_data['daily']
    trading_dates = daily_df.index

    state_rows = compute_rsi_states(
        tf_data, trading_dates,
        rsi_period=args.rsi_period,
        percentile_window=args.pct_window,
    )

    signals = build_signals(tf_filter=args.tf)
    print(f"\nRunning {len(signals)} signals...")
    t0 = time.time()

    results = []
    for name, fn in signals:
        r = forward_analysis(
            daily_df, state_rows, fn, name,
            start_year=args.start_year,
            end_year=args.end_year,
            max_forward=args.days,
            capital=args.capital,
        )
        results.append(r)
        n = r['n']
        if n >= args.min_n:
            hit10 = r['profit_rates'].get(0.10, 0)
            hit5s = r['stop_rates'].get(0.05, 0)
            e30   = np.mean(r['hold_pnl'].get(30, [0])) if r['hold_pnl'].get(30) else 0
            print(f"  {name:<46}  n={n:>4}  +10%:{hit10:>4.0%}  -5%:{hit5s:>4.0%}  E[30d]=${e30:>8,.0f}")

    print(f"\nDone in {time.time()-t0:.0f}s")

    # Summary table (only signals with n >= min_n)
    filtered = [r for r in results if r['n'] >= args.min_n]
    print_summary_table(filtered, fwd=args.days)

    # Detailed breakdown
    if args.detail:
        print(f"\n{'='*75}")
        print("DETAILED BREAKDOWN (n >= 3)")
        print(f"{'='*75}")
        for r in sorted(filtered, key=lambda r: r.get('profit_rates', {}).get(0.10, 0), reverse=True):
            print_result(r)
    else:
        print(f"\n{'='*75}")
        print("TOP 5 BY +10% HIT RATE  — use --detail for all")
        print(f"{'='*75}")
        top5 = sorted(
            [r for r in filtered if r['n'] >= 5],
            key=lambda r: r.get('profit_rates', {}).get(0.10, 0),
            reverse=True
        )[:5]
        for r in top5:
            print_result(r)

    print(f"\n{'='*75}")
    print("DONE")
    print(f"{'='*75}")


if __name__ == '__main__':
    main()
