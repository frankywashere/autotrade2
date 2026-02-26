#!/usr/bin/env python3
"""
tf_state_oos.py — IS / OOS Validation for Multi-TF Channel Confluence Signals

Splits the full history into:
  IS  = 2015-2023   (9 years in-sample)
  OOS-2024 = 2024   (1 year OOS)
  OOS-2025 = 2025   (1 year OOS, truly unseen)

Runs all A/B/C/D/E signals from tf_state_backtest.py with hold=30d, stop=20%
(optimum from tf_state_targets.py research), then prints a comparison table:

  Signal  |  IS n  IS WR  IS P&L  |  OOS24 n  OOS24 WR  OOS24 P&L  |  OOS25 n  OOS25 WR  OOS25 P&L
  --------|...

Also runs a walk-forward analysis: for each IS/OOS window pair, picks the
best IS signal and measures if it persists OOS.

Usage:
    python3 -m v15.validation.tf_state_oos --tsla data/TSLAMin.txt
    python3 -m v15.validation.tf_state_oos --tsla data/TSLAMin.txt --hold 30 --stop 0.15
    python3 -m v15.validation.tf_state_oos --tsla data/TSLAMin.txt --hold 45
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.tf_state_backtest import (
    load_all_tfs,
    compute_daily_states,
    run_backtest,
    SIGNALS,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def run_split(daily_df, state_rows, hold, stop, capital, start_year, end_year, label):
    """Run all signals for a given year range.  Returns list of result dicts."""
    print(f"\n  [{label}] hold={hold}d  stop={stop:.0%}  years={start_year}-{end_year}")
    results = []
    for name, fn in SIGNALS:
        r = run_backtest(
            daily_df, state_rows, fn, name,
            capital=capital,
            max_hold_days=hold,
            stop_pct=stop,
            start_year=start_year,
            end_year=end_year,
        )
        r['label'] = label
        results.append(r)
    return results


def print_oos_table(is_res, oos24_res, oos25_res, hold, capital):
    """Print 3-column IS / OOS24 / OOS25 comparison table."""
    is_map    = {r['name']: r for r in is_res}
    oos24_map = {r['name']: r for r in oos24_res}
    oos25_map = {r['name']: r for r in oos25_res}

    # Sort by IS P&L descending
    names = sorted(is_map.keys(), key=lambda n: is_map[n]['pnl'], reverse=True)

    def _fmt(r):
        if r['n'] == 0:
            return f"{'':>4}  {'':>4}  {'':>10}"
        return f"{r['n']:>4}  {r['wr']:>3.0%}  ${r['pnl']:>9,.0f}"

    print(f"\n{'='*115}")
    print(f"IS/OOS COMPARISON  hold={hold}d  stop=20%  capital=${capital:,.0f}")
    print(f"{'='*115}")
    hdr = (f"{'Signal':<46}  "
           f"{'--- IS 2015-2023 ---':>26}  "
           f"{'--- OOS 2024 ---':>22}  "
           f"{'--- OOS 2025 ---':>22}")
    print(hdr)
    sub = (f"{'':46}  "
           f"{'n':>4}  {'WR':>4}  {'P&L':>10}  "
           f"{'n':>4}  {'WR':>4}  {'P&L':>10}  "
           f"{'n':>4}  {'WR':>4}  {'P&L':>10}")
    print(sub)
    print('-' * 115)

    for name in names:
        ri  = is_map.get(name, {'n':0,'wr':0,'pnl':0})
        r24 = oos24_map.get(name, {'n':0,'wr':0,'pnl':0})
        r25 = oos25_map.get(name, {'n':0,'wr':0,'pnl':0})
        if ri['n'] == 0 and r24['n'] == 0 and r25['n'] == 0:
            continue
        print(f"  {name:<46}  {_fmt(ri)}  {_fmt(r24)}  {_fmt(r25)}")

    # Totals
    is_tot = sum(r['pnl'] for r in is_res)
    o24_tot = sum(r['pnl'] for r in oos24_res)
    o25_tot = sum(r['pnl'] for r in oos25_res)
    print('-' * 115)
    print(f"  {'TOTAL (all signals)':<46}  {'':>4}  {'':>4}  ${is_tot:>9,.0f}  "
          f"{'':>4}  {'':>4}  ${o24_tot:>9,.0f}  "
          f"{'':>4}  {'':>4}  ${o25_tot:>9,.0f}")


def walk_forward(daily_df, state_rows, hold, stop, capital):
    """
    Rolling IS/OOS walk-forward.
    Windows: IS=5yr, OOS=1yr  (6 windows covering 2015-2025)
    """
    windows = [
        (2015, 2019, 2020),
        (2016, 2020, 2021),
        (2017, 2021, 2022),
        (2018, 2022, 2023),
        (2019, 2023, 2024),
        (2020, 2024, 2025),
    ]

    print(f"\n{'='*90}")
    print(f"WALK-FORWARD ANALYSIS  (IS=5yr, OOS=1yr)  hold={hold}d  stop={stop:.0%}")
    print(f"{'='*90}")
    hdr = f"{'IS window':>15}  {'OOS':>6}  {'Best IS signal':<46}  {'IS P&L':>10}  {'OOS n':>6}  {'OOS P&L':>10}  {'OOS/IS':>7}"
    print(hdr)
    print('-' * 110)

    total_oos_pnl = 0.0
    total_is_pnl  = 0.0
    wins = 0

    for (is_start, is_end, oos_year) in windows:
        # Run IS
        is_results = []
        for name, fn in SIGNALS:
            r = run_backtest(
                daily_df, state_rows, fn, name,
                capital=capital, max_hold_days=hold, stop_pct=stop,
                start_year=is_start, end_year=is_end,
            )
            is_results.append(r)

        best_is = max(is_results, key=lambda r: r['pnl'])
        if best_is['n'] == 0:
            print(f"  {is_start}-{is_end}         {oos_year}   {'(no signals in IS)':>46}")
            continue

        # Run OOS with best IS signal
        best_fn = next(fn for name, fn in SIGNALS if name == best_is['name'])
        oos_r = run_backtest(
            daily_df, state_rows, best_fn, best_is['name'],
            capital=capital, max_hold_days=hold, stop_pct=stop,
            start_year=oos_year, end_year=oos_year,
        )

        ratio = oos_r['pnl'] / best_is['pnl'] if best_is['pnl'] > 0 else 0.0
        total_oos_pnl += oos_r['pnl']
        total_is_pnl  += best_is['pnl']
        if oos_r['pnl'] > 0:
            wins += 1

        print(f"  {is_start}-{is_end}         {oos_year}   "
              f"{best_is['name']:<46}  "
              f"${best_is['pnl']:>9,.0f}  "
              f"{oos_r['n']:>6}  "
              f"${oos_r['pnl']:>9,.0f}  "
              f"{ratio:>6.2f}x")

    print('-' * 110)
    ratio_total = total_oos_pnl / total_is_pnl if total_is_pnl > 0 else 0.0
    print(f"  {'TOTAL':>21}  {'':>46}  ${total_is_pnl:>9,.0f}  {'':>6}  "
          f"${total_oos_pnl:>9,.0f}  {ratio_total:>6.2f}x")
    print(f"  OOS positive: {wins}/{len(windows)} windows")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='tf_state IS/OOS split + walk-forward validation')
    parser.add_argument('--tsla', type=str, default='data/TSLAMin.txt')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end',   type=str, default='2025-12-31')
    parser.add_argument('--hold',  type=int, default=30,
                        help='Hold days (default: 30 — optimal from tf_state_targets)')
    parser.add_argument('--stop',  type=float, default=0.20,
                        help='Stop loss pct (default: 0.20)')
    parser.add_argument('--capital', type=float, default=100_000.0)
    parser.add_argument('--skip-wf', action='store_true', dest='skip_wf',
                        help='Skip walk-forward (faster)')
    args = parser.parse_args()

    print(f"\n{'='*75}")
    print("TF STATE SIGNALS — IS / OOS VALIDATION")
    print(f"Hold: {args.hold}d  Stop: {args.stop:.0%}  Capital: ${args.capital:,.0f}")
    print(f"{'='*75}")

    tf_data = load_all_tfs(args.tsla, args.start, args.end)
    daily_df = tf_data['daily']
    trading_dates = daily_df.index

    print(f"\nComputing TF states...")
    state_rows = compute_daily_states(tf_data, trading_dates)

    # IS and OOS splits
    is_res    = run_split(daily_df, state_rows, args.hold, args.stop, args.capital,
                          2015, 2023, 'IS 2015-2023')
    oos24_res = run_split(daily_df, state_rows, args.hold, args.stop, args.capital,
                          2024, 2024, 'OOS 2024')
    oos25_res = run_split(daily_df, state_rows, args.hold, args.stop, args.capital,
                          2025, 2025, 'OOS 2025')

    print_oos_table(is_res, oos24_res, oos25_res, args.hold, args.capital)

    if not args.skip_wf:
        walk_forward(daily_df, state_rows, args.hold, args.stop, args.capital)

    print(f"\n{'='*75}")
    print("DONE")
    print(f"{'='*75}")


if __name__ == '__main__':
    main()
