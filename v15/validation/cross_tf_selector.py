#!/usr/bin/env python3
"""
Cross-TF Trade Selector — 1h entries gated by active 4h signal.

Hypothesis: 1h trades that occur WHILE a same-direction 4h bounce is active
are higher quality than random 1h trades. The 4h signal acts as a quality gate.

Approach (post-hoc analysis — no changes to backtest engine):
  1. Run 1h baseline backtest per year → get full trade list with entry timestamps
  2. Run 4h baseline backtest per year → get full trade list with entry timestamps
  3. For each 1h trade, check if a same-direction 4h bounce was active at that time
     (active = 4h entry_time ≤ 1h entry_time ≤ 4h entry_time + hold_bars * 4h)
  4. Split 1h trades into aligned (4h-gated) vs non-aligned, compare stats

Usage:
    python3 -m v15.validation.cross_tf_selector
    python3 -m v15.validation.cross_tf_selector --years 2015-2024 --oos-year 2025
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.medium_tf_backtest import (
    _load_and_resample,
    _prepare_year,
    _run_year,
    TF_PARAMS,
)


# ---------------------------------------------------------------------------
# Alignment logic
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string, return None on failure."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except Exception:
        try:
            return datetime.strptime(ts_str[:19], '%Y-%m-%dT%H:%M:%S')
        except Exception:
            return None


def find_aligned(trades_1h: list, trades_4h: list,
                 direction_match: bool = True,
                 max_gap_hours: int = 20) -> list:
    """
    Return boolean mask: True for each 1h trade that was entered while a
    same-direction 4h bounce was active.

    A 4h trade is 'active' at time T when:
        4h_entry <= T  AND  T < 4h_entry + hold_bars * 4h

    Args:
        trades_1h: list of Trade objects from 1h backtest
        trades_4h: list of Trade objects from 4h backtest
        direction_match: only match same-direction trades
        max_gap_hours: cap on 4h trade lifetime for matching (safety)
    """
    # Pre-parse 4h entry/exit windows (bounce trades only)
    windows_4h = []
    for t in trades_4h:
        if getattr(t, 'signal_type', 'bounce') != 'bounce':
            continue
        ts = _parse_ts(getattr(t, 'entry_time', ''))
        if ts is None:
            continue
        hold = max(getattr(t, 'hold_bars', 1), 1)
        # Each 4h bar = 4 hours
        duration_h = min(hold * 4, max_gap_hours)
        windows_4h.append({
            'entry': ts,
            'exit': ts + timedelta(hours=duration_h),
            'direction': getattr(t, 'direction', ''),
        })

    mask = []
    for t in trades_1h:
        ts = _parse_ts(getattr(t, 'entry_time', ''))
        if ts is None:
            mask.append(False)
            continue
        direction = getattr(t, 'direction', '')
        matched = False
        for w in windows_4h:
            if direction_match and w['direction'] != direction:
                continue
            if w['entry'] <= ts <= w['exit']:
                matched = True
                break
        mask.append(matched)

    return mask


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _compute_stats(trades: list, mask: list, label: str) -> dict:
    """Compute stats for trades where mask=True."""
    subset = [t for t, m in zip(trades, mask) if m]
    if not subset:
        return {'label': label, 'n': 0, 'wr': 0.0, 'avg_pnl': 0.0, 'total_pnl': 0.0,
                'pf': 0.0, 'avg_hold': 0.0}
    pnls = [t.pnl for t in subset]
    wins = sum(1 for p in pnls if p > 0)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    holds = [getattr(t, 'hold_bars', 0) for t in subset]
    return {
        'label': label,
        'n': len(subset),
        'wr': wins / len(subset),
        'avg_pnl': np.mean(pnls),
        'total_pnl': sum(pnls),
        'pf': gross_profit / max(gross_loss, 1e-6),
        'avg_hold': np.mean(holds) if holds else 0.0,
    }


def _print_stats_table(rows: list):
    hdr = (f"{'Subset':<22} {'n':>6} {'WR':>7} {'Avg P&L':>11} "
           f"{'Total P&L':>14} {'PF':>6} {'AvgHold':>9}")
    print(hdr)
    print('-' * 80)
    for r in rows:
        if r['n'] == 0:
            print(f"  {r['label']:<20} {'0':>6} {'N/A':>7}")
            continue
        print(f"  {r['label']:<20} {r['n']:>6,} {r['wr']:>6.1%} ${r['avg_pnl']:>9,.0f} "
              f"${r['total_pnl']:>12,.0f} {r['pf']:>5.2f} {r['avg_hold']:>7.1f}h")


# ---------------------------------------------------------------------------
# Per-year run
# ---------------------------------------------------------------------------

def _run_year_both_tfs(year_data_1h: dict, year_data_4h: dict,
                       capital: float, vix_df) -> Optional[tuple]:
    """
    Run 1h and 4h backtests for one year, return (trades_1h, trades_4h).
    """
    result_1h = _run_year(year_data_1h, '1h', capital, vix_df, signal_filters=None)
    result_4h = _run_year(year_data_4h, '4h', capital, vix_df, signal_filters=None)

    if result_1h is None or result_4h is None:
        return None

    trades_1h = result_1h[1]  # list of Trade objects
    trades_4h = result_4h[1]
    return trades_1h, trades_4h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Cross-TF trade selector analysis')
    parser.add_argument('--tsla',     type=str, default='data/TSLAMin.txt')
    parser.add_argument('--spy',      type=str, default='data/SPYMin.txt')
    parser.add_argument('--years',    type=str, default='2015-2024')
    parser.add_argument('--oos-year', type=int, default=2025)
    parser.add_argument('--capital',  type=float, default=100_000.0)
    parser.add_argument('--max-gap',  type=int, default=20,
                        help='Max hours a 4h trade is considered active (default: 20 = 5 bars)')
    args = parser.parse_args()

    parts = args.years.split('-')
    start_year = int(parts[0])
    end_year   = int(parts[1]) if len(parts) > 1 else start_year
    is_years   = list(range(start_year, end_year + 1))

    print(f"\n{'='*70}")
    print(f"CROSS-TF SELECTOR — 1h gated by active 4h signal")
    print(f"IS: {start_year}-{end_year}  OOS: {args.oos_year}")
    print(f"Max 4h active window: {args.max_gap}h")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Load & resample to both TFs
    # ------------------------------------------------------------------
    print("\nLoading data...")
    t0 = time.time()
    spy_path = args.spy if os.path.isfile(args.spy) else None

    # Load both TF resamplings
    tsla_1h, spy_1h, daily_tsla, weekly_tsla, daily_spy, monthly_tsla = _load_and_resample(
        args.tsla, spy_path, '1h')
    tsla_4h, spy_4h, _, _, _, _ = _load_and_resample(args.tsla, spy_path, '4h')
    print(f"  Data loaded in {time.time()-t0:.1f}s")

    # VIX
    vix_df = None
    try:
        from v15.validation.vix_loader import load_vix_daily
        vix_df = load_vix_daily(start='2014-01-01', end='2025-12-31')
        print(f"  VIX: {len(vix_df):,} rows")
    except Exception as e:
        print(f"  VIX load failed: {e}")

    # ------------------------------------------------------------------
    # IS analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"IS ANALYSIS — {start_year}-{end_year}")
    print(f"{'='*70}")

    all_trades_1h_is = []
    all_trades_4h_is = []

    for year in is_years:
        yd_1h = _prepare_year(tsla_1h, spy_1h, daily_tsla, weekly_tsla, year,
                               monthly_tsla=monthly_tsla, primary_tf='1h')
        yd_4h = _prepare_year(tsla_4h, spy_4h, daily_tsla, weekly_tsla, year,
                               monthly_tsla=monthly_tsla, primary_tf='4h')
        if yd_1h is None or yd_4h is None:
            print(f"  {year}: skipped (no data)")
            continue

        t_yr = time.time()
        res = _run_year_both_tfs(yd_1h, yd_4h, args.capital, vix_df)
        if res is None:
            print(f"  {year}: too few bars")
            continue

        trades_1h, trades_4h = res
        all_trades_1h_is.extend(trades_1h)
        all_trades_4h_is.extend(trades_4h)

        # Quick per-year alignment count
        bounces_1h = [t for t in trades_1h if getattr(t, 'signal_type', '') == 'bounce']
        bounces_4h = [t for t in trades_4h if getattr(t, 'signal_type', '') == 'bounce']
        mask = find_aligned(bounces_1h, bounces_4h, max_gap_hours=args.max_gap)
        aligned = sum(mask)
        pct = aligned / max(len(bounces_1h), 1)
        pnl_aligned = sum(t.pnl for t, m in zip(bounces_1h, mask) if m)
        pnl_all = sum(t.pnl for t in bounces_1h)
        print(f"  {year}: 1h={len(bounces_1h):4} trades  4h={len(bounces_4h):3} trades  "
              f"aligned={aligned:4} ({pct:.0%})  "
              f"aligned_pnl=${pnl_aligned:>10,.0f}  total_pnl=${pnl_all:>10,.0f}  ({time.time()-t_yr:.1f}s)")

    # ------------------------------------------------------------------
    # IS aggregate stats
    # ------------------------------------------------------------------
    bounces_1h_is = [t for t in all_trades_1h_is if getattr(t, 'signal_type', '') == 'bounce']
    bounces_4h_is = [t for t in all_trades_4h_is if getattr(t, 'signal_type', '') == 'bounce']

    mask_is = find_aligned(bounces_1h_is, bounces_4h_is, max_gap_hours=args.max_gap)
    mask_not_is = [not m for m in mask_is]

    rows_is = [
        _compute_stats(bounces_1h_is, [True]*len(bounces_1h_is), '1h ALL (baseline)'),
        _compute_stats(bounces_1h_is, mask_is,     '1h + 4h aligned'),
        _compute_stats(bounces_1h_is, mask_not_is, '1h NOT aligned'),
        _compute_stats(bounces_4h_is, [True]*len(bounces_4h_is), '4h baseline'),
    ]

    print(f"\n{'='*70}")
    print(f"IS AGGREGATE — {start_year}-{end_year}")
    print(f"{'='*70}")
    _print_stats_table(rows_is)

    # Alignment summary
    n_aligned = sum(mask_is)
    n_total   = len(bounces_1h_is)
    print(f"\n  Alignment rate: {n_aligned:,}/{n_total:,} = {n_aligned/max(n_total,1):.1%} of 1h trades aligned with active 4h")
    if n_aligned > 0:
        r_aligned = _compute_stats(bounces_1h_is, mask_is, '')
        r_all     = _compute_stats(bounces_1h_is, [True]*n_total, '')
        lift_avg  = r_aligned['avg_pnl'] / max(r_all['avg_pnl'], 1)
        lift_wr   = r_aligned['wr'] - r_all['wr']
        print(f"  Avg P&L lift (aligned vs all): {lift_avg:.2f}x")
        print(f"  WR lift (aligned vs all): {lift_wr:+.1%}")

    # ------------------------------------------------------------------
    # OOS validation
    # ------------------------------------------------------------------
    if args.oos_year > 0:
        print(f"\n{'='*70}")
        print(f"OOS VALIDATION — {args.oos_year}")
        print(f"{'='*70}")

        yd_1h_oos = _prepare_year(tsla_1h, spy_1h, daily_tsla, weekly_tsla, args.oos_year,
                                   monthly_tsla=monthly_tsla, primary_tf='1h')
        yd_4h_oos = _prepare_year(tsla_4h, spy_4h, daily_tsla, weekly_tsla, args.oos_year,
                                   monthly_tsla=monthly_tsla, primary_tf='4h')

        if yd_1h_oos is None or yd_4h_oos is None:
            print(f"  {args.oos_year}: no OOS data")
        else:
            t_oos = time.time()
            res_oos = _run_year_both_tfs(yd_1h_oos, yd_4h_oos, args.capital, vix_df)
            if res_oos is None:
                print(f"  {args.oos_year}: too few bars")
            else:
                trades_1h_oos, trades_4h_oos = res_oos
                bounces_1h_oos = [t for t in trades_1h_oos if getattr(t, 'signal_type', '') == 'bounce']
                bounces_4h_oos = [t for t in trades_4h_oos if getattr(t, 'signal_type', '') == 'bounce']

                mask_oos     = find_aligned(bounces_1h_oos, bounces_4h_oos, max_gap_hours=args.max_gap)
                mask_not_oos = [not m for m in mask_oos]

                rows_oos = [
                    _compute_stats(bounces_1h_oos, [True]*len(bounces_1h_oos), '1h ALL (baseline)'),
                    _compute_stats(bounces_1h_oos, mask_oos,     '1h + 4h aligned'),
                    _compute_stats(bounces_1h_oos, mask_not_oos, '1h NOT aligned'),
                    _compute_stats(bounces_4h_oos, [True]*len(bounces_4h_oos), '4h baseline'),
                ]

                n_aligned_oos = sum(mask_oos)
                n_total_oos   = len(bounces_1h_oos)
                print(f"  OOS alignment: {n_aligned_oos}/{n_total_oos} = "
                      f"{n_aligned_oos/max(n_total_oos,1):.1%} of 1h trades")
                print(f"  Elapsed: {time.time()-t_oos:.1f}s\n")
                _print_stats_table(rows_oos)

                if n_aligned_oos > 0:
                    r_aligned_oos = _compute_stats(bounces_1h_oos, mask_oos, '')
                    r_all_oos     = _compute_stats(bounces_1h_oos, [True]*n_total_oos, '')
                    lift_avg_oos  = r_aligned_oos['avg_pnl'] / max(r_all_oos['avg_pnl'], 1)
                    lift_wr_oos   = r_aligned_oos['wr'] - r_all_oos['wr']
                    print(f"\n  OOS Avg P&L lift (aligned vs all): {lift_avg_oos:.2f}x")
                    print(f"  OOS WR lift (aligned vs all): {lift_wr_oos:+.1%}")

    print(f"\n{'='*70}")
    print("CROSS-TF SELECTOR COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
