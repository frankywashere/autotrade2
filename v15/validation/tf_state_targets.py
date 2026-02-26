#!/usr/bin/env python3
"""
Multi-TF Signal Forward-Return Target Analysis

For each market insights signal, tracks forward returns after entry and measures:
  - Hit rate at each profit target (+3%, +5%, +10%, +15%, +20%)
  - Hit rate at each stop level   (-3%, -5%, -10%, -15%, -20%)
  - Average / median days to reach each target
  - Which target hits FIRST (profit vs stop race)
  - Expectancy at various hold durations (5d, 10d, 20d, 30d)

This determines the NATURAL hold period and optimal exit levels from
the signal structure itself — no need to guess.

Usage:
    python3 -m v15.validation.tf_state_targets --tsla data/TSLAMin.txt
    python3 -m v15.validation.tf_state_targets --tsla data/TSLAMin.txt --days 60
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Reuse TF state helpers from tf_state_backtest
from v15.validation.tf_state_backtest import (
    load_all_tfs,
    compute_daily_states,
    SIGNALS,
    _mt,
    _count_near_bottom,
    _count_near_bottom_tfs,
    _safe_call,
)

# ── Target levels to test ────────────────────────────────────────────────────
PROFIT_TARGETS = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
STOP_TARGETS   = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
HOLD_DURATIONS = [5, 10, 20, 30, 45, 60]   # days — expectancy at each hold period
MAX_FORWARD    = 60                          # max days to track forward


# ── Forward-return analysis engine ──────────────────────────────────────────

def forward_analysis(
    daily_df: pd.DataFrame,
    state_rows: list,
    signal_fn,
    signal_name: str,
    start_year: int = 2015,
    end_year: int = 2024,
    max_forward: int = MAX_FORWARD,
    capital: float = 100_000.0,
) -> dict:
    """
    For each signal firing, compute forward-return profile.
    Returns dict of statistics.
    """
    dates  = daily_df.index
    opens  = daily_df['open'].values.astype(float)
    highs  = daily_df['high'].values.astype(float)
    lows   = daily_df['low'].values.astype(float)
    closes = daily_df['close'].values.astype(float)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # Find all signal firing dates (no re-entry logic — all firings independent)
    firings = []
    for row in state_rows:
        date = row['date']
        if date.year < start_year or date.year > end_year:
            continue
        states = {tf: row.get(tf) for tf in ['5min', '1h', '4h', 'daily', 'weekly']}
        if _safe_call(signal_fn, states):
            di = date_to_idx.get(date)
            if di is not None and di + 1 < len(dates):
                firings.append(di + 1)  # entry at next-day open

    if not firings:
        return {'name': signal_name, 'n': 0}

    # For each firing, track day-by-day forward return
    # Record: days_to_profit[tgt], days_to_stop[tgt], hold_pnl[hold_days]
    profit_hits  = {t: [] for t in PROFIT_TARGETS}   # days to hit
    stop_hits    = {t: [] for t in STOP_TARGETS}      # days to hit
    profit_rates = {t: 0  for t in PROFIT_TARGETS}    # count hit
    stop_rates   = {t: 0  for t in STOP_TARGETS}

    # Race stats: which hits first? (+10% vs -10%)
    race_pairs = [(0.10, 0.10), (0.05, 0.05), (0.15, 0.10), (0.10, 0.05)]
    race_results = {p: {'profit_first': 0, 'stop_first': 0, 'neither': 0}
                    for p in race_pairs}

    # Expectancy: hold N days from entry, take close price
    hold_pnl = {h: [] for h in HOLD_DURATIONS}

    for entry_di in firings:
        entry_price = opens[entry_di]
        if entry_price <= 0:
            continue

        # Track intraday high/low over forward window
        days_to_profit_first = {}  # tgt -> first day that hits
        days_to_stop_first   = {}

        for fwd in range(max_forward):
            bar = entry_di + fwd
            if bar >= len(dates):
                break

            # Use intraday range: high for profit, low for stop
            # (entry bar: only close available, not high/low intraday from entry open)
            if fwd == 0:
                day_high  = max(entry_price, closes[bar])
                day_low   = min(entry_price, closes[bar])
            else:
                day_high = highs[bar]
                day_low  = lows[bar]

            pct_high = (day_high - entry_price) / entry_price
            pct_low  = (day_low  - entry_price) / entry_price

            for tgt in PROFIT_TARGETS:
                if tgt not in days_to_profit_first and pct_high >= tgt:
                    days_to_profit_first[tgt] = fwd

            for tgt in STOP_TARGETS:
                if tgt not in days_to_stop_first and pct_low <= -tgt:
                    days_to_stop_first[tgt] = fwd

            # Collect hold-duration P&L at close
            if fwd + 1 in hold_pnl:
                pct = (closes[bar] - entry_price) / entry_price
                hold_pnl[fwd + 1].append(pct * capital)

        # Aggregate
        for tgt in PROFIT_TARGETS:
            if tgt in days_to_profit_first:
                profit_rates[tgt]  += 1
                profit_hits[tgt].append(days_to_profit_first[tgt])

        for tgt in STOP_TARGETS:
            if tgt in days_to_stop_first:
                stop_rates[tgt]  += 1
                stop_hits[tgt].append(days_to_stop_first[tgt])

        # Race: profit vs stop
        for (pt, st) in race_pairs:
            dp = days_to_profit_first.get(pt)
            ds = days_to_stop_first.get(st)
            if dp is not None and (ds is None or dp <= ds):
                race_results[(pt, st)]['profit_first'] += 1
            elif ds is not None and (dp is None or ds < dp):
                race_results[(pt, st)]['stop_first']   += 1
            else:
                race_results[(pt, st)]['neither']       += 1

    n = len(firings)
    return {
        'name':          signal_name,
        'n':             n,
        'profit_rates':  {t: profit_rates[t] / n for t in PROFIT_TARGETS},
        'stop_rates':    {t: stop_rates[t]   / n for t in STOP_TARGETS},
        'profit_days':   {t: profit_hits[t]      for t in PROFIT_TARGETS},
        'stop_days':     {t: stop_hits[t]        for t in STOP_TARGETS},
        'race_results':  race_results,
        'hold_pnl':      hold_pnl,
    }


def _fmt_days(day_list):
    if not day_list:
        return '  --  '
    return f'{np.mean(day_list):>4.1f}d (med {np.median(day_list):.0f}d)'


def print_result(r):
    n = r['n']
    if n == 0:
        print(f"  [{r['name']}]  n=0, no firings")
        return

    print(f"\n  [{r['name']}]  n={n} instances")
    print()

    # Profit targets
    print(f"    {'PROFIT TARGET':>16}  {'Hit Rate':>9}  {'Avg Days':>8}  {'Median':>7}  {'Count':>6}")
    print(f"    {'-'*55}")
    for tgt in PROFIT_TARGETS:
        rate  = r['profit_rates'][tgt]
        days  = r['profit_days'][tgt]
        avg_d = f"{np.mean(days):.1f}" if days else '--'
        med_d = f"{np.median(days):.0f}" if days else '--'
        bar   = '#' * int(rate * 20) + '.' * (20 - int(rate * 20))
        print(f"    +{tgt:>4.0%}          {rate:>8.0%}  {avg_d:>8}  {med_d:>7}  {len(days):>6}  {bar}")

    print()

    # Stop targets
    print(f"    {'STOP LEVEL':>16}  {'Hit Rate':>9}  {'Avg Days':>8}  {'Median':>7}  {'Count':>6}")
    print(f"    {'-'*55}")
    for tgt in STOP_TARGETS:
        rate  = r['stop_rates'][tgt]
        days  = r['stop_days'][tgt]
        avg_d = f"{np.mean(days):.1f}" if days else '--'
        med_d = f"{np.median(days):.0f}" if days else '--'
        bar   = '#' * int(rate * 20) + '.' * (20 - int(rate * 20))
        print(f"    -{tgt:>4.0%}          {rate:>8.0%}  {avg_d:>8}  {med_d:>7}  {len(days):>6}  {bar}")

    print()

    # Race: which hits first?
    print(f"    RACE — which target hits first?")
    for (pt, st), rc in r['race_results'].items():
        pf = rc['profit_first'] / n
        sf = rc['stop_first']   / n
        ni = rc['neither']      / n
        print(f"    +{pt:.0%} vs -{st:.0%}:  profit first {pf:.0%}  |  stop first {sf:.0%}  |  neither in {MAX_FORWARD}d {ni:.0%}")

    print()

    # Expectancy at hold durations
    print(f"    EXPECTANCY at hold durations (avg P&L on $100K, all firings):")
    print(f"    {'Hold':>6}  {'Avg P&L':>10}  {'Win%':>7}  {'n':>4}")
    print(f"    {'-'*35}")
    for h in HOLD_DURATIONS:
        pnls = r['hold_pnl'].get(h, [])
        if not pnls:
            print(f"    {h:>5}d  {'--':>10}")
            continue
        avg_pnl = np.mean(pnls)
        win_pct = sum(1 for p in pnls if p > 0) / len(pnls)
        print(f"    {h:>5}d  ${avg_pnl:>9,.0f}  {win_pct:>6.0%}  {len(pnls):>4}")


def print_summary_table(results, fwd=MAX_FORWARD):
    """Compact summary: profit hit rate at +5%, +10% and expectancy at 10d, 20d."""
    print(f"\n{'='*95}")
    print("SUMMARY TABLE  (all signals)")
    print(f"{'='*95}")
    hdr = (f"{'Signal':<46} {'n':>4}  "
           f"{'hit+5%':>7} {'hit+10%':>8} {'hit+15%':>8}  "
           f"{'hit-5%':>7} {'hit-10%':>8}  "
           f"{'E[10d]':>8} {'E[20d]':>8} {'E[30d]':>8}  "
           f"{'race+10/-10':>12}")
    print(hdr)
    print('-' * 95)

    ranked = sorted(results, key=lambda r: r.get('profit_rates', {}).get(0.10, 0), reverse=True)

    for r in ranked:
        if r['n'] == 0:
            print(f"  {r['name']:<46} {'0':>4}  {'--':>7} {'--':>8} {'--':>8}  {'--':>7} {'--':>8}  {'--':>8} {'--':>8} {'--':>8}  {'--':>12}")
            continue

        pr = r.get('profit_rates', {})
        sr = r.get('stop_rates',   {})
        hp = r.get('hold_pnl',     {})
        rc = r.get('race_results', {}).get((0.10, 0.10), {})

        def _e(h):
            pnls = hp.get(h, [])
            return f"${np.mean(pnls):>7,.0f}" if pnls else '     --'

        race_str = (f"{rc.get('profit_first',0)/r['n']:>4.0%}/"
                    f"{rc.get('stop_first',0)/r['n']:>3.0%}/"
                    f"{rc.get('neither',0)/r['n']:>3.0%}")

        print(f"  {r['name']:<46} {r['n']:>4}  "
              f"{pr.get(0.05,0):>6.0%} {pr.get(0.10,0):>7.0%} {pr.get(0.15,0):>7.0%}  "
              f"{sr.get(0.05,0):>6.0%} {sr.get(0.10,0):>7.0%}  "
              f"{_e(10):>8} {_e(20):>8} {_e(30):>8}  "
              f"{race_str:>12}")

    print(f"\n  Race format: +10% first / -10% first / neither in {fwd}d")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-TF dashboard signal forward-return target analysis')
    parser.add_argument('--tsla', type=str, default='data/TSLAMin.txt')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end',   type=str, default='2025-12-31')
    parser.add_argument('--days',  type=int, default=MAX_FORWARD,
                        help='Max forward days to track (default: 60)')
    parser.add_argument('--start-year', type=int, default=2015, dest='start_year')
    parser.add_argument('--end-year',   type=int, default=2024, dest='end_year')
    parser.add_argument('--capital', type=float, default=100_000.0)
    parser.add_argument('--detail', action='store_true',
                        help='Print full detail for every signal (default: summary only)')
    args = parser.parse_args()
    fwd = args.days  # forward window — passed explicitly everywhere

    print(f"\n{'='*70}")
    print("MULTI-TF SIGNAL FORWARD-RETURN TARGET ANALYSIS")
    print(f"Tracks: how long to reach +3/5/7/10/15/20% and -3/5/7/10/15/20%")
    print(f"Forward window: {fwd} trading days")
    print(f"IS period: {args.start_year}-{args.end_year}")
    print(f"{'='*70}")

    tf_data = load_all_tfs(args.tsla, args.start, args.end)
    daily_df = tf_data['daily']
    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates)

    print(f"\nRunning forward analysis for {len(SIGNALS)} signal combinations...")
    t0 = time.time()
    results = []
    for name, fn in SIGNALS:
        r = forward_analysis(
            daily_df, state_rows, fn, name,
            start_year=args.start_year,
            end_year=args.end_year,
            max_forward=fwd,
            capital=args.capital,
        )
        results.append(r)
        n = r['n']
        if n > 0:
            hit10 = r['profit_rates'].get(0.10, 0)
            hit5s = r['stop_rates'].get(0.05, 0)
            e20   = np.mean(r['hold_pnl'].get(20, [0])) if r['hold_pnl'].get(20) else 0
            print(f"  {name:<46}  n={n:>3}  +10%:{hit10:>4.0%}  -5%:{hit5s:>4.0%}  E[20d]=${e20:>7,.0f}")
        else:
            print(f"  {name:<46}  n=  0  no firings")

    print(f"\nDone in {time.time()-t0:.0f}s")

    # Summary table
    print_summary_table(results, fwd=fwd)

    # Detailed breakdown for signals with n >= 5
    if args.detail:
        print(f"\n{'='*70}")
        print("DETAILED BREAKDOWN (n >= 5)")
        print(f"{'='*70}")
        for r in sorted(results, key=lambda r: r.get('profit_rates', {}).get(0.10, 0), reverse=True):
            if r['n'] >= 5:
                print_result(r)
    else:
        # Auto-show top 5 by +10% hit rate with n >= 5
        print(f"\n{'='*70}")
        print("TOP 5 SIGNALS BY +10% HIT RATE (detailed)  — use --detail for all")
        print(f"{'='*70}")
        top5 = sorted(
            [r for r in results if r['n'] >= 5],
            key=lambda r: r.get('profit_rates', {}).get(0.10, 0),
            reverse=True
        )[:5]
        for r in top5:
            print_result(r)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
