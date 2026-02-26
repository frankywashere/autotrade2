#!/usr/bin/env python3
"""
Short-Term TF State Forward-Return Analysis

Tests dashboard state patterns for SHORT-TERM prediction horizons:
  1d, 2d, 5d, 10d, 20d  (vs existing scripts that test 30-45d)

Specifically addresses the mixed-TF divergence pattern observed on the dashboard:
  5min: AT channel top (rally exhaustion, pos_pct > 0.80)
  1h:   momentum turning (rally exhaustion)
  4h + weekly: turning up from sell-off (bullish exhaustion)
  → Question: does short-term direction favor bulls (4h/wkly) or bears (5min top)?

15 signals across 4 groups:
  Group 1 — 5min top conditions (new — never tested for short-term)
  Group 2 — Higher TF bullish exhaustion (turning up from sell-off)
  Group 3 — Divergence patterns: 5min top + higher TF bullish (user's exact case)
  Group 4 — Consensus / comparison baseline

Three phases:
  Phase 1 — Forward return analysis (IS 2015-2024, all firings independent)
  Phase 2 — IS/OOS backtest for all signals  (hold=2d, stop=5%)
  Phase 3 — Walk-forward: IS=5yr, OOS=1yr, 6 windows

Usage:
    python3 -m v15.validation.short_term_tf_targets
    python3 -m v15.validation.short_term_tf_targets --tsla data/TSLAMin.txt
    python3 -m v15.validation.short_term_tf_targets --detail
    python3 -m v15.validation.short_term_tf_targets --skip-phase3
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.tf_state_backtest import (
    load_all_tfs,
    compute_daily_states,
    run_backtest,
    _mt,
    _count_near_bottom,
)

# ── Short-term specific parameters ───────────────────────────────────────────
SHORT_HOLD_DURATIONS = [1, 2, 5, 10, 20]   # days — short-term focus
MAX_FORWARD_DAYS = 20                        # only track 20 days forward
PROFIT_TARGETS = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
STOP_TARGETS   = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

# Phase 2 params (tighter for short-term signals)
PHASE2_HOLD = 2      # 2-day hold
PHASE2_STOP = 0.05   # 5% stop (tighter)


# ── Signal helpers ────────────────────────────────────────────────────────────

def _5min_at_top(s):
    """5min price in upper channel zone (pos > 0.80)."""
    return bool(s.get('5min') and s['5min']['pos_pct'] > 0.80)


def _5min_near_top(s):
    """5min price in upper-mid zone (pos > 0.65) — broader top filter."""
    return bool(s.get('5min') and s['5min']['pos_pct'] > 0.65)


def _4h_turning_up(s):
    """4h momentum turning up from lower half of channel (sell-off exhaustion)."""
    return bool(s.get('4h') and s['4h']['is_turning'] and s['4h']['pos_pct'] < 0.5)


def _weekly_turning_up(s):
    """Weekly momentum turning up from lower half of channel."""
    return bool(s.get('weekly') and s['weekly']['is_turning'] and s['weekly']['pos_pct'] < 0.5)


def _count_bullish_tfs(s):
    """Count TFs with pos_pct > 0.5 (upper half of channel = bullish zone)."""
    return sum(1 for tf in ['5min', '1h', '4h', 'daily', 'weekly']
               if s.get(tf) and s[tf]['pos_pct'] > 0.5)


def _all_neutral(s):
    """All TFs in mid-channel zone (0.25-0.75) — no strong directional signal."""
    tfs_data = [tf for tf in ['5min', '1h', '4h', 'daily', 'weekly'] if s.get(tf)]
    if not tfs_data:
        return False
    return all(0.25 < s[tf]['pos_pct'] < 0.75 for tf in tfs_data)


# ── 15 signal definitions ────────────────────────────────────────────────────

SIGNALS_SHORT = [
    # ── Group 1: 5min top conditions (new — never tested short-term) ─────────
    # These are "resistance" signals — price at upper channel boundary.
    # Hypothesis: short-term mean-reversion down (sell-off from top).

    ('G1_5min_top',
     lambda s: _5min_at_top(s)),

    ('G1_5min_top_stressed',
     lambda s: _5min_at_top(s) and bool(s.get('5min') and s['5min']['stressed'])),

    # is_turning=True near top (pos>0.65) = momentum turning sign at upper zone
    # Per plan: channel surfer generates turning signals at extremes, so near-top
    # + is_turning implies downward turn (rally exhaustion)
    ('G1_5min_top_turning',
     lambda s: _5min_near_top(s) and bool(s.get('5min') and s['5min']['is_turning'])),

    # ── Group 2: Higher TF bullish exhaustion (turning up from sell-off) ─────
    # Hypothesis: medium-term bullish following sell-off exhaustion.

    ('G2_4h_turning_up',
     lambda s: _4h_turning_up(s)),

    ('G2_weekly_turning_up',
     lambda s: _weekly_turning_up(s)),

    ('G2_4h_weekly_turning',
     lambda s: _4h_turning_up(s) and _weekly_turning_up(s)),

    # ── Group 3: Divergence (5min top + higher TF bullish — user's pattern) ──
    # Core question: when 5min is at top but 4h/weekly are bullish, which wins?

    ('G3_5min_top_4h_turning',
     lambda s: _5min_at_top(s) and _4h_turning_up(s)),

    ('G3_5min_top_weekly_turning',
     lambda s: _5min_at_top(s) and _weekly_turning_up(s)),

    # Exact user pattern: 5min at top + both 4h and weekly turning up
    ('G3_5min_top_4h_wkly',
     lambda s: _5min_at_top(s) and _4h_turning_up(s) and _weekly_turning_up(s)),

    # Full user pattern: 5min near_top + 5min is_turning + 4h turning + weekly turning
    ('G3_full_user_pattern',
     lambda s: (_5min_near_top(s) and
                bool(s.get('5min') and s['5min']['is_turning']) and
                _4h_turning_up(s) and _weekly_turning_up(s))),

    # ── Group 4: Consensus / comparison baseline ──────────────────────────────
    # Consensus measures multiple TFs agreeing on direction (bullish = pos > 0.5).

    ('G4_consensus_2_bullish',
     lambda s: _count_bullish_tfs(s) >= 2),

    ('G4_consensus_3_bullish',
     lambda s: _count_bullish_tfs(s) >= 3),

    ('G4_consensus_5_bullish',
     lambda s: _count_bullish_tfs(s) >= 5),

    # Control: 5min at bottom — we know this works, confirms infrastructure
    ('G4_5min_bottom',
     lambda s: bool(s.get('5min') and s['5min']['pos_pct'] < 0.20)),

    # Baseline: no strong signal (all TFs mid-channel)
    ('G4_all_neutral',
     lambda s: _all_neutral(s)),
]


# ── Short-term forward analysis ───────────────────────────────────────────────

def short_forward_analysis(
    daily_df: pd.DataFrame,
    state_rows: list,
    signal_fn,
    signal_name: str,
    start_year: int = 2015,
    end_year: int = 2024,
    max_forward: int = MAX_FORWARD_DAYS,
    capital: float = 100_000.0,
    hold_durations: list | None = None,
) -> dict:
    """
    Forward-return analysis for short-term hold periods.
    Counts all firings independently (no re-entry exclusion) for max sample size.
    Entry at next-day open.
    """
    if hold_durations is None:
        hold_durations = SHORT_HOLD_DURATIONS

    dates  = daily_df.index
    opens  = daily_df['open'].values.astype(float)
    highs  = daily_df['high'].values.astype(float)
    lows   = daily_df['low'].values.astype(float)
    closes = daily_df['close'].values.astype(float)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # Find all signal firing dates (independent — no re-entry exclusion)
    firings = []
    for row in state_rows:
        date = row['date']
        if date.year < start_year or date.year > end_year:
            continue
        states = {tf: row.get(tf) for tf in ['5min', '1h', '4h', 'daily', 'weekly']}
        try:
            fires = bool(signal_fn(states))
        except Exception:
            fires = False
        if fires:
            di = date_to_idx.get(date)
            if di is not None and di + 1 < len(dates):
                firings.append(di + 1)  # entry at next-day open

    if not firings:
        return {'name': signal_name, 'n': 0, 'hold_durations': hold_durations}

    profit_hits  = {t: [] for t in PROFIT_TARGETS}
    stop_hits    = {t: [] for t in STOP_TARGETS}
    profit_rates = {t: 0  for t in PROFIT_TARGETS}
    stop_rates   = {t: 0  for t in STOP_TARGETS}

    # Race pairs — short-term relevant levels
    race_pairs = [(0.03, 0.03), (0.05, 0.05), (0.05, 0.03), (0.10, 0.05)]
    race_results = {p: {'profit_first': 0, 'stop_first': 0, 'neither': 0}
                    for p in race_pairs}

    hold_pnl = {h: [] for h in hold_durations}

    for entry_di in firings:
        entry_price = opens[entry_di]
        if entry_price <= 0:
            continue

        days_to_profit_first = {}
        days_to_stop_first   = {}

        for fwd in range(max_forward):
            bar = entry_di + fwd
            if bar >= len(dates):
                break

            # Entry bar: use close as proxy for intraday high/low
            if fwd == 0:
                day_high = max(entry_price, closes[bar])
                day_low  = min(entry_price, closes[bar])
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

            # hold_pnl: fwd=0 is day 1 (1d hold), fwd=1 is day 2, etc.
            if fwd + 1 in hold_pnl:
                pct = (closes[bar] - entry_price) / entry_price
                hold_pnl[fwd + 1].append(pct * capital)

        for tgt in PROFIT_TARGETS:
            if tgt in days_to_profit_first:
                profit_rates[tgt] += 1
                profit_hits[tgt].append(days_to_profit_first[tgt])

        for tgt in STOP_TARGETS:
            if tgt in days_to_stop_first:
                stop_rates[tgt] += 1
                stop_hits[tgt].append(days_to_stop_first[tgt])

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
        'name':           signal_name,
        'n':              n,
        'profit_rates':   {t: profit_rates[t] / n for t in PROFIT_TARGETS},
        'stop_rates':     {t: stop_rates[t]   / n for t in STOP_TARGETS},
        'profit_days':    {t: profit_hits[t]      for t in PROFIT_TARGETS},
        'stop_days':      {t: stop_hits[t]        for t in STOP_TARGETS},
        'race_results':   race_results,
        'hold_pnl':       hold_pnl,
        'hold_durations': hold_durations,
    }


# ── Phase 1 printing ──────────────────────────────────────────────────────────

def print_phase1_summary(results):
    """Compact summary table for Phase 1, sorted by E[2d]."""
    print(f"\n{'='*120}")
    print("PHASE 1 SUMMARY — SHORT-TERM FORWARD RETURNS (IS 2015-2024, all firings independent)")
    print(f"{'='*120}")
    hdr = (f"{'Signal':<36} {'n':>5}  "
           f"{'hit+3%':>7} {'hit+5%':>7} {'hit+10%':>8}  "
           f"{'hit-3%':>7} {'hit-5%':>7}  "
           f"{'E[1d]':>8} {'E[2d]':>8} {'E[5d]':>8} {'E[10d]':>8} {'E[20d]':>8}  "
           f"{'race+5/-5':>10}")
    print(hdr)
    print('-' * 120)

    def _e2(r):
        if r['n'] == 0:
            return -999_999
        pnls = r.get('hold_pnl', {}).get(2, [])
        return np.mean(pnls) if pnls else -999_999

    ranked = sorted(results, key=_e2, reverse=True)

    for r in ranked:
        if r['n'] == 0:
            print(f"  {r['name']:<36}     0  {'--':>7} {'--':>7} {'--':>8}  "
                  f"{'--':>7} {'--':>7}  {'--':>8} {'--':>8} {'--':>8} {'--':>8} {'--':>8}  {'--':>10}")
            continue

        pr = r.get('profit_rates', {})
        sr = r.get('stop_rates', {})
        hp = r.get('hold_pnl', {})
        rc = r.get('race_results', {}).get((0.05, 0.05), {})

        def _e(h):
            pnls = hp.get(h, [])
            return f"${np.mean(pnls):>7,.0f}" if pnls else '     --'

        race_str = (f"{rc.get('profit_first',0)/r['n']:>3.0%}/"
                    f"{rc.get('stop_first',0)/r['n']:>3.0%}")

        print(f"  {r['name']:<36} {r['n']:>5}  "
              f"{pr.get(0.03,0):>6.0%} {pr.get(0.05,0):>7.0%} {pr.get(0.10,0):>8.0%}  "
              f"{sr.get(0.03,0):>6.0%} {sr.get(0.05,0):>7.0%}  "
              f"{_e(1):>8} {_e(2):>8} {_e(5):>8} {_e(10):>8} {_e(20):>8}  "
              f"{race_str:>10}")

    print(f"\n  Sorted by E[2d].  Race = +5% first / -5% first")
    return ranked


def print_detailed_result(r):
    """Print full hit-rate + expectancy breakdown for one signal."""
    n = r['n']
    if n == 0:
        print(f"  [{r['name']}]  n=0, no firings")
        return

    hd = r.get('hold_durations', SHORT_HOLD_DURATIONS)
    print(f"\n  [{r['name']}]  n={n} firings")

    print(f"\n    PROFIT TARGETS (within {MAX_FORWARD_DAYS}d):")
    print(f"    {'Target':>8}  {'Hit%':>7}  {'AvgDays':>8}  {'MedDays':>8}  Count   Bar")
    print(f"    {'-'*65}")
    for tgt in PROFIT_TARGETS:
        rate  = r['profit_rates'][tgt]
        days  = r['profit_days'][tgt]
        avg_d = f"{np.mean(days):.1f}" if days else '--'
        med_d = f"{np.median(days):.0f}" if days else '--'
        bar   = '#' * int(rate * 20) + '.' * (20 - int(rate * 20))
        print(f"    +{tgt:>5.0%}     {rate:>6.0%}  {avg_d:>8}  {med_d:>8}  {len(days):>5}   {bar}")

    print(f"\n    STOP LEVELS (within {MAX_FORWARD_DAYS}d):")
    print(f"    {'Level':>8}  {'Hit%':>7}  {'AvgDays':>8}  {'MedDays':>8}  Count   Bar")
    print(f"    {'-'*65}")
    for tgt in STOP_TARGETS:
        rate  = r['stop_rates'][tgt]
        days  = r['stop_days'][tgt]
        avg_d = f"{np.mean(days):.1f}" if days else '--'
        med_d = f"{np.median(days):.0f}" if days else '--'
        bar   = '#' * int(rate * 20) + '.' * (20 - int(rate * 20))
        print(f"    -{tgt:>5.0%}     {rate:>6.0%}  {avg_d:>8}  {med_d:>8}  {len(days):>5}   {bar}")

    print(f"\n    RACE — which target hits first?")
    for (pt, st), rc in r['race_results'].items():
        pf = rc['profit_first'] / n
        sf = rc['stop_first']   / n
        ni = rc['neither']      / n
        print(f"    +{pt:.0%} vs -{st:.0%}:  profit first {pf:.0%}  |  "
              f"stop first {sf:.0%}  |  neither in {MAX_FORWARD_DAYS}d {ni:.0%}")

    print(f"\n    EXPECTANCY at hold durations (avg P&L on $100K):")
    print(f"    {'Hold':>6}  {'Avg P&L':>10}  {'Win%':>7}  {'n':>5}")
    print(f"    {'-'*38}")
    for h in hd:
        pnls = r['hold_pnl'].get(h, [])
        if not pnls:
            print(f"    {h:>5}d  {'--':>10}")
            continue
        avg_pnl = np.mean(pnls)
        win_pct = sum(1 for p in pnls if p > 0) / len(pnls)
        print(f"    {h:>5}d  ${avg_pnl:>9,.0f}  {win_pct:>6.0%}  {len(pnls):>5}")


# ── Phase 2: IS/OOS backtest ──────────────────────────────────────────────────

def run_phase2(daily_df, state_rows, capital):
    """
    Phase 2: IS/OOS backtest for all 15 signals.
    hold=2d, stop=5% (tighter for short-term).
    """
    print(f"\n{'='*100}")
    print(f"PHASE 2 — IS/OOS BACKTEST  hold={PHASE2_HOLD}d  stop={PHASE2_STOP:.0%}  capital=${capital:,.0f}")
    print(f"{'='*100}")
    hdr = (f"{'Signal':<36}  "
           f"{'--- IS 2015-2024 ---':>34}  "
           f"{'--- OOS 2025 ---':>34}")
    print(hdr)
    sub = (f"{'':36}  "
           f"{'n':>4}  {'WR':>4}  {'P&L':>10}  {'avg/tr':>8}  stops  "
           f"{'n':>4}  {'WR':>4}  {'P&L':>10}  {'avg/tr':>8}  stops")
    print(sub)
    print('-' * 100)

    def _fmt(r):
        if r['n'] == 0:
            return f"{'--':>4}  {'--':>4}  {'$--':>10}  {'$--':>8}  {'--':>5}"
        return (f"{r['n']:>4}  {r['wr']:>3.0%}  "
                f"${r['pnl']:>9,.0f}  ${r['avg']:>7,.0f}  "
                f"{r.get('stops',0):>5}")

    all_is = []
    for name, fn in SIGNALS_SHORT:
        is_r = run_backtest(
            daily_df, state_rows, fn, name,
            capital=capital, max_hold_days=PHASE2_HOLD,
            stop_pct=PHASE2_STOP,
            start_year=2015, end_year=2024,
        )
        oos_r = run_backtest(
            daily_df, state_rows, fn, name,
            capital=capital, max_hold_days=PHASE2_HOLD,
            stop_pct=PHASE2_STOP,
            start_year=2025, end_year=2025,
        )
        all_is.append((name, is_r, oos_r))
        print(f"  {name:<36}  {_fmt(is_r)}  {_fmt(oos_r)}")

    # Sort by IS P&L and show top signals
    print(f"\n  Top signals by IS P&L:")
    sorted_is = sorted(all_is, key=lambda x: x[1]['pnl'], reverse=True)
    for name, is_r, oos_r in sorted_is[:5]:
        if is_r['n'] > 0:
            ratio = oos_r['pnl'] / is_r['pnl'] if is_r['pnl'] > 0 else 0.0
            print(f"    {name:<36}  IS=${is_r['pnl']:>9,.0f}  OOS=${oos_r['pnl']:>9,.0f}  "
                  f"ratio={ratio:.2f}x")

    return all_is


# ── Phase 3: Walk-forward ─────────────────────────────────────────────────────

def run_phase3(daily_df, state_rows, capital):
    """
    Phase 3: Rolling walk-forward, IS=5yr, OOS=1yr, 6 windows.
    Picks best IS signal each window, applies to OOS.
    """
    windows = [
        (2015, 2019, 2020),
        (2016, 2020, 2021),
        (2017, 2021, 2022),
        (2018, 2022, 2023),
        (2019, 2023, 2024),
        (2020, 2024, 2025),
    ]

    print(f"\n{'='*105}")
    print(f"PHASE 3 — WALK-FORWARD  (IS=5yr, OOS=1yr)  hold={PHASE2_HOLD}d  stop={PHASE2_STOP:.0%}")
    print(f"{'='*105}")
    hdr = (f"{'IS window':>12}  {'OOS':>5}  {'Best IS signal':<36}  "
           f"{'IS P&L':>10}  {'OOS n':>6}  {'OOS P&L':>10}  {'OOS/IS':>7}")
    print(hdr)
    print('-' * 105)

    total_oos = 0.0
    total_is  = 0.0
    wins = 0

    for (is_start, is_end, oos_year) in windows:
        is_results = []
        for name, fn in SIGNALS_SHORT:
            r = run_backtest(
                daily_df, state_rows, fn, name,
                capital=capital, max_hold_days=PHASE2_HOLD,
                stop_pct=PHASE2_STOP,
                start_year=is_start, end_year=is_end,
            )
            is_results.append((name, fn, r))

        best_name, best_fn, best_is = max(is_results, key=lambda x: x[2]['pnl'])

        if best_is['n'] == 0:
            print(f"  {is_start}-{is_end}        {oos_year}   {'(no signals in IS)':>36}")
            continue

        oos_r = run_backtest(
            daily_df, state_rows, best_fn, best_name,
            capital=capital, max_hold_days=PHASE2_HOLD,
            stop_pct=PHASE2_STOP,
            start_year=oos_year, end_year=oos_year,
        )

        ratio = oos_r['pnl'] / best_is['pnl'] if best_is['pnl'] > 0 else 0.0
        total_oos += oos_r['pnl']
        total_is  += best_is['pnl']
        if oos_r['pnl'] > 0:
            wins += 1

        print(f"  {is_start}-{is_end}        {oos_year}   "
              f"{best_name:<36}  "
              f"${best_is['pnl']:>9,.0f}  "
              f"{oos_r['n']:>6}  "
              f"${oos_r['pnl']:>9,.0f}  "
              f"{ratio:>6.2f}x")

    print('-' * 105)
    ratio_total = total_oos / total_is if total_is > 0 else 0.0
    print(f"  {'TOTAL':>17}  {'':>5}  {'':>36}  "
          f"${total_is:>9,.0f}  {'':>6}  "
          f"${total_oos:>9,.0f}  {ratio_total:>6.2f}x")
    print(f"  OOS positive: {wins}/{len(windows)} windows")

    # Also track specific signals across all windows
    print(f"\n  Key signal walk-forward (fixed signals, all 6 windows):")
    key_signals = ['G3_full_user_pattern', 'G3_5min_top_4h_wkly',
                   'G1_5min_top', 'G2_4h_weekly_turning', 'G4_5min_bottom']
    for name, fn in SIGNALS_SHORT:
        if name not in key_signals:
            continue
        print(f"\n  [{name}]")
        sig_wins = 0
        sig_total_oos = 0.0
        for (is_start, is_end, oos_year) in windows:
            is_r = run_backtest(
                daily_df, state_rows, fn, name,
                capital=capital, max_hold_days=PHASE2_HOLD,
                stop_pct=PHASE2_STOP,
                start_year=is_start, end_year=is_end,
            )
            oos_r = run_backtest(
                daily_df, state_rows, fn, name,
                capital=capital, max_hold_days=PHASE2_HOLD,
                stop_pct=PHASE2_STOP,
                start_year=oos_year, end_year=oos_year,
            )
            sig_total_oos += oos_r['pnl']
            if oos_r['pnl'] > 0:
                sig_wins += 1
            ratio = oos_r['pnl'] / is_r['pnl'] if is_r['pnl'] > 0 else 0.0
            print(f"    IS {is_start}-{is_end} OOS {oos_year}:  "
                  f"IS n={is_r['n']:>3} ${is_r['pnl']:>8,.0f}  "
                  f"OOS n={oos_r['n']:>3} ${oos_r['pnl']:>8,.0f}  "
                  f"ratio={ratio:>5.2f}x")
        print(f"    Total OOS ${sig_total_oos:>9,.0f}  wins {sig_wins}/6")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Short-term TF state forward-return analysis (1d-20d)')
    parser.add_argument('--tsla', type=str, default='data/TSLAMin.txt',
                        help='Path to 1-min TSLA data')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end',   type=str, default='2025-12-31')
    parser.add_argument('--capital', type=float, default=100_000.0)
    parser.add_argument('--skip-phase2', action='store_true', dest='skip_phase2',
                        help='Skip Phase 2 IS/OOS backtest')
    parser.add_argument('--skip-phase3', action='store_true', dest='skip_phase3',
                        help='Skip Phase 3 walk-forward')
    parser.add_argument('--detail', action='store_true',
                        help='Print full hit-rate detail for all signals (default: top 5 only)')
    args = parser.parse_args()

    print(f"\n{'='*75}")
    print("SHORT-TERM TF STATE FORWARD-RETURN ANALYSIS")
    print(f"Hold periods: {SHORT_HOLD_DURATIONS} days  |  Max forward: {MAX_FORWARD_DAYS}d")
    print(f"Phase 2: hold={PHASE2_HOLD}d  stop={PHASE2_STOP:.0%}  capital=${args.capital:,.0f}")
    print(f"{'='*75}")

    # Load data
    tf_data = load_all_tfs(args.tsla, args.start, args.end)
    daily_df = tf_data['daily']
    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates)

    # ── Phase 1: Forward return analysis (IS 2015-2024) ──────────────────────
    print(f"\n{'='*75}")
    print("PHASE 1 — FORWARD RETURN ANALYSIS (IS 2015-2024)")
    print(f"{'='*75}")
    print(f"Running {len(SIGNALS_SHORT)} signals, tracking {MAX_FORWARD_DAYS} days forward...")
    print(f"(All firings independent — no re-entry exclusion)\n")

    t0 = time.time()
    phase1_results = []
    for name, fn in SIGNALS_SHORT:
        r = short_forward_analysis(
            daily_df, state_rows, fn, name,
            start_year=2015, end_year=2024,
            max_forward=MAX_FORWARD_DAYS,
            capital=args.capital,
        )
        phase1_results.append(r)
        if r['n'] > 0:
            e2 = np.mean(r['hold_pnl'].get(2, [0])) if r['hold_pnl'].get(2) else 0
            e5 = np.mean(r['hold_pnl'].get(5, [0])) if r['hold_pnl'].get(5) else 0
            hit5  = r['profit_rates'].get(0.05, 0)
            stop3 = r['stop_rates'].get(0.03, 0)
            print(f"  {name:<36}  n={r['n']:>4}  +5%:{hit5:>4.0%}  -3%:{stop3:>4.0%}  "
                  f"E[2d]=${e2:>7,.0f}  E[5d]=${e5:>7,.0f}")
        else:
            print(f"  {name:<36}  n=   0  no firings")

    print(f"\nDone in {time.time()-t0:.0f}s")

    # Summary table (sorted by E[2d])
    ranked = print_phase1_summary(phase1_results)

    # Detailed breakdown
    if args.detail:
        print(f"\n{'='*75}")
        print("DETAILED BREAKDOWN (all signals, n >= 3)")
        print(f"{'='*75}")
        for r in ranked:
            if r['n'] >= 3:
                print_detailed_result(r)
    else:
        # Show top 5 by E[2d] + always show the user-specific divergence signals + control
        top5_names = {r['name'] for r in ranked if r['n'] >= 5}
        always_show = {'G3_full_user_pattern', 'G3_5min_top_4h_wkly',
                       'G4_5min_bottom', 'G4_all_neutral', 'G1_5min_top'}
        to_show = top5_names | always_show
        print(f"\n{'='*75}")
        print("DETAILED BREAKDOWN (top 5 by E[2d] + key divergence signals)")
        print(f"{'='*75}")
        shown = set()
        for r in ranked:
            if r['name'] in to_show and r['n'] >= 3 and r['name'] not in shown:
                print_detailed_result(r)
                shown.add(r['name'])

    # ── Phase 2: IS/OOS backtest ─────────────────────────────────────────────
    if not args.skip_phase2:
        run_phase2(daily_df, state_rows, args.capital)

    # ── Phase 3: Walk-forward ────────────────────────────────────────────────
    if not args.skip_phase3:
        run_phase3(daily_df, state_rows, args.capital)

    # ── Interpretation guide ──────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*75}")
    print("  G4_5min_bottom   → control: buy dip. E[2d] > 0 confirms infrastructure works.")
    print("  G1_5min_top      → 5min at channel top. E[2d] < 0 = short-term mean-reversion.")
    print("  G2_4h_weekly_turning → higher TF turning bull. Short-term bullish edge?")
    print("  G3_5min_top_4h_wkly  → exact user pattern (5min top + 4h/wkly bullish).")
    print("  G3_full_user_pattern → same + 5min is_turning near top.")
    print()
    print("  Positive E[2d] for G3 signals → 4h/weekly direction dominates short-term.")
    print("  Negative E[2d]  → 5min resistance wins, expect pullback before rally.")
    print("  Near-zero E[2d] → mixed/no short-term edge; check E[5d] for 1-week view.")

    print(f"\n{'='*75}")
    print("DONE")
    print(f"{'='*75}")


if __name__ == '__main__':
    main()
