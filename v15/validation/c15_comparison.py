#!/usr/bin/env python3
"""
c15 Comparison: Backtester vs Live Scanner Alignment

Runs each backtester engine in 2 modes (full, full+AM-only) on the same
2025-01-01 to 2026-03-04 window, then prints a comparison table.

Scenarios:
  1. Full backtester      — each signal's proper engine, no AM block
  2. Full backtester + AM — same as #1 but entries restricted to 9:30-10:30 ET
  (Live scanner scenario is noted manually from forward_sim_v2 output)

Usage:
    python3 -m v15.validation.c15_comparison --tsla data/TSLAMin.txt
    python3 -m v15.validation.c15_comparison --tsla data/TSLAMin.txt --start 2025-01-01 --end 2026-03-04
    python3 -m v15.validation.c15_comparison --surfer-only --tsla data/TSLAMin.txt
    python3 -m v15.validation.c15_comparison --combo-only --tsla data/TSLAMin.txt
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class ScenarioResult:
    """Result from one backtester scenario."""
    signal_type: str
    scenario: str
    trades: int
    wins: int
    total_pnl: float
    max_dd_pct: float

    @property
    def wr_pct(self) -> float:
        return self.wins / max(self.trades, 1) * 100


def run_combo_scenarios(tsla_path: Optional[str], start: str, end: str) -> List[ScenarioResult]:
    """Run combo backtest for CS-5TF and champion DW combo."""
    from v15.validation.combo_backtest import (
        phase1_precompute, simulate_trades, _make_cs_tf5_combo,
        _build_filter_cascade, CAPITAL,
    )

    print("\n" + "=" * 70)
    print("  COMBO BACKTEST (CS-5TF / CS-DW)")
    print("=" * 70)

    # Phase 1: precompute signals
    print("\nPhase 1: Computing signals...")
    t0 = time.time()
    signals, daily_df, spy_daily, vix_daily, weekly_tsla = phase1_precompute(
        tsla_path, start, end)
    print(f"  Phase 1 done in {time.time() - t0:.1f}s ({len(signals)} days)")

    results = []

    # CS-5TF combo (W: TF5)
    print("\nRunning CS-5TF (full)...")
    tf5_fn = _make_cs_tf5_combo()
    tf5_trades = simulate_trades(signals, tf5_fn, 'CS-5TF')
    tf5_r = _combo_result('CS-5TF', 'Full backtester', tf5_trades, CAPITAL)
    results.append(tf5_r)
    print(f"  {tf5_r.trades} trades, {tf5_r.wr_pct:.1f}% WR, ${tf5_r.total_pnl:+,.0f}")

    # CS-5TF: AM-only is same as full (daily bar entries at open = AM)
    print("  (AM-only = same as full for daily-bar combo)")
    results.append(ScenarioResult(
        signal_type='CS-5TF', scenario='Full + AM rule',
        trades=tf5_r.trades, wins=tf5_r.wins,
        total_pnl=tf5_r.total_pnl, max_dd_pct=tf5_r.max_dd_pct,
    ))

    # CS-DW champion combo (DI: v36 with trail^12)
    # Need the extra precomputed data
    print("\nPre-computing filter/SPY/VIX data for DW combo...")
    cascade_vix = _build_filter_cascade(vix=True)
    cascade_vix.precompute_vix_cooldown(vix_daily)

    spy_above_sma20 = set()
    spy_above_055pct = set()
    spy_dist_map = {}
    spy_dist_5 = {}
    spy_dist_50 = {}
    vix_map = {}
    spy_return_map = {}
    spy_ret_2d = {}

    if spy_daily is not None and len(spy_daily) > 20:
        spy_close = spy_daily['close'].values.astype(float)
        spy_sma20 = pd.Series(spy_close).rolling(20).mean().values
        for i in range(20, len(spy_close)):
            if spy_sma20[i] > 0:
                dist_pct = (spy_close[i] - spy_sma20[i]) / spy_sma20[i] * 100
                if spy_close[i] > spy_sma20[i]:
                    spy_above_sma20.add(spy_daily.index[i])
                if dist_pct >= 0.55:
                    spy_above_055pct.add(spy_daily.index[i])
                spy_dist_map[spy_daily.index[i]] = dist_pct

        for win, dist_dict in [(5, spy_dist_5), (50, spy_dist_50)]:
            if len(spy_daily) > win:
                sma = pd.Series(spy_close).rolling(win).mean().values
                for i in range(win, len(spy_close)):
                    if sma[i] > 0:
                        dist_dict[spy_daily.index[i]] = (spy_close[i] - sma[i]) / sma[i] * 100

    if vix_daily is not None:
        for idx, row in vix_daily.iterrows():
            vix_map[idx] = row['close']

    if spy_daily is not None:
        spy_close_arr = spy_daily['close'].values.astype(float)
        for i in range(1, len(spy_close_arr)):
            spy_return_map[spy_daily.index[i]] = (spy_close_arr[i] - spy_close_arr[i-1]) / spy_close_arr[i-1] * 100
        for i in range(2, len(spy_close_arr)):
            spy_ret_2d[spy_daily.index[i]] = (spy_close_arr[i] - spy_close_arr[i-2]) / spy_close_arr[i-2] * 100

    # Import the latest DW combo maker
    try:
        from v15.validation.combo_backtest import _make_v36_di
        dw_fn = _make_v36_di(cascade_vix, spy_above_sma20, spy_above_055pct,
                             spy_dist_map, spy_dist_5, spy_dist_50,
                             vix_map, spy_return_map, spy_ret_2d)
        dw_name = 'CS-DW (v36)'
    except ImportError:
        # Fallback to v35
        from v15.validation.combo_backtest import _make_v35_dg
        dw_fn = _make_v35_dg(cascade_vix, spy_above_sma20, spy_above_055pct,
                             spy_dist_map, spy_dist_5, spy_dist_50,
                             vix_map, spy_return_map, spy_ret_2d)
        dw_name = 'CS-DW (v35)'

    print(f"\nRunning {dw_name} (full, trail^12)...")
    dw_trades = simulate_trades(signals, dw_fn, dw_name, cooldown=0, trail_power=12)
    dw_r = _combo_result(dw_name, 'Full backtester', dw_trades, CAPITAL)
    results.append(dw_r)
    print(f"  {dw_r.trades} trades, {dw_r.wr_pct:.1f}% WR, ${dw_r.total_pnl:+,.0f}")

    # DW + AM: same reasoning — daily bars, entries at open
    results.append(ScenarioResult(
        signal_type=dw_name, scenario='Full + AM rule',
        trades=dw_r.trades, wins=dw_r.wins,
        total_pnl=dw_r.total_pnl, max_dd_pct=dw_r.max_dd_pct,
    ))

    return results


def _filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to regular trading hours (9:30-16:00 ET).

    Timestamps are assumed UTC (no tz). Regular hours in UTC:
    EDT (Mar-Nov): 13:30-20:00, EST (Nov-Mar): 14:30-21:00.
    We approximate with ET = (UTC hour - 5) % 24.
    """
    pre = len(df)
    et_hour = (df.index.hour - 5) % 24
    et_min = df.index.minute
    mask = ((et_hour == 9) & (et_min >= 30)) | ((et_hour >= 10) & (et_hour <= 15))
    df = df[mask]
    print(f"  RTH filter: {pre:,} → {len(df):,} bars "
          f"({len(df)/max(pre,1)*100:.0f}% kept)", flush=True)
    return df


def run_surfer_scenarios(tsla_path: Optional[str], start: str, end: str) -> List[ScenarioResult]:
    """Run surfer ML backtest on 5-min RTH data (full and AM-only).

    Uses 5-min bars (not 1-min) because the channel detection windows
    [10,15,20,30,40] need ~50-200 min of context. eval_interval=3
    (every 15 min) matches the backtester's tuned trail/exit logic.
    """
    from v15.validation.tf_state_backtest import load_all_tfs
    from v15.data.native_tf import fetch_native_tf
    from v15.core.surfer_backtest import run_backtest

    print("\n" + "=" * 70, flush=True)
    print("  SURFER ML BACKTEST (5-min bars, RTH only, eval every bar)", flush=True)
    print("=" * 70, flush=True)

    # Load all TFs, then filter 5-min to RTH
    print("\nLoading data...", flush=True)
    t0 = time.time()
    tf_data = load_all_tfs(tsla_path, start, end)
    tsla_5m_rth = _filter_rth(tf_data['5min'])
    higher_tf_dict = {k: v for k, v in tf_data.items() if k != '5min'}

    spy_df = fetch_native_tf('SPY', 'daily', start, end)
    vix_df = fetch_native_tf('^VIX', 'daily', start, end)
    print(f"  Load time: {time.time() - t0:.1f}s", flush=True)
    print(f"  Base: {len(tsla_5m_rth):,} 5-min RTH bars "
          f"({tsla_5m_rth.index[0]} to {tsla_5m_rth.index[-1]})", flush=True)

    # Load ML model
    ml_model = None
    gbt_path = os.path.join('surfer_models', 'gbt_model.pkl')
    if os.path.exists(gbt_path):
        from v15.core.surfer_ml import GBTModel
        ml_model = GBTModel.load(gbt_path)
        print(f"  ML model loaded: {type(ml_model).__name__}", flush=True)
    else:
        print(f"  WARNING: No ML model at {gbt_path}", flush=True)

    # Default sizing: $10K base, initial equity $100K, all arch boosts active
    # No max_trade_usd cap — let TOD/DOW/VIX multipliers work as designed
    sizing_kwargs = dict()

    # Suppress verbose backtester output
    _quiet = type('Q', (), {'write': lambda s, *a: None, 'flush': lambda s: None})()

    results = []

    # Full backtester (no AM restriction)
    print("\nRunning Surfer ML (full)...", flush=True)
    t0 = time.time()
    old_stdout = sys.stdout
    sys.stdout = _quiet
    try:
        m_full, trades_full, _ = run_backtest(
            tsla_df=tsla_5m_rth,
            higher_tf_dict=higher_tf_dict,
            spy_df_input=spy_df,
            vix_df_input=vix_df,
            ml_model=ml_model,
            eval_interval=3,  # every 3 bars = 15 min (backtester's tuned default)
            am_only=False,
            **sizing_kwargs,
        )
    finally:
        sys.stdout = old_stdout
    elapsed = time.time() - t0
    # Compute flat-$100K P&L from per-trade pnl_pct (ignores compounding)
    flat_pnl_full = sum(t.pnl_pct * 100_000 for t in trades_full)
    print(f"  Done in {elapsed:.0f}s: {m_full.total_trades} trades, "
          f"WR={m_full.win_rate:.1%}, Flat P&L=${flat_pnl_full:+,.0f}", flush=True)
    results.append(ScenarioResult(
        signal_type='Surfer ML', scenario='Full backtester',
        trades=m_full.total_trades, wins=m_full.wins,
        total_pnl=flat_pnl_full, max_dd_pct=m_full.max_drawdown_pct,
    ))

    # Full + AM only
    print("\nRunning Surfer ML (AM-only)...", flush=True)
    t0 = time.time()
    sys.stdout = _quiet
    try:
        m_am, trades_am, _ = run_backtest(
            tsla_df=tsla_5m_rth,
            higher_tf_dict=higher_tf_dict,
            spy_df_input=spy_df,
            vix_df_input=vix_df,
            ml_model=ml_model,
            eval_interval=1,
            am_only=True,
            **sizing_kwargs,
        )
    finally:
        sys.stdout = old_stdout
    elapsed = time.time() - t0
    flat_pnl_am = sum(t.pnl_pct * 100_000 for t in trades_am)
    print(f"  Done in {elapsed:.0f}s: {m_am.total_trades} trades, "
          f"WR={m_am.win_rate:.1%}, Flat P&L=${flat_pnl_am:+,.0f}", flush=True)
    results.append(ScenarioResult(
        signal_type='Surfer ML', scenario='Full + AM rule',
        trades=m_am.total_trades, wins=m_am.wins,
        total_pnl=flat_pnl_am, max_dd_pct=m_am.max_drawdown_pct,
    ))

    # Entry hour distribution
    from collections import Counter
    for label, trades in [('FULL', trades_full), ('AM-ONLY', trades_am)]:
        hours = Counter()
        for t in trades:
            if t.entry_time:
                ts = pd.Timestamp(t.entry_time)
                hours[(ts.hour - 5) % 24] += 1
        if hours:
            print(f"\n  {label} entry hours (ET):", flush=True)
            for h in sorted(hours):
                print(f"    {h:2d}:00  {hours[h]:4d} ({hours[h]/len(trades)*100:4.1f}%)",
                      flush=True)

    return results


def _combo_result(signal_type: str, scenario: str,
                  trades, capital: float) -> ScenarioResult:
    """Build ScenarioResult from combo Trade list."""
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    total_pnl = sum(t.pnl for t in trades)
    if n > 0:
        pnls = np.array([t.pnl for t in trades])
        cum = np.cumsum(pnls)
        dd = np.maximum.accumulate(cum) - cum
        max_dd_pct = float(dd.max()) / capital * 100 if capital > 0 else 0.0
    else:
        max_dd_pct = 0.0
    return ScenarioResult(
        signal_type=signal_type, scenario=scenario,
        trades=n, wins=wins, total_pnl=total_pnl, max_dd_pct=max_dd_pct,
    )


def print_comparison(results: List[ScenarioResult]):
    """Print final comparison table."""
    print("\n" + "=" * 80)
    print("  c15 ALIGNMENT COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Signal Type':<18} {'Scenario':<22} {'Trades':>7} {'WR%':>7} "
          f"{'P&L':>12} {'MaxDD%':>8}")
    print("-" * 80)

    for r in results:
        print(f"{r.signal_type:<18} {r.scenario:<22} {r.trades:>7} "
              f"{r.wr_pct:>6.1f}% ${r.total_pnl:>+10,.0f} {r.max_dd_pct:>7.1f}%")

    print("-" * 80)
    print("\nNotes:")
    print("  - Combo entries are at daily open (inherently AM) → AM rule has no effect")
    print("  - Surfer ML AM rule: entries restricted to 9:30-10:30 ET only")
    print("  - 'Live scanner config' results come from forward_sim_v2 (run separately)")


def main():
    parser = argparse.ArgumentParser(description='c15: Backtester vs Live Scanner Alignment')
    parser.add_argument('--tsla', type=str, default=None,
                        help='Path to TSLAMin.txt (1-min data)')
    parser.add_argument('--start', type=str, default='2025-01-01')
    parser.add_argument('--end', type=str, default='2026-03-04')
    parser.add_argument('--combo-only', action='store_true',
                        help='Only run combo backtest scenarios')
    parser.add_argument('--surfer-only', action='store_true',
                        help='Only run surfer ML backtest scenarios')
    args = parser.parse_args()

    # Auto-detect TSLAMin.txt
    if args.tsla is None:
        for candidate in ['data/TSLAMin.txt', '../data/TSLAMin.txt',
                          'C:/AI/x14/data/TSLAMin.txt',
                          os.path.expanduser('~/data/TSLAMin.txt')]:
            if os.path.isfile(candidate):
                args.tsla = candidate
                break

    print(f"c15 Alignment Comparison -- {args.start} to {args.end}")
    print(f"TSLAMin: {args.tsla or 'not found'}")

    all_results = []

    if not args.surfer_only:
        combo_results = run_combo_scenarios(args.tsla, args.start, args.end)
        all_results.extend(combo_results)

    if not args.combo_only:
        surfer_results = run_surfer_scenarios(args.tsla, args.start, args.end)
        all_results.extend(surfer_results)

    print_comparison(all_results)


if __name__ == '__main__':
    main()
