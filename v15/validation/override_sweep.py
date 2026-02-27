#!/usr/bin/env python3
"""
Override Sweep: Grid search for higher-TF override + exit config.

Cache-and-replay architecture: runs market analysis ONCE, caches all
ChannelAnalysis objects per bar, then replays SimScanner trading logic
with different override/exit configs applied to cached analyses.

Phase 1: Override sweep (48 combos, baseline exits)
Phase 2: Exit sweep (36 combos, best override from Phase 1)

Usage:
    python -m v15.validation.override_sweep
    python -m v15.validation.override_sweep --tf 1h --start 2026-02-10 --end 2026-02-26
    python -m v15.validation.override_sweep --phase 2  # exit sweep only (uses best override)
"""
import argparse
import copy
import itertools
import os
import sys
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.core.channel_surfer import (
    ChannelAnalysis,
    OverrideConfig,
    apply_higher_tf_override,
    prepare_multi_tf_analysis,
)
from v15.validation.forward_sim_v2 import (
    SimClosedTrade,
    SimScanner,
    TF_SIM_CONFIG,
    build_native_data,
    download_data,
    slice_native_data,
)


# ---------------------------------------------------------------------------
# Phase 1: Override grid
# ---------------------------------------------------------------------------

OVERRIDE_GRID = {
    'mode': ['none', 'suppress', 'flip', 'boost_only'],
    'position_threshold': [0.20, 0.25, 0.30],
    'required_agreement': [1, 2],
    'override_tfs': [['daily', 'weekly'], ['4h', 'daily', 'weekly']],
    'require_stabilizing': [False, True],
}

# ---------------------------------------------------------------------------
# Phase 2: Exit grid
# ---------------------------------------------------------------------------

EXIT_GRID = {
    'trail_enabled': [True, False],
    'stop_multiplier': [1.0, 2.0, 3.0],
    'eod_close_enabled': [True, False],
    'timeout_multiplier': [1.0, 2.0, 5.0],
}


# ---------------------------------------------------------------------------
# Cached analysis storage
# ---------------------------------------------------------------------------

@dataclass
class CachedBar:
    """Cached per-bar data for replay."""
    bar_time: pd.Timestamp
    price: float
    high: float
    low: float
    analysis: Optional[ChannelAnalysis]  # None if signal generation failed


# ---------------------------------------------------------------------------
# Phase 1: Cache market analysis (run once)
# ---------------------------------------------------------------------------

def cache_analyses(data: dict, native_data: dict, sim_start: str, sim_end: str,
                   tf_system: str, verbose: bool = True) -> List[CachedBar]:
    """Walk bars for a TF system, generate and cache ChannelAnalysis per bar.

    Returns list of CachedBar objects for the FULL bar range (not just sim window),
    with analysis=None for bars outside the sim window.
    """
    config = TF_SIM_CONFIG[tf_system]
    walk_bars = data[config['data_key']]
    target_tfs = config['target_tfs']
    native_tf = config['native_tf']

    total_bars = len(walk_bars)
    cached: List[CachedBar] = []

    # Build sim window timestamps
    sim_start_ts = pd.Timestamp(sim_start)
    sim_end_ts = pd.Timestamp(sim_end) + pd.Timedelta(hours=23, minutes=59)
    walk_tz = walk_bars.index.tz
    if walk_tz is not None:
        if sim_start_ts.tzinfo is None:
            sim_start_ts = sim_start_ts.tz_localize(walk_tz)
        if sim_end_ts.tzinfo is None:
            sim_end_ts = sim_end_ts.tz_localize(walk_tz)
    else:
        if sim_start_ts.tzinfo is not None:
            sim_start_ts = sim_start_ts.tz_localize(None)
        if sim_end_ts.tzinfo is not None:
            sim_end_ts = sim_end_ts.tz_localize(None)

    if verbose:
        print(f"\n  Caching analyses for {tf_system} system ({total_bars:,} bars)...")

    t_start = time.time()
    last_progress = -10
    analyses_cached = 0

    spy_walk = data[config['spy_data_key']]

    for bar_idx in range(total_bars):
        bar = walk_bars.iloc[bar_idx]
        bar_time = walk_bars.index[bar_idx]
        price = float(bar['close'])
        high = float(bar['high'])
        low = float(bar['low'])

        # Progress
        if verbose:
            pct = int(bar_idx / total_bars * 100)
            if pct >= last_progress + 10:
                elapsed = time.time() - t_start
                rate = bar_idx / elapsed if elapsed > 0 else 0
                eta = (total_bars - bar_idx) / rate if rate > 0 else 0
                print(f"    {pct}% ({bar_idx:,}/{total_bars:,} bars, "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
                last_progress = pct

        # Only cache analyses during sim window (but store all bars for exits)
        analysis = None
        if sim_start_ts <= bar_time <= sim_end_ts:
            # Slice native data to prevent look-ahead
            sliced = slice_native_data(native_data, bar_time)
            sliced['TSLA'][native_tf] = walk_bars.iloc[:bar_idx + 1]

            # SPY
            bt_spy = bar_time
            if spy_walk.index.tz is not None and bt_spy.tzinfo is None:
                bt_spy = bt_spy.tz_localize(spy_walk.index.tz)
            elif spy_walk.index.tz is None and bt_spy.tzinfo is not None:
                bt_spy = bt_spy.tz_localize(None)
            sliced['SPY'][native_tf] = spy_walk[spy_walk.index <= bt_spy]

            # 5min data for 5min system
            live_5min = None
            if tf_system == '5min':
                live_5min = data['tsla_5min']
                ct_5m = bar_time
                if live_5min.index.tz is not None and ct_5m.tzinfo is None:
                    ct_5m = ct_5m.tz_localize(live_5min.index.tz)
                elif live_5min.index.tz is None and ct_5m.tzinfo is not None:
                    ct_5m = ct_5m.tz_localize(None)
                live_5min = live_5min[live_5min.index <= ct_5m]

            try:
                # Generate signal WITHOUT override (raw signal)
                analysis = prepare_multi_tf_analysis(
                    native_data=sliced,
                    live_5min_tsla=live_5min,
                    target_tfs=target_tfs,
                    override_config=None,  # No override — we apply later during replay
                )
                analyses_cached += 1
            except Exception:
                analysis = None

        cached.append(CachedBar(
            bar_time=bar_time,
            price=price,
            high=high,
            low=low,
            analysis=analysis,
        ))

    elapsed = time.time() - t_start
    if verbose:
        print(f"  Cached {analyses_cached:,} analyses in {elapsed:.1f}s")

    return cached


# ---------------------------------------------------------------------------
# Replay with override config
# ---------------------------------------------------------------------------

def replay_with_config(cached_bars: List[CachedBar],
                       override_config: Optional[OverrideConfig],
                       scanner: SimScanner,
                       tf_system: str) -> SimScanner:
    """Replay cached analyses through SimScanner with a specific override config.

    Mutates and returns the scanner.
    """
    config = TF_SIM_CONFIG[tf_system]
    scanner.TIMEOUT_MINUTES = config['timeout_minutes']

    for cb in cached_bars:
        # 1. Check exits on every bar
        scanner.check_exits(cb.price, cb.high, cb.low, cb.bar_time)

        # 2. Only evaluate signals where we have a cached analysis
        if cb.analysis is None:
            continue

        # Apply override to cached (raw) analysis
        analysis = cb.analysis
        if override_config is not None and override_config.mode != 'none':
            # Re-apply override to the raw signal
            modified_signal, override_info = apply_higher_tf_override(
                analysis.signal, analysis.tf_states, override_config,
            )
            analysis = replace(
                analysis,
                signal=modified_signal,
                override_info=override_info,
            )

        scanner.evaluate_signal(analysis, cb.price, cb.bar_time)

    # Force-close remaining
    if cached_bars and scanner.positions:
        last = cached_bars[-1]
        scanner.force_close_all(last.price, last.bar_time)

    return scanner


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(trades: List[SimClosedTrade], initial_capital: float) -> dict:
    """Compute summary metrics for a set of trades."""
    n = len(trades)
    if n == 0:
        return {
            'trades': 0, 'wins': 0, 'wr': 0.0, 'pf': 0.0,
            'pnl': 0.0, 'avg_hold': 0.0, 'overrides': 0,
        }

    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    win_pnl = sum(t.pnl for t in trades if t.pnl > 0)
    loss_pnl = sum(t.pnl for t in trades if t.pnl <= 0)
    pf = abs(win_pnl / loss_pnl) if loss_pnl != 0 else float('inf')
    total_pnl = sum(t.pnl for t in trades)
    avg_hold = np.mean([t.hold_minutes for t in trades])
    overrides = sum(1 for t in trades if t.override_applied)

    return {
        'trades': n, 'wins': wins, 'wr': wr, 'pf': pf,
        'pnl': total_pnl, 'avg_hold': avg_hold, 'overrides': overrides,
    }


# ---------------------------------------------------------------------------
# Phase 1: Override sweep
# ---------------------------------------------------------------------------

def run_override_sweep(cached_bars: List[CachedBar], tf_system: str,
                       initial_capital: float = 100_000,
                       verbose: bool = True) -> List[dict]:
    """Sweep override configs with baseline exit settings."""
    results = []

    # Generate all combos
    combos = []
    for mode in OVERRIDE_GRID['mode']:
        if mode == 'none':
            combos.append({'mode': 'none', 'position_threshold': 0,
                           'required_agreement': 0, 'override_tfs': [],
                           'require_stabilizing': False})
            continue
        for thresh in OVERRIDE_GRID['position_threshold']:
            for agree in OVERRIDE_GRID['required_agreement']:
                for tfs in OVERRIDE_GRID['override_tfs']:
                    for stab in OVERRIDE_GRID['require_stabilizing']:
                        combos.append({
                            'mode': mode,
                            'position_threshold': thresh,
                            'required_agreement': agree,
                            'override_tfs': tfs,
                            'require_stabilizing': stab,
                        })

    if verbose:
        print(f"\n  Phase 1: Override sweep ({len(combos)} combos)...")

    t_start = time.time()

    for i, combo in enumerate(combos):
        if combo['mode'] == 'none':
            oc = None
        else:
            oc = OverrideConfig(
                mode=combo['mode'],
                position_threshold=combo['position_threshold'],
                required_agreement=combo['required_agreement'],
                override_tfs=combo['override_tfs'],
                require_stabilizing=combo.get('require_stabilizing', False),
            )

        scanner = SimScanner(initial_capital)
        replay_with_config(cached_bars, oc, scanner, tf_system)

        metrics = compute_metrics(scanner.closed_trades, initial_capital)
        metrics['combo'] = combo
        results.append(metrics)

    elapsed = time.time() - t_start
    if verbose:
        print(f"  Completed {len(combos)} combos in {elapsed:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Phase 2: Exit sweep
# ---------------------------------------------------------------------------

def run_exit_sweep(cached_bars: List[CachedBar], tf_system: str,
                   best_override: Optional[OverrideConfig],
                   initial_capital: float = 100_000,
                   verbose: bool = True) -> List[dict]:
    """Sweep exit configs with the best override from Phase 1."""
    results = []

    combos = []
    for trail in EXIT_GRID['trail_enabled']:
        for stop_m in EXIT_GRID['stop_multiplier']:
            for eod in EXIT_GRID['eod_close_enabled']:
                for timeout_m in EXIT_GRID['timeout_multiplier']:
                    combos.append({
                        'trail_enabled': trail,
                        'stop_multiplier': stop_m,
                        'eod_close_enabled': eod,
                        'timeout_multiplier': timeout_m,
                    })

    if verbose:
        print(f"\n  Phase 2: Exit sweep ({len(combos)} combos)...")

    t_start = time.time()

    for combo in combos:
        scanner = SimScanner(
            initial_capital,
            trail_enabled=combo['trail_enabled'],
            stop_multiplier=combo['stop_multiplier'],
            eod_close_enabled=combo['eod_close_enabled'],
            timeout_multiplier=combo['timeout_multiplier'],
        )
        replay_with_config(cached_bars, best_override, scanner, tf_system)

        metrics = compute_metrics(scanner.closed_trades, initial_capital)
        metrics['combo'] = combo
        results.append(metrics)

    elapsed = time.time() - t_start
    if verbose:
        print(f"  Completed {len(combos)} combos in {elapsed:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_override_results(results: List[dict], tf_system: str,
                           sim_start: str, sim_end: str):
    """Print Phase 1 results table."""
    print(f"\n{'='*110}")
    print(f"PHASE 1: OVERRIDE SWEEP ({tf_system} system, {sim_start} to {sim_end})")
    print(f"{'='*110}")
    print(f"  {'Mode':<12} {'Thresh':>6} {'Agree':>6} {'TFs':<18} {'Stab':>5} "
          f"{'Trades':>7} {'WR':>6} {'PF':>6} {'P&L':>10} {'Overrides':>9}")
    print(f"  {'-'*95}")

    # Sort by P&L descending
    sorted_results = sorted(results, key=lambda r: r['pnl'], reverse=True)
    best_pnl = sorted_results[0]['pnl'] if sorted_results else 0

    for r in sorted_results:
        c = r['combo']
        mode = c['mode']
        thresh = f"{c['position_threshold']:.2f}" if mode != 'none' else '-'
        agree = str(c['required_agreement']) if mode != 'none' else '-'
        tfs = ','.join(c['override_tfs']) if mode != 'none' and c['override_tfs'] else '-'
        stab = 'yes' if c.get('require_stabilizing') else 'no'
        if mode == 'none':
            stab = '-'
        star = ' *' if r['pnl'] == best_pnl and r['pnl'] > 0 else ''
        pf_str = f"{r['pf']:.2f}" if r['pf'] < 100 else 'inf'

        print(f"  {mode:<12} {thresh:>6} {agree:>6} {tfs:<18} {stab:>5} "
              f"{r['trades']:>7} {r['wr']:>5.0f}% {pf_str:>6} "
              f"${r['pnl']:>+9,.0f} {r['overrides']:>9}{star}")


def print_exit_results(results: List[dict], tf_system: str,
                       sim_start: str, sim_end: str):
    """Print Phase 2 results table."""
    print(f"\n{'='*100}")
    print(f"PHASE 2: EXIT SWEEP ({tf_system} system, {sim_start} to {sim_end})")
    print(f"{'='*100}")
    print(f"  {'Trail':<7} {'StopX':>6} {'EOD':<6} {'TimeX':>7} "
          f"{'Trades':>7} {'WR':>6} {'PF':>6} {'P&L':>10} {'AvgHold':>10}")
    print(f"  {'-'*70}")

    sorted_results = sorted(results, key=lambda r: r['pnl'], reverse=True)
    best_pnl = sorted_results[0]['pnl'] if sorted_results else 0

    for r in sorted_results:
        c = r['combo']
        trail = 'yes' if c['trail_enabled'] else 'no'
        eod = 'yes' if c['eod_close_enabled'] else 'no'
        pf_str = f"{r['pf']:.2f}" if r['pf'] < 100 else 'inf'
        hold_hrs = r['avg_hold'] / 60 if r['avg_hold'] > 0 else 0
        star = ' *' if r['pnl'] == best_pnl and r['pnl'] > 0 else ''

        print(f"  {trail:<7} {c['stop_multiplier']:>5.1f}x {eod:<6} "
              f"{c['timeout_multiplier']:>5.1f}x "
              f"{r['trades']:>7} {r['wr']:>5.0f}% {pf_str:>6} "
              f"${r['pnl']:>+9,.0f} {hold_hrs:>8.1f}h{star}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Override + Exit Sweep')
    parser.add_argument('--start', default='2026-02-10',
                        help='Sim start date (default: 2026-02-10)')
    parser.add_argument('--end', default='2026-02-26',
                        help='Sim end date (default: 2026-02-26)')
    parser.add_argument('--capital', type=float, default=100_000,
                        help='Initial capital (default: 100000)')
    parser.add_argument('--tf', nargs='+', default=['1h', '5min'],
                        help='TF systems to sweep (default: 1h 5min)')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=None,
                        help='Run only Phase 1 or 2 (default: both)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    args = parser.parse_args()

    verbose = not args.quiet

    print(f"\n{'#'*80}")
    print(f"# OVERRIDE + EXIT SWEEP")
    print(f"# Period: {args.start} to {args.end}")
    print(f"# Capital: ${args.capital:,.0f}")
    print(f"# TF systems: {args.tf}")
    print(f"{'#'*80}")

    # Download data once
    t0 = time.time()
    data = download_data()
    native_data = build_native_data(data)
    print(f"  Downloaded in {time.time() - t0:.1f}s")

    for tf_system in args.tf:
        if tf_system not in TF_SIM_CONFIG:
            print(f"\n  [WARN] Unknown TF system '{tf_system}', skipping")
            continue

        print(f"\n\n{'#'*80}")
        print(f"# {tf_system.upper()} SYSTEM")
        print(f"{'#'*80}")

        # Cache analyses (once per TF system)
        cached = cache_analyses(data, native_data, args.start, args.end,
                                tf_system, verbose=verbose)

        # Phase 1: Override sweep
        best_override = None
        if args.phase is None or args.phase == 1:
            override_results = run_override_sweep(
                cached, tf_system, args.capital, verbose=verbose)
            print_override_results(override_results, tf_system, args.start, args.end)

            # Find best override (highest P&L, excluding 'none')
            overrides_only = [r for r in override_results if r['combo']['mode'] != 'none']
            if overrides_only:
                best = max(overrides_only, key=lambda r: r['pnl'])
                bc = best['combo']
                best_override = OverrideConfig(
                    mode=bc['mode'],
                    position_threshold=bc['position_threshold'],
                    required_agreement=bc['required_agreement'],
                    override_tfs=bc['override_tfs'],
                    require_stabilizing=bc.get('require_stabilizing', False),
                )
                stab_str = ', stab=yes' if bc.get('require_stabilizing') else ''
                print(f"\n  Best override: mode={bc['mode']}, "
                      f"thresh={bc['position_threshold']}, "
                      f"agree={bc['required_agreement']}, "
                      f"tfs={bc['override_tfs']}{stab_str}  →  "
                      f"P&L=${best['pnl']:+,.0f}")

        # Phase 2: Exit sweep
        if args.phase is None or args.phase == 2:
            if best_override is None and args.phase == 2:
                # Default to a reasonable override for phase-2-only runs
                best_override = OverrideConfig(mode='suppress', position_threshold=0.25,
                                               required_agreement=1)
                print(f"\n  Using default override for Phase 2: {best_override}")

            exit_results = run_exit_sweep(
                cached, tf_system, best_override, args.capital, verbose=verbose)
            print_exit_results(exit_results, tf_system, args.start, args.end)

            # Best exit config
            best_exit = max(exit_results, key=lambda r: r['pnl'])
            ec = best_exit['combo']
            hold_hrs = best_exit['avg_hold'] / 60 if best_exit['avg_hold'] > 0 else 0
            print(f"\n  Best exit: trail={'yes' if ec['trail_enabled'] else 'no'}, "
                  f"stop={ec['stop_multiplier']:.1f}x, "
                  f"eod={'yes' if ec['eod_close_enabled'] else 'no'}, "
                  f"timeout={ec['timeout_multiplier']:.1f}x  →  "
                  f"P&L=${best_exit['pnl']:+,.0f}, "
                  f"avg hold={hold_hrs:.1f}h")

    print(f"\n{'#'*80}")
    print(f"# SWEEP COMPLETE")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
