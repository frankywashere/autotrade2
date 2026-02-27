#!/usr/bin/env python3
"""
Combined Backtest -- Grid search over signal filter combinations.

Tests 9 configurations of the unified filter cascade (SQ gate, break predictor,
swing regime) on 2015-2024 IS data, then validates the winner on 2025 OOS.

Usage:
    python3 -m v15.validation.combined_backtest \
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt

    # Quick test (single config):
    python3 -m v15.validation.combined_backtest \
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt --config baseline
"""

import argparse
import sys
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# Filter configurations to test
# ---------------------------------------------------------------------------

CONFIGS = {
    'baseline':      {'sq_gate': 0.0,  'break_pred': False, 'swing': False, 'momentum': None},
    'sq50':          {'sq_gate': 0.50, 'break_pred': False, 'swing': False, 'momentum': None},
    'sq55':          {'sq_gate': 0.55, 'break_pred': False, 'swing': False, 'momentum': None},
    'bp_only':       {'sq_gate': 0.0,  'break_pred': True,  'swing': False, 'momentum': None},
    'swing_only':    {'sq_gate': 0.0,  'break_pred': False, 'swing': True,  'momentum': None},
    'sq50_bp':       {'sq_gate': 0.50, 'break_pred': True,  'swing': False, 'momentum': None},
    'sq50_swing':    {'sq_gate': 0.50, 'break_pred': False, 'swing': True,  'momentum': None},
    'all_50':        {'sq_gate': 0.50, 'break_pred': True,  'swing': True,  'momentum': None},
    'all_55':        {'sq_gate': 0.55, 'break_pred': True,  'swing': True,  'momentum': None},
    # --- MTF Momentum configs ---
    'mtf_conflict':  {'sq_gate': 0.0,  'break_pred': False, 'swing': False, 'momentum': 'conflict'},
    'mtf_exhaust':   {'sq_gate': 0.0,  'break_pred': False, 'swing': False, 'momentum': 'exhaust'},
    'mtf_full':      {'sq_gate': 0.0,  'break_pred': False, 'swing': False, 'momentum': 'full'},
    'sq50_mtf':      {'sq_gate': 0.50, 'break_pred': False, 'swing': False, 'momentum': 'full'},
}


def build_filter_cascade(cfg: dict):
    """Build a SignalFilterCascade from a config dict."""
    from v15.core.signal_filters import SignalFilterCascade

    momentum_mode = cfg.get('momentum')  # None | 'conflict' | 'exhaust' | 'full'

    if cfg['sq_gate'] == 0 and not cfg['break_pred'] and not cfg['swing'] and not momentum_mode:
        return None  # Baseline: no filters

    # MTF momentum params
    mtf_enabled = momentum_mode is not None
    # 'conflict' -> boost disabled (factor=1.0); 'exhaust' -> block disabled (penalty=1.0); 'full' -> both
    mtf_boost = 1.2 if momentum_mode in ('exhaust', 'full') else 1.0
    mtf_penalty = 0.3 if momentum_mode in ('conflict', 'full') else 1.0

    return SignalFilterCascade(
        sq_gate_threshold=cfg['sq_gate'],
        break_predictor_enabled=cfg['break_pred'],
        swing_regime_enabled=cfg['swing'],
        swing_boost=1.2,
        break_penalty=0.5,
        momentum_filter_enabled=mtf_enabled,
        momentum_boost=mtf_boost,
        momentum_conflict_penalty=mtf_penalty,
        momentum_context_tfs=['1h', '4h', 'daily'],
        momentum_min_tfs=2,
    )


def run_year_backtest(year_data, year, capital, vix_df, signal_filters=None,
                       bounce_cap=12.0, max_trade_usd=1_000_000.0):
    """Run a single year backtest with optional signal filters."""
    from v15.core.surfer_backtest import run_backtest

    tsla_5min = year_data['tsla_5min']
    if len(tsla_5min) < 200:
        return None

    result = run_backtest(
        days=0,
        eval_interval=6,
        max_hold_bars=60,
        position_size=capital / 10,
        min_confidence=0.45,
        use_multi_tf=True,
        tsla_df=tsla_5min,
        higher_tf_dict=year_data['higher_tf_data'],
        spy_df_input=year_data.get('spy_5min'),
        vix_df_input=vix_df,
        realistic=True,
        slippage_bps=3.0,
        commission_per_share=0.005,
        max_leverage=4.0,
        bounce_cap=bounce_cap,
        max_trade_usd=max_trade_usd,
        initial_capital=capital,
        capture_features=False,
        signal_filters=signal_filters,
    )

    metrics, trades, equity_curve = result[:3]
    return metrics, trades, equity_curve


def aggregate_results(results: dict) -> dict:
    """Aggregate per-year results into summary metrics."""
    if not results:
        return {}

    total_trades = sum(r[0].total_trades for r in results.values())
    total_wins = sum(r[0].wins for r in results.values())
    total_pnl = sum(r[0].total_pnl for r in results.values())
    gross_profit = sum(r[0].gross_profit for r in results.values())
    gross_loss = sum(r[0].gross_loss for r in results.values())
    max_dd = max(r[0].max_drawdown_pct for r in results.values())

    wr = total_wins / max(total_trades, 1)
    pf = gross_profit / max(abs(gross_loss), 1e-6)
    avg_pnl = total_pnl / max(total_trades, 1)
    yr_pnls = [r[0].total_pnl for r in results.values()]
    sharpe = float(np.mean(yr_pnls) / np.std(yr_pnls)) if len(yr_pnls) >= 2 and np.std(yr_pnls) > 0 else 0.0
    trades_per_yr = total_trades / max(len(results), 1)

    return {
        'trades': total_trades,
        'wins': total_wins,
        'wr': wr,
        'pf': pf,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'trades_per_yr': trades_per_yr,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
    }


def print_comparison(all_agg: dict):
    """Print a comparison table of all configurations."""
    print(f"\n{'='*120}")
    print("GRID SEARCH RESULTS -- COMPARISON TABLE")
    print(f"{'='*120}")

    header = f"{'Config':<14} {'Trades':>7} {'WR':>7} {'PF':>6} {'Total P&L':>14} {'Avg P&L':>10} {'Sharpe':>7} {'Trd/Yr':>7} {'MaxDD':>8}"
    print(header)
    print('-' * 120)

    baseline_pnl = all_agg.get('baseline', {}).get('total_pnl', 0)

    for name, agg in all_agg.items():
        if not agg:
            continue
        delta = agg['total_pnl'] - baseline_pnl
        delta_str = f"({delta:+,.0f})" if name != 'baseline' else ''
        print(f"{name:<14} {agg['trades']:>7,} {agg['wr']:>6.1%} {agg['pf']:>6.2f} "
              f"${agg['total_pnl']:>12,.0f} ${agg['avg_pnl']:>8,.0f} {agg['sharpe']:>7.2f} "
              f"{agg['trades_per_yr']:>7.0f} {agg['max_dd']:>7.1%}  {delta_str}")

    print(f"{'='*120}")


def main():
    parser = argparse.ArgumentParser(description='Combined backtest -- grid search over filter configs')
    parser.add_argument('--tsla', type=str, default='data/TSLAMin.txt')
    parser.add_argument('--spy', type=str, default='data/SPYMin.txt')
    parser.add_argument('--years', type=str, default='2015-2024',
                        help='IS year range (default: 2015-2024)')
    parser.add_argument('--oos-year', type=int, default=2025,
                        help='OOS year (default: 2025)')
    parser.add_argument('--capital', type=float, default=100_000.0)
    parser.add_argument('--bounce-cap', type=float, default=12.0)
    parser.add_argument('--max-trade-usd', type=float, default=1_000_000.0)
    parser.add_argument('--config', type=str, default=None,
                        help='Run single config only (e.g. baseline, sq50, all_50)')
    args = parser.parse_args()

    # Parse year range
    parts = args.years.split('-')
    start_year = int(parts[0])
    end_year = int(parts[1]) if len(parts) > 1 else start_year
    is_years = list(range(start_year, end_year + 1))

    # Select configs to run
    if args.config:
        if args.config not in CONFIGS:
            print(f"ERROR: Unknown config '{args.config}'. Available: {list(CONFIGS.keys())}")
            sys.exit(1)
        configs_to_run = {args.config: CONFIGS[args.config]}
        # Always include baseline for comparison
        if args.config != 'baseline':
            configs_to_run = {'baseline': CONFIGS['baseline'], **configs_to_run}
    else:
        configs_to_run = CONFIGS

    # ======================================================================
    # LOAD DATA
    # ======================================================================
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    t0 = time.time()

    from v15.core.historical_data import prepare_backtest_data, prepare_year_data, resample_to_tf, load_minute_data

    full_data = prepare_backtest_data(args.tsla, args.spy)

    # Add weekly TSLA (needed for swing regime S1041)
    if 'weekly' not in full_data['higher_tf_data']:
        print("  Resampling to weekly...")
        tsla_1min = full_data.get('tsla_1min')
        if tsla_1min is not None:
            full_data['higher_tf_data']['weekly'] = resample_to_tf(tsla_1min, '1W')
            print(f"  weekly: {len(full_data['higher_tf_data']['weekly']):,} bars")

    # Daily SPY (needed for swing regime S1041)
    daily_spy = None
    if args.spy and os.path.isfile(args.spy):
        print("  Loading SPY minute data for daily resampling...")
        spy_1min = load_minute_data(args.spy)
        daily_spy = resample_to_tf(spy_1min, '1D')
        print(f"  SPY daily: {len(daily_spy):,} bars")

    # Daily VIX
    vix_df = None
    try:
        from v15.validation.vix_loader import load_vix_daily
        vix_df = load_vix_daily(start='2014-01-01', end='2025-12-31')
    except Exception as e:
        print(f"  VIX load failed: {e}")

    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # ======================================================================
    # RUN GRID SEARCH (IS)
    # ======================================================================
    print(f"\n{'='*70}")
    print(f"RUNNING GRID SEARCH -- {len(configs_to_run)} configs x {len(is_years)} years")
    print(f"  bounce_cap={args.bounce_cap}x, max_trade_usd=${args.max_trade_usd:,.0f}, "
          f"capital=${args.capital:,.0f}")
    print(f"{'='*70}")

    all_is_results = {}

    for cfg_name, cfg in configs_to_run.items():
        print(f"\n--- Config: {cfg_name} (SQ={cfg['sq_gate']:.0%}, BP={cfg['break_pred']}, "
              f"Swing={cfg['swing']}, MTF={cfg.get('momentum')}) ---")

        # Build filter cascade
        cascade = build_filter_cascade(cfg)

        # Precompute swing regime if needed
        if cascade is not None and cascade.swing_regime_enabled:
            daily_tsla = full_data['higher_tf_data'].get('daily')
            weekly_tsla = full_data['higher_tf_data'].get('weekly')
            cascade.precompute_swing_regime(daily_tsla, daily_spy, vix_df, weekly_tsla)

        year_results = {}
        t_cfg = time.time()

        for year in is_years:
            year_data = prepare_year_data(full_data, year)
            if year_data is None:
                print(f"  {year}: no data, skipping")
                continue

            # For each year, reset cascade stats but keep precomputed swing regime
            if cascade is not None:
                for k in cascade.stats:
                    cascade.stats[k] = 0

            t_yr = time.time()
            result = run_year_backtest(
                year_data, year, args.capital, vix_df,
                signal_filters=cascade,
                bounce_cap=args.bounce_cap, max_trade_usd=args.max_trade_usd,
            )
            if result is None:
                print(f"  {year}: too few bars, skipping")
                continue

            year_results[year] = result
            m = result[0]
            elapsed = time.time() - t_yr
            print(f"  {year}: ${m.total_pnl:>10,.0f}  {m.total_trades:>4} trades  "
                  f"WR={m.win_rate:.1%}  PF={m.profit_factor:.2f}  ({elapsed:.1f}s)")

        all_is_results[cfg_name] = year_results

        # Print filter stats
        if cascade is not None:
            agg = aggregate_results(year_results)
            print(f"  Aggregate: ${agg.get('total_pnl', 0):,.0f}  "
                  f"Sharpe={agg.get('sharpe', 0):.2f}  Trades/yr={agg.get('trades_per_yr', 0):.0f}")
            print(f"  Elapsed: {time.time() - t_cfg:.1f}s")

    # ======================================================================
    # COMPARISON TABLE (IS)
    # ======================================================================
    all_is_agg = {name: aggregate_results(results) for name, results in all_is_results.items()}
    print_comparison(all_is_agg)

    # ======================================================================
    # FIND WINNER
    # ======================================================================
    baseline_pnl = all_is_agg.get('baseline', {}).get('total_pnl', 0)
    best_name = 'baseline'
    best_score = 0  # Combined score: Sharpe improvement + P&L improvement

    for name, agg in all_is_agg.items():
        if name == 'baseline' or not agg:
            continue
        pnl_lift = (agg['total_pnl'] - baseline_pnl) / max(abs(baseline_pnl), 1)
        sharpe_lift = agg['sharpe'] - all_is_agg['baseline'].get('sharpe', 0)
        score = sharpe_lift * 0.6 + pnl_lift * 0.4  # Weight Sharpe more than raw P&L
        if score > best_score:
            best_score = score
            best_name = name

    print(f"\n  WINNER: {best_name} (score={best_score:.4f})")

    # ======================================================================
    # OOS VALIDATION
    # ======================================================================
    if args.oos_year > 0:
        print(f"\n{'='*70}")
        print(f"OOS VALIDATION -- {args.oos_year}")
        print(f"{'='*70}")

        # Run baseline + winner on OOS
        oos_configs = {'baseline': CONFIGS['baseline']}
        if best_name != 'baseline':
            oos_configs[best_name] = CONFIGS[best_name]

        oos_year_data = prepare_year_data(full_data, args.oos_year)
        if oos_year_data is None:
            print(f"  {args.oos_year}: no data available for OOS")
        else:
            for cfg_name, cfg in oos_configs.items():
                cascade = build_filter_cascade(cfg)
                if cascade is not None and cascade.swing_regime_enabled:
                    daily_tsla = full_data['higher_tf_data'].get('daily')
                    weekly_tsla = full_data['higher_tf_data'].get('weekly')
                    cascade.precompute_swing_regime(daily_tsla, daily_spy, vix_df, weekly_tsla)

                t_oos = time.time()
                result = run_year_backtest(
                    oos_year_data, args.oos_year, args.capital, vix_df,
                    signal_filters=cascade,
                    bounce_cap=args.bounce_cap, max_trade_usd=args.max_trade_usd,
                )
                if result is None:
                    print(f"  {cfg_name}: too few bars")
                    continue

                m = result[0]
                elapsed = time.time() - t_oos
                print(f"  {cfg_name}: ${m.total_pnl:>10,.0f}  {m.total_trades:>4} trades  "
                      f"WR={m.win_rate:.1%}  PF={m.profit_factor:.2f}  "
                      f"Sharpe=N/A(1yr)  ({elapsed:.1f}s)")

                if cascade is not None:
                    print(f"    Filter stats:\n{cascade.summary()}")

    # ======================================================================
    # FINAL SUMMARY
    # ======================================================================
    print(f"\n{'='*70}")
    print("COMBINED BACKTEST COMPLETE")
    print(f"{'='*70}")
    print(f"  Winner: {best_name}")
    bw = all_is_agg.get(best_name, {})
    bl = all_is_agg.get('baseline', {})
    if bw and bl:
        print(f"  IS P&L: ${bw['total_pnl']:,.0f} vs baseline ${bl['total_pnl']:,.0f} "
              f"({bw['total_pnl'] - bl['total_pnl']:+,.0f})")
        print(f"  IS Sharpe: {bw['sharpe']:.2f} vs baseline {bl['sharpe']:.2f}")
        print(f"  IS Trades: {bw['trades']:,} vs baseline {bl['trades']:,}")
        print(f"  IS WR: {bw['wr']:.1%} vs baseline {bl['wr']:.1%}")


if __name__ == '__main__':
    main()
