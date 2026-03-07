#!/usr/bin/env python3
"""
Grid search for Surfer ML breakout_stop_mult tuning.

The causal fix exposed that the ultra-tight breakout_stop_mult=0.05 was overfit
to optimistic intrabar ordering (97% WR -> 19.6% after fix). This grid search
explores wider stop multipliers to find the causal-safe optimum.

Usage:
    python -m v15.validation.grid_surfer_ml                    # all combos, parallel
    python -m v15.validation.grid_surfer_ml --workers 12       # limit parallelism
    python -m v15.validation.grid_surfer_ml --dry-run           # show grid without running
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from itertools import product

# Ensure project root on path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)


# ── Grid Parameters ──────────────────────────────────────────────────────────

BREAKOUT_STOP_MULTS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
STOP_PCTS = [0.010, 0.015, 0.020, 0.030]   # Default stop % (bounce baseline)
TP_PCTS = [0.012, 0.020, 0.030]             # Default TP %

# Fixed params
DATA_PATH = None  # Auto-detect
START = '2025-01-01'
END = '2026-03-01'
EQUITY = 100_000.0


def run_single(params: dict) -> dict:
    """Run a single backtest with given params. Returns metrics dict."""
    from v15.validation.unified_backtester.data_provider import DataProvider
    from v15.validation.unified_backtester.engine import BacktestEngine
    from v15.validation.unified_backtester.portfolio import PortfolioManager
    from v15.validation.unified_backtester.algos.surfer_ml import (
        SurferMLAlgo, DEFAULT_SURFER_ML_CONFIG,
    )
    from v15.validation.unified_backtester.results import compute_metrics

    # Find data
    data_path = params.get('data_path')
    if not data_path:
        for candidate in ['data/TSLAMin.txt', '../data/TSLAMin.txt',
                          'C:/AI/x14/data/TSLAMin.txt']:
            if os.path.isfile(candidate):
                data_path = candidate
                break
    if not data_path:
        return {'error': 'No data file found', **params}

    try:
        data = DataProvider(
            tsla_1min_path=data_path,
            start=params['start'],
            end=params['end'],
            rth_only=True,
        )

        config = deepcopy(DEFAULT_SURFER_ML_CONFIG)
        config.initial_equity = params['equity']
        config.max_equity_per_trade = params['equity']
        config.params['breakout_stop_mult'] = params['breakout_stop_mult']
        config.params['stop_pct'] = params['stop_pct']
        config.params['tp_pct'] = params['tp_pct']
        config.algo_id = (f"sml_bsm{params['breakout_stop_mult']:.2f}"
                          f"_sp{params['stop_pct']:.3f}"
                          f"_tp{params['tp_pct']:.3f}")

        algo = SurferMLAlgo(config, data)
        portfolio = PortfolioManager()
        portfolio.register_algo(
            algo_id=config.algo_id,
            initial_equity=config.initial_equity,
            max_per_trade=config.max_equity_per_trade,
            max_positions=config.max_positions,
            cost_model=config.cost_model,
        )

        engine = BacktestEngine(data, [algo], portfolio, verbose=False)
        engine.run()

        trades = portfolio.get_trades(config.algo_id)
        metrics = compute_metrics(trades, config.initial_equity)

        return {
            'breakout_stop_mult': params['breakout_stop_mult'],
            'stop_pct': params['stop_pct'],
            'tp_pct': params['tp_pct'],
            **metrics,
        }
    except Exception as e:
        return {
            'breakout_stop_mult': params['breakout_stop_mult'],
            'stop_pct': params['stop_pct'],
            'tp_pct': params['tp_pct'],
            'error': str(e),
            'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
            'profit_factor': 0, 'max_drawdown_pct': 0, 'sharpe_ratio': 0,
        }


def main():
    parser = argparse.ArgumentParser(description='Surfer ML Grid Search')
    parser.add_argument('--workers', type=int, default=None,
                        help='Max parallel workers (default: CPU count - 2)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to TSLAMin.txt')
    parser.add_argument('--start', type=str, default=START)
    parser.add_argument('--end', type=str, default=END)
    parser.add_argument('--dry-run', action='store_true',
                        help='Show grid without running')
    parser.add_argument('--csv', type=str, default='surfer_ml_grid_results.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    combos = list(product(BREAKOUT_STOP_MULTS, STOP_PCTS, TP_PCTS))
    print(f"Grid search: {len(combos)} combinations")
    print(f"  breakout_stop_mult: {BREAKOUT_STOP_MULTS}")
    print(f"  stop_pct: {STOP_PCTS}")
    print(f"  tp_pct: {TP_PCTS}")
    print(f"  Date range: {args.start} to {args.end}")

    if args.dry_run:
        for i, (bsm, sp, tp) in enumerate(combos, 1):
            print(f"  [{i:3d}] bsm={bsm:.2f}  stop={sp:.3f}  tp={tp:.3f}")
        return

    workers = args.workers or max(1, os.cpu_count() - 2)
    print(f"  Workers: {workers}")
    print()

    param_list = [
        {
            'breakout_stop_mult': bsm,
            'stop_pct': sp,
            'tp_pct': tp,
            'equity': EQUITY,
            'start': args.start,
            'end': args.end,
            'data_path': args.data,
        }
        for bsm, sp, tp in combos
    ]

    results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single, p): p for p in param_list}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            results.append(result)
            err = result.get('error', '')
            if err:
                print(f"  [{done:3d}/{len(combos)}] ERROR: {err}")
            else:
                print(f"  [{done:3d}/{len(combos)}] "
                      f"bsm={result['breakout_stop_mult']:.2f} "
                      f"stop={result['stop_pct']:.3f} "
                      f"tp={result['tp_pct']:.3f} | "
                      f"trades={result['total_trades']:3d} "
                      f"WR={result['win_rate']:.1f}% "
                      f"PnL=${result['total_pnl']:>8,.0f} "
                      f"PF={result['profit_factor']:.2f} "
                      f"DD={result['max_drawdown_pct']:.1f}% "
                      f"Sharpe={result['sharpe_ratio']:.2f}")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(results)} runs in {elapsed:.0f}s "
          f"({elapsed/len(results):.1f}s/run avg)")

    # Sort by profit factor (descending), then by total PnL
    results.sort(key=lambda r: (-r.get('profit_factor', 0),
                                 -r.get('total_pnl', 0)))

    # Print top 10
    print(f"\n{'='*80}")
    print(f"  TOP 10 CONFIGURATIONS (by Profit Factor)")
    print(f"{'='*80}")
    print(f"  {'BSM':>5} {'Stop%':>6} {'TP%':>6} | "
          f"{'Trades':>6} {'WR%':>5} {'PnL':>10} "
          f"{'PF':>6} {'DD%':>6} {'Sharpe':>7}")
    print(f"  {'-'*70}")
    for r in results[:10]:
        if r.get('error'):
            continue
        print(f"  {r['breakout_stop_mult']:>5.2f} "
              f"{r['stop_pct']:>6.3f} "
              f"{r['tp_pct']:>6.3f} | "
              f"{r['total_trades']:>6} "
              f"{r['win_rate']:>5.1f} "
              f"${r['total_pnl']:>9,.0f} "
              f"{r['profit_factor']:>6.2f} "
              f"{r['max_drawdown_pct']:>6.1f} "
              f"{r['sharpe_ratio']:>7.2f}")

    # Save CSV
    if args.csv:
        import csv
        fields = ['breakout_stop_mult', 'stop_pct', 'tp_pct',
                  'total_trades', 'winners', 'losers', 'win_rate',
                  'total_pnl', 'avg_pnl', 'profit_factor',
                  'max_drawdown_pct', 'sharpe_ratio', 'total_return_pct',
                  'avg_winner', 'avg_loser', 'best_trade', 'worst_trade',
                  'avg_hold_bars', 'error']
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nResults saved to {args.csv}")


if __name__ == '__main__':
    main()
