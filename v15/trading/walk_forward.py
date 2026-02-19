#!/usr/bin/env python3
"""
Walk-Forward Validation for the Trading Engine.

Splits the 60-day yfinance dataset into overlapping windows and runs
the backtest independently on each to test robustness.

Usage:
    python3 -m v15.trading.walk_forward [--checkpoint PATH]
"""
import sys
import time
from pathlib import Path
from collections import defaultdict


def run_walk_forward(checkpoint_path: str, calibration_path: str = None):
    """Run walk-forward validation."""
    from v15.inference import Predictor
    from v15.trading.run_backtest import fetch_data, fetch_native_tf_data
    from v15.trading.signals import RegimeAdaptiveSignalEngine
    from v15.trading.position_sizer import PositionSizer
    from v15.trading.backtester import Backtester, BacktestConfig

    # Load model
    print(f"[MODEL] Loading checkpoint: {checkpoint_path}")
    predictor = Predictor.load(checkpoint_path, calibration_path=calibration_path)
    print("[MODEL] Loaded")

    # Fetch all data
    tsla_df, spy_df, vix_df = fetch_data(60)
    native_data = fetch_native_tf_data()
    print()

    total_bars = len(tsla_df)

    # Define windows: each window needs at least 1000 bars warmup + trading bars
    # With 4315 total bars, we can create overlapping windows
    #
    # Strategy: 3 non-overlapping windows of ~1100 trading bars each
    # (after 1000-bar warmup)
    # Window 1: bars 0-2100 (warmup: 0-999, trade: 1000-2100)
    # Window 2: bars 1100-3200 (warmup: 1100-2099, trade: 2100-3200)
    # Window 3: bars 2200-4315 (warmup: 2200-3199, trade: 3200-4315)

    warmup_bars = 1000
    window_size = 2100  # 1000 warmup + 1100 trading
    step_size = 1100    # Non-overlapping trading regions

    windows = []
    start = 0
    while start + warmup_bars + 200 < total_bars:  # Need at least 200 trading bars
        end = min(start + window_size, total_bars)
        trading_start = start + warmup_bars
        trading_bars = end - trading_start
        windows.append({
            'start': start,
            'end': end,
            'trading_start': trading_start,
            'trading_bars': trading_bars,
            'start_date': tsla_df.index[trading_start],
            'end_date': tsla_df.index[end - 1],
        })
        start += step_size
        if end >= total_bars:
            break

    print(f"Walk-Forward Validation: {len(windows)} windows")
    print(f"Total data: {total_bars} bars ({total_bars * 5 / 60 / 6.5:.0f} trading days)")
    print(f"{'='*80}")

    all_results = []

    for i, w in enumerate(windows):
        print(f"\n--- Window {i+1}/{len(windows)} ---")
        print(f"  Bars: {w['start']}-{w['end']} "
              f"(trade: {w['trading_start']}-{w['end']}, {w['trading_bars']} bars)")
        print(f"  Dates: {w['start_date']} to {w['end_date']}")

        # Slice data for this window
        w_tsla = tsla_df.iloc[w['start']:w['end']]
        w_spy = spy_df.iloc[w['start']:w['end']]
        w_vix = vix_df.iloc[w['start']:w['end']]

        config = BacktestConfig(
            initial_capital=100000.0,
            eval_interval_bars=12,
            max_hold_bars=390,
        )
        signal_engine = RegimeAdaptiveSignalEngine()
        sizer = PositionSizer(capital=100000.0)
        backtester = Backtester(
            predictor=predictor,
            signal_engine=signal_engine,
            position_sizer=sizer,
            config=config,
        )

        try:
            result = backtester.run(
                w_tsla, w_spy, w_vix,
                native_bars_by_tf=native_data,
            )
            m = result.metrics
            print(f"  Trades: {m.total_trades}, WR: {m.win_rate:.0%}, "
                  f"P&L: ${m.total_pnl:,.2f} ({m.total_pnl/1000:.2f}%), "
                  f"PF: {m.profit_factor:.2f}, Sharpe: {m.sharpe_ratio:.2f}, "
                  f"DD: {m.max_drawdown:.1%}")
            all_results.append({
                'window': i + 1,
                'start_date': str(w['start_date']),
                'end_date': str(w['end_date']),
                'trading_bars': w['trading_bars'],
                'trades': m.total_trades,
                'win_rate': m.win_rate,
                'pnl': m.total_pnl,
                'pf': m.profit_factor,
                'sharpe': m.sharpe_ratio,
                'max_dd': m.max_drawdown,
                'avg_pnl': m.avg_pnl if m.total_trades > 0 else 0,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                'window': i + 1,
                'start_date': str(w['start_date']),
                'end_date': str(w['end_date']),
                'trading_bars': w['trading_bars'],
                'trades': 0, 'win_rate': 0, 'pnl': 0, 'pf': 0,
                'sharpe': 0, 'max_dd': 0, 'avg_pnl': 0,
            })

    # Aggregate results
    print(f"\n{'='*80}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*80}")

    total_trades = sum(r['trades'] for r in all_results)
    total_pnl = sum(r['pnl'] for r in all_results)
    profitable_windows = sum(1 for r in all_results if r['pnl'] > 0)
    windows_with_trades = sum(1 for r in all_results if r['trades'] > 0)

    print(f"\nWindows: {len(all_results)} total, "
          f"{profitable_windows} profitable, "
          f"{windows_with_trades} with trades")
    print(f"Total trades: {total_trades}")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Avg P&L/window: ${total_pnl/len(all_results):,.2f}")

    if total_trades > 0:
        all_wrs = [r['win_rate'] for r in all_results if r['trades'] > 0]
        avg_wr = sum(all_wrs) / len(all_wrs) if all_wrs else 0
        print(f"Avg win rate: {avg_wr:.0%}")

    # Per-window table
    print(f"\n{'Window':>6s} {'Dates':>40s} {'Trades':>6s} {'WR':>5s} "
          f"{'P&L':>10s} {'PF':>6s} {'Sharpe':>7s} {'DD':>5s}")
    print(f"{'-'*90}")
    for r in all_results:
        dates = f"{r['start_date'][:10]} to {r['end_date'][:10]}"
        print(f"{r['window']:6d} {dates:>40s} {r['trades']:6d} "
              f"{r['win_rate']:5.0%} ${r['pnl']:9,.2f} "
              f"{r['pf']:6.2f} {r['sharpe']:7.2f} {r['max_dd']:5.1%}")

    # Consistency score: what fraction of windows are profitable?
    consistency = profitable_windows / max(len(all_results), 1)
    print(f"\nConsistency score: {consistency:.0%} of windows profitable")
    if consistency >= 0.67:
        print("  --> ROBUST: Edge appears consistent across time periods")
    elif consistency >= 0.50:
        print("  --> MODERATE: Edge present but regime-dependent")
    else:
        print("  --> WEAK: Edge may be overfitted to specific period")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Walk-forward validation')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--calibration', default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        for c in ['/tmp/x23_best_per_tf.pt', 'models/x23_best_per_tf.pt']:
            if Path(c).exists():
                checkpoint = c
                break
    if checkpoint is None:
        print("ERROR: No checkpoint found")
        sys.exit(1)

    calibration = args.calibration
    if calibration is None:
        cp_dir = Path(checkpoint).parent
        for name in ['temperature_calibration_x23.json', 'temperature_calibration.json']:
            if (cp_dir / name).exists():
                calibration = str(cp_dir / name)
                break

    start = time.time()
    run_walk_forward(checkpoint, calibration)
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
