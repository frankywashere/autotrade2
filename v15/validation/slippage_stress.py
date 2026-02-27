"""
Slippage stress test: Sweep slippage from 3-50 bps, find breakeven.

Tests how much execution cost the edge can absorb before going negative.

Usage:
    python3 -m v15.validation.slippage_stress --tsla data/TSLAMin.txt
"""

import argparse
import time

from v15.core.historical_data import prepare_backtest_data


def run_at_slippage(full_data: dict, slippage_bps: float, args) -> tuple:
    """Run full-period backtest at given slippage. Returns (metrics, trades)."""
    from v15.core.surfer_backtest import run_backtest

    tsla_5min = full_data['tsla_5min']
    result = run_backtest(
        days=0,
        eval_interval=args.eval_interval,
        max_hold_bars=args.max_hold,
        position_size=args.capital / 10,
        min_confidence=args.min_confidence,
        use_multi_tf=True,
        tsla_df=tsla_5min,
        higher_tf_dict=full_data['higher_tf_data'],
        spy_df_input=full_data.get('spy_5min'),
        realistic=True,
        slippage_bps=slippage_bps,
        commission_per_share=args.commission,
        max_leverage=args.max_leverage,
        initial_capital=args.capital,
    )
    metrics = result[0]
    trades = result[1] if len(result) > 1 else []
    return metrics, trades


def compute_profit_factor(trades) -> float:
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss < 1e-10:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def find_breakeven(full_data: dict, args, low: float, high: float, tol: float = 0.5) -> float:
    """Binary search for breakeven slippage (where P&L crosses zero)."""
    for _ in range(20):
        mid = (low + high) / 2
        if high - low < tol:
            break
        metrics, _ = run_at_slippage(full_data, mid, args)
        if metrics.total_pnl > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def main():
    parser = argparse.ArgumentParser(description='Slippage stress test')
    parser.add_argument('--tsla', required=True, help='Path to TSLAMin.txt')
    parser.add_argument('--spy', default=None, help='Path to SPYMin.txt (optional)')
    parser.add_argument('--capital', type=float, default=100000.0)
    parser.add_argument('--max-leverage', type=float, default=4.0)
    parser.add_argument('--commission', type=float, default=0.005)
    parser.add_argument('--eval-interval', type=int, default=6)
    parser.add_argument('--max-hold', type=int, default=60)
    parser.add_argument('--min-confidence', type=float, default=0.45)
    args = parser.parse_args()

    t_start = time.time()

    print("Loading data...")
    full_data = prepare_backtest_data(args.tsla, args.spy)
    print(f"  {len(full_data['tsla_5min']):,} 5-min bars loaded")

    slippage_levels = [3, 5, 7, 10, 15, 20, 30, 50]
    results = []

    print(f"\n{'='*80}")
    print(f"  SLIPPAGE STRESS TEST")
    print(f"  Capital: ${args.capital:,.0f}, Max Leverage: {args.max_leverage}x")
    print(f"{'='*80}")
    print(f"  {'Slippage':>10} {'Trades':>7} {'Win%':>6} {'PF':>7} {'Total P&L':>14} {'Avg Trade':>11}")
    print(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*7} {'-'*14} {'-'*11}")

    last_positive_slip = 3.0
    first_negative_slip = None

    for slip in slippage_levels:
        print(f"  Running at {slip} bps...", end='', flush=True)
        metrics, trades = run_at_slippage(full_data, slip, args)

        wr = metrics.wins / max(metrics.total_trades, 1)
        pf = compute_profit_factor(trades)
        avg = metrics.total_pnl / max(metrics.total_trades, 1)
        pf_str = f"{pf:.2f}" if pf < 100 else "inf"

        marker = ""
        if metrics.total_pnl <= 0 and first_negative_slip is None:
            first_negative_slip = slip
            marker = " <-- edge gone"
        elif metrics.total_pnl > 0:
            last_positive_slip = slip

        print(f"\r  {f'{slip} bps':>10} {metrics.total_trades:>7} {wr:>5.0%} {pf_str:>7} "
              f"{f'${metrics.total_pnl:>+,.0f}':>14} {f'${avg:>+,.0f}':>11}{marker}")

        results.append((slip, metrics, trades))

    # Binary search for exact breakeven
    if first_negative_slip is not None:
        print(f"\n  Searching for exact breakeven between {last_positive_slip} and {first_negative_slip} bps...")
        breakeven = find_breakeven(full_data, args, last_positive_slip, first_negative_slip)
        safety_margin = breakeven / 3.0  # relative to 3 bps baseline
        print(f"\n  BREAKEVEN: ~{breakeven:.1f} bps")
        print(f"  Safety margin: {safety_margin:.1f}x over 3 bps baseline")
    else:
        print(f"\n  Edge survives all tested slippage levels (up to {slippage_levels[-1]} bps)")
        print(f"  Breakeven is above {slippage_levels[-1]} bps — robust edge")

    # Summary
    baseline = results[0]
    print(f"\n  Baseline (3 bps): WR={baseline[1].wins/max(baseline[1].total_trades,1):.0%}, "
          f"PF={compute_profit_factor(baseline[2]):.2f}, P&L=${baseline[1].total_pnl:>+,.0f}")

    elapsed = time.time() - t_start
    print(f"\nTotal wall time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == '__main__':
    main()
