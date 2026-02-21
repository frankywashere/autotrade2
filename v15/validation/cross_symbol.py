"""
Cross-symbol validation: Run Channel Surfer on SPY vs TSLA side-by-side.

Tests whether the physics-based edge generalizes beyond TSLA.

Usage:
    python3 -m v15.validation.cross_symbol --tsla data/TSLAMin.txt --spy data/SPYMin.txt --year-by-year
"""

import argparse
import time

import numpy as np

from v15.core.historical_data import prepare_backtest_data, prepare_year_data


def run_single_period(data: dict, args, label: str = ""):
    """Run backtest on a single period, return (metrics, trades, equity_curve)."""
    from v15.core.surfer_backtest import run_backtest

    tsla_5min = data['tsla_5min']
    if len(tsla_5min) < 200:
        print(f"  [{label}] Not enough data ({len(tsla_5min)} bars)")
        return None, None, None

    result = run_backtest(
        days=0,
        eval_interval=args.eval_interval,
        max_hold_bars=args.max_hold,
        position_size=args.capital / 10,
        min_confidence=args.min_confidence,
        use_multi_tf=True,
        tsla_df=tsla_5min,
        higher_tf_dict=data['higher_tf_data'],
        spy_df_input=data.get('spy_5min'),
        realistic=True,
        slippage_bps=args.slippage,
        commission_per_share=args.commission,
        max_leverage=args.max_leverage,
        initial_capital=args.capital,
    )
    if len(result) == 3:
        return result
    return result[0], result[1], []


def prepare_spy_as_primary(spy_path: str) -> dict:
    """Load SPY as the primary symbol (no secondary)."""
    # prepare_backtest_data's first arg is the primary symbol
    return prepare_backtest_data(spy_path)


def compute_profit_factor(trades) -> float:
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss < 1e-10:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_sharpe(yearly_returns: list) -> float:
    if len(yearly_returns) < 2:
        return 0.0
    arr = np.array(yearly_returns)
    if np.std(arr) < 1e-10:
        return 0.0
    return float(np.mean(arr) / np.std(arr))


def print_comparison_table(tsla_results: list, spy_results: list, capital: float):
    """Print side-by-side year-by-year comparison."""
    print(f"\n{'='*110}")
    print(f"  CROSS-SYMBOL VALIDATION — TSLA vs SPY")
    print(f"  Initial Capital: ${capital:,.0f} each")
    print(f"{'='*110}")
    print(f"  {'':8s} {'--- TSLA ---':>40s}    {'--- SPY ---':>40s}")
    print(f"  {'Year':<6} {'Trades':>7} {'Win%':>6} {'PF':>6} {'P&L ($)':>12}   "
          f"{'Trades':>7} {'Win%':>6} {'PF':>6} {'P&L ($)':>12}")
    print(f"  {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*12}   "
          f"{'-'*7} {'-'*6} {'-'*6} {'-'*12}")

    tsla_totals = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'all_trades': []}
    spy_totals = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'all_trades': []}
    tsla_yearly = []
    spy_yearly = []

    for (year, t_metrics, t_trades, _), (_, s_metrics, s_trades, _) in zip(tsla_results, spy_results):
        # TSLA columns
        if t_metrics and t_metrics.total_trades > 0:
            t_wr = t_metrics.wins / t_metrics.total_trades
            t_pf = compute_profit_factor(t_trades)
            t_pf_str = f"{t_pf:.2f}" if t_pf < 100 else "inf"
            t_pnl_str = f"${t_metrics.total_pnl:>+,.0f}"
            tsla_totals['trades'] += t_metrics.total_trades
            tsla_totals['wins'] += t_metrics.wins
            tsla_totals['pnl'] += t_metrics.total_pnl
            tsla_totals['all_trades'].extend(t_trades)
            tsla_yearly.append(t_metrics.total_pnl / capital)
        else:
            t_wr = 0
            t_pf_str = "--"
            t_pnl_str = "--"
            tsla_yearly.append(0.0)

        # SPY columns
        if s_metrics and s_metrics.total_trades > 0:
            s_wr = s_metrics.wins / s_metrics.total_trades
            s_pf = compute_profit_factor(s_trades)
            s_pf_str = f"{s_pf:.2f}" if s_pf < 100 else "inf"
            s_pnl_str = f"${s_metrics.total_pnl:>+,.0f}"
            spy_totals['trades'] += s_metrics.total_trades
            spy_totals['wins'] += s_metrics.wins
            spy_totals['pnl'] += s_metrics.total_pnl
            spy_totals['all_trades'].extend(s_trades)
            spy_yearly.append(s_metrics.total_pnl / capital)
        else:
            s_wr = 0
            s_pf_str = "--"
            s_pnl_str = "--"
            spy_yearly.append(0.0)

        t_wr_s = f"{t_wr:.0%}" if t_metrics and t_metrics.total_trades > 0 else "--"
        s_wr_s = f"{s_wr:.0%}" if s_metrics and s_metrics.total_trades > 0 else "--"
        t_n = t_metrics.total_trades if t_metrics else 0
        s_n = s_metrics.total_trades if s_metrics else 0

        print(f"  {year:<6} {t_n:>7} {t_wr_s:>6} {t_pf_str:>6} {t_pnl_str:>12}   "
              f"{s_n:>7} {s_wr_s:>6} {s_pf_str:>6} {s_pnl_str:>12}")

    # Totals
    print(f"  {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*12}   "
          f"{'-'*7} {'-'*6} {'-'*6} {'-'*12}")

    t_total_wr = tsla_totals['wins'] / max(tsla_totals['trades'], 1)
    s_total_wr = spy_totals['wins'] / max(spy_totals['trades'], 1)
    t_total_pf = compute_profit_factor(tsla_totals['all_trades'])
    s_total_pf = compute_profit_factor(spy_totals['all_trades'])
    t_pf_str = f"{t_total_pf:.2f}" if t_total_pf < 100 else "inf"
    s_pf_str = f"{s_total_pf:.2f}" if s_total_pf < 100 else "inf"

    t_pnl_total = f"${tsla_totals['pnl']:>+,.0f}"
    s_pnl_total = f"${spy_totals['pnl']:>+,.0f}"
    print(f"  {'TOTAL':<6} {tsla_totals['trades']:>7} {t_total_wr:>5.0%} {t_pf_str:>6} "
          f"{t_pnl_total:>12}   "
          f"{spy_totals['trades']:>7} {s_total_wr:>5.0%} {s_pf_str:>6} "
          f"{s_pnl_total:>12}")

    # Verdict
    print(f"\n  TSLA Sharpe: {compute_sharpe(tsla_yearly):.2f}")
    print(f"  SPY  Sharpe: {compute_sharpe(spy_yearly):.2f}")

    if spy_totals['trades'] > 0:
        if s_total_wr > 0.60 and s_total_pf > 1.5:
            print(f"\n  VERDICT: Edge GENERALIZES to SPY (WR={s_total_wr:.0%}, PF={s_pf_str})")
        elif s_total_pf > 1.0:
            print(f"\n  VERDICT: Partial generalization — SPY profitable but weaker")
        else:
            print(f"\n  VERDICT: Edge does NOT generalize to SPY")
    else:
        print(f"\n  VERDICT: No SPY trades generated")


def main():
    parser = argparse.ArgumentParser(description='Cross-symbol validation: TSLA vs SPY')
    parser.add_argument('--tsla', required=True, help='Path to TSLAMin.txt')
    parser.add_argument('--spy', required=True, help='Path to SPYMin.txt')
    parser.add_argument('--capital', type=float, default=100000.0)
    parser.add_argument('--max-leverage', type=float, default=4.0)
    parser.add_argument('--slippage', type=float, default=3.0)
    parser.add_argument('--commission', type=float, default=0.005)
    parser.add_argument('--eval-interval', type=int, default=6)
    parser.add_argument('--max-hold', type=int, default=60)
    parser.add_argument('--min-confidence', type=float, default=0.45)
    parser.add_argument('--year-by-year', action='store_true')
    parser.add_argument('--start-year', type=int, default=None)
    parser.add_argument('--end-year', type=int, default=None)
    args = parser.parse_args()

    t_start = time.time()

    # Load both datasets
    print("Loading TSLA data...")
    tsla_full = prepare_backtest_data(args.tsla)

    print("\nLoading SPY data (as primary symbol)...")
    spy_full = prepare_spy_as_primary(args.spy)

    tsla_5min = tsla_full['tsla_5min']
    spy_5min = spy_full['tsla_5min']  # prepare_backtest_data returns 'tsla_5min' key regardless

    first_year = max(tsla_5min.index[0].year, spy_5min.index[0].year)
    last_year = min(tsla_5min.index[-1].year, spy_5min.index[-1].year)
    print(f"\nOverlapping data: {first_year} to {last_year}")

    start_y = args.start_year or first_year
    end_y = args.end_year or last_year

    tsla_results = []
    spy_results = []

    for year in range(start_y, end_y + 1):
        print(f"\n{'='*40}")
        print(f"  YEAR {year}")
        print(f"{'='*40}")

        # TSLA
        tsla_year = prepare_year_data(tsla_full, year)
        if tsla_year:
            print(f"  Running TSLA ({len(tsla_year['tsla_5min']):,} bars)...")
            t_metrics, t_trades, t_eq = run_single_period(tsla_year, args, label=f"TSLA-{year}")
        else:
            t_metrics, t_trades, t_eq = None, [], []
        tsla_results.append((year, t_metrics, t_trades or [], t_eq or []))

        # SPY
        spy_year = prepare_year_data(spy_full, year)
        if spy_year:
            print(f"  Running SPY  ({len(spy_year['tsla_5min']):,} bars)...")
            s_metrics, s_trades, s_eq = run_single_period(spy_year, args, label=f"SPY-{year}")
        else:
            s_metrics, s_trades, s_eq = None, [], []
        spy_results.append((year, s_metrics, s_trades or [], s_eq or []))

    print_comparison_table(tsla_results, spy_results, args.capital)

    elapsed = time.time() - t_start
    print(f"\nTotal wall time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == '__main__':
    main()
