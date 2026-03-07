"""
Results — Metrics computation and reporting for backtest results.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .portfolio import CompletedTrade, PortfolioManager


def compute_metrics(trades: List[CompletedTrade], initial_equity: float = 100_000.0) -> dict:
    """Compute standard trading metrics from a list of completed trades."""
    if not trades:
        return {
            'total_trades': 0, 'winners': 0, 'losers': 0,
            'win_rate': 0.0, 'total_pnl': 0.0, 'avg_pnl': 0.0,
            'profit_factor': 0.0, 'max_drawdown_pct': 0.0,
            'avg_hold_bars': 0, 'avg_winner': 0.0, 'avg_loser': 0.0,
            'best_trade': 0.0, 'worst_trade': 0.0,
            'sharpe_ratio': 0.0, 'total_return_pct': 0.0,
        }

    pnls = [t.net_pnl for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0

    # Drawdown from cumulative P&L
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    dd = peak - cum_pnl
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0
    max_dd_pct = max_dd / initial_equity * 100 if initial_equity > 0 else 0.0

    # Sharpe (daily returns approximation from trade P&L)
    pnl_arr = np.array(pnls)
    sharpe = 0.0
    if len(pnl_arr) > 1 and pnl_arr.std() > 0:
        sharpe = float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(252))

    return {
        'total_trades': len(trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(trades) * 100,
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / len(trades),
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'max_drawdown_pct': max_dd_pct,
        'avg_hold_bars': int(np.mean([t.hold_bars for t in trades])),
        'avg_winner': np.mean(winners) if winners else 0.0,
        'avg_loser': np.mean(losers) if losers else 0.0,
        'best_trade': max(pnls),
        'worst_trade': min(pnls),
        'sharpe_ratio': sharpe,
        'total_return_pct': total_pnl / initial_equity * 100 if initial_equity > 0 else 0.0,
    }


def print_report(trades: List[CompletedTrade], algo_id: str = '',
                 initial_equity: float = 100_000.0):
    """Print a formatted report for one algorithm's results."""
    m = compute_metrics(trades, initial_equity)
    header = f"Results for {algo_id}" if algo_id else "Backtest Results"
    print(f"\n{'='*60}")
    print(f"  {header}")
    print(f"{'='*60}")
    print(f"  Trades:      {m['total_trades']:>8}")
    print(f"  Win Rate:    {m['win_rate']:>7.1f}%  ({m['winners']}W / {m['losers']}L)")
    print(f"  Total P&L:   ${m['total_pnl']:>10,.0f}")
    print(f"  Avg P&L:     ${m['avg_pnl']:>10,.0f}")
    print(f"  Best Trade:  ${m['best_trade']:>10,.0f}")
    print(f"  Worst Trade: ${m['worst_trade']:>10,.0f}")
    print(f"  Avg Winner:  ${m['avg_winner']:>10,.0f}")
    print(f"  Avg Loser:   ${m['avg_loser']:>10,.0f}")
    print(f"  Profit Factor: {m['profit_factor']:>6.2f}")
    print(f"  Max Drawdown:  {m['max_drawdown_pct']:>6.2f}%")
    print(f"  Avg Hold:    {m['avg_hold_bars']:>8} bars")
    print(f"  Return:      {m['total_return_pct']:>7.1f}%")
    print(f"  Sharpe:      {m['sharpe_ratio']:>7.2f}")
    print(f"{'='*60}")

    # Exit reason breakdown
    reason_counts = {}
    for t in trades:
        reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1
    if reason_counts:
        print(f"  Exit reasons: {reason_counts}")


def print_summary_table(portfolio: PortfolioManager, algo_ids: List[str]):
    """Print a comparison table across multiple algorithms."""
    print(f"\n{'='*100}")
    print(f"  {'Algo':<20} {'Trades':>7} {'WR':>7} {'Total PnL':>12} {'Avg PnL':>10} {'PF':>7} {'MaxDD':>7} {'Return':>8}")
    print(f"  {'-'*90}")

    for algo_id in algo_ids:
        trades = portfolio.get_trades(algo_id)
        m = compute_metrics(trades, portfolio._algos[algo_id].initial_equity)
        print(f"  {algo_id:<20} {m['total_trades']:>7} {m['win_rate']:>6.1f}% "
              f"${m['total_pnl']:>10,.0f} ${m['avg_pnl']:>8,.0f} "
              f"{m['profit_factor']:>6.1f} {m['max_drawdown_pct']:>6.1f}% "
              f"{m['total_return_pct']:>7.1f}%")
    print(f"{'='*100}")


def trades_to_csv(trades: List[CompletedTrade], path: str):
    """Export trades to CSV."""
    if not trades:
        print(f"No trades to export")
        return
    rows = []
    for t in trades:
        rows.append({
            'pos_id': t.pos_id,
            'algo_id': t.algo_id,
            'direction': t.direction,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'shares': t.shares,
            'gross_pnl': round(t.gross_pnl, 2),
            'net_pnl': round(t.net_pnl, 2),
            'pnl_pct': round(t.pnl_pct * 100, 4),
            'exit_reason': t.exit_reason,
            'confidence': t.confidence,
            'signal_type': t.signal_type,
            'hold_bars': t.hold_bars,
            'mae_pct': round(t.mae_pct * 100, 4),
            'mfe_pct': round(t.mfe_pct * 100, 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Exported {len(trades)} trades to {path}")
