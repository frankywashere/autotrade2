"""
Performance Analytics Service

Calculates trading performance metrics from logged trades
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.models.database import Trade


class PerformanceService:
    """
    Calculate performance metrics from manual trades
    """

    def calculate_performance(self) -> Dict:
        """
        Calculate overall performance metrics

        Returns:
            Dict with win rate, total P&L, Sharpe ratio, etc.
        """
        # Get all closed trades
        trades = list(Trade.select().where(Trade.exit_time.is_null(False)))

        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_pnl_pct': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }

        # Convert to DataFrame for easy analysis
        data = []
        for trade in trades:
            data.append({
                'pnl': trade.pnl or 0,
                'pnl_pct': trade.pnl_pct or 0,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price or trade.entry_price
            })

        df = pd.DataFrame(data)

        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L metrics
        total_pnl = df['pnl'].sum()
        total_pnl_pct = df['pnl_pct'].sum()

        # Average win/loss
        wins = df[df['pnl'] > 0]['pnl']
        losses = df[df['pnl'] < 0]['pnl']
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        # Sharpe ratio (annualized, assuming 252 trading days)
        returns = df['pnl_pct'].values
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Max drawdown (cumulative)
        cumulative_pnl = df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdowns = cumulative_pnl - running_max
        max_drawdown_dollars = drawdowns.min()

        # Calculate as percentage of peak
        if running_max.max() > 0:
            max_drawdown_pct = (max_drawdown_dollars / running_max.max()) * 100
        else:
            max_drawdown_pct = 0.0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown_pct
        }

    def get_pnl_over_time(self) -> Dict:
        """
        Get cumulative P&L over time for charting

        Returns:
            Dict with dates and cumulative P&L
        """
        trades = list(Trade
                     .select()
                     .where(Trade.exit_time.is_null(False))
                     .order_by(Trade.exit_time))

        if not trades:
            return {'dates': [], 'cumulative_pnl': []}

        dates = []
        cumulative_pnl = []
        running_total = 0

        for trade in trades:
            dates.append(trade.exit_time.isoformat())
            running_total += trade.pnl or 0
            cumulative_pnl.append(running_total)

        return {
            'dates': dates,
            'cumulative_pnl': cumulative_pnl
        }

    def get_returns_distribution(self) -> Dict:
        """
        Get distribution of returns for histogram

        Returns:
            Dict with returns and counts
        """
        trades = list(Trade
                     .select()
                     .where(Trade.exit_time.is_null(False)))

        if not trades:
            return {'returns': [], 'counts': []}

        returns = [trade.pnl_pct or 0 for trade in trades]

        # Create histogram bins
        hist, bin_edges = np.histogram(returns, bins=20)

        return {
            'returns': bin_edges.tolist(),
            'counts': hist.tolist()
        }


# Global singleton instance
performance_service = PerformanceService()
