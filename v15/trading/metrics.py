"""
Trade Metrics and Equity Curve tracking.

Tracks all trades, computes P&L, Sharpe ratio, drawdown, win rate, etc.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class Trade:
    """Record of a completed trade."""
    entry_time: datetime
    exit_time: datetime
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    shares: int
    pnl: float  # Realized P&L (after costs)
    pnl_pct: float  # P&L as percentage of entry value
    commission: float  # Total commission paid
    slippage: float  # Total slippage cost
    signal_confidence: float  # Signal confidence at entry
    regime: str  # Market regime at entry
    primary_tf: str  # Primary TF used
    exit_reason: str  # 'stop_loss', 'take_profit', 'signal_flip', 'timeout'
    hold_bars: int  # Number of bars held


@dataclass
class EquityCurve:
    """Tracks equity over time."""
    timestamps: List[datetime] = field(default_factory=list)
    equity: List[float] = field(default_factory=list)
    drawdowns: List[float] = field(default_factory=list)

    def add_point(self, timestamp: datetime, equity_val: float):
        self.timestamps.append(timestamp)
        self.equity.append(equity_val)
        peak = max(self.equity)
        dd = (peak - equity_val) / peak if peak > 0 else 0.0
        self.drawdowns.append(dd)


@dataclass
class TradeMetrics:
    """Comprehensive trade performance metrics."""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: EquityCurve = field(default_factory=EquityCurve)

    def add_trade(self, trade: Trade):
        self.trades.append(trade)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losing_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return len(self.winning_trades) / len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_return_pct(self) -> float:
        if not self.equity_curve.equity:
            return 0.0
        initial = self.equity_curve.equity[0]
        final = self.equity_curve.equity[-1]
        return (final - initial) / initial * 100 if initial > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)

    @property
    def avg_win(self) -> float:
        wins = self.winning_trades
        if not wins:
            return 0.0
        return sum(t.pnl for t in wins) / len(wins)

    @property
    def avg_loss(self) -> float:
        losses = self.losing_trades
        if not losses:
            return 0.0
        return sum(t.pnl for t in losses) / len(losses)

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss. > 1 means profitable."""
        gross_profit = sum(t.pnl for t in self.winning_trades)
        gross_loss = abs(sum(t.pnl for t in self.losing_trades))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as fraction."""
        if not self.equity_curve.drawdowns:
            return 0.0
        return max(self.equity_curve.drawdowns)

    @property
    def sharpe_ratio(self) -> float:
        """
        Annualized Sharpe ratio.
        Assumes 5-min bars, ~78 bars/day, ~252 days/year.
        """
        if len(self.trades) < 2:
            return 0.0

        returns = [t.pnl_pct for t in self.trades]
        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std_ret = math.sqrt(var) if var > 0 else 0.0

        if std_ret == 0:
            return 0.0

        # Annualize: assume average ~2 trades per day
        trades_per_year = len(self.trades) / max(self._trading_days(), 1) * 252
        annualization = math.sqrt(trades_per_year) if trades_per_year > 0 else 1.0

        return (mean_ret / std_ret) * annualization

    @property
    def sortino_ratio(self) -> float:
        """Sortino ratio (only penalizes downside volatility)."""
        if len(self.trades) < 2:
            return 0.0

        returns = [t.pnl_pct for t in self.trades]
        mean_ret = sum(returns) / len(returns)
        downside_returns = [r for r in returns if r < 0]

        if not downside_returns:
            return float('inf') if mean_ret > 0 else 0.0

        downside_var = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        downside_std = math.sqrt(downside_var)

        if downside_std == 0:
            return 0.0

        trades_per_year = len(self.trades) / max(self._trading_days(), 1) * 252
        annualization = math.sqrt(trades_per_year) if trades_per_year > 0 else 1.0

        return (mean_ret / downside_std) * annualization

    @property
    def avg_hold_bars(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.hold_bars for t in self.trades) / len(self.trades)

    @property
    def total_commissions(self) -> float:
        return sum(t.commission for t in self.trades)

    @property
    def total_slippage(self) -> float:
        return sum(t.slippage for t in self.trades)

    def _trading_days(self) -> float:
        """Estimate number of trading days from trade timestamps."""
        if len(self.trades) < 2:
            return 1.0
        first = self.trades[0].entry_time
        last = self.trades[-1].exit_time
        delta = (last - first).total_seconds() / 86400
        return max(delta * 5 / 7, 1.0)  # Approximate trading days

    def by_regime(self) -> dict:
        """Break down metrics by regime."""
        regimes = {}
        for t in self.trades:
            if t.regime not in regimes:
                regimes[t.regime] = []
            regimes[t.regime].append(t)

        result = {}
        for regime, trades in regimes.items():
            wins = [t for t in trades if t.pnl > 0]
            result[regime] = {
                'trades': len(trades),
                'win_rate': len(wins) / len(trades) if trades else 0,
                'total_pnl': sum(t.pnl for t in trades),
                'avg_pnl': sum(t.pnl for t in trades) / len(trades) if trades else 0,
            }
        return result

    def by_tf(self) -> dict:
        """Break down metrics by primary timeframe."""
        tfs = {}
        for t in self.trades:
            if t.primary_tf not in tfs:
                tfs[t.primary_tf] = []
            tfs[t.primary_tf].append(t)

        result = {}
        for tf, trades in tfs.items():
            wins = [t for t in trades if t.pnl > 0]
            result[tf] = {
                'trades': len(trades),
                'win_rate': len(wins) / len(trades) if trades else 0,
                'total_pnl': sum(t.pnl for t in trades),
                'avg_pnl': sum(t.pnl for t in trades) / len(trades) if trades else 0,
            }
        return result

    def summary(self) -> str:
        """Print a formatted summary of all metrics."""
        lines = [
            "=" * 60,
            "TRADING PERFORMANCE SUMMARY",
            "=" * 60,
            f"Total Trades:     {self.total_trades}",
            f"Win Rate:         {self.win_rate:.1%}",
            f"Total P&L:        ${self.total_pnl:,.2f}",
            f"Total Return:     {self.total_return_pct:.2f}%",
            f"Avg P&L/Trade:    ${self.avg_pnl:,.2f}",
            f"Avg Win:          ${self.avg_win:,.2f}",
            f"Avg Loss:         ${self.avg_loss:,.2f}",
            f"Profit Factor:    {self.profit_factor:.2f}",
            f"Max Drawdown:     {self.max_drawdown:.1%}",
            f"Sharpe Ratio:     {self.sharpe_ratio:.2f}",
            f"Sortino Ratio:    {self.sortino_ratio:.2f}",
            f"Avg Hold (bars):  {self.avg_hold_bars:.1f}",
            f"Total Commissions:${self.total_commissions:,.2f}",
            f"Total Slippage:   ${self.total_slippage:,.2f}",
            "",
            "--- By Regime ---",
        ]
        for regime, stats in self.by_regime().items():
            lines.append(
                f"  {regime}: {stats['trades']} trades, "
                f"{stats['win_rate']:.0%} win, ${stats['total_pnl']:,.2f}"
            )

        lines.append("")
        lines.append("--- By Timeframe ---")
        for tf, stats in self.by_tf().items():
            lines.append(
                f"  {tf}: {stats['trades']} trades, "
                f"{stats['win_rate']:.0%} win, ${stats['total_pnl']:,.2f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)
