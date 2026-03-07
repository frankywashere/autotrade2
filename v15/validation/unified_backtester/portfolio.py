"""
PortfolioManager — Tracks equity, positions, trades, and transaction costs.

Each algorithm gets its own equity pool. Positions are isolated per-algo.
"""

import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import pandas as pd

from .algo_base import CostModel


@dataclass
class Position:
    """An open position being tracked."""
    pos_id: str
    algo_id: str
    direction: str              # 'long' or 'short'
    entry_price: float
    entry_time: pd.Timestamp
    shares: int
    notional: float             # shares * entry_price
    stop_price: float
    tp_price: float
    confidence: float
    signal_type: str            # 'bounce', 'break', 'intraday'
    best_price: float           # Best price seen (for trailing stop)
    worst_price: float          # Worst price seen (for MAE)
    hold_bars: int = 0          # Bars held since entry
    metadata: dict = field(default_factory=dict)


@dataclass
class CompletedTrade:
    """A completed (closed) trade."""
    pos_id: str
    algo_id: str
    direction: str
    entry_price: float
    entry_time: pd.Timestamp
    exit_price: float
    exit_time: pd.Timestamp
    shares: int
    gross_pnl: float            # Before costs
    net_pnl: float              # After slippage + commission
    pnl_pct: float              # net_pnl / notional
    exit_reason: str
    confidence: float
    signal_type: str
    hold_bars: int
    mae_pct: float = 0.0       # Max adverse excursion %
    mfe_pct: float = 0.0       # Max favorable excursion %
    metadata: dict = field(default_factory=dict)


class AlgoEquity:
    """Equity tracking for a single algorithm."""

    def __init__(self, algo_id: str, initial_equity: float, max_per_trade: float,
                 max_positions: int, cost_model: CostModel):
        self.algo_id = algo_id
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.max_per_trade = max_per_trade
        self.max_positions = max_positions
        self.cost_model = cost_model
        self.positions: Dict[str, Position] = {}
        self.trades: List[CompletedTrade] = []
        self.equity_curve: List[dict] = []  # [{time, equity}]
        self.peak_equity = initial_equity
        self.max_drawdown = 0.0

    def record_equity(self, time: pd.Timestamp):
        """Snapshot current equity (cash + unrealized P&L)."""
        self.equity_curve.append({'time': time, 'equity': self.equity})
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd


class PortfolioManager:
    """Manages equity, positions, and trades for all algorithms."""

    def __init__(self):
        self._algos: Dict[str, AlgoEquity] = {}

    def register_algo(self, algo_id: str, initial_equity: float,
                      max_per_trade: float, max_positions: int,
                      cost_model: CostModel):
        """Register an algorithm with its own equity pool."""
        self._algos[algo_id] = AlgoEquity(
            algo_id=algo_id,
            initial_equity=initial_equity,
            max_per_trade=max_per_trade,
            max_positions=max_positions,
            cost_model=cost_model,
        )

    def can_open(self, algo_id: str) -> bool:
        """Check if algo can open a new position (within limits)."""
        ae = self._algos[algo_id]
        return len(ae.positions) < ae.max_positions

    def compute_shares(self, algo_id: str, price: float,
                       requested_shares: int = 0,
                       confidence: float = 1.0,
                       flat_sizing: bool = True) -> int:
        """Compute position size in shares.

        If requested_shares > 0, uses that (capped by max_per_trade).
        If flat_sizing, uses max_per_trade / price.
        Otherwise, scales by confidence: (max_per_trade * confidence) / price.
        """
        ae = self._algos[algo_id]
        if requested_shares > 0:
            max_shares = int(ae.max_per_trade / price) if price > 0 else 0
            return min(requested_shares, max_shares)

        if flat_sizing:
            notional = ae.max_per_trade
        else:
            notional = ae.max_per_trade * min(confidence, 1.0)

        return max(1, int(notional / price)) if price > 0 else 0

    def open_position(self, algo_id: str, direction: str, price: float,
                      shares: int, stop_price: float, tp_price: float,
                      confidence: float, signal_type: str,
                      time: pd.Timestamp, metadata: dict = None) -> Optional[Position]:
        """Open a position for the given algo.

        Returns the Position object, or None if limits exceeded.
        """
        ae = self._algos[algo_id]
        if len(ae.positions) >= ae.max_positions:
            return None

        pos_id = str(uuid.uuid4())[:8]
        notional = shares * price

        pos = Position(
            pos_id=pos_id,
            algo_id=algo_id,
            direction=direction,
            entry_price=price,
            entry_time=time,
            shares=shares,
            notional=notional,
            stop_price=stop_price,
            tp_price=tp_price,
            confidence=confidence,
            signal_type=signal_type,
            best_price=price,
            worst_price=price,
            metadata=metadata or {},
        )
        ae.positions[pos_id] = pos
        return pos

    def close_position(self, pos_id: str, price: float, time: pd.Timestamp,
                       reason: str) -> Optional[CompletedTrade]:
        """Close a position, compute P&L after costs."""
        # Find which algo owns this position
        for ae in self._algos.values():
            if pos_id in ae.positions:
                pos = ae.positions.pop(pos_id)
                cost = ae.cost_model

                # Exit slippage (entry slippage already baked into entry_price)
                if pos.direction == 'long':
                    exit_fill = price * (1.0 - cost.slippage_pct)
                else:
                    exit_fill = price * (1.0 + cost.slippage_pct)

                # Gross P&L (from slippage-adjusted prices)
                if pos.direction == 'long':
                    gross_pnl = (exit_fill - pos.entry_price) * pos.shares
                else:
                    gross_pnl = (pos.entry_price - exit_fill) * pos.shares

                # Commission (both sides)
                commission = cost.commission_per_share * pos.shares * 2
                net_pnl = gross_pnl - commission

                pnl_pct = net_pnl / pos.notional if pos.notional > 0 else 0.0

                # MAE/MFE
                if pos.direction == 'long':
                    mae_pct = (pos.entry_price - pos.worst_price) / pos.entry_price if pos.entry_price > 0 else 0
                    mfe_pct = (pos.best_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
                else:
                    mae_pct = (pos.worst_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
                    mfe_pct = (pos.entry_price - pos.best_price) / pos.entry_price if pos.entry_price > 0 else 0

                trade = CompletedTrade(
                    pos_id=pos.pos_id,
                    algo_id=pos.algo_id,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    entry_time=pos.entry_time,
                    exit_price=price,
                    exit_time=time,
                    shares=pos.shares,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=reason,
                    confidence=pos.confidence,
                    signal_type=pos.signal_type,
                    hold_bars=pos.hold_bars,
                    mae_pct=mae_pct,
                    mfe_pct=mfe_pct,
                    metadata=pos.metadata,
                )

                # Update equity
                ae.equity += net_pnl
                ae.trades.append(trade)
                return trade
        return None

    def update_position(self, pos_id: str, high: float, low: float):
        """Update best/worst prices and increment hold bars for a position."""
        for ae in self._algos.values():
            if pos_id in ae.positions:
                pos = ae.positions[pos_id]
                pos.hold_bars += 1
                if pos.direction == 'long':
                    pos.best_price = max(pos.best_price, high)
                    pos.worst_price = min(pos.worst_price, low)
                else:
                    pos.best_price = min(pos.best_price, low)
                    pos.worst_price = max(pos.worst_price, high)
                return

    def get_open_positions(self, algo_id: str = None) -> List[Position]:
        """Get open positions, optionally filtered by algo."""
        if algo_id:
            ae = self._algos.get(algo_id)
            return list(ae.positions.values()) if ae else []
        positions = []
        for ae in self._algos.values():
            positions.extend(ae.positions.values())
        return positions

    def get_trades(self, algo_id: str = None) -> List[CompletedTrade]:
        """Get completed trades, optionally filtered by algo."""
        if algo_id:
            ae = self._algos.get(algo_id)
            return list(ae.trades) if ae else []
        trades = []
        for ae in self._algos.values():
            trades.extend(ae.trades)
        return trades

    def get_equity(self, algo_id: str) -> float:
        """Current equity for this algo."""
        ae = self._algos.get(algo_id)
        return ae.equity if ae else 0.0

    def record_equity(self, time: pd.Timestamp, algo_id: str = None):
        """Snapshot equity for one or all algos."""
        if algo_id:
            if algo_id in self._algos:
                self._algos[algo_id].record_equity(time)
        else:
            for ae in self._algos.values():
                ae.record_equity(time)

    def get_equity_curve(self, algo_id: str) -> pd.DataFrame:
        """Get equity curve as DataFrame for an algo."""
        ae = self._algos.get(algo_id)
        if not ae or not ae.equity_curve:
            return pd.DataFrame(columns=['time', 'equity'])
        return pd.DataFrame(ae.equity_curve)
