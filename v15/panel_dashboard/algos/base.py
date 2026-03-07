"""
AlgoAdapter ABC — Live trading adapter interface for signal algorithms.

Each adapter wraps an existing scanner's signal logic and conforms to this
interface so ScannerManager can evaluate/exit/trail all algos uniformly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Signal:
    """Entry signal from an adapter."""
    algo_id: str
    action: str            # 'BUY' or 'SELL'
    direction: str         # 'long' or 'short'
    confidence: float
    signal_type: str       # 'cs', 'ml_breakout', 'intraday', etc.
    stop_price: float
    tp_price: float
    shares: int = 0        # 0 = let ScannerManager compute from equity/sizing
    trail_width: float = 0.0
    ou_half_life: float = 0.0
    el_flagged: bool = False
    trail_width_mult: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ExitSignal:
    """Exit signal from an adapter."""
    trade_id: int
    exit_reason: str       # 'tp', 'sl', 'trailing', 'timeout', 'manual', 'eod'
    exit_price: float = 0.0  # 0 = use current price


class AlgoAdapter(ABC):
    """Live trading adapter for a signal algorithm."""

    def __init__(self, algo_id: str, signal_source: str, config: dict):
        self.algo_id = algo_id
        self.signal_source = signal_source
        self.config = config
        self.enabled = True

    @abstractmethod
    def evaluate(self, price: float, analysis: dict,
                 open_trades: list[dict],
                 features: dict = None) -> Optional[Signal]:
        """Check for entry signal.

        open_trades: from DB for this algo_id, for anti-pyramid/max-position gating.
        Returns Signal or None.
        """

    @abstractmethod
    def check_exit(self, trade: dict, price: float,
                   bid: float = 0, ask: float = 0) -> Optional[ExitSignal]:
        """Check if an open trade should exit. Returns ExitSignal or None."""

    @abstractmethod
    def update_trailing(self, trade: dict, price: float) -> dict:
        """Update trailing stop / best price tracking.

        Returns dict of changed fields (e.g., {'best_price': X, 'trail_width': Y, 'stop_price': Z}).
        ScannerManager MUST persist these via trade_db.update_trade_state(trade_id, **changes)
        and modify the resting IB stop order if stop_price changed.

        Returns empty dict if nothing changed.
        """
