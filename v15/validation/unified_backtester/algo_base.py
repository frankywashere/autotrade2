"""
AlgoBase — Abstract base class for pluggable algorithms.

Every algorithm implements on_bar() for entries and check_exits() for exits.
The engine calls these at the appropriate times based on the algo's primary TF.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .data_provider import DataProvider


@dataclass
class CostModel:
    """Transaction cost model."""
    slippage_pct: float = 0.0001        # Per side (0.01% default)
    commission_per_share: float = 0.005  # Per side


@dataclass
class AlgoConfig:
    """Configuration for a single algorithm instance."""
    algo_id: str                          # Unique name: 'cs-5tf', 'surfer-ml', 'intraday-I'
    initial_equity: float = 100_000.0     # Starting equity for this algo
    max_equity_per_trade: float = 100_000.0  # Max notional per trade
    max_positions: int = 2                # Max simultaneous open positions
    primary_tf: str = '5min'              # Timeframe for on_bar dispatch
    eval_interval: int = 1                # Call on_bar every N primary bars
    cost_model: CostModel = field(default_factory=CostModel)
    params: dict = field(default_factory=dict)  # Algo-specific parameters
    exit_check_tf: str = '1min'           # TF for exit checking (1min = highest resolution)
    # Optional active hours hint: engine skips on_bar() outside these hours.
    # Exits still run regardless. Algo can do additional filtering internally.
    # None = no restriction (all hours active).
    active_start: object = None           # datetime.time or None
    active_end: object = None             # datetime.time or None


@dataclass
class Signal:
    """Entry signal produced by an algorithm."""
    algo_id: str
    direction: str          # 'long' or 'short'
    price: float            # Entry price (current bar close)
    confidence: float
    stop_pct: float
    tp_pct: float
    signal_type: str        # 'bounce', 'break', 'intraday', etc.
    shares: int = 0         # 0 = let portfolio compute from equity/price
    delayed_entry: bool = False  # True = enter at next session open (for daily algos)
    metadata: dict = field(default_factory=dict)


@dataclass
class ExitSignal:
    """Exit signal for an open position."""
    pos_id: str
    price: float            # Exit price
    reason: str             # 'trail', 'stop', 'tp', 'timeout', 'signal_flip', 'eod'


class AlgoBase(ABC):
    """Abstract base class for all trading algorithms.

    Subclasses must implement:
    - on_bar(): Called at each primary TF bar, returns entry signals
    - check_exits(): Called at each exit_check_tf bar, returns exit signals
    """

    def __init__(self, config: AlgoConfig, data: DataProvider):
        self.config = config
        self.data = data

    @property
    def algo_id(self) -> str:
        return self.config.algo_id

    @abstractmethod
    def on_bar(self, time: pd.Timestamp, bar: dict,
               open_positions: list) -> List[Signal]:
        """Called every eval_interval primary TF bars. Return entry signals.

        Args:
            time: Timestamp of the current bar
            bar: OHLCV dict for the current primary TF bar
            open_positions: List of Position objects currently open for this algo

        Returns:
            List of Signal objects (may be empty)

        IMPORTANT: The algo can access historical data via self.data.get_bars(tf, time).
        It must NOT access bars after `time`. The DataProvider enforces this.
        """
        ...

    @abstractmethod
    def check_exits(self, time: pd.Timestamp, bar: dict,
                    open_positions: list) -> List[ExitSignal]:
        """Called every exit_check_tf bar for algos with open positions.

        Args:
            time: Timestamp of the current 1-min (or exit_check_tf) bar
            bar: OHLCV dict for the current exit check bar
            open_positions: List of Position objects currently open for this algo

        Returns:
            List of ExitSignal objects for positions to close
        """
        ...

    def on_fill(self, trade: 'CompletedTrade'):
        """Called after a trade closes. Override to update internal state.

        Useful for tracking win/loss streaks, cooldown counters, etc.
        """
        pass

    def on_position_opened(self, position: 'Position'):
        """Called after a new position is opened. Override to update state.

        Useful for tracking anti-pyramid counters, daily trade counts, etc.
        """
        pass

    def warmup_bars(self) -> int:
        """Number of primary TF bars needed before first signal.

        Override if your algo needs historical context (e.g., 260 bars for
        channel detection warmup).
        """
        return 0
