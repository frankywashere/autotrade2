"""
AlgoBase — Abstract base class for pluggable algorithms.

Every algorithm implements on_bar() for entries and check_exits() for exits.
The engine calls these at the appropriate times based on the algo's primary TF.

Used by BOTH the offline BacktestEngine and the live LiveEngine.
Algos access data through self.data (DataProvider or LiveDataProvider).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    """Transaction cost model."""
    slippage_pct: float = 0.0001        # Per side (0.01% default)
    commission_per_share: float = 0.005  # Per side


@dataclass
class TradeContext:
    """ML feature context passed to on_bar() for signal quality prediction.

    Built by BacktestEngine (from PortfolioManager) or LiveEngine (from TradeDB).
    Used by _extract_signal_features() for GBT/EL/ER predictions.
    """
    recent_trades: list = field(default_factory=list)  # Last N closed trades (dicts)
    daily_pnl: float = 0.0              # Today's realized P&L
    win_streak: int = 0                 # Current consecutive wins
    loss_streak: int = 0                # Current consecutive losses
    equity: float = 100_000.0           # Current equity
    spy_price: float = 0.0              # Current SPY price
    vix_price: float = 0.0              # Current VIX price


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
    stop_check_mode: str = 'sequential'     # 'current', 'fixed', 'pessimistic', 'sequential'
    stop_check_interval: int = 1           # 1-min bars between broker stop checks (fixed/pessimistic)
    stop_check_delay: int = 0             # 1-min bars to skip before first check after lock (fixed/pessimistic)
    exit_grace_bars: int = 5              # 1-min bars after entry before stop checks activate (sequential)
    seq_check_price: str = 'low'          # Price field for stop check: 'low', 'open', 'close' (sequential)
    seq_check_interval: int = 1           # Check every N 1-min bars after grace: 1=every bar, 5=5-min (sequential)
    # 5-sec sub-loop knobs (only active when 5s data available):
    stop_update_secs: int = 60            # How often to ratchet best_price + recompute effective_stop (seconds).
                                          # 5=every 5s bar, 60=every 1min (default), 300=every 5min
    stop_check_secs: int = 5             # How often to check if price breached the stop (seconds).
                                          # 5=every 5s bar (default), 60=every 1min
    grace_ratchet_secs: int = 60          # How often to ratchet best_price during grace period (seconds).
                                          # 0=no ratcheting during grace, 5=every 5s, 60=every 1min (default)
    live_orders: bool = False             # Whether to place real IB orders (live only)
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

    Used by both BacktestEngine and LiveEngine. Data access goes through
    self.data which may be DataProvider (backtest) or LiveDataProvider (live).
    """

    def __init__(self, config: AlgoConfig, data):
        self.config = config
        self.data = data

    @property
    def algo_id(self) -> str:
        return self.config.algo_id

    @abstractmethod
    def on_bar(self, time: pd.Timestamp, bar: dict,
               open_positions: list,
               context: TradeContext = None) -> List[Signal]:
        """Called every eval_interval primary TF bars. Return entry signals.

        Args:
            time: Timestamp of the current bar
            bar: OHLCV dict for the current primary TF bar
            open_positions: List of Position objects currently open for this algo
            context: ML feature context (trade history, equity, SPY/VIX)

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

    def get_effective_stop(self, position) -> Optional[float]:
        """Returns current effective stop for a position.

        Used by LiveEngine to sync broker-side resting stops.
        Override in algos with trailing stops (e.g., SurferMLAlgo).
        """
        return getattr(position, 'stop_price', None)

    def serialize_state(self, pos_id: str) -> dict:
        """Serialize algo-specific position state for crash recovery.

        Called by LiveEngine on state changes. Override to persist
        trail state, EL/ER flags, etc.
        """
        return {}

    def restore_state(self, pos_id: str, state: dict):
        """Restore algo-specific position state after restart.

        Called by LiveEngine during recovery.
        """
        pass
