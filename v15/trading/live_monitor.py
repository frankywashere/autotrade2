"""
Live Trading Monitor — signal evaluation, position tracking, exit alerts.

No auto-trading. No broker integration. Dashboard notifications only.
Persistence via JSON at ~/.x14/trading_state.json.
"""
import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import pytz

_ET = pytz.timezone('US/Eastern')


def _now_et() -> datetime:
    """Current time in US/Eastern (consistent with market hours and charts)."""
    return datetime.now(_ET)


def _is_market_open() -> bool:
    """True if current ET time is within regular trading hours (9:30-16:00, weekdays)."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    t = now.time()
    from datetime import time as _time
    return _time(9, 30) <= t < _time(16, 0)

from .signals import (
    TradeSignal, SignalType, MarketRegime,
    RegimeAdaptiveSignalEngine, HazardClock,
)
from .position_sizer import PositionSizer, PositionRecommendation
from .signal_filter import (
    FilteredSignal, classify_strategy_signals, scale_position, compute_momentum,
)

if TYPE_CHECKING:
    from ..inference import PerTFPrediction

STATE_PATH = Path.home() / ".x14" / "trading_state.json"
MAX_SIGNAL_HISTORY = 500


@dataclass
class LivePosition:
    """A manually-entered position being tracked."""
    pos_id: str
    strategy: str
    direction: str            # 'long' or 'short'
    entry_price: float
    shares: int
    entry_time: str           # ISO format
    stop_loss_price: float
    take_profit_price: float
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    best_price: float         # Best price since entry (for trail)
    signal_confidence: float
    primary_tf: str
    regime: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'LivePosition':
        return cls(**d)


@dataclass
class ClosedTrade:
    """Record of a closed trade."""
    pos_id: str
    strategy: str
    direction: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    entry_time: str
    exit_time: str
    exit_reason: str
    hold_minutes: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ClosedTrade':
        return cls(**d)


class TradingMonitor:
    """
    State machine for live signal monitoring and position tracking.

    Designed to be instantiated once per dashboard session (stored in
    st.session_state) and called on every auto-refresh cycle.
    """

    # Horizon-specific max hold times in minutes
    HORIZON_MAX_HOLD_MINUTES = {
        'short': 78 * 5,     # ~6.5 hours (1 trading day of 5-min bars)
        'medium': 156 * 5,   # ~13 hours (2 trading days)
        'long': 390 * 5,     # ~32.5 hours (1 trading week)
    }

    def __init__(self, initial_equity: float = 100000.0):
        self.signal_engine = RegimeAdaptiveSignalEngine()
        self.position_sizer = PositionSizer(capital=initial_equity)

        # State — loaded from disk
        self.positions: Dict[str, LivePosition] = {}
        self.signal_history: List[dict] = []
        self.closed_trades: List[ClosedTrade] = []
        self.equity: float = initial_equity
        self.peak_equity: float = initial_equity

        self._load_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        if not STATE_PATH.exists():
            return
        try:
            data = json.loads(STATE_PATH.read_text())
            self.equity = data.get('equity', self.equity)
            self.peak_equity = data.get('peak_equity', self.peak_equity)
            self.signal_history = data.get('signal_history', [])
            self.positions = {
                k: LivePosition.from_dict(v)
                for k, v in data.get('positions', {}).items()
            }
            self.closed_trades = [
                ClosedTrade.from_dict(t)
                for t in data.get('closed_trades', [])
            ]
        except Exception as e:
            print(f"[MONITOR] Failed to load state: {e}")

    def _save_state(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'equity': self.equity,
            'peak_equity': self.peak_equity,
            'signal_history': self.signal_history[-MAX_SIGNAL_HISTORY:],
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'closed_trades': [t.to_dict() for t in self.closed_trades],
        }
        STATE_PATH.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Signal evaluation (called on each refresh)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        per_tf_predictions: Dict[str, 'PerTFPrediction'],
        current_price: float,
        vix_level: float,
        tsla_close_series=None,
    ) -> List[FilteredSignal]:
        """Evaluate current predictions and return actionable signals.

        Args:
            per_tf_predictions: Per-TF predictions from the model.
            current_price: Current TSLA close price.
            vix_level: Current VIX close price.
            tsla_close_series: Optional pandas Series of close prices for momentum.

        Returns:
            List of FilteredSignal sorted by score (highest first).
        """
        if not per_tf_predictions:
            return []

        # Generate horizon signals
        horizon_signals = self.signal_engine.generate_horizon_signals(
            per_tf_predictions=per_tf_predictions,
        )
        if not horizon_signals:
            return []

        # Compute momentum (if price history available)
        mom_1d = 0.0
        mom_3d = 0.0
        if tsla_close_series is not None and len(tsla_close_series) > 0:
            mom_1d = compute_momentum(tsla_close_series, current_price, 78)
            mom_3d = compute_momentum(tsla_close_series, current_price, 234)

        # Classify strategies
        strategy_signals = classify_strategy_signals(horizon_signals, mom_1d, mom_3d)

        # Build FilteredSignal list, skipping strategies with open positions
        results: List[FilteredSignal] = []
        from ..config import TF_TO_HORIZON

        for strat_key, (signal, score) in strategy_signals.items():
            # Skip if already have a position for this strategy
            if strat_key in self.positions:
                continue

            if signal.entry_urgency <= 0.3:
                continue

            horizon = TF_TO_HORIZON.get(signal.primary_tf, 'medium')

            # Size position
            position = self.position_sizer.size_position(signal, current_price)
            if not position.should_trade:
                continue

            # Scale position
            scale_position(
                signal, position, horizon_signals,
                vix_level, self.equity, current_price,
            )

            fs = FilteredSignal(
                strategy=strat_key,
                signal=signal,
                score=score,
                horizon=horizon,
                position=position,
            )
            results.append(fs)

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Block entries outside regular trading hours
        if not _is_market_open():
            return []

        # Record to history
        now_str = _now_et().isoformat()
        for fs in results:
            self.signal_history.append({
                'time': now_str,
                'strategy': fs.strategy,
                'direction': fs.signal.signal_type.value,
                'confidence': round(fs.signal.confidence, 3),
                'urgency': round(fs.signal.entry_urgency, 3),
                'primary_tf': fs.signal.primary_tf,
                'regime': fs.signal.regime.regime.value,
                'score': round(fs.score, 3),
                'acted': False,
            })

        # Trim history
        if len(self.signal_history) > MAX_SIGNAL_HISTORY:
            self.signal_history = self.signal_history[-MAX_SIGNAL_HISTORY:]

        self._save_state()
        return results

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def enter_position(
        self,
        filtered_signal: FilteredSignal,
        actual_entry_price: float,
        shares_override: Optional[int] = None,
    ) -> str:
        """Record entry into a tracked position.

        Returns:
            Position ID string.
        """
        sig = filtered_signal.signal
        pos = filtered_signal.position
        shares = shares_override if shares_override is not None else pos.shares

        direction = 'long' if sig.signal_type == SignalType.LONG else 'short'

        # Compute stops from entry price
        if direction == 'long':
            stop_price = actual_entry_price * (1 - pos.stop_loss_pct)
            tp_price = actual_entry_price * (1 + pos.take_profit_pct)
        else:
            stop_price = actual_entry_price * (1 + pos.stop_loss_pct)
            tp_price = actual_entry_price * (1 - pos.take_profit_pct)

        pos_id = f"pos_{_now_et().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

        live_pos = LivePosition(
            pos_id=pos_id,
            strategy=filtered_signal.strategy,
            direction=direction,
            entry_price=actual_entry_price,
            shares=shares,
            entry_time=_now_et().isoformat(),
            stop_loss_price=stop_price,
            take_profit_price=tp_price,
            stop_loss_pct=pos.stop_loss_pct,
            take_profit_pct=pos.take_profit_pct,
            trailing_stop_pct=0.02,  # Default 2% trail
            best_price=actual_entry_price,
            signal_confidence=sig.confidence,
            primary_tf=sig.primary_tf,
            regime=sig.regime.regime.value,
        )

        self.positions[filtered_signal.strategy] = live_pos

        # Mark signal as acted-on in history
        for entry in reversed(self.signal_history):
            if entry['strategy'] == filtered_signal.strategy and not entry['acted']:
                entry['acted'] = True
                break

        self._save_state()
        return pos_id

    def check_exits(
        self,
        current_price: float,
        high: float,
        low: float,
    ) -> List[Tuple[str, str, float]]:
        """Check all open positions for exit conditions.

        Returns:
            List of (pos_id, exit_reason, suggested_exit_price) tuples.
            Does NOT auto-close — user must confirm.
        """
        from ..config import TF_TO_HORIZON

        alerts: List[Tuple[str, str, float]] = []

        for strat_key, pos in self.positions.items():
            # Update best price
            if pos.direction == 'long':
                if high > pos.best_price:
                    pos.best_price = high
            else:
                if pos.best_price == 0 or low < pos.best_price:
                    pos.best_price = low

            # Compute bars_held from wall clock time
            entry_time = datetime.fromisoformat(pos.entry_time)
            if entry_time.tzinfo is None:
                entry_time = _ET.localize(entry_time)
            elapsed_minutes = (_now_et() - entry_time).total_seconds() / 60.0
            bars_held = elapsed_minutes / 5.0  # 5-min bars

            horizon = TF_TO_HORIZON.get(pos.primary_tf, 'medium')
            max_hold_minutes = self.HORIZON_MAX_HOLD_MINUTES.get(horizon, 390 * 5)
            max_hold_bars = max_hold_minutes / 5.0

            # Progressive trail tightening
            hold_pct = min(1.0, bars_held / max(max_hold_bars, 1))
            tightening = 1.0 - hold_pct * 0.6
            effective_trail = pos.trailing_stop_pct * tightening

            # Profit lock: tighter trail when deeply profitable
            if pos.direction == 'long':
                tp_dist = pos.take_profit_price - pos.entry_price
                profit_pct = (pos.best_price - pos.entry_price) / tp_dist if tp_dist > 0 else 0
            else:
                tp_dist = pos.entry_price - pos.take_profit_price
                profit_pct = (pos.entry_price - pos.best_price) / tp_dist if tp_dist > 0 else 0
            if profit_pct >= 0.50:
                effective_trail *= 0.6

            # Check exit conditions
            exit_price = None
            exit_reason = None

            if pos.direction == 'long':
                if low <= pos.stop_loss_price:
                    exit_price, exit_reason = pos.stop_loss_price, 'stop_loss'
                elif pos.best_price > pos.entry_price:
                    trailing_stop = pos.best_price * (1 - effective_trail)
                    if trailing_stop > pos.stop_loss_price and low <= trailing_stop:
                        exit_price, exit_reason = trailing_stop, 'trailing_stop'
                if exit_price is None and high >= pos.take_profit_price:
                    exit_price, exit_reason = pos.take_profit_price, 'take_profit'
            else:
                if high >= pos.stop_loss_price:
                    exit_price, exit_reason = pos.stop_loss_price, 'stop_loss'
                elif pos.best_price > 0 and pos.best_price < pos.entry_price:
                    trailing_stop = pos.best_price * (1 + effective_trail)
                    if trailing_stop < pos.stop_loss_price and high >= trailing_stop:
                        exit_price, exit_reason = trailing_stop, 'trailing_stop'
                if exit_price is None and low <= pos.take_profit_price:
                    exit_price, exit_reason = pos.take_profit_price, 'take_profit'

            # Timeout
            if exit_price is None and elapsed_minutes >= max_hold_minutes:
                exit_price, exit_reason = current_price, 'timeout'

            if exit_price is not None:
                alerts.append((pos.pos_id, exit_reason, exit_price))

        # Save updated best_price values
        if alerts:
            self._save_state()

        return alerts

    def exit_position(
        self,
        strategy_key: str,
        actual_exit_price: float,
        exit_reason: str = 'manual',
    ) -> Optional[ClosedTrade]:
        """Close a tracked position and record the trade.

        Args:
            strategy_key: Strategy key (e.g. 'trend', 'bounce').
            actual_exit_price: The price the user actually exited at.
            exit_reason: Reason for exit.

        Returns:
            ClosedTrade summary, or None if position not found.
        """
        pos = self.positions.pop(strategy_key, None)
        if pos is None:
            return None

        entry_time = datetime.fromisoformat(pos.entry_time)
        if entry_time.tzinfo is None:
            entry_time = _ET.localize(entry_time)
        hold_minutes = (_now_et() - entry_time).total_seconds() / 60.0

        if pos.direction == 'long':
            pnl = (actual_exit_price - pos.entry_price) * pos.shares
        else:
            pnl = (pos.entry_price - actual_exit_price) * pos.shares

        entry_value = pos.entry_price * pos.shares
        pnl_pct = pnl / entry_value if entry_value > 0 else 0.0

        trade = ClosedTrade(
            pos_id=pos.pos_id,
            strategy=pos.strategy,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=actual_exit_price,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_time=pos.entry_time,
            exit_time=_now_et().isoformat(),
            exit_reason=exit_reason,
            hold_minutes=hold_minutes,
        )

        self.closed_trades.append(trade)
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.position_sizer.update_equity(self.equity)

        self._save_state()
        return trade

    # ------------------------------------------------------------------
    # Risk properties
    # ------------------------------------------------------------------

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as a fraction (0 = at peak)."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)

    def total_exposure(self, current_price: float) -> float:
        """Total dollar exposure across all open positions."""
        return sum(p.shares * current_price for p in self.positions.values())

    def exposure_pct(self, current_price: float) -> float:
        """Total exposure as a fraction of equity."""
        if self.equity <= 0:
            return 0.0
        return self.total_exposure(current_price) / self.equity

    def unrealized_pnl(self, current_price: float) -> float:
        """Total unrealized P&L across all open positions."""
        total = 0.0
        for pos in self.positions.values():
            if pos.direction == 'long':
                total += (current_price - pos.entry_price) * pos.shares
            else:
                total += (pos.entry_price - current_price) * pos.shares
        return total
