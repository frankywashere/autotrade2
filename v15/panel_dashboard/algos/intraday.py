"""
Intraday algo adapter.

5-min intraday signals with confidence-based trailing stop.
"""

import datetime
import logging
from copy import deepcopy
from typing import Optional

from .base import AlgoAdapter, Signal, ExitSignal

logger = logging.getLogger(__name__)

DEFAULT_INTRADAY_CONFIG = {
    'signal_source': 'intraday',
    'equity': 100_000,
    'flat_sizing': True,
    'trail_power': 6,
    'trail_base': 0.006,
    'max_positions': 2,
    'max_trades_per_day': 20,
    'intraday_start': datetime.time(9, 30),
    'intraday_end': datetime.time(15, 25),
}


class IntradayAdapter(AlgoAdapter):
    """Adapter for intraday signals."""

    def __init__(self, algo_id: str, config: dict = None):
        cfg = deepcopy(DEFAULT_INTRADAY_CONFIG)
        if config:
            cfg.update(config)
        super().__init__(algo_id, 'intraday', cfg)

        self._bar_count = 0
        self._daily_trade_count = 0
        self._last_signal_time = None
        self._current_date = None

    def _reset_daily_if_needed(self, now: datetime.datetime = None):
        if now is None:
            now = datetime.datetime.now()
        today = now.date()
        if self._current_date != today:
            self._current_date = today
            self._daily_trade_count = 0
            self._bar_count = 0

    def evaluate(self, price: float, analysis: dict,
                 open_trades: list[dict],
                 features: dict = None) -> Optional[Signal]:
        """Evaluate intraday signal from 5-min features."""
        now = features.get('now') if features else None
        self._reset_daily_if_needed(now)

        if not features or not features.get('intraday_signal'):
            return None

        sig = features['intraday_signal']
        if sig.get('action') == 'HOLD':
            return None

        # Max trades per day
        if self._daily_trade_count >= self.config['max_trades_per_day']:
            return None

        # Anti-pyramid
        if len(open_trades) >= self.config['max_positions']:
            return None

        # Time window check
        if now:
            t = now.time() if hasattr(now, 'time') else None
            if t:
                start = self.config['intraday_start']
                end = self.config['intraday_end']
                if t < start or t > end:
                    return None

        action = sig['action']
        confidence = sig.get('confidence', 0.5)
        stop_price = sig.get('stop_price', price * 0.98)
        tp_price = sig.get('tp_price', price * 1.05)

        # Trail width: 0.006 * (1 - conf) ^ trail_power
        trail_power = self.config['trail_power']
        trail_base = self.config['trail_base']
        trail_width = trail_base * (1 - confidence) ** trail_power

        # Sizing
        equity = self.config['equity']
        shares = int(equity / price)

        direction = 'long' if action == 'BUY' else 'short'

        self._daily_trade_count += 1

        return Signal(
            algo_id=self.algo_id,
            action=action,
            direction=direction,
            confidence=confidence,
            signal_type='intraday',
            stop_price=stop_price,
            tp_price=tp_price,
            shares=shares,
            trail_width=trail_width,
        )

    def check_exit(self, trade: dict, price: float,
                   bid: float = 0, ask: float = 0) -> Optional[ExitSignal]:
        """Check trailing stop exit for intraday trades."""
        best = trade.get('best_price', trade['entry_price'])
        trail_width = trade.get('trail_width', 0.006)
        stop_price = trade.get('stop_price', 0)
        direction = trade.get('direction', 'long')

        if direction == 'long':
            trail_stop = best * (1 - trail_width)
            if price <= max(trail_stop, stop_price):
                return ExitSignal(
                    trade_id=trade['id'],
                    exit_reason='trailing' if price <= trail_stop else 'sl',
                    exit_price=price,
                )
            if trade.get('tp_price') and price >= trade['tp_price']:
                return ExitSignal(
                    trade_id=trade['id'],
                    exit_reason='tp',
                    exit_price=price,
                )
        else:
            trail_stop = best * (1 + trail_width)
            if price >= min(trail_stop, stop_price):
                return ExitSignal(
                    trade_id=trade['id'],
                    exit_reason='trailing' if price >= trail_stop else 'sl',
                    exit_price=price,
                )
            if trade.get('tp_price') and price <= trade['tp_price']:
                return ExitSignal(
                    trade_id=trade['id'],
                    exit_reason='tp',
                    exit_price=price,
                )

        return None

    def update_trailing(self, trade: dict, price: float) -> dict:
        """Update best price and trailing stop."""
        changes = {}
        direction = trade.get('direction', 'long')
        best = trade.get('best_price', trade['entry_price'])
        trail_width = trade.get('trail_width', 0.006)

        if direction == 'long':
            if price > best:
                changes['best_price'] = price
                new_stop = price * (1 - trail_width)
                old_stop = trade.get('stop_price', 0)
                if new_stop > old_stop:
                    changes['stop_price'] = new_stop
        else:
            if price < best:
                changes['best_price'] = price
                new_stop = price * (1 + trail_width)
                old_stop = trade.get('stop_price', float('inf'))
                if new_stop < old_stop:
                    changes['stop_price'] = new_stop

        hold_bars = trade.get('hold_bars', 0) + 1
        changes['hold_bars'] = hold_bars

        return changes
