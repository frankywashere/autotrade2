"""
OE-Sig5 algo adapter.

Wraps the evolved bounce signal (TSLA/SPY/VIX + weekly channels).
"""

import logging
from copy import deepcopy
from typing import Optional

from .base import AlgoAdapter, Signal, ExitSignal

logger = logging.getLogger(__name__)

DEFAULT_OE_SIG5_CONFIG = {
    'signal_source': 'oe_sig5',
    'equity': 100_000,
    'flat_sizing': True,
    'trail_power': 12,
    'cooldown_days': 0,
    'max_positions': 1,
}


class OESig5Adapter(AlgoAdapter):
    """Adapter for OE-Sig5 evolved bounce signals."""

    def __init__(self, algo_id: str, config: dict = None):
        cfg = deepcopy(DEFAULT_OE_SIG5_CONFIG)
        if config:
            cfg.update(config)
        super().__init__(algo_id, 'oe_sig5', cfg)

        self._cooldown_until = None

    def evaluate(self, price: float, analysis: dict,
                 open_trades: list[dict],
                 features: dict = None) -> Optional[Signal]:
        """Evaluate OE-Sig5 bounce signal."""
        if not features or not features.get('oe_signal'):
            return None

        sig = features['oe_signal']
        if sig.get('action') == 'HOLD':
            return None

        if len(open_trades) >= self.config['max_positions']:
            return None

        action = sig['action']
        confidence = sig.get('confidence', 0.5)
        stop_price = sig.get('stop_price', price * 0.98)
        tp_price = sig.get('tp_price', price * 1.05)

        equity = self.config['equity']
        shares = int(equity / price)

        direction = 'long' if action == 'BUY' else 'short'

        trail_power = self.config['trail_power']
        trail_width = 0.006 * (1 - confidence) ** trail_power

        return Signal(
            algo_id=self.algo_id,
            action=action,
            direction=direction,
            confidence=confidence,
            signal_type='oe_bounce',
            stop_price=stop_price,
            tp_price=tp_price,
            shares=shares,
            trail_width=trail_width,
        )

    def check_exit(self, trade: dict, price: float,
                   bid: float = 0, ask: float = 0) -> Optional[ExitSignal]:
        """Check trailing stop exit for OE trades."""
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
