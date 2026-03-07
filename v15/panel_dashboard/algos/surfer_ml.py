"""
Surfer ML algo adapter.

Wraps GBT soft gate + profit-tier trailing with EL/ER sub-models.
"""

import logging
from copy import deepcopy
from typing import Optional

from .base import AlgoAdapter, Signal, ExitSignal

logger = logging.getLogger(__name__)

DEFAULT_SURFER_ML_CONFIG = {
    'signal_source': 'surfer_ml',
    'equity': 100_000,
    'flat_sizing': True,
    'max_positions': 2,
    'ml_model_dir': None,
}


class SurferMLAdapter(AlgoAdapter):
    """Adapter for Surfer ML signals (GBT + profit-tier trail)."""

    def __init__(self, algo_id: str, config: dict = None):
        cfg = deepcopy(DEFAULT_SURFER_ML_CONFIG)
        if config:
            cfg.update(config)
        super().__init__(algo_id, 'surfer_ml', cfg)

        self._feature_buffer = {}
        self._last_prediction = None
        self._cooldown_until = None

    def evaluate(self, price: float, analysis: dict,
                 open_trades: list[dict],
                 features: dict = None) -> Optional[Signal]:
        """Evaluate ML signal from GBT model output."""
        if not features or not features.get('ml_signal'):
            return None

        sig = features['ml_signal']
        if sig.get('action') == 'HOLD':
            return None

        if len(open_trades) >= self.config['max_positions']:
            return None

        action = sig['action']
        confidence = sig.get('confidence', 0.5)
        stop_price = sig.get('stop_price', price * 0.98)
        tp_price = sig.get('tp_price', price * 1.05)
        ou_half_life = sig.get('ou_half_life', 5.0)
        el_flagged = sig.get('el_flagged', False)
        trail_width_mult = sig.get('trail_width_mult', 1.0)

        equity = self.config['equity']
        shares = int(equity / price)

        direction = 'long' if action == 'BUY' else 'short'

        # Surfer ML uses profit-tier trailing (twm/el/fast_rev)
        trail_width = sig.get('trail_width', 0.01)

        return Signal(
            algo_id=self.algo_id,
            action=action,
            direction=direction,
            confidence=confidence,
            signal_type=sig.get('signal_type', 'ml_breakout'),
            stop_price=stop_price,
            tp_price=tp_price,
            shares=shares,
            trail_width=trail_width,
            ou_half_life=ou_half_life,
            el_flagged=el_flagged,
            trail_width_mult=trail_width_mult,
        )

    def check_exit(self, trade: dict, price: float,
                   bid: float = 0, ask: float = 0) -> Optional[ExitSignal]:
        """Check profit-tier trailing exit for ML trades."""
        best = trade.get('best_price', trade['entry_price'])
        trail_width = trade.get('trail_width', 0.01)
        stop_price = trade.get('stop_price', 0)
        direction = trade.get('direction', 'long')
        trail_width_mult = trade.get('trail_width_mult', 1.0)

        effective_trail = trail_width * trail_width_mult

        if direction == 'long':
            trail_stop = best * (1 - effective_trail)
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
            trail_stop = best * (1 + effective_trail)
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
        """Update best price and profit-tier trailing stop."""
        changes = {}
        direction = trade.get('direction', 'long')
        best = trade.get('best_price', trade['entry_price'])
        trail_width = trade.get('trail_width', 0.01)
        trail_width_mult = trade.get('trail_width_mult', 1.0)

        effective_trail = trail_width * trail_width_mult

        if direction == 'long':
            if price > best:
                changes['best_price'] = price
                new_stop = price * (1 - effective_trail)
                old_stop = trade.get('stop_price', 0)
                if new_stop > old_stop:
                    changes['stop_price'] = new_stop
        else:
            if price < best:
                changes['best_price'] = price
                new_stop = price * (1 + effective_trail)
                old_stop = trade.get('stop_price', float('inf'))
                if new_stop < old_stop:
                    changes['stop_price'] = new_stop

        hold_bars = trade.get('hold_bars', 0) + 1
        changes['hold_bars'] = hold_bars

        return changes
