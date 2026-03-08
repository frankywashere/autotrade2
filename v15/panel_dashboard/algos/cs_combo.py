"""
CS-5TF and CS-DW algo adapters.

Channel Surfer daily signals with exponential trailing stop.
"""

import logging
from copy import deepcopy
from typing import Optional

from .base import AlgoAdapter, Signal, ExitSignal

logger = logging.getLogger(__name__)

DEFAULT_CS_COMBO_CONFIG = {
    'signal_source': 'CS-5TF',
    'equity': 100_000,
    'flat_sizing': True,
    'trail_power': 12,
    'cooldown_days': 0,
    'max_positions': 2,
    'am_block_hour': 0,
}

DEFAULT_CS_DW_CONFIG = {
    **DEFAULT_CS_COMBO_CONFIG,
    'signal_source': 'CS-DW',
}


class CSComboAdapter(AlgoAdapter):
    """Adapter for CS-5TF or CS-DW signals."""

    def __init__(self, algo_id: str, config: dict = None):
        cfg = deepcopy(DEFAULT_CS_COMBO_CONFIG if 'dw' not in algo_id.lower()
                       else DEFAULT_CS_DW_CONFIG)
        if config:
            cfg.update(config)
        super().__init__(algo_id, cfg['signal_source'], cfg)

        self._cooldown_until = None
        self._last_analysis_result = None

    def evaluate(self, price: float, analysis: dict,
                 open_trades: list[dict],
                 features: dict = None) -> Optional[Signal]:
        """Evaluate CS signal from multi-TF analysis.

        analysis: output from prepare_multi_tf_analysis() with .signal attribute
        """
        if not analysis:
            return None

        sig = analysis.get('signal')
        if sig is None or sig.get('action') == 'HOLD':
            return None

        # Anti-pyramid: skip if already at max positions for this algo
        if len(open_trades) >= self.config['max_positions']:
            return None

        # Cooldown check
        if self._cooldown_until and features and features.get('now'):
            if features['now'] < self._cooldown_until:
                return None

        action = sig['action']
        confidence = sig.get('confidence', 0.5)
        signal_type = sig.get('signal_type', 'bounce')

        # Compute stop/TP from channel analysis
        stop_price = sig.get('stop_price', price * 0.98)
        tp_price = sig.get('tp_price', price * 1.05)

        # Sizing
        equity = self.config['equity']
        shares = int(equity / price)

        direction = 'long' if action == 'BUY' else 'short'

        # Trail width: exponential trail = 0.006 * (1 - conf) ^ trail_power
        trail_power = self.config['trail_power']
        trail_width = 0.006 * (1 - confidence) ** trail_power

        return Signal(
            algo_id=self.algo_id,
            action=action,
            direction=direction,
            confidence=confidence,
            signal_type=signal_type,
            stop_price=stop_price,
            tp_price=tp_price,
            shares=shares,
            trail_width=trail_width,
        )

    def check_exit(self, trade: dict, price: float,
                   bid: float = 0, ask: float = 0) -> Optional[ExitSignal]:
        """Check trailing stop exit for CS trades."""
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

        # Increment hold_bars
        hold_bars = trade.get('hold_bars', 0) + 1
        changes['hold_bars'] = hold_bars

        return changes
