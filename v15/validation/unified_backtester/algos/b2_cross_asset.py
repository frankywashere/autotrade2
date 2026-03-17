"""
B2 Cross-Asset Algo — OpenEvolve-discovered signal.

Hybrid three-mode signal using TSLA/SPY/VIX 5-min bars:
1. Beta-adjusted mean reversion (TSLA/SPY ratio z-score + Bollinger Bands)
2. Momentum/breakout (15-bar range + opening range breakout)
3. Extreme excess return fade (20-bar beta-adjusted)

Discovered by Phase B2 OpenEvolve (180+ iterations of evolution).
Training: 2015-2024 on 1-min bars. Holdout: 2025 positive ($117K on $100K).
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..algo_base import AlgoBase, AlgoConfig, Signal, ExitSignal, CostModel

logger = logging.getLogger(__name__)


# Import the evolved signal function
from v15.validation.openevolve_new_signal.output.best_program import generate_signals as _b2_generate_signals


DEFAULT_B2_CONFIG = AlgoConfig(
    algo_id='b2-cross',
    initial_equity=100_000.0,
    max_equity_per_trade=100_000.0,
    max_positions=2,
    primary_tf='5min',
    eval_interval=1,            # Every 5-min bar
    exit_check_tf='5min',
    cost_model=CostModel(
        slippage_pct=0.0,
        commission_per_share=0.0,
    ),
    params={
        'flat_sizing': True,
        'max_hold_bars': 60,    # 5 hours in 5-min bars
    },
)


class B2CrossAssetAlgo(AlgoBase):
    """Cross-asset signal from B2 OpenEvolve discovery."""

    def __init__(self, config: AlgoConfig = None, data=None):
        super().__init__(config or DEFAULT_B2_CONFIG, data)
        self._pos_state: Dict[str, dict] = {}

    def warmup_bars(self) -> int:
        return 100  # Need 100 5-min bars for lookback

    def on_bar(self, time: pd.Timestamp, bar: dict,
               open_positions: list, context=None) -> List[Signal]:
        """Generate signals from B2 cross-asset logic."""
        try:
            # Get last 100 5-min bars for each symbol
            tsla_bars = self.data.get_bars('5min', time, symbol='TSLA')
            spy_bars = self.data.get_bars('5min', time, symbol='SPY')
            vix_bars = self.data.get_bars('5min', time, symbol='VIX')

            if len(tsla_bars) < 50 or len(spy_bars) < 50 or len(vix_bars) < 10:
                return []

            tsla_bars = tsla_bars.tail(100)
            spy_bars = spy_bars.tail(100)
            vix_bars = vix_bars.tail(100)

            # Build position info dict (matching B2 evaluator interface)
            has_long = any(p.direction == 'long' for p in open_positions)
            has_short = any(p.direction == 'short' for p in open_positions)
            position_info = {
                'has_long': has_long,
                'has_short': has_short,
                'n_positions': len(open_positions),
                'max_positions': self.config.max_positions,
            }

            # Call B2 evolved signal function
            signals_dicts = _b2_generate_signals(
                tsla_bars, spy_bars, vix_bars, time, position_info
            )

            if not signals_dicts:
                return []

            # Convert to Signal objects
            result = []
            for sig_dict in signals_dicts:
                if not isinstance(sig_dict, dict):
                    continue

                direction = sig_dict.get('direction', '').lower()
                if direction not in ('long', 'short'):
                    continue

                # Anti-pyramid: no same-direction or same-type positions
                if direction == 'long' and has_long:
                    continue
                if direction == 'short' and has_short:
                    continue

                confidence = float(np.clip(sig_dict.get('confidence', 0.5), 0.01, 1.0))
                stop_pct = float(np.clip(sig_dict.get('stop_pct', 0.008), 0.002, 0.030))
                tp_pct = float(np.clip(sig_dict.get('tp_pct', 0.015), 0.003, 0.050))

                result.append(Signal(
                    algo_id=self.config.algo_id,
                    direction=direction,
                    price=bar['close'],
                    confidence=confidence,
                    stop_pct=stop_pct,
                    tp_pct=tp_pct,
                    signal_type='b2_cross_asset',
                    metadata={
                        'signal_bar_high': bar['high'],
                        'signal_bar_low': bar['low'],
                    },
                ))

            return result

        except Exception as e:
            logger.error("B2 on_bar FAILED: %s", e, exc_info=True)
            return []

    def on_position_opened(self, position):
        """Initialize state for exit tracking."""
        self._pos_state[position.pos_id] = {
            'window_high': position.metadata.get('signal_bar_high', position.entry_price),
            'window_low': position.metadata.get('signal_bar_low', position.entry_price),
        }

    def check_exits(self, time: pd.Timestamp, bar: dict,
                    open_positions: list) -> List[ExitSignal]:
        """Simple exit logic: stop, TP, timeout."""
        exits = []
        max_hold = self.config.max_hold_bars if self.config.max_hold_bars > 0 else self.config.params.get('max_hold_bars', 60)
        eval_interval = self.config.eval_interval

        for pos in open_positions:
            state = self._pos_state.get(pos.pos_id, {})

            # Accumulate window high/low
            state.setdefault('window_high', bar['high'])
            state.setdefault('window_low', bar['low'])
            state['window_high'] = max(state['window_high'], bar['high'])
            state['window_low'] = min(state['window_low'], bar['low'])

            state.setdefault('exit_bar_count', 0)
            state['exit_bar_count'] += 1
            if state['exit_bar_count'] < eval_interval:
                continue
            state['exit_bar_count'] = 0

            high = state['window_high']
            low = state['window_low']
            close = bar['close']
            state['window_high'] = bar['high']
            state['window_low'] = bar['low']

            if pos.direction == 'long':
                if low <= pos.stop_price:
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.stop_price, reason='stop'))
                    continue
                if high >= pos.tp_price:
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.tp_price, reason='tp'))
                    continue
            else:
                if high >= pos.stop_price:
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.stop_price, reason='stop'))
                    continue
                if low <= pos.tp_price:
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.tp_price, reason='tp'))
                    continue

            # Timeout
            if pos.hold_bars >= max_hold:
                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='timeout'))

        return exits

    def on_fill(self, trade):
        """Clean up state."""
        self._pos_state.pop(trade.pos_id, None)

    def serialize_state(self, pos_id: str) -> dict:
        return dict(self._pos_state.get(pos_id, {}))

    def restore_state(self, pos_id: str, state: dict):
        self._pos_state[pos_id] = state
