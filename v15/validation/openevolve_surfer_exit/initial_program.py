"""
Phase C initial program: Current surfer-ml exit logic.

Defines get_effective_stop() and check_exits() that the evaluator
monkey-patches onto SurferMLAlgo. This is the starting point —
the LLM evolves from here.

IMPORTANT: These functions use `self` — they are methods that will be
bound to the SurferMLAlgo instance. Available on `self`:
  - self._pos_state[pos.pos_id] — per-position state dict
  - self.config.max_hold_bars — max bars before timeout
  - self.config.eval_interval — check every N bars

Position object fields:
  - pos.pos_id, pos.direction ('long'/'short')
  - pos.entry_price, pos.stop_price, pos.tp_price
  - pos.best_price (engine-ratcheted high watermark)
  - pos.hold_bars (5-min bars held)
  - pos.signal_type ('bounce' or 'break')
  - pos.metadata (dict with signal_bar_high/low, el_flagged, etc.)

ExitSignal(pos_id, price, reason) — reason: 'stop','trail','tp','ou_timeout','timeout'
"""

from typing import List, Optional
import pandas as pd


def get_effective_stop(self, position) -> Optional[float]:
    """Compute effective stop from position.best_price (engine-ratcheted).

    Pure function — no internal state needed. The engine controls how
    often best_price updates (every 5s, 1min, 5min, etc.), which
    determines how often the stop level changes.
    """
    state = self._pos_state.get(position.pos_id, {})
    entry = position.entry_price
    if entry <= 0:
        return position.stop_price

    trailing = position.best_price  # Engine-ratcheted high watermark
    is_breakout = position.signal_type == 'break'
    tp_dist = abs(position.tp_price - entry) / entry if entry > 0 else 0.01
    initial_stop_dist = abs(position.stop_price - entry) / entry if entry > 0 else 0.01

    twm = state.get('trail_width_mult', 1.0)
    el = state.get('el_flagged', False)
    fast_rev = state.get('fast_reversion', False) and not is_breakout

    if position.direction == 'long':
        profit_from_best = (trailing - entry) / entry

        if is_breakout:
            if initial_stop_dist < 0.001 and profit_from_best > 0.0001:
                trail_price = trailing * (1 - initial_stop_dist * 0.50 * twm)
                effective_stop = max(position.stop_price, trail_price)
            elif profit_from_best > 0.015:
                trail_price = trailing * (1 - initial_stop_dist * 0.01 * twm)
                effective_stop = max(position.stop_price, trail_price)
            elif profit_from_best > 0.008:
                trail_price = trailing * (1 - initial_stop_dist * 0.02 * twm)
                effective_stop = max(position.stop_price, trail_price)
            else:
                tier3_thresh = 0.002 if el else 0.0008
                trail_mult = 0.20 if el else 0.01
                if profit_from_best > tier3_thresh:
                    trail_price = trailing * (1 - initial_stop_dist * trail_mult * twm)
                    effective_stop = max(position.stop_price, trail_price)
                else:
                    effective_stop = position.stop_price
        else:
            # Bounce: ratio-based tiers
            profit_from_entry = (trailing - entry) / entry
            profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
            tight = el or fast_rev

            if profit_ratio >= 0.80:
                trail_price = trailing * (1 - initial_stop_dist * 0.005 * twm)
                effective_stop = max(position.stop_price, trail_price)
            elif profit_ratio >= (0.60 if tight else 0.55):
                trail_price = trailing * (1 - initial_stop_dist * 0.02 * twm)
                effective_stop = max(position.stop_price, trail_price)
            elif profit_ratio >= (0.30 if tight else 0.40):
                mult = 0.08 if tight else 0.06
                trail_price = trailing * (1 - initial_stop_dist * mult * twm)
                effective_stop = max(position.stop_price, trail_price)
            elif profit_ratio >= (0.10 if tight else 0.15):
                effective_stop = max(position.stop_price, entry * 1.0005)
            else:
                effective_stop = position.stop_price

        return effective_stop

    else:  # short
        profit_from_best = (entry - trailing) / entry

        if is_breakout:
            if initial_stop_dist < 0.001 and profit_from_best > 0.0001:
                trail_price = trailing * (1 + initial_stop_dist * 0.50 * twm)
                effective_stop = min(position.stop_price, trail_price)
            elif profit_from_best > 0.015:
                trail_price = trailing * (1 + initial_stop_dist * 0.01 * twm)
                effective_stop = min(position.stop_price, trail_price)
            elif profit_from_best > 0.008:
                trail_price = trailing * (1 + initial_stop_dist * 0.02 * twm)
                effective_stop = min(position.stop_price, trail_price)
            else:
                tier3_thresh = 0.002 if el else 0.0003
                trail_mult = 0.20 if el else 0.01
                if profit_from_best > tier3_thresh:
                    trail_price = trailing * (1 + initial_stop_dist * trail_mult * twm)
                    effective_stop = min(position.stop_price, trail_price)
                else:
                    effective_stop = position.stop_price
        else:
            profit_from_entry = (entry - trailing) / entry
            profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
            tight = el or fast_rev

            if profit_ratio >= 0.80:
                trail_price = trailing * (1 + initial_stop_dist * 0.005 * twm)
                effective_stop = min(position.stop_price, trail_price)
            elif profit_ratio >= (0.60 if tight else 0.55):
                trail_price = trailing * (1 + initial_stop_dist * 0.02 * twm)
                effective_stop = min(position.stop_price, trail_price)
            elif profit_ratio >= (0.30 if tight else 0.40):
                mult = 0.08 if tight else 0.06
                trail_price = trailing * (1 + initial_stop_dist * mult * twm)
                effective_stop = min(position.stop_price, trail_price)
            elif profit_ratio >= (0.10 if tight else 0.15):
                effective_stop = min(position.stop_price, entry * 0.9995)
            else:
                effective_stop = position.stop_price

        return effective_stop


def check_exits(self, time, bar, open_positions):
    """Check exits: stop/trail via get_effective_stop(), TP, and timeouts.

    In sequential mode, stop/trail exits are filtered by the engine
    (handled per-bar by _check_sequential_stops instead). TP/timeout
    still evaluated here at eval_interval boundaries using window
    high/low for accurate detection.
    """
    from v15.validation.unified_backtester.algo_base import ExitSignal

    exits = []
    max_hold = self.config.max_hold_bars if self.config.max_hold_bars > 0 else self.config.params.get('max_hold_bars', 60)
    eval_interval = self.config.eval_interval

    for pos in open_positions:
        state = self._pos_state.get(pos.pos_id, {})

        # Accumulate window high/low across bars for TP detection
        bar_high = bar['high']
        bar_low = bar['low']
        state.setdefault('window_high', bar_high)
        state.setdefault('window_low', bar_low)
        state['window_high'] = max(state['window_high'], bar_high)
        state['window_low'] = min(state['window_low'], bar_low)

        # Only evaluate exit every eval_interval bars
        state.setdefault('exit_bar_count', 0)
        state['exit_bar_count'] += 1
        if state['exit_bar_count'] < eval_interval:
            continue
        state['exit_bar_count'] = 0

        # Use accumulated window high/low for this eval cycle
        high = state['window_high']
        low = state['window_low']
        close = bar['close']
        # Reset window for next cycle
        state['window_high'] = bar['high']
        state['window_low'] = bar['low']

        is_breakout = pos.signal_type == 'break'
        ou_hl = state.get('ou_half_life', 5.0)

        # Stop/trail check — delegates to pure get_effective_stop()
        effective_stop = self.get_effective_stop(pos)

        if pos.direction == 'long':
            if low <= effective_stop:
                reason = 'stop' if effective_stop == pos.stop_price else 'trail'
                exits.append(ExitSignal(pos_id=pos.pos_id, price=effective_stop, reason=reason))
                continue
            if high >= pos.tp_price:
                exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.tp_price, reason='tp'))
                continue
            hold_5m = pos.hold_bars
            if not is_breakout and hold_5m >= max(6, int(ou_hl * 3)):
                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                continue
        else:  # short
            if high >= effective_stop:
                reason = 'stop' if effective_stop == pos.stop_price else 'trail'
                exits.append(ExitSignal(pos_id=pos.pos_id, price=effective_stop, reason=reason))
                continue
            if low <= pos.tp_price:
                exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.tp_price, reason='tp'))
                continue
            hold_5m = pos.hold_bars
            if not is_breakout and hold_5m >= max(6, int(ou_hl * 3)):
                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                continue

        # Hard timeout (max_hold is in 5-min bars)
        if pos.hold_bars >= max_hold:
            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='timeout'))

    return exits
