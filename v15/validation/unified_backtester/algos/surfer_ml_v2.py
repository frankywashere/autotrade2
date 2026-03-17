"""
Surfer ML V2 Algorithm — Evolved exit logic from Phase C OpenEvolve.

Subclasses SurferMLAlgo: IDENTICAL entries (physics + ML gating), but
replaces get_effective_stop() and check_exits() with evolved versions.

Key innovations over V1:
- Stall detection: progressive trail tightening when best_price stops advancing
- Momentum tracking: consecutive up/down bars, 3-bar momentum chains
- Time decay: hold_frac-based trail tightening as position ages
- Profit lock floor: locks in escalating % of peak profit (+ stall boost)
- Pre-stop protection: exit at 70-76% of way to stop when no peak profit
- Early-hold adverse exits: cut losers fast in first 25% of hold
- Retrace detection: exit when retracing from peak without momentum support
- Volatility-adaptive: min trail gap from recent bar range
- Adverse candle detection: strong opposing candle + below/above entry + no profit → exit fast
- EMA-slope micro-profit protection: moderate profit + EMA declining + close near entry → protect small wins
- Adaptive stall thresholds: big winners get more room to consolidate
- Near-TP trail tighter: bounce profit_ratio>=0.80 → 0.0040, short >=0.75 → 0.0025
"""

import math
from typing import List, Optional

from .surfer_ml import SurferMLAlgo
from ..algo_base import AlgoConfig, AlgoBase, ExitSignal, CostModel


DEFAULT_SURFER_ML_V2_CONFIG = AlgoConfig(
    algo_id='surfer-ml-v2',
    initial_equity=100_000.0,
    max_equity_per_trade=100_000.0,
    max_positions=2,
    primary_tf='5min',
    eval_interval=3,
    exit_check_tf='5min',
    cost_model=CostModel(
        slippage_pct=0.0,
        commission_per_share=0.0,
    ),
    params={
        'flat_sizing': True,
        'min_confidence': 0.01,
        'max_hold_bars': 60,
        'ou_half_life': 5.0,
        'stop_pct': 0.015,
        'tp_pct': 0.012,
        'ml_model_dir': None,
        'atr_period': 14,
        'breakout_stop_mult': 1.00,
    },
)


class SurferMLV2Algo(SurferMLAlgo):
    """Surfer ML V2 — evolved exit logic, identical entries.

    Phase C OpenEvolve result: $70.9K PnL, 0.747 Sharpe, 12.6% DD
    vs V1 baseline: $38K PnL, ~0.5 Sharpe, ~15% DD
    """

    def __init__(self, config: AlgoConfig = None, data=None):
        super().__init__(config or DEFAULT_SURFER_ML_V2_CONFIG, data)

    def on_position_opened(self, position):
        """Initialize state for evolved exit logic.

        Seeds V1 keys plus new keys needed by evolved exits.
        """
        sig_high = position.metadata.get('signal_bar_high', position.entry_price)
        sig_low = position.metadata.get('signal_bar_low', position.entry_price)
        self._pos_state[position.pos_id] = {
            # V1 keys (from signal metadata)
            'el_flagged': position.metadata.get('el_flagged', False),
            'trail_width_mult': position.metadata.get('trail_width_mult', 1.0),
            'fast_reversion': position.metadata.get('fast_reversion', False),
            'ou_half_life': position.metadata.get('ou_half_life', 5.0),
            'window_high': sig_high,
            'window_low': sig_low,
            # V2 keys (evolved exit state)
            'last_close': position.entry_price,
            'prev_close': position.entry_price,
            'prev_prev_close': position.entry_price,
            'consec_up_bars': 0,
            'consec_down_bars': 0,
            'stall_bars': 0,
            'recent_bar_range': 0.002,
            'close_ema': position.entry_price,
            'prev_ema': position.entry_price,
            'tracked_best': position.best_price,
            'exit_bar_count': 0,
            'prev_close_ema': position.entry_price,
        }

    def get_effective_stop(self, position) -> Optional[float]:
        state = self._pos_state.get(position.pos_id, {})
        entry = position.entry_price
        if entry <= 0:
            return position.stop_price

        trailing = position.best_price
        last_close = state.get('last_close', trailing)

        is_breakout = position.signal_type == 'break'
        tp_dist = abs(position.tp_price - entry) / entry if entry > 0 else 0.01
        initial_stop_dist = abs(position.stop_price - entry) / entry if entry > 0 else 0.01

        twm = state.get('trail_width_mult', 1.0)
        el = state.get('el_flagged', False)
        fast_rev = state.get('fast_reversion', False) and not is_breakout

        prev_close = state.get('prev_close', last_close)
        prev_prev_close = state.get('prev_prev_close', last_close)
        long_momentum = last_close > prev_close > prev_prev_close
        short_momentum = last_close < prev_close < prev_prev_close

        consec_up = state.get('consec_up_bars', 0)
        consec_down = state.get('consec_down_bars', 0)
        strong_long_momentum = consec_up >= 3
        strong_short_momentum = consec_down >= 3

        stall_bars = state.get('stall_bars', 0)
        if stall_bars >= 25:
            stall_mult = 0.14
        elif stall_bars >= 22:
            stall_mult = 0.18
        elif stall_bars >= 20:
            stall_mult = 0.22
        elif stall_bars >= 18:
            stall_mult = 0.30
        elif stall_bars >= 15:
            stall_mult = 0.35
        elif stall_bars >= 12:
            stall_mult = 0.42
        elif stall_bars >= 10:
            stall_mult = 0.48
        elif stall_bars >= 8:
            stall_mult = 0.49
        elif stall_bars >= 6:
            stall_mult = 0.57
        elif stall_bars >= 3:
            stall_mult = 0.77
        elif stall_bars >= 2:
            stall_mult = 0.90
        elif stall_bars >= 1:
            stall_mult = 0.97
        else:
            stall_mult = 1.0

        stall_tighten = stall_bars >= 3

        max_hold = self.config.max_hold_bars if self.config.max_hold_bars > 0 else 60
        hold_frac = position.hold_bars / max(max_hold, 1)
        if hold_frac >= 0.95:
            time_mult = 0.40
        elif hold_frac >= 0.90:
            time_mult = 0.50
        elif hold_frac >= 0.75:
            time_mult = 0.68
        elif hold_frac >= 0.60:
            time_mult = 0.83
        elif hold_frac >= 0.50:
            time_mult = 0.88
        elif hold_frac >= 0.40:
            time_mult = 0.93
        elif hold_frac >= 0.30:
            time_mult = 0.97
        elif hold_frac >= 0.20:
            time_mult = 0.99
        else:
            time_mult = 1.0

        compound_mult = stall_mult * time_mult

        recent_range = state.get('recent_bar_range', 0.002)

        if position.direction == 'long':
            if is_breakout:
                if recent_range > 0.004:
                    retrace_threshold = 0.27
                elif recent_range < 0.001:
                    retrace_threshold = 0.22
                else:
                    retrace_threshold = 0.25
            else:
                if recent_range > 0.004:
                    retrace_threshold = 0.40
                elif recent_range < 0.001:
                    retrace_threshold = 0.30
                else:
                    retrace_threshold = 0.35
        else:
            if is_breakout:
                if recent_range > 0.004:
                    retrace_threshold = 0.27
                elif recent_range < 0.001:
                    retrace_threshold = 0.22
                else:
                    retrace_threshold = 0.25
            else:
                if recent_range > 0.004:
                    retrace_threshold = 0.35
                elif recent_range < 0.001:
                    retrace_threshold = 0.25
                else:
                    retrace_threshold = 0.30

        min_gap = recent_range * 0.20

        if position.direction == 'long':
            peak_profit = (trailing - entry) / entry
            current_profit = (last_close - entry) / entry

            stall_lock_boost = min(0.14, stall_bars * 0.014)

            if peak_profit >= 0.120:
                lock_floor = entry * (1.0 + peak_profit * (0.97 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.100:
                lock_floor = entry * (1.0 + peak_profit * (0.95 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.080:
                lock_floor = entry * (1.0 + peak_profit * (0.93 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.060:
                lock_floor = entry * (1.0 + peak_profit * (0.91 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.050:
                lock_floor = entry * (1.0 + peak_profit * (0.88 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.040:
                lock_floor = entry * (1.0 + peak_profit * (0.92 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.035:
                lock_floor = entry * (1.0 + peak_profit * (0.90 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.030:
                lock_floor = entry * (1.0 + peak_profit * (0.89 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.025:
                lock_floor = entry * (1.0 + peak_profit * (0.81 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.020:
                lock_floor = entry * (1.0 + peak_profit * (0.79 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.015:
                lock_floor = entry * (1.0 + peak_profit * (0.70 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.010:
                lock_floor = entry * (1.0 + peak_profit * (0.58 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.008:
                lock_floor = entry * (1.0 + peak_profit * (0.51 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.006:
                lock_floor = entry * (1.0 + peak_profit * (0.42 + stall_lock_boost))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.005:
                lock_floor = entry * (1.0 + peak_profit * (0.31 + stall_lock_boost * 0.7))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.004:
                lock_floor = entry * (1.0 + peak_profit * (0.22 + stall_lock_boost * 0.5))
                lock_floor = max(position.stop_price, lock_floor)
            elif peak_profit >= 0.003:
                lock_floor = entry * (1.0 + peak_profit * (0.10 + stall_lock_boost * 0.5))
                lock_floor = max(position.stop_price, lock_floor)
            else:
                lock_floor = position.stop_price

            if is_breakout:
                if peak_profit <= 0.0001:
                    effective_stop = position.stop_price
                else:
                    if 0.004 <= peak_profit < 0.008 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.25 and not long_momentum and last_close < prev_close:
                            lock_price = entry * (1.0 + peak_profit * 0.20)
                            return max(position.stop_price, lock_price)

                    if peak_profit >= 0.012 and current_profit > 0 and consec_down >= 2:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.62 and not long_momentum:
                            lock_price = entry * (1.0 + peak_profit * 0.62)
                            return max(position.stop_price, lock_price)

                    if peak_profit >= 0.100 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.92 and not long_momentum:
                            lock_price = entry * (1.0 + peak_profit * 0.95)
                            return max(position.stop_price, lock_price)

                    if peak_profit >= 0.080 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.90 and not long_momentum:
                            lock_price = entry * (1.0 + peak_profit * 0.93)
                            return max(position.stop_price, lock_price)

                    if peak_profit >= 0.060 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.87 and not long_momentum:
                            lock_price = entry * (1.0 + peak_profit * 0.90)
                            return max(position.stop_price, lock_price)

                    if peak_profit >= 0.050 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.84 and not long_momentum:
                            lock_price = entry * (1.0 + peak_profit * 0.88)
                            return max(position.stop_price, lock_price)

                    if peak_profit >= 0.040 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.82 and not long_momentum:
                            lock_price = entry * (1.0 + peak_profit * 0.85)
                            return max(position.stop_price, lock_price)

                    if peak_profit >= 0.030 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.75 and not long_momentum:
                            lock_price = entry * (1.0 + peak_profit * 0.80)
                            return max(position.stop_price, lock_price)

                    if peak_profit >= 0.025 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.70 and not long_momentum:
                            lock_price = entry * (1.0 + peak_profit * 0.72)
                            return max(position.stop_price, lock_price)

                    if peak_profit >= 0.018 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.62 and not long_momentum:
                            lock_price = entry * (1.0 + peak_profit * 0.65)
                            return max(position.stop_price, lock_price)

                    if peak_profit >= 0.015 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.61 and not long_momentum:
                            lock_price = entry * (1.0 + peak_profit * 0.60)
                            return max(position.stop_price, lock_price)

                    if peak_profit >= 0.012 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.60 and not long_momentum:
                            lock_price = entry * (1.0 + peak_profit * 0.55)
                            return max(position.stop_price, lock_price)

                    if peak_profit > 0.003 and current_profit > 0 and not long_momentum:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < retrace_threshold:
                            lock_price = entry * (1.0 + peak_profit * 0.30)
                            return max(position.stop_price, lock_price)

                    vol_adj = 1.0
                    if recent_range > 0.005 and long_momentum:
                        vol_adj = 1.10

                    if peak_profit > 0.022:
                        poly_mult = max(0.02, 0.10 * (1.0 - min(peak_profit / 0.025, 1.0)) ** 1.1)
                    elif peak_profit > 0.020:
                        poly_mult = max(0.02, 0.12 * (1.0 - min(peak_profit / 0.025, 1.0)) ** 1.2)
                    elif peak_profit > 0.015:
                        poly_mult = max(0.03, 0.22 * (1.0 - min(peak_profit / 0.025, 1.0)) ** 1.5)
                    else:
                        poly_mult = max(0.03, 0.35 * (1.0 - min(peak_profit / 0.025, 1.0)) ** 1.8)
                    poly_mult *= 0.90 * vol_adj

                    if recent_range > 0.005 and peak_profit > 0.010 and long_momentum:
                        poly_mult *= 1.20

                    if long_momentum and peak_profit > 0.004:
                        local_stall_mult = 1.0
                        if strong_long_momentum:
                            poly_mult *= 2.0
                        else:
                            poly_mult *= 1.50
                    else:
                        local_stall_mult = stall_mult
                    local_compound_mult = local_stall_mult * time_mult

                    if initial_stop_dist < 0.001:
                        trail_price = trailing * (1 - initial_stop_dist * 0.50 * twm * local_compound_mult)
                    else:
                        trail_price = trailing * (1 - initial_stop_dist * poly_mult * twm * local_compound_mult)

                    trail_price = min(trail_price, trailing * (1.0 - min_gap))
                    effective_stop = max(position.stop_price, trail_price)
            else:
                profit_ratio = peak_profit / max(tp_dist, 1e-6)
                tight = el or fast_rev or stall_tighten

                if peak_profit > 0.004 and current_profit > 0 and not long_momentum:
                    retrace_ratio = current_profit / max(peak_profit, 1e-6)
                    if retrace_ratio < retrace_threshold:
                        lock_price = entry * (1.0 + peak_profit * 0.30)
                        return max(position.stop_price, lock_price)

                if profit_ratio >= 0.96:
                    trail_price = trailing * (1 - initial_stop_dist * 0.0010 * compound_mult * twm)
                    effective_stop = max(position.stop_price, trail_price)
                elif profit_ratio >= 0.92:
                    trail_price = trailing * (1 - initial_stop_dist * 0.0020 * compound_mult * twm)
                    effective_stop = max(position.stop_price, trail_price)
                elif profit_ratio >= 0.80:
                    trail_price = trailing * (1 - initial_stop_dist * 0.0040 * compound_mult * twm)
                    effective_stop = max(position.stop_price, trail_price)
                elif profit_ratio >= 0.72:
                    trail_price = trailing * (1 - initial_stop_dist * 0.006 * compound_mult * twm)
                    effective_stop = max(position.stop_price, trail_price)
                elif profit_ratio >= (0.60 if tight else 0.50):
                    trail_price = trailing * (1 - initial_stop_dist * 0.02 * compound_mult * twm)
                    effective_stop = max(position.stop_price, trail_price)
                elif profit_ratio >= (0.30 if tight else 0.38):
                    base_mult = 0.08 if tight else 0.06
                    trail_price = trailing * (1 - initial_stop_dist * base_mult * compound_mult * twm)
                    effective_stop = max(position.stop_price, trail_price)
                elif profit_ratio >= (0.08 if tight else 0.15):
                    effective_stop = max(position.stop_price, entry * 1.0005)
                else:
                    effective_stop = position.stop_price

            return max(effective_stop, lock_floor)

        else:  # short
            peak_profit = (entry - trailing) / entry
            current_profit = (entry - last_close) / entry

            stall_lock_boost = min(0.14, stall_bars * 0.014)

            if peak_profit >= 0.100:
                lock_floor = entry * (1.0 - peak_profit * (0.92 + stall_lock_boost))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.080:
                lock_floor = entry * (1.0 - peak_profit * (0.90 + stall_lock_boost))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.060:
                lock_floor = entry * (1.0 - peak_profit * (0.89 + stall_lock_boost))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.050:
                lock_floor = entry * (1.0 - peak_profit * (0.86 + stall_lock_boost))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.040:
                lock_floor = entry * (1.0 - peak_profit * (0.87 + stall_lock_boost))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.030:
                lock_floor = entry * (1.0 - peak_profit * (0.87 + stall_lock_boost))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.025:
                lock_floor = entry * (1.0 - peak_profit * (0.85 + stall_lock_boost))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.020:
                lock_floor = entry * (1.0 - peak_profit * (0.84 + stall_lock_boost))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.015:
                lock_floor = entry * (1.0 - peak_profit * (0.75 + stall_lock_boost))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.012:
                lock_floor = entry * (1.0 - peak_profit * (0.65 + stall_lock_boost))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.008:
                lock_floor = entry * (1.0 - peak_profit * (0.59 + stall_lock_boost))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.006:
                lock_floor = entry * (1.0 - peak_profit * (0.47 + stall_lock_boost))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.005:
                lock_floor = entry * (1.0 - peak_profit * (0.40 + stall_lock_boost * 0.7))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.004:
                lock_floor = entry * (1.0 - peak_profit * (0.33 + stall_lock_boost))
                lock_floor = min(position.stop_price, lock_floor)
            elif peak_profit >= 0.003:
                lock_floor = entry * (1.0 - peak_profit * (0.10 + stall_lock_boost * 0.5))
                lock_floor = min(position.stop_price, lock_floor)
            else:
                lock_floor = position.stop_price

            if is_breakout:
                if peak_profit <= 0.0001:
                    effective_stop = position.stop_price
                else:
                    if 0.003 <= peak_profit < 0.006 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.25 and not short_momentum:
                            lock_price = entry * (1.0 - peak_profit * 0.20)
                            return min(position.stop_price, lock_price)

                    if peak_profit >= 0.010 and current_profit > 0 and consec_up >= 2:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.62 and not short_momentum:
                            lock_price = entry * (1.0 - peak_profit * 0.60)
                            return min(position.stop_price, lock_price)

                    if peak_profit >= 0.100 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.85 and not short_momentum:
                            lock_price = entry * (1.0 - peak_profit * 0.91)
                            return min(position.stop_price, lock_price)

                    if peak_profit >= 0.080 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.83 and not short_momentum:
                            lock_price = entry * (1.0 - peak_profit * 0.89)
                            return min(position.stop_price, lock_price)

                    if peak_profit >= 0.060 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.80 and not short_momentum:
                            lock_price = entry * (1.0 - peak_profit * 0.86)
                            return min(position.stop_price, lock_price)

                    if peak_profit >= 0.050 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.77 and not short_momentum:
                            lock_price = entry * (1.0 - peak_profit * 0.82)
                            return min(position.stop_price, lock_price)

                    if peak_profit >= 0.040 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.74 and not short_momentum:
                            lock_price = entry * (1.0 - peak_profit * 0.78)
                            return min(position.stop_price, lock_price)

                    if peak_profit >= 0.030 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.70 and not short_momentum:
                            lock_price = entry * (1.0 - peak_profit * 0.74)
                            return min(position.stop_price, lock_price)

                    if peak_profit >= 0.020 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.68 and not short_momentum:
                            lock_price = entry * (1.0 - peak_profit * 0.70)
                            return min(position.stop_price, lock_price)

                    if peak_profit >= 0.015 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.62 and not short_momentum:
                            lock_price = entry * (1.0 - peak_profit * 0.60)
                            return min(position.stop_price, lock_price)

                    if peak_profit >= 0.008 and current_profit > 0:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < 0.60 and not short_momentum:
                            lock_price = entry * (1.0 - peak_profit * 0.55)
                            return min(position.stop_price, lock_price)

                    if peak_profit > 0.004 and current_profit > 0 and not short_momentum:
                        retrace_ratio = current_profit / max(peak_profit, 1e-6)
                        if retrace_ratio < retrace_threshold:
                            lock_price = entry * (1.0 - peak_profit * 0.30)
                            return min(position.stop_price, lock_price)

                    if peak_profit > 0.015:
                        curve_mult = max(0.03, 0.17 * math.exp(-peak_profit * 40))
                    elif peak_profit > 0.012:
                        curve_mult = max(0.03, 0.20 * math.exp(-peak_profit * 40))
                    else:
                        curve_mult = max(0.03, 0.28 * math.exp(-peak_profit * 50))
                    curve_mult = max(0.03, min(0.28, curve_mult))

                    if short_momentum and peak_profit > 0.004:
                        local_stall_mult = 1.0
                        if strong_short_momentum:
                            curve_mult *= 1.65
                        else:
                            curve_mult *= 1.45
                    else:
                        local_stall_mult = stall_mult
                        if strong_long_momentum and peak_profit > 0.006:
                            curve_mult *= 0.70
                    local_compound_mult = local_stall_mult * time_mult

                    if initial_stop_dist < 0.001:
                        trail_price = trailing * (1 + initial_stop_dist * 0.50 * twm * local_compound_mult)
                    else:
                        trail_price = trailing * (1 + initial_stop_dist * curve_mult * twm * local_compound_mult)

                    trail_price = max(trail_price, trailing * (1.0 + min_gap))
                    effective_stop = min(position.stop_price, trail_price)
            else:
                profit_ratio = peak_profit / max(tp_dist, 1e-6)
                tight = el or fast_rev or stall_tighten

                if peak_profit > 0.004 and current_profit > 0 and not short_momentum:
                    retrace_ratio = current_profit / max(peak_profit, 1e-6)
                    if retrace_ratio < retrace_threshold:
                        lock_price = entry * (1.0 - peak_profit * 0.30)
                        return min(position.stop_price, lock_price)

                if profit_ratio >= 0.93:
                    trail_price = trailing * (1 + initial_stop_dist * 0.0006 * compound_mult * twm)
                    effective_stop = min(position.stop_price, trail_price)
                elif profit_ratio >= 0.88:
                    trail_price = trailing * (1 + initial_stop_dist * 0.0012 * compound_mult * twm)
                    effective_stop = min(position.stop_price, trail_price)
                elif profit_ratio >= 0.75:
                    trail_price = trailing * (1 + initial_stop_dist * 0.0025 * compound_mult * twm)
                    effective_stop = min(position.stop_price, trail_price)
                elif profit_ratio >= 0.65:
                    trail_price = trailing * (1 + initial_stop_dist * 0.005 * compound_mult * twm)
                    effective_stop = min(position.stop_price, trail_price)
                elif profit_ratio >= 0.55:
                    trail_price = trailing * (1 + initial_stop_dist * 0.015 * compound_mult * twm)
                    effective_stop = min(position.stop_price, trail_price)
                elif profit_ratio >= (0.28 if tight else 0.38):
                    base_mult = 0.09 if tight else 0.07
                    trail_price = trailing * (1 + initial_stop_dist * base_mult * compound_mult * twm)
                    effective_stop = min(position.stop_price, trail_price)
                elif profit_ratio >= (0.10 if tight else 0.15):
                    effective_stop = min(position.stop_price, entry * 0.9995)
                else:
                    effective_stop = position.stop_price

            return min(effective_stop, lock_floor)

    def check_exits(self, time, bar, open_positions):
        """No-peak loss cut 0.62->0.60 (isd>0.005), 0.67->0.65 (isd 0.002-0.005); bounce 0.020+ stall>=3; bounce 0.008-0.013 stall>=3 retrace<0.36; micro-profit stall>=8."""
        from v15.validation.unified_backtester.algo_base import ExitSignal

        exits = []
        max_hold = self.config.max_hold_bars if self.config.max_hold_bars > 0 else self.config.params.get('max_hold_bars', 60)
        eval_interval = self.config.eval_interval

        for pos in open_positions:
            state = self._pos_state.get(pos.pos_id, {})

            bar_high = bar['high']
            bar_low = bar['low']
            bar_close = bar['close']
            bar_open = bar.get('open', bar_close)

            state['last_close'] = bar_close

            state.setdefault('window_high', bar_high)
            state.setdefault('window_low', bar_low)
            state['window_high'] = max(state['window_high'], bar_high)
            state['window_low'] = min(state['window_low'], bar_low)

            bar_range = (bar_high - bar_low) / max(bar_close, 1.0)
            prev_range = state.get('recent_bar_range', bar_range)
            state['recent_bar_range'] = 0.2 * bar_range + 0.8 * prev_range

            prev_ema = state.get('close_ema', bar_close)
            state['prev_ema'] = prev_ema
            state['close_ema'] = 0.35 * bar_close + 0.65 * prev_ema

            prev_close = state.get('prev_close', bar_close)
            prev_prev_close = state.get('prev_prev_close', bar_close)
            state['prev_prev_close'] = prev_close
            state['prev_close'] = bar_close

            if bar_close > prev_close:
                state['consec_up_bars'] = state.get('consec_up_bars', 0) + 1
                state['consec_down_bars'] = 0
            elif bar_close < prev_close:
                state['consec_down_bars'] = state.get('consec_down_bars', 0) + 1
                state['consec_up_bars'] = 0
            else:
                state['consec_up_bars'] = max(0, state.get('consec_up_bars', 0) - 1)
                state['consec_down_bars'] = max(0, state.get('consec_down_bars', 0) - 1)

            consec_up = state.get('consec_up_bars', 0)
            consec_down = state.get('consec_down_bars', 0)

            if 'tracked_best' not in state:
                state['tracked_best'] = pos.best_price
                state['stall_bars'] = 0
            else:
                prev_best = state['tracked_best']
                if pos.direction == 'long':
                    if pos.best_price > prev_best:
                        state['stall_bars'] = 0
                        state['tracked_best'] = pos.best_price
                    else:
                        state['stall_bars'] = state.get('stall_bars', 0) + 1
                        if consec_up >= 4:
                            state['stall_bars'] = 0
                        elif consec_up >= 3:
                            state['stall_bars'] = max(0, state['stall_bars'] - 2)
                        elif consec_up >= 2:
                            state['stall_bars'] = max(0, state['stall_bars'] - 1)
                else:
                    if pos.best_price < prev_best:
                        state['stall_bars'] = 0
                        state['tracked_best'] = pos.best_price
                    else:
                        state['stall_bars'] = state.get('stall_bars', 0) + 1
                        if consec_down >= 4:
                            state['stall_bars'] = 0
                        elif consec_down >= 3:
                            state['stall_bars'] = max(0, state['stall_bars'] - 2)
                        elif consec_down >= 2:
                            state['stall_bars'] = max(0, state['stall_bars'] - 1)

            candle_range_abs = bar_high - bar_low
            if candle_range_abs > 1e-6:
                candle_close_pct = (bar_close - bar_low) / candle_range_abs
            else:
                candle_close_pct = 0.5

            current_bar_range_frac = (bar_high - bar_low) / max(bar_close, 1.0)

            state.setdefault('exit_bar_count', 0)
            state['exit_bar_count'] += 1
            if state['exit_bar_count'] < eval_interval:
                continue
            state['exit_bar_count'] = 0

            high = state['window_high']
            low = state['window_low']
            close = bar['close']
            close_ema = state['close_ema']
            ema_declining = close_ema < state.get('prev_ema', close_ema)
            ema_rising = close_ema > state.get('prev_ema', close_ema)
            state['window_high'] = bar['high']
            state['window_low'] = bar['low']

            is_breakout = pos.signal_type == 'break'
            ou_hl = state.get('ou_half_life', 5.0)
            el = state.get('el_flagged', False)

            max_hold_bars = self.config.max_hold_bars if self.config.max_hold_bars > 0 else 60
            hold_frac = pos.hold_bars / max(max_hold_bars, 1)

            effective_stop = self.get_effective_stop(pos)

            stall_bars = state.get('stall_bars', 0)

            if pos.direction == 'long':
                if low <= effective_stop:
                    reason = 'stop' if effective_stop == pos.stop_price else 'trail'
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=effective_stop, reason=reason))
                    continue
                if high >= pos.tp_price:
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.tp_price, reason='tp'))
                    continue
                hold_5m = pos.hold_bars

                if hold_frac >= 0.85 and abs(close - pos.entry_price) / max(pos.entry_price, 1e-6) < 0.002:
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                    continue

                if stall_bars >= 20 and (pos.best_price - pos.entry_price) / max(pos.entry_price, 1e-6) < 0.004:
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                    continue

                _isd_long = abs(pos.stop_price - pos.entry_price) / max(pos.entry_price, 1e-6)
                _peak_long = (pos.best_price - pos.entry_price) / max(pos.entry_price, 1e-6)
                _loss_long = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                # Tightened: 0.62->0.60 and 0.67->0.65 to cut no-peak losers earlier
                if _isd_long > 0.005:
                    if _loss_long > _isd_long * 0.60 and _peak_long < 0.003:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                        continue
                elif _isd_long > 0.002:
                    if _loss_long > _isd_long * 0.65 and _peak_long < 0.002:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                        continue

                if is_breakout:
                    peak_profit = (pos.best_price - pos.entry_price) / max(pos.entry_price, 1e-6)
                    long_momentum = bar_close > prev_close > prev_prev_close
                    recent_range = state.get('recent_bar_range', 0.002)
                    strong_long_momentum = consec_up >= 3

                    if hold_5m >= 2 and close < pos.entry_price * 0.9970 and peak_profit < 0.001:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if hold_5m >= 3 and close < pos.entry_price * 0.998 and peak_profit < 0.002:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if peak_profit > 0.015 and close > pos.entry_price * 1.003:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        prev_bar_profit = (prev_close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if prev_bar_profit > 0 and current_profit > 0:
                            single_bar_drop = (prev_bar_profit - current_profit) / max(prev_bar_profit, 1e-6)
                            if single_bar_drop > 0.42 and not long_momentum:
                                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                                continue

                    if peak_profit > 0.025 and current_bar_range_frac > 0.005 and candle_close_pct < 0.20 and close > pos.entry_price * 1.010 and not long_momentum:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if peak_profit > 0.030 and consec_down >= 2 and close > pos.entry_price * 1.010:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.60:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if consec_down >= 3 and peak_profit > 0.010 and close > pos.entry_price * 1.004:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0.003:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if consec_down >= 4 and peak_profit > 0.008 and close > pos.entry_price * 1.004:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0.003:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if consec_down >= 2 and peak_profit > 0.018 and close > pos.entry_price * 1.008:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.55:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if consec_down >= 2 and peak_profit > 0.012 and close > pos.entry_price * 1.006:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.61:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if recent_range < 0.0007 and hold_5m >= 8 and peak_profit > 0.005:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if 0.001 < current_profit < peak_profit * 0.60:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if hold_5m >= 5 and pos.best_price < pos.entry_price * 1.002 and close <= pos.entry_price * 1.0005:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if peak_profit >= 0.006 and candle_close_pct < 0.25 and not long_momentum:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0.002 and close > pos.entry_price * 1.002:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if peak_profit > 0.040:
                        _long_stall_thr = 4
                    elif peak_profit > 0.030:
                        _long_stall_thr = 3
                    elif peak_profit > 0.015:
                        _long_stall_thr = 2
                    elif peak_profit > 0.009:
                        _long_stall_thr = 3
                    elif peak_profit > 0.006:
                        _long_stall_thr = 4
                    else:
                        _long_stall_thr = 5
                    if stall_bars >= _long_stall_thr and peak_profit > 0.007 and close > pos.entry_price * 1.003 and not long_momentum and candle_close_pct < 0.45:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                        continue

                    if stall_bars >= 10 and abs(close - pos.entry_price) / max(pos.entry_price, 1) < 0.003:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if peak_profit > 0.004 and close < close_ema * 0.9997 and close > pos.entry_price * 1.001 and not long_momentum:
                        if not (recent_range > 0.005 and long_momentum):
                            if not strong_long_momentum:
                                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                                continue

                    if peak_profit > 0.008 and bar_close < prev_close < prev_prev_close:
                        if close > pos.entry_price * 1.002:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if peak_profit >= 0.006 and not long_momentum:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.30:
                            if bar_close < prev_close and close > pos.entry_price * 1.001:
                                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                                continue

                    if peak_profit >= 0.010:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.55:
                            if bar_close < prev_close < prev_prev_close:
                                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                                continue
                else:
                    peak_profit = (pos.best_price - pos.entry_price) / max(pos.entry_price, 1e-6)
                    long_momentum = bar_close > prev_close > prev_prev_close

                    quick_fail_bars = max(2, int(ou_hl * 0.27))
                    if hold_5m >= quick_fail_bars and pos.best_price <= pos.entry_price * 1.0003 and close <= pos.entry_price:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if hold_5m >= 2 and close < pos.entry_price * 0.9965 and peak_profit < 0.001:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if current_bar_range_frac > 0.003 and candle_close_pct < 0.15 and close < pos.entry_price and peak_profit < 0.005 and hold_5m >= 1:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if consec_down >= 3 and close < pos.entry_price * 0.997 and hold_frac <= 0.25 and peak_profit < 0.002:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if consec_down >= 3 and close < pos.entry_price * 0.999 and hold_frac > 0.30 and peak_profit < 0.005:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if hold_frac > 0.40 and close < pos.entry_price * 0.997 and peak_profit < 0.003:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if hold_frac > 0.55 and close < pos.entry_price * 0.998 and peak_profit < 0.004:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if 0.003 <= peak_profit <= 0.012 and ema_declining and close < pos.entry_price * 1.0010 and close > pos.entry_price * 0.9995 and not long_momentum and hold_5m >= 3:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                        continue

                    if peak_profit > 0.015 and close > pos.entry_price * 1.005:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        prev_bar_profit = (prev_close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if prev_bar_profit > 0 and current_profit > 0:
                            single_bar_drop = (prev_bar_profit - current_profit) / max(prev_bar_profit, 1e-6)
                            if single_bar_drop > 0.45 and not long_momentum:
                                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                                continue

                    if peak_profit > 0.015 and consec_down >= 2 and close > pos.entry_price * 1.005:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.50:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    # Tightened 0.020+ tier: stall>=3 retrace<0.40 (was stall>=4 retrace<0.42)
                    if 0.020 <= peak_profit and stall_bars >= 3 and close > pos.entry_price * 1.004:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.40 and not long_momentum:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if 0.013 <= peak_profit < 0.020 and stall_bars >= 3 and close > pos.entry_price * 1.004:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.40 and not long_momentum:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    # Tightened 0.008-0.013 tier: stall>=3 retrace<0.36 (was stall>=4 retrace<0.38)
                    if 0.008 <= peak_profit < 0.013 and stall_bars >= 3 and close > pos.entry_price * 1.002:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.36 and not long_momentum:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if 0.005 <= peak_profit < 0.008 and stall_bars >= 5 and close > pos.entry_price * 1.001:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.35 and not long_momentum:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    # Micro-profit stall tier: reduced stall>=9->stall>=8
                    if 0.003 <= peak_profit < 0.005 and stall_bars >= 8 and close > pos.entry_price * 1.0005:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.28 and not long_momentum:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if peak_profit > 0.006 and consec_down >= 2 and close > pos.entry_price * 1.001:
                        current_profit = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.37:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if peak_profit > 0.008 and close < close_ema and not long_momentum and bar_close < prev_close < prev_prev_close:
                        if close > pos.entry_price * 1.003:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if peak_profit > 0.006 and close < close_ema and close > pos.entry_price * 1.002 and not long_momentum:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                        continue

                    if el and close < pos.entry_price:
                        el_timeout = max(3, int(ou_hl * 0.6))
                        if hold_5m >= el_timeout:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                            continue

                    drift_ratio = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                    if close < pos.entry_price:
                        near_be_timeout = max(4, int(ou_hl * 1.3))
                    elif -0.0010 <= drift_ratio <= 0.0005:
                        near_be_timeout = max(4, int(ou_hl * 1.6))
                    else:
                        near_be_timeout = max(4, int(ou_hl * 2.3))
                    if stall_bars >= 5:
                        near_be_timeout = max(3, int(near_be_timeout * 0.80))
                    if hold_5m >= near_be_timeout and abs(close - pos.entry_price) / max(pos.entry_price, 1e-6) < 0.0010:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if pos.entry_price * 1.0005 <= close < pos.entry_price * 1.003 and peak_profit < 0.010:
                        weak_timeout = max(5, int(ou_hl * 2.5))
                        if hold_5m >= weak_timeout:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                            continue

                    if close < pos.entry_price * 0.985:
                        early_timeout = max(1, int(ou_hl * 0.08))
                    elif close < pos.entry_price * 0.990:
                        early_timeout = max(1, int(ou_hl * 0.15))
                    elif close < pos.entry_price * 0.993:
                        early_timeout = max(2, int(ou_hl * 0.22))
                    elif close < pos.entry_price * 0.995:
                        early_timeout = max(2, int(ou_hl * 0.30))
                    else:
                        early_timeout = max(3, int(ou_hl * 0.43))
                    if hold_5m >= early_timeout and close < pos.entry_price:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if close >= pos.entry_price:
                        if close > pos.entry_price * 1.010:
                            final_timeout = max(8, int(ou_hl * 5.5))
                        elif close > pos.entry_price * 1.005:
                            final_timeout = max(8, int(ou_hl * 4.7))
                        else:
                            final_timeout = max(8, int(ou_hl * 4.5))
                    else:
                        final_timeout = max(6, int(ou_hl * 3.0))
                    if hold_5m >= final_timeout:
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

                if hold_frac >= 0.85 and abs(close - pos.entry_price) / max(pos.entry_price, 1e-6) < 0.002:
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                    continue

                if stall_bars >= 20 and (pos.entry_price - pos.best_price) / max(pos.entry_price, 1e-6) < 0.004:
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                    continue

                _isd_short = abs(pos.stop_price - pos.entry_price) / max(pos.entry_price, 1e-6)
                _peak_short = (pos.entry_price - pos.best_price) / max(pos.entry_price, 1e-6)
                _loss_short = (close - pos.entry_price) / max(pos.entry_price, 1e-6)
                # Tightened: 0.62->0.60 and 0.67->0.65 to cut no-peak losers earlier
                if _isd_short > 0.005:
                    if _loss_short > _isd_short * 0.60 and _peak_short < 0.003:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                        continue
                elif _isd_short > 0.002:
                    if _loss_short > _isd_short * 0.65 and _peak_short < 0.002:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                        continue

                if is_breakout:
                    peak_profit = (pos.entry_price - pos.best_price) / max(pos.entry_price, 1e-6)
                    short_momentum = bar_close < prev_close < prev_prev_close
                    strong_short_momentum = consec_down >= 3

                    if hold_5m >= 2 and close > pos.entry_price * 1.0030 and peak_profit < 0.001:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if hold_5m >= 3 and close > pos.entry_price * 1.002 and peak_profit < 0.002:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if peak_profit > 0.015 and close < pos.entry_price * 0.997:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        prev_bar_profit = (pos.entry_price - prev_close) / max(pos.entry_price, 1e-6)
                        if prev_bar_profit > 0 and current_profit > 0:
                            single_bar_drop = (prev_bar_profit - current_profit) / max(prev_bar_profit, 1e-6)
                            if single_bar_drop > 0.42 and not short_momentum:
                                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                                continue

                    if peak_profit > 0.020 and current_bar_range_frac > 0.005 and candle_close_pct > 0.80 and close < pos.entry_price * 0.990 and not short_momentum:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if peak_profit > 0.030 and consec_up >= 2 and close < pos.entry_price * 0.990:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.60:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if consec_up >= 3 and peak_profit > 0.008 and close < pos.entry_price * 0.996:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0.003:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if consec_up >= 2 and peak_profit > 0.018 and close < pos.entry_price * 0.992:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.56:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if consec_up >= 2 and peak_profit > 0.010 and close < pos.entry_price * 0.995:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.62:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    recent_range = state.get('recent_bar_range', 0.002)
                    if recent_range < 0.0007 and hold_5m >= 8 and peak_profit > 0.005:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if 0.001 < current_profit < peak_profit * 0.60:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if hold_5m >= 4 and pos.best_price > pos.entry_price * 0.998 and close >= pos.entry_price * 0.9995:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if peak_profit >= 0.006 and candle_close_pct > 0.75 and not short_momentum:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0.002 and close < pos.entry_price * 0.998:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if peak_profit > 0.025:
                        stall_threshold = 3
                    elif peak_profit > 0.012:
                        stall_threshold = 2
                    elif peak_profit > 0.010:
                        stall_threshold = 2
                    elif peak_profit > 0.008:
                        stall_threshold = 3
                    elif peak_profit > 0.006:
                        stall_threshold = 4
                    else:
                        stall_threshold = 5
                    if stall_bars >= stall_threshold and peak_profit > 0.006 and close < pos.entry_price * 0.998 and not short_momentum:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                        continue

                    if stall_bars >= 9 and abs(close - pos.entry_price) / max(pos.entry_price, 1) < 0.003:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if peak_profit >= 0.008:
                        prev_ema_val = state.get('prev_close_ema', close_ema)
                        if close > close_ema and prev_close <= prev_ema_val and bar_close > prev_close > prev_prev_close:
                            current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                            if current_profit > 0 and close < pos.entry_price * 0.997:
                                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                                state['prev_close_ema'] = close_ema
                                continue
                    state['prev_close_ema'] = close_ema

                    if peak_profit > 0.003 and close > close_ema * 1.0003 and close < pos.entry_price * 0.999 and not short_momentum:
                        if not strong_short_momentum:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if peak_profit > 0.006 and bar_close > prev_close > prev_prev_close:
                        if close < pos.entry_price * 0.998:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if peak_profit >= 0.006:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.50:
                            if bar_close > prev_close > prev_prev_close:
                                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                                continue
                else:
                    peak_profit = (pos.entry_price - pos.best_price) / max(pos.entry_price, 1e-6)
                    short_momentum = bar_close < prev_close < prev_prev_close
                    strong_short_momentum = consec_down >= 3

                    quick_fail_bars = max(2, int(ou_hl * 0.25))
                    if hold_5m >= quick_fail_bars and pos.best_price >= pos.entry_price * 0.9997 and close >= pos.entry_price:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if hold_5m >= 2 and close > pos.entry_price * 1.0035 and peak_profit < 0.001:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if current_bar_range_frac > 0.003 and candle_close_pct > 0.85 and close > pos.entry_price and peak_profit < 0.005 and hold_5m >= 1:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if consec_up >= 2 and close > pos.entry_price * 1.002 and peak_profit < 0.003 and hold_5m >= 2:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if consec_up >= 3 and close > pos.entry_price * 1.003 and hold_frac <= 0.25 and peak_profit < 0.002:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if consec_up >= 3 and close > pos.entry_price * 1.001 and hold_frac > 0.30 and peak_profit < 0.005:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if hold_frac > 0.40 and close > pos.entry_price * 1.003 and peak_profit < 0.003:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if hold_frac > 0.55 and close > pos.entry_price * 1.002 and peak_profit < 0.004:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if 0.003 <= peak_profit <= 0.012 and ema_rising and close > pos.entry_price * 0.9990 and close < pos.entry_price * 1.0005 and not short_momentum and hold_5m >= 3:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                        continue

                    if peak_profit > 0.015 and close < pos.entry_price * 0.994:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        prev_bar_profit = (pos.entry_price - prev_close) / max(pos.entry_price, 1e-6)
                        if prev_bar_profit > 0 and current_profit > 0:
                            single_bar_drop = (prev_bar_profit - current_profit) / max(prev_bar_profit, 1e-6)
                            if single_bar_drop > 0.45 and not short_momentum:
                                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                                continue

                    if peak_profit > 0.015 and consec_up >= 2 and close < pos.entry_price * 0.994:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.50:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    # Tightened 0.020+ tier: stall>=3 retrace<0.40 (was stall>=4 retrace<0.42)
                    if 0.020 <= peak_profit and stall_bars >= 3 and close < pos.entry_price * 0.996:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.40 and not short_momentum:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if 0.013 <= peak_profit < 0.020 and stall_bars >= 3 and close < pos.entry_price * 0.996:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.40 and not short_momentum:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    # Tightened 0.008-0.013 tier: stall>=3 retrace<0.36 (was stall>=4 retrace<0.38)
                    if 0.008 <= peak_profit < 0.013 and stall_bars >= 3 and close < pos.entry_price * 0.998:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.36 and not short_momentum:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if 0.005 <= peak_profit < 0.008 and stall_bars >= 5 and close < pos.entry_price * 0.999:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.35 and not short_momentum:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    # Micro-profit stall tier: reduced stall>=9->stall>=8
                    if 0.003 <= peak_profit < 0.005 and stall_bars >= 8 and close < pos.entry_price * 0.9995:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.28 and not short_momentum:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if peak_profit > 0.006 and consec_up >= 2 and close < pos.entry_price * 0.999:
                        current_profit = (pos.entry_price - close) / max(pos.entry_price, 1e-6)
                        if current_profit > 0 and (current_profit / max(peak_profit, 1e-6)) < 0.37:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                            continue

                    if peak_profit > 0.006 and close > close_ema and close < pos.entry_price * 0.998 and not short_momentum:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='trail'))
                        continue

                    if el and close > pos.entry_price:
                        el_timeout = max(2, int(ou_hl * 0.5))
                        if hold_5m >= el_timeout:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                            continue

                    near_be_timeout = max(3, int(ou_hl * 0.85))
                    if hold_5m >= near_be_timeout and abs(close - pos.entry_price) / max(pos.entry_price, 1e-6) < 0.0012:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if pos.entry_price * 0.998 < close <= pos.entry_price * 0.9995 and peak_profit < 0.008:
                        weak_short_timeout = max(4, int(ou_hl * 2.0))
                        if hold_5m >= weak_short_timeout:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                            continue

                    if close > pos.entry_price * 1.015:
                        extreme_adverse_timeout = max(1, int(ou_hl * 0.07))
                        if hold_5m >= extreme_adverse_timeout:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                            continue

                    if close > pos.entry_price * 1.010:
                        ultra_adverse_timeout = max(1, int(ou_hl * 0.10))
                        if hold_5m >= ultra_adverse_timeout:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                            continue

                    early_timeout = max(2, int(ou_hl * 0.11))
                    if hold_5m >= early_timeout and close > pos.entry_price and not strong_short_momentum:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

                    if close > pos.entry_price and bar_close > prev_close and peak_profit < 0.003:
                        failed_short_timeout = max(2, int(ou_hl * 0.12))
                        if hold_5m >= failed_short_timeout:
                            exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                            continue

                    if close <= pos.entry_price:
                        if close < pos.entry_price * 0.990:
                            final_timeout = max(10, int(ou_hl * 6.8))
                        else:
                            final_timeout = max(10, int(ou_hl * 4.5))
                    else:
                        final_timeout = max(6, int(ou_hl * 2.5))
                    if hold_5m >= final_timeout:
                        exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                        continue

            if pos.hold_bars >= max_hold:
                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='timeout'))

        return exits
