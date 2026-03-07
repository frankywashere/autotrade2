"""
Surfer ML Algorithm — Plug-in for unified backtester.

Replicates surfer_backtest.py: Channel Surfer physics signal + GBT soft gate,
profit-tier trailing stops (breakout 3-tier + bounce ratio), EL/ER sub-models.

Signal: prepare_multi_tf_analysis() every eval_interval bars
Trail: Profit-tier system (not exponential) with twm/el/fast_rev modifiers
Sizing: Flat $100K (c16) or risk-based
"""

import datetime as dt
import time as _time_mod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..algo_base import AlgoBase, AlgoConfig, Signal, ExitSignal, CostModel
from ..data_provider import DataProvider
from ..portfolio import Position


DEFAULT_SURFER_ML_CONFIG = AlgoConfig(
    algo_id='surfer-ml',
    initial_equity=100_000.0,
    max_equity_per_trade=100_000.0,
    max_positions=2,             # 1 long + 1 short (matching surfer_backtest)
    primary_tf='5min',
    eval_interval=3,            # Every 3 bars = 15 min (matching surfer_backtest)
    exit_check_tf='5min',       # Exit checking on 5-min bars
    cost_model=CostModel(
        slippage_pct=0.0,       # No slippage (matching surfer_backtest non-realistic mode)
        commission_per_share=0.0,  # No commission (matching non-realistic)
    ),
    params={
        'flat_sizing': True,     # c16: flat $100K
        'min_confidence': 0.01,
        'max_hold_bars': 60,     # 5 hours (5-min bars)
        'ou_half_life': 5.0,     # Default OU half-life for bounce timeout
        'stop_pct': 0.015,       # Default stop (overridden by signal)
        'tp_pct': 0.012,         # Default TP (overridden by signal)
        'ml_model_dir': None,    # Path to surfer_models/ directory
        'atr_period': 14,        # ATR lookback period (5-min bars)
        'breakout_stop_mult': 0.05,  # Ultra-tight breakout stop (5% of suggested)
    },
)


class SurferMLAlgo(AlgoBase):
    """Surfer ML algorithm — physics signal + ML gating + profit-tier trail.

    Signal generation uses prepare_multi_tf_analysis() at 5-min intervals.
    Exit logic uses the profit-tier trailing stop from surfer_backtest.py.
    """

    def __init__(self, config: AlgoConfig = None, data: DataProvider = None):
        super().__init__(config or DEFAULT_SURFER_ML_CONFIG, data)

        # Load ML models if available
        self._gbt_model = None
        self._el_model = None
        self._er_model = None
        self._fast_rev_model = None
        self._load_models()

        # Track positions' internal state (for profit-tier trail)
        self._pos_state: Dict[str, dict] = {}  # pos_id -> trail state

    def _load_models(self):
        """Load GBT + sub-models if model directory provided."""
        import pickle
        from pathlib import Path

        model_dir = self.config.params.get('ml_model_dir')
        if not model_dir:
            # Try default location
            for candidate in ['surfer_models', '../surfer_models',
                              'C:/AI/x14/surfer_models']:
                if Path(candidate).is_dir():
                    model_dir = candidate
                    break

        if not model_dir or not Path(model_dir).is_dir():
            print("  No ML models found, running physics-only mode")
            return

        model_dir = Path(model_dir)
        for name, attr in [
            ('gbt_model.pkl', '_gbt_model'),
            ('extreme_loser_model.pkl', '_el_model'),
            ('extended_run_model.pkl', '_er_model'),
            ('momentum_reversal_model.pkl', '_fast_rev_model'),
        ]:
            path = model_dir / name
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        setattr(self, attr, pickle.load(f))
                except Exception as e:
                    print(f"  Warning: Failed to load {name}: {e}")

        loaded = sum(1 for m in [self._gbt_model, self._el_model,
                                  self._er_model, self._fast_rev_model] if m is not None)
        print(f"  Loaded {loaded}/4 ML models from {model_dir}")

    def warmup_bars(self) -> int:
        return 300  # Need enough history for channel detection

    def on_bar(self, time: pd.Timestamp, bar: dict,
               open_positions: list) -> List[Signal]:
        """Run Channel Surfer analysis, optionally gate with GBT model.

        Replicates the original surfer_backtest signal generation path:
        - 100-bar 5-min lookback for channel detection
        - Higher TFs: completed bars only, tail(100)
        - Direct analyze_channels() call (not prepare_multi_tf_analysis)
        """
        try:
            from v15.core.channel import detect_channels_multi_window, select_best_channel
            from v15.core.channel_surfer import analyze_channels, TF_WINDOWS
        except ImportError:
            return []

        # Anti-pyramid: skip if already have position in same direction
        existing_dirs = {p.direction for p in open_positions}
        existing_types = {getattr(p, 'signal_type', '') for p in open_positions}

        # Get 5-min data: last 100 bars (matching original line 1443)
        df5 = self.data.get_bars('5min', time)
        if len(df5) < 20:
            return []
        df_slice = df5.tail(100)

        # Detect channels on 5-min slice (matching original lines 1450-1457)
        try:
            multi = detect_channels_multi_window(df_slice, windows=[10, 15, 20, 30, 40])
            best_ch, _ = select_best_channel(multi)
        except Exception:
            return []

        if best_ch is None or not best_ch.valid:
            return []

        # Build multi-TF channel dict (matching original lines 1460-1506)
        slice_closes = df_slice['close'].values
        channels_by_tf = {'5min': best_ch}
        prices_by_tf = {'5min': slice_closes}
        current_prices = {'5min': float(slice_closes[-1])}
        volumes_dict = {}
        if 'volume' in df_slice.columns:
            volumes_dict['5min'] = df_slice['volume'].values

        # Add higher TF channels (matching original lines 1477-1506)
        _TF_PERIOD = {
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            'daily': pd.Timedelta(days=1),
        }
        for tf_label in ('1h', '4h', 'daily'):
            try:
                tf_df = self.data.get_bars(tf_label, time)
            except (ValueError, KeyError):
                continue
            if len(tf_df) == 0:
                continue
            # Only include completed bars
            tf_period = _TF_PERIOD.get(tf_label, pd.Timedelta(hours=1))
            tf_available = tf_df[tf_df.index + tf_period <= time]
            tf_recent = tf_available.tail(100)
            if len(tf_recent) < 30:
                continue
            tf_windows = TF_WINDOWS.get(tf_label, [20, 30, 40])
            try:
                tf_multi = detect_channels_multi_window(tf_recent, windows=tf_windows)
                tf_ch, _ = select_best_channel(tf_multi)
                if tf_ch and tf_ch.valid:
                    channels_by_tf[tf_label] = tf_ch
                    prices_by_tf[tf_label] = tf_recent['close'].values
                    current_prices[tf_label] = float(tf_recent['close'].iloc[-1])
                    if 'volume' in tf_recent.columns:
                        volumes_dict[tf_label] = tf_recent['volume'].values
            except Exception:
                continue

        try:
            analysis = analyze_channels(
                channels_by_tf, prices_by_tf, current_prices,
                volumes_by_tf=volumes_dict if volumes_dict else None,
            )
        except Exception:
            return []

        sig = analysis.signal
        if sig.action not in ('BUY', 'SELL'):
            return []

        conf = sig.confidence
        if conf < self.config.params.get('min_confidence', 0.01):
            return []

        # Anti-pyramid: no same-direction or same-type positions
        direction = 'long' if sig.action == 'BUY' else 'short'
        if direction in existing_dirs:
            return []
        signal_type = sig.signal_type or 'bounce'
        if signal_type in existing_types:
            return []

        # Base stop/TP from signal
        stop_pct = sig.suggested_stop_pct or self.config.params['stop_pct']
        tp_pct = sig.suggested_tp_pct or self.config.params['tp_pct']

        # ATR-adjusted stops (matching surfer_backtest lines 2199-2208)
        atr_val = self._compute_current_atr(time)
        entry_price = bar['close']
        if atr_val > 0 and entry_price > 0:
            if signal_type == 'bounce':
                atr_floor = (0.5 * atr_val) / entry_price
                atr_cap = (1.5 * atr_val) / entry_price
            else:
                atr_floor = (1.5 * atr_val) / entry_price
                atr_cap = (3.0 * atr_val) / entry_price
            stop_pct = np.clip(stop_pct, atr_floor, atr_cap)

        # Ultra-tight breakout stops (surfer_backtest line 2227)
        if signal_type == 'break':
            stop_pct *= self.config.params.get('breakout_stop_mult', 0.05)

        # TP widening for high-confidence bounces (surfer_backtest line 2244)
        if signal_type == 'bounce' and conf > 0.65:
            tp_pct *= 1.30

        # EL/ER predictions (stored in metadata for exit logic)
        el_flagged = False
        twm = 1.0
        fast_rev = False

        # Extended Run predictor
        if self._er_model is not None:
            # TODO: extract features for ER prediction
            pass

        # Extreme Loser detector
        if self._el_model is not None:
            # TODO: extract features for EL prediction
            pass

        return [Signal(
            algo_id=self.config.algo_id,
            direction=direction,
            price=bar['close'],
            confidence=conf,
            stop_pct=stop_pct,
            tp_pct=tp_pct,
            signal_type=signal_type,
            metadata={
                'el_flagged': el_flagged,
                'trail_width_mult': twm,
                'fast_reversion': fast_rev,
                'ou_half_life': self.config.params.get('ou_half_life', 5.0),
                # Signal bar high/low for seeding first exit window
                # (original includes signal bar in first eval window)
                'signal_bar_high': bar['high'],
                'signal_bar_low': bar['low'],
            },
        )]

    def _compute_current_atr(self, time: pd.Timestamp) -> float:
        """Compute current ATR from 5-min bars up to given time."""
        bars = self.data.get_bars('5min', time)
        period = self.config.params.get('atr_period', 14)
        if len(bars) < period + 1:
            return 0.0
        recent = bars.tail(period + 1)
        highs = recent['high'].values
        lows = recent['low'].values
        closes = recent['close'].values
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        return float(np.mean(tr))

    def on_position_opened(self, position: 'Position'):
        """Initialize profit-tier trail state for new position.

        Seeds the first exit window with the signal bar's high/low to match
        the original's 4-bar overlapping window (signal bar + 3 eval bars).
        """
        # Seed window with signal bar data (original includes signal bar in first window)
        sig_high = position.metadata.get('signal_bar_high', position.entry_price)
        sig_low = position.metadata.get('signal_bar_low', position.entry_price)
        self._pos_state[position.pos_id] = {
            'trailing_stop': position.entry_price,
            'el_flagged': position.metadata.get('el_flagged', False),
            'trail_width_mult': position.metadata.get('trail_width_mult', 1.0),
            'fast_reversion': position.metadata.get('fast_reversion', False),
            'ou_half_life': position.metadata.get('ou_half_life', 5.0),
            # Pre-seed window with signal bar (the bar before fill)
            'window_high': sig_high,
            'window_low': sig_low,
        }

    def check_exits(self, time: pd.Timestamp, bar: dict,
                    open_positions: list) -> List[ExitSignal]:
        """Profit-tier trailing stop system (matching surfer_backtest).

        The original surfer_backtest only checks exits every eval_interval bars
        (every 3 bars = 15 min), using the window high/low across those bars.
        We accumulate high/low across bars and only evaluate on eval boundaries.
        """
        exits = []
        max_hold = self.config.params.get('max_hold_bars', 60)  # In 5-min bars
        eval_interval = self.config.eval_interval  # 3
        # hold_bars counts 1-min bars; convert to 5-min equivalent
        _1m_to_5m = 5

        for pos in open_positions:
            state = self._pos_state.get(pos.pos_id, {})

            # Accumulate window high/low across bars (matching original's window_high/window_low)
            bar_high = bar['high']
            bar_low = bar['low']
            state.setdefault('window_high', bar_high)
            state.setdefault('window_low', bar_low)
            state['window_high'] = max(state['window_high'], bar_high)
            state['window_low'] = min(state['window_low'], bar_low)

            # Only evaluate exit every eval_interval bars (matching original's eval loop)
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

            entry = pos.entry_price
            is_breakout = pos.signal_type == 'break'
            tp_dist = abs(pos.tp_price - entry) / entry if entry > 0 else 0.01
            initial_stop_dist = abs(pos.stop_price - entry) / entry if entry > 0 else 0.01

            twm = state.get('trail_width_mult', 1.0)
            el = state.get('el_flagged', False)
            fast_rev = state.get('fast_reversion', False) and not is_breakout
            ou_hl = state.get('ou_half_life', 5.0)
            trailing = state.get('trailing_stop', entry)

            if pos.direction == 'long':
                # Update trailing stop (best price)
                if high > trailing:
                    trailing = high
                    state['trailing_stop'] = trailing

                if is_breakout:
                    profit_from_best = (trailing - entry) / entry
                    if initial_stop_dist < 0.001 and profit_from_best > 0.0001:
                        trail_price = trailing * (1 - initial_stop_dist * 0.50 * twm)
                        effective_stop = max(pos.stop_price, trail_price)
                    elif profit_from_best > 0.015:
                        trail_price = trailing * (1 - initial_stop_dist * 0.01 * twm)
                        effective_stop = max(pos.stop_price, trail_price)
                    elif profit_from_best > 0.008:
                        trail_price = trailing * (1 - initial_stop_dist * 0.02 * twm)
                        effective_stop = max(pos.stop_price, trail_price)
                    else:
                        tier3_thresh = 0.002 if el else 0.0008
                        trail_mult = 0.20 if el else 0.01
                        if profit_from_best > tier3_thresh:
                            trail_price = trailing * (1 - initial_stop_dist * trail_mult * twm)
                            effective_stop = max(pos.stop_price, trail_price)
                        else:
                            effective_stop = pos.stop_price
                else:
                    # Bounce: ratio-based tiers
                    profit_from_entry = (trailing - entry) / entry
                    profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
                    tight = el or fast_rev

                    if profit_ratio >= 0.80:
                        trail_price = trailing * (1 - initial_stop_dist * 0.005 * twm)
                        effective_stop = max(pos.stop_price, trail_price)
                    elif profit_ratio >= (0.60 if tight else 0.55):
                        trail_price = trailing * (1 - initial_stop_dist * 0.02 * twm)
                        effective_stop = max(pos.stop_price, trail_price)
                    elif profit_ratio >= (0.30 if tight else 0.40):
                        mult = 0.08 if tight else 0.06
                        trail_price = trailing * (1 - initial_stop_dist * mult * twm)
                        effective_stop = max(pos.stop_price, trail_price)
                    elif profit_ratio >= (0.10 if tight else 0.15):
                        effective_stop = max(pos.stop_price, entry * 1.0005)
                    else:
                        effective_stop = pos.stop_price

                # Check exit conditions
                if low <= effective_stop:
                    reason = 'stop' if effective_stop == pos.stop_price else 'trail'
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=effective_stop, reason=reason))
                    continue
                if high >= pos.tp_price:
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.tp_price, reason='tp'))
                    continue
                hold_5m = pos.hold_bars // _1m_to_5m
                if not is_breakout and hold_5m >= max(6, int(ou_hl * 3)):
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                    continue

            else:  # short
                if trailing == 0 or low < trailing:
                    trailing = low
                    state['trailing_stop'] = trailing

                if is_breakout:
                    profit_from_best = (entry - trailing) / entry
                    if initial_stop_dist < 0.001 and profit_from_best > 0.0001:
                        trail_price = trailing * (1 + initial_stop_dist * 0.50 * twm)
                        effective_stop = min(pos.stop_price, trail_price)
                    elif profit_from_best > 0.015:
                        trail_price = trailing * (1 + initial_stop_dist * 0.01 * twm)
                        effective_stop = min(pos.stop_price, trail_price)
                    elif profit_from_best > 0.008:
                        trail_price = trailing * (1 + initial_stop_dist * 0.02 * twm)
                        effective_stop = min(pos.stop_price, trail_price)
                    else:
                        tier3_thresh = 0.002 if el else 0.0003
                        trail_mult = 0.20 if el else 0.01
                        if profit_from_best > tier3_thresh:
                            trail_price = trailing * (1 + initial_stop_dist * trail_mult * twm)
                            effective_stop = min(pos.stop_price, trail_price)
                        else:
                            effective_stop = pos.stop_price
                else:
                    profit_from_entry = (entry - trailing) / entry
                    profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
                    tight = el or fast_rev

                    if profit_ratio >= 0.80:
                        trail_price = trailing * (1 + initial_stop_dist * 0.005 * twm)
                        effective_stop = min(pos.stop_price, trail_price)
                    elif profit_ratio >= (0.60 if tight else 0.55):
                        trail_price = trailing * (1 + initial_stop_dist * 0.02 * twm)
                        effective_stop = min(pos.stop_price, trail_price)
                    elif profit_ratio >= (0.30 if tight else 0.40):
                        mult = 0.08 if tight else 0.06
                        trail_price = trailing * (1 + initial_stop_dist * mult * twm)
                        effective_stop = min(pos.stop_price, trail_price)
                    elif profit_ratio >= (0.10 if tight else 0.15):
                        effective_stop = min(pos.stop_price, entry * 0.9995)
                    else:
                        effective_stop = pos.stop_price

                if high >= effective_stop:
                    reason = 'stop' if effective_stop == pos.stop_price else 'trail'
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=effective_stop, reason=reason))
                    continue
                if low <= pos.tp_price:
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.tp_price, reason='tp'))
                    continue
                hold_5m = pos.hold_bars // _1m_to_5m
                if not is_breakout and hold_5m >= max(6, int(ou_hl * 3)):
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='ou_timeout'))
                    continue

            # Hard timeout (max_hold is in 5-min bars)
            hold_5m = pos.hold_bars // _1m_to_5m
            if hold_5m >= max_hold:
                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='timeout'))

        return exits

    def on_fill(self, trade):
        """Clean up position state."""
        self._pos_state.pop(trade.pos_id, None)
