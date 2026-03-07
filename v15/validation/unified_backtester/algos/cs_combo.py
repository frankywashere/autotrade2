"""
CS-Combo Algorithm — Plug-in for unified backtester.

Replicates combo_backtest.py: Channel Surfer signal on daily bars,
entry at next-day open, trail = 0.025 * (1-conf)^power.

Signal source: prepare_multi_tf_analysis() (physics-based multi-window channels)

Two variants:
  - CS-5TF (cs-combo): All TFs — 5min, 1h, 4h, daily, weekly, monthly
  - CS-DW  (cs-dw):    Daily + weekly only (matching live scanner CS-DW)
"""

import datetime as dt
import time as _time_mod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..algo_base import AlgoBase, AlgoConfig, Signal, ExitSignal, CostModel
from ..data_provider import DataProvider
from ..portfolio import Position


# Default config matching combo_backtest.py with c16 settings
DEFAULT_CS_COMBO_CONFIG = AlgoConfig(
    algo_id='cs-combo',
    initial_equity=100_000.0,
    max_equity_per_trade=100_000.0,
    max_positions=1,
    primary_tf='daily',
    eval_interval=1,
    exit_check_tf='daily',
    cost_model=CostModel(
        slippage_pct=0.0001,        # 0.01% per side (matching combo_backtest)
        commission_per_share=0.005,
    ),
    params={
        'flat_sizing': True,         # c16: flat $100K
        'trail_base': 0.025,         # 2.5% base
        'trail_power': 12,           # c16: dodecic (was 4 for original)
        'stop_pct': 0.02,            # 2% hard stop
        'tp_pct': 0.04,              # 4% take profit
        'max_hold_days': 10,
        'cooldown_days': 2,
        'min_confidence': 0.45,
        'combo_type': 'raw_cs',      # 'raw_cs' or future combo gating functions
        # 'target_tfs': None,        # None = all TFs; list = restrict analysis
    },
)

# CS-DW: Daily + Weekly only (matching live scanner CS-DW config)
DEFAULT_CS_DW_CONFIG = AlgoConfig(
    algo_id='cs-dw',
    initial_equity=100_000.0,
    max_equity_per_trade=100_000.0,
    max_positions=1,
    primary_tf='daily',
    eval_interval=1,
    exit_check_tf='daily',
    cost_model=CostModel(
        slippage_pct=0.0001,
        commission_per_share=0.005,
    ),
    params={
        'flat_sizing': True,
        'trail_base': 0.025,
        'trail_power': 12,
        'stop_pct': 0.02,
        'tp_pct': 0.04,
        'max_hold_days': 10,
        'cooldown_days': 0,          # CS-DW: no cooldown (cd=0 in c15 comparison)
        'min_confidence': 0.45,
        'combo_type': 'raw_cs',
        'target_tfs': ['daily', 'weekly'],  # DW = daily + weekly only
    },
)


class CSComboAlgo(AlgoBase):
    """Channel Surfer combo algorithm — daily signal, next-day open entry.

    Precomputes all CS signals at init (matching combo_backtest Phase 1),
    then looks them up during the walk-forward simulation.
    """

    def __init__(self, config: AlgoConfig = None, data: DataProvider = None):
        super().__init__(config or DEFAULT_CS_COMBO_CONFIG, data)
        self._cooldown_remaining = 0
        self._current_day = None

        # Load signals: from cache (preferred) or live computation
        print("  Loading CS signals...")
        t0 = _time_mod.time()
        cache_path = self.config.params.get('signal_cache')
        if cache_path:
            self._day_signals = self._load_from_cache(cache_path)
        else:
            self._day_signals = self._precompute_signals()
        print(f"  Done: {len(self._day_signals)} days with signals in {_time_mod.time() - t0:.1f}s")

    def _load_from_cache(self, cache_path: str) -> Dict:
        """Load precomputed signals from combo_backtest pickle cache."""
        import pickle
        from pathlib import Path

        path = Path(cache_path)
        if not path.exists():
            print(f"  WARNING: Cache not found at {cache_path}, falling back to live computation")
            return self._precompute_signals()

        with open(path, 'rb') as f:
            cache_data = pickle.load(f)

        signals = {}
        min_conf = self.config.params.get('min_confidence', 0.45)
        stop_default = self.config.params.get('stop_pct', 0.02)
        tp_default = self.config.params.get('tp_pct', 0.04)

        for day_sig in cache_data['signals']:
            if day_sig.cs_action in ('BUY', 'SELL') and day_sig.cs_confidence >= 0.01:
                day = day_sig.date.date() if hasattr(day_sig.date, 'date') else day_sig.date
                # Apply floor to stop/TP (matching _floor_stop_tp in combo_backtest)
                stop = max(day_sig.cs_stop_pct or stop_default, stop_default)
                tp = max(day_sig.cs_tp_pct or tp_default, tp_default)
                signals[day] = {
                    'action': day_sig.cs_action,
                    'confidence': day_sig.cs_confidence,
                    'stop_pct': stop,
                    'tp_pct': tp,
                    'signal_type': day_sig.cs_signal_type,
                }

        return signals

    def _precompute_signals(self) -> Dict:
        """Run prepare_multi_tf_analysis for each trading day.

        Returns dict: {date: {action, confidence, stop_pct, tp_pct, signal_type, ...}}
        """
        try:
            from v15.core.channel_surfer import prepare_multi_tf_analysis
        except ImportError as e:
            print(f"  WARNING: Cannot import channel_surfer: {e}")
            return {}

        signals = {}
        trading_days = self.data.trading_days

        # Build native_data dict with available TFs (filtered by target_tfs if set)
        target_tfs = self.config.params.get('target_tfs')
        if target_tfs:
            tfs_needed = list(target_tfs)
        else:
            tfs_needed = ['5min', '1h', '4h', 'daily', 'weekly', 'monthly']
        full_data = {}
        for tf in tfs_needed:
            if tf in self.data._tf_data and len(self.data._tf_data[tf]) > 0:
                full_data[tf] = self.data._tf_data[tf]

        for i, day in enumerate(trading_days):
            day_ts = pd.Timestamp(day)

            # Slice data up to this day (no lookahead)
            native_slice = {}
            for tf, df in full_data.items():
                # For daily+ TFs, include up to this date
                if tf in ('daily', 'weekly', 'monthly'):
                    mask = df.index <= day_ts + pd.Timedelta(days=1)
                else:
                    mask = df.index <= day_ts + pd.Timedelta(hours=16)
                sliced = df.loc[mask]
                if len(sliced) >= 15:
                    native_slice[tf] = sliced

            if not native_slice:
                continue

            try:
                analysis = prepare_multi_tf_analysis(
                    native_data={'TSLA': native_slice})
                sig = analysis.signal

                if sig.action in ('BUY', 'SELL') and sig.confidence >= 0.01:
                    signals[day] = {
                        'action': sig.action,
                        'confidence': sig.confidence,
                        'stop_pct': sig.suggested_stop_pct or self.config.params['stop_pct'],
                        'tp_pct': sig.suggested_tp_pct or self.config.params['tp_pct'],
                        'signal_type': sig.signal_type,
                        'primary_tf': getattr(sig, 'primary_tf', ''),
                    }
            except Exception as e:
                if i % 50 == 0:
                    print(f"    CS error on {day}: {e}")

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(trading_days)} days processed")

        return signals

    def warmup_bars(self) -> int:
        return 0  # Signals precomputed

    def on_bar(self, time: pd.Timestamp, bar: dict,
               open_positions: list) -> List[Signal]:
        """At end of trading day: look up precomputed CS signal."""
        params = self.config.params

        day = time.date()
        if day != self._current_day:
            self._current_day = day
            # Cooldown: skip this day, then decrement
            # (matching original: decrement + continue = skip 'cooldown' days)
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                return []

        # Skip if in cooldown or already in a trade
        if self._cooldown_remaining > 0:
            return []
        if open_positions:
            return []

        # Look up today's precomputed signal
        sig_data = self._day_signals.get(day)
        if sig_data is None:
            return []

        # Apply minimum confidence filter
        if sig_data['confidence'] < params.get('min_confidence', 0.45):
            return []

        direction = 'long' if sig_data['action'] == 'BUY' else 'short'

        return [Signal(
            algo_id=self.config.algo_id,
            direction=direction,
            price=bar['close'],  # Will be overridden to next bar's open
            confidence=sig_data['confidence'],
            stop_pct=sig_data['stop_pct'],
            tp_pct=sig_data['tp_pct'],
            signal_type=sig_data.get('signal_type', 'cs'),
            delayed_entry=True,  # Fill at next RTH open
        )]

    def check_exits(self, time: pd.Timestamp, bar: dict,
                    open_positions: list) -> List[ExitSignal]:
        """Check trailing stop, stop loss, take profit, and timeout on daily bars."""
        exits = []
        params = self.config.params
        trail_base = params.get('trail_base', 0.025)
        trail_power = params.get('trail_power', 12)
        max_hold = params.get('max_hold_days', 10)

        for pos in open_positions:
            high = bar['high']
            low = bar['low']
            close = bar['close']

            # Compute trail for this trade
            trail_pct = trail_base * (1.0 - pos.confidence) ** trail_power

            if pos.direction == 'long':
                best = max(pos.best_price, high)

                # Trailing stop (only activates when in profit — matching combo_backtest)
                trailing_stop = best * (1.0 - trail_pct)
                if best > pos.entry_price:
                    effective_stop = max(pos.stop_price, trailing_stop)
                else:
                    effective_stop = pos.stop_price

                # Check stop hit
                if low <= effective_stop:
                    is_trailing = best > pos.entry_price and trailing_stop > pos.stop_price
                    reason = 'trailing' if is_trailing else 'stop'
                    exits.append(ExitSignal(
                        pos_id=pos.pos_id, price=effective_stop, reason=reason))
                    continue

                # Check take profit
                if high >= pos.tp_price:
                    exits.append(ExitSignal(
                        pos_id=pos.pos_id, price=pos.tp_price, reason='tp'))
                    continue

            else:  # short
                best = min(pos.best_price, low)

                trailing_stop = best * (1.0 + trail_pct)
                if best < pos.entry_price:
                    effective_stop = min(pos.stop_price, trailing_stop)
                else:
                    effective_stop = pos.stop_price

                if high >= effective_stop:
                    is_trailing = best < pos.entry_price and trailing_stop < pos.stop_price
                    reason = 'trailing' if is_trailing else 'stop'
                    exits.append(ExitSignal(
                        pos_id=pos.pos_id, price=effective_stop, reason=reason))
                    continue

                if low <= pos.tp_price:
                    exits.append(ExitSignal(
                        pos_id=pos.pos_id, price=pos.tp_price, reason='tp'))
                    continue

            # Check timeout (hold_bars counts 1-min bars, convert to days: ~390 bars/day)
            hold_days = pos.hold_bars / 390.0
            if hold_days >= max_hold:
                exits.append(ExitSignal(
                    pos_id=pos.pos_id, price=close, reason='timeout'))

        return exits

    def on_fill(self, trade):
        """Set cooldown after trade close."""
        self._cooldown_remaining = self.config.params.get('cooldown_days', 2)
