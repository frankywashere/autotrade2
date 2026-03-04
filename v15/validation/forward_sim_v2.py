#!/usr/bin/env python3
"""
Forward Sim V2: Live Scanner Replica.

Replicates the exact SurferLiveScanner trading logic on historical 5-min bars.
Unlike forward_sim.py (which uses run_backtest()), this processes every 5-min bar
through prepare_multi_tf_analysis() with proper native TF data, uses risk-based
position sizing, and applies all 8 exit types from the live scanner.

Usage:
    python -m v15.validation.forward_sim_v2 --start 2026-02-23 --end 2026-02-26
    python -m v15.validation.forward_sim_v2 --start 2026-02-23 --end 2026-02-26 --capital 100000
"""
import argparse
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SimPosition:
    """Hypothetical open position (mirrors HypotheticalPosition from live scanner)."""
    pos_id: str
    direction: str           # 'long' or 'short'
    entry_price: float
    entry_time: pd.Timestamp
    shares: int
    notional: float
    stop_price: float
    tp_price: float
    signal_type: str         # 'bounce' or 'break'
    primary_tf: str
    confidence: float
    best_price: float        # For trailing stop
    reason: str = ''
    breakeven_applied: bool = False
    override_applied: bool = False
    signal_source: str = ''          # 'CS-5TF', 'CS-DW', 'surfer_ml', 'intraday'
    initial_stop_pct: float = 0.02   # Original stop % for trail calc
    pos_hard_stop_pct: Optional[float] = None  # Per-position hard stop override


@dataclass
class SimClosedTrade:
    """Record of a closed trade."""
    pos_id: str
    direction: str
    entry_price: float
    exit_price: float
    shares: int
    notional: float
    pnl: float
    pnl_pct: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    exit_reason: str
    hold_minutes: float
    signal_type: str
    primary_tf: str
    confidence: float
    override_applied: bool = False
    signal_source: str = ''


# ---------------------------------------------------------------------------
# SimScanner — standalone replica of SurferLiveScanner logic
# ---------------------------------------------------------------------------

class SimScanner:
    """Replicates SurferLiveScanner logic with explicit timestamps (no datetime.now())."""

    # Constants matching surfer_live_scanner.py
    SLIPPAGE_PCT = 0.0005           # 0.05% per side
    COMMISSION_PER_SHARE = 0.005    # $0.005/share (IBKR tiered)
    TIMEOUT_MINUTES = 300           # 5 hours
    EOD_CLOSE_HOUR_ET = 15
    EOD_CLOSE_MINUTE_ET = 45
    EQUITY_CEILING_PCT = 0.05       # Close if unrealized >= 5% of equity
    NEAR_TP_PCT = 0.01              # Close early if within 1% of TP
    NEAR_TP_MIN_GAIN = 100.0        # Min dollar gain for near-TP
    NEAR_TP_WINDOW_MIN = 30         # Near-TP only in first 30 min
    BREAKEVEN_TRIGGER_MIN = 30      # Move stop to entry after 30 min if in profit
    OUTLIER_MULT = 2.0              # Close if unrealized >= 2x TP gain
    MIN_CONFIDENCE = 0.45
    RISK_PER_TRADE = 0.02           # 2% of equity
    MAX_LEVERAGE = 4.0
    MAX_BUYING_POWER_PCT = 0.25
    MAX_POSITIONS = 2
    DAILY_LOSS_LIMIT = -2000.0

    def __init__(self, initial_capital: float = 100_000,
                 trail_enabled: bool = True,
                 stop_multiplier: float = 1.0,
                 eod_close_enabled: bool = True,
                 timeout_multiplier: float = 1.0,
                 signal_source: str = '',
                 hard_stop_pct: Optional[float] = None):
        self.equity = initial_capital
        self.initial_capital = initial_capital
        self.positions: Dict[str, SimPosition] = {}
        self.closed_trades: List[SimClosedTrade] = []
        self.daily_pnl = 0.0
        self.current_date: Optional[str] = None
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        # Configurable exit params
        self.trail_enabled = trail_enabled
        self.stop_multiplier = stop_multiplier
        self.eod_close_enabled = eod_close_enabled
        self.timeout_multiplier = timeout_multiplier
        # Signal source routing (empty = legacy behavior)
        self.signal_source = signal_source
        self.hard_stop_pct = hard_stop_pct

    def _reset_daily_if_needed(self, bar_time: pd.Timestamp):
        """Reset daily P&L tracking on new date."""
        day_str = bar_time.strftime('%Y-%m-%d')
        if day_str != self.current_date:
            self.daily_pnl = 0.0
            self.current_date = day_str

    def _is_eod(self, bar_time: pd.Timestamp) -> bool:
        """Check if bar_time is at or past 3:45 PM ET."""
        try:
            import pytz
            et = pytz.timezone('US/Eastern')
            if bar_time.tzinfo is not None:
                bar_et = bar_time.astimezone(et)
            else:
                bar_et = et.localize(bar_time)
            return (bar_et.hour > self.EOD_CLOSE_HOUR_ET or
                    (bar_et.hour == self.EOD_CLOSE_HOUR_ET and
                     bar_et.minute >= self.EOD_CLOSE_MINUTE_ET))
        except Exception:
            return False

    def evaluate_signal(self, analysis, current_price: float, bar_time: pd.Timestamp,
                        tf_filter: Optional[set] = None) -> bool:
        """Evaluate a ChannelAnalysis for entry. Returns True if position entered.

        Sizing and stop logic branch by self.signal_source:
        - CS-5TF/CS-DW: conf × capital sizing, floor stops 2%/4%
        - surfer_ml: risk-based sizing, ATR-clipped stops
        - '' (legacy): risk-based sizing

        Args:
            tf_filter: If set, only accept signals where primary_tf is in this set.
        """
        self._reset_daily_if_needed(bar_time)

        sig = analysis.signal

        # TF filter (for portfolio mode)
        if tf_filter and sig.primary_tf not in tf_filter:
            return False

        # Kill switch / daily loss limit
        if self.daily_pnl <= self.DAILY_LOSS_LIMIT:
            return False

        # HOLD = no action
        if sig.action == 'HOLD':
            return False

        # Confidence gate
        if sig.confidence < self.MIN_CONFIDENCE:
            return False

        # Max positions (intraday has its own cap in evaluate_intraday)
        if len(self.positions) >= self.MAX_POSITIONS:
            return False

        # Duplicate direction check
        direction = 'long' if sig.action == 'BUY' else 'short'
        for pos in self.positions.values():
            if pos.direction == direction:
                return False

        # --- Stop/TP logic (branched by signal source) ---
        stop_pct = sig.suggested_stop_pct
        if stop_pct <= 0:
            stop_pct = 0.02
        tp_pct = sig.suggested_tp_pct
        signal_type = getattr(sig, 'signal_type', 'bounce')

        if self.signal_source in ('CS-5TF', 'CS-DW'):
            # combo_backtest.py: floor at 2%/4%, no ATR clipping
            stop_pct = max(stop_pct, 0.02)
            tp_pct = max(tp_pct, 0.04)
        elif self.signal_source == 'surfer_ml':
            # surfer_backtest.py: ATR-clipped stops
            atr_val = getattr(analysis, 'atr', None) or getattr(sig, 'atr', None)
            if atr_val and atr_val > 0 and current_price > 0:
                if signal_type == 'bounce':
                    atr_floor = (0.5 * atr_val) / current_price
                    atr_cap = (1.5 * atr_val) / current_price
                else:  # break
                    atr_floor = (1.5 * atr_val) / current_price
                    atr_cap = (3.0 * atr_val) / current_price
                stop_pct = max(atr_floor, min(stop_pct, atr_cap))
                if signal_type == 'break':
                    stop_pct *= 0.05
                    if stop_pct < 0.00030:
                        return False
            # TP widening for high-confidence bounces
            if signal_type == 'bounce' and sig.confidence > 0.65:
                tp_pct *= 1.30
        else:
            # Legacy: apply stop multiplier
            stop_pct = stop_pct * self.stop_multiplier

        # --- Position sizing (branched by signal source) ---
        if self.signal_source in ('CS-5TF', 'CS-DW'):
            # combo_backtest.py: confidence-scaled, full capital base
            position_value = self.initial_capital * min(sig.confidence, 1.0)
            shares = max(1, int(position_value / current_price))
        elif self.signal_source == 'surfer_ml':
            # surfer_backtest.py: risk-based sizing with type multipliers
            risk_dollars = self.equity * self.RISK_PER_TRADE
            stop_distance = stop_pct * current_price
            shares = int(risk_dollars / stop_distance) if stop_distance > 0 else 0
            size_mult = 1.0
            if signal_type == 'bounce':
                size_mult = 1.5 if direction == 'long' else 2.5
            elif signal_type == 'break':
                ch_health = getattr(sig, 'channel_health', 0.5)
                size_mult = 0.6 if ch_health > 0.50 else (1.4 if ch_health < 0.30 else 1.0)
            shares = int(shares * size_mult)
        else:
            # Legacy: risk-based
            risk_dollars = self.equity * self.RISK_PER_TRADE
            stop_distance = stop_pct * current_price
            shares = int(risk_dollars / stop_distance) if stop_distance > 0 else 0

        # Cap at max buying power
        buying_power = self.equity * self.MAX_LEVERAGE
        max_notional = buying_power * self.MAX_BUYING_POWER_PCT
        max_shares = int(max_notional / current_price) if current_price > 0 else 0
        shares = min(shares, max_shares)

        if shares <= 0:
            return False

        notional = shares * current_price

        # Compute stop/TP prices
        if direction == 'long':
            stop_price = current_price * (1 - stop_pct)
            tp_price = current_price * (1 + tp_pct)
        else:
            stop_price = current_price * (1 + stop_pct)
            tp_price = current_price * (1 - tp_pct)

        pos_id = str(uuid.uuid4())[:8]
        override_applied = False
        if hasattr(analysis, 'override_info') and analysis.override_info:
            override_applied = analysis.override_info.get('override_applied', False)

        pos = SimPosition(
            pos_id=pos_id,
            direction=direction,
            entry_price=current_price,
            entry_time=bar_time,
            shares=shares,
            notional=notional,
            stop_price=stop_price,
            tp_price=tp_price,
            signal_type=signal_type,
            primary_tf=sig.primary_tf,
            confidence=sig.confidence,
            best_price=current_price,
            reason=sig.reason,
            override_applied=override_applied,
            signal_source=self.signal_source,
            initial_stop_pct=stop_pct,
        )
        self.positions[pos_id] = pos
        return True

    def evaluate_intraday(self, features: dict, current_price: float,
                          bar_time: pd.Timestamp, bar_idx: int) -> bool:
        """Evaluate intraday signal from precomputed features. Returns True if position entered.

        Args:
            features: Dict with 'vwap_dist', 'vol_ratio', 'vwap_slope', 'rsi_slope',
                      'spread_pct', 'gap_pct' arrays + 'cp5', 'daily_cp', 'h1_cp',
                      'h4_cp', 'daily_slope', 'h1_slope', 'h4_slope' scalars.
            current_price: Current TSLA price
            bar_time: Timestamp of current bar
            bar_idx: Index into precomputed feature arrays
        """
        from v15.trading.intraday_signals import sig_union_enhanced

        self._reset_daily_if_needed(bar_time)

        # PM-only window check (13:00-15:25 ET)
        try:
            import pytz
            et = pytz.timezone('US/Eastern')
            if bar_time.tzinfo is not None:
                bar_et = bar_time.astimezone(et)
            else:
                bar_et = et.localize(bar_time)
            from datetime import time as _time
            if not (_time(13, 0) <= bar_et.time() <= _time(15, 25)):
                return False
        except Exception:
            return False

        # Daily loss limit
        if self.daily_pnl <= self.DAILY_LOSS_LIMIT:
            return False

        # Cap at 30 intraday positions
        if len(self.positions) >= 30:
            return False

        # Get scalar features for this bar
        def _get(key):
            v = features.get(key)
            if v is None:
                return float('nan')
            if isinstance(v, np.ndarray):
                return float(v[bar_idx]) if bar_idx < len(v) else float('nan')
            return float(v)  # Already scalar

        result = sig_union_enhanced(
            cp5=_get('cp5'),
            vwap_dist=_get('vwap_dist'),
            daily_cp=_get('daily_cp'),
            h1_cp=_get('h1_cp'),
            h4_cp=_get('h4_cp'),
            vol_ratio=_get('vol_ratio'),
            vwap_slope=_get('vwap_slope'),
            bullish_1m=float('nan'),  # Would need 1-min data
            gap_pct=_get('gap_pct'),
            rsi_slope=_get('rsi_slope'),
            daily_slope=_get('daily_slope'),
            h1_slope=_get('h1_slope'),
            h4_slope=_get('h4_slope'),
            spread_pct=_get('spread_pct'),
        )

        if result is None:
            return False

        _, confidence, stop_pct, tp_pct = result

        if confidence < self.MIN_CONFIDENCE:
            return False

        # Flat sizing: $capital / price (matching intraday backtest conf_size=False)
        shares = max(1, int(self.initial_capital / current_price)) if current_price > 0 else 0
        if shares <= 0:
            return False

        notional = shares * current_price
        stop_price = current_price * (1 - stop_pct)
        tp_price = current_price * (1 + tp_pct)

        pos_id = str(uuid.uuid4())[:8]
        pos = SimPosition(
            pos_id=pos_id,
            direction='long',  # Intraday is long-only
            entry_price=current_price,
            entry_time=bar_time,
            shares=shares,
            notional=notional,
            stop_price=stop_price,
            tp_price=tp_price,
            signal_type='intraday',
            primary_tf='5min',
            confidence=confidence,
            best_price=current_price,
            reason='Intraday FD Enh-Union',
            signal_source='intraday',
            initial_stop_pct=stop_pct,
        )
        self.positions[pos_id] = pos
        return True

    @staticmethod
    def _calc_trail_price(pos: SimPosition) -> Optional[float]:
        """Profit-tier trailing stop, exact copy of SurferLiveScanner._calc_trail_price()."""
        entry = pos.entry_price
        if entry <= 0:
            return None
        initial_stop_dist = abs(pos.stop_price - entry) / entry
        tp_dist = abs(pos.tp_price - entry) / entry
        is_breakout = pos.signal_type == 'break'

        if pos.direction == 'long':
            if pos.best_price <= entry:
                return None
            if is_breakout:
                profit_from_best = (pos.best_price - entry) / entry
                if profit_from_best > 0.015:
                    trail_pct = initial_stop_dist * 0.01
                elif profit_from_best > 0.008:
                    trail_pct = initial_stop_dist * 0.02
                elif profit_from_best > 0.0008:
                    trail_pct = initial_stop_dist * 0.01
                else:
                    return None
            else:  # bounce
                profit_from_entry = (pos.best_price - entry) / entry
                profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
                if profit_ratio >= 0.80:
                    trail_pct = initial_stop_dist * 0.005
                elif profit_ratio >= 0.55:
                    trail_pct = initial_stop_dist * 0.02
                elif profit_ratio >= 0.40:
                    trail_pct = initial_stop_dist * 0.06
                else:
                    return None
            return max(pos.stop_price, pos.best_price * (1.0 - trail_pct))

        else:  # short
            if pos.best_price >= entry:
                return None
            if is_breakout:
                profit_from_best = (entry - pos.best_price) / entry
                if profit_from_best > 0.015:
                    trail_pct = initial_stop_dist * 0.01
                elif profit_from_best > 0.008:
                    trail_pct = initial_stop_dist * 0.02
                elif profit_from_best > 0.0008:
                    trail_pct = initial_stop_dist * 0.01
                else:
                    return None
            else:  # bounce
                profit_from_entry = (entry - pos.best_price) / entry
                profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
                if profit_ratio >= 0.80:
                    trail_pct = initial_stop_dist * 0.005
                elif profit_ratio >= 0.55:
                    trail_pct = initial_stop_dist * 0.02
                elif profit_ratio >= 0.40:
                    trail_pct = initial_stop_dist * 0.06
                else:
                    return None
            return min(pos.stop_price, pos.best_price * (1.0 + trail_pct))

    # CS trail constants (matching surfer_live_scanner.py)
    CS_TRAIL_BASE = 0.025
    CS_TRAIL_POWER = 8
    CS_MAX_HOLD_DAYS = 10

    def check_exits(self, current_price: float, bar_high: float, bar_low: float,
                    bar_time: pd.Timestamp) -> List[SimClosedTrade]:
        """Check all open positions for exit conditions. Returns list of closed trades.

        Exit logic branches by pos.signal_source when set, otherwise uses legacy behavior.
        """
        closed: List[SimClosedTrade] = []
        to_close: List[str] = []
        is_eod = self._is_eod(bar_time)

        # Also check for intraday EOD (15:50 ET)
        is_intraday_eod = False
        try:
            import pytz
            et = pytz.timezone('US/Eastern')
            if bar_time.tzinfo is not None:
                bar_et = bar_time.astimezone(et)
            else:
                bar_et = et.localize(bar_time)
            from datetime import time as _time
            is_intraday_eod = bar_et.time() >= _time(15, 50)
        except Exception:
            pass

        for pos_id, pos in self.positions.items():
            exit_reason = None
            exit_price = current_price

            hold_minutes = (bar_time - pos.entry_time).total_seconds() / 60

            if pos.direction == 'long':
                unrealized_pnl = (current_price - pos.entry_price) * pos.shares
            else:
                unrealized_pnl = (pos.entry_price - current_price) * pos.shares

            is_cs = pos.signal_source in ('CS-5TF', 'CS-DW')
            is_intraday = pos.signal_source == 'intraday'
            is_ml = pos.signal_source == 'surfer_ml'
            use_legacy = not pos.signal_source

            # --- Breakeven stop adjustment (surfer_ml and legacy only) ---
            if (not is_cs and not is_intraday
                    and not pos.breakeven_applied
                    and hold_minutes >= self.BREAKEVEN_TRIGGER_MIN
                    and unrealized_pnl > 0):
                if pos.direction == 'long':
                    pos.stop_price = max(pos.stop_price, pos.entry_price)
                else:
                    pos.stop_price = min(pos.stop_price, pos.entry_price)
                pos.breakeven_applied = True

            # Update best price
            if pos.direction == 'long':
                if bar_high > pos.best_price:
                    pos.best_price = bar_high
            else:
                if bar_low < pos.best_price:
                    pos.best_price = bar_low

            # --- Intraday EOD close at 15:50 ET ---
            if exit_reason is None and is_intraday and is_intraday_eod:
                exit_reason = 'intraday_eod'

            # --- Non-intraday EOD force close (3:45 PM ET) ---
            if exit_reason is None and not is_intraday and self.eod_close_enabled and is_eod:
                exit_reason = 'eod_close'

            # --- Hard stop cap (per-position override or scanner-level) ---
            hs_pct = pos.pos_hard_stop_pct if pos.pos_hard_stop_pct is not None else self.hard_stop_pct
            if exit_reason is None and hs_pct is not None:
                if pos.direction == 'long':
                    hard_stop_price = pos.entry_price * (1 - hs_pct)
                    if bar_low <= hard_stop_price:
                        exit_reason = 'hard_stop'
                        exit_price = hard_stop_price
                elif pos.direction == 'short':
                    hard_stop_price = pos.entry_price * (1 + hs_pct)
                    if bar_high >= hard_stop_price:
                        exit_reason = 'hard_stop'
                        exit_price = hard_stop_price

            # --- Legacy advanced exits (only for legacy/ML mode) ---
            if use_legacy or is_ml:
                if exit_reason is None and hold_minutes < self.NEAR_TP_WINDOW_MIN:
                    if unrealized_pnl >= self.NEAR_TP_MIN_GAIN:
                        if (pos.direction == 'long' and
                                bar_high >= pos.tp_price * (1 - self.NEAR_TP_PCT)):
                            exit_reason = 'near_tp'
                        elif (pos.direction == 'short' and
                              bar_low <= pos.tp_price * (1 + self.NEAR_TP_PCT)):
                            exit_reason = 'near_tp'

                if exit_reason is None:
                    if unrealized_pnl >= self.equity * self.EQUITY_CEILING_PCT:
                        exit_reason = 'equity_ceiling'

                if exit_reason is None:
                    if pos.direction == 'long':
                        expected_tp_gain = (pos.tp_price - pos.entry_price) * pos.shares
                    else:
                        expected_tp_gain = (pos.entry_price - pos.tp_price) * pos.shares
                    if expected_tp_gain > 0 and unrealized_pnl >= self.OUTLIER_MULT * expected_tp_gain:
                        exit_reason = 'outlier_winner'

            # --- Take profit ---
            if exit_reason is None:
                if pos.direction == 'long' and bar_high >= pos.tp_price:
                    exit_reason = 'take_profit'
                    exit_price = pos.tp_price
                elif pos.direction == 'short' and bar_low <= pos.tp_price:
                    exit_reason = 'take_profit'
                    exit_price = pos.tp_price

            # --- Trailing stop (branched by signal source) ---
            if exit_reason is None and self.trail_enabled:
                if is_intraday:
                    # Intraday: trail = 0.006 * (1 - conf)^6
                    trail_pct = 0.006 * (1.0 - pos.confidence) ** 6
                    if pos.direction == 'long' and pos.best_price > pos.entry_price:
                        trail_price = pos.best_price * (1.0 - trail_pct)
                        trail_price = max(trail_price, pos.stop_price)
                        if bar_low <= trail_price:
                            exit_reason = 'trailing_stop'
                            exit_price = trail_price
                elif is_cs:
                    # CS-5TF/CS-DW: exponential trail = 0.025 * (1 - conf)^8
                    trail_pct = self.CS_TRAIL_BASE * (1.0 - pos.confidence) ** self.CS_TRAIL_POWER
                    if pos.direction == 'long' and pos.best_price > pos.entry_price:
                        trail_price = pos.best_price * (1.0 - trail_pct)
                        trail_price = max(trail_price, pos.stop_price)
                        if bar_low <= trail_price:
                            exit_reason = 'trailing_stop'
                            exit_price = trail_price
                    elif pos.direction == 'short' and pos.best_price < pos.entry_price:
                        trail_price = pos.best_price * (1.0 + trail_pct)
                        trail_price = min(trail_price, pos.stop_price)
                        if bar_high >= trail_price:
                            exit_reason = 'trailing_stop'
                            exit_price = trail_price
                else:
                    # Surfer ML / legacy: profit-tier trail
                    trail_price = self._calc_trail_price(pos)
                    if trail_price is not None:
                        if pos.direction == 'long' and bar_low <= trail_price:
                            exit_reason = 'trailing_stop'
                            exit_price = trail_price
                        elif pos.direction == 'short' and bar_high >= trail_price:
                            exit_reason = 'trailing_stop'
                            exit_price = trail_price

            # --- Timeout (branched by signal source) ---
            if exit_reason is None:
                if is_cs:
                    hold_days = hold_minutes / (60 * 6.5)
                    if hold_days > self.CS_MAX_HOLD_DAYS:
                        exit_reason = 'timeout'
                elif is_intraday:
                    pass  # EOD close handles intraday timeout
                elif is_ml:
                    timeout = 600 if pos.signal_type == 'break' else 300
                    if hold_minutes > timeout:
                        exit_reason = 'timeout'
                else:
                    # Legacy
                    if hold_minutes > self.TIMEOUT_MINUTES * self.timeout_multiplier:
                        exit_reason = 'timeout'

            if exit_reason:
                trade = self._close_position(pos, exit_price, exit_reason, bar_time)
                closed.append(trade)
                to_close.append(pos_id)

        for pos_id in to_close:
            del self.positions[pos_id]

        return closed

    def _close_position(self, pos: SimPosition, exit_price: float,
                        exit_reason: str, bar_time: pd.Timestamp) -> SimClosedTrade:
        """Close position with slippage + commission. Update equity."""
        if pos.direction == 'long':
            raw_pnl = (exit_price - pos.entry_price) * pos.shares
        else:
            raw_pnl = (pos.entry_price - exit_price) * pos.shares

        slippage_cost = pos.notional * self.SLIPPAGE_PCT * 2  # entry + exit
        commission = pos.shares * self.COMMISSION_PER_SHARE * 2
        pnl = raw_pnl - slippage_cost - commission
        pnl_pct = pnl / pos.notional if pos.notional > 0 else 0

        # Update equity and daily tracking
        self.equity += pnl
        self.daily_pnl += pnl

        # Track drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = (self.peak_equity - self.equity) / self.peak_equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        hold_minutes = (bar_time - pos.entry_time).total_seconds() / 60

        trade = SimClosedTrade(
            pos_id=pos.pos_id,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            shares=pos.shares,
            notional=pos.notional,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_time=pos.entry_time,
            exit_time=bar_time,
            exit_reason=exit_reason,
            hold_minutes=hold_minutes,
            signal_type=pos.signal_type,
            primary_tf=pos.primary_tf,
            confidence=pos.confidence,
            override_applied=pos.override_applied,
            signal_source=pos.signal_source,
        )
        self.closed_trades.append(trade)
        return trade

    def force_close_all(self, current_price: float, bar_time: pd.Timestamp) -> List[SimClosedTrade]:
        """Force-close all remaining positions (end of sim)."""
        closed = []
        for pos_id, pos in list(self.positions.items()):
            trade = self._close_position(pos, current_price, 'sim_end', bar_time)
            closed.append(trade)
        self.positions.clear()
        return closed


# ---------------------------------------------------------------------------
# Per-TF simulation config
# ---------------------------------------------------------------------------

TF_SIM_CONFIG = {
    '5min': {
        'data_key': 'tsla_5min',
        'spy_data_key': 'spy_5min',
        'native_tf': '5min',
        'target_tfs': ['5min', '1h', '4h', 'daily', 'weekly'],
        'timeout_minutes': 300,       # 5 hours
    },
    '1h': {
        'data_key': 'tsla_1h',
        'spy_data_key': 'spy_1h',
        'native_tf': '1h',
        'target_tfs': ['1h', '4h', 'daily', 'weekly'],
        'timeout_minutes': 600,       # ~10 bars
    },
    '4h': {
        'data_key': 'tsla_4h',
        'spy_data_key': 'spy_4h',
        'native_tf': '4h',
        'target_tfs': ['4h', 'daily', 'weekly'],
        'timeout_minutes': 1200,      # ~5 bars
    },
    'daily': {
        'data_key': 'tsla_daily',
        'spy_data_key': 'spy_daily',
        'native_tf': 'daily',
        'target_tfs': ['daily', 'weekly'],
        'timeout_minutes': 1950,      # ~5 trading days
    },
    'weekly': {
        'data_key': 'tsla_weekly',
        'spy_data_key': 'spy_weekly',
        'native_tf': 'weekly',
        'target_tfs': ['weekly'],
        'timeout_minutes': 9750,      # ~5 weeks
    },
}


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _clean_yf(df):
    """Normalize yfinance DataFrame: flatten MultiIndex columns, lowercase."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if 'adj close' in df.columns:
        df = df.drop(columns=['adj close'])
    return df


def _load_min_file(path: str, start: str, end: str):
    """Load a semicolon-delimited 1-min file (TSLAMin.txt / SPYMin.txt) and resample.

    The local data timestamps are in UTC. We localize to UTC, filter to RTH
    (9:30-16:00 ET = 14:30-21:00 UTC), then convert to US/Eastern so the sim's
    time-of-day logic works correctly.
    """
    import pytz

    df1m = pd.read_csv(
        path, sep=';', header=None,
        names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
        parse_dates=['datetime'], date_format='%Y%m%d %H%M%S',
        index_col='datetime',
    )
    df1m.index = pd.to_datetime(df1m.index).tz_localize('UTC')
    df1m = df1m[start:end]

    # Convert to ET for RTH filtering
    et = pytz.timezone('US/Eastern')
    df1m.index = df1m.index.tz_convert(et)

    # Filter to RTH only: 9:30-16:00 ET
    hours = df1m.index.hour
    minutes = df1m.index.minute
    rth_mask = ((hours > 9) | ((hours == 9) & (minutes >= 30))) & (hours < 16)
    df1m = df1m[rth_mask]
    print(f"    RTH filter: {rth_mask.sum():,} of {len(rth_mask):,} bars kept")

    def _resample(df, rule):
        return df.resample(rule).agg(
            {'open': 'first', 'high': 'max', 'low': 'min',
             'close': 'last', 'volume': 'sum'}
        ).dropna(subset=['close'])

    tf5m = _resample(df1m, '5min')
    tf1h = _resample(df1m, '1h')
    tf4h = _resample(df1m, '4h')
    return tf5m, tf1h, tf4h


def download_data_local(start: str = '2025-01-01', end: str = '2025-09-26'):
    """Load data from local 1-min files + yfinance for daily/weekly/VIX.

    Local 1-min data is in UTC — converted to ET and filtered to RTH (9:30-16:00 ET).
    SPY 5min is supplemented with yfinance (last 60 days) to extend coverage
    beyond SPYMin.txt's end date.
    """
    import yfinance as yf

    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'data')
    tsla_path = os.path.join(base, 'TSLAMin.txt')
    spy_path = os.path.join(base, 'SPYMin.txt')

    print(f"\nLoading local data ({start} to {end})...")

    # TSLA intraday from 1-min file (UTC → ET, RTH filtered)
    print("  TSLA:")
    tsla_5min, tsla_1h, tsla_4h = _load_min_file(tsla_path, start, end)
    print(f"  TSLA 5min : {len(tsla_5min):,} bars  "
          f"({tsla_5min.index[0]} to {tsla_5min.index[-1]})")

    # SPY intraday from 1-min file (UTC → ET, RTH filtered)
    print("  SPY:")
    spy_5min, spy_1h, spy_4h = _load_min_file(spy_path, start, end)
    spy_local_end = spy_5min.index[-1] if len(spy_5min) > 0 else None
    print(f"  SPY 5min  : {len(spy_5min):,} bars  "
          f"({spy_5min.index[0]} to {spy_5min.index[-1]})")

    # Supplement SPY with yfinance 5min (last 60 days) if local data ends early
    if spy_local_end is not None:
        try:
            yf_spy_5m = _clean_yf(yf.download('SPY', period='60d', interval='5m', progress=False))
            if yf_spy_5m is not None and len(yf_spy_5m) > 0:
                # yfinance data is tz-aware (America/New_York), match local data tz
                if yf_spy_5m.index.tz is None:
                    import pytz
                    yf_spy_5m.index = yf_spy_5m.index.tz_localize('America/New_York')
                elif str(yf_spy_5m.index.tz) != str(spy_5min.index.tz):
                    yf_spy_5m.index = yf_spy_5m.index.tz_convert(spy_5min.index.tz)
                # Only append bars after local data ends
                new_bars = yf_spy_5m[yf_spy_5m.index > spy_local_end]
                if len(new_bars) > 0:
                    spy_5min = pd.concat([spy_5min, new_bars])
                    print(f"  SPY 5min+ : {len(new_bars):,} bars from yfinance "
                          f"(now through {spy_5min.index[-1]})")
        except Exception as e:
            print(f"  SPY yfinance supplement failed: {e}")

    # SPY 1h from yfinance (730 days — fills the gap between local and recent)
    spy_1h_yf = _clean_yf(yf.download('SPY', period='730d', interval='1h', progress=False))
    if spy_1h_yf is not None and len(spy_1h_yf) > 0:
        if spy_1h_yf.index.tz is None:
            import pytz
            spy_1h_yf.index = spy_1h_yf.index.tz_localize('America/New_York')
        elif spy_1h.index.tz is not None and str(spy_1h_yf.index.tz) != str(spy_1h.index.tz):
            spy_1h_yf.index = spy_1h_yf.index.tz_convert(spy_1h.index.tz)
        # Append bars after local 1h data
        if len(spy_1h) > 0:
            new_1h = spy_1h_yf[spy_1h_yf.index > spy_1h.index[-1]]
            if len(new_1h) > 0:
                spy_1h = pd.concat([spy_1h, new_1h])
                print(f"  SPY 1h+   : extended to {spy_1h.index[-1]} via yfinance")

    # Daily, weekly from yfinance (full history, no 60d limit)
    tsla_daily = _clean_yf(yf.download('TSLA', period='5y', interval='1d', progress=False))
    spy_daily = _clean_yf(yf.download('SPY', period='5y', interval='1d', progress=False))
    tsla_weekly = _clean_yf(yf.download('TSLA', period='max', interval='1wk', progress=False))
    spy_weekly = _clean_yf(yf.download('SPY', period='max', interval='1wk', progress=False))
    print(f"  TSLA daily: {len(tsla_daily):,} bars  weekly: {len(tsla_weekly):,}")

    # VIX
    vix_daily = _clean_yf(yf.download('^VIX', period='2y', interval='1d', progress=False))
    print(f"  VIX daily : {len(vix_daily):,} bars")

    return {
        'tsla_5min': tsla_5min, 'spy_5min': spy_5min,
        'tsla_1h': tsla_1h, 'spy_1h': spy_1h,
        'tsla_4h': tsla_4h, 'spy_4h': spy_4h,
        'tsla_daily': tsla_daily, 'spy_daily': spy_daily,
        'tsla_weekly': tsla_weekly, 'spy_weekly': spy_weekly,
        'vix_daily': vix_daily,
    }


def download_data():
    """Download TSLA, SPY, VIX at multiple TFs from yfinance."""
    import yfinance as yf

    print("\nDownloading data from yfinance...")

    # 5min (max 60 days)
    tsla_5min = _clean_yf(yf.download('TSLA', period='60d', interval='5m', progress=False))
    spy_5min = _clean_yf(yf.download('SPY', period='60d', interval='5m', progress=False))
    print(f"  TSLA 5min : {len(tsla_5min):,} bars  "
          f"({tsla_5min.index[0]} to {tsla_5min.index[-1]})")

    # 1h (max 730 days)
    tsla_1h = _clean_yf(yf.download('TSLA', period='730d', interval='1h', progress=False))
    spy_1h = _clean_yf(yf.download('SPY', period='730d', interval='1h', progress=False))
    print(f"  TSLA 1h   : {len(tsla_1h):,} bars")

    # Daily (5 years for good channel history)
    tsla_daily = _clean_yf(yf.download('TSLA', period='5y', interval='1d', progress=False))
    spy_daily = _clean_yf(yf.download('SPY', period='5y', interval='1d', progress=False))
    print(f"  TSLA daily: {len(tsla_daily):,} bars")

    # Weekly
    tsla_weekly = _clean_yf(yf.download('TSLA', period='max', interval='1wk', progress=False))
    spy_weekly = _clean_yf(yf.download('SPY', period='max', interval='1wk', progress=False))
    print(f"  TSLA wkly : {len(tsla_weekly):,} bars")

    # VIX
    vix_daily = _clean_yf(yf.download('^VIX', period='2y', interval='1d', progress=False))
    print(f"  VIX daily : {len(vix_daily):,} bars")

    # 4h: resample from 1h
    tsla_4h = tsla_1h.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    spy_4h = spy_1h.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    print(f"  TSLA 4h   : {len(tsla_4h):,} bars (resampled)")

    return {
        'tsla_5min': tsla_5min, 'spy_5min': spy_5min,
        'tsla_1h': tsla_1h, 'spy_1h': spy_1h,
        'tsla_4h': tsla_4h, 'spy_4h': spy_4h,
        'tsla_daily': tsla_daily, 'spy_daily': spy_daily,
        'tsla_weekly': tsla_weekly, 'spy_weekly': spy_weekly,
        'vix_daily': vix_daily,
    }


def build_native_data(data: dict) -> dict:
    """Build the native_data dict structure that prepare_multi_tf_analysis() expects.

    Structure: {'TSLA': {'5min': df, '1h': df, ...}, 'SPY': {...}, '^VIX': {...}}
    """
    native = {
        'TSLA': {
            '5min': data['tsla_5min'],
            '1h': data['tsla_1h'],
            '4h': data['tsla_4h'],
            'daily': data['tsla_daily'],
            'weekly': data['tsla_weekly'],
        },
        'SPY': {
            '5min': data['spy_5min'],
            '1h': data['spy_1h'],
            '4h': data['spy_4h'],
            'daily': data['spy_daily'],
            'weekly': data['spy_weekly'],
        },
        '^VIX': {
            'daily': data['vix_daily'],
        },
    }
    return native


# ---------------------------------------------------------------------------
# Look-ahead prevention
# ---------------------------------------------------------------------------

def slice_native_data(native_data: dict, current_time: pd.Timestamp) -> dict:
    """Return native_data with only bars that closed BEFORE current_time.

    For higher TFs, a bar's open time < current_time means it closed before
    the current 5-min bar. This prevents look-ahead bias.

    Handles tz-aware/naive mismatch: normalizes comparison to tz-naive.
    """
    # Normalize current_time to tz-naive for comparison
    ct_naive = current_time.tz_localize(None) if current_time.tzinfo is not None else current_time

    sliced = {}
    for symbol, tf_dict in native_data.items():
        sliced[symbol] = {}
        for tf, df in tf_dict.items():
            if df.index.tz is not None:
                # tz-aware index: compare directly with tz-aware current_time
                ct = current_time if current_time.tzinfo is not None else current_time.tz_localize(df.index.tz)
                sliced[symbol][tf] = df[df.index < ct]
            else:
                # tz-naive index: compare with tz-naive current_time
                sliced[symbol][tf] = df[df.index < ct_naive]
    return sliced


# ---------------------------------------------------------------------------
# Per-TF simulation loop
# ---------------------------------------------------------------------------

def _run_tf_sim(data: dict, native_data: dict, sim_start: str, sim_end: str,
                scanner: SimScanner, tf_system: str,
                verbose: bool = True, label: str = '') -> None:
    """Run simulation for a specific TF by walking that TF's native bars.

    For 1h system: walks 1h bars, calls prepare_multi_tf_analysis() with
    target_tfs=['1h','4h','daily','weekly'] so 1h becomes primary_tf.
    """
    from v15.core.channel_surfer import prepare_multi_tf_analysis

    config = TF_SIM_CONFIG[tf_system]
    walk_bars = data[config['data_key']]
    spy_walk = data[config['spy_data_key']]
    target_tfs = config['target_tfs']
    native_tf = config['native_tf']

    # Override timeout for this TF
    scanner.TIMEOUT_MINUTES = config['timeout_minutes']

    total_bars = len(walk_bars)

    # Build sim window timestamps matching walking bars' tz
    sim_start_ts = pd.Timestamp(sim_start)
    sim_end_ts = pd.Timestamp(sim_end) + pd.Timedelta(hours=23, minutes=59)
    walk_tz = walk_bars.index.tz
    if walk_tz is not None:
        if sim_start_ts.tzinfo is None:
            sim_start_ts = sim_start_ts.tz_localize(walk_tz)
        if sim_end_ts.tzinfo is None:
            sim_end_ts = sim_end_ts.tz_localize(walk_tz)
    else:
        if sim_start_ts.tzinfo is not None:
            sim_start_ts = sim_start_ts.tz_localize(None)
        if sim_end_ts.tzinfo is not None:
            sim_end_ts = sim_end_ts.tz_localize(None)

    sim_bars = 0
    signals_evaluated = 0
    last_progress = -10

    if verbose:
        print(f"\n{'- '*40}")
        print(f"  {label}" if label else f"  Running {tf_system} simulation...")
        print(f"  Walking {tf_system} bars: {total_bars:,}  |  "
              f"Capital: ${scanner.equity:,.0f}  |  target_tfs: {target_tfs}")

    t_start = time.time()

    for bar_idx in range(total_bars):
        bar = walk_bars.iloc[bar_idx]
        bar_time = walk_bars.index[bar_idx]
        price = float(bar['close'])
        high = float(bar['high'])
        low = float(bar['low'])

        # Progress
        if verbose:
            pct = int(bar_idx / total_bars * 100)
            if pct >= last_progress + 10:
                elapsed = time.time() - t_start
                rate = bar_idx / elapsed if elapsed > 0 else 0
                eta = (total_bars - bar_idx) / rate if rate > 0 else 0
                print(f"    {pct}% ({bar_idx:,}/{total_bars:,} bars, "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                      f"{len(scanner.closed_trades)} trades)")
                last_progress = pct

        # 1. Check exits FIRST (runs on all bars, not just sim window)
        closed = scanner.check_exits(price, high, low, bar_time)
        if closed and verbose:
            for t in closed:
                print(f"    EXIT  {bar_time}  {t.direction.upper():<5} "
                      f"${t.entry_price:.2f}->${t.exit_price:.2f}  "
                      f"P&L=${t.pnl:>+8,.0f}  ({t.exit_reason})  "
                      f"[{t.primary_tf}/{t.signal_type}]")

        # Only evaluate signals during sim window
        if bar_time < sim_start_ts or bar_time > sim_end_ts:
            continue

        sim_bars += 1

        # 2. Slice native data to prevent look-ahead on higher TFs
        sliced = slice_native_data(native_data, bar_time)

        # 3. Override walking TF to include current bar (inclusive)
        sliced['TSLA'][native_tf] = walk_bars.iloc[:bar_idx + 1]

        # SPY at walking TF: include bars up to current time
        bt_spy = bar_time
        if spy_walk.index.tz is not None and bt_spy.tzinfo is None:
            bt_spy = bt_spy.tz_localize(spy_walk.index.tz)
        elif spy_walk.index.tz is None and bt_spy.tzinfo is not None:
            bt_spy = bt_spy.tz_localize(None)
        sliced['SPY'][native_tf] = spy_walk[spy_walk.index <= bt_spy]

        # 4. Generate signal
        # For 5min system, pass live_5min_tsla; for others, None
        live_5min = None
        if tf_system == '5min':
            live_5min = data['tsla_5min']
            ct_5m = bar_time
            if live_5min.index.tz is not None and ct_5m.tzinfo is None:
                ct_5m = ct_5m.tz_localize(live_5min.index.tz)
            elif live_5min.index.tz is None and ct_5m.tzinfo is not None:
                ct_5m = ct_5m.tz_localize(None)
            live_5min = live_5min[live_5min.index <= ct_5m]

        try:
            analysis = prepare_multi_tf_analysis(
                native_data=sliced,
                live_5min_tsla=live_5min,
                target_tfs=target_tfs,
            )
        except Exception as e:
            if verbose and sim_bars <= 5:
                print(f"    [WARN] Signal error at {bar_time}: {e}")
            continue

        signals_evaluated += 1

        # 5. Evaluate entry (no tf_filter needed - target_tfs already constrains)
        entered = scanner.evaluate_signal(analysis, price, bar_time)
        if entered and verbose:
            latest_pos = list(scanner.positions.values())[-1]
            print(f"    ENTRY {bar_time}  {latest_pos.direction.upper():<5} "
                  f"${price:.2f}  {latest_pos.shares} shares  "
                  f"conf={latest_pos.confidence:.2f}  "
                  f"[{latest_pos.primary_tf}/{latest_pos.signal_type}]  "
                  f"SL=${latest_pos.stop_price:.2f} TP=${latest_pos.tp_price:.2f}")

    # Force-close remaining positions
    if len(walk_bars) > 0 and scanner.positions:
        last_price = float(walk_bars.iloc[-1]['close'])
        last_time = walk_bars.index[-1]
        remaining = scanner.force_close_all(last_price, last_time)
        if verbose:
            for t in remaining:
                print(f"    CLOSE {last_time}  {t.direction.upper():<5}  "
                      f"P&L=${t.pnl:>+8,.0f}  (sim_end)")

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n  {tf_system} complete in {elapsed:.1f}s  |  "
              f"Sim bars: {sim_bars:,}  |  Signals: {signals_evaluated:,}  |  "
              f"Trades: {len(scanner.closed_trades)}")


# ---------------------------------------------------------------------------
# Main simulation loop (5min baseline)
# ---------------------------------------------------------------------------

def _run_sim_core(data: dict, native_data: dict, sim_start_ts, sim_end_ts,
                  scanners: Dict[str, SimScanner],
                  tf_filters: Dict[str, Optional[set]],
                  verbose: bool = True, label: str = ''):
    """Core simulation loop shared by single and portfolio modes.

    Args:
        scanners: Dict of name -> SimScanner (one for single mode, N for portfolio)
        tf_filters: Dict of name -> set of allowed TFs (or None for all)
    """
    from v15.core.channel_surfer import prepare_multi_tf_analysis

    tsla_5min = data['tsla_5min']
    total_bars = len(tsla_5min)
    sim_bars = 0
    signals_evaluated = 0
    last_progress = -10

    if verbose:
        print(f"\n{'- '*40}")
        print(f"  {label}" if label else "  Running simulation...")
        total_cap = sum(s.equity for s in scanners.values())
        print(f"  Total 5min bars: {total_bars:,}  |  Capital: ${total_cap:,.0f}")

    t_start = time.time()

    for bar_idx in range(total_bars):
        bar = tsla_5min.iloc[bar_idx]
        bar_time = tsla_5min.index[bar_idx]
        price = float(bar['close'])
        high = float(bar['high'])
        low = float(bar['low'])

        # Progress reporting
        if verbose:
            pct = int(bar_idx / total_bars * 100)
            if pct >= last_progress + 10:
                elapsed = time.time() - t_start
                rate = bar_idx / elapsed if elapsed > 0 else 0
                eta = (total_bars - bar_idx) / rate if rate > 0 else 0
                total_trades = sum(len(s.closed_trades) for s in scanners.values())
                print(f"    {pct}% ({bar_idx:,}/{total_bars:,} bars, "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                      f"{total_trades} trades)")
                last_progress = pct

        # 1. Check exits FIRST for all scanners
        for name, scanner in scanners.items():
            closed = scanner.check_exits(price, high, low, bar_time)
            if closed and verbose:
                for t in closed:
                    prefix = f"[{name}] " if len(scanners) > 1 else ""
                    print(f"    {prefix}EXIT  {bar_time}  {t.direction.upper():<5} "
                          f"${t.entry_price:.2f}->${t.exit_price:.2f}  "
                          f"P&L=${t.pnl:>+8,.0f}  ({t.exit_reason})  "
                          f"[{t.primary_tf}/{t.signal_type}]")

        # Only evaluate signals during sim window
        if bar_time < sim_start_ts or bar_time > sim_end_ts:
            continue

        sim_bars += 1

        # 2. Slice native_data to prevent look-ahead
        sliced_native = slice_native_data(native_data, bar_time)

        # 3. Slice 5min data up to current bar (inclusive)
        sliced_native['TSLA']['5min'] = tsla_5min.iloc[:bar_idx + 1]
        sliced_native['SPY']['5min'] = data['spy_5min'].iloc[:min(bar_idx + 1, len(data['spy_5min']))]

        # 4. Generate signal via prepare_multi_tf_analysis()
        try:
            analysis = prepare_multi_tf_analysis(
                native_data=sliced_native,
                live_5min_tsla=tsla_5min.iloc[:bar_idx + 1],
            )
        except Exception as e:
            if verbose and sim_bars <= 5:
                print(f"    [WARN] Signal error at {bar_time}: {e}")
            continue

        signals_evaluated += 1

        # 5. Evaluate entry for each scanner (with its TF filter)
        for name, scanner in scanners.items():
            tf_filt = tf_filters.get(name)
            entered = scanner.evaluate_signal(analysis, price, bar_time, tf_filter=tf_filt)
            if entered and verbose:
                latest_pos = list(scanner.positions.values())[-1]
                prefix = f"[{name}] " if len(scanners) > 1 else ""
                print(f"    {prefix}ENTRY {bar_time}  {latest_pos.direction.upper():<5} "
                      f"${price:.2f}  {latest_pos.shares} shares  "
                      f"conf={latest_pos.confidence:.2f}  "
                      f"[{latest_pos.primary_tf}/{latest_pos.signal_type}]  "
                      f"SL=${latest_pos.stop_price:.2f} TP=${latest_pos.tp_price:.2f}")

    # Force-close remaining positions
    last_price = float(tsla_5min.iloc[-1]['close'])
    last_time = tsla_5min.index[-1]
    for name, scanner in scanners.items():
        if scanner.positions:
            remaining = scanner.force_close_all(last_price, last_time)
            if verbose:
                for t in remaining:
                    prefix = f"[{name}] " if len(scanners) > 1 else ""
                    print(f"    {prefix}CLOSE {last_time}  {t.direction.upper():<5}  "
                          f"P&L=${t.pnl:>+8,.0f}  (sim_end)")

    elapsed = time.time() - t_start
    if verbose:
        total_trades = sum(len(s.closed_trades) for s in scanners.values())
        print(f"\n  Complete in {elapsed:.1f}s  |  "
              f"Sim bars: {sim_bars:,}  |  Signals: {signals_evaluated:,}  |  "
              f"Trades: {total_trades}")


def run_simulation(data: dict, sim_start: str, sim_end: str,
                   capital: float = 100_000, verbose: bool = True) -> SimScanner:
    """Run a single forward simulation (all TFs accepted).

    ALL bars are processed (channel warmup from bar 0), but entries only
    count from sim_start onward.
    """
    native_data = build_native_data(data)
    tsla_5min = data['tsla_5min']

    sim_start_ts = pd.Timestamp(sim_start)
    sim_end_ts = pd.Timestamp(sim_end) + pd.Timedelta(hours=23, minutes=59)
    if tsla_5min.index.tz is not None:
        if sim_start_ts.tzinfo is None:
            sim_start_ts = sim_start_ts.tz_localize(tsla_5min.index.tz)
        if sim_end_ts.tzinfo is None:
            sim_end_ts = sim_end_ts.tz_localize(tsla_5min.index.tz)

    scanner = SimScanner(capital)
    scanners = {'all': scanner}
    tf_filters = {'all': None}

    _run_sim_core(data, native_data, sim_start_ts, sim_end_ts,
                  scanners, tf_filters, verbose=verbose,
                  label=f'SINGLE RUN (all TFs): {sim_start} to {sim_end}, ${capital:,.0f}')

    return scanner


def run_portfolio_simulation(data: dict, sim_start: str, sim_end: str,
                             allocations: Dict[str, Tuple[set, float]],
                             total_capital: float = 100_000,
                             verbose: bool = True) -> Dict[str, SimScanner]:
    """Run portfolio simulation with per-TF native bar walking.

    Each system walks its own TF's native bars and calls
    prepare_multi_tf_analysis() with target_tfs that exclude lower TFs,
    making the target TF the primary signal source.

    Args:
        allocations: Dict of system_name -> (set_of_tfs, weight).
            e.g. {'1h': ({'1h'}, 0.20), '1d': ({'daily'}, 0.80)}
            The first TF in each set determines which bars to walk.
        total_capital: Total capital to split across systems.

    Returns:
        Dict of system_name -> SimScanner with results.
    """
    native_data = build_native_data(data)

    scanners: Dict[str, SimScanner] = {}

    alloc_parts = []
    for name, (tfs, weight) in allocations.items():
        alloc_parts.append(f"{name}={weight:.0%}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  PORTFOLIO ({', '.join(alloc_parts)})")
        print(f"  {sim_start} to {sim_end}, ${total_capital:,.0f}")
        print(f"{'='*60}")

    # Map allocation TF names to TF_SIM_CONFIG keys
    tf_name_map = {
        '5min': '5min', '1h': '1h', '4h': '4h',
        'daily': 'daily', '1d': 'daily',
        'weekly': 'weekly',
    }

    for name, (tfs, weight) in allocations.items():
        cap = total_capital * weight
        scanner = SimScanner(cap)

        # Find the TF system to walk (first TF in the set that has a config)
        tf_system = None
        for tf in tfs:
            mapped = tf_name_map.get(tf)
            if mapped and mapped in TF_SIM_CONFIG:
                tf_system = mapped
                break

        if tf_system is None:
            if verbose:
                print(f"\n  [WARN] No TF config for '{name}' (tfs={tfs}), skipping")
            scanners[name] = scanner
            continue

        label = f"{name} system ({weight:.0%} = ${cap:,.0f})"
        _run_tf_sim(data, native_data, sim_start, sim_end,
                    scanner, tf_system, verbose=verbose, label=label)

        scanners[name] = scanner

    return scanners


def print_portfolio_results(scanners: Dict[str, SimScanner],
                            allocations: Dict[str, Tuple[set, float]],
                            sim_start: str, sim_end: str, total_capital: float):
    """Print formatted portfolio results."""
    alloc_str = ' + '.join(f"{w:.0%} {n}" for n, (_, w) in allocations.items())

    print(f"\n{'='*80}")
    print(f"PORTFOLIO: {alloc_str}")
    print(f"Period: {sim_start} to {sim_end}  |  Capital: ${total_capital:,.0f}")
    print(f"{'='*80}")

    # Per-system summary
    print(f"\nPER-SYSTEM BREAKDOWN:")
    print(f"  {'System':<10} {'Capital':>10} {'Trades':>7} {'P&L':>12} {'WR':>6} "
          f"{'ROI':>8} {'Final Eq':>12}")
    print(f"  {'-'*68}")

    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    all_trades: List[SimClosedTrade] = []

    for name, (tfs, weight) in allocations.items():
        scanner = scanners[name]
        cap = total_capital * weight
        trades = scanner.closed_trades
        n = len(trades)
        pnl = sum(t.pnl for t in trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / n * 100 if n > 0 else 0
        roi = pnl / cap * 100 if cap > 0 else 0

        total_pnl += pnl
        total_trades += n
        total_wins += wins
        all_trades.extend(trades)

        print(f"  {name:<10} ${cap:>9,.0f} {n:>7} ${pnl:>+10,.0f} {wr:>5.0f}% "
              f"{roi:>+7.2f}% ${scanner.equity:>11,.0f}")

    # Combined trade list (sorted by time)
    all_trades.sort(key=lambda t: t.entry_time)

    if all_trades:
        print(f"\nALL TRADES (chronological):")
        print(f"  {'#':>3}  {'Time':<20} {'Dir':<6} {'Price':>8} {'Shares':>7} "
              f"{'Type':<8} {'TF':<6} {'Exit':<15} {'P&L':>10}")
        print(f"  {'-'*95}")

        for i, t in enumerate(all_trades, 1):
            time_str = t.entry_time.strftime('%m/%d %H:%M') if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:16]
            print(f"  {i:>3}  {time_str:<20} {t.direction.upper():<6} "
                  f"${t.entry_price:>7.2f} {t.shares:>7} "
                  f"{t.signal_type:<8} {t.primary_tf:<6} {t.exit_reason:<15} "
                  f"${t.pnl:>+9,.0f}")

    # Daily summary
    daily: Dict[str, List[SimClosedTrade]] = {}
    for t in all_trades:
        day = t.entry_time.strftime('%Y-%m-%d') if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:10]
        daily.setdefault(day, []).append(t)

    if daily:
        print(f"\nDAILY SUMMARY:")
        print(f"  {'Date':<12} {'Trades':>7} {'P&L':>12}")
        print(f"  {'-'*35}")
        for day in sorted(daily.keys()):
            dt = daily[day]
            day_pnl = sum(t.pnl for t in dt)
            print(f"  {day:<12} {len(dt):>7} ${day_pnl:>+10,.0f}")

    # Totals
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    roi = total_pnl / total_capital * 100
    final_equity = sum(s.equity for s in scanners.values())
    max_dd = max(s.max_drawdown for s in scanners.values()) * 100

    print(f"\n  TOTALS:  Trades={total_trades}  |  WR={wr:.0f}%  |  "
          f"P&L=${total_pnl:>+,.0f}  |  ROI={roi:>+.2f}%  |  "
          f"Final=${final_equity:>,.0f}  |  MaxDD={max_dd:.2f}%")
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Multi-scanner simulation (all 4 signal types)
# ---------------------------------------------------------------------------

def run_all_scanners_sim(data: dict, sim_start: str, sim_end: str,
                         capital: float = 100_000,
                         hard_stop_pct: Optional[float] = None,
                         equity_mode: str = 'isolated',
                         am_only: bool = False,
                         late_hard_stop_pct: Optional[float] = None,
                         vix_am_threshold: Optional[float] = None,
                         dw_eod_close: bool = True,
                         trade_gate=None,
                         am_cutoff_minutes: int = 60,
                         late_short_hard_stop_pct: Optional[float] = None,
                         verbose: bool = True) -> Dict[str, SimScanner]:
    """Run simulation with all 4 scanner types on 5-min bars.

    Scanner types:
      CS-5TF  — Channel Surfer, all 5 TFs, conf-scaled sizing, exponential trail
      CS-DW   — Channel Surfer, daily+weekly only, conf-scaled sizing, exponential trail
      ML      — Surfer ML, all TFs, risk-based sizing, profit-tier trail
      Intra   — Intraday, 5-min features + sig_union_enhanced, flat sizing

    Args:
        equity_mode: 'isolated' = each scanner gets $capital independently.
                     'shared' = all scanners draw from one pool.
        am_only: If True, CS-5TF/CS-DW/ML only enter during first hour (9:30-10:30 ET).
                 Intraday unaffected (has its own 13:00-15:25 window).
        late_hard_stop_pct: If set, CS-5TF/CS-DW/ML entries AFTER 10:30 ET get this
                            tighter per-position hard stop (e.g. 0.006 = 0.6%).
        vix_am_threshold: If set, restrict CS/ML to first hour ONLY when previous
                          day's VIX close >= this threshold. Ignored if am_only=True.
        dw_eod_close: If False, CS-DW positions are NOT force-closed at EOD,
                      allowing multi-day holds for daily/weekly TF signals.
        am_cutoff_minutes: Minutes after 9:30 ET for AM cutoff (default 60 = 10:30).
                           Use 45 for 10:15 cutoff.
        late_short_hard_stop_pct: If set, SHORT entries by CS/ML AFTER the AM cutoff
                                  get this tight hard stop (e.g. 0.001 = 0.1%).
                                  Allows late shorts but with very tight risk.
    """
    from v15.core.channel_surfer import prepare_multi_tf_analysis
    from v15.validation.intraday_features import precompute_5min_features

    native_data = build_native_data(data)
    tsla_5min = data['tsla_5min']
    total_bars = len(tsla_5min)

    # Create 4 scanners
    scanner_configs = {
        'CS-5TF': {'signal_source': 'CS-5TF', 'tf_filter': None},
        'CS-DW':  {'signal_source': 'CS-DW',  'tf_filter': {'daily', 'weekly'}},
        'ML':     {'signal_source': 'surfer_ml', 'tf_filter': None},
        'Intra':  {'signal_source': 'intraday', 'tf_filter': None},
    }

    scanners: Dict[str, SimScanner] = {}
    for name, cfg in scanner_configs.items():
        eod = True
        if name == 'CS-DW' and not dw_eod_close:
            eod = False
        scanners[name] = SimScanner(
            initial_capital=capital,
            signal_source=cfg['signal_source'],
            hard_stop_pct=hard_stop_pct,
            eod_close_enabled=eod,
        )
        # CS and ML use full buying power (no 25% cap)
        if cfg['signal_source'] in ('CS-5TF', 'CS-DW', 'intraday'):
            scanners[name].MAX_BUYING_POWER_PCT = 1.0

    # Shared equity: all scanners reference a common equity pool
    shared_equity = [capital] if equity_mode == 'shared' else None
    if shared_equity:
        for s in scanners.values():
            s._shared_equity = shared_equity  # Monkey-patch for shared mode

    # Precompute intraday features from 5-min bars
    if verbose:
        print("  Precomputing intraday features from 5-min bars...")
    intraday_feats = precompute_5min_features(tsla_5min)

    # Build VIX lookup: date → previous trading day's VIX close
    vix_daily = data.get('vix_daily')
    _vix_by_date = {}
    if vix_daily is not None and len(vix_daily) > 0:
        vix_closes = vix_daily['close'].dropna()
        vix_dates = sorted(vix_closes.index)
        for i in range(1, len(vix_dates)):
            # For date vix_dates[i], the "previous day VIX" is vix_dates[i-1]'s close
            d = vix_dates[i].date() if hasattr(vix_dates[i], 'date') else vix_dates[i]
            prev_close = float(vix_closes.iloc[i - 1])
            _vix_by_date[d] = prev_close

    # Build calendar event sets for trade gate
    _event_dates = set()
    _near_event_dates = set()
    if trade_gate is not None:
        try:
            from v15.validation.market_calendar import CALENDAR_EVENTS
            from datetime import timedelta as _td
            for d_str in CALENDAR_EVENTS:
                _event_dates.add(d_str)
                d = pd.Timestamp(d_str).date()
                for offset in range(1, 4):
                    for sign in (-1, 1):
                        nd = d + _td(days=offset * sign)
                        if nd.weekday() < 5:
                            _near_event_dates.add(str(nd))
                            break
        except ImportError:
            pass

    # Build sim window timestamps
    sim_start_ts = pd.Timestamp(sim_start)
    sim_end_ts = pd.Timestamp(sim_end) + pd.Timedelta(hours=23, minutes=59)
    walk_tz = tsla_5min.index.tz
    if walk_tz is not None:
        if sim_start_ts.tzinfo is None:
            sim_start_ts = sim_start_ts.tz_localize(walk_tz)
        if sim_end_ts.tzinfo is None:
            sim_end_ts = sim_end_ts.tz_localize(walk_tz)

    sim_bars = 0
    last_progress = -10

    if verbose:
        _cutoff_time_str = f"9:30-{9 + (30 + am_cutoff_minutes) // 60}:{(30 + am_cutoff_minutes) % 60:02d} ET"
        hs_str = f"  Hard stop: {hard_stop_pct:.0%}" if hard_stop_pct else "  Hard stop: None"
        am_str = f"  AM-only: CS/ML entries restricted to {_cutoff_time_str}" if am_only else ""
        if vix_am_threshold is not None and not am_only:
            am_str = f"  VIX-gated AM-only: CS/ML restricted to first {am_cutoff_minutes}min when prev VIX >= {vix_am_threshold}"
        late_str = f"  Late hard stop: {late_hard_stop_pct:.1%} for CS/ML entries after {_cutoff_time_str}" if late_hard_stop_pct else ""
        if late_short_hard_stop_pct is not None:
            late_str += f"\n  Late SHORT hard stop: {late_short_hard_stop_pct:.1%} for CS/ML shorts after {_cutoff_time_str}"
        print(f"\n{'- '*40}")
        print(f"  ALL 4 SCANNERS: {sim_start} to {sim_end}")
        print(f"  Capital: ${capital:,.0f} per scanner ({equity_mode} mode)")
        print(hs_str)
        if am_str:
            print(am_str)
        if late_str:
            print(late_str)
        print(f"  Total 5min bars: {total_bars:,}")

    t_start = time.time()

    for bar_idx in range(total_bars):
        bar = tsla_5min.iloc[bar_idx]
        bar_time = tsla_5min.index[bar_idx]
        price = float(bar['close'])
        high = float(bar['high'])
        low = float(bar['low'])

        # Progress
        if verbose:
            pct = int(bar_idx / total_bars * 100)
            if pct >= last_progress + 10:
                elapsed = time.time() - t_start
                rate = bar_idx / elapsed if elapsed > 0 else 0
                eta = (total_bars - bar_idx) / rate if rate > 0 else 0
                total_trades = sum(len(s.closed_trades) for s in scanners.values())
                print(f"    {pct}% ({bar_idx:,}/{total_bars:,} bars, "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                      f"{total_trades} trades)")
                last_progress = pct

        # Sync shared equity before exit checks
        if shared_equity:
            for s in scanners.values():
                s.equity = shared_equity[0]

        # 1. Check exits for all scanners
        for name, scanner in scanners.items():
            closed = scanner.check_exits(price, high, low, bar_time)
            if shared_equity and closed:
                # Update shared pool from this scanner's equity changes
                shared_equity[0] = scanner.equity
            if closed and verbose:
                for t in closed:
                    print(f"    [{name}] EXIT  {bar_time}  {t.direction.upper():<5} "
                          f"${t.entry_price:.2f}->${t.exit_price:.2f}  "
                          f"P&L=${t.pnl:>+8,.0f}  ({t.exit_reason})")

        # Only evaluate signals during sim window
        if bar_time < sim_start_ts or bar_time > sim_end_ts:
            continue

        sim_bars += 1

        # 2. Slice native_data to prevent look-ahead
        sliced_native = slice_native_data(native_data, bar_time)
        sliced_native['TSLA']['5min'] = tsla_5min.iloc[:bar_idx + 1]
        sliced_native['SPY']['5min'] = data['spy_5min'].iloc[:min(bar_idx + 1, len(data['spy_5min']))]

        # 3. Generate signal via prepare_multi_tf_analysis() (shared by CS-5TF, ML)
        analysis = None
        try:
            analysis = prepare_multi_tf_analysis(
                native_data=sliced_native,
                live_5min_tsla=tsla_5min.iloc[:bar_idx + 1],
            )
        except Exception:
            pass

        # Sync shared equity before entries
        if shared_equity:
            for s in scanners.values():
                s.equity = shared_equity[0]

        # Determine if we're past the AM cutoff (default 60min = 10:30 ET)
        _in_first_hour = False
        _past_first_hour = False
        try:
            import pytz as _pz
            _et = _pz.timezone('US/Eastern')
            if bar_time.tzinfo is not None:
                _bt_et = bar_time.astimezone(_et)
            else:
                _bt_et = _et.localize(bar_time)
            from datetime import time as _time
            _bt_tod = _bt_et.time()
            _cutoff_h = 9 + (30 + am_cutoff_minutes) // 60
            _cutoff_m = (30 + am_cutoff_minutes) % 60
            _cutoff_time = _time(_cutoff_h, _cutoff_m)
            _in_first_hour = _time(9, 30) <= _bt_tod <= _cutoff_time
            _past_first_hour = _bt_tod > _cutoff_time
        except Exception:
            pass

        # AM-only / VIX-gated AM-only: skip CS/ML entries after cutoff
        _allow_cs_ml = True
        _late_short_mode = False  # True when shorts allowed but with tight HS
        _late_all_mode = False  # True when all trades allowed but with tight HS
        if am_only and _past_first_hour:
            if late_hard_stop_pct is not None:
                # Don't block — allow all trades through with tight hard stop
                _late_all_mode = True
            elif late_short_hard_stop_pct is not None:
                # Don't block — allow shorts through with tight hard stop
                _late_short_mode = True
            else:
                _allow_cs_ml = False
        elif vix_am_threshold is not None and _past_first_hour:
            _bar_date = bar_time.date() if hasattr(bar_time, 'date') else bar_time
            if getattr(bar_time, 'tzinfo', None) is not None:
                import pytz as _pz2
                _bar_date = bar_time.astimezone(_pz2.timezone('US/Eastern')).date()
            _prev_vix = _vix_by_date.get(_bar_date, 0)
            if _prev_vix >= vix_am_threshold:
                _allow_cs_ml = False

        # 4. Evaluate CS-5TF and ML (same analysis, different sizing/exits)
        if analysis is not None and _allow_cs_ml:
            # Pre-extract signal info for trade gate
            _sig = analysis.signal
            _sig_action = getattr(_sig, 'action', 'HOLD')
            _sig_conf = getattr(_sig, 'confidence', 0.0)
            _sig_tf = getattr(_sig, 'primary_tf', '5min')
            _sig_type = getattr(_sig, 'signal_type', 'bounce')

            # Trade gate: check if this signal should be allowed
            _gate_ok = True
            if trade_gate is not None and _sig_action != 'HOLD':
                _scanner_map = {'CS-5TF': 0.0, 'CS-DW': 1.0, 'ML': 2.0}
                _sigtype_map = {'bounce': 0.0, 'break': 1.0}
                _dir_map = {'BUY': 0.0, 'SELL': 1.0}
                _tf_map = {'5min': 0.0, '1h': 1.0, '4h': 2.0, 'daily': 3.0, 'weekly': 4.0}
                _bar_date_g = _bt_et.date() if '_bt_et' in dir() else bar_time.date()
                _prev_vix_g = _vix_by_date.get(_bar_date_g, 0)
                _entry_hr_g = _bt_et.hour + _bt_et.minute / 60.0 if '_bt_et' in dir() else 12.0
                _dow_g = float(_bar_date_g.weekday())
                _ev_day = 1.0 if str(_bar_date_g) in _event_dates else 0.0
                _near_ev = 1.0 if str(_bar_date_g) in _near_event_dates else 0.0

            for name in ('CS-5TF', 'ML'):
                scanner = scanners[name]
                tf_filt = scanner_configs[name]['tf_filter']

                # Late short mode: only allow SELL signals after AM cutoff
                if _late_short_mode and _sig_action != 'SELL':
                    continue

                # Apply trade gate before entry
                if trade_gate is not None and _sig_action != 'HOLD' and _sig_conf >= scanner.MIN_CONFIDENCE:
                    try:
                        _gate_ok = trade_gate(
                            scanner=_scanner_map.get(name, 0.0),
                            signal_type=_sigtype_map.get(_sig_type, 0.0),
                            direction=_dir_map.get(_sig_action, 0.0),
                            primary_tf=_tf_map.get(_sig_tf, 0.0),
                            confidence=_sig_conf,
                            entry_hour=_entry_hr_g,
                            prev_vix=_prev_vix_g,
                            day_of_week=_dow_g,
                            is_event_day=_ev_day,
                            is_near_event=_near_ev,
                        )
                    except Exception:
                        _gate_ok = True

                if not _gate_ok:
                    continue

                entered = scanner.evaluate_signal(analysis, price, bar_time, tf_filter=tf_filt)
                if entered:
                    # Apply late short hard stop
                    if _late_short_mode and late_short_hard_stop_pct is not None:
                        pos = list(scanner.positions.values())[-1]
                        pos.pos_hard_stop_pct = late_short_hard_stop_pct
                    # Apply late hard stop for entries after first hour
                    elif late_hard_stop_pct is not None and _past_first_hour:
                        pos = list(scanner.positions.values())[-1]
                        pos.pos_hard_stop_pct = late_hard_stop_pct
                    if shared_equity:
                        shared_equity[0] = scanner.equity
                    if verbose:
                        pos = list(scanner.positions.values())[-1]
                        hs_tag = f" HS={pos.pos_hard_stop_pct:.1%}" if pos.pos_hard_stop_pct else ""
                        print(f"    [{name}] ENTRY {bar_time}  {pos.direction.upper():<5} "
                              f"${price:.2f}  {pos.shares}sh  conf={pos.confidence:.2f}  "
                              f"[{pos.primary_tf}/{pos.signal_type}]{hs_tag}")

        # 5. Evaluate CS-DW (uses same analysis but with tf_filter)
        if analysis is not None and _allow_cs_ml:
            scanner = scanners['CS-DW']

            # Apply trade gate for CS-DW
            _gate_ok_dw = True
            if trade_gate is not None and _sig_action != 'HOLD' and _sig_conf >= scanner.MIN_CONFIDENCE:
                try:
                    _gate_ok_dw = trade_gate(
                        scanner=1.0,  # CS-DW
                        signal_type=_sigtype_map.get(_sig_type, 0.0),
                        direction=_dir_map.get(_sig_action, 0.0),
                        primary_tf=_tf_map.get(_sig_tf, 0.0),
                        confidence=_sig_conf,
                        entry_hour=_entry_hr_g,
                        prev_vix=_prev_vix_g,
                        day_of_week=_dow_g,
                        is_event_day=_ev_day,
                        is_near_event=_near_ev,
                    )
                except Exception:
                    _gate_ok_dw = True

            if _gate_ok_dw:
                # Late short mode: only allow SELL signals after AM cutoff
                if _late_short_mode and _sig_action != 'SELL':
                    pass  # Skip non-short entries in late short mode
                else:
                    entered = scanner.evaluate_signal(analysis, price, bar_time,
                                                      tf_filter={'daily', 'weekly'})
                    if entered:
                        # Apply late short hard stop
                        if _late_short_mode and late_short_hard_stop_pct is not None:
                            pos = list(scanner.positions.values())[-1]
                            pos.pos_hard_stop_pct = late_short_hard_stop_pct
                        # Apply late hard stop for entries after first hour
                        elif late_hard_stop_pct is not None and _past_first_hour:
                            pos = list(scanner.positions.values())[-1]
                            pos.pos_hard_stop_pct = late_hard_stop_pct
                        if shared_equity:
                            shared_equity[0] = scanner.equity
                        if verbose:
                            pos = list(scanner.positions.values())[-1]
                            hs_tag = f" HS={pos.pos_hard_stop_pct:.1%}" if pos.pos_hard_stop_pct else ""
                            print(f"    [CS-DW] ENTRY {bar_time}  {pos.direction.upper():<5} "
                                  f"${price:.2f}  {pos.shares}sh  conf={pos.confidence:.2f}  "
                                  f"[{pos.primary_tf}/{pos.signal_type}]{hs_tag}")

        # 6. Evaluate Intraday (uses precomputed features + analysis TF states)
        intra_features = dict(intraday_feats)  # Copy array refs
        # Extract channel positions from analysis if available
        if analysis is not None and hasattr(analysis, 'tf_states'):
            tf_st = analysis.tf_states
            for tf_key, feat_key in [('5min', 'cp5'), ('daily', 'daily_cp'),
                                     ('1h', 'h1_cp'), ('4h', 'h4_cp')]:
                if tf_key in tf_st and tf_st[tf_key].valid:
                    intra_features[feat_key] = tf_st[tf_key].position_pct
                else:
                    intra_features[feat_key] = float('nan')
            for tf_key, feat_key in [('daily', 'daily_slope'),
                                     ('1h', 'h1_slope'), ('4h', 'h4_slope')]:
                if tf_key in tf_st and tf_st[tf_key].valid:
                    intra_features[feat_key] = tf_st[tf_key].slope_pct
                else:
                    intra_features[feat_key] = float('nan')
        else:
            for k in ('cp5', 'daily_cp', 'h1_cp', 'h4_cp',
                      'daily_slope', 'h1_slope', 'h4_slope'):
                intra_features[k] = float('nan')

        scanner = scanners['Intra']
        entered = scanner.evaluate_intraday(intra_features, price, bar_time, bar_idx)
        if shared_equity and entered:
            shared_equity[0] = scanner.equity
        if entered and verbose:
            pos = list(scanner.positions.values())[-1]
            print(f"    [Intra] ENTRY {bar_time}  LONG  "
                  f"${price:.2f}  {pos.shares}sh  conf={pos.confidence:.2f}")

    # Force-close remaining positions
    last_price = float(tsla_5min.iloc[-1]['close'])
    last_time = tsla_5min.index[-1]
    for name, scanner in scanners.items():
        if scanner.positions:
            remaining = scanner.force_close_all(last_price, last_time)
            if shared_equity:
                shared_equity[0] = scanner.equity
            if verbose:
                for t in remaining:
                    print(f"    [{name}] CLOSE {last_time}  P&L=${t.pnl:>+8,.0f}  (sim_end)")

    elapsed = time.time() - t_start
    if verbose:
        total_trades = sum(len(s.closed_trades) for s in scanners.values())
        print(f"\n  Complete in {elapsed:.1f}s  |  Sim bars: {sim_bars:,}  |  "
              f"Trades: {total_trades}")

    return scanners


def print_multi_scanner_results(scanners: Dict[str, SimScanner],
                                sim_start: str, sim_end: str,
                                capital: float, equity_mode: str,
                                hard_stop_pct: Optional[float] = None,
                                label: str = ''):
    """Print formatted results for all 4 scanner types."""
    hs_str = f"{hard_stop_pct:.0%}" if hard_stop_pct else "None"

    print(f"\n{'='*80}")
    if label:
        print(f"{label}")
    print(f"ALL 4 SCANNERS: {sim_start} to {sim_end}")
    print(f"Capital: ${capital:,.0f}/scanner ({equity_mode})  |  Hard stop: {hs_str}")
    print(f"{'='*80}")

    # Per-scanner summary
    print(f"\n  {'Scanner':<10} {'Trades':>7} {'Wins':>5} {'WR':>6} {'P&L':>12} "
          f"{'MaxDD':>7} {'Final Eq':>12}")
    print(f"  {'-'*65}")

    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    all_trades: List[SimClosedTrade] = []

    for name in ('CS-5TF', 'CS-DW', 'ML', 'Intra'):
        scanner = scanners[name]
        trades = scanner.closed_trades
        n = len(trades)
        pnl = sum(t.pnl for t in trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / n * 100 if n > 0 else 0
        dd = scanner.max_drawdown * 100

        total_pnl += pnl
        total_trades += n
        total_wins += wins
        all_trades.extend(trades)

        print(f"  {name:<10} {n:>7} {wins:>5} {wr:>5.0f}% ${pnl:>+10,.0f} "
              f"{dd:>6.2f}% ${scanner.equity:>11,.0f}")

    # Overall
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    total_cap = capital * 4 if equity_mode == 'isolated' else capital
    roi = total_pnl / total_cap * 100

    print(f"  {'-'*65}")
    print(f"  {'TOTAL':<10} {total_trades:>7} {total_wins:>5} {wr:>5.0f}% "
          f"${total_pnl:>+10,.0f}         ${total_cap + total_pnl:>11,.0f}")
    print(f"  ROI: {roi:>+.2f}% on ${total_cap:,.0f}")

    # Exit reason breakdown
    exit_reasons: Dict[str, List[float]] = {}
    for t in all_trades:
        exit_reasons.setdefault(t.exit_reason, []).append(t.pnl)

    if exit_reasons:
        print(f"\n  EXIT REASONS:")
        print(f"  {'Reason':<16} {'Count':>6} {'P&L':>12} {'WR':>6}")
        print(f"  {'-'*44}")
        for reason in sorted(exit_reasons.keys()):
            pnls = exit_reasons[reason]
            total = sum(pnls)
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / len(pnls) * 100
            print(f"  {reason:<16} {len(pnls):>6} ${total:>+10,.0f} {wr:>5.0f}%")

    print(f"{'='*80}")

    return total_pnl, total_trades, total_wins


def run_sweep(data: dict, sim_start: str, sim_end: str,
              capital: float = 100_000, verbose: bool = False):
    """Run 3 × 2 = 6 scenarios: hard_stop × equity_mode.

    Hard stop: {1%, 2%, None} × Equity: {isolated, shared}
    """
    hard_stops = [0.01, 0.02, None]
    equity_modes = ['isolated', 'shared']

    results = []

    for hs in hard_stops:
        for em in equity_modes:
            hs_label = f"{hs:.0%}" if hs else "None"
            label = f"HS={hs_label} / {em}"
            print(f"\n{'#'*60}")
            print(f"# SWEEP: {label}")
            print(f"{'#'*60}")

            scanners = run_all_scanners_sim(
                data=data,
                sim_start=sim_start,
                sim_end=sim_end,
                capital=capital,
                hard_stop_pct=hs,
                equity_mode=em,
                verbose=verbose,
            )

            pnl, trades, wins = print_multi_scanner_results(
                scanners, sim_start, sim_end, capital, em, hs)

            max_dd = max(s.max_drawdown for s in scanners.values()) * 100
            wr = wins / trades * 100 if trades > 0 else 0
            results.append((label, trades, wins, wr, pnl, max_dd))

    # Comparison table
    print(f"\n\n{'='*80}")
    print(f"SWEEP COMPARISON: {sim_start} to {sim_end}")
    print(f"{'='*80}")
    print(f"  {'Scenario':<25} {'Trades':>7} {'WR':>6} {'P&L':>12} {'MaxDD':>7}")
    print(f"  {'-'*60}")
    for label, trades, wins, wr, pnl, dd in results:
        print(f"  {label:<25} {trades:>7} {wr:>5.0f}% ${pnl:>+10,.0f} {dd:>6.2f}%")
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(scanner: SimScanner, sim_start: str, sim_end: str, capital: float):
    """Print formatted results."""
    trades = scanner.closed_trades

    print(f"\n{'='*80}")
    print(f"FORWARD SIM V2 (Live Scanner Replica): {sim_start} to {sim_end}")
    print(f"{'='*80}")
    print(f"Capital: ${capital:,.0f}")

    if not trades:
        print("\n  No trades generated.")
        return

    # Trade table
    print(f"\nTRADES:")
    print(f"  {'#':>3}  {'Time':<20} {'Dir':<6} {'Price':>8} {'Shares':>7} "
          f"{'Type':<8} {'TF':<6} {'Exit':<15} {'P&L':>10} {'Equity':>12}")
    print(f"  {'-'*100}")

    running_equity = capital
    for i, t in enumerate(trades, 1):
        running_equity += t.pnl
        time_str = t.entry_time.strftime('%m/%d %H:%M') if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:16]
        print(f"  {i:>3}  {time_str:<20} {t.direction.upper():<6} "
              f"${t.entry_price:>7.2f} {t.shares:>7} "
              f"{t.signal_type:<8} {t.primary_tf:<6} {t.exit_reason:<15} "
              f"${t.pnl:>+9,.0f} ${running_equity:>11,.0f}")

    # Daily summary
    daily: Dict[str, List[SimClosedTrade]] = {}
    for t in trades:
        day = t.entry_time.strftime('%Y-%m-%d') if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:10]
        daily.setdefault(day, []).append(t)

    print(f"\nDAILY SUMMARY:")
    print(f"  {'Date':<12} {'Trades':>7} {'P&L':>12} {'Equity':>12} {'Win Rate':>10}")
    print(f"  {'-'*55}")

    running_eq = capital
    for day in sorted(daily.keys()):
        day_trades = daily[day]
        day_pnl = sum(t.pnl for t in day_trades)
        day_wins = sum(1 for t in day_trades if t.pnl > 0)
        day_wr = day_wins / len(day_trades) * 100 if day_trades else 0
        running_eq += day_pnl
        print(f"  {day:<12} {len(day_trades):>7} ${day_pnl:>+10,.0f} "
              f"${running_eq:>11,.0f} {day_wr:>9.0f}%")

    # Exit reason breakdown
    exit_reasons: Dict[str, List[float]] = {}
    for t in trades:
        exit_reasons.setdefault(t.exit_reason, []).append(t.pnl)

    print(f"\nEXIT REASONS:")
    print(f"  {'Reason':<18} {'Count':>6} {'Total P&L':>12} {'Avg P&L':>10} {'Win Rate':>10}")
    print(f"  {'-'*60}")
    for reason in sorted(exit_reasons.keys()):
        pnls = exit_reasons[reason]
        total = sum(pnls)
        avg = total / len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100
        print(f"  {reason:<18} {len(pnls):>6} ${total:>+10,.0f} ${avg:>+8,.0f} {wr:>9.0f}%")

    # TF breakdown
    tf_trades: Dict[str, List[float]] = {}
    for t in trades:
        tf_trades.setdefault(t.primary_tf, []).append(t.pnl)

    print(f"\nTIMEFRAME BREAKDOWN:")
    print(f"  {'TF':<10} {'Count':>6} {'Total P&L':>12} {'Avg P&L':>10} {'Win Rate':>10}")
    print(f"  {'-'*52}")
    for tf in sorted(tf_trades.keys()):
        pnls = tf_trades[tf]
        total = sum(pnls)
        avg = total / len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100
        print(f"  {tf:<10} {len(pnls):>6} ${total:>+10,.0f} ${avg:>+8,.0f} {wr:>9.0f}%")

    # Totals
    total_pnl = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    wr = wins / len(trades) * 100
    roi = total_pnl / capital * 100
    avg_trade = total_pnl / len(trades)
    avg_hold = np.mean([t.hold_minutes for t in trades])
    max_dd = scanner.max_drawdown * 100

    win_pnls = [t.pnl for t in trades if t.pnl > 0]
    loss_pnls = [t.pnl for t in trades if t.pnl <= 0]
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    pf = abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf')

    print(f"\n{'='*80}")
    print(f"TOTALS:")
    print(f"  Trades: {len(trades)}  |  Wins: {wins}  |  Losses: {losses}  |  "
          f"Win Rate: {wr:.0f}%")
    print(f"  Total P&L: ${total_pnl:>+,.0f}  |  ROI: {roi:>+.2f}%  |  "
          f"Final Equity: ${scanner.equity:>,.0f}")
    print(f"  Avg Trade: ${avg_trade:>+,.0f}  |  Avg Win: ${avg_win:>+,.0f}  |  "
          f"Avg Loss: ${avg_loss:>+,.0f}")
    print(f"  Profit Factor: {pf:.2f}  |  Max DD: {max_dd:.2f}%  |  "
          f"Avg Hold: {avg_hold:.0f} min")
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# VIX-gated AM-only sweep
# ---------------------------------------------------------------------------

def run_vix_sweep(data: dict, sim_start: str, sim_end: str,
                  capital: float = 100_000, hard_stop_pct: float = 0.02,
                  equity_mode: str = 'isolated'):
    """Run VIX-gated AM-only threshold sweep.

    Tests 6 scenarios: baseline, always AM-only, and 4 VIX thresholds.
    All use the same hard_stop_pct and equity_mode.
    """
    # Print VIX context for the sim window
    vix_daily = data.get('vix_daily')
    if vix_daily is not None and len(vix_daily) > 0:
        vix_closes = vix_daily['close'].dropna()
        sim_s = pd.Timestamp(sim_start)
        sim_e = pd.Timestamp(sim_end)
        if vix_closes.index.tz is not None:
            if sim_s.tzinfo is None:
                sim_s = sim_s.tz_localize(vix_closes.index.tz)
            if sim_e.tzinfo is None:
                sim_e = sim_e.tz_localize(vix_closes.index.tz)
        window_vix = vix_closes[(vix_closes.index >= sim_s) & (vix_closes.index <= sim_e)]
        print(f"\n{'='*80}")
        print(f"VIX CONTEXT: {sim_start} to {sim_end}")
        print(f"{'='*80}")
        for dt, val in window_vix.items():
            d_str = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)[:10]
            print(f"  {d_str}  VIX close = {val:.2f}")
        if len(window_vix) > 0:
            print(f"  Range: {window_vix.min():.2f} - {window_vix.max():.2f}  "
                  f"Mean: {window_vix.mean():.2f}")
        print()

    # Define scenarios: (label, am_only, vix_am_threshold)
    scenarios = [
        ('Baseline (no AM)',    False, None),
        ('Always AM-only',     True,  None),
        ('VIX >= 16',          False, 16.0),
        ('VIX >= 18',          False, 18.0),
        ('VIX >= 20',          False, 20.0),
        ('VIX >= 22',          False, 22.0),
        ('VIX >= 25',          False, 25.0),
        ('VIX >= 30',          False, 30.0),
    ]

    results = []
    for label, am_flag, vix_thresh in scenarios:
        print(f"\n{'#'*60}")
        print(f"# VIX SWEEP: {label}")
        print(f"{'#'*60}")

        scanners = run_all_scanners_sim(
            data=data,
            sim_start=sim_start,
            sim_end=sim_end,
            capital=capital,
            hard_stop_pct=hard_stop_pct,
            equity_mode=equity_mode,
            am_only=am_flag,
            vix_am_threshold=vix_thresh,
            dw_eod_close=False,
            verbose=False,
        )

        pnl, trades, wins = print_multi_scanner_results(
            scanners, sim_start, sim_end, capital, equity_mode, hard_stop_pct)

        max_dd = max(s.max_drawdown for s in scanners.values()) * 100
        wr = wins / trades * 100 if trades > 0 else 0
        results.append((label, trades, wins, wr, pnl, max_dd))

    # Comparison table
    print(f"\n\n{'='*80}")
    print(f"VIX-GATED AM-ONLY SWEEP: {sim_start} to {sim_end}")
    print(f"Hard stop: {hard_stop_pct:.0%}  |  Equity: {equity_mode}  |  Capital: ${capital:,.0f}")
    print(f"{'='*80}")
    print(f"  {'Scenario':<25} {'Trades':>7} {'WR':>6} {'P&L':>12} {'MaxDD':>7}")
    print(f"  {'-'*60}")
    for label, trades, wins, wr, pnl, dd in results:
        print(f"  {label:<25} {trades:>7} {wr:>5.0f}% ${pnl:>+10,.0f} {dd:>6.2f}%")
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Trade analysis — deep dive on eod_close / hard_stop patterns
# ---------------------------------------------------------------------------

def print_trade_analysis(scanners: Dict[str, SimScanner]):
    """Analyze all trades for patterns in losing exit types."""
    all_trades: List[SimClosedTrade] = []
    for name in ('CS-5TF', 'CS-DW', 'ML', 'Intra'):
        for t in scanners[name].closed_trades:
            t._scanner = name  # tag with scanner name
            all_trades.append(t)

    if not all_trades:
        print("  No trades to analyze.")
        return

    # ── 1. Direction × exit type matrix ──
    print(f"\n{'='*90}")
    print("TRADE ANALYSIS: Direction × Exit Type")
    print(f"{'='*90}")
    print(f"  {'Exit Reason':<18} {'Long #':>7} {'Long P&L':>12} {'Short #':>8} {'Short P&L':>12} "
          f"{'Long WR':>8} {'Short WR':>9}")
    print(f"  {'-'*78}")

    exit_types = sorted(set(t.exit_reason for t in all_trades))
    for reason in exit_types:
        longs = [t for t in all_trades if t.exit_reason == reason and t.direction == 'long']
        shorts = [t for t in all_trades if t.exit_reason == reason and t.direction == 'short']
        l_pnl = sum(t.pnl for t in longs)
        s_pnl = sum(t.pnl for t in shorts)
        l_wr = sum(1 for t in longs if t.pnl > 0) / len(longs) * 100 if longs else 0
        s_wr = sum(1 for t in shorts if t.pnl > 0) / len(shorts) * 100 if shorts else 0
        print(f"  {reason:<18} {len(longs):>7} ${l_pnl:>+10,.0f} {len(shorts):>8} ${s_pnl:>+10,.0f} "
              f"{l_wr:>7.0f}% {s_wr:>8.0f}%")

    # Overall direction split
    all_longs = [t for t in all_trades if t.direction == 'long']
    all_shorts = [t for t in all_trades if t.direction == 'short']
    l_tot = sum(t.pnl for t in all_longs)
    s_tot = sum(t.pnl for t in all_shorts)
    l_wr = sum(1 for t in all_longs if t.pnl > 0) / len(all_longs) * 100 if all_longs else 0
    s_wr = sum(1 for t in all_shorts if t.pnl > 0) / len(all_shorts) * 100 if all_shorts else 0
    print(f"  {'-'*78}")
    print(f"  {'TOTAL':<18} {len(all_longs):>7} ${l_tot:>+10,.0f} {len(all_shorts):>8} ${s_tot:>+10,.0f} "
          f"{l_wr:>7.0f}% {s_wr:>8.0f}%")

    # ── 2. Per-scanner × direction breakdown ──
    print(f"\n{'='*90}")
    print("PER-SCANNER DIRECTION BREAKDOWN")
    print(f"{'='*90}")
    print(f"  {'Scanner':<10} {'Long #':>7} {'Long P&L':>12} {'Long WR':>8} "
          f"{'Short #':>8} {'Short P&L':>12} {'Short WR':>9}")
    print(f"  {'-'*70}")
    for name in ('CS-5TF', 'CS-DW', 'ML', 'Intra'):
        trades = scanners[name].closed_trades
        longs = [t for t in trades if t.direction == 'long']
        shorts = [t for t in trades if t.direction == 'short']
        l_pnl = sum(t.pnl for t in longs)
        s_pnl = sum(t.pnl for t in shorts)
        l_wr = sum(1 for t in longs if t.pnl > 0) / len(longs) * 100 if longs else 0
        s_wr = sum(1 for t in shorts if t.pnl > 0) / len(shorts) * 100 if shorts else 0
        print(f"  {name:<10} {len(longs):>7} ${l_pnl:>+10,.0f} {l_wr:>7.0f}% "
              f"{len(shorts):>8} ${s_pnl:>+10,.0f} {s_wr:>8.0f}%")

    # ── 3. EOD close entry hour distribution ──
    eod_trades = [t for t in all_trades if t.exit_reason == 'eod_close']
    if eod_trades:
        print(f"\n{'='*90}")
        print(f"EOD_CLOSE TRADES: Entry Hour Distribution (ET)")
        print(f"{'='*90}")
        hour_buckets: Dict[int, List] = {}
        for t in eod_trades:
            try:
                import pytz
                et = pytz.timezone('US/Eastern')
                if t.entry_time.tzinfo is not None:
                    entry_et = t.entry_time.astimezone(et)
                else:
                    entry_et = et.localize(t.entry_time)
                h = entry_et.hour
            except Exception:
                h = t.entry_time.hour if hasattr(t.entry_time, 'hour') else -1
            hour_buckets.setdefault(h, []).append(t)

        print(f"  {'Hour ET':<10} {'Count':>7} {'Longs':>7} {'Shorts':>7} {'P&L':>12} {'WR':>6}")
        print(f"  {'-'*52}")
        for h in sorted(hour_buckets.keys()):
            trades_h = hour_buckets[h]
            n_l = sum(1 for t in trades_h if t.direction == 'long')
            n_s = sum(1 for t in trades_h if t.direction == 'short')
            pnl = sum(t.pnl for t in trades_h)
            wr = sum(1 for t in trades_h if t.pnl > 0) / len(trades_h) * 100
            print(f"  {h:>2}:00     {len(trades_h):>7} {n_l:>7} {n_s:>7} ${pnl:>+10,.0f} {wr:>5.0f}%")

        # Per-scanner breakdown of eod_close
        print(f"\n  EOD_CLOSE by Scanner × Direction:")
        print(f"  {'Scanner':<10} {'Long #':>7} {'Long P&L':>12} {'Short #':>8} {'Short P&L':>12}")
        print(f"  {'-'*52}")
        for name in ('CS-5TF', 'CS-DW', 'ML', 'Intra'):
            sc_eod = [t for t in eod_trades if t._scanner == name]
            longs = [t for t in sc_eod if t.direction == 'long']
            shorts = [t for t in sc_eod if t.direction == 'short']
            l_pnl = sum(t.pnl for t in longs)
            s_pnl = sum(t.pnl for t in shorts)
            if longs or shorts:
                print(f"  {name:<10} {len(longs):>7} ${l_pnl:>+10,.0f} {len(shorts):>8} ${s_pnl:>+10,.0f}")

    # ── 4. Hard stop trade list ──
    hs_trades = [t for t in all_trades if t.exit_reason == 'hard_stop']
    if hs_trades:
        print(f"\n{'='*90}")
        print(f"ALL HARD_STOP TRADES ({len(hs_trades)} total)")
        print(f"{'='*90}")
        print(f"  {'#':>3} {'Scanner':<8} {'Dir':<6} {'Entry Time':<18} {'Entry$':>8} "
              f"{'Exit$':>8} {'P&L':>10} {'TF':<8} {'Type':<8} {'Hold':>6}")
        print(f"  {'-'*100}")
        for i, t in enumerate(sorted(hs_trades, key=lambda x: x.entry_time), 1):
            time_str = t.entry_time.strftime('%m/%d %H:%M') if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:16]
            hold = f"{t.hold_minutes:.0f}m"
            print(f"  {i:>3} {t._scanner:<8} {t.direction.upper():<6} {time_str:<18} "
                  f"${t.entry_price:>7.2f} ${t.exit_price:>7.2f} ${t.pnl:>+9,.0f} "
                  f"{t.primary_tf:<8} {t.signal_type:<8} {hold:>6}")

        # Summary
        hs_longs = [t for t in hs_trades if t.direction == 'long']
        hs_shorts = [t for t in hs_trades if t.direction == 'short']
        print(f"\n  Hard stop summary: {len(hs_longs)} longs (${sum(t.pnl for t in hs_longs):+,.0f}), "
              f"{len(hs_shorts)} shorts (${sum(t.pnl for t in hs_shorts):+,.0f})")
        # Per-scanner
        for name in ('CS-5TF', 'ML'):
            sc_hs = [t for t in hs_trades if t._scanner == name]
            if sc_hs:
                l = sum(1 for t in sc_hs if t.direction == 'long')
                s = sum(1 for t in sc_hs if t.direction == 'short')
                print(f"  {name}: {l} longs, {s} shorts")

    # ── 5. TF breakdown for problematic exits ──
    print(f"\n{'='*90}")
    print("EOD_CLOSE + HARD_STOP by Primary Timeframe")
    print(f"{'='*90}")
    problem_trades = [t for t in all_trades if t.exit_reason in ('eod_close', 'hard_stop')]
    tf_buckets: Dict[str, List] = {}
    for t in problem_trades:
        tf_buckets.setdefault(t.primary_tf, []).append(t)
    print(f"  {'TF':<10} {'Count':>7} {'Longs':>7} {'Shorts':>7} {'P&L':>12} {'WR':>6}")
    print(f"  {'-'*52}")
    for tf in sorted(tf_buckets.keys()):
        trades_tf = tf_buckets[tf]
        n_l = sum(1 for t in trades_tf if t.direction == 'long')
        n_s = sum(1 for t in trades_tf if t.direction == 'short')
        pnl = sum(t.pnl for t in trades_tf)
        wr = sum(1 for t in trades_tf if t.pnl > 0) / len(trades_tf) * 100
        print(f"  {tf:<10} {len(trades_tf):>7} {n_l:>7} {n_s:>7} ${pnl:>+10,.0f} {wr:>5.0f}%")
    print(f"{'='*90}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Forward Sim V2: Live Scanner Replica')
    parser.add_argument('--start', default='2026-02-23',
                        help='Sim start date (default: 2026-02-23)')
    parser.add_argument('--end', default='2026-03-03',
                        help='Sim end date (default: 2026-03-03)')
    parser.add_argument('--capital', type=float, default=100_000,
                        help='Initial capital per scanner (default: 100000)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-bar output')
    parser.add_argument('--mode', choices=['all', 'single', 'portfolio', 'equal',
                                           'multi', 'sweep'],
                        default='multi',
                        help='Run mode (default: multi)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run sweep: 3 hard stops × 2 equity modes')
    parser.add_argument('--equity-mode', choices=['isolated', 'shared'],
                        default='isolated',
                        help='Equity mode for multi/sweep (default: isolated)')
    parser.add_argument('--hard-stop', type=float, default=None,
                        help='Hard stop cap as decimal (e.g. 0.02 = 2%%)')
    parser.add_argument('--vix-sweep', action='store_true',
                        help='Run VIX-gated AM-only threshold sweep')
    parser.add_argument('--local', action='store_true',
                        help='Use local 1-min files (TSLAMin/SPYMin) for longer OOS')
    parser.add_argument('--analyze', action='store_true',
                        help='Run baseline with DW multi-day holds + trade analysis')
    parser.add_argument('--am-only', action='store_true',
                        help='Restrict CS/ML to first hour (9:30-10:30 ET)')
    parser.add_argument('--hs-dump', action='store_true',
                        help='Dump all hard_stop trades with VIX context to CSV + analysis')
    parser.add_argument('--am-cutoff', type=int, default=60,
                        help='AM cutoff in minutes after 9:30 ET (default: 60 = 10:30)')
    parser.add_argument('--late-short-hs', type=float, default=None,
                        help='Late short hard stop pct (e.g. 0.001 = 0.1%%)')
    parser.add_argument('--late-hs', type=float, default=None,
                        help='Late hard stop pct for ALL trades after AM cutoff (e.g. 0.0025 = 0.25%%)')
    args = parser.parse_args()

    # --sweep flag overrides --mode
    if args.sweep:
        args.mode = 'sweep'

    print(f"\n{'#'*80}")
    print(f"# FORWARD SIM V2 (Live Scanner Replica)")
    print(f"# Sim window: {args.start} to {args.end}")
    print(f"# Capital: ${args.capital:,.0f}")
    print(f"# Mode: {args.mode}")
    print(f"{'#'*80}")

    # Download data (once, reused across all runs)
    t0 = time.time()
    if args.local:
        data = download_data_local(start=args.start, end=args.end)
    else:
        data = download_data()
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # ── VIX-gated AM-only sweep ──
    if args.vix_sweep:
        run_vix_sweep(
            data=data,
            sim_start=args.start,
            sim_end=args.end,
            capital=args.capital,
            hard_stop_pct=args.hard_stop if args.hard_stop else 0.02,
            equity_mode=args.equity_mode,
        )
        print(f"\n{'#'*80}")
        print(f"# ALL RUNS COMPLETE")
        print(f"{'#'*80}")
        return

    # ── Hard stop dump: export all HS trades with VIX context ──
    if args.hs_dump:
        hs = args.hard_stop if args.hard_stop else 0.02
        print(f"\n\n{'#'*80}")
        print(f"# HARD STOP DUMP (HS={hs:.0%}, {args.equity_mode}, DW EOD disabled)")
        print(f"{'#'*80}")
        scanners = run_all_scanners_sim(
            data=data, sim_start=args.start, sim_end=args.end,
            capital=args.capital, hard_stop_pct=hs,
            equity_mode=args.equity_mode, dw_eod_close=False,
            verbose=not args.quiet,
        )
        print_multi_scanner_results(scanners, args.start, args.end,
                                    args.capital, args.equity_mode, hs)

        # Build VIX lookup
        import pytz
        et_tz = pytz.timezone('US/Eastern')
        vix_daily = data.get('vix_daily')
        vix_by_date = {}
        if vix_daily is not None and len(vix_daily) > 0:
            vc = vix_daily['close'].dropna()
            vdates = sorted(vc.index)
            for i in range(1, len(vdates)):
                d = vdates[i].date() if hasattr(vdates[i], 'date') else vdates[i]
                vix_by_date[d] = float(vc.iloc[i - 1])

        # Collect all hard_stop trades
        hs_trades = []
        for name in ('CS-5TF', 'CS-DW', 'ML', 'Intra'):
            for t in scanners[name].closed_trades:
                if t.exit_reason == 'hard_stop':
                    t._scanner = name
                    hs_trades.append(t)
        hs_trades.sort(key=lambda x: x.entry_time)

        print(f"\n{'='*120}")
        print(f"ALL {len(hs_trades)} HARD_STOP TRADES — Full Detail")
        print(f"{'='*120}")
        print(f"  {'#':>3} {'Scanner':<8} {'Dir':<6} {'Entry ET':<18} {'Exit ET':<18} "
              f"{'Entry$':>8} {'Exit$':>8} {'P&L':>10} {'TF':<6} {'SigType':<8} "
              f"{'Hold':>6} {'VIX':>5} {'Conf':>5} {'EntHr':>5}")
        print(f"  {'-'*130}")

        # Collect data for analysis
        hs_data = []
        for i, t in enumerate(hs_trades, 1):
            entry_et = t.entry_time.astimezone(et_tz) if getattr(t.entry_time, 'tzinfo', None) else et_tz.localize(t.entry_time)
            exit_et = t.exit_time.astimezone(et_tz) if getattr(t.exit_time, 'tzinfo', None) else et_tz.localize(t.exit_time)
            entry_date = entry_et.date()
            prev_vix = vix_by_date.get(entry_date, 0)
            entry_hr = entry_et.hour + entry_et.minute / 60.0

            hs_data.append({
                'scanner': t._scanner, 'direction': t.direction,
                'entry_time': entry_et, 'exit_time': exit_et,
                'entry_price': t.entry_price, 'exit_price': t.exit_price,
                'pnl': t.pnl, 'pnl_pct': t.pnl_pct,
                'primary_tf': t.primary_tf, 'signal_type': t.signal_type,
                'hold_minutes': t.hold_minutes, 'prev_vix': prev_vix,
                'confidence': t.confidence, 'entry_hour': entry_hr,
                'entry_date': entry_date,
            })

            time_str = entry_et.strftime('%m/%d %H:%M')
            exit_str = exit_et.strftime('%m/%d %H:%M')
            print(f"  {i:>3} {t._scanner:<8} {t.direction.upper():<6} {time_str:<18} {exit_str:<18} "
                  f"${t.entry_price:>7.2f} ${t.exit_price:>7.2f} ${t.pnl:>+9,.0f} "
                  f"{t.primary_tf:<6} {t.signal_type:<8} {t.hold_minutes:>5.0f}m "
                  f"{prev_vix:>5.1f} {t.confidence:>5.2f} {entry_hr:>5.1f}")

        # ── Analysis: group by scanner ──
        print(f"\n{'='*80}")
        print("HARD STOP BREAKDOWN BY SCANNER")
        print(f"{'='*80}")
        from collections import Counter
        scanner_counts = Counter(d['scanner'] for d in hs_data)
        for sc, cnt in scanner_counts.most_common():
            sc_trades = [d for d in hs_data if d['scanner'] == sc]
            avg_pnl = sum(d['pnl'] for d in sc_trades) / len(sc_trades)
            avg_vix = sum(d['prev_vix'] for d in sc_trades) / len(sc_trades)
            longs = sum(1 for d in sc_trades if d['direction'] == 'long')
            shorts = cnt - longs
            print(f"  {sc:<8}: {cnt:>3} trades (L:{longs}/S:{shorts}), "
                  f"avg P&L=${avg_pnl:+,.0f}, avg VIX={avg_vix:.1f}")

        # ── Analysis: group by VIX level ──
        print(f"\n{'='*80}")
        print("HARD STOP BREAKDOWN BY VIX LEVEL")
        print(f"{'='*80}")
        vix_buckets = [(0, 16, 'VIX<16'), (16, 20, 'VIX 16-20'),
                       (20, 25, 'VIX 20-25'), (25, 35, 'VIX 25-35'), (35, 100, 'VIX 35+')]
        for lo, hi, label in vix_buckets:
            bucket = [d for d in hs_data if lo <= d['prev_vix'] < hi]
            if bucket:
                tot_pnl = sum(d['pnl'] for d in bucket)
                avg_pnl = tot_pnl / len(bucket)
                print(f"  {label:<12}: {len(bucket):>3} trades, total P&L=${tot_pnl:+,.0f}, avg=${avg_pnl:+,.0f}")

        # ── Analysis: group by entry hour ──
        print(f"\n{'='*80}")
        print("HARD STOP BREAKDOWN BY ENTRY HOUR (ET)")
        print(f"{'='*80}")
        hour_buckets_hs = {}
        for d in hs_data:
            h = int(d['entry_hour'])
            hour_buckets_hs.setdefault(h, []).append(d)
        for h in sorted(hour_buckets_hs.keys()):
            bucket = hour_buckets_hs[h]
            tot_pnl = sum(d['pnl'] for d in bucket)
            longs = sum(1 for d in bucket if d['direction'] == 'long')
            shorts = len(bucket) - longs
            print(f"  {h:>2}:00-{h:>2}:59 : {len(bucket):>3} trades (L:{longs}/S:{shorts}), "
                  f"P&L=${tot_pnl:+,.0f}")

        # ── Analysis: group by signal_type ──
        print(f"\n{'='*80}")
        print("HARD STOP BREAKDOWN BY SIGNAL TYPE")
        print(f"{'='*80}")
        sig_counts = Counter(d['signal_type'] for d in hs_data)
        for sig, cnt in sig_counts.most_common():
            sig_trades = [d for d in hs_data if d['signal_type'] == sig]
            tot_pnl = sum(d['pnl'] for d in sig_trades)
            avg_conf = sum(d['confidence'] for d in sig_trades) / len(sig_trades)
            print(f"  {sig:<12}: {cnt:>3} trades, P&L=${tot_pnl:+,.0f}, avg conf={avg_conf:.2f}")

        # ── Analysis: group by primary_tf ──
        print(f"\n{'='*80}")
        print("HARD STOP BREAKDOWN BY PRIMARY TIMEFRAME")
        print(f"{'='*80}")
        tf_counts = Counter(d['primary_tf'] for d in hs_data)
        for tf, cnt in tf_counts.most_common():
            tf_trades = [d for d in hs_data if d['primary_tf'] == tf]
            tot_pnl = sum(d['pnl'] for d in tf_trades)
            print(f"  {tf:<8}: {cnt:>3} trades, P&L=${tot_pnl:+,.0f}")

        # ── Analysis: group by month ──
        print(f"\n{'='*80}")
        print("HARD STOP BREAKDOWN BY MONTH")
        print(f"{'='*80}")
        month_buckets = {}
        for d in hs_data:
            m = d['entry_date'].strftime('%Y-%m')
            month_buckets.setdefault(m, []).append(d)
        for m in sorted(month_buckets.keys()):
            bucket = month_buckets[m]
            tot_pnl = sum(d['pnl'] for d in bucket)
            avg_vix = sum(d['prev_vix'] for d in bucket) / len(bucket)
            print(f"  {m}: {len(bucket):>3} trades, P&L=${tot_pnl:+,.0f}, avg VIX={avg_vix:.1f}")

        # ── Analysis: group by day of week ──
        print(f"\n{'='*80}")
        print("HARD STOP BREAKDOWN BY DAY OF WEEK")
        print(f"{'='*80}")
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        dow_buckets = {}
        for d in hs_data:
            dow = d['entry_date'].weekday()
            dow_buckets.setdefault(dow, []).append(d)
        for dow in sorted(dow_buckets.keys()):
            bucket = dow_buckets[dow]
            tot_pnl = sum(d['pnl'] for d in bucket)
            print(f"  {dow_names[dow]}: {len(bucket):>3} trades, P&L=${tot_pnl:+,.0f}")

        # ── Export to CSV ──
        csv_path = 'v15/validation/hard_stop_trades.csv'
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'scanner', 'direction', 'entry_time', 'exit_time',
                'entry_price', 'exit_price', 'pnl', 'pnl_pct',
                'primary_tf', 'signal_type', 'hold_minutes',
                'prev_vix', 'confidence', 'entry_hour', 'entry_date',
            ])
            writer.writeheader()
            for d in hs_data:
                row = dict(d)
                row['entry_time'] = str(row['entry_time'])
                row['exit_time'] = str(row['exit_time'])
                row['entry_date'] = str(row['entry_date'])
                writer.writerow(row)
        print(f"\n  Exported {len(hs_data)} hard_stop trades to {csv_path}")

        print(f"\n{'#'*80}")
        print(f"# HARD STOP DUMP COMPLETE")
        print(f"{'#'*80}")
        return

    # ── Analyze mode: baseline + DW multi-day + trade deep dive ──
    if args.analyze:
        hs = args.hard_stop if args.hard_stop else 0.02
        print(f"\n\n{'#'*80}")
        print(f"# TRADE ANALYSIS (DW EOD disabled, HS={hs:.0%}, {args.equity_mode})")
        if args.am_only:
            print(f"# AM-only: CS/ML restricted to 9:30-10:30 ET")
        print(f"{'#'*80}")
        scanners = run_all_scanners_sim(
            data=data,
            sim_start=args.start,
            sim_end=args.end,
            capital=args.capital,
            hard_stop_pct=hs,
            equity_mode=args.equity_mode,
            am_only=args.am_only,
            am_cutoff_minutes=args.am_cutoff,
            late_short_hard_stop_pct=args.late_short_hs,
            dw_eod_close=False,
            verbose=not args.quiet,
        )
        print_multi_scanner_results(scanners, args.start, args.end,
                                    args.capital, args.equity_mode, hs)
        print_trade_analysis(scanners)
        print(f"\n{'#'*80}")
        print(f"# ALL RUNS COMPLETE")
        print(f"{'#'*80}")
        return

    # ── Multi-scanner: all 4 signal types ──
    if args.mode == 'multi':
        am_tag = ""
        if args.am_only:
            am_tag = f", AM-only {args.am_cutoff}min"
        if args.late_short_hs is not None:
            am_tag += f", late short HS={args.late_short_hs:.1%}"
        if args.late_hs is not None:
            am_tag += f", late all HS={args.late_hs:.2%}"
        print(f"\n\n{'#'*80}")
        print(f"# ALL 4 SCANNERS ({args.equity_mode} mode{am_tag})")
        print(f"{'#'*80}")
        scanners = run_all_scanners_sim(
            data=data,
            sim_start=args.start,
            sim_end=args.end,
            capital=args.capital,
            hard_stop_pct=args.hard_stop,
            equity_mode=args.equity_mode,
            am_only=args.am_only,
            am_cutoff_minutes=args.am_cutoff,
            late_short_hard_stop_pct=args.late_short_hs,
            late_hard_stop_pct=args.late_hs,
            verbose=not args.quiet,
        )
        print_multi_scanner_results(scanners, args.start, args.end,
                                    args.capital, args.equity_mode,
                                    args.hard_stop)

    # ── Sweep: 3 hard stops × 2 equity modes ──
    elif args.mode == 'sweep':
        run_sweep(
            data=data,
            sim_start=args.start,
            sim_end=args.end,
            capital=args.capital,
            verbose=not args.quiet,
        )

    else:
        # Legacy modes
        run_single = args.mode in ('all', 'single')
        run_portfolio = args.mode in ('all', 'portfolio')
        run_equal = args.mode in ('all', 'equal')

        if run_single:
            print(f"\n\n{'#'*80}")
            print(f"# RUN 1: ALL-TF BASELINE (live scanner replica)")
            print(f"{'#'*80}")
            scanner = run_simulation(
                data=data,
                sim_start=args.start,
                sim_end=args.end,
                capital=args.capital,
                verbose=not args.quiet,
            )
            print_results(scanner, args.start, args.end, args.capital)

        if run_portfolio:
            print(f"\n\n{'#'*80}")
            print(f"# RUN 2: PORTFOLIO (20% 1h + 80% 1d)")
            print(f"{'#'*80}")
            alloc_2080 = {
                '1h':  ({'1h'}, 0.20),
                '1d':  ({'daily'}, 0.80),
            }
            scanners_2080 = run_portfolio_simulation(
                data=data,
                sim_start=args.start,
                sim_end=args.end,
                allocations=alloc_2080,
                total_capital=args.capital,
                verbose=not args.quiet,
            )
            print_portfolio_results(scanners_2080, alloc_2080,
                                    args.start, args.end, args.capital)

        if run_equal:
            print(f"\n\n{'#'*80}")
            print(f"# RUN 3: EQUAL SPLIT (20% each x 5 systems)")
            print(f"{'#'*80}")
            alloc_equal = {
                '5min':   ({'5min'}, 0.20),
                '1h':     ({'1h'}, 0.20),
                '4h':     ({'4h'}, 0.20),
                '1d':     ({'daily'}, 0.20),
                'weekly': ({'weekly'}, 0.20),
            }
            scanners_equal = run_portfolio_simulation(
                data=data,
                sim_start=args.start,
                sim_end=args.end,
                allocations=alloc_equal,
                total_capital=args.capital,
                verbose=not args.quiet,
            )
            print_portfolio_results(scanners_equal, alloc_equal,
                                    args.start, args.end, args.capital)

    print(f"\n{'#'*80}")
    print(f"# ALL RUNS COMPLETE")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
