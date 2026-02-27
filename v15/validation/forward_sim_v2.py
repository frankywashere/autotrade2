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
                 timeout_multiplier: float = 1.0):
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

        # Max positions
        if len(self.positions) >= self.MAX_POSITIONS:
            return False

        # Duplicate direction check
        direction = 'long' if sig.action == 'BUY' else 'short'
        for pos in self.positions.values():
            if pos.direction == direction:
                return False

        # Position sizing
        risk_dollars = self.equity * self.RISK_PER_TRADE
        stop_pct = sig.suggested_stop_pct
        if stop_pct <= 0:
            stop_pct = 0.02
        # Apply stop multiplier (wider stops)
        stop_pct = stop_pct * self.stop_multiplier

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
            tp_price = current_price * (1 + sig.suggested_tp_pct)
        else:
            stop_price = current_price * (1 + stop_pct)
            tp_price = current_price * (1 - sig.suggested_tp_pct)

        pos_id = str(uuid.uuid4())[:8]
        # Check if override was applied
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
            signal_type=getattr(sig, 'signal_type', 'bounce'),
            primary_tf=sig.primary_tf,
            confidence=sig.confidence,
            best_price=current_price,
            reason=sig.reason,
            override_applied=override_applied,
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

    def check_exits(self, current_price: float, bar_high: float, bar_low: float,
                    bar_time: pd.Timestamp) -> List[SimClosedTrade]:
        """Check all open positions for exit conditions. Returns list of closed trades."""
        closed: List[SimClosedTrade] = []
        to_close: List[str] = []
        is_eod = self._is_eod(bar_time)

        for pos_id, pos in self.positions.items():
            exit_reason = None
            exit_price = current_price

            # Compute hold time
            hold_minutes = (bar_time - pos.entry_time).total_seconds() / 60

            # Compute unrealized P&L
            if pos.direction == 'long':
                unrealized_pnl = (current_price - pos.entry_price) * pos.shares
            else:
                unrealized_pnl = (pos.entry_price - current_price) * pos.shares

            # --- Breakeven stop adjustment (not an exit) ---
            if (not pos.breakeven_applied
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

            # --- EOD force close (3:45 PM ET) ---
            if exit_reason is None and self.eod_close_enabled and is_eod:
                exit_reason = 'eod_close'

            # --- Near-TP early close (within first 30 min) ---
            if exit_reason is None and hold_minutes < self.NEAR_TP_WINDOW_MIN:
                if unrealized_pnl >= self.NEAR_TP_MIN_GAIN:
                    if (pos.direction == 'long' and
                            bar_high >= pos.tp_price * (1 - self.NEAR_TP_PCT)):
                        exit_reason = 'near_tp'
                    elif (pos.direction == 'short' and
                          bar_low <= pos.tp_price * (1 + self.NEAR_TP_PCT)):
                        exit_reason = 'near_tp'

            # --- Equity ceiling (unrealized >= 5% of equity) ---
            if exit_reason is None:
                if unrealized_pnl >= self.equity * self.EQUITY_CEILING_PCT:
                    exit_reason = 'equity_ceiling'

            # --- Outlier winner (unrealized >= 2x TP gain) ---
            if exit_reason is None:
                if pos.direction == 'long':
                    expected_tp_gain = (pos.tp_price - pos.entry_price) * pos.shares
                else:
                    expected_tp_gain = (pos.entry_price - pos.tp_price) * pos.shares
                if expected_tp_gain > 0 and unrealized_pnl >= self.OUTLIER_MULT * expected_tp_gain:
                    exit_reason = 'outlier_winner'

            # --- Hard stop DISABLED (matching live scanner) ---

            # --- Take profit ---
            if exit_reason is None:
                if pos.direction == 'long' and bar_high >= pos.tp_price:
                    exit_reason = 'take_profit'
                    exit_price = pos.tp_price
                elif pos.direction == 'short' and bar_low <= pos.tp_price:
                    exit_reason = 'take_profit'
                    exit_price = pos.tp_price

            # --- Trailing stop (profit-tier based) ---
            if exit_reason is None and self.trail_enabled:
                trail_price = self._calc_trail_price(pos)
                if trail_price is not None:
                    if pos.direction == 'long' and bar_low <= trail_price:
                        exit_reason = 'trailing_stop'
                        exit_price = trail_price
                    elif pos.direction == 'short' and bar_high >= trail_price:
                        exit_reason = 'trailing_stop'
                        exit_price = trail_price

            # --- Timeout (scaled by timeout_multiplier) ---
            if exit_reason is None:
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
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Forward Sim V2: Live Scanner Replica')
    parser.add_argument('--start', default='2026-02-23',
                        help='Sim start date (default: 2026-02-23)')
    parser.add_argument('--end', default='2026-02-26',
                        help='Sim end date (default: 2026-02-26)')
    parser.add_argument('--capital', type=float, default=100_000,
                        help='Initial capital (default: 100000)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-bar output')
    parser.add_argument('--mode', choices=['all', 'single', 'portfolio', 'equal'],
                        default='all',
                        help='Run mode: all (default, runs all 3), single, portfolio (20/80), equal (20/20/20/20/20)')
    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print(f"# FORWARD SIM V2 (Live Scanner Replica)")
    print(f"# Sim window: {args.start} to {args.end}")
    print(f"# Capital: ${args.capital:,.0f}")
    print(f"# Mode: {args.mode}")
    print(f"{'#'*80}")

    # Download data (once, reused across all runs)
    t0 = time.time()
    data = download_data()
    print(f"  Downloaded in {time.time() - t0:.1f}s")

    run_single = args.mode in ('all', 'single')
    run_portfolio = args.mode in ('all', 'portfolio')
    run_equal = args.mode in ('all', 'equal')

    # ── Run 1: Single scanner, all TFs (baseline / validation) ──
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

    # ── Run 2: Portfolio — 20% 1h + 80% 1d ──
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

    # ── Run 3: Equal — 20% each of 5min, 1h, 4h, 1d, weekly ──
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
