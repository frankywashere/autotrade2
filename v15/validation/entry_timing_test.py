#!/usr/bin/env python3
"""
Entry Timing Filter Test

Tests mean-reversion timing filters on CS-ALL signals to see if we can
improve entry timing vs. the baseline next-day-open entry.

Strategies:
  A: Immediate (baseline) — enter next-day open
  B: Pullback entry — wait up to 3 days for price to close against trade direction
  C: Gap filter — skip trades where next-day gaps > 1% in our direction
  D: RSI confirmation — BUY only if 5d RSI < 40, SELL only if 5d RSI > 60
  E: Intraday range filter — skip if signal day range > 5% of close
  F: Close-position filter — BUY if close in bottom 30% of range, SELL if top 30%
  G: Combined best — best individual filters combined

Usage:
    python -m v15.validation.entry_timing_test
"""

import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.combo_backtest import DaySignals  # needed for pickle unpickling

# ---------------------------------------------------------------------------
# Constants (match combo_backtest.py)
# ---------------------------------------------------------------------------

MIN_SIGNAL_CONFIDENCE = 0.45
CAPITAL = 100_000.0
DEFAULT_STOP_PCT = 0.02
DEFAULT_TP_PCT = 0.04
TRAILING_STOP_PCT = 0.015
MAX_HOLD_DAYS = 10
COOLDOWN_DAYS = 2
SLIPPAGE_PCT = 0.0001
COMMISSION_PER_SHARE = 0.005

CACHE_DIR = Path(__file__).parent / "combo_cache"
CACHE_FILE = CACHE_DIR / "combo_signals.pkl"


# ---------------------------------------------------------------------------
# Data structures (reuse DaySignals from cache, define Trade locally)
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A single completed trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    confidence: float
    shares: int
    pnl: float
    hold_days: int
    exit_reason: str
    source: str


# ---------------------------------------------------------------------------
# RSI computation (Wilder's smoothing, period=5)
# ---------------------------------------------------------------------------

def compute_rsi(closes: np.ndarray, period: int = 5) -> np.ndarray:
    """Compute Wilder's RSI from close prices. Returns array same length as closes."""
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple average
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder's smoothing
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


# ---------------------------------------------------------------------------
# Cost model (same as combo_backtest.py)
# ---------------------------------------------------------------------------

def apply_costs(entry_price: float, exit_price: float, shares: int,
                direction: str) -> float:
    """Compute raw PnL after slippage + commission."""
    slip_entry = entry_price * SLIPPAGE_PCT
    slip_exit = exit_price * SLIPPAGE_PCT
    comm = COMMISSION_PER_SHARE * shares * 2

    if direction == "LONG":
        pnl = (exit_price - slip_exit - entry_price - slip_entry) * shares - comm
    else:
        pnl = (entry_price - slip_entry - exit_price - slip_exit) * shares - comm
    return pnl


# ---------------------------------------------------------------------------
# Trade simulation engine
# ---------------------------------------------------------------------------

def run_trade_from_entry(signals, entry_idx: int, direction: str,
                         confidence: float) -> Optional[Trade]:
    """
    Simulate a single trade starting at entry_idx (the day we enter at open).
    Returns a Trade or None if entry is invalid.
    """
    if entry_idx >= len(signals):
        return None

    entry_day = signals[entry_idx]
    entry_price = entry_day.day_open
    if entry_price <= 0:
        return None

    # Position sizing
    position_value = CAPITAL * min(confidence, 1.0)
    shares = max(1, int(position_value / entry_price))

    if direction == "LONG":
        stop_price = entry_price * (1.0 - DEFAULT_STOP_PCT)
        tp_price = entry_price * (1.0 + DEFAULT_TP_PCT)
    else:
        stop_price = entry_price * (1.0 + DEFAULT_STOP_PCT)
        tp_price = entry_price * (1.0 - DEFAULT_TP_PCT)

    best_price = entry_price
    hold_days = 0

    for j in range(entry_idx, len(signals)):
        if j == entry_idx:
            # On entry day, we entered at open; still check intraday exit
            # but only count as hold_day=0 -> start counting from next day
            pass
        hold_days = j - entry_idx

        day = signals[j]
        price_h = day.day_high
        price_l = day.day_low
        price_c = day.day_close

        if direction == "LONG":
            best_price = max(best_price, price_h)
            trailing_stop = best_price * (1.0 - TRAILING_STOP_PCT)
            if best_price > entry_price:
                effective_stop = max(stop_price, trailing_stop)
            else:
                effective_stop = stop_price
            hit_stop = price_l <= effective_stop
            hit_tp = price_h >= tp_price
        else:
            best_price = min(best_price, price_l)
            trailing_stop = best_price * (1.0 + TRAILING_STOP_PCT)
            if best_price < entry_price:
                effective_stop = min(stop_price, trailing_stop)
            else:
                effective_stop = stop_price
            hit_stop = price_h >= effective_stop
            hit_tp = price_l <= tp_price

        exit_reason = None
        exit_price = 0.0

        if hold_days > 0 or j > entry_idx:
            # Don't exit on entry bar (we just entered at open)
            # Actually match combo_backtest: it increments hold_days then checks
            # The original sim enters at next_day open and starts checking from
            # the next iteration. Let's replicate: only check exits starting
            # from the day AFTER entry.
            pass

        if j == entry_idx:
            # Skip exit checks on entry day to match combo_backtest behavior
            # (combo_backtest enters at next_day open, then the for-loop
            # increments hold_days on the NEXT iteration before checking)
            continue

        if hit_stop:
            is_trailing = (
                (direction == "LONG" and best_price > entry_price and trailing_stop > stop_price) or
                (direction == "SHORT" and best_price < entry_price and trailing_stop < stop_price)
            )
            exit_reason = "trailing" if is_trailing else "stop"
            exit_price = effective_stop
        elif hit_tp:
            exit_reason = "tp"
            exit_price = tp_price
        elif hold_days >= MAX_HOLD_DAYS:
            exit_reason = "timeout"
            exit_price = price_c

        if exit_reason:
            pnl = apply_costs(entry_price, exit_price, shares, direction)
            return Trade(
                entry_date=entry_day.date,
                exit_date=day.date,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                confidence=confidence,
                shares=shares,
                pnl=pnl,
                hold_days=hold_days,
                exit_reason=exit_reason,
                source="CS",
            )

    # End of data — close trade
    if len(signals) > 0:
        last = signals[-1]
        exit_price = last.day_close
        pnl = apply_costs(entry_price, exit_price, shares, direction)
        return Trade(
            entry_date=entry_day.date,
            exit_date=last.date,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            confidence=confidence,
            shares=shares,
            pnl=pnl,
            hold_days=len(signals) - 1 - entry_idx,
            exit_reason="end",
            source="CS",
        )
    return None


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def strategy_a_immediate(signals, rsi_arr) -> List[Trade]:
    """Strategy A: Immediate entry (baseline). Enter next-day open."""
    trades = []
    cooldown_until = -1  # index after which we can trade again

    for i, day in enumerate(signals):
        if i <= cooldown_until:
            continue

        if day.cs_action not in ("BUY", "SELL"):
            continue
        if day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            continue

        direction = "LONG" if day.cs_action == "BUY" else "SHORT"
        entry_idx = i + 1

        trade = run_trade_from_entry(signals, entry_idx, direction, day.cs_confidence)
        if trade is None:
            continue

        trades.append(trade)
        # Find exit index for cooldown
        exit_idx = _find_date_index(signals, trade.exit_date, start=entry_idx)
        if exit_idx is not None:
            cooldown_until = exit_idx + COOLDOWN_DAYS
        else:
            cooldown_until = entry_idx + trade.hold_days + COOLDOWN_DAYS

    return trades


def strategy_b_pullback(signals, rsi_arr) -> List[Trade]:
    """Strategy B: Pullback entry. Wait up to 3 days for pullback, else enter day 4."""
    trades = []
    cooldown_until = -1

    for i, day in enumerate(signals):
        if i <= cooldown_until:
            continue

        if day.cs_action not in ("BUY", "SELL"):
            continue
        if day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            continue

        direction = "LONG" if day.cs_action == "BUY" else "SHORT"
        signal_close = day.day_close

        # Look for pullback in next 3 days
        entry_idx = None
        for offset in range(1, 4):  # days 1, 2, 3 after signal
            if i + offset >= len(signals):
                break
            check_day = signals[i + offset]
            if direction == "LONG":
                # Wait for close below signal-day close
                if check_day.day_close < signal_close:
                    entry_idx = i + offset + 1  # enter next open after pullback
                    break
            else:
                # SELL: wait for close above signal-day close
                if check_day.day_close > signal_close:
                    entry_idx = i + offset + 1
                    break

        # No pullback found — enter at day 4 open
        if entry_idx is None:
            entry_idx = i + 4  # day 4 open (0-indexed: signal=i, so i+4)

        trade = run_trade_from_entry(signals, entry_idx, direction, day.cs_confidence)
        if trade is None:
            continue

        trades.append(trade)
        exit_idx = _find_date_index(signals, trade.exit_date, start=entry_idx)
        if exit_idx is not None:
            cooldown_until = exit_idx + COOLDOWN_DAYS
        else:
            cooldown_until = entry_idx + trade.hold_days + COOLDOWN_DAYS

    return trades


def strategy_c_gap_filter(signals, rsi_arr) -> List[Trade]:
    """Strategy C: Gap filter. Skip if next-day gaps > 1% in trade direction."""
    trades = []
    cooldown_until = -1
    skipped = 0

    for i, day in enumerate(signals):
        if i <= cooldown_until:
            continue

        if day.cs_action not in ("BUY", "SELL"):
            continue
        if day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            continue

        direction = "LONG" if day.cs_action == "BUY" else "SHORT"
        entry_idx = i + 1

        if entry_idx >= len(signals):
            continue

        next_day = signals[entry_idx]
        gap_pct = (next_day.day_open - day.day_close) / day.day_close

        # Skip if gap > 1% in our direction
        if direction == "LONG" and gap_pct > 0.01:
            skipped += 1
            continue
        if direction == "SHORT" and gap_pct < -0.01:
            skipped += 1
            continue

        trade = run_trade_from_entry(signals, entry_idx, direction, day.cs_confidence)
        if trade is None:
            continue

        trades.append(trade)
        exit_idx = _find_date_index(signals, trade.exit_date, start=entry_idx)
        if exit_idx is not None:
            cooldown_until = exit_idx + COOLDOWN_DAYS
        else:
            cooldown_until = entry_idx + trade.hold_days + COOLDOWN_DAYS

    print(f"    Gap filter skipped {skipped} trades")
    return trades


def strategy_d_rsi(signals, rsi_arr) -> List[Trade]:
    """Strategy D: RSI confirmation. BUY only if 5d RSI < 40, SELL only if 5d RSI > 60."""
    trades = []
    cooldown_until = -1
    skipped = 0

    for i, day in enumerate(signals):
        if i <= cooldown_until:
            continue

        if day.cs_action not in ("BUY", "SELL"):
            continue
        if day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            continue

        direction = "LONG" if day.cs_action == "BUY" else "SHORT"
        rsi_val = rsi_arr[i]

        if direction == "LONG" and rsi_val >= 40:
            skipped += 1
            continue
        if direction == "SHORT" and rsi_val <= 60:
            skipped += 1
            continue

        entry_idx = i + 1
        trade = run_trade_from_entry(signals, entry_idx, direction, day.cs_confidence)
        if trade is None:
            continue

        trades.append(trade)
        exit_idx = _find_date_index(signals, trade.exit_date, start=entry_idx)
        if exit_idx is not None:
            cooldown_until = exit_idx + COOLDOWN_DAYS
        else:
            cooldown_until = entry_idx + trade.hold_days + COOLDOWN_DAYS

    print(f"    RSI filter skipped {skipped} trades")
    return trades


def strategy_e_range_filter(signals, rsi_arr) -> List[Trade]:
    """Strategy E: Intraday range filter. Skip if signal day range > 5% of close."""
    trades = []
    cooldown_until = -1
    skipped = 0

    for i, day in enumerate(signals):
        if i <= cooldown_until:
            continue

        if day.cs_action not in ("BUY", "SELL"):
            continue
        if day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            continue

        # Check signal day range
        if day.day_close > 0:
            day_range_pct = (day.day_high - day.day_low) / day.day_close
            if day_range_pct > 0.05:
                skipped += 1
                continue

        direction = "LONG" if day.cs_action == "BUY" else "SHORT"
        entry_idx = i + 1

        trade = run_trade_from_entry(signals, entry_idx, direction, day.cs_confidence)
        if trade is None:
            continue

        trades.append(trade)
        exit_idx = _find_date_index(signals, trade.exit_date, start=entry_idx)
        if exit_idx is not None:
            cooldown_until = exit_idx + COOLDOWN_DAYS
        else:
            cooldown_until = entry_idx + trade.hold_days + COOLDOWN_DAYS

    print(f"    Range filter skipped {skipped} trades")
    return trades


def strategy_f_close_position(signals, rsi_arr) -> List[Trade]:
    """Strategy F: Close-position filter. BUY only if close in bottom 30%, SELL top 30%."""
    trades = []
    cooldown_until = -1
    skipped = 0

    for i, day in enumerate(signals):
        if i <= cooldown_until:
            continue

        if day.cs_action not in ("BUY", "SELL"):
            continue
        if day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            continue

        direction = "LONG" if day.cs_action == "BUY" else "SHORT"

        # Close position within day range
        day_range = day.day_high - day.day_low
        if day_range <= 0:
            skipped += 1
            continue

        close_pos = (day.day_close - day.day_low) / day_range  # 0=low, 1=high

        if direction == "LONG" and close_pos >= 0.30:
            skipped += 1
            continue
        if direction == "SHORT" and close_pos <= 0.70:
            skipped += 1
            continue

        entry_idx = i + 1
        trade = run_trade_from_entry(signals, entry_idx, direction, day.cs_confidence)
        if trade is None:
            continue

        trades.append(trade)
        exit_idx = _find_date_index(signals, trade.exit_date, start=entry_idx)
        if exit_idx is not None:
            cooldown_until = exit_idx + COOLDOWN_DAYS
        else:
            cooldown_until = entry_idx + trade.hold_days + COOLDOWN_DAYS

    print(f"    Close-position filter skipped {skipped} trades")
    return trades


def strategy_g_combined(signals, rsi_arr, best_filters: List[str]) -> List[Trade]:
    """Strategy G: Combined best filters from A-F."""
    trades = []
    cooldown_until = -1
    skipped = 0

    for i, day in enumerate(signals):
        if i <= cooldown_until:
            continue

        if day.cs_action not in ("BUY", "SELL"):
            continue
        if day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            continue

        direction = "LONG" if day.cs_action == "BUY" else "SHORT"
        skip = False

        # Apply each selected filter
        if "gap" in best_filters:
            entry_idx = i + 1
            if entry_idx < len(signals):
                next_day = signals[entry_idx]
                gap_pct = (next_day.day_open - day.day_close) / day.day_close
                if direction == "LONG" and gap_pct > 0.01:
                    skip = True
                if direction == "SHORT" and gap_pct < -0.01:
                    skip = True

        if "rsi" in best_filters and not skip:
            rsi_val = rsi_arr[i]
            if direction == "LONG" and rsi_val >= 40:
                skip = True
            if direction == "SHORT" and rsi_val <= 60:
                skip = True

        if "range" in best_filters and not skip:
            if day.day_close > 0:
                day_range_pct = (day.day_high - day.day_low) / day.day_close
                if day_range_pct > 0.05:
                    skip = True

        if "close_pos" in best_filters and not skip:
            day_range = day.day_high - day.day_low
            if day_range > 0:
                close_pos = (day.day_close - day.day_low) / day_range
                if direction == "LONG" and close_pos >= 0.30:
                    skip = True
                if direction == "SHORT" and close_pos <= 0.70:
                    skip = True

        if skip:
            skipped += 1
            continue

        # Pullback entry if selected
        if "pullback" in best_filters:
            signal_close = day.day_close
            entry_idx = None
            for offset in range(1, 4):
                if i + offset >= len(signals):
                    break
                check_day = signals[i + offset]
                if direction == "LONG" and check_day.day_close < signal_close:
                    entry_idx = i + offset + 1
                    break
                if direction == "SHORT" and check_day.day_close > signal_close:
                    entry_idx = i + offset + 1
                    break
            if entry_idx is None:
                entry_idx = i + 4
        else:
            entry_idx = i + 1

        trade = run_trade_from_entry(signals, entry_idx, direction, day.cs_confidence)
        if trade is None:
            continue

        trades.append(trade)
        exit_idx = _find_date_index(signals, trade.exit_date, start=entry_idx)
        if exit_idx is not None:
            cooldown_until = exit_idx + COOLDOWN_DAYS
        else:
            cooldown_until = entry_idx + trade.hold_days + COOLDOWN_DAYS

    print(f"    Combined filter skipped {skipped} trades")
    return trades


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_date_index(signals, date, start=0):
    """Find index of a date in signals list."""
    for j in range(start, len(signals)):
        if signals[j].date == date:
            return j
    return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_strategy(name: str, trades: List[Trade]):
    """Print summary stats for a strategy."""
    n = len(trades)
    if n == 0:
        return {
            "name": name, "trades": 0, "wr": 0, "pnl": 0,
            "avg_win": 0, "avg_loss": 0, "sharpe": 0, "max_dd_pct": 0,
        }

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    wr = len(wins) / n * 100

    pnls = np.array([t.pnl for t in trades])
    avg_hold = max(np.mean([t.hold_days for t in trades]), 1)
    sharpe = (pnls.mean() / pnls.std() * np.sqrt(252 / avg_hold)
              ) if pnls.std() > 0 else 0.0

    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0
    max_dd_pct = max_dd / CAPITAL * 100

    # Exit reasons
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    # Direction breakdown
    longs = [t for t in trades if t.direction == "LONG"]
    shorts = [t for t in trades if t.direction == "SHORT"]

    # Profit factor
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  Trades: {n} | Wins: {len(wins)} ({wr:.1f}%) | Losses: {len(losses)}")
    print(f"  Total PnL: ${total_pnl:+,.0f} | Avg Win: ${avg_win:+,.0f} | Avg Loss: ${avg_loss:+,.0f}")
    print(f"  Sharpe: {sharpe:.2f} | Profit Factor: {pf:.2f} | Max DD: {max_dd_pct:.1f}%")
    print(f"  Avg Hold: {avg_hold:.1f} days | Exits: {reasons}")

    if longs:
        l_wr = sum(1 for t in longs if t.pnl > 0) / len(longs) * 100
        l_pnl = sum(t.pnl for t in longs)
        print(f"  Longs:  {len(longs)} trades, {l_wr:.0f}% WR, ${l_pnl:+,.0f}")
    if shorts:
        s_wr = sum(1 for t in shorts if t.pnl > 0) / len(shorts) * 100
        s_pnl = sum(t.pnl for t in shorts)
        print(f"  Shorts: {len(shorts)} trades, {s_wr:.0f}% WR, ${s_pnl:+,.0f}")

    return {
        "name": name, "trades": n, "wr": wr, "pnl": total_pnl,
        "avg_win": avg_win, "avg_loss": avg_loss, "sharpe": sharpe,
        "max_dd_pct": max_dd_pct, "pf": pf,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  ENTRY TIMING FILTER TEST")
    print("  Testing mean-reversion timing filters on CS-ALL signals")
    print("=" * 70)

    # Load cached signals
    if not CACHE_FILE.exists():
        print(f"\nERROR: Cache file not found: {CACHE_FILE}")
        print("Run combo_backtest.py first to generate the cache.")
        sys.exit(1)

    print(f"\nLoading cached signals from {CACHE_FILE}...")
    t0 = time.time()
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)

    signals = cache["signals"]
    daily_df = cache["daily_df"]
    print(f"Loaded {len(signals):,} trading days in {time.time() - t0:.1f}s")

    # Signal summary
    cs_buy = sum(1 for s in signals if s.cs_action == "BUY" and s.cs_confidence >= MIN_SIGNAL_CONFIDENCE)
    cs_sell = sum(1 for s in signals if s.cs_action == "SELL" and s.cs_confidence >= MIN_SIGNAL_CONFIDENCE)
    print(f"CS-ALL signals (conf >= {MIN_SIGNAL_CONFIDENCE}): {cs_buy} BUY, {cs_sell} SELL, {cs_buy + cs_sell} total")

    # Pre-compute 5-day RSI aligned to signals
    print("Computing 5-day RSI...")
    closes = daily_df["close"].values.astype(float)
    full_rsi = compute_rsi(closes, period=5)

    # Map signal dates to daily_df indices to get RSI values
    date_to_daily_idx = {d: idx for idx, d in enumerate(daily_df.index)}
    rsi_arr = np.full(len(signals), 50.0)
    for i, sig in enumerate(signals):
        daily_idx = date_to_daily_idx.get(sig.date)
        if daily_idx is not None and daily_idx < len(full_rsi):
            rsi_arr[i] = full_rsi[daily_idx]

    print(f"RSI range: {rsi_arr.min():.1f} - {rsi_arr.max():.1f}, mean: {rsi_arr.mean():.1f}")

    # Run all strategies
    results = []

    print("\n--- Running Strategy A: Immediate (baseline) ---")
    trades_a = strategy_a_immediate(signals, rsi_arr)
    results.append(report_strategy("A: Immediate (baseline)", trades_a))

    print("\n--- Running Strategy B: Pullback entry ---")
    trades_b = strategy_b_pullback(signals, rsi_arr)
    results.append(report_strategy("B: Pullback entry", trades_b))

    print("\n--- Running Strategy C: Gap filter ---")
    trades_c = strategy_c_gap_filter(signals, rsi_arr)
    results.append(report_strategy("C: Gap filter", trades_c))

    print("\n--- Running Strategy D: RSI confirmation ---")
    trades_d = strategy_d_rsi(signals, rsi_arr)
    results.append(report_strategy("D: RSI confirmation", trades_d))

    print("\n--- Running Strategy E: Intraday range filter ---")
    trades_e = strategy_e_range_filter(signals, rsi_arr)
    results.append(report_strategy("E: Range filter", trades_e))

    print("\n--- Running Strategy F: Close-position filter ---")
    trades_f = strategy_f_close_position(signals, rsi_arr)
    results.append(report_strategy("F: Close-position filter", trades_f))

    # Determine best filters for G
    # Compare each filter to baseline on Sharpe improvement
    baseline_sharpe = results[0]["sharpe"]
    baseline_pnl = results[0]["pnl"]
    filter_map = {
        "B: Pullback entry": "pullback",
        "C: Gap filter": "gap",
        "D: RSI confirmation": "rsi",
        "E: Range filter": "range",
        "F: Close-position filter": "close_pos",
    }

    print("\n" + "-" * 65)
    print("  Selecting best filters for Strategy G...")
    print("-" * 65)

    best_filters = []
    for r in results[1:]:
        name = r["name"]
        tag = filter_map.get(name)
        if tag is None:
            continue
        # A filter is "good" if it improves Sharpe OR improves PnL with decent WR
        sharpe_better = r["sharpe"] > baseline_sharpe
        pnl_better = r["pnl"] > baseline_pnl and r["wr"] >= results[0]["wr"] - 3
        if sharpe_better or pnl_better:
            best_filters.append(tag)
            print(f"  + {name}: Sharpe {r['sharpe']:.2f} vs {baseline_sharpe:.2f} -> SELECTED")
        else:
            print(f"  - {name}: Sharpe {r['sharpe']:.2f} vs {baseline_sharpe:.2f} -> skipped")

    if not best_filters:
        print("  No individual filter beat baseline, using gap + close_pos as default")
        best_filters = ["gap", "close_pos"]

    print(f"\n  Combined filters: {best_filters}")

    print(f"\n--- Running Strategy G: Combined ({' + '.join(best_filters)}) ---")
    trades_g = strategy_g_combined(signals, rsi_arr, best_filters)
    results.append(report_strategy(f"G: Combined ({'+'.join(best_filters)})", trades_g))

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"{'Strategy':<35} {'Trades':>6} {'WR%':>6} {'PnL':>11} {'AvgWin':>9} "
          f"{'AvgLoss':>9} {'Sharpe':>7} {'PF':>6} {'DD%':>6}")
    print(f"{'-'*90}")

    for r in results:
        n = r["trades"]
        if n == 0:
            print(f"{r['name']:<35} {'0':>6} {'---':>6} {'---':>11} {'---':>9} "
                  f"{'---':>9} {'---':>7} {'---':>6} {'---':>6}")
            continue
        print(f"{r['name']:<35} {n:>6} {r['wr']:>5.1f}% ${r['pnl']:>+9,.0f} "
              f"${r['avg_win']:>+8,.0f} ${r['avg_loss']:>+8,.0f} "
              f"{r['sharpe']:>7.2f} {r.get('pf', 0):>5.2f} {r['max_dd_pct']:>5.1f}%")

    # Delta vs baseline
    print(f"\n{'='*90}")
    print(f"  DELTA vs BASELINE (Strategy A)")
    print(f"{'='*90}")
    print(f"{'Strategy':<35} {'dTrades':>8} {'dWR':>7} {'dPnL':>11} {'dSharpe':>8}")
    print(f"{'-'*90}")

    base = results[0]
    for r in results[1:]:
        dt = r["trades"] - base["trades"]
        dwr = r["wr"] - base["wr"]
        dpnl = r["pnl"] - base["pnl"]
        dsh = r["sharpe"] - base["sharpe"]
        print(f"{r['name']:<35} {dt:>+8} {dwr:>+6.1f}% ${dpnl:>+9,.0f} {dsh:>+7.2f}")

    print(f"\n{'='*90}")
    print("  Done.")


if __name__ == "__main__":
    main()
