#!/usr/bin/env python3
"""
Seasonality Test: Day-of-week and month-of-year filters for TSLA combo backtest.

Hypothesis: certain days/months may have systematically better/worse performance
for channel-based signals.

Loads cached signals from combo_cache/combo_signals.pkl, runs CS-ALL baseline
simulation, then analyzes performance by:
  - Day of week (signal day vs entry day)
  - Month of year
  - Quarter
  - Signal type (bounce vs break)
  - Direction x month

Then tests filtered strategies that skip worst-performing time periods.

Usage:
    cd C:\AI\x14
    python -m v15.validation.seasonality_test
"""

import os
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ---------------------------------------------------------------------------
# Constants (match combo_backtest.py exactly)
# ---------------------------------------------------------------------------

MIN_SIGNAL_CONFIDENCE = 0.45
CAPITAL = 100_000.0
DEFAULT_STOP_PCT = 0.02
DEFAULT_TP_PCT = 0.04
TRAILING_STOP_PCT = 0.015
MAX_HOLD_DAYS = 10
COOLDOWN_DAYS = 2
SLIPPAGE_PCT = 0.0001       # 0.01% per side
COMMISSION_PER_SHARE = 0.005

CACHE_DIR = Path(__file__).parent / 'combo_cache'
CACHE_FILE = CACHE_DIR / 'combo_signals.pkl'

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
QUARTER_NAMES = ['Q1', 'Q2', 'Q3', 'Q4']

# ---------------------------------------------------------------------------
# Import DaySignals from combo_backtest (needed for pickle)
# ---------------------------------------------------------------------------
from v15.validation.combo_backtest import DaySignals


# ---------------------------------------------------------------------------
# Trade dataclass (same as combo_backtest)
# ---------------------------------------------------------------------------

@dataclass
class Trade:
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
    # Extra fields for seasonality analysis
    signal_date: pd.Timestamp = None
    signal_type: str = 'bounce'      # bounce or break
    cs_action: str = 'BUY'           # original action


# ---------------------------------------------------------------------------
# Trade simulation (matches combo_backtest.py exactly)
# ---------------------------------------------------------------------------

def _apply_costs(entry_price: float, exit_price: float, shares: int,
                 direction: str) -> float:
    slip_entry = entry_price * SLIPPAGE_PCT
    slip_exit = exit_price * SLIPPAGE_PCT
    comm = COMMISSION_PER_SHARE * shares * 2

    if direction == 'LONG':
        pnl = (exit_price - slip_exit - entry_price - slip_entry) * shares - comm
    else:
        pnl = (entry_price - slip_entry - exit_price - slip_exit) * shares - comm
    return pnl


def _floor_stop_tp(stop, tp):
    return max(stop, DEFAULT_STOP_PCT), max(tp, DEFAULT_TP_PCT)


def simulate_cs_all(signals: list,
                    skip_signal_days: set = None,
                    skip_entry_days: set = None,
                    skip_months: set = None,
                    skip_quarters: set = None,
                    only_signal_days: set = None) -> List[Trade]:
    """
    Run CS-ALL simulation with optional day/month/quarter filters.

    Filters are applied to the SIGNAL date (not entry date) unless
    skip_entry_days is used.

    skip_signal_days: set of weekday ints (0=Mon..4=Fri) to skip
    skip_entry_days: set of weekday ints for entry day filtering
    skip_months: set of month ints (1-12) to skip
    skip_quarters: set of quarter ints (1-4) to skip
    only_signal_days: if set, ONLY trade on these weekday ints
    """
    trades: List[Trade] = []
    in_trade = False
    cooldown_remaining = 0

    entry_date = None
    entry_price = 0.0
    direction = ''
    confidence = 0.0
    shares = 0
    stop_price = 0.0
    tp_price = 0.0
    best_price = 0.0
    hold_days = 0
    source = ''
    signal_date = None
    signal_type = 'bounce'
    cs_action_str = 'BUY'

    for day_idx, day in enumerate(signals):
        if in_trade:
            hold_days += 1
            price_h = day.day_high
            price_l = day.day_low
            price_c = day.day_close

            if direction == 'LONG':
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

            if hit_stop:
                exit_reason = ('trailing'
                               if (direction == 'LONG' and best_price > entry_price
                                   and trailing_stop > stop_price)
                               or (direction == 'SHORT' and best_price < entry_price
                                   and trailing_stop < stop_price)
                               else 'stop')
                exit_price = effective_stop
            elif hit_tp:
                exit_reason = 'tp'
                exit_price = tp_price
            elif hold_days >= MAX_HOLD_DAYS:
                exit_reason = 'timeout'
                exit_price = price_c

            if exit_reason:
                pnl = _apply_costs(entry_price, exit_price, shares, direction)
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=day.date,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    confidence=confidence,
                    shares=shares,
                    pnl=pnl,
                    hold_days=hold_days,
                    exit_reason=exit_reason,
                    source=source,
                    signal_date=signal_date,
                    signal_type=signal_type,
                    cs_action=cs_action_str,
                ))
                in_trade = False
                cooldown_remaining = COOLDOWN_DAYS
                continue

        # Cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        # Check CS-ALL signal
        if day.cs_action not in ('BUY', 'SELL'):
            continue
        if day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            continue

        # --- Apply seasonality filters on SIGNAL day ---
        sig_dow = day.date.weekday()  # 0=Mon..4=Fri
        sig_month = day.date.month    # 1-12
        sig_quarter = (sig_month - 1) // 3 + 1  # 1-4

        if skip_signal_days and sig_dow in skip_signal_days:
            continue
        if only_signal_days is not None and sig_dow not in only_signal_days:
            continue
        if skip_months and sig_month in skip_months:
            continue
        if skip_quarters and sig_quarter in skip_quarters:
            continue

        # Entry at next-day open
        if day_idx + 1 >= len(signals):
            break
        next_day = signals[day_idx + 1]

        # --- Apply entry-day filter ---
        if skip_entry_days:
            entry_dow = next_day.date.weekday()
            if entry_dow in skip_entry_days:
                continue

        entry_price = next_day.day_open
        if entry_price <= 0:
            continue

        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)

        entry_date = next_day.date
        signal_date = day.date
        signal_type = day.cs_signal_type
        cs_action_str = day.cs_action
        confidence = day.cs_confidence
        source = 'CS'

        position_value = CAPITAL * min(day.cs_confidence, 1.0)
        shares = max(1, int(position_value / entry_price))

        if day.cs_action == 'BUY':
            direction = 'LONG'
            stop_price = entry_price * (1.0 - s)
            tp_price = entry_price * (1.0 + t)
            best_price = entry_price
        else:
            direction = 'SHORT'
            stop_price = entry_price * (1.0 + s)
            tp_price = entry_price * (1.0 - t)
            best_price = entry_price

        in_trade = True
        hold_days = 0

    # Close open trade at end
    if in_trade and len(signals) > 0:
        last = signals[-1]
        exit_price = last.day_close
        pnl = _apply_costs(entry_price, exit_price, shares, direction)
        trades.append(Trade(
            entry_date=entry_date,
            exit_date=last.date,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            confidence=confidence,
            shares=shares,
            pnl=pnl,
            hold_days=hold_days,
            exit_reason='end',
            source=source,
            signal_date=signal_date,
            signal_type=signal_type,
            cs_action=cs_action_str,
        ))

    return trades


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _wr(trades: List[Trade]) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.pnl > 0) / len(trades) * 100


def _avg_pnl(trades: List[Trade]) -> float:
    if not trades:
        return 0.0
    return np.mean([t.pnl for t in trades])


def _total_pnl(trades: List[Trade]) -> float:
    return sum(t.pnl for t in trades)


def _sharpe(trades: List[Trade]) -> float:
    if len(trades) < 2:
        return 0.0
    pnls = np.array([t.pnl for t in trades])
    if pnls.std() == 0:
        return 0.0
    avg_hold = max(np.mean([t.hold_days for t in trades]), 1)
    return pnls.mean() / pnls.std() * np.sqrt(252 / avg_hold)


def _max_dd_pct(trades: List[Trade]) -> float:
    if not trades:
        return 0.0
    pnls = np.array([t.pnl for t in trades])
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    return float(dd.max()) / CAPITAL * 100


def _profit_factor(trades: List[Trade]) -> float:
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    if gross_loss == 0:
        return float('inf') if gross_win > 0 else 0.0
    return gross_win / gross_loss


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze_seasonality(trades: List[Trade]):
    """Full seasonality breakdown of CS-ALL trades."""

    print_section("BASELINE: CS-ALL (286 expected trades)")
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    total = _total_pnl(trades)
    print(f"  Trades: {n} | WR: {_wr(trades):.1f}% | PnL: ${total:+,.0f} | "
          f"Sharpe: {_sharpe(trades):.2f} | MaxDD: {_max_dd_pct(trades):.1f}%")

    # ------------------------------------------------------------------
    # 3a) Win rate by day-of-week of SIGNAL day
    # ------------------------------------------------------------------
    print_section("3a) Win Rate by Day-of-Week (SIGNAL day)")
    by_sig_dow = defaultdict(list)
    for t in trades:
        dow = t.signal_date.weekday()
        by_sig_dow[dow].append(t)

    print(f"  {'Day':<6} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'AvgPnL':>10} "
          f"{'TotalPnL':>11} {'PF':>6}")
    print(f"  {'-'*55}")
    for dow in range(5):
        subset = by_sig_dow[dow]
        if not subset:
            print(f"  {DAY_NAMES[dow]:<6} {'0':>7}")
            continue
        w = sum(1 for t in subset if t.pnl > 0)
        print(f"  {DAY_NAMES[dow]:<6} {len(subset):>7} {w:>6} {_wr(subset):>6.1f}% "
              f"${_avg_pnl(subset):>+9,.0f} ${_total_pnl(subset):>+10,.0f} "
              f"{_profit_factor(subset):>5.2f}")

    # ------------------------------------------------------------------
    # 3b) Win rate by day-of-week of ENTRY day
    # ------------------------------------------------------------------
    print_section("3b) Win Rate by Day-of-Week (ENTRY day)")
    by_entry_dow = defaultdict(list)
    for t in trades:
        dow = t.entry_date.weekday()
        by_entry_dow[dow].append(t)

    print(f"  {'Day':<6} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'AvgPnL':>10} "
          f"{'TotalPnL':>11} {'PF':>6}")
    print(f"  {'-'*55}")
    for dow in range(5):
        subset = by_entry_dow[dow]
        if not subset:
            print(f"  {DAY_NAMES[dow]:<6} {'0':>7}")
            continue
        w = sum(1 for t in subset if t.pnl > 0)
        print(f"  {DAY_NAMES[dow]:<6} {len(subset):>7} {w:>6} {_wr(subset):>6.1f}% "
              f"${_avg_pnl(subset):>+9,.0f} ${_total_pnl(subset):>+10,.0f} "
              f"{_profit_factor(subset):>5.2f}")

    # ------------------------------------------------------------------
    # 3c) Win rate by month
    # ------------------------------------------------------------------
    print_section("3c) Win Rate by Month")
    by_month = defaultdict(list)
    for t in trades:
        m = t.signal_date.month
        by_month[m].append(t)

    print(f"  {'Month':<6} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'AvgPnL':>10} "
          f"{'TotalPnL':>11} {'PF':>6}")
    print(f"  {'-'*55}")
    for m in range(1, 13):
        subset = by_month[m]
        if not subset:
            print(f"  {MONTH_NAMES[m-1]:<6} {'0':>7}")
            continue
        w = sum(1 for t in subset if t.pnl > 0)
        print(f"  {MONTH_NAMES[m-1]:<6} {len(subset):>7} {w:>6} {_wr(subset):>6.1f}% "
              f"${_avg_pnl(subset):>+9,.0f} ${_total_pnl(subset):>+10,.0f} "
              f"{_profit_factor(subset):>5.2f}")

    # ------------------------------------------------------------------
    # 3d) Win rate by quarter
    # ------------------------------------------------------------------
    print_section("3d) Win Rate by Quarter")
    by_quarter = defaultdict(list)
    for t in trades:
        q = (t.signal_date.month - 1) // 3 + 1
        by_quarter[q].append(t)

    print(f"  {'Qtr':<6} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'AvgPnL':>10} "
          f"{'TotalPnL':>11} {'PF':>6}")
    print(f"  {'-'*55}")
    for q in range(1, 5):
        subset = by_quarter[q]
        if not subset:
            print(f"  {QUARTER_NAMES[q-1]:<6} {'0':>7}")
            continue
        w = sum(1 for t in subset if t.pnl > 0)
        print(f"  {QUARTER_NAMES[q-1]:<6} {len(subset):>7} {w:>6} {_wr(subset):>6.1f}% "
              f"${_avg_pnl(subset):>+9,.0f} ${_total_pnl(subset):>+10,.0f} "
              f"{_profit_factor(subset):>5.2f}")

    # ------------------------------------------------------------------
    # 3e) Average PnL by day-of-week (same data, just emphasizing avg)
    # ------------------------------------------------------------------
    print_section("3e) Average PnL by Day-of-Week")
    print(f"  {'Day':<6} {'AvgPnL':>10} {'Median':>10} {'StdDev':>10} {'Best':>10} {'Worst':>10}")
    print(f"  {'-'*56}")
    for dow in range(5):
        subset = by_sig_dow[dow]
        if not subset:
            continue
        pnls = [t.pnl for t in subset]
        print(f"  {DAY_NAMES[dow]:<6} ${np.mean(pnls):>+9,.0f} ${np.median(pnls):>+9,.0f} "
              f"${np.std(pnls):>9,.0f} ${max(pnls):>+9,.0f} ${min(pnls):>+9,.0f}")

    # ------------------------------------------------------------------
    # 3f) Average PnL by month
    # ------------------------------------------------------------------
    print_section("3f) Average PnL by Month")
    print(f"  {'Month':<6} {'AvgPnL':>10} {'Median':>10} {'StdDev':>10} {'Best':>10} {'Worst':>10}")
    print(f"  {'-'*56}")
    for m in range(1, 13):
        subset = by_month[m]
        if not subset:
            continue
        pnls = [t.pnl for t in subset]
        print(f"  {MONTH_NAMES[m-1]:<6} ${np.mean(pnls):>+9,.0f} ${np.median(pnls):>+9,.0f} "
              f"${np.std(pnls):>9,.0f} ${max(pnls):>+9,.0f} ${min(pnls):>+9,.0f}")

    # ------------------------------------------------------------------
    # 3g) Monday vs Friday
    # ------------------------------------------------------------------
    print_section("3g) Monday vs Friday Entry")
    for dow, name in [(0, 'Monday'), (4, 'Friday')]:
        subset = by_entry_dow[dow]
        if not subset:
            print(f"  {name}: No trades")
            continue
        print(f"  {name} entries: {len(subset)} trades | "
              f"WR: {_wr(subset):.1f}% | "
              f"AvgPnL: ${_avg_pnl(subset):+,.0f} | "
              f"Total: ${_total_pnl(subset):+,.0f}")

    # ------------------------------------------------------------------
    # 3h) Win rate by signal_type (bounce vs break)
    # ------------------------------------------------------------------
    print_section("3h) Win Rate by Signal Type")
    by_type = defaultdict(list)
    for t in trades:
        by_type[t.signal_type].append(t)

    print(f"  {'Type':<12} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'AvgPnL':>10} "
          f"{'TotalPnL':>11} {'PF':>6}")
    print(f"  {'-'*60}")
    for stype in sorted(by_type.keys()):
        subset = by_type[stype]
        w = sum(1 for t in subset if t.pnl > 0)
        print(f"  {stype:<12} {len(subset):>7} {w:>6} {_wr(subset):>6.1f}% "
              f"${_avg_pnl(subset):>+9,.0f} ${_total_pnl(subset):>+10,.0f} "
              f"{_profit_factor(subset):>5.2f}")

    # ------------------------------------------------------------------
    # 3i) Win rate by direction per month
    # ------------------------------------------------------------------
    print_section("3i) Win Rate by Direction x Month")
    print(f"  {'Month':<6} {'LONG_N':>7} {'LONG_WR':>8} {'LONG_PnL':>10} | "
          f"{'SHORT_N':>7} {'SHORT_WR':>8} {'SHORT_PnL':>10}")
    print(f"  {'-'*70}")
    for m in range(1, 13):
        subset = by_month[m]
        if not subset:
            continue
        longs = [t for t in subset if t.direction == 'LONG']
        shorts = [t for t in subset if t.direction == 'SHORT']
        l_n = len(longs)
        s_n = len(shorts)
        l_wr = _wr(longs)
        s_wr = _wr(shorts)
        l_pnl = _total_pnl(longs)
        s_pnl = _total_pnl(shorts)
        print(f"  {MONTH_NAMES[m-1]:<6} {l_n:>7} {l_wr:>7.1f}% ${l_pnl:>+9,.0f} | "
              f"{s_n:>7} {s_wr:>7.1f}% ${s_pnl:>+9,.0f}")

    # Return analysis data for filter strategy construction
    return by_sig_dow, by_entry_dow, by_month, by_quarter


def find_worst_best(by_group: dict, group_names: list, offset: int = 0):
    """Find worst and best groups by win rate (with min 5 trades)."""
    group_stats = []
    for key, subset in sorted(by_group.items()):
        if len(subset) < 5:
            continue
        wr = _wr(subset)
        avg = _avg_pnl(subset)
        name = group_names[key - offset] if isinstance(group_names, list) else str(key)
        group_stats.append((key, name, len(subset), wr, avg, _total_pnl(subset)))

    group_stats.sort(key=lambda x: x[3])  # sort by WR ascending
    return group_stats


# ---------------------------------------------------------------------------
# Filtered strategy simulation
# ---------------------------------------------------------------------------

def run_filtered_strategies(signals: list, baseline_trades: List[Trade],
                            by_sig_dow: dict, by_entry_dow: dict,
                            by_month: dict, by_quarter: dict):
    """Run S1-S6 filtered strategies and compare to baseline."""

    print_section("FILTERED STRATEGY ANALYSIS")

    # Determine worst/best groups
    dow_stats = find_worst_best(by_sig_dow, DAY_NAMES, offset=0)
    month_stats = find_worst_best(by_month, MONTH_NAMES, offset=1)
    quarter_stats = find_worst_best(by_quarter, QUARTER_NAMES, offset=1)

    print("\n  Day-of-week ranking (by WR, signal day):")
    for key, name, n, wr, avg, total in dow_stats:
        print(f"    {name}: {n} trades, {wr:.1f}% WR, ${avg:+,.0f} avg, ${total:+,.0f} total")

    print("\n  Month ranking (by WR):")
    for key, name, n, wr, avg, total in month_stats:
        print(f"    {name}: {n} trades, {wr:.1f}% WR, ${avg:+,.0f} avg, ${total:+,.0f} total")

    print("\n  Quarter ranking (by WR):")
    for key, name, n, wr, avg, total in quarter_stats:
        print(f"    {name}: {n} trades, {wr:.1f}% WR, ${avg:+,.0f} avg, ${total:+,.0f} total")

    # Define strategies
    worst_day = dow_stats[0][0] if dow_stats else None
    worst_2_days = set(s[0] for s in dow_stats[:2]) if len(dow_stats) >= 2 else set()
    best_2_days = set(s[0] for s in dow_stats[-2:]) if len(dow_stats) >= 2 else set()
    worst_month = month_stats[0][0] if month_stats else None
    worst_quarter = quarter_stats[0][0] if quarter_stats else None
    best_day = dow_stats[-1][0] if dow_stats else None
    best_month_key = month_stats[-1][0] if month_stats else None

    strategies = []

    # S1: Skip worst day
    if worst_day is not None:
        worst_day_name = DAY_NAMES[worst_day]
        s1 = simulate_cs_all(signals, skip_signal_days={worst_day})
        strategies.append((f"S1: Skip {worst_day_name} (worst day)", s1))

    # S2: Skip worst 2 days
    if len(worst_2_days) >= 2:
        d_names = '+'.join(DAY_NAMES[d] for d in sorted(worst_2_days))
        s2 = simulate_cs_all(signals, skip_signal_days=worst_2_days)
        strategies.append((f"S2: Skip {d_names} (worst 2)", s2))

    # S3: Skip worst month
    if worst_month is not None:
        worst_m_name = MONTH_NAMES[worst_month - 1]
        s3 = simulate_cs_all(signals, skip_months={worst_month})
        strategies.append((f"S3: Skip {worst_m_name} (worst month)", s3))

    # S4: Skip worst quarter
    if worst_quarter is not None:
        worst_q_name = QUARTER_NAMES[worst_quarter - 1]
        s4 = simulate_cs_all(signals, skip_quarters={worst_quarter})
        strategies.append((f"S4: Skip {worst_q_name} (worst quarter)", s4))

    # S5: Best 2 days only
    if len(best_2_days) >= 2:
        d_names = '+'.join(DAY_NAMES[d] for d in sorted(best_2_days))
        s5 = simulate_cs_all(signals, only_signal_days=best_2_days)
        strategies.append((f"S5: Only {d_names} (best 2 days)", s5))

    # S6: Best combo (best day + skip worst month)
    if best_day is not None and worst_month is not None:
        best_day_name = DAY_NAMES[best_day]
        worst_m_name = MONTH_NAMES[worst_month - 1]
        s6 = simulate_cs_all(signals,
                             only_signal_days={best_day},
                             skip_months={worst_month})
        strategies.append((f"S6: Only {best_day_name} + skip {worst_m_name}", s6))

    # Print comparison table
    print_section("STRATEGY COMPARISON")
    print(f"  {'Strategy':<42} {'Trades':>6} {'WR%':>6} {'PnL':>10} {'AvgPnL':>9} "
          f"{'Sharpe':>7} {'MaxDD%':>7} {'PF':>6}")
    print(f"  {'-'*95}")

    # Baseline
    bl = baseline_trades
    print(f"  {'BASELINE: CS-ALL':<42} {len(bl):>6} {_wr(bl):>5.1f}% "
          f"${_total_pnl(bl):>+9,.0f} ${_avg_pnl(bl):>+8,.0f} "
          f"{_sharpe(bl):>7.2f} {_max_dd_pct(bl):>6.1f}% "
          f"{_profit_factor(bl):>5.2f}")

    for name, strades in strategies:
        if not strades:
            print(f"  {name:<42} {'0':>6}")
            continue
        print(f"  {name:<42} {len(strades):>6} {_wr(strades):>5.1f}% "
              f"${_total_pnl(strades):>+9,.0f} ${_avg_pnl(strades):>+8,.0f} "
              f"{_sharpe(strades):>7.2f} {_max_dd_pct(strades):>6.1f}% "
              f"{_profit_factor(strades):>5.2f}")

    # Delta vs baseline
    print(f"\n  {'--- Delta vs Baseline ---':^95}")
    print(f"  {'Strategy':<42} {'dTrades':>7} {'dWR':>7} {'dPnL':>10} {'dSharpe':>8}")
    print(f"  {'-'*75}")
    bl_wr = _wr(bl)
    bl_pnl = _total_pnl(bl)
    bl_sh = _sharpe(bl)
    for name, strades in strategies:
        if not strades:
            continue
        dt = len(strades) - len(bl)
        dwr = _wr(strades) - bl_wr
        dpnl = _total_pnl(strades) - bl_pnl
        dsh = _sharpe(strades) - bl_sh
        print(f"  {name:<42} {dt:>+7} {dwr:>+6.1f}% ${dpnl:>+9,.0f} {dsh:>+7.2f}")

    # ------------------------------------------------------------------
    # Year-by-year breakdown for best strategy
    # ------------------------------------------------------------------
    if strategies:
        # Find best by Sharpe
        best_strat = max(strategies, key=lambda x: _sharpe(x[1]) if x[1] else -999)
        best_name, best_trades = best_strat

        print_section(f"YEAR-BY-YEAR: {best_name}")
        by_year = defaultdict(list)
        for t in best_trades:
            by_year[t.entry_date.year].append(t)

        bl_by_year = defaultdict(list)
        for t in bl:
            bl_by_year[t.entry_date.year].append(t)

        print(f"  {'Year':<6} {'BL_N':>5} {'BL_WR':>6} {'BL_PnL':>10} | "
              f"{'S_N':>5} {'S_WR':>6} {'S_PnL':>10} {'dWR':>7} {'dPnL':>10}")
        print(f"  {'-'*75}")
        all_years = sorted(set(list(by_year.keys()) + list(bl_by_year.keys())))
        for yr in all_years:
            bl_sub = bl_by_year[yr]
            s_sub = by_year[yr]
            bl_n = len(bl_sub)
            s_n = len(s_sub)
            bl_wr_y = _wr(bl_sub)
            s_wr_y = _wr(s_sub)
            bl_pnl_y = _total_pnl(bl_sub)
            s_pnl_y = _total_pnl(s_sub)
            dwr = s_wr_y - bl_wr_y
            dpnl = s_pnl_y - bl_pnl_y
            print(f"  {yr:<6} {bl_n:>5} {bl_wr_y:>5.1f}% ${bl_pnl_y:>+9,.0f} | "
                  f"{s_n:>5} {s_wr_y:>5.1f}% ${s_pnl_y:>+9,.0f} {dwr:>+6.1f}% ${dpnl:>+9,.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  TSLA Seasonality Test: Day-of-Week & Month Filters")
    print("=" * 70)

    # Load cached signals
    if not CACHE_FILE.exists():
        print(f"ERROR: Cache file not found: {CACHE_FILE}")
        print("Run combo_backtest.py first to generate the cache.")
        sys.exit(1)

    print(f"\nLoading cached signals from {CACHE_FILE}...")
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)

    signals = cache['signals']
    print(f"Loaded {len(signals):,} trading days")

    # Quick sanity
    cs_buy = sum(1 for s in signals if s.cs_action == 'BUY' and s.cs_confidence >= MIN_SIGNAL_CONFIDENCE)
    cs_sell = sum(1 for s in signals if s.cs_action == 'SELL' and s.cs_confidence >= MIN_SIGNAL_CONFIDENCE)
    print(f"CS signals >= {MIN_SIGNAL_CONFIDENCE}: BUY={cs_buy}, SELL={cs_sell}")
    print(f"Date range: {signals[0].date.date()} to {signals[-1].date.date()}")

    # Run baseline CS-ALL simulation
    print("\nRunning baseline CS-ALL simulation...")
    baseline_trades = simulate_cs_all(signals)
    print(f"Baseline: {len(baseline_trades)} trades")

    # Seasonality analysis
    by_sig_dow, by_entry_dow, by_month, by_quarter = analyze_seasonality(baseline_trades)

    # Filtered strategies
    run_filtered_strategies(signals, baseline_trades,
                            by_sig_dow, by_entry_dow, by_month, by_quarter)

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
