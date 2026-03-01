#!/usr/bin/env python3
"""
Adaptive Trailing Stop Study

Tests multiple trailing-stop strategies on the CS-ALL signal set to determine
whether ATR-based or regime-adaptive stops improve over the fixed 1.5% trail.

Strategies tested:
  1. Fixed 1.5% trailing (baseline)
  2. ATR-based: trail = 1.0 * ATR(14) as % of price
  3. ATR-based: trail = 1.5 * ATR(14) as % of price
  4. Asymmetric: 1.0% short, 2.0% long
  5. Time-decay: start 2.5%, tighten by 0.15%/day
  6. Volatility regime: 1.0% when ATR < median, 2.0% when ATR > median
  7. Confidence-scaled: trail = 2.5% * (1 - confidence)

Usage:
    python -m v15.validation.adaptive_stop_test
"""

import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ---------------------------------------------------------------------------
# Constants (match combo_backtest.py)
# ---------------------------------------------------------------------------

MIN_SIGNAL_CONFIDENCE = 0.45
CAPITAL = 100_000.0
DEFAULT_STOP_PCT = 0.02
DEFAULT_TP_PCT = 0.04
MAX_HOLD_DAYS = 10
COOLDOWN_DAYS = 2
SLIPPAGE_PCT = 0.0001       # 0.01% per side
COMMISSION_PER_SHARE = 0.005
TRAIN_END_YEAR = 2021

CACHE_FILE = Path(__file__).parent / 'combo_cache' / 'combo_signals.pkl'

# ---------------------------------------------------------------------------
# DaySignals / Trade (duplicated to keep this script standalone)
# ---------------------------------------------------------------------------

@dataclass
class DaySignals:
    """Pre-computed signals for a single trading day."""
    date: pd.Timestamp
    cs_action: str = 'HOLD'
    cs_confidence: float = 0.0
    cs_stop_pct: float = DEFAULT_STOP_PCT
    cs_tp_pct: float = DEFAULT_TP_PCT
    cs_signal_type: str = 'bounce'
    cs_primary_tf: str = ''
    cs_reason: str = ''
    v5_take_bounce: bool = False
    v5_confidence: float = 0.0
    v5_delay_hours: int = 0
    day_open: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    day_close: float = 0.0
    cs_position_score: float = 0.0
    cs_energy_score: float = 0.0
    cs_entropy_score: float = 0.0
    cs_confluence_score: float = 0.0
    cs_timing_score: float = 0.0
    cs_channel_health: float = 0.0
    cs_tf_states: Optional[Dict] = None


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
# ATR computation
# ---------------------------------------------------------------------------

def compute_atr_series(daily_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ATR(period) from daily OHLCV, returning a Series aligned to daily_df index.
    Returns ATR as a percentage of closing price for each bar.
    """
    high = daily_df['high'].values.astype(float)
    low = daily_df['low'].values.astype(float)
    close = daily_df['close'].values.astype(float)

    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))

    # EMA-style ATR
    atr = np.zeros(n)
    atr[:period] = np.nan
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # ATR as % of price
    atr_pct = np.zeros(n)
    for i in range(n):
        if close[i] > 0 and not np.isnan(atr[i]):
            atr_pct[i] = atr[i] / close[i]
        else:
            atr_pct[i] = 0.02  # fallback

    return pd.Series(atr_pct, index=daily_df.index, name='atr_pct')


# ---------------------------------------------------------------------------
# Trade cost helper
# ---------------------------------------------------------------------------

def _apply_costs(entry_price: float, exit_price: float, shares: int,
                 direction: str) -> float:
    """Compute raw PnL after slippage + commission."""
    slip_entry = entry_price * SLIPPAGE_PCT
    slip_exit = exit_price * SLIPPAGE_PCT
    comm = COMMISSION_PER_SHARE * shares * 2

    if direction == 'LONG':
        pnl = (exit_price - slip_exit - entry_price - slip_entry) * shares - comm
    else:
        pnl = (entry_price - slip_entry - exit_price - slip_exit) * shares - comm
    return pnl


# ---------------------------------------------------------------------------
# CS-ALL combo function
# ---------------------------------------------------------------------------

def _floor_stop_tp(stop, tp):
    return max(stop, DEFAULT_STOP_PCT), max(tp, DEFAULT_TP_PCT)


def cs_all_combo(day: DaySignals):
    """CS-ALL: BUY + SELL signals."""
    if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, day.cs_confidence, s, t, 'CS')
    return None


# ---------------------------------------------------------------------------
# Trailing stop strategy interface
# ---------------------------------------------------------------------------

class TrailingStopStrategy:
    """Base class for trailing stop strategies."""

    def __init__(self, name: str):
        self.name = name

    def get_trailing_pct(self, direction: str, hold_days: int,
                         confidence: float, entry_date: pd.Timestamp,
                         atr_lookup: Dict[pd.Timestamp, float]) -> float:
        """Return the trailing stop % for the current bar."""
        raise NotImplementedError


class FixedTrailing(TrailingStopStrategy):
    """Strategy 1: Fixed trailing stop."""
    def __init__(self, pct: float = 0.015):
        super().__init__(f"Fixed {pct*100:.1f}%")
        self.pct = pct

    def get_trailing_pct(self, direction, hold_days, confidence, entry_date, atr_lookup):
        return self.pct


class ATRTrailing(TrailingStopStrategy):
    """Strategy 2/3: ATR-based trailing stop."""
    def __init__(self, multiplier: float = 1.0):
        super().__init__(f"ATR x{multiplier:.1f}")
        self.multiplier = multiplier

    def get_trailing_pct(self, direction, hold_days, confidence, entry_date, atr_lookup):
        atr_pct = atr_lookup.get(entry_date, 0.02)
        return self.multiplier * atr_pct


class AsymmetricTrailing(TrailingStopStrategy):
    """Strategy 4: Different trails for longs vs shorts."""
    def __init__(self, long_pct: float = 0.02, short_pct: float = 0.01):
        super().__init__(f"Asym L={long_pct*100:.1f}% S={short_pct*100:.1f}%")
        self.long_pct = long_pct
        self.short_pct = short_pct

    def get_trailing_pct(self, direction, hold_days, confidence, entry_date, atr_lookup):
        return self.long_pct if direction == 'LONG' else self.short_pct


class TimeDecayTrailing(TrailingStopStrategy):
    """Strategy 5: Start wide, tighten over time."""
    def __init__(self, start_pct: float = 0.025, decay_per_day: float = 0.0015,
                 floor_pct: float = 0.005):
        super().__init__(f"TimeDecay {start_pct*100:.1f}% -{decay_per_day*100:.2f}%/d")
        self.start_pct = start_pct
        self.decay_per_day = decay_per_day
        self.floor_pct = floor_pct

    def get_trailing_pct(self, direction, hold_days, confidence, entry_date, atr_lookup):
        trail = self.start_pct - self.decay_per_day * hold_days
        return max(trail, self.floor_pct)


class VolRegimeTrailing(TrailingStopStrategy):
    """Strategy 6: Wider trail in high-vol, tighter in low-vol."""
    def __init__(self, low_pct: float = 0.01, high_pct: float = 0.02,
                 median_atr: float = 0.0):
        super().__init__(f"VolRegime L={low_pct*100:.1f}% H={high_pct*100:.1f}%")
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.median_atr = median_atr

    def get_trailing_pct(self, direction, hold_days, confidence, entry_date, atr_lookup):
        atr_pct = atr_lookup.get(entry_date, self.median_atr)
        if atr_pct < self.median_atr:
            return self.low_pct
        else:
            return self.high_pct


class ConfidenceScaledTrailing(TrailingStopStrategy):
    """Strategy 7: High confidence = tighter trail."""
    def __init__(self, base_pct: float = 0.025):
        super().__init__(f"ConfScale base={base_pct*100:.1f}%")
        self.base_pct = base_pct

    def get_trailing_pct(self, direction, hold_days, confidence, entry_date, atr_lookup):
        # trail = base * (1 - confidence)
        # confidence ~0.45-1.0 => trail ~1.375%-0.0%
        trail = self.base_pct * (1.0 - confidence)
        return max(trail, 0.003)  # floor at 0.3%


# ---------------------------------------------------------------------------
# Simulation engine (modified for pluggable trailing stop)
# ---------------------------------------------------------------------------

def simulate_with_strategy(signals: List[DaySignals],
                           combo_fn: Callable,
                           strategy: TrailingStopStrategy,
                           atr_lookup: Dict[pd.Timestamp, float]) -> List[Trade]:
    """
    Run trade simulation with a pluggable trailing stop strategy.
    Matches combo_backtest.py logic except trailing stop is dynamic.
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

    for day_idx, day in enumerate(signals):
        if in_trade:
            hold_days += 1
            price_h = day.day_high
            price_l = day.day_low
            price_c = day.day_close

            # Get current trailing stop %
            trailing_pct = strategy.get_trailing_pct(
                direction, hold_days, confidence, entry_date, atr_lookup)

            # Update best price for trailing stop
            if direction == 'LONG':
                best_price = max(best_price, price_h)
                trailing_stop = best_price * (1.0 - trailing_pct)
                # Only activate trailing once profitable
                if best_price > entry_price:
                    effective_stop = max(stop_price, trailing_stop)
                else:
                    effective_stop = stop_price
                hit_stop = price_l <= effective_stop
                hit_tp = price_h >= tp_price
            else:  # SHORT
                best_price = min(best_price, price_l)
                trailing_stop = best_price * (1.0 + trailing_pct)
                if best_price < entry_price:
                    effective_stop = min(stop_price, trailing_stop)
                else:
                    effective_stop = stop_price
                hit_stop = price_h >= effective_stop
                hit_tp = price_l <= tp_price

            exit_reason = None
            exit_price = 0.0

            if hit_stop:
                is_trailing = (
                    (direction == 'LONG' and best_price > entry_price and trailing_stop > stop_price) or
                    (direction == 'SHORT' and best_price < entry_price and trailing_stop < stop_price)
                )
                exit_reason = 'trailing' if is_trailing else 'stop'
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
                ))
                in_trade = False
                cooldown_remaining = COOLDOWN_DAYS
                continue

        # Cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        # Check for new signal
        result = combo_fn(day)
        if result is None:
            continue
        action, conf, s_pct, t_pct, src = result
        if action is None or conf < MIN_SIGNAL_CONFIDENCE:
            continue

        # Entry at next-day open
        if day_idx + 1 >= len(signals):
            break
        next_day = signals[day_idx + 1]
        entry_price = next_day.day_open
        if entry_price <= 0:
            continue

        entry_date = next_day.date
        confidence = conf
        source = src
        stop_pct = s_pct
        tp_pct = t_pct

        # Position sizing
        position_value = CAPITAL * min(conf, 1.0)
        shares = max(1, int(position_value / entry_price))

        if action == 'BUY':
            direction = 'LONG'
            stop_price = entry_price * (1.0 - stop_pct)
            tp_price = entry_price * (1.0 + tp_pct)
            best_price = entry_price
        elif action == 'SELL':
            direction = 'SHORT'
            stop_price = entry_price * (1.0 + stop_pct)
            tp_price = entry_price * (1.0 - tp_pct)
            best_price = entry_price
        else:
            continue

        in_trade = True
        hold_days = 0

    # Close any open trade at end
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
        ))

    return trades


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_strategy(name: str, trades: List[Trade]):
    """Print detailed stats for one strategy."""
    n = len(trades)
    if n == 0:
        print(f"\n  {name}: No trades")
        return {}

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

    pnls = np.array([t.pnl for t in trades])
    avg_hold = np.mean([t.hold_days for t in trades])
    sharpe = (pnls.mean() / pnls.std() * np.sqrt(252 / max(avg_hold, 1))
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
    longs = [t for t in trades if t.direction == 'LONG']
    shorts = [t for t in trades if t.direction == 'SHORT']

    # Train/test
    train = [t for t in trades if t.entry_date.year <= TRAIN_END_YEAR]
    test = [t for t in trades if t.entry_date.year > TRAIN_END_YEAR]

    print(f"\n{'='*64}")
    print(f"  {name}")
    print(f"{'='*64}")
    print(f"  Trades: {n} | Wins: {len(wins)} ({len(wins)/n*100:.1f}%) | "
          f"Losses: {len(losses)} ({len(losses)/n*100:.1f}%)")
    print(f"  Total PnL: ${total_pnl:+,.0f} | Avg Win: ${avg_win:+,.0f} | Avg Loss: ${avg_loss:+,.0f}")
    print(f"  Sharpe: {sharpe:.2f} | Max DD: {max_dd_pct:.1f}% | Avg Hold: {avg_hold:.1f}d")
    print(f"  Exits: {reasons}")

    if longs:
        l_wr = sum(1 for t in longs if t.pnl > 0) / len(longs) * 100
        l_pnl = sum(t.pnl for t in longs)
        print(f"  Longs:  {len(longs)} trades, {l_wr:.0f}% WR, ${l_pnl:+,.0f}")
    if shorts:
        s_wr = sum(1 for t in shorts if t.pnl > 0) / len(shorts) * 100
        s_pnl = sum(t.pnl for t in shorts)
        print(f"  Shorts: {len(shorts)} trades, {s_wr:.0f}% WR, ${s_pnl:+,.0f}")

    for label, subset in [('Train (<=2021)', train), ('Test (>=2022)', test)]:
        if not subset:
            print(f"  --- {label}: 0 trades ---")
            continue
        sw = sum(1 for t in subset if t.pnl > 0)
        sp = sum(t.pnl for t in subset)
        sn = len(subset)
        print(f"  --- {label}: {sn} trades, {sw/sn*100:.0f}% WR, ${sp:+,.0f} ---")

    return {
        'name': name, 'trades': n, 'wins': len(wins), 'wr': len(wins)/n*100,
        'total_pnl': total_pnl, 'avg_win': avg_win, 'avg_loss': avg_loss,
        'sharpe': sharpe, 'max_dd_pct': max_dd_pct, 'avg_hold': avg_hold,
        'exits': reasons,
    }


def print_summary_table(results: List[dict]):
    """Print comparison table across all strategies."""
    print(f"\n{'='*100}")
    print(f"  ADAPTIVE TRAILING STOP — COMPARISON TABLE")
    print(f"{'='*100}")
    hdr = (f"{'Strategy':<38} {'Trades':>6} {'WR%':>6} {'PnL':>11} {'AvgWin':>9} "
           f"{'AvgLoss':>9} {'Sharpe':>7} {'MaxDD%':>7} {'AvgHold':>7}")
    print(hdr)
    print(f"{'-'*100}")

    for r in results:
        if not r:
            continue
        print(f"{r['name']:<38} {r['trades']:>6} {r['wr']:>5.1f}% ${r['total_pnl']:>+9,.0f} "
              f"${r['avg_win']:>+8,.0f} ${r['avg_loss']:>+8,.0f} {r['sharpe']:>7.2f} "
              f"{r['max_dd_pct']:>6.1f}% {r['avg_hold']:>6.1f}d")

    # Best by key metrics
    print(f"\n--- Best by Metric ---")
    if results:
        best_pnl = max(results, key=lambda r: r.get('total_pnl', -1e9))
        best_wr = max(results, key=lambda r: r.get('wr', 0))
        best_sharpe = max(results, key=lambda r: r.get('sharpe', -1e9))
        best_dd = min(results, key=lambda r: r.get('max_dd_pct', 1e9))
        print(f"  Best PnL:    {best_pnl['name']} (${best_pnl['total_pnl']:+,.0f})")
        print(f"  Best WR:     {best_wr['name']} ({best_wr['wr']:.1f}%)")
        print(f"  Best Sharpe: {best_sharpe['name']} ({best_sharpe['sharpe']:.2f})")
        print(f"  Lowest DD:   {best_dd['name']} ({best_dd['max_dd_pct']:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  ADAPTIVE TRAILING STOP STUDY")
    print("  Signal set: CS-ALL (BUY + SELL)")
    print("=" * 64)

    # Load cache
    if not CACHE_FILE.exists():
        print(f"\nERROR: Cache not found at {CACHE_FILE}")
        print("Run combo_backtest.py first to generate the cache.")
        sys.exit(1)

    print(f"\nLoading cache from {CACHE_FILE}...")
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)

    signals = cache['signals']
    daily_df = cache['daily_df']
    print(f"Loaded {len(signals):,} signal days, {len(daily_df):,} daily bars")

    # Sanity
    cs_buy = sum(1 for s in signals if s.cs_action == 'BUY')
    cs_sell = sum(1 for s in signals if s.cs_action == 'SELL')
    print(f"CS signals: {cs_buy} BUY, {cs_sell} SELL")

    # Compute ATR series
    print("\nComputing ATR(14) series...")
    atr_series = compute_atr_series(daily_df, period=14)
    atr_lookup = atr_series.to_dict()

    # Compute median ATR for vol-regime strategy
    valid_atr = atr_series.dropna()
    median_atr = float(valid_atr.median()) if len(valid_atr) > 0 else 0.02
    print(f"ATR(14) stats: median={median_atr*100:.2f}%, "
          f"mean={valid_atr.mean()*100:.2f}%, "
          f"min={valid_atr.min()*100:.2f}%, max={valid_atr.max()*100:.2f}%")

    # Define strategies
    strategies = [
        FixedTrailing(0.015),                                       # 1. Baseline
        ATRTrailing(1.0),                                           # 2. ATR x1.0
        ATRTrailing(1.5),                                           # 3. ATR x1.5
        AsymmetricTrailing(long_pct=0.02, short_pct=0.01),          # 4. Asymmetric
        TimeDecayTrailing(0.025, 0.0015, 0.005),                    # 5. Time-decay
        VolRegimeTrailing(0.01, 0.02, median_atr),                  # 6. Vol regime
        ConfidenceScaledTrailing(0.025),                             # 7. Confidence
    ]

    # Run each strategy
    all_results = []
    for strat in strategies:
        print(f"\nRunning: {strat.name}...")
        t0 = time.time()
        trades = simulate_with_strategy(signals, cs_all_combo, strat, atr_lookup)
        elapsed = time.time() - t0
        print(f"  {len(trades)} trades in {elapsed:.2f}s")
        result = report_strategy(strat.name, trades)
        all_results.append(result)

    # Summary comparison
    print_summary_table(all_results)

    print("\nDone.")


if __name__ == '__main__':
    main()
