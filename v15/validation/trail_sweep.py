#!/usr/bin/env python3
"""Sweep trailing stop parameters on TF4+VIX combo to find optimal formula."""

import pickle, sys, os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.combo_backtest import (
    DaySignals, Trade, _apply_costs, _floor_stop_tp, _count_tf_confirming,
    _SigProxy, _AnalysisProxy, _build_filter_cascade,
    MIN_SIGNAL_CONFIDENCE, CAPITAL, DEFAULT_STOP_PCT, DEFAULT_TP_PCT,
    MAX_HOLD_DAYS, COOLDOWN_DAYS, SLIPPAGE_PCT, COMMISSION_PER_SHARE,
    TRAIN_END_YEAR,
)


def simulate_with_trail(signals, combo_fn, trail_base, trail_formula='linear'):
    """Run trade sim with configurable trail formula.

    Formulas:
    - linear: trail = base * (1 - conf)
    - quadratic: trail = base * (1 - conf)^2
    - sqrt: trail = base * sqrt(1 - conf)
    - flat: trail = base (fixed, no confidence scaling)
    """
    trades = []
    in_trade = False
    cooldown_remaining = 0
    entry_date = entry_price = confidence = shares = 0
    stop_price = tp_price = best_price = hold_days = 0
    direction = source = ''
    trail_pct = trail_base

    for day_idx, day in enumerate(signals):
        if in_trade:
            hold_days += 1
            price_h, price_l, price_c = day.day_high, day.day_low, day.day_close

            if direction == 'LONG':
                best_price = max(best_price, price_h)
                trailing_stop = best_price * (1.0 - trail_pct)
                effective_stop = max(stop_price, trailing_stop) if best_price > entry_price else stop_price
                hit_stop = price_l <= effective_stop
                hit_tp = price_h >= tp_price
            else:
                best_price = min(best_price, price_l)
                trailing_stop = best_price * (1.0 + trail_pct)
                effective_stop = min(stop_price, trailing_stop) if best_price < entry_price else stop_price
                hit_stop = price_h >= effective_stop
                hit_tp = price_l <= tp_price

            exit_reason = exit_price = None
            if hit_stop:
                exit_reason = 'trailing' if (direction == 'LONG' and best_price > entry_price and trailing_stop > stop_price) or \
                                            (direction == 'SHORT' and best_price < entry_price and trailing_stop < stop_price) else 'stop'
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
                    entry_date=entry_date, exit_date=day.date, direction=direction,
                    entry_price=entry_price, exit_price=exit_price, confidence=confidence,
                    shares=shares, pnl=pnl, hold_days=hold_days, exit_reason=exit_reason,
                    source=source,
                ))
                in_trade = False
                cooldown_remaining = COOLDOWN_DAYS
                continue

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        result = combo_fn(day)
        if result is None:
            continue
        action, conf, s_pct, t_pct, src = result
        if action is None or conf < MIN_SIGNAL_CONFIDENCE:
            continue

        if day_idx + 1 >= len(signals):
            break
        next_day = signals[day_idx + 1]
        entry_price = next_day.day_open
        if entry_price <= 0:
            continue

        entry_date = next_day.date
        confidence = conf
        source = src

        # Compute trail_pct based on formula
        if trail_formula == 'linear':
            trail_pct = trail_base * (1.0 - conf)
        elif trail_formula == 'quadratic':
            trail_pct = trail_base * (1.0 - conf) ** 2
        elif trail_formula == 'sqrt':
            trail_pct = trail_base * np.sqrt(1.0 - conf)
        elif trail_formula == 'flat':
            trail_pct = trail_base
        else:
            trail_pct = trail_base * (1.0 - conf)

        position_value = CAPITAL * min(conf, 1.0)
        shares = max(1, int(position_value / entry_price))

        if action == 'BUY':
            direction = 'LONG'
            stop_price = entry_price * (1.0 - s_pct)
            tp_price = entry_price * (1.0 + t_pct)
            best_price = entry_price
        elif action == 'SELL':
            direction = 'SHORT'
            stop_price = entry_price * (1.0 + s_pct)
            tp_price = entry_price * (1.0 - t_pct)
            best_price = entry_price
        else:
            continue

        in_trade = True
        hold_days = 0

    return trades


def make_tf4_vix_combo(cascade_vix):
    """TF4 + VIX cooldown."""
    def fn(day):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) < 4:
                return None
            sig = _SigProxy(day)
            ana = _AnalysisProxy(day.cs_tf_states)
            ok, adj, _ = cascade_vix.evaluate(sig, ana, feature_vec=None,
                                              bar_datetime=day.date,
                                              higher_tf_data=None, spy_df=None, vix_df=None)
            if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                return None
            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return (day.cs_action, adj, s, t, 'CS')
        return None
    return fn


def summarize(trades, label):
    n = len(trades)
    if n == 0:
        return
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    total = sum(t.pnl for t in trades)
    pnls = np.array([t.pnl for t in trades])
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    mdd = float(dd.max()) / CAPITAL * 100

    train = [t for t in trades if t.entry_date.year <= TRAIN_END_YEAR]
    test = [t for t in trades if t.entry_date.year > TRAIN_END_YEAR]
    t_wr = sum(1 for t in train if t.pnl > 0) / len(train) * 100 if train else 0
    s_wr = sum(1 for t in test if t.pnl > 0) / len(test) * 100 if test else 0
    big_l = min(t.pnl for t in trades)

    print(f"{label:<35} {n:>4} {wr:>5.1f}% ${total:>+9,.0f}  "
          f"Sh={sharpe:>5.1f}  DD={mdd:>4.1f}%  "
          f"Train={t_wr:.0f}% Test={s_wr:.0f}%  BigL=${big_l:>+7,.0f}")


def main():
    # Load cache
    cache_dir = Path(__file__).parent / 'combo_cache'
    cache_path = cache_dir / 'combo_signals.pkl'
    print(f"Loading from {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    signals = data['signals']
    print(f"  {len(signals)} days\n")

    # Build VIX cascade - load from cache or fetch
    vix_daily = data.get('vix_daily')
    if vix_daily is None:
        from v15.data.native_tf import fetch_native_tf
        vix_daily = fetch_native_tf('^VIX', 'daily', '2016-01-01', '2025-12-31')
        vix_daily.columns = [c.lower() for c in vix_daily.columns]
        vix_daily.index = pd.to_datetime(vix_daily.index).tz_localize(None)
    cascade_vix = _build_filter_cascade(vix=True)
    cascade_vix.precompute_vix_cooldown(vix_daily)

    combo_fn = make_tf4_vix_combo(cascade_vix)

    print("=" * 100)
    print("  TRAILING STOP PARAMETER SWEEP (TF4+VIX combo)")
    print("=" * 100)

    # Sweep trail_base with linear formula
    print("\n--- Linear: trail = base * (1 - conf) ---")
    for base in [0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050]:
        trades = simulate_with_trail(signals, combo_fn, base, 'linear')
        summarize(trades, f"  base={base:.3f} linear")

    # Sweep formulas at base=0.025
    print("\n--- Formula variants at base=0.025 ---")
    for formula in ['linear', 'quadratic', 'sqrt', 'flat']:
        trades = simulate_with_trail(signals, combo_fn, 0.025, formula)
        summarize(trades, f"  0.025 {formula}")

    # Sweep formulas at base=0.030
    print("\n--- Formula variants at base=0.030 ---")
    for formula in ['linear', 'quadratic', 'sqrt', 'flat']:
        trades = simulate_with_trail(signals, combo_fn, 0.030, formula)
        summarize(trades, f"  0.030 {formula}")

    # Sweep formulas at base=0.035
    print("\n--- Formula variants at base=0.035 ---")
    for formula in ['linear', 'quadratic', 'sqrt', 'flat']:
        trades = simulate_with_trail(signals, combo_fn, 0.035, formula)
        summarize(trades, f"  0.035 {formula}")

    print("\nDone.")


if __name__ == '__main__':
    main()
