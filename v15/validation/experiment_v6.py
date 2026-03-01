#!/usr/bin/env python3
"""
v6 Experiment Suite: Push WR and profit beyond v5 quadratic trail.

Experiments:
1. Loss forensics - Analyze what the ~6 remaining losses in X/Y have in common
2. Cubic/quartic trail formulas
3. Min hold period before trail activates
4. ATR-scaled trail (volatility-adaptive)
5. VIX-level scaled trail
6. Two-stage trail (tight then widen)
7. Asymmetric long/short parameters
8. Momentum persistence (consecutive days aligned)
9. Confidence threshold sweep
10. Position sizing experiments (Kelly criterion, etc.)
11. Combined multi-factor trail
"""

import pickle, sys, os, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.combo_backtest import (
    DaySignals, Trade, _apply_costs, _floor_stop_tp, _count_tf_confirming,
    _SigProxy, _AnalysisProxy, _build_filter_cascade,
    MIN_SIGNAL_CONFIDENCE, CAPITAL, DEFAULT_STOP_PCT, DEFAULT_TP_PCT,
    MAX_HOLD_DAYS, COOLDOWN_DAYS, SLIPPAGE_PCT, COMMISSION_PER_SHARE,
    TRAIN_END_YEAR, TRAILING_STOP_BASE,
)


# ===========================================================================
# Generalized simulator with configurable trail formula
# ===========================================================================

def simulate_v6(signals, combo_fn, trail_config):
    """
    Run trade sim with configurable trail parameters.

    trail_config dict keys:
    - formula: 'quadratic', 'cubic', 'quartic', 'exponential', 'adaptive'
    - base: float (default 0.025)
    - min_hold: int - minimum hold days before trail activates (default 0)
    - atr_scale: bool - scale trail by recent ATR (default False)
    - vix_scale: bool - scale trail by VIX level (default False)
    - vix_data: dict mapping date -> VIX close (needed if vix_scale)
    - atr_data: dict mapping date -> ATR% (needed if atr_scale)
    - two_stage: bool - use two-stage trail (default False)
    - stage2_profit_pct: float - profit threshold to switch to wider trail
    - stage2_multiplier: float - how much wider stage 2 trail is
    - min_trail: float - minimum trail pct (floor)
    - max_trail: float - maximum trail pct (ceiling)
    """
    formula = trail_config.get('formula', 'quadratic')
    base = trail_config.get('base', 0.025)
    min_hold = trail_config.get('min_hold', 0)
    atr_scale = trail_config.get('atr_scale', False)
    vix_scale = trail_config.get('vix_scale', False)
    vix_data = trail_config.get('vix_data', {})
    atr_data = trail_config.get('atr_data', {})
    two_stage = trail_config.get('two_stage', False)
    stage2_profit_pct = trail_config.get('stage2_profit_pct', 0.01)
    stage2_multiplier = trail_config.get('stage2_multiplier', 2.0)
    min_trail = trail_config.get('min_trail', 0.0001)  # 0.01%
    max_trail = trail_config.get('max_trail', 0.05)    # 5%

    trades = []
    in_trade = False
    cooldown_remaining = 0
    entry_date = entry_price = confidence = shares = 0
    stop_price = tp_price = best_price = hold_days = 0
    direction = source = ''
    trail_pct = base

    for day_idx, day in enumerate(signals):
        if in_trade:
            hold_days += 1
            price_h, price_l, price_c = day.day_high, day.day_low, day.day_close

            # Determine effective trail_pct (possibly two-stage)
            current_trail = trail_pct
            if two_stage and hold_days > min_hold:
                if direction == 'LONG':
                    profit_pct = (best_price - entry_price) / entry_price
                else:
                    profit_pct = (entry_price - best_price) / entry_price
                if profit_pct > stage2_profit_pct:
                    current_trail = trail_pct * stage2_multiplier

            # Only activate trailing after min_hold
            use_trail = hold_days > min_hold

            if direction == 'LONG':
                best_price = max(best_price, price_h)
                if use_trail:
                    trailing_stop = best_price * (1.0 - current_trail)
                    effective_stop = max(stop_price, trailing_stop) if best_price > entry_price else stop_price
                else:
                    trailing_stop = 0
                    effective_stop = stop_price
                hit_stop = price_l <= effective_stop
                hit_tp = price_h >= tp_price
            else:
                best_price = min(best_price, price_l)
                if use_trail:
                    trailing_stop = best_price * (1.0 + current_trail)
                    effective_stop = min(stop_price, trailing_stop) if best_price < entry_price else stop_price
                else:
                    trailing_stop = float('inf')
                    effective_stop = stop_price
                hit_stop = price_h >= effective_stop
                hit_tp = price_l <= tp_price

            exit_reason = exit_price = None
            if hit_stop:
                exit_reason = 'trailing' if use_trail and (
                    (direction == 'LONG' and best_price > entry_price and trailing_stop > stop_price) or
                    (direction == 'SHORT' and best_price < entry_price and trailing_stop < stop_price)
                ) else 'stop'
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
        c = 1.0 - conf
        if formula == 'linear':
            trail_pct = base * c
        elif formula == 'quadratic':
            trail_pct = base * c ** 2
        elif formula == 'cubic':
            trail_pct = base * c ** 3
        elif formula == 'quartic':
            trail_pct = base * c ** 4
        elif formula == 'exponential':
            trail_pct = base * np.exp(-3 * conf)  # steep decay with confidence
        elif formula == 'sigmoid':
            # S-curve: tight at high conf, wider at low
            trail_pct = base / (1 + np.exp(10 * (conf - 0.5)))
        elif formula == 'adaptive':
            # Base formula + ATR + VIX adjustments
            trail_pct = base * c ** 2
        else:
            trail_pct = base * c ** 2

        # ATR scaling
        if atr_scale and day.date in atr_data:
            atr_val = atr_data[day.date]
            # Scale trail relative to median ATR
            median_atr = trail_config.get('median_atr', 0.03)
            atr_ratio = atr_val / median_atr if median_atr > 0 else 1.0
            trail_pct *= atr_ratio

        # VIX scaling
        if vix_scale and day.date in vix_data:
            vix_val = vix_data[day.date]
            # Low VIX (<15) = tighter trail, High VIX (>30) = wider trail
            vix_ratio = vix_val / 20.0  # normalize to VIX=20 baseline
            trail_pct *= max(0.5, min(vix_ratio, 2.0))

        # Clamp trail
        trail_pct = max(min_trail, min(trail_pct, max_trail))

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


def summarize(trades, label, verbose=False):
    n = len(trades)
    if n == 0:
        print(f"{label:<45} {'0':>4} trades")
        return {}
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    total = sum(t.pnl for t in trades)
    pnls = np.array([t.pnl for t in trades])
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    mdd = float(dd.max()) / CAPITAL * 100
    big_l = min(t.pnl for t in trades)

    train = [t for t in trades if t.entry_date.year <= TRAIN_END_YEAR]
    test = [t for t in trades if t.entry_date.year > TRAIN_END_YEAR]
    t_wr = sum(1 for t in train if t.pnl > 0) / len(train) * 100 if train else 0
    s_wr = sum(1 for t in test if t.pnl > 0) / len(test) * 100 if test else 0

    print(f"{label:<45} {n:>4} {wr:>5.1f}% ${total:>+9,.0f}  "
          f"Sh={sharpe:>5.1f}  DD={mdd:>4.1f}%  "
          f"Tr={t_wr:.0f}% Ts={s_wr:.0f}%  BL=${big_l:>+7,.0f}")

    if verbose:
        losses = [t for t in trades if t.pnl <= 0]
        for t in losses:
            print(f"  LOSS: {t.entry_date.date()} {t.direction} "
                  f"${t.entry_price:.1f}->${t.exit_price:.1f} "
                  f"PnL=${t.pnl:+,.0f} conf={t.confidence:.2f} "
                  f"hold={t.hold_days}d exit={t.exit_reason} [{t.source}]")

    return {
        'n': n, 'wr': wr, 'pnl': total, 'sharpe': sharpe,
        'mdd': mdd, 'big_l': big_l, 'train_wr': t_wr, 'test_wr': s_wr,
    }


# ===========================================================================
# Experiment functions
# ===========================================================================

def exp_loss_forensics(signals, combo_fn, label="X: TF4+VIX"):
    """Deep analysis of remaining losses."""
    print("\n" + "=" * 100)
    print(f"  EXPERIMENT 1: LOSS FORENSICS ({label})")
    print("=" * 100)

    # Run with v5 settings first
    trades = simulate_v6(signals, combo_fn, {'formula': 'quadratic', 'base': 0.025})
    losses = [t for t in trades if t.pnl <= 0]
    wins = [t for t in trades if t.pnl > 0]

    print(f"\nTotal: {len(trades)} trades, {len(losses)} losses:")
    for t in losses:
        # Find the signal day (day before entry)
        sig_day = None
        for s in signals:
            if s.date == t.entry_date:
                sig_day = s
                break
        tf_count = _count_tf_confirming(sig_day, t.direction.replace('LONG', 'BUY').replace('SHORT', 'SELL')) if sig_day else '?'
        print(f"  {t.entry_date.date()} {t.direction:>5} ${t.entry_price:>7.1f}->${t.exit_price:>7.1f} "
              f"PnL=${t.pnl:>+8,.0f} conf={t.confidence:.3f} hold={t.hold_days}d "
              f"exit={t.exit_reason} TFs={tf_count} [{t.source}]")

    # Analyze patterns in losses vs wins
    print(f"\n  Loss patterns:")
    loss_confs = [t.confidence for t in losses]
    win_confs = [t.confidence for t in wins]
    print(f"    Loss conf range: {min(loss_confs):.3f} - {max(loss_confs):.3f} (mean {np.mean(loss_confs):.3f})")
    print(f"    Win conf range:  {min(win_confs):.3f} - {max(win_confs):.3f} (mean {np.mean(win_confs):.3f})")

    loss_holds = [t.hold_days for t in losses]
    win_holds = [t.hold_days for t in wins]
    print(f"    Loss hold days: {min(loss_holds)}-{max(loss_holds)} (mean {np.mean(loss_holds):.1f})")
    print(f"    Win hold days:  {min(win_holds)}-{max(win_holds)} (mean {np.mean(win_holds):.1f})")

    loss_dirs = defaultdict(int)
    for t in losses:
        loss_dirs[t.direction] += 1
    win_dirs = defaultdict(int)
    for t in wins:
        win_dirs[t.direction] += 1
    print(f"    Loss direction: {dict(loss_dirs)}")
    print(f"    Win direction:  {dict(win_dirs)}")

    loss_reasons = defaultdict(int)
    for t in losses:
        loss_reasons[t.exit_reason] += 1
    print(f"    Loss exit reasons: {dict(loss_reasons)}")

    # Check if losses cluster in time
    loss_years = defaultdict(int)
    for t in losses:
        loss_years[t.entry_date.year] += 1
    print(f"    Loss by year: {dict(sorted(loss_years.items()))}")

    loss_months = defaultdict(int)
    for t in losses:
        loss_months[t.entry_date.month] += 1
    print(f"    Loss by month: {dict(sorted(loss_months.items()))}")

    # Day of week analysis
    loss_dow = defaultdict(int)
    for t in losses:
        loss_dow[t.entry_date.day_name()] += 1
    win_dow = defaultdict(int)
    for t in wins:
        win_dow[t.entry_date.day_name()] += 1
    print(f"    Loss by day-of-week: {dict(loss_dow)}")
    print(f"    Win by day-of-week: {dict(win_dow)}")

    return losses


def exp_trail_formulas(signals, combo_fn):
    """Test different trail formulas."""
    print("\n" + "=" * 100)
    print("  EXPERIMENT 2: TRAIL FORMULA SWEEP")
    print("=" * 100)

    configs = [
        ('quadratic (v5 baseline)', {'formula': 'quadratic', 'base': 0.025}),
        ('cubic', {'formula': 'cubic', 'base': 0.025}),
        ('cubic b=0.030', {'formula': 'cubic', 'base': 0.030}),
        ('cubic b=0.035', {'formula': 'cubic', 'base': 0.035}),
        ('cubic b=0.040', {'formula': 'cubic', 'base': 0.040}),
        ('quartic', {'formula': 'quartic', 'base': 0.025}),
        ('quartic b=0.035', {'formula': 'quartic', 'base': 0.035}),
        ('quartic b=0.050', {'formula': 'quartic', 'base': 0.050}),
        ('exponential', {'formula': 'exponential', 'base': 0.025}),
        ('sigmoid', {'formula': 'sigmoid', 'base': 0.025}),
        ('sigmoid b=0.035', {'formula': 'sigmoid', 'base': 0.035}),
    ]

    results = {}
    for label, cfg in configs:
        trades = simulate_v6(signals, combo_fn, cfg)
        r = summarize(trades, f"  {label}")
        results[label] = r

    return results


def exp_min_hold(signals, combo_fn):
    """Test minimum hold period before trail activates."""
    print("\n" + "=" * 100)
    print("  EXPERIMENT 3: MINIMUM HOLD PERIOD")
    print("=" * 100)

    results = {}
    for min_h in [0, 1, 2, 3, 4, 5]:
        cfg = {'formula': 'quadratic', 'base': 0.025, 'min_hold': min_h}
        trades = simulate_v6(signals, combo_fn, cfg)
        r = summarize(trades, f"  min_hold={min_h}")
        results[min_h] = r

    # Also test with cubic
    print("\n  With cubic formula:")
    for min_h in [0, 1, 2, 3]:
        cfg = {'formula': 'cubic', 'base': 0.035, 'min_hold': min_h}
        trades = simulate_v6(signals, combo_fn, cfg)
        r = summarize(trades, f"  cubic b=0.035 min_hold={min_h}")
        results[f'cubic_{min_h}'] = r

    return results


def exp_two_stage_trail(signals, combo_fn):
    """Test two-stage trailing stop."""
    print("\n" + "=" * 100)
    print("  EXPERIMENT 4: TWO-STAGE TRAIL")
    print("=" * 100)

    results = {}
    # Idea: start with tight trail, then widen after reaching X% profit
    # This lets winners run further
    configs = [
        ('baseline (no 2-stage)', {'formula': 'quadratic', 'base': 0.025}),
        ('2stg: 1% -> 2x', {'formula': 'quadratic', 'base': 0.025,
                             'two_stage': True, 'stage2_profit_pct': 0.01, 'stage2_multiplier': 2.0}),
        ('2stg: 1% -> 3x', {'formula': 'quadratic', 'base': 0.025,
                             'two_stage': True, 'stage2_profit_pct': 0.01, 'stage2_multiplier': 3.0}),
        ('2stg: 2% -> 2x', {'formula': 'quadratic', 'base': 0.025,
                             'two_stage': True, 'stage2_profit_pct': 0.02, 'stage2_multiplier': 2.0}),
        ('2stg: 0.5% -> 1.5x', {'formula': 'quadratic', 'base': 0.025,
                                  'two_stage': True, 'stage2_profit_pct': 0.005, 'stage2_multiplier': 1.5}),
    ]

    for label, cfg in configs:
        trades = simulate_v6(signals, combo_fn, cfg)
        r = summarize(trades, f"  {label}")
        results[label] = r

    return results


def exp_atr_scaling(signals, combo_fn, daily_df):
    """Scale trail by ATR (volatility-adaptive)."""
    print("\n" + "=" * 100)
    print("  EXPERIMENT 5: ATR-SCALED TRAIL")
    print("=" * 100)

    # Pre-compute ATR% for each date
    from v15.features.utils import calc_atr
    highs = daily_df['high'].values.astype(float)
    lows = daily_df['low'].values.astype(float)
    closes = daily_df['close'].values.astype(float)
    atr_arr = calc_atr(highs, lows, closes, period=14)
    atr_pct = atr_arr / closes

    atr_data = {}
    for i, date in enumerate(daily_df.index):
        if i < len(atr_pct) and not np.isnan(atr_pct[i]):
            atr_data[date] = float(atr_pct[i])

    median_atr = float(np.nanmedian(atr_pct))
    print(f"  Median ATR%: {median_atr:.4f} ({median_atr*100:.2f}%)")

    results = {}
    configs = [
        ('baseline (no ATR)', {'formula': 'quadratic', 'base': 0.025}),
        ('ATR-scaled quad', {'formula': 'quadratic', 'base': 0.025,
                             'atr_scale': True, 'atr_data': atr_data, 'median_atr': median_atr}),
        ('ATR-scaled cubic b=0.035', {'formula': 'cubic', 'base': 0.035,
                                       'atr_scale': True, 'atr_data': atr_data, 'median_atr': median_atr}),
    ]

    for label, cfg in configs:
        trades = simulate_v6(signals, combo_fn, cfg)
        r = summarize(trades, f"  {label}")
        results[label] = r

    return results


def exp_vix_scaling(signals, combo_fn, vix_daily):
    """Scale trail by VIX level."""
    print("\n" + "=" * 100)
    print("  EXPERIMENT 6: VIX-LEVEL SCALED TRAIL")
    print("=" * 100)

    if vix_daily is None:
        print("  VIX data not available, skipping")
        return {}

    vix_data = {}
    for date, row in vix_daily.iterrows():
        vix_data[date] = float(row['close'])

    results = {}
    configs = [
        ('baseline', {'formula': 'quadratic', 'base': 0.025}),
        ('VIX-scaled quad', {'formula': 'quadratic', 'base': 0.025,
                              'vix_scale': True, 'vix_data': vix_data}),
        ('VIX-scaled quad b=0.020', {'formula': 'quadratic', 'base': 0.020,
                                      'vix_scale': True, 'vix_data': vix_data}),
        ('VIX-scaled cubic b=0.035', {'formula': 'cubic', 'base': 0.035,
                                       'vix_scale': True, 'vix_data': vix_data}),
    ]

    for label, cfg in configs:
        trades = simulate_v6(signals, combo_fn, cfg)
        r = summarize(trades, f"  {label}")
        results[label] = r

    return results


def exp_confidence_threshold(signals, combo_fn_maker):
    """Sweep minimum confidence threshold."""
    print("\n" + "=" * 100)
    print("  EXPERIMENT 7: CONFIDENCE THRESHOLD SWEEP")
    print("=" * 100)

    results = {}
    for min_conf in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        # Rebuild combo with new threshold
        fn = combo_fn_maker(min_conf)
        trades = simulate_v6(signals, fn, {'formula': 'quadratic', 'base': 0.025})
        r = summarize(trades, f"  min_conf={min_conf:.2f}")
        results[min_conf] = r

    return results


def exp_asymmetric_params(signals, combo_fn):
    """Test different parameters for longs vs shorts."""
    print("\n" + "=" * 100)
    print("  EXPERIMENT 8: ASYMMETRIC LONG/SHORT TRAIL")
    print("=" * 100)

    # Run long-only and short-only variants
    def make_long_only(fn):
        def wrapper(day):
            r = fn(day)
            if r and r[0] == 'SELL':
                return None
            return r
        return wrapper

    def make_short_only(fn):
        def wrapper(day):
            r = fn(day)
            if r and r[0] == 'BUY':
                return None
            return r
        return wrapper

    results = {}

    # Analyze longs and shorts separately with different formulas
    for formula in ['quadratic', 'cubic']:
        for base in [0.020, 0.025, 0.030, 0.035]:
            cfg = {'formula': formula, 'base': base}

            # Long only
            trades_l = simulate_v6(signals, make_long_only(combo_fn), cfg)
            r = summarize(trades_l, f"  LONG {formula} b={base}")
            results[f'long_{formula}_{base}'] = r

            # Short only
            trades_s = simulate_v6(signals, make_short_only(combo_fn), cfg)
            r = summarize(trades_s, f"  SHORT {formula} b={base}")
            results[f'short_{formula}_{base}'] = r

    return results


def exp_momentum_persistence(signals):
    """Require multiple consecutive days of TF alignment."""
    print("\n" + "=" * 100)
    print("  EXPERIMENT 9: MOMENTUM PERSISTENCE FILTER")
    print("=" * 100)

    # Track how many consecutive days each TF has been aligned
    tf_align_streak = {}  # {tf: streak_count}

    def make_persistence_combo(min_streak=2, min_tfs=4):
        """Require TFs to have been aligned for min_streak consecutive days."""
        prev_tf_states = {}
        streaks = defaultdict(int)

        def fn(day: DaySignals):
            nonlocal prev_tf_states, streaks

            if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
                # Update streaks even on non-signal days
                if day.cs_tf_states:
                    for tf, state in day.cs_tf_states.items():
                        if not state.get('valid', False):
                            streaks[tf] = 0
                            continue
                        md = state.get('momentum_direction', 0.0)
                        prev_md = prev_tf_states.get(tf, 0.0)
                        if (md > 0 and prev_md > 0) or (md < 0 and prev_md < 0):
                            streaks[tf] += 1
                        else:
                            streaks[tf] = 1 if md != 0 else 0
                        prev_tf_states[tf] = md
                return None

            action = day.cs_action

            # Update streaks
            if day.cs_tf_states:
                for tf, state in day.cs_tf_states.items():
                    if not state.get('valid', False):
                        streaks[tf] = 0
                        continue
                    md = state.get('momentum_direction', 0.0)
                    prev_md = prev_tf_states.get(tf, 0.0)
                    if (md > 0 and prev_md > 0) or (md < 0 and prev_md < 0):
                        streaks[tf] += 1
                    else:
                        streaks[tf] = 1 if md != 0 else 0
                    prev_tf_states[tf] = md

            # Count TFs with sufficient streak
            confirmed = 0
            if day.cs_tf_states:
                for tf, state in day.cs_tf_states.items():
                    if not state.get('valid', False):
                        continue
                    md = state.get('momentum_direction', 0.0)
                    aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                    if aligned and streaks.get(tf, 0) >= min_streak:
                        confirmed += 1

            if confirmed < min_tfs:
                return None

            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return (day.cs_action, day.cs_confidence, s, t, 'CS')
        return fn

    results = {}
    for streak in [1, 2, 3, 4, 5]:
        for min_tfs in [3, 4]:
            fn = make_persistence_combo(min_streak=streak, min_tfs=min_tfs)
            trades = simulate_v6(signals, fn, {'formula': 'quadratic', 'base': 0.025})
            r = summarize(trades, f"  streak>={streak} TF>={min_tfs}")
            results[f's{streak}_tf{min_tfs}'] = r

    return results


def exp_stop_tp_sweep(signals, combo_fn):
    """Sweep stop loss and take profit percentages."""
    print("\n" + "=" * 100)
    print("  EXPERIMENT 10: STOP/TP PARAMETER SWEEP")
    print("=" * 100)

    # Override stop/tp in the combo function
    def make_override_fn(fn, new_stop, new_tp):
        def wrapper(day):
            r = fn(day)
            if r is None:
                return None
            action, conf, _, _, src = r
            return (action, conf, new_stop, new_tp, src)
        return wrapper

    results = {}
    for stop in [0.015, 0.020, 0.025, 0.030]:
        for tp in [0.03, 0.04, 0.05, 0.06, 0.08]:
            fn = make_override_fn(combo_fn, stop, tp)
            trades = simulate_v6(signals, fn, {'formula': 'quadratic', 'base': 0.025})
            r = summarize(trades, f"  stop={stop*100:.1f}% tp={tp*100:.0f}%")
            results[f's{stop}_t{tp}'] = r

    return results


def exp_combined_best(signals, combo_fn, daily_df, vix_daily):
    """Combine the best discoveries from all experiments."""
    print("\n" + "=" * 100)
    print("  EXPERIMENT 11: COMBINED BEST CONFIGURATIONS")
    print("=" * 100)

    # Pre-compute ATR data
    from v15.features.utils import calc_atr
    highs = daily_df['high'].values.astype(float)
    lows = daily_df['low'].values.astype(float)
    closes = daily_df['close'].values.astype(float)
    atr_arr = calc_atr(highs, lows, closes, period=14)
    atr_pct = atr_arr / closes
    atr_data = {}
    for i, date in enumerate(daily_df.index):
        if i < len(atr_pct) and not np.isnan(atr_pct[i]):
            atr_data[date] = float(atr_pct[i])
    median_atr = float(np.nanmedian(atr_pct))

    vix_data = {}
    if vix_daily is not None:
        for date, row in vix_daily.iterrows():
            vix_data[date] = float(row['close'])

    configs = [
        ('v5 baseline', {'formula': 'quadratic', 'base': 0.025}),
        ('cubic + ATR', {'formula': 'cubic', 'base': 0.035,
                          'atr_scale': True, 'atr_data': atr_data, 'median_atr': median_atr}),
        ('quad + min_hold=1', {'formula': 'quadratic', 'base': 0.025, 'min_hold': 1}),
        ('cubic + min_hold=1', {'formula': 'cubic', 'base': 0.035, 'min_hold': 1}),
        ('quad + 2stage', {'formula': 'quadratic', 'base': 0.025,
                           'two_stage': True, 'stage2_profit_pct': 0.01, 'stage2_multiplier': 2.0}),
        ('cubic + ATR + min1', {'formula': 'cubic', 'base': 0.035,
                                 'atr_scale': True, 'atr_data': atr_data, 'median_atr': median_atr,
                                 'min_hold': 1}),
        ('cubic + VIX + ATR', {'formula': 'cubic', 'base': 0.035,
                                'atr_scale': True, 'atr_data': atr_data, 'median_atr': median_atr,
                                'vix_scale': True, 'vix_data': vix_data}),
        ('quartic + ATR', {'formula': 'quartic', 'base': 0.050,
                            'atr_scale': True, 'atr_data': atr_data, 'median_atr': median_atr}),
    ]

    results = {}
    best_wr = 0
    best_label = ''
    for label, cfg in configs:
        trades = simulate_v6(signals, combo_fn, cfg)
        r = summarize(trades, f"  {label}", verbose=(label != 'v5 baseline'))
        results[label] = r
        if r and r.get('wr', 0) > best_wr and r.get('n', 0) >= 50:
            best_wr = r['wr']
            best_label = label

    if best_label:
        print(f"\n  BEST: {best_label} at {best_wr:.1f}% WR")

    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    cache_path = Path(__file__).parent / 'combo_cache' / 'combo_signals.pkl'
    print(f"Loading from {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    signals = data['signals']
    daily_df = data['daily_df']
    vix_daily = data.get('vix_daily')
    print(f"  {len(signals)} days, {signals[0].date.date()} to {signals[-1].date.date()}\n")

    # Build VIX cascade for X: TF4+VIX combo
    cascade_vix = _build_filter_cascade(vix=True)
    if vix_daily is not None:
        cascade_vix.precompute_vix_cooldown(vix_daily)

    # Build the X: TF4+VIX combo function (our best CS-only combo)
    def make_tf4_vix(day):
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

    # Also build Y: TF4+VIX+V5 for some experiments
    def make_tf4_vix_v5(day):
        action = None
        conf = 0.0
        stop = DEFAULT_STOP_PCT
        tp = DEFAULT_TP_PCT
        src = 'CS'
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) >= 4:
                action = day.cs_action
                conf = day.cs_confidence
                stop, tp = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        if day.v5_take_bounce and day.v5_confidence >= 0.56:
            if action == 'BUY':
                conf = max(conf, day.v5_confidence)
                src = 'CS+V5'
            elif action is None:
                action = 'BUY'
                conf = day.v5_confidence
                src = 'V5'
        if action is None or conf < MIN_SIGNAL_CONFIDENCE:
            return None
        sig = _SigProxy(day, action=action, conf=conf)
        ana = _AnalysisProxy(day.cs_tf_states)
        ok, adj, _ = cascade_vix.evaluate(sig, ana, feature_vec=None,
                                          bar_datetime=day.date,
                                          higher_tf_data=None, spy_df=None, vix_df=None)
        if not ok or adj < MIN_SIGNAL_CONFIDENCE:
            return None
        return (action, adj, stop, tp, src)

    # Run all experiments
    all_results = {}

    # 1. Loss forensics
    losses = exp_loss_forensics(signals, make_tf4_vix, "X: TF4+VIX")
    print("\n  Also Y: TF4+VIX+V5:")
    exp_loss_forensics(signals, make_tf4_vix_v5, "Y: TF4+VIX+V5")

    # 2. Trail formulas
    print("\n  --- On X: TF4+VIX ---")
    r = exp_trail_formulas(signals, make_tf4_vix)
    all_results['trail_formulas_X'] = r

    print("\n  --- On Y: TF4+VIX+V5 ---")
    r = exp_trail_formulas(signals, make_tf4_vix_v5)
    all_results['trail_formulas_Y'] = r

    # 3. Min hold period
    r = exp_min_hold(signals, make_tf4_vix)
    all_results['min_hold'] = r

    # 4. Two-stage trail
    r = exp_two_stage_trail(signals, make_tf4_vix)
    all_results['two_stage'] = r

    # 5. ATR scaling
    r = exp_atr_scaling(signals, make_tf4_vix, daily_df)
    all_results['atr_scaling'] = r

    # 6. VIX scaling
    r = exp_vix_scaling(signals, make_tf4_vix, vix_daily)
    all_results['vix_scaling'] = r

    # 7. Confidence threshold
    def make_conf_combo(min_conf):
        def fn(day):
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= min_conf:
                if _count_tf_confirming(day, day.cs_action) < 4:
                    return None
                sig = _SigProxy(day)
                ana = _AnalysisProxy(day.cs_tf_states)
                ok, adj, _ = cascade_vix.evaluate(sig, ana, feature_vec=None,
                                                  bar_datetime=day.date,
                                                  higher_tf_data=None, spy_df=None, vix_df=None)
                if not ok or adj < min_conf:
                    return None
                s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                return (day.cs_action, adj, s, t, 'CS')
            return None
        return fn

    r = exp_confidence_threshold(signals, make_conf_combo)
    all_results['conf_threshold'] = r

    # 8. Asymmetric long/short
    r = exp_asymmetric_params(signals, make_tf4_vix)
    all_results['asymmetric'] = r

    # 9. Momentum persistence
    r = exp_momentum_persistence(signals)
    all_results['persistence'] = r

    # 10. Stop/TP sweep
    r = exp_stop_tp_sweep(signals, make_tf4_vix)
    all_results['stop_tp'] = r

    # 11. Combined best
    r = exp_combined_best(signals, make_tf4_vix, daily_df, vix_daily)
    all_results['combined'] = r

    # Also run combined on Y
    print("\n  --- Combined on Y: TF4+VIX+V5 ---")
    r = exp_combined_best(signals, make_tf4_vix_v5, daily_df, vix_daily)
    all_results['combined_Y'] = r

    print("\n" + "=" * 100)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
