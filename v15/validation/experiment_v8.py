#!/usr/bin/env python3
"""
V8 Experiments: Cooldown optimization, s1_tf3 combos, gap/range/DOW filters,
rolling Sharpe regime, asymmetric trail fix, confidence x TF interaction.

Key findings from v7 to build on:
  - Cooldown=0 boosts X from 137→165 trades at 98.2% WR, $331K
  - s1_tf3 (streak≥1, 3+ TFs) = 210 trades, 97.1%, $352K, max loss only -$98
  - Asymmetric trail was BUGGY (all zeros) — fix and retest
  - TP/max_hold don't matter (trail exits everything)
"""

import pickle, sys, os, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming,
    _SigProxy, _AnalysisProxy, _build_filter_cascade, _floor_stop_tp,
    MIN_SIGNAL_CONFIDENCE, CAPITAL, DEFAULT_STOP_PCT, DEFAULT_TP_PCT,
    TRAILING_STOP_BASE, MAX_HOLD_DAYS, SLIPPAGE_PCT, COMMISSION_PER_SHARE,
    simulate_trades, report_combo,
)


# ---------------------------------------------------------------------------
# Custom simulator with configurable cooldown & trail
# ---------------------------------------------------------------------------

def simulate_custom(signals, combo_fn, name, cooldown=2, trail_fn=None,
                    max_hold=MAX_HOLD_DAYS):
    """simulate_trades clone with configurable cooldown and trail function.

    trail_fn(conf, direction) -> trail_pct
    If trail_fn is None, uses default quartic.
    """
    from v15.validation.combo_backtest import _apply_costs

    trades = []
    in_trade = False
    cooldown_remaining = 0

    for day_idx, day in enumerate(signals):
        if in_trade:
            hold_days += 1
            price_h = day.day_high
            price_l = day.day_low
            price_c = day.day_close

            # Trailing stop logic matching combo_backtest.py exactly
            if direction == 'LONG':
                best_price = max(best_price, price_h)
                trailing_stop = best_price * (1.0 - trail_pct)
                # Only activate trailing once profitable
                if best_price > entry_price:
                    effective_stop = max(stop_price, trailing_stop)
                else:
                    effective_stop = stop_price
                hit_stop = price_l <= effective_stop
                hit_tp = price_h >= tp_price
            else:  # SHORT
                best_price = min(best_price, price_l)
                trailing_stop = best_price * (1.0 + trail_pct)
                if best_price < entry_price:
                    effective_stop = min(stop_price, trailing_stop)
                else:
                    effective_stop = stop_price
                hit_stop = price_h >= effective_stop
                hit_tp = price_l <= tp_price

            exit_reason = None
            exit_price = 0.0

            if hit_stop:
                exit_reason = ('trailing' if (direction == 'LONG' and best_price > entry_price and trailing_stop > stop_price) or
                               (direction == 'SHORT' and best_price < entry_price and trailing_stop < stop_price) else 'stop')
                exit_price = effective_stop
            elif hit_tp:
                exit_reason = 'tp'
                exit_price = tp_price
            elif hold_days >= max_hold:
                exit_reason = 'timeout'
                exit_price = price_c

            if exit_reason is None:
                continue

            pnl = _apply_costs(entry_price, exit_price, shares, direction)
            trades.append(Trade(
                entry_date=entry_date, exit_date=day.date,
                direction=direction, entry_price=entry_price,
                exit_price=exit_price, confidence=confidence,
                shares=shares, pnl=pnl, hold_days=hold_days,
                exit_reason=exit_reason, source=source,
            ))
            in_trade = False
            cooldown_remaining = cooldown
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
        stop_pct = s_pct
        tp_pct = t_pct

        if trail_fn is not None:
            trail_pct = trail_fn(conf, action)
        else:
            trail_pct = TRAILING_STOP_BASE * (1.0 - conf) ** 4

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

    if in_trade and len(signals) > 0:
        last = signals[-1]
        exit_price = last.day_close
        pnl = _apply_costs(entry_price, exit_price, shares, direction)
        trades.append(Trade(
            entry_date=entry_date, exit_date=last.date,
            direction=direction, entry_price=entry_price,
            exit_price=exit_price, confidence=confidence,
            shares=shares, pnl=pnl, hold_days=hold_days,
            exit_reason='end', source=source,
        ))

    return trades


def _summary_line(trades, name=''):
    """One-line summary of trade results."""
    n = len(trades)
    if n == 0:
        return f"  {name:<52} {'0':>6}  {'---':>5}  {'---':>10}  {'---':>5}  {'---':>5}  {'---':>5}  {'---':>8}"
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    total = sum(t.pnl for t in trades)
    pnls = np.array([t.pnl for t in trades])
    sharpe = (pnls.mean() / pnls.std() * np.sqrt(252 / max(np.mean([t.hold_days for t in trades]), 1))
              ) if pnls.std() > 0 else 0
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    mdd = float(dd.max()) / CAPITAL * 100 if len(dd) > 0 else 0
    big_l = min(t.pnl for t in trades)
    # Train/test WR
    train = [t for t in trades if t.entry_date.year <= 2021]
    test = [t for t in trades if t.entry_date.year > 2021]
    tr_wr = sum(1 for t in train if t.pnl > 0) / len(train) * 100 if train else 0
    ts_wr = sum(1 for t in test if t.pnl > 0) / len(test) * 100 if test else 0
    return (f"  {name:<52} {n:>5} {wr:>5.1f}% ${total:>+9,.0f}  Sh={sharpe:>5.1f}  "
            f"DD={mdd:>4.1f}%  Tr={tr_wr:.0f}% Ts={ts_wr:.0f}%  BL=${big_l:>+8,.0f}")


# ---------------------------------------------------------------------------
# Combo factories
# ---------------------------------------------------------------------------

def make_s1_tf3_combo():
    """Streak≥1, 3+ TFs aligned (from v7 relaxed persistence: 210 trades, 97.1% WR, -$98 BL)."""
    prev_tf_states = {}
    streaks = defaultdict(int)
    def fn(day):
        nonlocal prev_tf_states, streaks
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    streaks[tf] = 0; continue
                md = state.get('momentum_direction', 0.0)
                prev_md = prev_tf_states.get(tf, 0.0)
                if (md > 0 and prev_md > 0) or (md < 0 and prev_md < 0):
                    streaks[tf] += 1
                else:
                    streaks[tf] = 1 if md != 0 else 0
                prev_tf_states[tf] = md
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            return None
        action = day.cs_action
        confirmed = 0
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False): continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks.get(tf, 0) >= 1:
                    confirmed += 1
        if confirmed < 3: return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, day.cs_confidence, s, t, 'CS')
    return fn


def make_s1_tf3_vix_combo(cascade_vix):
    """s1_tf3 + VIX cooldown filter."""
    prev_tf_states = {}
    streaks = defaultdict(int)
    def fn(day):
        nonlocal prev_tf_states, streaks
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    streaks[tf] = 0; continue
                md = state.get('momentum_direction', 0.0)
                prev_md = prev_tf_states.get(tf, 0.0)
                if (md > 0 and prev_md > 0) or (md < 0 and prev_md < 0):
                    streaks[tf] += 1
                else:
                    streaks[tf] = 1 if md != 0 else 0
                prev_tf_states[tf] = md
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            return None
        action = day.cs_action
        confirmed = 0
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False): continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks.get(tf, 0) >= 1:
                    confirmed += 1
        if confirmed < 3: return None
        # VIX filter
        sig = _SigProxy(day)
        ana = _AnalysisProxy(day.cs_tf_states)
        ok, adj, _ = cascade_vix.evaluate(
            sig, ana, feature_vec=None, bar_datetime=day.date,
            higher_tf_data=None, spy_df=None, vix_df=None,
        )
        if not ok or adj < MIN_SIGNAL_CONFIDENCE:
            return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, adj, s, t, 'CS')
    return fn


def make_tf4_vix_combo(cascade_vix):
    """X: TF4+VIX (baseline for comparison)."""
    def fn(day):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) < 4:
                return None
            sig = _SigProxy(day)
            ana = _AnalysisProxy(day.cs_tf_states)
            ok, adj, _ = cascade_vix.evaluate(
                sig, ana, feature_vec=None, bar_datetime=day.date,
                higher_tf_data=None, spy_df=None, vix_df=None,
            )
            if not ok or adj < MIN_SIGNAL_CONFIDENCE:
                return None
            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return (day.cs_action, adj, s, t, 'CS')
        return None
    return fn


def make_tf4_vix_v5_combo(cascade_vix):
    """Y: TF4+VIX+V5 (baseline)."""
    def fn(day):
        action = None
        conf = 0.0
        src = 'CS'
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) >= 4:
                sig = _SigProxy(day)
                ana = _AnalysisProxy(day.cs_tf_states)
                ok, adj, _ = cascade_vix.evaluate(
                    sig, ana, feature_vec=None, bar_datetime=day.date,
                    higher_tf_data=None, spy_df=None, vix_df=None,
                )
                if ok and adj >= MIN_SIGNAL_CONFIDENCE:
                    action = day.cs_action
                    conf = adj
        # V5 override
        if day.v5_take_bounce:
            if action == 'BUY':
                conf = max(conf, day.v5_confidence)
                src = 'CS+V5'
            elif action is None:
                action = 'BUY'
                conf = day.v5_confidence
                src = 'V5'
        if action is None or conf < MIN_SIGNAL_CONFIDENCE:
            return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (action, conf, s, t, src)
    return fn


def make_gap_filter_combo(base_fn, signals, max_gap_pct=0.02):
    """Wrap a base combo with overnight gap filter.
    Block signals when |open - prev_close| / prev_close > threshold."""
    prev_close = {}
    for i, day in enumerate(signals):
        prev_close[day.date] = day.day_close

    def fn(day, _idx=[0]):
        result = base_fn(day)
        if result is None:
            return None
        # Check gap on this day (signal day, trade enters next day)
        # Actually we want to check entry day gap, but we evaluate signal on day before
        # So we check if current day had a big gap
        idx = None
        for i, s in enumerate(signals):
            if s.date == day.date:
                idx = i
                break
        if idx is not None and idx > 0:
            prev = signals[idx - 1]
            gap = abs(day.day_open - prev.day_close) / prev.day_close
            if gap > max_gap_pct:
                return None
        return result
    return fn


def make_range_filter_combo(base_fn, signals, max_range_pct=0.05, min_range_pct=0.005):
    """Block signals when daily range is too wide (whipsaw) or too narrow (dead)."""
    def fn(day):
        result = base_fn(day)
        if result is None:
            return None
        if day.day_close > 0:
            daily_range = (day.day_high - day.day_low) / day.day_close
            if daily_range > max_range_pct or daily_range < min_range_pct:
                return None
        return result
    return fn


def make_dow_filter_combo(base_fn, blocked_days):
    """Block signals on specific days of week (0=Mon, 4=Fri)."""
    def fn(day):
        if hasattr(day.date, 'dayofweek') and day.date.dayofweek in blocked_days:
            return None
        return base_fn(day)
    return fn


# ---------------------------------------------------------------------------
# Precompute helpers
# ---------------------------------------------------------------------------

def compute_rolling_sharpe(signals, window=20):
    """Rolling Sharpe ratio from daily close returns."""
    closes = np.array([s.day_close for s in signals], dtype=float)
    dates = [s.date for s in signals]
    returns = np.diff(closes) / closes[:-1]
    returns = np.insert(returns, 0, 0.0)  # pad first

    sharpe_by_date = {}
    for i in range(window, len(returns)):
        r = returns[i-window:i]
        if r.std() > 0:
            sharpe_by_date[dates[i]] = r.mean() / r.std() * np.sqrt(252)
        else:
            sharpe_by_date[dates[i]] = 0.0
    return sharpe_by_date


def analyze_day_of_week(trades, name=''):
    """Analyze win rate by day of week for a set of trades."""
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    dow_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0})
    for t in trades:
        dow = t.entry_date.dayofweek
        if t.pnl > 0:
            dow_stats[dow]['wins'] += 1
        else:
            dow_stats[dow]['losses'] += 1
        dow_stats[dow]['pnl'] += t.pnl

    print(f"\n  Day-of-week analysis for {name}:")
    print(f"  {'Day':<5} {'Trades':>6} {'Wins':>5} {'Loss':>5} {'WR':>7} {'PnL':>10}")
    for dow in range(5):
        s = dow_stats[dow]
        n = s['wins'] + s['losses']
        if n == 0: continue
        wr = s['wins'] / n * 100
        print(f"  {dow_names[dow]:<5} {n:>6} {s['wins']:>5} {s['losses']:>5} {wr:>6.1f}% ${s['pnl']:>+9,.0f}")


def analyze_gap_correlation(signals, trades, name=''):
    """Check if overnight gap size correlates with trade outcome."""
    # Build date->prev_close lookup
    prev_close = {}
    for i in range(1, len(signals)):
        prev_close[signals[i].date] = signals[i-1].day_close

    win_gaps = []
    loss_gaps = []
    for t in trades:
        pc = prev_close.get(t.entry_date)
        if pc and pc > 0:
            gap = (t.entry_price - pc) / pc  # positive = gap up
            if t.direction == 'SHORT':
                gap = -gap  # flip for shorts
            if t.pnl > 0:
                win_gaps.append(gap)
            else:
                loss_gaps.append(gap)

    if win_gaps:
        print(f"\n  Gap analysis for {name}:")
        print(f"  Win gaps:  mean={np.mean(win_gaps)*100:+.3f}%, std={np.std(win_gaps)*100:.3f}%")
        if loss_gaps:
            print(f"  Loss gaps: mean={np.mean(loss_gaps)*100:+.3f}%, std={np.std(loss_gaps)*100:.3f}%")
        else:
            print(f"  Loss gaps: none (100% WR)")


def analyze_confidence_x_tf(signals, combo_fn, name=''):
    """Analyze WR by confidence bucket x TF count interaction."""
    from v15.validation.combo_backtest import _apply_costs

    # Run through signals, track features per trade
    trade_features = []
    in_trade = False
    cooldown_remaining = 0

    for day_idx, day in enumerate(signals):
        if in_trade:
            hold_days += 1
            price_h, price_l, price_c = day.day_high, day.day_low, day.day_close

            if direction == 'LONG':
                best_price = max(best_price, price_h)
                trailing_stop = best_price * (1.0 - trail_pct)
                if best_price > entry_price:
                    effective_stop = max(stop_price, trailing_stop)
                else:
                    effective_stop = stop_price
                hit_stop = price_l <= effective_stop
                hit_tp = price_h >= tp_price
            else:
                best_price = min(best_price, price_l)
                trailing_stop = best_price * (1.0 + trail_pct)
                if best_price < entry_price:
                    effective_stop = min(stop_price, trailing_stop)
                else:
                    effective_stop = stop_price
                hit_stop = price_h >= effective_stop
                hit_tp = price_l <= tp_price

            exit_p = None
            if hit_stop:
                exit_p = effective_stop
                reason = ('trailing' if (direction == 'LONG' and best_price > entry_price and trailing_stop > stop_price) or
                          (direction == 'SHORT' and best_price < entry_price and trailing_stop < stop_price) else 'stop')
            elif hit_tp:
                exit_p, reason = tp_price, 'tp'
            elif hold_days >= MAX_HOLD_DAYS:
                exit_p, reason = price_c, 'timeout'
            else:
                continue

            pnl = _apply_costs(entry_price, exit_p, shares, direction)
            trade_features.append({
                'conf': feat_conf, 'tf_count': feat_tf, 'win': pnl > 0, 'pnl': pnl
            })
            in_trade = False
            cooldown_remaining = 2
            continue

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        result = combo_fn(day)
        if result is None: continue
        action, conf, s_pct, t_pct, src = result
        if action is None or conf < MIN_SIGNAL_CONFIDENCE: continue
        if day_idx + 1 >= len(signals): break

        next_day = signals[day_idx + 1]
        entry_price = next_day.day_open
        if entry_price <= 0: continue

        feat_conf = conf
        feat_tf = _count_tf_confirming(day, action)

        confidence = conf
        trail_pct = TRAILING_STOP_BASE * (1.0 - conf) ** 4
        position_value = CAPITAL * min(conf, 1.0)
        shares = max(1, int(position_value / entry_price))
        stop_pct = s_pct
        tp_pct = t_pct

        if action == 'BUY':
            direction = 'LONG'
            stop_price = entry_price * (1.0 - stop_pct)
            tp_price = entry_price * (1.0 + tp_pct)
            best_price = entry_price
        else:
            direction = 'SHORT'
            stop_price = entry_price * (1.0 + stop_pct)
            tp_price = entry_price * (1.0 - tp_pct)
            best_price = entry_price
        in_trade = True
        hold_days = 0

    if not trade_features:
        return

    df = pd.DataFrame(trade_features)
    print(f"\n  Confidence x TF count interaction for {name}:")
    print(f"  {'Conf Range':<15} {'TF Count':>8} {'Trades':>6} {'WR':>7} {'AvgPnL':>9}")

    conf_bins = [(0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.0)]
    for lo, hi in conf_bins:
        for tf in [3, 4, 5, 6]:
            mask = (df['conf'] >= lo) & (df['conf'] < hi) & (df['tf_count'] == tf)
            sub = df[mask]
            if len(sub) == 0: continue
            wr = sub['win'].mean() * 100
            avg_pnl = sub['pnl'].mean()
            print(f"  {lo:.2f}-{hi:.2f}       TF={tf:>2}    {len(sub):>5}  {wr:>6.1f}%  ${avg_pnl:>+8,.0f}")


# ---------------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------------

def main():
    cache_path = Path(__file__).parent / 'combo_cache' / 'combo_signals.pkl'
    print(f"Loading from {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    signals = data['signals']
    vix_daily = data.get('vix_daily')
    print(f"  {len(signals)} days, {signals[0].date.date()} to {signals[-1].date.date()}\n")

    # Build VIX cascade
    cascade_vix = _build_filter_cascade(vix=True)
    if vix_daily is not None:
        cascade_vix.precompute_vix_cooldown(vix_daily)
        print(f"[FILTER] VIX cooldown precomputed\n")

    # Base combos for testing
    def cs_all(day):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return (day.cs_action, day.cs_confidence, s, t, 'CS')
        return None

    # =====================================================================
    # EXPERIMENT 1: COOLDOWN SWEEP ON TOP COMBOS
    # =====================================================================
    print("=" * 100)
    print("  EXPERIMENT 1: COOLDOWN SWEEP ON TOP COMBOS")
    print("=" * 100)

    top_combos = {
        'X: TF4+VIX': make_tf4_vix_combo(cascade_vix),
        'Y: TF4+VIX+V5': make_tf4_vix_v5_combo(cascade_vix),
        's1_tf3': make_s1_tf3_combo(),
        's1_tf3+VIX': make_s1_tf3_vix_combo(cascade_vix),
        'B: CS-ALL': cs_all,
    }

    for combo_name, combo_fn in top_combos.items():
        print(f"\n  --- {combo_name} ---")
        for cd in [0, 1, 2, 3]:
            # Need fresh combo_fn for stateful combos (persistence tracking)
            if 's1_tf3' in combo_name and 'VIX' in combo_name:
                fn = make_s1_tf3_vix_combo(cascade_vix)
            elif 's1_tf3' in combo_name:
                fn = make_s1_tf3_combo()
            elif combo_name == 'X: TF4+VIX':
                fn = make_tf4_vix_combo(cascade_vix)
            elif combo_name == 'Y: TF4+VIX+V5':
                fn = make_tf4_vix_v5_combo(cascade_vix)
            else:
                fn = combo_fn
            trades = simulate_custom(signals, fn, combo_name, cooldown=cd)
            print(_summary_line(trades, f"cd={cd}"))

    # =====================================================================
    # EXPERIMENT 2: s1_tf3 VARIANTS
    # =====================================================================
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 2: s1_tf3 VARIANTS")
    print("=" * 100)

    # s1_tf3 standalone
    for label, fn_maker in [
        ('s1_tf3 (baseline cd=2)', lambda: make_s1_tf3_combo()),
        ('s1_tf3+VIX (cd=2)', lambda: make_s1_tf3_vix_combo(cascade_vix)),
        ('s1_tf3 (cd=0)', lambda: make_s1_tf3_combo()),
        ('s1_tf3+VIX (cd=0)', lambda: make_s1_tf3_vix_combo(cascade_vix)),
    ]:
        cd = 0 if 'cd=0' in label else 2
        trades = simulate_custom(signals, fn_maker(), label, cooldown=cd)
        print(_summary_line(trades, label))

    # =====================================================================
    # EXPERIMENT 3: DAY-OF-WEEK ANALYSIS
    # =====================================================================
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 3: DAY-OF-WEEK ANALYSIS")
    print("=" * 100)

    # Analyze DOW for top combos
    for combo_name in ['X: TF4+VIX', 'Y: TF4+VIX+V5', 's1_tf3']:
        if combo_name == 'X: TF4+VIX':
            fn = make_tf4_vix_combo(cascade_vix)
        elif combo_name == 'Y: TF4+VIX+V5':
            fn = make_tf4_vix_v5_combo(cascade_vix)
        else:
            fn = make_s1_tf3_combo()
        trades = simulate_custom(signals, fn, combo_name, cooldown=2)
        analyze_day_of_week(trades, combo_name)

    # Test blocking worst days
    print("\n  --- DOW filtering (X: TF4+VIX) ---")
    for blocked in [[], [0], [4], [0, 4], [0, 1], [3, 4]]:
        fn = make_dow_filter_combo(make_tf4_vix_combo(cascade_vix), blocked)
        trades = simulate_custom(signals, fn, f"block {blocked}", cooldown=2)
        day_names = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri'}
        blocked_str = '+'.join(day_names.get(d, str(d)) for d in blocked) if blocked else 'none'
        print(_summary_line(trades, f"block {blocked_str}"))

    # =====================================================================
    # EXPERIMENT 4: GAP ANALYSIS
    # =====================================================================
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 4: GAP ANALYSIS")
    print("=" * 100)

    for combo_name in ['X: TF4+VIX', 's1_tf3']:
        if combo_name == 'X: TF4+VIX':
            fn = make_tf4_vix_combo(cascade_vix)
        else:
            fn = make_s1_tf3_combo()
        trades = simulate_custom(signals, fn, combo_name, cooldown=2)
        analyze_gap_correlation(signals, trades, combo_name)

    # Gap filter sweep
    print("\n  --- Gap filter sweep (X: TF4+VIX) ---")
    for max_gap in [0.01, 0.015, 0.02, 0.03, 0.05, 1.0]:
        base = make_tf4_vix_combo(cascade_vix)
        fn = make_gap_filter_combo(base, signals, max_gap_pct=max_gap)
        trades = simulate_custom(signals, fn, f"gap<={max_gap*100:.1f}%", cooldown=2)
        print(_summary_line(trades, f"gap<={max_gap*100:.1f}%"))

    # =====================================================================
    # EXPERIMENT 5: DAILY RANGE FILTER
    # =====================================================================
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 5: DAILY RANGE FILTER")
    print("=" * 100)

    # Range stats
    ranges = [(s.day_high - s.day_low) / s.day_close for s in signals if s.day_close > 0]
    print(f"  Daily range stats: mean={np.mean(ranges)*100:.2f}%, "
          f"median={np.median(ranges)*100:.2f}%, "
          f"p10={np.percentile(ranges, 10)*100:.2f}%, "
          f"p90={np.percentile(ranges, 90)*100:.2f}%")

    print("\n  --- Range filter sweep (X: TF4+VIX) ---")
    for max_r in [0.03, 0.04, 0.05, 0.06, 0.08, 1.0]:
        base = make_tf4_vix_combo(cascade_vix)
        fn = make_range_filter_combo(base, signals, max_range_pct=max_r, min_range_pct=0.0)
        trades = simulate_custom(signals, fn, f"range<={max_r*100:.0f}%", cooldown=2)
        print(_summary_line(trades, f"range<={max_r*100:.0f}%"))

    # =====================================================================
    # EXPERIMENT 6: ROLLING SHARPE REGIME
    # =====================================================================
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 6: ROLLING SHARPE REGIME FILTER")
    print("=" * 100)

    sharpe_by_date = compute_rolling_sharpe(signals, window=20)
    sharpe_vals = list(sharpe_by_date.values())
    print(f"  Rolling 20d Sharpe: mean={np.mean(sharpe_vals):.2f}, "
          f"std={np.std(sharpe_vals):.2f}, "
          f"min={np.min(sharpe_vals):.2f}, max={np.max(sharpe_vals):.2f}")

    # Check Sharpe at loss dates
    for combo_name, combo_fn_maker in [
        ('X: TF4+VIX', lambda: make_tf4_vix_combo(cascade_vix)),
    ]:
        trades = simulate_custom(signals, combo_fn_maker(), combo_name, cooldown=2)
        losses = [t for t in trades if t.pnl <= 0]
        wins = [t for t in trades if t.pnl > 0]
        print(f"\n  Sharpe at entry dates ({combo_name}):")
        for t in losses:
            sv = sharpe_by_date.get(t.entry_date, None)
            print(f"    LOSS {t.entry_date.date()} PnL=${t.pnl:+,.0f} Sharpe20={sv:.2f}" if sv else
                  f"    LOSS {t.entry_date.date()} PnL=${t.pnl:+,.0f} Sharpe20=N/A")

        win_sharpes = [sharpe_by_date.get(t.entry_date, None) for t in wins]
        win_sharpes = [s for s in win_sharpes if s is not None]
        if win_sharpes:
            print(f"  Win Sharpe20 mean: {np.mean(win_sharpes):.2f} (std {np.std(win_sharpes):.2f})")

    # Sharpe filter sweep
    print("\n  --- Sharpe filter sweep (block when rolling Sharpe < threshold) ---")
    for thresh in [-3.0, -2.0, -1.0, 0.0, 0.5, 1.0]:
        def make_sharpe_filter(base_fn, threshold):
            def fn(day):
                sv = sharpe_by_date.get(day.date)
                if sv is not None and sv < threshold:
                    return None
                return base_fn(day)
            return fn
        fn = make_sharpe_filter(make_tf4_vix_combo(cascade_vix), thresh)
        trades = simulate_custom(signals, fn, f"sharpe>={thresh:.1f}", cooldown=2)
        print(_summary_line(trades, f"sharpe>={thresh:.1f}"))

    # =====================================================================
    # EXPERIMENT 7: ASYMMETRIC TRAIL (FIXED)
    # =====================================================================
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 7: ASYMMETRIC TRAIL (FIXED)")
    print("=" * 100)

    # The v7 bug was that all results showed 0. Let's fix by properly passing trail_fn.
    trail_configs = [
        ('quartic/quartic (baseline)',
         lambda c, d: TRAILING_STOP_BASE * (1.0 - c) ** 4),
        ('cubic long / quartic short',
         lambda c, d: TRAILING_STOP_BASE * (1.0 - c) ** 3 if d == 'BUY' else TRAILING_STOP_BASE * (1.0 - c) ** 4),
        ('quartic long / cubic short',
         lambda c, d: TRAILING_STOP_BASE * (1.0 - c) ** 4 if d == 'BUY' else TRAILING_STOP_BASE * (1.0 - c) ** 3),
        ('quintic/quartic',
         lambda c, d: TRAILING_STOP_BASE * (1.0 - c) ** 5 if d == 'BUY' else TRAILING_STOP_BASE * (1.0 - c) ** 4),
        ('sextic (power=6) both',
         lambda c, d: TRAILING_STOP_BASE * (1.0 - c) ** 6),
        ('octic (power=8) both',
         lambda c, d: TRAILING_STOP_BASE * (1.0 - c) ** 8),
        ('tight base=0.015 quartic',
         lambda c, d: 0.015 * (1.0 - c) ** 4),
        ('wide base=0.035 quartic',
         lambda c, d: 0.035 * (1.0 - c) ** 4),
    ]

    for label, trail_fn in trail_configs:
        fn = make_tf4_vix_combo(cascade_vix)
        trades = simulate_custom(signals, fn, label, cooldown=2, trail_fn=trail_fn)
        print(_summary_line(trades, label))

    # =====================================================================
    # EXPERIMENT 8: CONFIDENCE x TF COUNT INTERACTION
    # =====================================================================
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 8: CONFIDENCE x TF COUNT INTERACTION")
    print("=" * 100)

    analyze_confidence_x_tf(signals, cs_all, "B: CS-ALL")

    # =====================================================================
    # EXPERIMENT 9: COMBINED BEST — s1_tf3+VIX, cooldown=0, best trail
    # =====================================================================
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 9: COMBINED BEST CONFIGURATIONS")
    print("=" * 100)

    configs = [
        ('X: TF4+VIX cd=0 quartic',
         lambda: make_tf4_vix_combo(cascade_vix), 0, None),
        ('X: TF4+VIX cd=0 sextic',
         lambda: make_tf4_vix_combo(cascade_vix), 0,
         lambda c, d: TRAILING_STOP_BASE * (1.0 - c) ** 6),
        ('Y: TF4+VIX+V5 cd=0 quartic',
         lambda: make_tf4_vix_v5_combo(cascade_vix), 0, None),
        ('Y: TF4+VIX+V5 cd=0 sextic',
         lambda: make_tf4_vix_v5_combo(cascade_vix), 0,
         lambda c, d: TRAILING_STOP_BASE * (1.0 - c) ** 6),
        ('s1_tf3 cd=0 quartic',
         lambda: make_s1_tf3_combo(), 0, None),
        ('s1_tf3+VIX cd=0 quartic',
         lambda: make_s1_tf3_vix_combo(cascade_vix), 0, None),
        ('s1_tf3+VIX cd=0 sextic',
         lambda: make_s1_tf3_vix_combo(cascade_vix), 0,
         lambda c, d: TRAILING_STOP_BASE * (1.0 - c) ** 6),
        ('s1_tf3+VIX cd=1 quartic',
         lambda: make_s1_tf3_vix_combo(cascade_vix), 1, None),
    ]

    for label, fn_maker, cd, trail_fn in configs:
        trades = simulate_custom(signals, fn_maker(), label, cooldown=cd, trail_fn=trail_fn)
        print(_summary_line(trades, label))

    # =====================================================================
    # EXPERIMENT 10: ENTRY PRICE RELATIONSHIP TO CHANNEL
    # =====================================================================
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 10: CHANNEL SCORE ANALYSIS AT TRADE ENTRIES")
    print("=" * 100)

    fn = make_tf4_vix_combo(cascade_vix)
    trades_x = simulate_custom(signals, fn, "X", cooldown=2)

    # Match trades to signal days to get channel scores
    signal_dates = {s.date: s for s in signals}
    win_scores = {'position': [], 'energy': [], 'confluence': [], 'timing': [], 'health': []}
    loss_scores = {'position': [], 'energy': [], 'confluence': [], 'timing': [], 'health': []}

    for t in trades_x:
        # Signal was day before entry
        for i, s in enumerate(signals):
            if i + 1 < len(signals) and signals[i+1].date == t.entry_date:
                target = win_scores if t.pnl > 0 else loss_scores
                target['position'].append(s.cs_position_score)
                target['energy'].append(s.cs_energy_score)
                target['confluence'].append(s.cs_confluence_score)
                target['timing'].append(s.cs_timing_score)
                target['health'].append(s.cs_channel_health)
                break

    print(f"\n  Channel scores at entry (X: TF4+VIX):")
    print(f"  {'Score':<15} {'Win Mean':>10} {'Win Std':>10} {'Loss Mean':>10} {'Loss Std':>10}")
    for key in ['position', 'energy', 'confluence', 'timing', 'health']:
        wm = np.mean(win_scores[key]) if win_scores[key] else 0
        ws = np.std(win_scores[key]) if win_scores[key] else 0
        lm = np.mean(loss_scores[key]) if loss_scores[key] else 0
        ls = np.std(loss_scores[key]) if loss_scores[key] else 0
        print(f"  {key:<15} {wm:>10.3f} {ws:>10.3f} {lm:>10.3f} {ls:>10.3f}")

    # Channel score filter sweep
    print("\n  --- Channel health filter (X: TF4+VIX) ---")
    for min_health in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]:
        def make_health_filter(base_fn, threshold):
            def fn(day):
                if day.cs_channel_health < threshold:
                    return None
                return base_fn(day)
            return fn
        fn = make_health_filter(make_tf4_vix_combo(cascade_vix), min_health)
        trades = simulate_custom(signals, fn, f"health>={min_health:.1f}", cooldown=2)
        print(_summary_line(trades, f"health>={min_health:.1f}"))

    print("\n\n" + "=" * 100)
    print("  ALL v8 EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
