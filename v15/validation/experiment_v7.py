#!/usr/bin/env python3
"""
v7 Physics-Based Filter Experiments

1. Permutation Entropy filter on remaining losses
2. Hurst Exponent / DFA regime detection
3. Relaxed persistence combos (more trades while keeping WR)
4. Asymmetric trail (cubic for longs, quartic for shorts)
5. Profit maximization: wider TP + quartic trail
"""

import pickle, sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.combo_backtest import (
    DaySignals, Trade, _apply_costs, _floor_stop_tp, _count_tf_confirming,
    _SigProxy, _AnalysisProxy, _build_filter_cascade,
    MIN_SIGNAL_CONFIDENCE, CAPITAL, DEFAULT_STOP_PCT, DEFAULT_TP_PCT,
    MAX_HOLD_DAYS, COOLDOWN_DAYS, TRAILING_STOP_BASE,
)


# ===========================================================================
# Physics-based filter implementations
# ===========================================================================

def permutation_entropy(series, order=3, delay=1, normalize=True):
    """Compute permutation entropy of a time series."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < (order - 1) * delay + 1:
        return np.nan

    pattern_counts = {}
    total = 0

    for i in range(n - (order - 1) * delay):
        indices = [i + j * delay for j in range(order)]
        window = series[indices]
        pattern = tuple(np.argsort(np.argsort(window)))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        total += 1

    if total == 0:
        return np.nan

    probs = np.array(list(pattern_counts.values())) / total
    pe = -np.sum(probs * np.log2(probs + 1e-15))

    if normalize:
        import math
        max_entropy = np.log2(math.factorial(order))
        pe /= max_entropy if max_entropy > 0 else 1.0

    return pe


def rolling_permutation_entropy(closes, window=60, order=3):
    """Compute rolling PE on log returns."""
    log_returns = np.diff(np.log(closes))
    n = len(log_returns)
    result = np.full(len(closes), np.nan)
    for i in range(window, n):
        result[i + 1] = permutation_entropy(log_returns[i - window:i], order=order)
    return result


def hurst_rs(series, min_window=10, max_window=None):
    """Rescaled Range Hurst exponent. Pure numpy."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if max_window is None:
        max_window = min(n // 2, 100)

    window_sizes = []
    rs_values = []

    for w in range(min_window, max_window + 1, max(1, (max_window - min_window) // 15)):
        rs_list = []
        for start in range(0, n - w + 1, w):
            segment = series[start:start + w]
            if len(segment) < 2:
                continue
            mean = segment.mean()
            devs = np.cumsum(segment - mean)
            R = devs.max() - devs.min()
            S = segment.std(ddof=1)
            if S > 1e-10:
                rs_list.append(R / S)
        if rs_list:
            window_sizes.append(w)
            rs_values.append(np.mean(rs_list))

    if len(window_sizes) < 3:
        return 0.5

    log_w = np.log(np.array(window_sizes))
    log_rs = np.log(np.array(rs_values))
    H, _ = np.polyfit(log_w, log_rs, 1)
    return float(H)


def rolling_hurst(closes, window=120, step=1):
    """Compute rolling Hurst exponent on log returns."""
    log_returns = np.diff(np.log(closes))
    n = len(log_returns)
    result = np.full(len(closes), np.nan)
    for i in range(window, n, step):
        result[i + 1] = hurst_rs(log_returns[i - window:i])
    return result


# ===========================================================================
# Simulator with configurable per-direction trail
# ===========================================================================

def simulate_asym(signals, combo_fn, long_formula='quartic', long_base=0.025,
                  short_formula='quartic', short_base=0.025):
    """Simulate with asymmetric long/short trail parameters."""
    trades = []
    in_trade = False
    cooldown_remaining = 0
    entry_date = entry_price = confidence = shares = 0
    stop_price = tp_price = best_price = hold_days = 0
    direction = source = ''
    trail_pct = 0.025

    def _compute_trail(conf, formula, base):
        c = 1.0 - conf
        if formula == 'quadratic':
            return base * c ** 2
        elif formula == 'cubic':
            return base * c ** 3
        elif formula == 'quartic':
            return base * c ** 4
        return base * c ** 2

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
                exit_reason = 'trailing' if (
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

        if action == 'BUY':
            direction = 'LONG'
            trail_pct = _compute_trail(conf, long_formula, long_base)
            stop_price = entry_price * (1.0 - s_pct)
            tp_price = entry_price * (1.0 + t_pct)
            best_price = entry_price
        elif action == 'SELL':
            direction = 'SHORT'
            trail_pct = _compute_trail(conf, short_formula, short_base)
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
        print(f"{label:<50} {'0':>4} trades")
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

    train = [t for t in trades if t.entry_date.year <= 2021]
    test = [t for t in trades if t.entry_date.year > 2021]
    t_wr = sum(1 for t in train if t.pnl > 0) / len(train) * 100 if train else 0
    s_wr = sum(1 for t in test if t.pnl > 0) / len(test) * 100 if test else 0

    print(f"{label:<50} {n:>4} {wr:>5.1f}% ${total:>+9,.0f}  "
          f"Sh={sharpe:>5.1f}  DD={mdd:>4.1f}%  "
          f"Tr={t_wr:.0f}% Ts={s_wr:.0f}%  BL=${big_l:>+7,.0f}")

    if verbose:
        for t in trades:
            if t.pnl <= 0:
                print(f"  LOSS: {t.entry_date.date()} {t.direction} PnL=${t.pnl:+,.0f} "
                      f"conf={t.confidence:.2f} [{t.source}]")
    return {'n': n, 'wr': wr, 'pnl': total, 'sharpe': sharpe, 'mdd': mdd, 'big_l': big_l}


# ===========================================================================
# Experiments
# ===========================================================================

def exp_pe_filter(signals, combo_fn, daily_df, label="X: TF4+VIX"):
    """Test permutation entropy as a signal filter."""
    print("\n" + "=" * 100)
    print(f"  EXPERIMENT 1: PERMUTATION ENTROPY FILTER ({label})")
    print("=" * 100)

    closes = daily_df['close'].values.astype(float)
    dates = daily_df.index

    # Pre-compute rolling PE with different windows
    print("  Computing rolling PE...")
    pe_60 = rolling_permutation_entropy(closes, window=60, order=3)
    pe_30 = rolling_permutation_entropy(closes, window=30, order=3)
    pe_100 = rolling_permutation_entropy(closes, window=100, order=4)

    # Build lookup by date
    pe_by_date = {}
    for i, date in enumerate(dates):
        pe_by_date[date] = {
            'pe60': pe_60[i] if i < len(pe_60) else np.nan,
            'pe30': pe_30[i] if i < len(pe_30) else np.nan,
            'pe100': pe_100[i] if i < len(pe_100) else np.nan,
        }

    print(f"  PE60 range: {np.nanmin(pe_60):.4f} - {np.nanmax(pe_60):.4f}")
    print(f"  PE30 range: {np.nanmin(pe_30):.4f} - {np.nanmax(pe_30):.4f}")

    # First, check PE values at all loss dates
    from v15.validation.combo_backtest import simulate_trades
    trades = simulate_trades(signals, combo_fn, label)
    losses = [t for t in trades if t.pnl <= 0]
    wins = [t for t in trades if t.pnl > 0]

    print(f"\n  PE values at LOSS entry dates:")
    loss_pe_vals = []
    for t in losses:
        pe_val = pe_by_date.get(t.entry_date, {}).get('pe60', np.nan)
        loss_pe_vals.append(pe_val)
        print(f"    {t.entry_date.date()} PnL=${t.pnl:+,.0f} PE60={pe_val:.4f}")

    print(f"\n  PE values at WIN entry dates (sample):")
    win_pe_vals = []
    for t in wins:
        pe_val = pe_by_date.get(t.entry_date, {}).get('pe60', np.nan)
        win_pe_vals.append(pe_val)

    win_pe_vals = [v for v in win_pe_vals if not np.isnan(v)]
    loss_pe_vals = [v for v in loss_pe_vals if not np.isnan(v)]

    if win_pe_vals and loss_pe_vals:
        print(f"    Win PE60 mean: {np.mean(win_pe_vals):.4f} (std {np.std(win_pe_vals):.4f})")
        print(f"    Loss PE60 mean: {np.mean(loss_pe_vals):.4f} (std {np.std(loss_pe_vals):.4f})")

        # Test PE as a filter
        print(f"\n  Sweep PE threshold (block if PE < threshold):")
        valid_pe = [v for v in pe_60 if not np.isnan(v)]
        for pctile in [5, 10, 15, 20, 25, 30]:
            threshold = np.percentile(valid_pe, pctile)

            def make_pe_combo(fn, pe_data, pe_thresh):
                def wrapper(day):
                    r = fn(day)
                    if r is None:
                        return r
                    pe_val = pe_data.get(day.date, {}).get('pe60', np.nan)
                    if not np.isnan(pe_val) and pe_val < pe_thresh:
                        return None
                    return r
                return wrapper

            pe_fn = make_pe_combo(combo_fn, pe_by_date, threshold)
            trades_pe = simulate_trades(signals, pe_fn, f"PE<{pctile}pct")
            summarize(trades_pe, f"  block PE60 < {pctile}th pctile ({threshold:.4f})")


def exp_hurst_filter(signals, combo_fn, daily_df, label="X: TF4+VIX"):
    """Test Hurst exponent as a signal filter."""
    print("\n" + "=" * 100)
    print(f"  EXPERIMENT 2: HURST EXPONENT FILTER ({label})")
    print("=" * 100)

    closes = daily_df['close'].values.astype(float)
    dates = daily_df.index

    print("  Computing rolling Hurst (120-day window, step=5)...")
    hurst_arr = rolling_hurst(closes, window=120, step=5)

    # Forward-fill NaN gaps from step=5
    last_val = 0.5
    for i in range(len(hurst_arr)):
        if not np.isnan(hurst_arr[i]):
            last_val = hurst_arr[i]
        else:
            hurst_arr[i] = last_val

    hurst_by_date = {}
    for i, date in enumerate(dates):
        hurst_by_date[date] = hurst_arr[i] if i < len(hurst_arr) else 0.5

    valid_h = [v for v in hurst_arr if not np.isnan(v)]
    print(f"  Hurst range: {min(valid_h):.4f} - {max(valid_h):.4f}")
    print(f"  Hurst mean: {np.mean(valid_h):.4f}")

    # Check Hurst at loss dates
    from v15.validation.combo_backtest import simulate_trades
    trades = simulate_trades(signals, combo_fn, label)
    losses = [t for t in trades if t.pnl <= 0]
    wins = [t for t in trades if t.pnl > 0]

    print(f"\n  Hurst at LOSS dates:")
    for t in losses:
        h = hurst_by_date.get(t.entry_date, 0.5)
        print(f"    {t.entry_date.date()} PnL=${t.pnl:+,.0f} H={h:.4f}")

    loss_h = [hurst_by_date.get(t.entry_date, 0.5) for t in losses]
    win_h = [hurst_by_date.get(t.entry_date, 0.5) for t in wins]
    print(f"\n  Win Hurst mean: {np.mean(win_h):.4f}")
    print(f"  Loss Hurst mean: {np.mean(loss_h):.4f}")

    # Sweep Hurst threshold (block if trending too strongly)
    print(f"\n  Block when Hurst > threshold (trending regime):")
    for h_thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
        def make_hurst_combo(fn, h_data, h_thresh):
            def wrapper(day):
                r = fn(day)
                if r is None:
                    return r
                h = h_data.get(day.date, 0.5)
                if h > h_thresh:
                    return None
                return r
            return wrapper

        h_fn = make_hurst_combo(combo_fn, hurst_by_date, h_thresh)
        trades_h = simulate_trades(signals, h_fn, f"H>{h_thresh}")
        summarize(trades_h, f"  block Hurst > {h_thresh}")


def exp_relaxed_persistence(signals):
    """Try relaxing persistence combos to get more trades."""
    print("\n" + "=" * 100)
    print("  EXPERIMENT 3: RELAXED PERSISTENCE COMBOS")
    print("=" * 100)

    from v15.validation.combo_backtest import simulate_trades

    # Import the persistence combo maker
    from v15.validation.combo_backtest import _make_momentum_persist_combo

    configs = [
        ('s1 tf3', 1, 3),
        ('s1 tf4', 1, 4),
        ('s2 tf3', 2, 3),
        ('s2 tf4', 2, 4),  # = AA: Persist24
        ('s3 tf3', 3, 3),
        ('s3 tf4', 3, 4),
    ]

    for label, streak, tfs in configs:
        fn = _make_momentum_persist_combo(min_streak=streak, min_tfs=tfs)
        trades = simulate_trades(signals, fn, label)
        summarize(trades, f"  {label}")


def exp_asymmetric_trail(signals, combo_fn, label="X: TF4+VIX"):
    """Test asymmetric trail parameters for longs vs shorts."""
    print("\n" + "=" * 100)
    print(f"  EXPERIMENT 4: ASYMMETRIC TRAIL ({label})")
    print("=" * 100)

    configs = [
        ('quartic/quartic (baseline)', 'quartic', 0.025, 'quartic', 0.025),
        ('cubic long / quartic short', 'cubic', 0.025, 'quartic', 0.025),
        ('cubic long / quartic short b=0.02', 'cubic', 0.020, 'quartic', 0.020),
        ('quartic long / cubic short', 'quartic', 0.025, 'cubic', 0.025),
        ('quadratic long / quartic short', 'quadratic', 0.025, 'quartic', 0.025),
    ]

    for lbl, lf, lb, sf, sb in configs:
        trades = simulate_asym(signals, combo_fn, lf, lb, sf, sb)
        summarize(trades, f"  {lbl}")


def exp_profit_maximization(signals, combo_fn, label="X: TF4+VIX"):
    """Try wider TP/stop combos to maximize profit with quartic trail."""
    print("\n" + "=" * 100)
    print(f"  EXPERIMENT 5: PROFIT MAXIMIZATION ({label})")
    print("=" * 100)

    from v15.validation.combo_backtest import simulate_trades

    def make_tp_override(fn, new_tp):
        def wrapper(day):
            r = fn(day)
            if r is None:
                return None
            action, conf, s, t, src = r
            return (action, conf, s, max(t, new_tp), src)
        return wrapper

    # Also test removing TP entirely (rely solely on trail)
    def make_no_tp(fn):
        def wrapper(day):
            r = fn(day)
            if r is None:
                return None
            action, conf, s, _, src = r
            return (action, conf, s, 1.0, src)  # 100% TP = essentially no TP
        return wrapper

    configs = [
        ('baseline (4% TP)', combo_fn),
        ('6% TP', make_tp_override(combo_fn, 0.06)),
        ('8% TP', make_tp_override(combo_fn, 0.08)),
        ('10% TP', make_tp_override(combo_fn, 0.10)),
        ('15% TP', make_tp_override(combo_fn, 0.15)),
        ('no TP (trail only)', make_no_tp(combo_fn)),
    ]

    for lbl, fn in configs:
        trades = simulate_trades(signals, fn, lbl)
        summarize(trades, f"  {lbl}")

    # Also test longer hold periods
    print("\n  With extended max hold:")
    for max_h in [10, 15, 20, 30]:
        # Temporarily override MAX_HOLD_DAYS
        import v15.validation.combo_backtest as cb
        orig = cb.MAX_HOLD_DAYS
        cb.MAX_HOLD_DAYS = max_h
        trades = simulate_trades(signals, combo_fn, f"hold={max_h}")
        summarize(trades, f"  max_hold={max_h}")
        cb.MAX_HOLD_DAYS = orig


def exp_cooldown_sweep(signals, combo_fn, label="X: TF4+VIX"):
    """Sweep cooldown days — fewer cooldown = more trades."""
    print("\n" + "=" * 100)
    print(f"  EXPERIMENT 6: COOLDOWN SWEEP ({label})")
    print("=" * 100)

    import v15.validation.combo_backtest as cb

    for cd in [0, 1, 2, 3, 5]:
        orig = cb.COOLDOWN_DAYS
        cb.COOLDOWN_DAYS = cd
        trades = cb.simulate_trades(signals, combo_fn, f"cd={cd}")
        summarize(trades, f"  cooldown={cd} days")
        cb.COOLDOWN_DAYS = orig


def main():
    cache_path = Path(__file__).parent / 'combo_cache' / 'combo_signals.pkl'
    print(f"Loading from {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    signals = data['signals']
    daily_df = data['daily_df']
    vix_daily = data.get('vix_daily')
    print(f"  {len(signals)} days, {signals[0].date.date()} to {signals[-1].date.date()}\n")

    # Build VIX cascade
    cascade_vix = _build_filter_cascade(vix=True)
    if vix_daily is not None:
        cascade_vix.precompute_vix_cooldown(vix_daily)

    # X: TF4+VIX combo function
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

    # Y: TF4+VIX+V5
    def make_tf4_vix_v5(day):
        action = conf = None
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
        if action is None or conf is None or conf < MIN_SIGNAL_CONFIDENCE:
            return None
        sig = _SigProxy(day, action=action, conf=conf)
        ana = _AnalysisProxy(day.cs_tf_states)
        ok, adj, _ = cascade_vix.evaluate(sig, ana, feature_vec=None,
                                          bar_datetime=day.date,
                                          higher_tf_data=None, spy_df=None, vix_df=None)
        if not ok or adj < MIN_SIGNAL_CONFIDENCE:
            return None
        return (action, adj, stop, tp, src)

    # Run experiments
    exp_pe_filter(signals, make_tf4_vix, daily_df, "X: TF4+VIX")
    exp_hurst_filter(signals, make_tf4_vix, daily_df, "X: TF4+VIX")
    exp_relaxed_persistence(signals)
    exp_asymmetric_trail(signals, make_tf4_vix, "X: TF4+VIX")
    exp_asymmetric_trail(signals, make_tf4_vix_v5, "Y: TF4+VIX+V5")
    exp_profit_maximization(signals, make_tf4_vix, "X: TF4+VIX")
    exp_profit_maximization(signals, make_tf4_vix_v5, "Y: TF4+VIX+V5")
    exp_cooldown_sweep(signals, make_tf4_vix, "X: TF4+VIX")

    print("\n" + "=" * 100)
    print("  ALL v7 EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
