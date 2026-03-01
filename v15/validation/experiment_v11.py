#!/usr/bin/env python3
"""
V11 Experiments: Expand trade count while preserving ultra-high win rate.

Key insight: AI (s1tf3+VIX cd=0) = 269 trades, 97.8%, $464K profit
             AJ (TF4+VIX+SPY) = 96 trades, 99.0%, $198K, BL=-$14

Can we get AI-level trade count with AJ-level safety?

FIX: cs_streak is NOT a DaySignals field — it's computed via closure in
_make_s1_tf3_combo(). Must import and wrap original combos, not reimplement.
"""

import pickle, sys, os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming,
    _SigProxy, _AnalysisProxy, _build_filter_cascade, _floor_stop_tp,
    MIN_SIGNAL_CONFIDENCE, CAPITAL, DEFAULT_STOP_PCT, DEFAULT_TP_PCT,
    TRAILING_STOP_BASE, MAX_HOLD_DAYS,
    simulate_trades,
    _make_s1_tf3_combo, _make_s1_tf3_vix_combo,
)


def _summary_line(trades, name=''):
    n = len(trades)
    if n == 0:
        return f"  {name:<55} {'0':>5}  {'---':>5}  {'---':>10}  {'---':>5}  {'---':>5}  {'---':>8}"
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
    train = [t for t in trades if t.entry_date.year <= 2021]
    test = [t for t in trades if t.entry_date.year > 2021]
    tr_wr = sum(1 for t in train if t.pnl > 0) / len(train) * 100 if train else 0
    ts_wr = sum(1 for t in test if t.pnl > 0) / len(test) * 100 if test else 0
    return (f"  {name:<55} {n:>5} {wr:>5.1f}% ${total:>+9,.0f}  Sh={sharpe:>5.1f}  "
            f"DD={mdd:>4.1f}%  Tr={tr_wr:.0f}% Ts={ts_wr:.0f}%  BL=${big_l:>+8,.0f}")


# ---------------------------------------------------------------------------
# Wrapper: add filters on top of any existing combo function
# ---------------------------------------------------------------------------

def wrap_with_filters(base_combo_fn, spy_set=None, require_health=None,
                      max_vol_ratio=None, vol_ratio_by_date=None,
                      max_intra_pos=None, intra_pos_by_date=None):
    """Wrap any combo function with additional pre-filters.
    Checks filters BEFORE calling base_combo_fn, so base_combo_fn's
    internal state (like streak tracking) only advances on eligible days."""
    def fn(day):
        # Always call base first so streak state updates for every day
        result = base_combo_fn(day)
        if result is None:
            return None

        # Now apply additional filters
        if spy_set is not None and day.date not in spy_set:
            return None
        if require_health is not None and day.cs_channel_health < require_health:
            return None
        if max_vol_ratio is not None and vol_ratio_by_date is not None:
            vr = vol_ratio_by_date.get(day.date)
            if vr is not None and vr > max_vol_ratio:
                return None
        if max_intra_pos is not None and intra_pos_by_date is not None:
            ip = intra_pos_by_date.get(day.date)
            if ip is not None and ip > max_intra_pos:
                return None

        return result
    return fn


def _make_tf4_vix_base(cascade_vix):
    """Simple TF4+VIX combo for baseline comparisons."""
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


def _make_tf3_vix_combo(cascade_vix):
    """TF3+VIX (no streak) — same as TF4 but relaxed to 3 TFs."""
    def fn(day):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            if _count_tf_confirming(day, day.cs_action) < 3:
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


# ---------------------------------------------------------------------------
# Pre-computations
# ---------------------------------------------------------------------------

def precompute_spy_sma(spy_daily, windows=[20, 50]):
    result = {}
    if spy_daily is None or len(spy_daily) < max(windows):
        return result
    spy_close = spy_daily['close'].values.astype(float)
    for w in windows:
        sma = pd.Series(spy_close).rolling(w).mean().values
        above = set()
        for i in range(w, len(spy_close)):
            if spy_close[i] > sma[i]:
                above.add(spy_daily.index[i])
        result[w] = above
    return result


def precompute_vol_ratio(signals):
    closes = np.array([s.day_close for s in signals], dtype=float)
    returns = np.diff(closes) / closes[:-1]
    returns = np.insert(returns, 0, 0.0)
    vol5 = pd.Series(np.abs(returns)).rolling(5).mean().values
    vol20 = pd.Series(np.abs(returns)).rolling(20).mean().values
    vol_ratio = np.where(vol20 > 0, vol5 / vol20, 1.0)
    return {signals[i].date: vol_ratio[i] for i in range(len(signals))
            if not np.isnan(vol_ratio[i])}


def precompute_intraday_position(signals):
    result = {}
    for s in signals:
        rng = s.day_high - s.day_low
        if rng > 0:
            pos = (s.day_close - s.day_low) / rng
        else:
            pos = 0.5
        result[s.date] = pos
    return result


def precompute_spy_distances(spy_daily):
    """SPY distance from SMA20 and SMA50 as percentage."""
    if spy_daily is None or len(spy_daily) < 50:
        return {}, {}
    spy_close = spy_daily['close'].values.astype(float)
    spy_sma20 = pd.Series(spy_close).rolling(20).mean().values
    spy_sma50 = pd.Series(spy_close).rolling(50).mean().values
    dist20 = {}
    dist50 = {}
    for i in range(50, len(spy_close)):
        d = spy_daily.index[i]
        dist20[d] = (spy_close[i] - spy_sma20[i]) / spy_sma20[i] * 100
        dist50[d] = (spy_close[i] - spy_sma50[i]) / spy_sma50[i] * 100
    return dist20, dist50


# ---------------------------------------------------------------------------
# EXPERIMENT 1: s1_tf3 + SPY (using original combo functions with wrapping)
# ---------------------------------------------------------------------------

def run_s1tf3_spy_experiment(signals, cascade_vix, spy_sma_sets, vol_ratio_by_date):
    print("=" * 100)
    print("  EXPERIMENT 1: s1_tf3 + SPY REGIME (fixed streak handling)")
    print("=" * 100)

    spy20 = spy_sma_sets.get(20)
    spy50 = spy_sma_sets.get(50)

    # Baselines using original functions
    trades = simulate_trades(signals, _make_s1_tf3_combo(), "AD: s1_tf3")
    print(_summary_line(trades, "AD: s1_tf3 baseline"))

    trades = simulate_trades(signals, _make_s1_tf3_vix_combo(cascade_vix), "AE: s1tf3+VIX")
    print(_summary_line(trades, "AE: s1tf3+VIX baseline"))

    # s1_tf3 (no VIX) + SPY20
    fn = wrap_with_filters(_make_s1_tf3_combo(), spy_set=spy20)
    trades = simulate_trades(signals, fn, "s1_tf3+SPY20", cooldown=2)
    print(_summary_line(trades, "s1_tf3+SPY20 cd=2"))

    fn = wrap_with_filters(_make_s1_tf3_combo(), spy_set=spy20)
    trades = simulate_trades(signals, fn, "s1_tf3+SPY20 cd=0", cooldown=0)
    print(_summary_line(trades, "s1_tf3+SPY20 cd=0"))

    # s1_tf3+VIX + SPY20 (triple: streak + VIX + SPY)
    for cd in [2, 0]:
        for tp in [4, 6]:
            fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20)
            label = f"s1tf3+VIX+SPY20 cd={cd} tp={tp}"
            trades = simulate_trades(signals, fn, label, cooldown=cd, trail_power=tp)
            print(_summary_line(trades, label))

    # s1_tf3+VIX + SPY50 (more restrictive)
    fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy50)
    trades = simulate_trades(signals, fn, "s1tf3+VIX+SPY50 cd=0", cooldown=0)
    print(_summary_line(trades, "s1tf3+VIX+SPY50 cd=0"))

    # s1_tf3+VIX + SPY20 + health
    for min_h in [0.2, 0.3, 0.35]:
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix),
                               spy_set=spy20, require_health=min_h)
        label = f"s1tf3+VIX+SPY+h>={min_h} cd=0"
        trades = simulate_trades(signals, fn, label, cooldown=0)
        print(_summary_line(trades, label))

    # s1_tf3+VIX + SPY20 + vol_ratio
    for max_vr in [1.0, 1.2, 1.5]:
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix),
                               spy_set=spy20, max_vol_ratio=max_vr,
                               vol_ratio_by_date=vol_ratio_by_date)
        label = f"s1tf3+VIX+SPY+vr<{max_vr} cd=0"
        trades = simulate_trades(signals, fn, label, cooldown=0)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 2: INTRADAY POSITION WITH s1_tf3
# ---------------------------------------------------------------------------

def run_intraday_with_s1tf3(signals, cascade_vix, spy_sma_sets, intra_pos_by_date):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 2: INTRADAY POSITION + s1_tf3")
    print("=" * 100)

    spy20 = spy_sma_sets.get(20)

    # Check intra-pos at loss dates for s1_tf3+VIX
    trades_ae = simulate_trades(signals, _make_s1_tf3_vix_combo(cascade_vix), "AE")
    print("  Intraday position at s1_tf3+VIX loss dates:")
    for t in trades_ae:
        if t.pnl <= 0:
            ip = intra_pos_by_date.get(t.entry_date)
            if ip is not None:
                print(f"    LOSS {t.entry_date.date()} PnL=${t.pnl:>+7,.0f} intra_pos={ip:.3f}")

    # Sweep intraday filter on s1_tf3+VIX+SPY
    print("\n  --- s1_tf3+VIX+SPY with intraday position filter ---")
    for max_pos in [0.3, 0.5, 0.7, 0.9]:
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix),
                               spy_set=spy20, max_intra_pos=max_pos,
                               intra_pos_by_date=intra_pos_by_date)
        label = f"s1tf3+VIX+SPY intra<{max_pos}"
        trades = simulate_trades(signals, fn, label, cooldown=0)
        print(_summary_line(trades, label))

    # Also test on TF4+VIX+SPY baseline
    print("\n  --- TF4+VIX+SPY with intraday position filter ---")
    for max_pos in [0.3, 0.5, 0.7, 0.9]:
        fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix),
                               spy_set=spy20, max_intra_pos=max_pos,
                               intra_pos_by_date=intra_pos_by_date)
        label = f"TF4+VIX+SPY intra<{max_pos}"
        trades = simulate_trades(signals, fn, label, cooldown=0)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 3: SPY DISTANCE THRESHOLD
# ---------------------------------------------------------------------------

def run_spy_distance_experiment(signals, cascade_vix, spy_daily, spy_dist20, spy_dist50):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 3: SPY DISTANCE-ABOVE-SMA SWEEP")
    print("=" * 100)

    if not spy_dist20:
        print("  SPY data not available")
        return

    # Check at loss dates
    trades_x = simulate_trades(signals, _make_tf4_vix_base(cascade_vix), "X")
    print("  SPY distance at loss dates:")
    for t in trades_x:
        if t.pnl <= 0:
            d20 = spy_dist20.get(t.entry_date, None)
            d50 = spy_dist50.get(t.entry_date, None)
            d20_str = f"{d20:.2f}%" if d20 is not None else "N/A"
            d50_str = f"{d50:.2f}%" if d50 is not None else "N/A"
            print(f"    LOSS {t.entry_date.date()} PnL=${t.pnl:>+7,.0f} "
                  f"SPY dist20={d20_str} dist50={d50_str}")

    # Sweep distance thresholds with TF4+VIX
    print("\n  --- TF4+VIX + SPY distance sweep ---")
    for min_dist in [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]:
        above_set = set(d for d, dist in spy_dist20.items() if dist >= min_dist)
        fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=above_set)
        label = f"TF4+VIX SPY>SMA20+{min_dist}%"
        trades = simulate_trades(signals, fn, label, cooldown=0)
        print(_summary_line(trades, label))

    # Sweep with s1_tf3+VIX
    print("\n  --- s1_tf3+VIX + SPY distance sweep ---")
    for min_dist in [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]:
        above_set = set(d for d, dist in spy_dist20.items() if dist >= min_dist)
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=above_set)
        label = f"s1tf3+VIX SPY>SMA20+{min_dist}%"
        trades = simulate_trades(signals, fn, label, cooldown=0)
        print(_summary_line(trades, label))

    # SPY distance from SMA50
    print("\n  --- TF4+VIX + SPY dist50 sweep ---")
    for min_dist in [0.0, 0.5, 1.0, 2.0, 3.0]:
        above_set = set(d for d, dist in spy_dist50.items() if dist >= min_dist)
        fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=above_set)
        label = f"TF4+VIX SPY>SMA50+{min_dist}%"
        trades = simulate_trades(signals, fn, label, cooldown=0)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 4: PRIOR TRADE GAP FILTER (100% WR when gap<=10d)
# ---------------------------------------------------------------------------

def run_gap_filter_experiment(signals, cascade_vix, spy_sma_sets):
    """The v11 clustering result showed: after a win with gap<=10d = 100% WR.
    Can we build a combo that uses this? Tricky because it depends on trade
    history, not just signals."""
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 4: TRADE GAP ANALYSIS (extended)")
    print("=" * 100)

    spy20 = spy_sma_sets.get(20)

    # Analyze gap patterns for multiple base combos
    for label, combo_fn, cd, tp in [
        ('X: TF4+VIX cd=0', _make_tf4_vix_base(cascade_vix), 0, 4),
        ('AE: s1tf3+VIX cd=0', _make_s1_tf3_vix_combo(cascade_vix), 0, 4),
        ('AJ: TF4+VIX+SPY cd=0', wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy20), 0, 4),
    ]:
        trades = simulate_trades(signals, combo_fn, label, cooldown=cd, trail_power=tp)
        print(f"\n  {label}: {len(trades)} trades")

        for i, t in enumerate(trades):
            if t.pnl <= 0:
                gap = (t.entry_date - trades[i-1].exit_date).days if i > 0 else 999
                print(f"    LOSS {t.entry_date.date()} PnL=${t.pnl:>+7,.0f} gap={gap}d")

        # WR by gap
        for max_gap in [3, 5, 7, 10, 15, 20, 30]:
            follow_wins = 0
            follow_total = 0
            for i in range(1, len(trades)):
                gap = (trades[i].entry_date - trades[i-1].exit_date).days
                if gap <= max_gap and trades[i-1].pnl > 0:
                    follow_total += 1
                    if trades[i].pnl > 0:
                        follow_wins += 1
            if follow_total > 0:
                wr = follow_wins / follow_total * 100
                marker = " <--" if wr >= 100 else ""
                print(f"    After win, gap<={max_gap}d: {follow_total} trades, {wr:.1f}% WR{marker}")

    # Also analyze for s1_tf3+VIX+SPY
    print("\n  --- s1_tf3+VIX+SPY gap analysis ---")
    fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20)
    trades = simulate_trades(signals, fn, "s1tf3+VIX+SPY cd=0", cooldown=0)
    print(f"  s1tf3+VIX+SPY cd=0: {len(trades)} trades")
    for i, t in enumerate(trades):
        if t.pnl <= 0:
            gap = (t.entry_date - trades[i-1].exit_date).days if i > 0 else 999
            print(f"    LOSS {t.entry_date.date()} PnL=${t.pnl:>+7,.0f} gap={gap}d")

    for max_gap in [3, 5, 7, 10, 15, 20]:
        follow_wins = 0
        follow_total = 0
        for i in range(1, len(trades)):
            gap = (trades[i].entry_date - trades[i-1].exit_date).days
            if gap <= max_gap and trades[i-1].pnl > 0:
                follow_total += 1
                if trades[i].pnl > 0:
                    follow_wins += 1
        if follow_total > 0:
            wr = follow_wins / follow_total * 100
            marker = " <--" if wr >= 100 else ""
            print(f"    After win, gap<={max_gap}d: {follow_total} trades, {wr:.1f}% WR{marker}")


# ---------------------------------------------------------------------------
# EXPERIMENT 5: BEST COMBOS COMPREHENSIVE SWEEP
# ---------------------------------------------------------------------------

def run_comprehensive_sweep(signals, cascade_vix, spy_sma_sets, vol_ratio_by_date):
    """Run all the best combos from previous experiments in one place."""
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 5: COMPREHENSIVE BEST COMBOS")
    print("=" * 100)

    spy20 = spy_sma_sets.get(20)
    spy50 = spy_sma_sets.get(50)

    configs = [
        # Baselines
        ('X: TF4+VIX',           _make_tf4_vix_base(cascade_vix), 2, 4),
        ('X cd=0',               _make_tf4_vix_base(cascade_vix), 0, 4),
        ('AE: s1tf3+VIX',        _make_s1_tf3_vix_combo(cascade_vix), 2, 4),
        ('AE cd=0',              _make_s1_tf3_vix_combo(cascade_vix), 0, 4),
        ('AI: AE cd=0 sex',      _make_s1_tf3_vix_combo(cascade_vix), 0, 6),

        # SPY combos
        ('AJ: TF4+VIX+SPY',
         wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy20), 2, 4),
        ('AL: AJ cd=0',
         wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy20), 0, 4),

        # NEW: s1_tf3 + SPY combos
        ('AN: s1tf3+VIX+SPY',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20), 2, 4),
        ('AO: AN cd=0',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20), 0, 4),
        ('AP: AN cd=0 sex',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy20), 0, 6),

        # s1_tf3+VIX+SPY + health
        ('AQ: AN+h0.3 cd=0',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix),
                           spy_set=spy20, require_health=0.3), 0, 4),

        # s1_tf3+VIX+SPY + vol_ratio
        ('AR: AN+vr1.5 cd=0',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix),
                           spy_set=spy20, max_vol_ratio=1.5,
                           vol_ratio_by_date=vol_ratio_by_date), 0, 4),

        # TF3 (no streak) + SPY
        ('AS: TF3+VIX+SPY cd=0',
         wrap_with_filters(_make_tf3_vix_combo(cascade_vix), spy_set=spy20), 0, 4),

        # TF4+SPY+health (100% WR combos from v10)
        ('AT: TF4+SPY+h0.3 cd=0',
         wrap_with_filters(_make_tf4_vix_base(cascade_vix),
                           spy_set=spy20, require_health=0.3), 0, 4),

        # s1_tf3+SPY50 (more restrictive SPY)
        ('AU: s1tf3+VIX+SPY50 cd=0',
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy50), 0, 4),
    ]

    for label, combo_fn, cd, tp in configs:
        trades = simulate_trades(signals, combo_fn, label, cooldown=cd, trail_power=tp)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cache_path = Path(__file__).parent / 'combo_cache' / 'combo_signals.pkl'
    print(f"Loading from {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    signals = data['signals']
    vix_daily = data.get('vix_daily')
    spy_daily = data.get('spy_daily')
    print(f"  {len(signals)} days, {signals[0].date.date()} to {signals[-1].date.date()}\n")

    cascade_vix = _build_filter_cascade(vix=True)
    if vix_daily is not None:
        cascade_vix.precompute_vix_cooldown(vix_daily)
        print(f"[FILTER] VIX cooldown precomputed\n")

    # Pre-compute all auxiliary data
    spy_sma_sets = precompute_spy_sma(spy_daily, windows=[20, 50])
    vol_ratio_by_date = precompute_vol_ratio(signals)
    intra_pos_by_date = precompute_intraday_position(signals)
    spy_dist20, spy_dist50 = precompute_spy_distances(spy_daily)

    run_s1tf3_spy_experiment(signals, cascade_vix, spy_sma_sets, vol_ratio_by_date)
    run_intraday_with_s1tf3(signals, cascade_vix, spy_sma_sets, intra_pos_by_date)
    run_spy_distance_experiment(signals, cascade_vix, spy_daily, spy_dist20, spy_dist50)
    run_gap_filter_experiment(signals, cascade_vix, spy_sma_sets)
    run_comprehensive_sweep(signals, cascade_vix, spy_sma_sets, vol_ratio_by_date)

    print("\n\n" + "=" * 100)
    print("  ALL v11 EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
