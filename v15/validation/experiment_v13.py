#!/usr/bin/env python3
"""
V13 Experiments: Push past 99 trades at 100% WR and close BC's gap.

Current best:
  BQ: 99 trades, 100% WR, $190K (SPY>SMA20+0.6%)
  BC: 156 trades, 98.7%, $277K, BL=-$13 (s1tf3+VIX+SPY+short>=0.65 sex)
  AZ: 112 trades, 99.1%, $209K, BL=-$12 (s1tf3+VIX+SPY+LONG sex)

BC's 2 losses:
  1. 2024-09-04 SHORT $210.59 PnL=-$13 conf=0.870 (unavoidable tiny loss)
  2. One more loss with conf < 0.65 but > 0.45

Can we find a filter that keeps more than 99 trades at 100% WR?

Experiments:
  1. Hybrid SPY filter (different thresholds for LONG vs SHORT)
  2. TF3+VIX (no streak) + SPY combos (more trades from relaxed TF req)
  3. V5 override additions to 100% combos
  4. Multi-SMA consensus (SPY above BOTH SMA20 and SMA50)
  5. Confidence-weighted position sizing analysis
  6. Direction-specific trail power
  7. Adaptive SPY threshold (lower when TSLA also trending)
  8. Cross-asset correlation regime
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
        return f"  {name:<60} {'0':>5}  {'---':>5}  {'---':>10}  {'---':>5}  {'---':>8}"
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
    return (f"  {name:<60} {n:>5} {wr:>5.1f}% ${total:>+9,.0f}  Sh={sharpe:>5.1f}  "
            f"DD={mdd:>4.1f}%  BL=${big_l:>+8,.0f}  Tr={tr_wr:.0f}% Ts={ts_wr:.0f}%")


# ---------------------------------------------------------------------------
# Combo factories
# ---------------------------------------------------------------------------

def _make_tf4_vix_base(cascade_vix):
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
    """TF3+VIX (no streak) -- more trades than TF4."""
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


def _make_tf4_vix_v5_base(cascade_vix):
    """TF4+VIX+V5 override (Y base)."""
    def fn(day: DaySignals):
        action = None
        conf = 0.0
        src = ''
        s_pct = DEFAULT_STOP_PCT
        t_pct = DEFAULT_TP_PCT

        # CS signal
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
                    s_pct, t_pct = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                    src = 'CS'

        # V5 override
        if day.v5_take_bounce and day.v5_confidence >= MIN_SIGNAL_CONFIDENCE:
            if action == 'BUY':
                conf = max(conf, day.v5_confidence)
                src = 'CS+V5'
            elif action is None:
                action = 'BUY'
                conf = day.v5_confidence
                src = 'V5'

        if action is None:
            return None
        return (action, conf, s_pct, t_pct, src)
    return fn


def wrap_with_filters(base_fn, spy_set=None, long_only=False, short_min_conf=None,
                      min_confidence=None, long_spy_set=None, short_spy_set=None):
    """Wrap combo fn with filters. Supports per-direction SPY sets."""
    def fn(day):
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if long_only and action == 'SELL':
            return None
        if min_confidence is not None and conf < min_confidence:
            return None
        if short_min_conf is not None and action == 'SELL' and conf < short_min_conf:
            return None
        if spy_set is not None and day.date not in spy_set:
            return None
        # Per-direction SPY filter
        if long_spy_set is not None and action == 'BUY' and day.date not in long_spy_set:
            return None
        if short_spy_set is not None and action == 'SELL' and day.date not in short_spy_set:
            return None

        return result
    return fn


# ---------------------------------------------------------------------------
# Pre-computations
# ---------------------------------------------------------------------------

def precompute_spy_distance_set(spy_daily, window=20, min_dist_pct=0.0):
    if spy_daily is None or len(spy_daily) < window:
        return set()
    spy_close = spy_daily['close'].values.astype(float)
    sma = pd.Series(spy_close).rolling(window).mean().values
    above = set()
    for i in range(window, len(spy_close)):
        if sma[i] > 0:
            dist = (spy_close[i] - sma[i]) / sma[i] * 100
            if dist >= min_dist_pct:
                above.add(spy_daily.index[i])
    return above


def precompute_spy_below_set(spy_daily, window=20, max_dist_pct=0.0):
    """SPY below SMA by at most max_dist_pct (for shorts in downtrends)."""
    if spy_daily is None or len(spy_daily) < window:
        return set()
    spy_close = spy_daily['close'].values.astype(float)
    sma = pd.Series(spy_close).rolling(window).mean().values
    below = set()
    for i in range(window, len(spy_close)):
        if sma[i] > 0:
            dist = (spy_close[i] - sma[i]) / sma[i] * 100
            if dist <= max_dist_pct:
                below.add(spy_daily.index[i])
    return below


def precompute_tsla_sma(signals, window=20):
    closes = np.array([s.day_close for s in signals], dtype=float)
    sma = pd.Series(closes).rolling(window).mean().values
    above = set()
    for i in range(window, len(signals)):
        if closes[i] > sma[i]:
            above.add(signals[i].date)
    return above


def precompute_spy_sma50(spy_daily):
    if spy_daily is None or len(spy_daily) < 50:
        return set()
    spy_close = spy_daily['close'].values.astype(float)
    sma50 = pd.Series(spy_close).rolling(50).mean().values
    above = set()
    for i in range(50, len(spy_close)):
        if spy_close[i] > sma50[i]:
            above.add(spy_daily.index[i])
    return above


# ---------------------------------------------------------------------------
# EXPERIMENT 1: HYBRID SPY FILTER (different thresholds per direction)
# ---------------------------------------------------------------------------

def run_hybrid_spy(signals, cascade_vix, spy_daily):
    print("=" * 100)
    print("  EXPERIMENT 1: HYBRID SPY FILTER (LONG vs SHORT thresholds)")
    print("=" * 100)
    print("  Key idea: LONGs need SPY uptrend, SHORTs need SPY downtrend or flat")
    print("  The 2024-09-04 SHORT loss: SPY was above SMA20 -> short in uptrend = bad")

    # Pre-compute various SPY sets
    spy_above_0 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_above_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    spy_above_1 = precompute_spy_distance_set(spy_daily, 20, 1.0)

    # Standard approach: same threshold for both directions
    print("\n  --- TF4+VIX + uniform SPY threshold ---")
    for dist in [0.0, 0.3, 0.6, 1.0]:
        spy_set = precompute_spy_distance_set(spy_daily, 20, dist)
        fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy_set)
        label = f"TF4+VIX SPY>={dist}% uniform cd=0"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
        print(_summary_line(trades, label))

    # Hybrid: LONG needs SPY > 0%, SHORT needs SPY > higher threshold
    print("\n  --- TF4+VIX + hybrid SPY (LONG>=X%, SHORT>=Y%) ---")
    configs = [
        (0.0, 0.6, "LONG>=0 SHORT>=0.6"),
        (0.0, 1.0, "LONG>=0 SHORT>=1.0"),
        (0.0, 1.5, "LONG>=0 SHORT>=1.5"),
        (0.3, 0.6, "LONG>=0.3 SHORT>=0.6"),
        (0.3, 1.0, "LONG>=0.3 SHORT>=1.0"),
        (0.6, 1.0, "LONG>=0.6 SHORT>=1.0"),
        (0.6, 1.5, "LONG>=0.6 SHORT>=1.5"),
    ]
    for long_dist, short_dist, desc in configs:
        long_set = precompute_spy_distance_set(spy_daily, 20, long_dist)
        short_set = precompute_spy_distance_set(spy_daily, 20, short_dist)
        fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix),
                               long_spy_set=long_set, short_spy_set=short_set)
        label = f"TF4+VIX {desc} cd=0"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
        print(_summary_line(trades, label))

    # Same for s1_tf3+VIX
    print("\n  --- s1tf3+VIX + hybrid SPY sex ---")
    for long_dist, short_dist, desc in configs:
        long_set = precompute_spy_distance_set(spy_daily, 20, long_dist)
        short_set = precompute_spy_distance_set(spy_daily, 20, short_dist)
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix),
                               long_spy_set=long_set, short_spy_set=short_set)
        label = f"s1tf3+VIX {desc} cd=0 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 2: TF3+VIX (no streak) + SPY combos
# ---------------------------------------------------------------------------

def run_tf3_combos(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 2: TF3+VIX (no streak) + SPY COMBOS")
    print("=" * 100)
    print("  TF3 = more trades (relaxed from TF4). Can we maintain WR with SPY filter?")

    spy_above_0 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_above_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    spy_above_1 = precompute_spy_distance_set(spy_daily, 20, 1.0)

    # Baselines
    fn = _make_tf3_vix_combo(cascade_vix)
    trades = simulate_trades(signals, fn, "TF3+VIX cd=0", cooldown=0, trail_power=4)
    print(_summary_line(trades, "TF3+VIX cd=0 (baseline)"))

    fn = _make_tf3_vix_combo(cascade_vix)
    trades = simulate_trades(signals, fn, "TF3+VIX cd=0 sex", cooldown=0, trail_power=6)
    print(_summary_line(trades, "TF3+VIX cd=0 sex"))

    # TF3+VIX+SPY combos
    print("\n  --- TF3+VIX + SPY ---")
    for dist, tp in [(0.0, 4), (0.0, 6), (0.3, 4), (0.6, 4), (0.6, 6), (1.0, 4)]:
        spy_set = precompute_spy_distance_set(spy_daily, 20, dist)
        fn = wrap_with_filters(_make_tf3_vix_combo(cascade_vix), spy_set=spy_set)
        label = f"TF3+VIX+SPY>={dist}% cd=0 t^{tp}"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=tp)
        print(_summary_line(trades, label))

    # TF3+VIX+SPY + short conf
    print("\n  --- TF3+VIX+SPY + short conf boost ---")
    for dist, sc in [(0.0, 0.65), (0.0, 0.70), (0.3, 0.65), (0.6, 0.65)]:
        spy_set = precompute_spy_distance_set(spy_daily, 20, dist)
        fn = wrap_with_filters(_make_tf3_vix_combo(cascade_vix),
                               spy_set=spy_set, short_min_conf=sc)
        label = f"TF3+VIX+SPY>={dist}% shrt>={sc} cd=0 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # TF3+VIX+SPY LONG only
    print("\n  --- TF3+VIX+SPY LONG only ---")
    for dist in [0.0, 0.3, 0.6]:
        spy_set = precompute_spy_distance_set(spy_daily, 20, dist)
        fn = wrap_with_filters(_make_tf3_vix_combo(cascade_vix),
                               spy_set=spy_set, long_only=True)
        label = f"TF3+VIX+SPY>={dist}% LONG cd=0 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 3: V5 OVERRIDE + 100% WR COMBOS
# ---------------------------------------------------------------------------

def run_v5_additions(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 3: V5 BOUNCE OVERRIDE + 100% WR COMBOS")
    print("=" * 100)
    print("  Can V5 add trades without breaking 100% WR?")

    spy_above_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    spy_above_1 = precompute_spy_distance_set(spy_daily, 20, 1.0)

    # Baseline with V5 override
    fn = wrap_with_filters(_make_tf4_vix_v5_base(cascade_vix), spy_set=spy_above_06)
    label = "TF4+VIX+V5+SPY>=0.6% cd=0"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
    print(_summary_line(trades, label))

    fn = wrap_with_filters(_make_tf4_vix_v5_base(cascade_vix), spy_set=spy_above_1)
    label = "TF4+VIX+V5+SPY>=1.0% cd=0"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
    print(_summary_line(trades, label))

    # V5 only with SPY filter
    fn_v5_spy = lambda day: (('BUY', day.v5_confidence, DEFAULT_STOP_PCT, DEFAULT_TP_PCT, 'V5')
                              if day.v5_take_bounce and day.v5_confidence >= MIN_SIGNAL_CONFIDENCE
                              else None)
    fn = wrap_with_filters(fn_v5_spy, spy_set=spy_above_06)
    label = "V5+SPY>=0.6% cd=0"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
    print(_summary_line(trades, label))

    fn = wrap_with_filters(fn_v5_spy, spy_set=spy_above_1)
    label = "V5+SPY>=1.0% cd=0"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
    print(_summary_line(trades, label))

    # s1_tf3+VIX+V5+SPY (add V5 bounce to s1_tf3 combo)
    def make_s1_tf3_vix_v5(cascade_vix):
        """s1_tf3+VIX with V5 override."""
        base = _make_s1_tf3_vix_combo(cascade_vix)
        def fn(day):
            result = base(day)
            # V5 override: add bounce signals
            if day.v5_take_bounce and day.v5_confidence >= MIN_SIGNAL_CONFIDENCE:
                if result is None:
                    return ('BUY', day.v5_confidence, DEFAULT_STOP_PCT, DEFAULT_TP_PCT, 'V5')
                elif result[0] == 'BUY':
                    action, conf, s, t, src = result
                    return (action, max(conf, day.v5_confidence), s, t, 'CS+V5')
            return result
        return fn

    fn = wrap_with_filters(make_s1_tf3_vix_v5(cascade_vix), spy_set=spy_above_06)
    label = "s1tf3+VIX+V5+SPY>=0.6% cd=0 sex"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
    print(_summary_line(trades, label))

    fn = wrap_with_filters(make_s1_tf3_vix_v5(cascade_vix),
                           spy_set=spy_above_06, short_min_conf=0.65)
    label = "s1tf3+VIX+V5+SPY>=0.6% shrt65 cd=0 sex"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
    print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 4: MULTI-SMA CONSENSUS
# ---------------------------------------------------------------------------

def run_multi_sma(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 4: MULTI-SMA CONSENSUS")
    print("=" * 100)
    print("  Require SPY above BOTH SMA20 and SMA50 for extra safety")

    spy20 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy50 = precompute_spy_sma50(spy_daily)
    spy_both = spy20 & spy50
    print(f"  SPY above SMA20: {len(spy20)} days")
    print(f"  SPY above SMA50: {len(spy50)} days")
    print(f"  SPY above BOTH: {len(spy_both)} days")

    # TF4+VIX + dual SMA
    configs = [
        ("TF4+VIX+SPY20 cd=0", spy20, 4, 4),
        ("TF4+VIX+SPY50 cd=0", spy50, 4, 4),
        ("TF4+VIX+SPY20+50 cd=0", spy_both, 4, 4),
    ]
    for label, spy_set, tf, tp in configs:
        fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy_set)
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=tp)
        print(_summary_line(trades, label))

    # s1_tf3+VIX + dual SMA
    configs2 = [
        ("s1tf3+VIX+SPY20 cd=0 sex", spy20, 6),
        ("s1tf3+VIX+SPY50 cd=0 sex", spy50, 6),
        ("s1tf3+VIX+SPY20+50 cd=0 sex", spy_both, 6),
    ]
    for label, spy_set, tp in configs2:
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy_set)
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=tp)
        print(_summary_line(trades, label))

    # Dual SMA + short conf
    fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix),
                           spy_set=spy_both, short_min_conf=0.65)
    label = "s1tf3+VIX+SPY20+50 shrt65 cd=0 sex"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
    print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 5: ADAPTIVE SPY (lower threshold when TSLA also trending)
# ---------------------------------------------------------------------------

def run_adaptive_spy(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 5: ADAPTIVE SPY (relax when TSLA also trends)")
    print("=" * 100)

    tsla_above_sma20 = precompute_tsla_sma(signals, window=20)
    print(f"  TSLA above SMA20: {len(tsla_above_sma20)} days")

    # Concept: SPY>=0.6% OR (SPY>=0% AND TSLA>SMA20)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)

    # Adaptive: SPY>0 + TSLA>SMA20, union with SPY>0.6 (no TSLA req)
    adaptive_set = spy_06 | (spy_00 & tsla_above_sma20)
    only_tsla_add = (spy_00 & tsla_above_sma20) - spy_06
    print(f"  SPY>=0.6%: {len(spy_06)} days")
    print(f"  SPY>=0% + TSLA>SMA20: {len(spy_00 & tsla_above_sma20)} days")
    print(f"  Adaptive (union): {len(adaptive_set)} days (+{len(only_tsla_add)} extra)")

    fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy_06)
    label = "TF4+VIX+SPY>=0.6% cd=0 (baseline)"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
    print(_summary_line(trades, label))

    fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=adaptive_set)
    label = "TF4+VIX+SPY>=0.6%|TSLA cd=0"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
    print(_summary_line(trades, label))

    # Try multiple relaxation levels
    for spy_strict, spy_relax in [(0.6, 0.0), (0.6, 0.3), (1.0, 0.0), (1.0, 0.3), (1.0, 0.6)]:
        strict = precompute_spy_distance_set(spy_daily, 20, spy_strict)
        relax = precompute_spy_distance_set(spy_daily, 20, spy_relax)
        combo = strict | (relax & tsla_above_sma20)
        fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=combo)
        label = f"TF4+VIX SPY>={spy_strict}%|({spy_relax}%+TSLA) cd=0"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
        print(_summary_line(trades, label))

    # Same for s1_tf3
    print()
    for spy_strict, spy_relax in [(0.6, 0.0), (1.0, 0.0)]:
        strict = precompute_spy_distance_set(spy_daily, 20, spy_strict)
        relax = precompute_spy_distance_set(spy_daily, 20, spy_relax)
        combo = strict | (relax & tsla_above_sma20)
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=combo)
        label = f"s1tf3+VIX SPY>={spy_strict}%|({spy_relax}%+TSLA) cd=0 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # Short conf + adaptive
    for spy_strict, spy_relax, sc in [(0.6, 0.0, 0.65), (0.6, 0.0, 0.70)]:
        strict = precompute_spy_distance_set(spy_daily, 20, spy_strict)
        relax = precompute_spy_distance_set(spy_daily, 20, spy_relax)
        combo = strict | (relax & tsla_above_sma20)
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix),
                               spy_set=combo, short_min_conf=sc)
        label = f"s1tf3+VIX SPY>={spy_strict}%|({spy_relax}%+TSLA) shrt{sc} sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 6: DIRECTION-SPECIFIC ANALYSIS
# ---------------------------------------------------------------------------

def run_direction_analysis(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 6: DIRECTION-SPECIFIC ANALYSIS")
    print("=" * 100)

    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)

    # Analyze: how many longs vs shorts at each SPY threshold?
    print("\n  --- Trade direction breakdown by SPY threshold ---")
    for dist in [0.0, 0.3, 0.6, 1.0]:
        spy_set = precompute_spy_distance_set(spy_daily, 20, dist)
        fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy_set)
        trades = simulate_trades(signals, fn, f"test", cooldown=0, trail_power=4)
        longs = [t for t in trades if t.direction == 'LONG']
        shorts = [t for t in trades if t.direction == 'SHORT']
        l_wr = sum(1 for t in longs if t.pnl > 0) / max(len(longs), 1) * 100
        s_wr = sum(1 for t in shorts if t.pnl > 0) / max(len(shorts), 1) * 100
        l_pnl = sum(t.pnl for t in longs)
        s_pnl = sum(t.pnl for t in shorts)
        print(f"    SPY>={dist}%: {len(longs)}L ({l_wr:.0f}% ${l_pnl:+,.0f}) + "
              f"{len(shorts)}S ({s_wr:.0f}% ${s_pnl:+,.0f}) = {len(trades)} total")


# ---------------------------------------------------------------------------
# EXPERIMENT 7: GRAND COMBO SWEEP (combine best from all experiments)
# ---------------------------------------------------------------------------

def run_grand_sweep(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 7: GRAND COMBO SWEEP")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_03 = precompute_spy_distance_set(spy_daily, 20, 0.3)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    spy_1 = precompute_spy_distance_set(spy_daily, 20, 1.0)
    tsla20 = precompute_tsla_sma(signals, 20)
    spy50 = precompute_spy_sma50(spy_daily)

    # Adaptive sets
    adapt_06_0 = spy_06 | (spy_00 & tsla20)
    adapt_1_0 = spy_1 | (spy_00 & tsla20)
    adapt_1_06 = spy_1 | (spy_06 & tsla20)

    configs = [
        # Current leaders (for comparison)
        ("BQ: SPY>=0.6% cd=0", _make_tf4_vix_base(cascade_vix), spy_06, None, False, 4),
        ("AW: SPY>=1.0% cd=0", _make_tf4_vix_base(cascade_vix), spy_1, None, False, 4),

        # Adaptive SPY combos
        ("BR: SPY>=0.6%|TSLA cd=0", _make_tf4_vix_base(cascade_vix), adapt_06_0, None, False, 4),
        ("BS: SPY>=1.0%|TSLA cd=0", _make_tf4_vix_base(cascade_vix), adapt_1_0, None, False, 4),
        ("BT: SPY>=1.0%|(0.6%+TSLA) cd=0", _make_tf4_vix_base(cascade_vix), adapt_1_06, None, False, 4),

        # s1_tf3 adaptive
        ("BU: s1tf3 SPY>=0.6%|TSLA cd=0 sex", _make_s1_tf3_vix_combo(cascade_vix), adapt_06_0, None, False, 6),
        ("BV: s1tf3 SPY>=0.6%|TSLA shrt65 sex", _make_s1_tf3_vix_combo(cascade_vix), adapt_06_0, 0.65, False, 6),

        # TF3 (relaxed TF) combos
        ("BW: TF3+VIX+SPY>=0.6% cd=0", _make_tf3_vix_combo(cascade_vix), spy_06, None, False, 4),
        ("BX: TF3+VIX+SPY>=0.6% LONG cd=0 sex", _make_tf3_vix_combo(cascade_vix), spy_06, None, True, 6),
        ("BY: TF3+VIX+SPY>=0.6% shrt65 cd=0 sex", _make_tf3_vix_combo(cascade_vix), spy_06, 0.65, False, 6),

        # V5 additions
        ("BZ: TF4+V5+SPY>=0.6% cd=0", _make_tf4_vix_v5_base(cascade_vix), spy_06, None, False, 4),
        ("CA: TF4+V5+SPY>=1.0% cd=0", _make_tf4_vix_v5_base(cascade_vix), spy_1, None, False, 4),
    ]

    for label, base_fn, spy_set, short_conf, long_only, tp in configs:
        fn = wrap_with_filters(base_fn, spy_set=spy_set, short_min_conf=short_conf, long_only=long_only)
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=tp)
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
        print("[FILTER] VIX cooldown precomputed\n")

    run_hybrid_spy(signals, cascade_vix, spy_daily)
    run_tf3_combos(signals, cascade_vix, spy_daily)
    run_v5_additions(signals, cascade_vix, spy_daily)
    run_multi_sma(signals, cascade_vix, spy_daily)
    run_adaptive_spy(signals, cascade_vix, spy_daily)
    run_direction_analysis(signals, cascade_vix, spy_daily)
    run_grand_sweep(signals, cascade_vix, spy_daily)

    print("\n\n" + "=" * 100)
    print("  ALL v13 EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
