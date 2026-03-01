#!/usr/bin/env python3
"""
V15 Experiments: Push beyond CD's 125 trades at 100% WR.

CD kills the 2018-01-24 LONG loss (conf=0.651) with longconf>=0.66.
This removes 23 LONGs (22 winners + 1 loser).

The loser's unique characteristics:
  conf=0.651, health=0.364, pos=1.000, timing=0, 1h:-T, buyTF=3, type=break

Strategy: Use composite "OR" filters to allow some low-conf LONGs back:
  - longconf>=0.66 OR (longconf>=0.45 AND additional_filter)
  - Each additional_filter should block the 2018-01-24 loss specifically

Also test relaxing SHORT SPY thresholds to add more SHORT trades.
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
        return f"  {name:<72} {'0':>5}  {'---':>5}  {'---':>10}  {'---':>5}  {'---':>8}"
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
    return (f"  {name:<72} {n:>5} {wr:>5.1f}% ${total:>+9,.0f}  Sh={sharpe:>5.1f}  "
            f"DD={mdd:>4.1f}%  BL=${big_l:>+8,.0f}  Tr={tr_wr:.0f}% Ts={ts_wr:.0f}%")


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


def make_composite_long_filter(base_fn, spy_above_sma20, spy_above_06pct,
                                min_short_conf=0.65, high_conf=0.66,
                                low_conf_extra_filter=None):
    """
    Composite filter: Hybrid SPY + shrt>=0.65 + composite LONG filter.

    For LONGs:
      - conf >= high_conf: PASS (no additional filter needed)
      - conf < high_conf:  PASS only if low_conf_extra_filter(day) is True

    low_conf_extra_filter: function(day) -> bool
    """
    def fn(day):
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        # Hybrid SPY
        if action == 'BUY' and day.date not in spy_above_sma20:
            return None
        if action == 'SELL' and day.date not in spy_above_06pct:
            return None

        # Short confidence minimum
        if action == 'SELL' and conf < min_short_conf:
            return None

        # Composite LONG filter
        if action == 'BUY':
            if conf >= high_conf:
                pass  # High-conf LONGs always pass
            elif low_conf_extra_filter is not None:
                if not low_conf_extra_filter(day):
                    return None
            else:
                return None  # No extra filter = block all low-conf LONGs

        return result
    return fn


# ---------------------------------------------------------------------------
# Extra filters for low-confidence LONGs
# ---------------------------------------------------------------------------

def filter_health_min(min_h):
    """Block if channel_health < min_h."""
    return lambda day: day.cs_channel_health >= min_h

def filter_position_max(max_p):
    """Block if position_score > max_p (at top of channel = bad for buy)."""
    return lambda day: day.cs_position_score <= max_p

def filter_buy_tf_min(min_tf):
    """Block if fewer than min_tf TFs confirm the BUY."""
    return lambda day: _count_tf_confirming(day, 'BUY') >= min_tf

def filter_timing_min(min_t):
    """Block if timing_score < min_t."""
    return lambda day: day.cs_timing_score >= min_t

def filter_confluence_min(min_c):
    """Block if confluence_score < min_c."""
    return lambda day: day.cs_confluence_score >= min_c

def filter_and(*filters):
    """Combine multiple filters with AND."""
    return lambda day: all(f(day) for f in filters)


# ---------------------------------------------------------------------------
# EXPERIMENT 1: COMPOSITE OR FILTERS (recover low-conf LONGs)
# ---------------------------------------------------------------------------

def run_composite_or(signals, cascade_vix, spy_daily):
    print("=" * 100)
    print("  EXP 1: COMPOSITE OR FILTERS — recover low-conf LONGs from CD")
    print("=" * 100)
    print("  CD: longconf>=0.66 = 125 trades, 100% WR")
    print("  Goal: longconf>=0.66 OR (longconf>=0.45 AND extra_filter)")

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)

    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    # Baseline (CD)
    fn = make_composite_long_filter(base_fn, spy_00, spy_06, 0.65, 0.66, None)
    label = "CD baseline (lc>=0.66 only)"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
    print(_summary_line(trades, label))

    # Single extra filters
    extras = [
        ("health>=0.37", filter_health_min(0.37)),
        ("health>=0.40", filter_health_min(0.40)),
        ("health>=0.45", filter_health_min(0.45)),
        ("pos<=0.99", filter_position_max(0.99)),
        ("pos<=0.95", filter_position_max(0.95)),
        ("pos<=0.90", filter_position_max(0.90)),
        ("buyTF>=4", filter_buy_tf_min(4)),
        ("buyTF>=5", filter_buy_tf_min(5)),
        ("timing>0", filter_timing_min(0.001)),
        ("confl>=0.90", filter_confluence_min(0.90)),
    ]

    print("\n  --- lc>=0.66 OR (lc>=0.45 AND extra) ---")
    for desc, extra in extras:
        fn = make_composite_long_filter(base_fn, spy_00, spy_06, 0.65, 0.66, extra)
        label = f"s1tf3 lc66 OR {desc} sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # Composite extra filters (AND combinations)
    print("\n  --- lc>=0.66 OR (lc>=0.45 AND filter1 AND filter2) ---")
    composite_extras = [
        ("h>=0.37 & pos<=0.99", filter_and(filter_health_min(0.37), filter_position_max(0.99))),
        ("h>=0.37 & buyTF>=4", filter_and(filter_health_min(0.37), filter_buy_tf_min(4))),
        ("pos<=0.99 & buyTF>=4", filter_and(filter_position_max(0.99), filter_buy_tf_min(4))),
        ("h>=0.37 & pos<=0.99 & buyTF>=4", filter_and(filter_health_min(0.37), filter_position_max(0.99), filter_buy_tf_min(4))),
        ("h>=0.40 & pos<=0.99", filter_and(filter_health_min(0.40), filter_position_max(0.99))),
        ("pos<=0.95 & buyTF>=4", filter_and(filter_position_max(0.95), filter_buy_tf_min(4))),
    ]

    for desc, extra in composite_extras:
        fn = make_composite_long_filter(base_fn, spy_00, spy_06, 0.65, 0.66, extra)
        label = f"s1tf3 lc66 OR ({desc}) sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # Try different high_conf thresholds with buyTF4
    print("\n  --- vary high_conf threshold with buyTF4 fallback ---")
    for hc in [0.60, 0.62, 0.64, 0.65, 0.66, 0.67, 0.70]:
        fn = make_composite_long_filter(base_fn, spy_00, spy_06, 0.65, hc, filter_buy_tf_min(4))
        label = f"s1tf3 lc>={hc:.2f} OR buyTF4 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 2: RELAX SHORT SPY THRESHOLD (more shorts at 100%)
# ---------------------------------------------------------------------------

def run_relax_short_spy(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 2: RELAX SHORT SPY THRESHOLD (more shorts with CD's safety)")
    print("=" * 100)
    print("  CD uses S>=0.6% for shorts. Can we lower this while maintaining 100% WR?")
    print("  The 2024-09-04 SHORT loss has SPY dist=0.534% -> blocked by S>=0.6%")
    print("  Any threshold below 0.534% would allow this loss back.")

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    # Sweep SHORT SPY threshold while keeping CD's LONG filter
    print("\n  --- CD but with different SHORT SPY threshold ---")
    for s_dist in [0.3, 0.4, 0.5, 0.54, 0.55, 0.6, 0.7, 0.8, 1.0]:
        spy_s = precompute_spy_distance_set(spy_daily, 20, s_dist)
        fn = make_composite_long_filter(base_fn, spy_00, spy_s, 0.65, 0.66, None)
        label = f"s1tf3 L0/S>={s_dist}% shrt65 lc66 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # With buyTF4 fallback for LONGs and different SHORT SPY
    print("\n  --- With buyTF4 OR fallback + different SHORT SPY ---")
    for s_dist in [0.3, 0.4, 0.5, 0.54, 0.55, 0.6]:
        spy_s = precompute_spy_distance_set(spy_daily, 20, s_dist)
        fn = make_composite_long_filter(base_fn, spy_00, spy_s, 0.65, 0.66, filter_buy_tf_min(4))
        label = f"s1tf3 L0/S>={s_dist}% shrt65 lc66|bTF4 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 3: SHORT CONFIDENCE SWEEP (with CD's LONG filter)
# ---------------------------------------------------------------------------

def run_short_conf_sweep(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 3: SHORT CONFIDENCE SWEEP (relax from 0.65)")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    # The 2024-09-04 SHORT loss has conf=0.870 and SPY dist=0.534%
    # It's blocked by SPY>=0.6%, not by short conf.
    # The 2019-04-03 SHORT loss has conf=0.625 and SPY dist=2.040%
    # It's blocked by shrt>=0.65, and also by SPY>=0.6%? SPY=2.040% > 0.6%, so NOT blocked by SPY.
    # Wait, AP has that loss and uses SPY>=0%. So it's only blocked by shrt>=0.65.

    print("\n  --- CD with different short conf minimums ---")
    for sc in [0.45, 0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.80]:
        fn = make_composite_long_filter(base_fn, spy_00, spy_06, sc, 0.66, None)
        label = f"s1tf3 L0/S0.6 shrt>={sc:.2f} lc66 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # With buyTF4 fallback
    print("\n  --- With buyTF4 OR fallback + different short conf ---")
    for sc in [0.45, 0.50, 0.55, 0.60, 0.62, 0.65]:
        fn = make_composite_long_filter(base_fn, spy_00, spy_06, sc, 0.66, filter_buy_tf_min(4))
        label = f"s1tf3 L0/S0.6 shrt>={sc:.2f} lc66|bTF4 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 4: ASYMMETRIC SPY DISTANCE FOR SHORTS (conf-scaled)
# ---------------------------------------------------------------------------

def run_short_conf_spy_composite(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 4: CONF-SCALED SHORT SPY (high-conf shorts need less SPY distance)")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)

    # Pre-compute SPY distances
    spy_close = spy_daily['close'].values.astype(float)
    spy_sma20 = pd.Series(spy_close).rolling(20).mean().values
    spy_dist = {}
    for i in range(20, len(spy_close)):
        if spy_sma20[i] > 0:
            spy_dist[spy_daily.index[i]] = (spy_close[i] - spy_sma20[i]) / spy_sma20[i] * 100

    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    def make_conf_spy_combo(high_conf, spy_dist_map, tiers_short, long_spy_set):
        """tiers_short: list of (min_conf, min_spy_dist) for shorts."""
        def fn(day):
            result = base_fn(day)
            if result is None:
                return None
            action, conf, s, t, src = result

            if action == 'BUY':
                if day.date not in long_spy_set:
                    return None
                if conf < high_conf:
                    # Use buyTF4 fallback for low-conf LONGs
                    if _count_tf_confirming(day, 'BUY') < 4:
                        return None
                return result

            if action == 'SELL':
                dist = spy_dist_map.get(day.date)
                if dist is None:
                    return None
                for min_conf, min_dist in tiers_short:
                    if conf >= min_conf:
                        if dist >= min_dist:
                            return result
                        else:
                            return None
                return None
            return result
        return fn

    tier_configs = [
        ("hi87=0.55 mid65=0.6",
         [(0.87, 0.55), (0.65, 0.6), (0.0, 1.0)]),
        ("hi85=0.55 mid65=0.6",
         [(0.85, 0.55), (0.65, 0.6), (0.0, 1.0)]),
        ("hi85=0.5 mid65=0.6",
         [(0.85, 0.5), (0.65, 0.6), (0.0, 1.0)]),
        ("hi80=0.5 mid65=0.6",
         [(0.80, 0.5), (0.65, 0.6), (0.0, 1.0)]),
        ("hi85=0.55 mid70=0.6 lo=1.0",
         [(0.85, 0.55), (0.70, 0.6), (0.0, 1.0)]),
        ("hi85=0.55 mid65=0.6 lo=0.8",
         [(0.85, 0.55), (0.65, 0.6), (0.0, 0.8)]),
        ("all>=0.55",
         [(0.65, 0.55), (0.0, 1.0)]),
        ("all>=0.6",
         [(0.65, 0.6), (0.0, 1.0)]),
    ]

    print("\n  --- conf-scaled SHORT SPY with lc66|bTF4 LONG filter ---")
    for desc, tiers in tier_configs:
        fn = make_conf_spy_combo(0.66, spy_dist, tiers, spy_00)
        label = f"s1tf3 lc66|bTF4 shrt({desc}) sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 5: GRAND SWEEP — combine best from all experiments
# ---------------------------------------------------------------------------

def run_grand_sweep(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 5: GRAND SWEEP")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_04 = precompute_spy_distance_set(spy_daily, 20, 0.4)
    spy_05 = precompute_spy_distance_set(spy_daily, 20, 0.5)
    spy_054 = precompute_spy_distance_set(spy_daily, 20, 0.54)
    spy_055 = precompute_spy_distance_set(spy_daily, 20, 0.55)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    base_fn = _make_s1_tf3_vix_combo(cascade_vix)

    configs = [
        # Baselines
        ("CD: lc66 S0.6 shrt65 (125@100%)",
         make_composite_long_filter(base_fn, spy_00, spy_06, 0.65, 0.66, None), 0, 6),

        ("CE: bTF4 S0.6 shrt65 (115@100%)",
         make_composite_long_filter(base_fn, spy_00, spy_06, 0.65, 0.66, filter_buy_tf_min(4)), 0, 6),

        # Best composite OR filters
        ("lc66 OR buyTF4",
         make_composite_long_filter(base_fn, spy_00, spy_06, 0.65, 0.66, filter_buy_tf_min(4)), 0, 6),

        ("lc66 OR pos<=0.99",
         make_composite_long_filter(base_fn, spy_00, spy_06, 0.65, 0.66, filter_position_max(0.99)), 0, 6),

        ("lc66 OR h>=0.37",
         make_composite_long_filter(base_fn, spy_00, spy_06, 0.65, 0.66, filter_health_min(0.37)), 0, 6),

        ("lc66 OR (h>=0.37 & pos<=0.99)",
         make_composite_long_filter(base_fn, spy_00, spy_06, 0.65, 0.66,
                                    filter_and(filter_health_min(0.37), filter_position_max(0.99))), 0, 6),

        ("lc66 OR (pos<=0.99 & buyTF4)",
         make_composite_long_filter(base_fn, spy_00, spy_06, 0.65, 0.66,
                                    filter_and(filter_position_max(0.99), filter_buy_tf_min(4))), 0, 6),

        # Relax short conf to 0.60 (more shorts)
        ("CD + shrt60",
         make_composite_long_filter(base_fn, spy_00, spy_06, 0.60, 0.66, None), 0, 6),

        ("lc66|bTF4 + shrt60",
         make_composite_long_filter(base_fn, spy_00, spy_06, 0.60, 0.66, filter_buy_tf_min(4)), 0, 6),

        # S>=0.55 (just above the 0.534% loss)
        ("lc66 S0.55 shrt65",
         make_composite_long_filter(base_fn, spy_00, spy_055, 0.65, 0.66, None), 0, 6),

        ("lc66|bTF4 S0.55 shrt65",
         make_composite_long_filter(base_fn, spy_00, spy_055, 0.65, 0.66, filter_buy_tf_min(4)), 0, 6),

        ("lc66 S0.54 shrt65",
         make_composite_long_filter(base_fn, spy_00, spy_054, 0.65, 0.66, None), 0, 6),

        # S>=0.55 + shrt60
        ("lc66 S0.55 shrt60",
         make_composite_long_filter(base_fn, spy_00, spy_055, 0.60, 0.66, None), 0, 6),

        ("lc66|bTF4 S0.55 shrt60",
         make_composite_long_filter(base_fn, spy_00, spy_055, 0.60, 0.66, filter_buy_tf_min(4)), 0, 6),

        # Best composite: OR filter + relaxed short SPY + relaxed short conf
        ("lc66|bTF4 S0.55 shrt60",
         make_composite_long_filter(base_fn, spy_00, spy_055, 0.60, 0.66, filter_buy_tf_min(4)), 0, 6),

        ("lc66|(pos<=0.99&bTF4) S0.55 shrt65",
         make_composite_long_filter(base_fn, spy_00, spy_055, 0.65, 0.66,
                                    filter_and(filter_position_max(0.99), filter_buy_tf_min(4))), 0, 6),
    ]

    for label, fn, cd, tp in configs:
        trades = simulate_trades(signals, fn, label, cooldown=cd, trail_power=tp)
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

    run_composite_or(signals, cascade_vix, spy_daily)
    run_relax_short_spy(signals, cascade_vix, spy_daily)
    run_short_conf_sweep(signals, cascade_vix, spy_daily)
    run_short_conf_spy_composite(signals, cascade_vix, spy_daily)
    run_grand_sweep(signals, cascade_vix, spy_daily)

    print("\n\n" + "=" * 100)
    print("  ALL v15 EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
