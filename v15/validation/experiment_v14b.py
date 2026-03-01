#!/usr/bin/env python3
"""
V14b: Targeted experiments to eliminate the 2018-01-24 LONG loss from
the 148-trade 99.3% WR combo (s1tf3/TF3+VIX L>=0% S>=0.6% shrt>=0.65 sex).

The loss characteristics:
  - Date: 2018-01-24 (signal: 2018-01-23)
  - Direction: LONG
  - Confidence: 0.651 (low-ish)
  - Channel health: 0.364 (below median 0.334... wait above actually)
  - Position score: 1.000 (at TOP of channel -- terrible for a buy!)
  - Timing score: 0.000 (no timing support)
  - Primary TF: 5min (shortest)
  - Signal type: break
  - TF states: 1h:-T, 4h:+, 5min:+, daily:-, weekly:+T (3 confirming, 1h turning)

Target filters (any ONE should kill this loss):
  1. LONG confidence >= 0.66 (conf 0.651 < 0.66)
  2. channel_health >= 0.37 (health 0.364 < 0.37)
  3. position_score < 0.99 for BUY (pos 1.000 >= 0.99 = buying at top = bad)
  4. timing_score > 0 for entry (timing 0.000 = no timing support)
  5. Skip 5min-primary signals
  6. Minimum confluence > 0.50 for BUY (confluence 0.875 -- won't help)
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
    _make_tf4_vix_combo,
)


def _summary_line(trades, name=''):
    n = len(trades)
    if n == 0:
        return f"  {name:<70} {'0':>5}  {'---':>5}  {'---':>10}  {'---':>5}  {'---':>8}"
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
    return (f"  {name:<70} {n:>5} {wr:>5.1f}% ${total:>+9,.0f}  Sh={sharpe:>5.1f}  "
            f"DD={mdd:>4.1f}%  BL=${big_l:>+8,.0f}  Tr={tr_wr:.0f}% Ts={ts_wr:.0f}%")


def _make_tf3_vix_combo(cascade_vix):
    """TF3+VIX (no streak)."""
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


def wrap_extended(base_fn, long_spy_set=None, short_spy_set=None,
                  spy_set=None, short_min_conf=None,
                  long_min_conf=None, min_channel_health=None,
                  max_buy_position=None, min_timing=None,
                  skip_5min_primary=False, min_buy_confluence=None,
                  long_min_health=None, buy_min_confirming=None):
    """Extended filter wrapper with all v14b targeted filters."""
    def fn(day):
        result = base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        # SPY filters
        if spy_set is not None and day.date not in spy_set:
            return None
        if long_spy_set is not None and action == 'BUY' and day.date not in long_spy_set:
            return None
        if short_spy_set is not None and action == 'SELL' and day.date not in short_spy_set:
            return None

        # Confidence filters
        if short_min_conf is not None and action == 'SELL' and conf < short_min_conf:
            return None
        if long_min_conf is not None and action == 'BUY' and conf < long_min_conf:
            return None

        # Channel quality
        if min_channel_health is not None and day.cs_channel_health < min_channel_health:
            return None
        if long_min_health is not None and action == 'BUY' and day.cs_channel_health < long_min_health:
            return None

        # Position score filter: for BUY, skip if at top of channel
        if max_buy_position is not None and action == 'BUY':
            if day.cs_position_score > max_buy_position:
                return None

        # Timing score filter
        if min_timing is not None and day.cs_timing_score < min_timing:
            return None

        # Skip 5min-primary signals
        if skip_5min_primary and day.cs_primary_tf == '5min':
            return None

        # Confluence filter for BUY
        if min_buy_confluence is not None and action == 'BUY':
            if day.cs_confluence_score < min_buy_confluence:
                return None

        # Minimum confirming TFs for BUY (stricter than base combo)
        if buy_min_confirming is not None and action == 'BUY':
            n_tf = _count_tf_confirming(day, 'BUY')
            if n_tf < buy_min_confirming:
                return None

        return result
    return fn


# ---------------------------------------------------------------------------
# EXPERIMENT 1: LONG confidence minimum sweep
# ---------------------------------------------------------------------------

def run_long_conf_sweep(signals, cascade_vix, spy_daily):
    print("=" * 100)
    print("  EXP 1: LONG CONFIDENCE MINIMUM (target: conf 0.651 loss)")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)

    # Test on s1tf3+VIX L0/S0.6 shrt65 sex (base: 148 trades, 99.3%)
    print("\n  --- s1tf3+VIX L0/S0.6 shrt65 + LONG conf minimum ---")
    for lc in [0.45, 0.55, 0.60, 0.65, 0.66, 0.67, 0.70, 0.75]:
        fn = wrap_extended(_make_s1_tf3_vix_combo(cascade_vix),
                           long_spy_set=spy_00, short_spy_set=spy_06,
                           short_min_conf=0.65, long_min_conf=lc)
        label = f"s1tf3 L0/S0.6 shrt65 longconf>={lc:.2f} sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # Same on TF3+VIX
    print("\n  --- TF3+VIX L0/S0.6 shrt65 + LONG conf minimum ---")
    for lc in [0.45, 0.55, 0.60, 0.65, 0.66, 0.67, 0.70, 0.75]:
        fn = wrap_extended(_make_tf3_vix_combo(cascade_vix),
                           long_spy_set=spy_00, short_spy_set=spy_06,
                           short_min_conf=0.65, long_min_conf=lc)
        label = f"TF3+VIX L0/S0.6 shrt65 longconf>={lc:.2f} sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 2: Channel health fine sweep
# ---------------------------------------------------------------------------

def run_health_fine_sweep(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 2: CHANNEL HEALTH FINE SWEEP (target: health 0.364 loss)")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)

    # Global health sweep on s1tf3
    print("\n  --- s1tf3+VIX L0/S0.6 shrt65 + global health ---")
    for h in [0.25, 0.30, 0.33, 0.35, 0.37, 0.40, 0.42, 0.45]:
        fn = wrap_extended(_make_s1_tf3_vix_combo(cascade_vix),
                           long_spy_set=spy_00, short_spy_set=spy_06,
                           short_min_conf=0.65, min_channel_health=h)
        label = f"s1tf3 L0/S0.6 shrt65 health>={h:.2f} sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # LONG-only health (apply health filter only to BUY signals)
    print("\n  --- s1tf3+VIX L0/S0.6 shrt65 + LONG-only health ---")
    for h in [0.25, 0.30, 0.33, 0.35, 0.37, 0.40, 0.42, 0.45]:
        fn = wrap_extended(_make_s1_tf3_vix_combo(cascade_vix),
                           long_spy_set=spy_00, short_spy_set=spy_06,
                           short_min_conf=0.65, long_min_health=h)
        label = f"s1tf3 L0/S0.6 shrt65 longH>={h:.2f} sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 3: Position score filter (BUY at top of channel = bad)
# ---------------------------------------------------------------------------

def run_position_filter(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 3: POSITION SCORE FILTER (target: pos=1.000 BUY = at top of channel)")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)

    # First: distribution of position scores for BUY signals in winning trades
    fn_base = wrap_extended(_make_s1_tf3_vix_combo(cascade_vix),
                            long_spy_set=spy_00, short_spy_set=spy_06,
                            short_min_conf=0.65)
    trades_base = simulate_trades(signals, fn_base, "base", cooldown=0, trail_power=6)

    # Get signal days for each trade
    sig_by_date = {s.date: s for s in signals}
    buy_positions = []
    for t in trades_base:
        # Signal fires day before entry
        for idx, s in enumerate(signals):
            if idx + 1 < len(signals) and signals[idx + 1].date == t.entry_date:
                if t.direction == 'LONG':
                    buy_positions.append((s.cs_position_score, t.pnl))
                break

    if buy_positions:
        pos_vals = [p[0] for p in buy_positions]
        win_pos = [p[0] for p in buy_positions if p[1] > 0]
        loss_pos = [p[0] for p in buy_positions if p[1] <= 0]
        print(f"\n  BUY position scores: mean={np.mean(pos_vals):.3f}, "
              f"p25={np.percentile(pos_vals, 25):.3f}, p50={np.percentile(pos_vals, 50):.3f}, "
              f"p75={np.percentile(pos_vals, 75):.3f}")
        if win_pos:
            print(f"  Winners: mean={np.mean(win_pos):.3f}, min={min(win_pos):.3f}, max={max(win_pos):.3f}")
        if loss_pos:
            print(f"  Losers:  mean={np.mean(loss_pos):.3f}, min={min(loss_pos):.3f}, max={max(loss_pos):.3f}")
        # Count how many winners have position > 0.99
        n_above_99 = sum(1 for p in buy_positions if p[0] > 0.99 and p[1] > 0)
        n_below_99 = sum(1 for p in buy_positions if p[0] <= 0.99 and p[1] > 0)
        print(f"  Winners with pos > 0.99: {n_above_99}, with pos <= 0.99: {n_below_99}")

    # Position score sweep for BUY
    print("\n  --- s1tf3+VIX L0/S0.6 shrt65 + max BUY position ---")
    for max_pos in [0.90, 0.95, 0.98, 0.99, 0.999, 1.001]:
        fn = wrap_extended(_make_s1_tf3_vix_combo(cascade_vix),
                           long_spy_set=spy_00, short_spy_set=spy_06,
                           short_min_conf=0.65, max_buy_position=max_pos)
        label = f"s1tf3 L0/S0.6 shrt65 buyPos<={max_pos:.3f} sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # Same for TF3+VIX
    print("\n  --- TF3+VIX L0/S0.6 shrt65 + max BUY position ---")
    for max_pos in [0.90, 0.95, 0.98, 0.99, 0.999, 1.001]:
        fn = wrap_extended(_make_tf3_vix_combo(cascade_vix),
                           long_spy_set=spy_00, short_spy_set=spy_06,
                           short_min_conf=0.65, max_buy_position=max_pos)
        label = f"TF3+VIX L0/S0.6 shrt65 buyPos<={max_pos:.3f} sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 4: Skip 5min-primary signals
# ---------------------------------------------------------------------------

def run_skip_5min_primary(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 4: SKIP 5min PRIMARY TF SIGNALS (target: primary_tf=5min)")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)

    # Count how many signals have 5min as primary
    cs_signals = [s for s in signals if s.cs_action in ('BUY', 'SELL') and s.cs_confidence >= MIN_SIGNAL_CONFIDENCE]
    n_5min = sum(1 for s in cs_signals if s.cs_primary_tf == '5min')
    print(f"\n  CS signals with 5min primary: {n_5min}/{len(cs_signals)} ({n_5min/max(len(cs_signals),1)*100:.1f}%)")

    # Distribution of primary TFs
    from collections import Counter
    tf_counts = Counter(s.cs_primary_tf for s in cs_signals)
    for tf, cnt in tf_counts.most_common():
        print(f"    {tf}: {cnt} ({cnt/len(cs_signals)*100:.1f}%)")

    # With and without 5min-primary filter
    print("\n  --- s1tf3+VIX L0/S0.6 shrt65 +/- skip 5min ---")
    fn = wrap_extended(_make_s1_tf3_vix_combo(cascade_vix),
                       long_spy_set=spy_00, short_spy_set=spy_06,
                       short_min_conf=0.65)
    label = "s1tf3 L0/S0.6 shrt65 (baseline) sex"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
    print(_summary_line(trades, label))

    fn = wrap_extended(_make_s1_tf3_vix_combo(cascade_vix),
                       long_spy_set=spy_00, short_spy_set=spy_06,
                       short_min_conf=0.65, skip_5min_primary=True)
    label = "s1tf3 L0/S0.6 shrt65 skip5min sex"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
    print(_summary_line(trades, label))

    # CC baseline with skip 5min
    fn = wrap_extended(_make_tf4_vix_base(cascade_vix),
                       long_spy_set=spy_00, short_spy_set=spy_06)
    label = "CC baseline (no skip) cd=0"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
    print(_summary_line(trades, label))

    fn = wrap_extended(_make_tf4_vix_base(cascade_vix),
                       long_spy_set=spy_00, short_spy_set=spy_06,
                       skip_5min_primary=True)
    label = "CC skip5min cd=0"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
    print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 5: BUY-specific TF4 (require 4 TFs for BUY, 3 for SELL)
# ---------------------------------------------------------------------------

def run_buy_tf4(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 5: ASYMMETRIC TF REQUIREMENT (BUY needs TF4, SELL needs TF3)")
    print("=" * 100)
    print("  The loss is a LONG with only 3 TFs. Require TF4 for longs only.")

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)

    # s1tf3+VIX with BUY needing TF4
    print("\n  --- s1tf3+VIX L0/S0.6 shrt65 + BUY needs TF4 ---")
    fn = wrap_extended(_make_s1_tf3_vix_combo(cascade_vix),
                       long_spy_set=spy_00, short_spy_set=spy_06,
                       short_min_conf=0.65, buy_min_confirming=4)
    label = "s1tf3 L0/S0.6 shrt65 buyTF4 sex"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
    print(_summary_line(trades, label))

    fn = wrap_extended(_make_s1_tf3_vix_combo(cascade_vix),
                       long_spy_set=spy_00, short_spy_set=spy_06,
                       short_min_conf=0.65, buy_min_confirming=5)
    label = "s1tf3 L0/S0.6 shrt65 buyTF5 sex"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
    print(_summary_line(trades, label))

    # TF3+VIX with BUY needing TF4
    fn = wrap_extended(_make_tf3_vix_combo(cascade_vix),
                       long_spy_set=spy_00, short_spy_set=spy_06,
                       short_min_conf=0.65, buy_min_confirming=4)
    label = "TF3+VIX L0/S0.6 shrt65 buyTF4 sex"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
    print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 6: COMBINED BEST FILTERS
# ---------------------------------------------------------------------------

def run_combined(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXP 6: COMBINED BEST FILTERS")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_03 = precompute_spy_distance_set(spy_daily, 20, 0.3)
    spy_04 = precompute_spy_distance_set(spy_daily, 20, 0.4)
    spy_05 = precompute_spy_distance_set(spy_daily, 20, 0.5)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)

    configs = [
        # Baselines
        ("CC baseline (106@100%)",
         _make_tf4_vix_base(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06), 0, 4),

        ("s1tf3 L0/S0.6 shrt65 (148@99.3%)",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65), 0, 6),

        # Single filters to kill the 2018-01-24 loss
        ("s1tf3 L0/S0.6 shrt65 longconf>=0.66",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65, long_min_conf=0.66), 0, 6),

        ("s1tf3 L0/S0.6 shrt65 buyTF4",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65, buy_min_confirming=4), 0, 6),

        ("s1tf3 L0/S0.6 shrt65 longH>=0.37",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65, long_min_health=0.37), 0, 6),

        # Same on TF3+VIX
        ("TF3+VIX L0/S0.6 shrt65 longconf>=0.66",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65, long_min_conf=0.66), 0, 6),

        ("TF3+VIX L0/S0.6 shrt65 buyTF4",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65, buy_min_confirming=4), 0, 6),

        ("TF3+VIX L0/S0.6 shrt65 longH>=0.37",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65, long_min_health=0.37), 0, 6),

        # Composite: buyTF4 + shrt65 (the loss has 3 TFs, this blocks it)
        ("s1tf3 L0/S0.6 shrt65 buyTF4 longconf>=0.66",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65,
              buy_min_confirming=4, long_min_conf=0.66), 0, 6),

        # TF3 base with buyTF4 (uses TF3 for shorts but TF4 for longs)
        ("TF3+VIX L0/S0.6 shrt65 buyTF4 longconf>=0.66",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65,
              buy_min_confirming=4, long_min_conf=0.66), 0, 6),

        # Wider hybrid: L0/S0.4 and L0/S0.5 with the killer filter
        ("s1tf3 L0/S0.4 shrt65 buyTF4",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_04, short_min_conf=0.65, buy_min_confirming=4), 0, 6),

        ("s1tf3 L0/S0.5 shrt65 buyTF4",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_05, short_min_conf=0.65, buy_min_confirming=4), 0, 6),

        ("s1tf3 L0/S0.3 shrt65 buyTF4",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_03, short_min_conf=0.65, buy_min_confirming=4), 0, 6),

        ("TF3+VIX L0/S0.4 shrt65 buyTF4",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_04, short_min_conf=0.65, buy_min_confirming=4), 0, 6),

        ("TF3+VIX L0/S0.5 shrt65 buyTF4",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_05, short_min_conf=0.65, buy_min_confirming=4), 0, 6),

        ("TF3+VIX L0/S0.3 shrt65 buyTF4",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_03, short_min_conf=0.65, buy_min_confirming=4), 0, 6),

        # Experiment with lower short SPY thresholds + buyTF4 (more trades)
        ("s1tf3 L0/S0.3 shrt65 buyTF4 longconf>=0.66",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_03, short_min_conf=0.65,
              buy_min_confirming=4, long_min_conf=0.66), 0, 6),

        ("TF3+VIX L0/S0.3 shrt65 buyTF4 longconf>=0.66",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_03, short_min_conf=0.65,
              buy_min_confirming=4, long_min_conf=0.66), 0, 6),

        # Also test with short_min_conf=0.60 (more shorts)
        ("s1tf3 L0/S0.3 shrt60 buyTF4",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_03, short_min_conf=0.60, buy_min_confirming=4), 0, 6),

        ("TF3+VIX L0/S0.3 shrt60 buyTF4",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_03, short_min_conf=0.60, buy_min_confirming=4), 0, 6),

        ("s1tf3 L0/S0.4 shrt60 buyTF4",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_04, short_min_conf=0.60, buy_min_confirming=4), 0, 6),

        ("TF3+VIX L0/S0.4 shrt60 buyTF4",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_04, short_min_conf=0.60, buy_min_confirming=4), 0, 6),

        # No short conf at all (just hybrid SPY + buyTF4)
        ("s1tf3 L0/S0.6 buyTF4",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, buy_min_confirming=4), 0, 6),

        ("s1tf3 L0/S0.3 buyTF4",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_03, buy_min_confirming=4), 0, 6),

        ("TF3+VIX L0/S0.6 buyTF4",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, buy_min_confirming=4), 0, 6),

        ("TF3+VIX L0/S0.3 buyTF4",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_03, buy_min_confirming=4), 0, 6),
    ]

    for label, base_fn, filter_kwargs, cd, tp in configs:
        fn = wrap_extended(base_fn, **filter_kwargs)
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

    run_long_conf_sweep(signals, cascade_vix, spy_daily)
    run_health_fine_sweep(signals, cascade_vix, spy_daily)
    run_position_filter(signals, cascade_vix, spy_daily)
    run_skip_5min_primary(signals, cascade_vix, spy_daily)
    run_buy_tf4(signals, cascade_vix, spy_daily)
    run_combined(signals, cascade_vix, spy_daily)

    print("\n\n" + "=" * 100)
    print("  ALL v14b EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
