#!/usr/bin/env python3
"""
V14 Experiments: Eliminate remaining losses in near-100% combos.

Current standings (top Pareto frontier):
  CC: 106 trades, 100% WR, $211K (TF4+VIX + hybrid SPY L>=0% S>=0.6%, cd=0)
  BY: 134 trades, 99.3% WR, $237K (TF3+VIX+SPY0.6% shrt65 sex) -- 1 loss
  AL: 113 trades, 99.1% WR, $224K (TF4+VIX+SPY cd=0) -- 1 loss
  AZ: 112 trades, 99.1% WR, $209K (s1tf3+VIX+SPY LONG sex) -- 1 loss
  BC: 156 trades, 98.7% WR, $277K (s1tf3+VIX+SPY shrt65 sex) -- 2 losses
  AP: 185 trades, 97.8% WR, $312K (s1tf3+VIX+SPY cd=0 sex) -- 4 losses

Key insight: ALL LONGs with SPY>SMA20 are 100% WR. Losses are from SHORTs.

Strategy: Identify each specific loss, find filters that block it without losing winners.

Experiments:
  1. Loss identification — full details for every loss in near-100% combos
  2. s1_tf3 + hybrid SPY fine sweep (many L/S threshold combos)
  3. Channel health/energy/position score minimum filters
  4. TF turning filter — skip when any TF is turning against trade
  5. Confidence-adaptive hybrid SPY — higher confidence shorts need less SPY distance
  6. s1_tf3 + hybrid SPY + short conf boost (triple filter)
  7. Grand combo sweep — best filters combined
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
        return f"  {name:<65} {'0':>5}  {'---':>5}  {'---':>10}  {'---':>5}  {'---':>8}"
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
    return (f"  {name:<65} {n:>5} {wr:>5.1f}% ${total:>+9,.0f}  Sh={sharpe:>5.1f}  "
            f"DD={mdd:>4.1f}%  BL=${big_l:>+8,.0f}  Tr={tr_wr:.0f}% Ts={ts_wr:.0f}%")


# ---------------------------------------------------------------------------
# Combo factories (reused from v13)
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


def precompute_tsla_sma(signals, window=20):
    closes = np.array([s.day_close for s in signals], dtype=float)
    sma = pd.Series(closes).rolling(window).mean().values
    above = set()
    for i in range(window, len(signals)):
        if closes[i] > sma[i]:
            above.add(signals[i].date)
    return above


def precompute_tsla_atr_pct(signals, window=14):
    """Compute rolling ATR as % of close price."""
    highs = np.array([s.day_high for s in signals], dtype=float)
    lows = np.array([s.day_low for s in signals], dtype=float)
    closes = np.array([s.day_close for s in signals], dtype=float)

    tr = np.zeros(len(signals))
    for i in range(1, len(signals)):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1]))
    atr = pd.Series(tr).rolling(window).mean().values
    atr_pct = {}
    for i in range(window, len(signals)):
        if closes[i] > 0:
            atr_pct[signals[i].date] = atr[i] / closes[i] * 100
    return atr_pct


def wrap_with_filters(base_fn, spy_set=None, long_only=False, short_min_conf=None,
                      min_confidence=None, long_spy_set=None, short_spy_set=None,
                      min_channel_health=None, min_energy=None, min_position=None,
                      no_turning_against=False, atr_range=None, atr_pct_map=None):
    """Extended filter wrapper with channel quality and ATR regime filters."""
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
        if long_spy_set is not None and action == 'BUY' and day.date not in long_spy_set:
            return None
        if short_spy_set is not None and action == 'SELL' and day.date not in short_spy_set:
            return None

        # Channel quality filters
        if min_channel_health is not None and day.cs_channel_health < min_channel_health:
            return None
        if min_energy is not None and day.cs_energy_score < min_energy:
            return None
        if min_position is not None:
            # For BUY: want low position (near bottom). For SELL: want high position (near top).
            if action == 'BUY' and day.cs_position_score > (1.0 - min_position):
                return None  # Too high for a buy
            if action == 'SELL' and day.cs_position_score < min_position:
                return None  # Too low for a sell

        # TF turning filter: skip if any TF momentum is turning against the trade
        if no_turning_against and day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    continue
                is_turning = state.get('momentum_is_turning', False)
                md = state.get('momentum_direction', 0.0)
                if is_turning:
                    # Turning: momentum is reversing. Bad if reversing against our direction.
                    if action == 'BUY' and md > 0:
                        # Momentum was positive but is turning (going negative) = bad for long
                        return None
                    if action == 'SELL' and md < 0:
                        # Momentum was negative but is turning (going positive) = bad for short
                        return None

        # ATR regime filter
        if atr_range is not None and atr_pct_map is not None:
            atr_val = atr_pct_map.get(day.date)
            if atr_val is not None:
                low, high = atr_range
                if atr_val < low or atr_val > high:
                    return None

        return result
    return fn


# ---------------------------------------------------------------------------
# EXPERIMENT 1: LOSS IDENTIFICATION
# ---------------------------------------------------------------------------

def run_loss_identification(signals, cascade_vix, spy_daily):
    print("=" * 100)
    print("  EXPERIMENT 1: LOSS IDENTIFICATION — every loss in near-100% combos")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)

    combos_to_analyze = [
        ("AL: TF4VIX+SPY cd=0",
         wrap_with_filters(_make_tf4_vix_base(cascade_vix), spy_set=spy_00),
         0, 4),
        ("BY: TF3VIX+SPY0.6 shrt65 cd=0 sex",
         wrap_with_filters(_make_tf3_vix_combo(cascade_vix), spy_set=spy_06, short_min_conf=0.65),
         0, 6),
        ("AZ: s1tf3VIX+SPY LONG cd=0 sex",
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy_00, long_only=True),
         0, 6),
        ("BC: s1tf3VIX+SPY shrt65 cd=0 sex",
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy_00, short_min_conf=0.65),
         0, 6),
        ("AP: s1tf3VIX+SPY cd=0 sex",
         wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy_00),
         0, 6),
    ]

    # Also compute SPY distance for each day for context
    spy_close = spy_daily['close'].values.astype(float)
    spy_sma20 = pd.Series(spy_close).rolling(20).mean().values
    spy_dist_by_date = {}
    for i in range(20, len(spy_close)):
        if spy_sma20[i] > 0:
            dist = (spy_close[i] - spy_sma20[i]) / spy_sma20[i] * 100
            spy_dist_by_date[spy_daily.index[i]] = dist

    # Build signal lookup for channel health/energy analysis
    sig_by_date = {s.date: s for s in signals}

    for label, fn, cd, tp in combos_to_analyze:
        trades = simulate_trades(signals, fn, label, cooldown=cd, trail_power=tp)
        losses = [t for t in trades if t.pnl <= 0]
        wins = [t for t in trades if t.pnl > 0]
        print(f"\n  {label}: {len(trades)} trades, {len(wins)} wins, {len(losses)} losses")
        print(f"  Total PnL: ${sum(t.pnl for t in trades):+,.0f}")

        if not losses:
            print("  *** 100% WIN RATE — no losses! ***")
            continue

        for loss in losses:
            print(f"\n    LOSS: {loss.entry_date.date()} -> {loss.exit_date.date()}")
            print(f"      Direction: {loss.direction}")
            print(f"      Entry: ${loss.entry_price:.2f} -> Exit: ${loss.exit_price:.2f}")
            print(f"      PnL: ${loss.pnl:+,.2f}")
            print(f"      Confidence: {loss.confidence:.3f}")
            print(f"      Hold days: {loss.hold_days}")
            print(f"      Exit reason: {loss.exit_reason}")
            print(f"      Source: {loss.source}")

            # Look up signal day context (signal day = day before entry)
            entry_idx = None
            for idx, s in enumerate(signals):
                if s.date == loss.entry_date:
                    entry_idx = idx
                    break
            if entry_idx is not None and entry_idx > 0:
                sig_day = signals[entry_idx - 1]  # Signal fires day before entry
                spy_dist = spy_dist_by_date.get(sig_day.date, None)
                print(f"      Signal day: {sig_day.date.date()}")
                print(f"      SPY dist from SMA20: {spy_dist:.3f}%" if spy_dist is not None else "      SPY dist: N/A")
                print(f"      Channel health: {sig_day.cs_channel_health:.3f}")
                print(f"      Energy score: {sig_day.cs_energy_score:.3f}")
                print(f"      Entropy score: {sig_day.cs_entropy_score:.3f}")
                print(f"      Confluence score: {sig_day.cs_confluence_score:.3f}")
                print(f"      Timing score: {sig_day.cs_timing_score:.3f}")
                print(f"      Position score: {sig_day.cs_position_score:.3f}")
                print(f"      CS action: {sig_day.cs_action}, CS conf: {sig_day.cs_confidence:.3f}")
                print(f"      Primary TF: {sig_day.cs_primary_tf}")
                print(f"      Signal type: {sig_day.cs_signal_type}")

                # TF momentum states
                if sig_day.cs_tf_states:
                    tf_info = []
                    for tf, state in sorted(sig_day.cs_tf_states.items()):
                        if state.get('valid', False):
                            md = state.get('momentum_direction', 0)
                            turn = state.get('momentum_is_turning', False)
                            dir_sym = '+' if md > 0 else ('-' if md < 0 else '0')
                            turn_sym = 'T' if turn else ''
                            tf_info.append(f"{tf}:{dir_sym}{turn_sym}")
                    print(f"      TF states: {', '.join(tf_info)}")

                # Count confirming TFs
                n_tf = _count_tf_confirming(sig_day, sig_day.cs_action)
                print(f"      Confirming TFs: {n_tf}")


# ---------------------------------------------------------------------------
# EXPERIMENT 2: s1_tf3 + HYBRID SPY FINE SWEEP
# ---------------------------------------------------------------------------

def run_s1tf3_hybrid_spy(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 2: s1_tf3 + HYBRID SPY FINE SWEEP")
    print("=" * 100)
    print("  Apply direction-specific SPY thresholds to s1_tf3 base (more trades)")

    # Pre-compute SPY distance sets at fine granularity
    thresholds = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    spy_sets = {t: precompute_spy_distance_set(spy_daily, 20, t) for t in thresholds}

    # Uniform SPY baselines
    print("\n  --- s1_tf3+VIX + uniform SPY cd=0 sex ---")
    for dist in thresholds:
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy_sets[dist])
        label = f"s1tf3+VIX SPY>={dist}% cd=0 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # Hybrid: LONG>=X%, SHORT>=Y%
    print("\n  --- s1_tf3+VIX + hybrid SPY (L>=X%, S>=Y%) cd=0 sex ---")
    configs = []
    for l_dist in [0.0, 0.2, 0.3]:
        for s_dist in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
            if s_dist > l_dist:
                configs.append((l_dist, s_dist))

    for l_dist, s_dist in configs:
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix),
                               long_spy_set=spy_sets[l_dist], short_spy_set=spy_sets[s_dist])
        label = f"s1tf3+VIX L>={l_dist}% S>={s_dist}% cd=0 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # Hybrid + short conf boost
    print("\n  --- s1_tf3+VIX + hybrid SPY + shrt conf cd=0 sex ---")
    for l_dist, s_dist in [(0.0, 0.3), (0.0, 0.4), (0.0, 0.5), (0.0, 0.6)]:
        for sc in [0.60, 0.65, 0.70, 0.80]:
            fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix),
                                   long_spy_set=spy_sets[l_dist], short_spy_set=spy_sets[s_dist],
                                   short_min_conf=sc)
            label = f"s1tf3 L>={l_dist}% S>={s_dist}% shrt>={sc} sex"
            trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
            print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 3: CHANNEL QUALITY FILTERS
# ---------------------------------------------------------------------------

def run_channel_quality(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 3: CHANNEL QUALITY FILTERS (health, energy, position)")
    print("=" * 100)

    # First, analyze the distribution of channel quality scores
    buy_signals = [s for s in signals if s.cs_action == 'BUY' and s.cs_confidence >= MIN_SIGNAL_CONFIDENCE]
    sell_signals = [s for s in signals if s.cs_action == 'SELL' and s.cs_confidence >= MIN_SIGNAL_CONFIDENCE]
    all_cs = buy_signals + sell_signals

    if all_cs:
        healths = [s.cs_channel_health for s in all_cs]
        energies = [s.cs_energy_score for s in all_cs]
        positions = [s.cs_position_score for s in all_cs]
        print(f"\n  Channel Health: mean={np.mean(healths):.3f}, "
              f"p10={np.percentile(healths, 10):.3f}, p25={np.percentile(healths, 25):.3f}, "
              f"p50={np.percentile(healths, 50):.3f}, p75={np.percentile(healths, 75):.3f}")
        print(f"  Energy Score:   mean={np.mean(energies):.3f}, "
              f"p10={np.percentile(energies, 10):.3f}, p25={np.percentile(energies, 25):.3f}, "
              f"p50={np.percentile(energies, 50):.3f}, p75={np.percentile(energies, 75):.3f}")
        print(f"  Position Score: mean={np.mean(positions):.3f}, "
              f"p10={np.percentile(positions, 10):.3f}, p25={np.percentile(positions, 25):.3f}, "
              f"p50={np.percentile(positions, 50):.3f}, p75={np.percentile(positions, 75):.3f}")

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)

    # Channel health sweep on CC base (TF4+VIX+hybrid SPY)
    print("\n  --- Channel health minimum on CC base (TF4+VIX hybrid SPY cd=0) ---")
    for min_h in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix),
                               long_spy_set=spy_00, short_spy_set=spy_06,
                               min_channel_health=min_h)
        label = f"CC+health>={min_h:.1f} cd=0"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
        print(_summary_line(trades, label))

    # Channel health on s1_tf3+VIX+SPY
    print("\n  --- Channel health on s1_tf3+VIX+SPY cd=0 sex ---")
    for min_h in [0.0, 0.2, 0.3, 0.4, 0.5]:
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix),
                               spy_set=spy_00, min_channel_health=min_h)
        label = f"s1tf3+VIX+SPY+health>={min_h:.1f} cd=0 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # Energy score sweep
    print("\n  --- Energy score minimum ---")
    for min_e in [0.0, 0.2, 0.3, 0.4, 0.5]:
        fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix),
                               long_spy_set=spy_00, short_spy_set=spy_06,
                               min_energy=min_e)
        label = f"CC+energy>={min_e:.1f} cd=0"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
        print(_summary_line(trades, label))

    # Combined health + energy
    print("\n  --- Health + Energy combined ---")
    for min_h, min_e in [(0.3, 0.3), (0.4, 0.3), (0.3, 0.4), (0.4, 0.4)]:
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix),
                               spy_set=spy_00, min_channel_health=min_h, min_energy=min_e)
        label = f"s1tf3+VIX+SPY h>={min_h} e>={min_e} sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 4: TF TURNING FILTER
# ---------------------------------------------------------------------------

def run_tf_turning(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 4: TF TURNING FILTER")
    print("=" * 100)
    print("  Skip signals where ANY TF momentum is turning against the trade direction")

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)

    # First, check how often turning occurs in the signal set
    turn_count = 0
    total_cs = 0
    for s in signals:
        if s.cs_action in ('BUY', 'SELL') and s.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            total_cs += 1
            if s.cs_tf_states:
                for tf, state in s.cs_tf_states.items():
                    if state.get('valid', False) and state.get('momentum_is_turning', False):
                        md = state.get('momentum_direction', 0)
                        if (s.cs_action == 'BUY' and md > 0) or (s.cs_action == 'SELL' and md < 0):
                            turn_count += 1
                            break
    print(f"\n  CS signals with at-least-1-TF turning against: {turn_count}/{total_cs} ({turn_count/max(total_cs,1)*100:.1f}%)")

    # CC baseline vs CC + no-turning
    print("\n  --- CC baseline vs CC+no-turning ---")
    fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix),
                           long_spy_set=spy_00, short_spy_set=spy_06)
    label = "CC baseline cd=0"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
    print(_summary_line(trades, label))

    fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix),
                           long_spy_set=spy_00, short_spy_set=spy_06,
                           no_turning_against=True)
    label = "CC+no_turn cd=0"
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
    print(_summary_line(trades, label))

    # s1_tf3+VIX+SPY with and without turning filter
    print("\n  --- s1_tf3+VIX+SPY with turning filter ---")
    for spy_set, spy_label in [(spy_00, "SPY>0%"), (spy_06, "SPY>0.6%")]:
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy_set)
        label = f"s1tf3+VIX+{spy_label} cd=0 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix), spy_set=spy_set,
                               no_turning_against=True)
        label = f"s1tf3+VIX+{spy_label}+no_turn cd=0 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

    # TF3+VIX + SPY + turning filter
    print("\n  --- TF3+VIX + SPY + turning filter ---")
    for spy_set, spy_label in [(spy_06, "SPY>0.6%")]:
        fn = wrap_with_filters(_make_tf3_vix_combo(cascade_vix), spy_set=spy_set,
                               short_min_conf=0.65)
        label = f"TF3+VIX+{spy_label}+shrt65 cd=0 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))

        fn = wrap_with_filters(_make_tf3_vix_combo(cascade_vix), spy_set=spy_set,
                               short_min_conf=0.65, no_turning_against=True)
        label = f"TF3+VIX+{spy_label}+shrt65+no_turn sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 5: ATR VOLATILITY REGIME
# ---------------------------------------------------------------------------

def run_atr_regime(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 5: TSLA ATR VOLATILITY REGIME")
    print("=" * 100)
    print("  Only trade when TSLA ATR% is in optimal range (not too volatile)")

    atr_pct_map = precompute_tsla_atr_pct(signals, window=14)
    if atr_pct_map:
        vals = list(atr_pct_map.values())
        print(f"\n  TSLA ATR%: mean={np.mean(vals):.2f}%, "
              f"p10={np.percentile(vals, 10):.2f}%, p25={np.percentile(vals, 25):.2f}%, "
              f"p50={np.percentile(vals, 50):.2f}%, p75={np.percentile(vals, 75):.2f}%, "
              f"p90={np.percentile(vals, 90):.2f}%")

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)

    # CC + ATR range filter
    print("\n  --- CC + ATR range filter ---")
    for low, high in [(0, 3), (0, 4), (0, 5), (0, 6), (1, 4), (1, 5), (2, 5), (2, 6)]:
        fn = wrap_with_filters(_make_tf4_vix_base(cascade_vix),
                               long_spy_set=spy_00, short_spy_set=spy_06,
                               atr_range=(low, high), atr_pct_map=atr_pct_map)
        label = f"CC+ATR[{low}-{high}%] cd=0"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
        print(_summary_line(trades, label))

    # s1_tf3+VIX+SPY + ATR
    print("\n  --- s1_tf3+VIX+SPY + ATR ---")
    for low, high in [(0, 4), (0, 5), (1, 5), (1, 6)]:
        fn = wrap_with_filters(_make_s1_tf3_vix_combo(cascade_vix),
                               spy_set=spy_00,
                               atr_range=(low, high), atr_pct_map=atr_pct_map)
        label = f"s1tf3+VIX+SPY+ATR[{low}-{high}%] cd=0 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 6: TF3+VIX HYBRID SPY + SHORT CONF SWEEP
# ---------------------------------------------------------------------------

def run_tf3_hybrid_sweep(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 6: TF3+VIX + HYBRID SPY + SHORT CONF SWEEP")
    print("=" * 100)
    print("  BY (TF3+VIX+SPY0.6%+shrt65) has 134 trades at 99.3%. Push to 100%?")

    thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    spy_sets = {t: precompute_spy_distance_set(spy_daily, 20, t) for t in thresholds}

    # Fine sweep: hybrid SPY on TF3+VIX
    print("\n  --- TF3+VIX + hybrid SPY (L>=X%, S>=Y%) + shrt conf cd=0 sex ---")
    for l_dist in [0.0, 0.3]:
        for s_dist in [0.4, 0.5, 0.6, 0.8, 1.0]:
            if s_dist <= l_dist:
                continue
            for sc in [0.60, 0.65, 0.70, 0.80]:
                fn = wrap_with_filters(_make_tf3_vix_combo(cascade_vix),
                                       long_spy_set=spy_sets[l_dist], short_spy_set=spy_sets[s_dist],
                                       short_min_conf=sc)
                label = f"TF3+VIX L>={l_dist}% S>={s_dist}% shrt>={sc} sex"
                trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
                wr = sum(1 for t in trades if t.pnl > 0) / max(len(trades), 1) * 100
                # Only print if interesting (100% WR or lots of trades)
                if wr >= 99.0 or len(trades) >= 120:
                    print(_summary_line(trades, label))

    # Without short conf boost (pure hybrid SPY)
    print("\n  --- TF3+VIX + hybrid SPY only (no short conf) cd=0 sex ---")
    for l_dist in [0.0, 0.3]:
        for s_dist in [0.4, 0.5, 0.6, 0.8, 1.0]:
            if s_dist <= l_dist:
                continue
            fn = wrap_with_filters(_make_tf3_vix_combo(cascade_vix),
                                   long_spy_set=spy_sets[l_dist], short_spy_set=spy_sets[s_dist])
            label = f"TF3+VIX L>={l_dist}% S>={s_dist}% cd=0 sex"
            trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
            wr = sum(1 for t in trades if t.pnl > 0) / max(len(trades), 1) * 100
            if wr >= 99.0 or len(trades) >= 120:
                print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 7: CONFIDENCE-ADAPTIVE SPY DISTANCE
# ---------------------------------------------------------------------------

def run_conf_adaptive_spy(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 7: CONFIDENCE-ADAPTIVE SPY DISTANCE")
    print("=" * 100)
    print("  Idea: high-conf shorts need less SPY distance, low-conf need more")
    print("  conf >= 0.85 -> SPY >= 0.3%, conf >= 0.70 -> SPY >= 0.6%, else -> SPY >= 1.0%")

    # Pre-compute SPY distances
    spy_daily_close = spy_daily['close'].values.astype(float)
    spy_sma20 = pd.Series(spy_daily_close).rolling(20).mean().values
    spy_dist = {}
    for i in range(20, len(spy_daily_close)):
        if spy_sma20[i] > 0:
            spy_dist[spy_daily.index[i]] = (spy_daily_close[i] - spy_sma20[i]) / spy_sma20[i] * 100

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)

    def make_conf_adaptive_spy_fn(base_fn, spy_dist_map, tiers):
        """tiers: list of (min_conf, min_spy_dist) sorted by min_conf descending."""
        def fn(day):
            result = base_fn(day)
            if result is None:
                return None
            action, conf, s_pct, t_pct, src = result
            # LONGs: always pass with SPY > SMA20
            if action == 'BUY':
                if day.date not in spy_00:
                    return None
                return result
            # SHORTs: adaptive SPY distance based on confidence
            dist = spy_dist_map.get(day.date)
            if dist is None:
                return None
            for min_conf, min_dist in tiers:
                if conf >= min_conf:
                    if dist >= min_dist:
                        return result
                    else:
                        return None
            return None  # Below all tiers
        return fn

    # Test various tier configurations
    tier_configs = [
        ("hi85=0.3 mid70=0.6 lo=1.0",
         [(0.85, 0.3), (0.70, 0.6), (0.0, 1.0)]),
        ("hi85=0.4 mid70=0.6 lo=0.8",
         [(0.85, 0.4), (0.70, 0.6), (0.0, 0.8)]),
        ("hi80=0.3 mid65=0.6 lo=1.0",
         [(0.80, 0.3), (0.65, 0.6), (0.0, 1.0)]),
        ("hi80=0.4 mid65=0.6 lo=0.8",
         [(0.80, 0.4), (0.65, 0.6), (0.0, 0.8)]),
        ("hi85=0.0 mid70=0.3 lo=0.6",
         [(0.85, 0.0), (0.70, 0.3), (0.0, 0.6)]),
        ("hi80=0.0 mid65=0.3 lo=0.6",
         [(0.80, 0.0), (0.65, 0.3), (0.0, 0.6)]),
    ]

    # On TF4+VIX base
    print("\n  --- TF4+VIX + conf-adaptive SPY cd=0 ---")
    for desc, tiers in tier_configs:
        fn = make_conf_adaptive_spy_fn(_make_tf4_vix_base(cascade_vix), spy_dist, tiers)
        label = f"TF4+VIX confSPY({desc}) cd=0"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=4)
        print(_summary_line(trades, label))

    # On s1_tf3+VIX base
    print("\n  --- s1_tf3+VIX + conf-adaptive SPY cd=0 sex ---")
    for desc, tiers in tier_configs:
        fn = make_conf_adaptive_spy_fn(_make_s1_tf3_vix_combo(cascade_vix), spy_dist, tiers)
        label = f"s1tf3+VIX confSPY({desc}) cd=0 sex"
        trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
        print(_summary_line(trades, label))


# ---------------------------------------------------------------------------
# EXPERIMENT 8: GRAND SWEEP — combine best discoveries
# ---------------------------------------------------------------------------

def run_grand_sweep(signals, cascade_vix, spy_daily):
    print("\n\n" + "=" * 100)
    print("  EXPERIMENT 8: GRAND SWEEP — combine best filters")
    print("=" * 100)

    spy_00 = precompute_spy_distance_set(spy_daily, 20, 0.0)
    spy_03 = precompute_spy_distance_set(spy_daily, 20, 0.3)
    spy_04 = precompute_spy_distance_set(spy_daily, 20, 0.4)
    spy_05 = precompute_spy_distance_set(spy_daily, 20, 0.5)
    spy_06 = precompute_spy_distance_set(spy_daily, 20, 0.6)
    spy_08 = precompute_spy_distance_set(spy_daily, 20, 0.8)
    spy_10 = precompute_spy_distance_set(spy_daily, 20, 1.0)
    atr_pct_map = precompute_tsla_atr_pct(signals, window=14)

    configs = [
        # CC baseline
        ("CC baseline",
         _make_tf4_vix_base(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06), 0, 4),

        # CC + channel health
        ("CC+health>=0.3",
         _make_tf4_vix_base(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, min_channel_health=0.3), 0, 4),

        # CC + no turning
        ("CC+no_turn",
         _make_tf4_vix_base(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, no_turning_against=True), 0, 4),

        # CC + ATR[0-5]
        ("CC+ATR[0-5]",
         _make_tf4_vix_base(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, atr_range=(0, 5), atr_pct_map=atr_pct_map), 0, 4),

        # s1_tf3+VIX hybrid SPY L0/S0.3 (more aggressive)
        ("s1tf3 L>=0% S>=0.3% sex",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_03), 0, 6),

        # s1_tf3+VIX hybrid SPY L0/S0.4
        ("s1tf3 L>=0% S>=0.4% sex",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_04), 0, 6),

        # s1_tf3+VIX hybrid SPY L0/S0.5
        ("s1tf3 L>=0% S>=0.5% sex",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_05), 0, 6),

        # s1_tf3+VIX hybrid SPY L0/S0.6
        ("s1tf3 L>=0% S>=0.6% sex",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06), 0, 6),

        # s1_tf3+VIX hybrid SPY L0/S0.6 + shrt65
        ("s1tf3 L0/S0.6 shrt65 sex",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65), 0, 6),

        # s1_tf3+VIX hybrid SPY L0/S0.6 + shrt70
        ("s1tf3 L0/S0.6 shrt70 sex",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.70), 0, 6),

        # s1_tf3+VIX hybrid SPY L0/S0.6 + health>=0.3
        ("s1tf3 L0/S0.6 h>=0.3 sex",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, min_channel_health=0.3), 0, 6),

        # s1_tf3+VIX hybrid SPY L0/S0.6 + no_turn
        ("s1tf3 L0/S0.6 no_turn sex",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, no_turning_against=True), 0, 6),

        # s1_tf3+VIX hybrid SPY L0/S0.6 + shrt65 + health
        ("s1tf3 L0/S0.6 shrt65 h>=0.3 sex",
         _make_s1_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65, min_channel_health=0.3), 0, 6),

        # TF3+VIX hybrid SPY combos
        ("TF3+VIX L>=0% S>=0.4% shrt65 sex",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_04, short_min_conf=0.65), 0, 6),

        ("TF3+VIX L>=0% S>=0.5% shrt65 sex",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_05, short_min_conf=0.65), 0, 6),

        ("TF3+VIX L>=0% S>=0.6% shrt65 sex",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65), 0, 6),

        ("TF3+VIX L>=0% S>=0.6% shrt70 sex",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.70), 0, 6),

        ("TF3+VIX L>=0% S>=0.8% shrt65 sex",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_08, short_min_conf=0.65), 0, 6),

        # TF3+VIX hybrid + no_turn
        ("TF3+VIX L0/S0.6 shrt65 no_turn sex",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65, no_turning_against=True), 0, 6),

        # TF3+VIX hybrid + health
        ("TF3+VIX L0/S0.6 shrt65 h>=0.3 sex",
         _make_tf3_vix_combo(cascade_vix),
         dict(long_spy_set=spy_00, short_spy_set=spy_06, short_min_conf=0.65, min_channel_health=0.3), 0, 6),
    ]

    for label, base_fn, filter_kwargs, cd, tp in configs:
        fn = wrap_with_filters(base_fn, **filter_kwargs)
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

    run_loss_identification(signals, cascade_vix, spy_daily)
    run_s1tf3_hybrid_spy(signals, cascade_vix, spy_daily)
    run_channel_quality(signals, cascade_vix, spy_daily)
    run_tf_turning(signals, cascade_vix, spy_daily)
    run_atr_regime(signals, cascade_vix, spy_daily)
    run_tf3_hybrid_sweep(signals, cascade_vix, spy_daily)
    run_conf_adaptive_spy(signals, cascade_vix, spy_daily)
    run_grand_sweep(signals, cascade_vix, spy_daily)

    print("\n\n" + "=" * 100)
    print("  ALL v14 EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
