#!/usr/bin/env python3
"""v28: Combine v27 recovery paths + explore new angles.
100% WR recoveries from v27: c90(+2), BUY VIX>50&h15(+1), Mon&h30(+1), tf2:h20&confl80(+1).
Test: do they stack? What's the max combo at 100% WR?
Also: adaptive trail, wider stops, ATR-based exits, position-score patterns."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _build_filter_cascade,
    _make_v26_cs, _make_v25_cr, _SigProxy, _AnalysisProxy, _floor_stop_tp
)

cache_path = Path(__file__).parent / 'combo_cache' / 'combo_signals.pkl'
with open(cache_path, 'rb') as f:
    data = pickle.load(f)
signals = data['signals']
vix_daily = data.get('vix_daily')
spy_daily = data.get('spy_daily')

cascade_vix = _build_filter_cascade(vix=True)
cascade_vix.precompute_vix_cooldown(vix_daily)

spy_close = spy_daily['close'].values.astype(float)
spy_dist_map, spy_dist_5, spy_dist_50 = {}, {}, {}
spy_above_sma20, spy_above_055pct = set(), set()
for win, dm in [(5, spy_dist_5), (20, spy_dist_map), (50, spy_dist_50)]:
    sma = pd.Series(spy_close).rolling(win).mean().values
    for i in range(win, len(spy_close)):
        if sma[i] > 0:
            d = (spy_close[i] - sma[i]) / sma[i] * 100
            dm[spy_daily.index[i]] = d
            if win == 20:
                if d >= 0: spy_above_sma20.add(spy_daily.index[i])
                if d >= 0.55: spy_above_055pct.add(spy_daily.index[i])

vix_map = {idx: row['close'] for idx, row in vix_daily.iterrows()}
spy_return_map = {}
for i in range(1, len(spy_close)):
    spy_return_map[spy_daily.index[i]] = (spy_close[i]-spy_close[i-1])/spy_close[i-1]*100
spy_ret_2d = {}
for i in range(2, len(spy_close)):
    spy_ret_2d[spy_daily.index[i]] = (spy_close[i]-spy_close[i-2])/spy_close[i-2]*100

# ── CS baseline ──
cs_fn = _make_v26_cs(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50,
                      vix_map, spy_return_map, spy_ret_2d)
cs_trades = simulate_trades(signals, cs_fn, 'CS', cooldown=0, trail_power=6)
cs_dates = {t.entry_date for t in cs_trades}
day_map = {day.date: day for day in signals}
print(f"CS baseline: {len(cs_trades)} trades, {sum(1 for t in cs_trades if t.pnl>0)/len(cs_trades)*100:.1f}% WR, ${sum(t.pnl for t in cs_trades):+,.0f}")

# ── All CS+CP base (no TF, no health) ──
def make_all_cs_base():
    def fn(day):
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            return None
        sig = _SigProxy(day)
        ana = _AnalysisProxy(day.cs_tf_states)
        ok, adj, _ = cascade_vix.evaluate(sig, ana, feature_vec=None, bar_datetime=day.date,
                                           higher_tf_data=None, spy_df=None, vix_df=None)
        if not ok or adj < MIN_SIGNAL_CONFIDENCE: return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, adj, s, t, 'CS')
    return fn

def make_cp_filtered(base_fn):
    def fn(day):
        result = base_fn(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        if action == 'BUY':
            spy_pass = False
            if day.date in spy_above_sma20: spy_pass = True
            elif conf >= 0.80: spy_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 5: spy_pass = True
            elif conf >= 0.70 and _count_tf_confirming(day, 'BUY') >= 4 and day.cs_position_score < 0.95: spy_pass = True
            elif conf >= 0.65 and _count_tf_confirming(day, 'BUY') >= 4 and day.cs_channel_health >= 0.40: spy_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.65: spy_pass = True
            if not spy_pass:
                if day.cs_confluence_score >= 0.9 and conf >= 0.45: spy_pass = True
                elif vix_map.get(day.date, 22) > 30 and conf >= 0.50: spy_pass = True
                elif vix_map.get(day.date, 22) > 25 and conf >= 0.55: spy_pass = True
                elif spy_return_map.get(day.date, 0) > 1.0 and conf >= 0.55: spy_pass = True
            if not spy_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1: spy_pass = True
                elif day.cs_channel_health >= 0.25 and day.cs_position_score < 0.95: spy_pass = True
            if not spy_pass: return None
            conf_pass = False
            if conf >= 0.66: conf_pass = True
            elif day.cs_position_score <= 0.99: conf_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 4: conf_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.55: conf_pass = True
            if not conf_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1: conf_pass = True
                elif day.cs_channel_health >= 0.25 and day.cs_position_score < 0.95: conf_pass = True
            if not conf_pass: return None
        if action == 'SELL':
            spy_pass = False
            if day.date in spy_above_055pct: spy_pass = True
            elif spy_dist_map.get(day.date, 999) < 0 and day.cs_channel_health >= 0.32: spy_pass = True
            elif 0 <= spy_dist_map.get(day.date, 999) < 0.55 and day.cs_position_score < 0.99: spy_pass = True
            elif 0 <= spy_dist_map.get(day.date, 999) < 0.55 and day.cs_position_score >= 0.99 and day.cs_channel_health >= 0.35: spy_pass = True
            if not spy_pass and spy_dist_5.get(day.date, -999) >= 0: spy_pass = True
            if not spy_pass:
                vix = vix_map.get(day.date, 22)
                if (vix < 20 or vix > 25) and day.cs_channel_health >= 0.25: spy_pass = True
            if not spy_pass and day.cs_channel_health >= 0.25:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() in (0, 2, 3): spy_pass = True
            if not spy_pass and spy_dist_50.get(day.date, -999) >= 1.0: spy_pass = True
            if not spy_pass:
                if _count_tf_confirming(day, 'SELL') >= 5 and day.cs_channel_health >= 0.20: spy_pass = True
                elif spy_ret_2d.get(day.date, 0) < -2.0 and day.cs_channel_health >= 0.15: spy_pass = True
            if not spy_pass: return None
            conf_pass = False
            if conf >= 0.65: conf_pass = True
            elif day.cs_channel_health >= 0.30: conf_pass = True
            elif day.cs_confluence_score >= 0.9 and day.cs_channel_health >= 0.25: conf_pass = True
            elif vix_map.get(day.date, 22) > 25 and day.cs_channel_health >= 0.20: conf_pass = True
            elif spy_return_map.get(day.date, 0) < -1.0 and day.cs_channel_health >= 0.20: conf_pass = True
            elif vix_map.get(day.date, 22) > 30 and day.cs_channel_health >= 0.15: conf_pass = True
            elif day.cs_channel_health >= 0.10 and conf >= 0.60 and _count_tf_confirming(day, 'SELL') >= 4: conf_pass = True
            if not conf_pass: return None
        return result
    return fn

all_base = make_all_cs_base()
all_filtered = make_cp_filtered(all_base)

# ── SECTION 1: Identify which dates each recovery path adds ──
print("\n=== RECOVERY PATH DATE IDENTIFICATION ===")

def get_new_dates(check, label):
    def fn(day):
        result = cs_fn(day)
        if result is not None: return result
        result = all_filtered(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        if check(day, action, conf): return result
        return None
    trades = simulate_trades(signals, fn, label, cooldown=0, trail_power=6)
    new = [t for t in trades if t.entry_date not in cs_dates]
    return new, trades

# Also need TF-level relaxation function
def make_tf_relax(tf_level, check):
    prev_tf_states = {}
    streaks = defaultdict(int)
    def tf_base(day):
        nonlocal prev_tf_states, streaks
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
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            return None
        action = day.cs_action
        confirmed = 0
        if day.cs_tf_states:
            for tf2, state in day.cs_tf_states.items():
                if not state.get('valid', False): continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks.get(tf2, 0) >= 1:
                    confirmed += 1
        if confirmed < tf_level: return None
        sig = _SigProxy(day)
        ana = _AnalysisProxy(day.cs_tf_states)
        ok, adj, _ = cascade_vix.evaluate(sig, ana, feature_vec=None, bar_datetime=day.date,
                                           higher_tf_data=None, spy_df=None, vix_df=None)
        if not ok or adj < MIN_SIGNAL_CONFIDENCE: return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, adj, s, t, 'CS')
    tf_filtered = make_cp_filtered(tf_base)
    def fn(day):
        result = cs_fn(day)
        if result is not None: return result
        result = tf_filtered(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        if check(day, action, conf): return result
        return None
    return fn

paths = {
    'c90': lambda d, a, c: c >= 0.90,
    'BUY VIX>50&h15': lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) > 50 and d.cs_channel_health >= 0.15,
    'Mon&h30': lambda d, a, c: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 0 and d.cs_channel_health >= 0.30,
}

path_dates = {}
for name, check in paths.items():
    new, all_t = get_new_dates(check, name)
    dates = sorted([t.entry_date for t in new])
    path_dates[name] = set(t.entry_date for t in new)
    pnls = {t.entry_date: t.pnl for t in new}
    print(f"\n{name}: +{len(new)} new trades")
    for dt in dates:
        day = day_map.get(dt)
        if day:
            print(f"  {str(dt)[:10]}: {day.cs_action} PnL=${pnls[dt]:+,.0f} h={day.cs_channel_health:.3f} "
                  f"c={day.cs_confidence:.3f} confl={day.cs_confluence_score:.2f} pos={day.cs_position_score:.2f}")

# TF2 relaxation
tf2_fn = make_tf_relax(2, lambda d, a, c: d.cs_channel_health >= 0.20 and d.cs_confluence_score >= 0.80)
tf2_trades = simulate_trades(signals, tf2_fn, 'tf2_relax', cooldown=0, trail_power=6)
tf2_new = [t for t in tf2_trades if t.entry_date not in cs_dates]
path_dates['tf2:h20&confl80'] = set(t.entry_date for t in tf2_new)
print(f"\ntf2:h20&confl80: +{len(tf2_new)} new trades")
for t in sorted(tf2_new, key=lambda x: x.entry_date):
    day = day_map.get(t.entry_date)
    if day:
        print(f"  {str(t.entry_date)[:10]}: {t.direction} PnL=${t.pnl:+,.0f} h={day.cs_channel_health:.3f} "
              f"c={day.cs_confidence:.3f} confl={day.cs_confluence_score:.2f} pos={day.cs_position_score:.2f}")

# Check overlaps
print("\n=== OVERLAP MATRIX ===")
all_path_names = list(path_dates.keys())
for i, n1 in enumerate(all_path_names):
    for n2 in all_path_names[i+1:]:
        overlap = path_dates[n1] & path_dates[n2]
        if overlap:
            print(f"  {n1} & {n2}: {len(overlap)} overlap(s) — {[str(d)[:10] for d in overlap]}")
        else:
            print(f"  {n1} & {n2}: no overlap")

# ── SECTION 2: Combine all 100% WR paths ──
print("\n=== COMBINED RECOVERY PATHS ===")

def make_combined(path_checks):
    """Combine multiple TF0-level recovery checks."""
    def fn(day):
        result = cs_fn(day)
        if result is not None: return result
        result = all_filtered(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        for check in path_checks:
            if check(day, action, conf):
                return result
        return None
    return fn

# Individual
combos = [
    ('c90 only', [paths['c90']]),
    ('VIX>50 only', [paths['BUY VIX>50&h15']]),
    ('Mon&h30 only', [paths['Mon&h30']]),
    # Pairs
    ('c90 + VIX>50', [paths['c90'], paths['BUY VIX>50&h15']]),
    ('c90 + Mon&h30', [paths['c90'], paths['Mon&h30']]),
    ('VIX>50 + Mon&h30', [paths['BUY VIX>50&h15'], paths['Mon&h30']]),
    # Triple
    ('c90 + VIX>50 + Mon', [paths['c90'], paths['BUY VIX>50&h15'], paths['Mon&h30']]),
]

for name, checks in combos:
    fn = make_combined(checks)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cs_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades (+{new}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

# Also: combined TF0 paths + TF2 relaxation
print("\n  -- With tf2:h20&confl80 relaxation --")
# Need a unified function that includes both TF0 recovery AND tf2 relaxation
def make_full_combined(tf0_checks, include_tf2_relax=False):
    """Full combined: CS + TF0 recovery paths + optional TF2 relaxation."""
    # For tf2 relaxation, need independent streak state
    prev_tf2 = {}
    streaks_tf2 = defaultdict(int)
    def fn(day):
        nonlocal prev_tf2, streaks_tf2
        # Always update tf2 streak state
        if include_tf2_relax and day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    streaks_tf2[tf] = 0
                    continue
                md = state.get('momentum_direction', 0.0)
                prev_md = prev_tf2.get(tf, 0.0)
                if (md > 0 and prev_md > 0) or (md < 0 and prev_md < 0):
                    streaks_tf2[tf] += 1
                else:
                    streaks_tf2[tf] = 1 if md != 0 else 0
                prev_tf2[tf] = md

        # Try CS first
        result = cs_fn(day)
        if result is not None: return result

        # Try TF0 recovery paths
        result = all_filtered(day)
        if result is not None:
            action, conf, s_pct, t_pct, src = result
            for check in tf0_checks:
                if check(day, action, conf):
                    return result

        # Try TF2 relaxation
        if include_tf2_relax:
            if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
                return None
            action = day.cs_action
            confirmed = 0
            if day.cs_tf_states:
                for tf, state in day.cs_tf_states.items():
                    if not state.get('valid', False): continue
                    md = state.get('momentum_direction', 0.0)
                    aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                    if aligned and streaks_tf2.get(tf, 0) >= 1:
                        confirmed += 1
            if confirmed >= 2:
                sig = _SigProxy(day)
                ana = _AnalysisProxy(day.cs_tf_states)
                ok, adj, _ = cascade_vix.evaluate(sig, ana, feature_vec=None, bar_datetime=day.date,
                                                   higher_tf_data=None, spy_df=None, vix_df=None)
                if ok and adj >= MIN_SIGNAL_CONFIDENCE:
                    s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
                    base_result = (day.cs_action, adj, s, t, 'CS')
                    cp_result = make_cp_filtered(lambda d: base_result)(day)
                    if cp_result is not None:
                        _, conf2, _, _, _ = cp_result
                        if day.cs_channel_health >= 0.20 and day.cs_confluence_score >= 0.80:
                            return cp_result
        return None
    return fn

combined_tests = [
    ('c90+VIX50+Mon+tf2', [paths['c90'], paths['BUY VIX>50&h15'], paths['Mon&h30']], True),
    ('c90+tf2', [paths['c90']], True),
    ('c90+VIX50+tf2', [paths['c90'], paths['BUY VIX>50&h15']], True),
]

for name, checks, tf2 in combined_tests:
    fn = make_full_combined(checks, include_tf2_relax=tf2)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cs_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades (+{new}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

# ── SECTION 3: Adaptive trail power ──
print("\n=== ADAPTIVE TRAIL POWER ===")
# What if we use higher trail power (wider trail) for lower-confidence trades?
# This could turn micro-losses into wins
for power in [5, 6, 7, 8, 10, 12]:
    fn = make_combined([paths['c90'], paths['BUY VIX>50&h15'], paths['Mon&h30']])
    trades = simulate_trades(signals, fn, f'p{power}', cooldown=0, trail_power=power)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    print(f"  trail_power={power:2d}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# ── SECTION 4: Different stop/tp for TF0 recovery trades ──
print("\n=== WIDER STOPS FOR RECOVERY TRADES ===")
# What if we give recovery trades wider stops (3%/6% instead of 2%/4%)?

def make_combined_wider(path_checks, stop_pct=3.0, tp_pct=6.0):
    def fn(day):
        result = cs_fn(day)
        if result is not None: return result
        result = all_filtered(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        for check in path_checks:
            if check(day, action, conf):
                # Override stops for recovery trades
                return (action, conf, stop_pct, tp_pct, src)
        return None
    return fn

for stop, tp in [(2.5, 5.0), (3.0, 6.0), (3.5, 7.0), (4.0, 8.0)]:
    fn = make_combined_wider([paths['c90'], paths['BUY VIX>50&h15'], paths['Mon&h30']], stop, tp)
    trades = simulate_trades(signals, fn, f's{stop}', cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cs_dates])
    print(f"  stop={stop}%/tp={tp}%: {n:3d} trades (+{new}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# ── SECTION 5: Adjacent-day signal clustering ──
print("\n=== ADJACENT DAY ANALYSIS ===")
# Look at gap trades that have a CS/CS+CP signal on adjacent days
# If a signal fires day N and day N+1, the day N+1 signal might be safer
all_cp_trades = simulate_trades(signals, all_filtered, 'allCP', cooldown=0, trail_power=6)
all_cp_dates = sorted([t.entry_date for t in all_cp_trades])
gap_trades = [t for t in all_cp_trades if t.entry_date not in cs_dates]

for t in sorted(gap_trades, key=lambda x: x.entry_date):
    day = day_map.get(t.entry_date)
    if not day: continue
    # Check if adjacent days have CS trades
    prev_in_cs = any(dt in cs_dates for dt in all_cp_dates if abs((dt - t.entry_date).days) <= 3 and dt != t.entry_date)
    adj_label = "NEAR_CS" if prev_in_cs else "ISOLATED"
    print(f"  {str(t.entry_date)[:10]}: {t.direction} PnL=${t.pnl:+,.0f} h={day.cs_channel_health:.3f} "
          f"pos={day.cs_position_score:.2f} confl={day.cs_confluence_score:.2f} [{adj_label}]")

# ── SECTION 6: New idea — Confluence score as primary filter ──
print("\n=== CONFLUENCE-BASED EXPANSION ===")
# Instead of health filter, what if we use confluence score as gate?
confl_tests = [
    ('confl>=0.90', lambda d, a, c: d.cs_confluence_score >= 0.90),
    ('confl>=0.85', lambda d, a, c: d.cs_confluence_score >= 0.85),
    ('confl>=0.80', lambda d, a, c: d.cs_confluence_score >= 0.80),
    ('confl>=0.75', lambda d, a, c: d.cs_confluence_score >= 0.75),
    ('confl>=0.70', lambda d, a, c: d.cs_confluence_score >= 0.70),
    ('confl90|h40', lambda d, a, c: d.cs_confluence_score >= 0.90 or d.cs_channel_health >= 0.40),
    ('confl80&h25', lambda d, a, c: d.cs_confluence_score >= 0.80 and d.cs_channel_health >= 0.25),
    ('confl70&h30', lambda d, a, c: d.cs_confluence_score >= 0.70 and d.cs_channel_health >= 0.30),
]

for name, check in confl_tests:
    fn = make_combined([check])  # Uses CS + all_filtered + check
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 339: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cs_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades (+{new}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

# ── SECTION 7: Directional analysis — are remaining trades mostly BUY or SELL? ──
print("\n=== DIRECTIONAL BREAKDOWN ===")
gap_buys = [t for t in gap_trades if t.direction == 'LONG']
gap_sells = [t for t in gap_trades if t.direction == 'SHORT']
print(f"Gap LONG:  {len(gap_buys)} ({sum(1 for t in gap_buys if t.pnl>0)}W/{sum(1 for t in gap_buys if t.pnl<=0)}L)")
print(f"Gap SHORT: {len(gap_sells)} ({sum(1 for t in gap_sells if t.pnl>0)}W/{sum(1 for t in gap_sells if t.pnl<=0)}L)")

# BUY-only and SELL-only recovery
print("\n  -- BUY-only recovery --")
buy_recoveries = [
    ('BUY pos0', lambda d, a, c: a == 'BUY' and d.cs_position_score == 0.0),
    ('BUY pos<50', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.50),
    ('BUY pos<20&h30', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.20 and d.cs_channel_health >= 0.30),
    ('BUY h40', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.40),
    ('BUY h45', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.45),
    ('BUY h50', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.50),
    ('BUY confl80&h30', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.80 and d.cs_channel_health >= 0.30),
    ('BUY confl90', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.90),
]

for name, check in buy_recoveries:
    fn = make_combined([check])
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 339: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cs_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades (+{new}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

print("\n  -- SELL-only recovery --")
sell_recoveries = [
    ('SELL pos>=90', lambda d, a, c: a == 'SELL' and d.cs_position_score >= 0.90),
    ('SELL pos100', lambda d, a, c: a == 'SELL' and d.cs_position_score >= 1.0),
    ('SELL h30', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.30),
    ('SELL h25&confl80', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.80),
    ('SELL confl90', lambda d, a, c: a == 'SELL' and d.cs_confluence_score >= 0.90),
]

for name, check in sell_recoveries:
    fn = make_combined([check])
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 339: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cs_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades (+{new}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

# ── SECTION 8: Year-of-trade analysis ──
print("\n=== YEAR-BY-YEAR GAP TRADE DISTRIBUTION ===")
year_dist = defaultdict(lambda: {'win': 0, 'loss': 0})
for t in gap_trades:
    yr = t.entry_date.year
    if t.pnl > 0:
        year_dist[yr]['win'] += 1
    else:
        year_dist[yr]['loss'] += 1

for yr in sorted(year_dist.keys()):
    w, l = year_dist[yr]['win'], year_dist[yr]['loss']
    print(f"  {yr}: {w}W / {l}L")

print("\nDone")
