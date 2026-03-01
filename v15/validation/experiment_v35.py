#!/usr/bin/env python3
"""v35: Explore new trade expansion frontiers beyond CV (402 trades, 100% WR).
1. Profile uncaptured signals — what does CV miss and why?
2. Test relaxed boundaries on existing filters
3. Explore new signal sources / filter combinations
"""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _build_filter_cascade,
    _make_v28_cu, _make_v29_cv, _SigProxy, _AnalysisProxy, _floor_stop_tp
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

args = (cascade_vix, spy_above_sma20, spy_above_055pct,
        spy_dist_map, spy_dist_5, spy_dist_50,
        vix_map, spy_return_map, spy_ret_2d)

cu_fn = _make_v28_cu(*args)
cv_fn = _make_v29_cv(*args)

# Get CV trades
cv_trades = simulate_trades(signals, cv_fn, 'CV', cooldown=0, trail_power=6)
cv_dates = {t.entry_date for t in cv_trades}

print(f"CV: {len(cv_trades)} trades, 100% WR, ${sum(t.pnl for t in cv_trades):+,.0f}")
print(f"CV trade dates: {len(cv_dates)}")

# ══════════════════════════════════════════════════════════
# FRONTIER 1: Uncaptured CS signals (HOLD days with active signals)
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FRONTIER 1: What signals does CV miss?")
print("="*70)

# All days with ANY CS signal (BUY or SELL)
cs_signal_days = [d for d in signals if d.cs_action in ('BUY', 'SELL')]
v5_signal_days = [d for d in signals if d.v5_take_bounce]
print(f"\nTotal CS BUY/SELL days: {len(cs_signal_days)}")
print(f"Total V5 bounce days: {len(v5_signal_days)}")

# Find days with CS signals that CV doesn't trade
missed_cs = [d for d in cs_signal_days if d.date not in cv_dates]
missed_v5 = [d for d in v5_signal_days if d.date not in cv_dates]
print(f"Missed CS signal days: {len(missed_cs)}")
print(f"Missed V5 bounce days: {len(missed_v5)}")

# Simulate what would happen if we traded ALL missed CS signals
# (with the same trail_power=6 trailing stop)
def sim_missed(missed_days, label):
    """Simulate trades on missed days using raw CS signals."""
    def missed_fn(day: DaySignals):
        if day.date in missed_dates:
            if day.cs_action == 'BUY':
                return ('BUY', day.cs_confidence, 2.0, 4.0, 'CS')
            elif day.cs_action == 'SELL':
                return ('SELL', day.cs_confidence, 2.0, 4.0, 'CS')
            elif day.v5_take_bounce:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    missed_dates = {d.date for d in missed_days}
    trades = simulate_trades(signals, missed_fn, label, cooldown=0, trail_power=6)
    return trades

missed_all = missed_cs + [d for d in missed_v5 if d.date not in {m.date for m in missed_cs}]
missed_trades = sim_missed(missed_all, 'missed')
n = len(missed_trades)
if n > 0:
    wins = sum(1 for t in missed_trades if t.pnl > 0)
    losses = sum(1 for t in missed_trades if t.pnl <= 0)
    wr = wins/n*100
    pnl = sum(t.pnl for t in missed_trades)
    bl = min(t.pnl for t in missed_trades)
    print(f"\nMissed signal trades: {n} trades, {wins}W/{losses}L, {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

    # Profile losses
    loss_trades = [t for t in missed_trades if t.pnl <= 0]
    print(f"\nLoss profile ({len(loss_trades)} losses):")
    for t in sorted(loss_trades, key=lambda x: x.pnl):
        day = next((d for d in signals if d.date == t.entry_date), None)
        if day:
            print(f"  {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} "
                  f"h={day.cs_channel_health:.3f} c={day.cs_confidence:.3f} "
                  f"pos={day.cs_position_score:.3f} confl={day.cs_confluence_score:.2f} "
                  f"TFs={_count_tf_confirming(day, t.direction)} "
                  f"VIX={vix_map.get(t.entry_date, 0):.1f}")

# ══════════════════════════════════════════════════════════
# FRONTIER 2: Relax V5 filter boundaries
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FRONTIER 2: V5 filter boundary relaxation")
print("="*70)

# Try wider V5 filters
for h_max in [0.50, 0.51, 0.52, 0.55, 0.60, 0.70, 1.0]:
    for pos_max in [0.85, 0.86, 0.87, 0.90, 0.95, 1.0]:
        def make_cv_variant(h_max_v=h_max, pos_max_v=pos_max):
            def fn(day: DaySignals):
                result = cu_fn(day)
                if result is not None:
                    return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < h_max_v and day.cs_position_score < pos_max_v:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        fn = make_cv_variant()
        trades = simulate_trades(signals, fn, 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n > 0 else 0
        pnl = sum(t.pnl for t in trades)
        bl = min(t.pnl for t in trades) if trades else 0
        marker = " <-- CV" if h_max == 0.50 and pos_max == 0.85 else ""
        marker = " ***" if wr >= 100 and n > 402 else marker
        if wr >= 99.5 or (h_max == 0.50 and pos_max == 0.85):
            print(f"  h<{h_max:.2f} pos<{pos_max:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ══════════════════════════════════════════════════════════
# FRONTIER 3: Gate-free bypass boundary relaxation
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FRONTIER 3: Gate-free bypass (h>=X) boundary sweep")
print("="*70)

# CU uses h>=0.38 OR confl>=0.90 for gate-free bypass
# What if we lower h threshold?
for h_thresh in [0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.30]:
    for confl_thresh in [0.90, 0.85, 0.80]:
        def make_cu_variant(h_t=h_thresh, c_t=confl_thresh):
            from v15.validation.combo_backtest import _make_v27_ct
            ct_fn = _make_v27_ct(*args)
            def fn(day: DaySignals):
                result = ct_fn(day)
                if result is not None:
                    return result
                # Gate-free bypass with variant thresholds
                if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                    if day.cs_channel_health >= h_t or day.cs_confluence_score >= c_t:
                        direction = day.cs_action
                        stop, tp = _floor_stop_tp(
                            getattr(day, 'cs_suggested_stop_pct', 2.0),
                            getattr(day, 'cs_suggested_tp_pct', 4.0))
                        return (direction, day.cs_confidence, stop, tp, 'CS')
                return None
            return fn
        fn = make_cu_variant()
        trades = simulate_trades(signals, fn, 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n > 0 else 0
        pnl = sum(t.pnl for t in trades)
        bl = min(t.pnl for t in trades) if trades else 0
        marker = " <-- CU" if h_thresh == 0.38 and confl_thresh == 0.90 else ""
        marker = " ***" if wr >= 100 and n > 377 else marker
        if wr >= 99.5 or (h_thresh == 0.38 and confl_thresh == 0.90):
            print(f"  h>={h_thresh:.2f} confl>={confl_thresh:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ══════════════════════════════════════════════════════════
# FRONTIER 4: Confidence threshold exploration
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FRONTIER 4: Lower MIN_SIGNAL_CONFIDENCE from 0.45")
print("="*70)

# What if we accept lower-confidence signals with extra filters?
for min_conf in [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.10]:
    def make_lowconf(mc=min_conf):
        def fn(day: DaySignals):
            # Full CV logic but with lower conf threshold
            result = cv_fn(day)
            if result is not None:
                return result
            # Try lower conf with strict health filter
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= mc:
                if day.cs_channel_health >= 0.50 and day.cs_confluence_score >= 0.90:
                    direction = day.cs_action
                    stop, tp = _floor_stop_tp(
                        getattr(day, 'cs_suggested_stop_pct', 2.0),
                        getattr(day, 'cs_suggested_tp_pct', 4.0))
                    return (direction, day.cs_confidence, stop, tp, 'CS')
            return None
        return fn
    fn = make_lowconf()
    trades = simulate_trades(signals, fn, 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n > 0 else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    new_t = n - 402
    marker = " ***" if wr >= 100 and n > 402 else ""
    print(f"  min_conf={min_conf:.2f} h>=0.50&confl>=0.90: {n:4d} trades (+{new_t:3d}), {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# Try different health/confl combos for low-conf recovery
print("\n  Low-conf recovery with various filters (min_conf=0.10):")
for h_min in [0.40, 0.45, 0.50, 0.55, 0.60]:
    for confl_min in [0.80, 0.85, 0.90, 0.95, 1.00]:
        def make_lowconf2(hm=h_min, cm=confl_min):
            def fn(day: DaySignals):
                result = cv_fn(day)
                if result is not None:
                    return result
                if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= 0.10:
                    if day.cs_channel_health >= hm and day.cs_confluence_score >= cm:
                        direction = day.cs_action
                        stop, tp = _floor_stop_tp(
                            getattr(day, 'cs_suggested_stop_pct', 2.0),
                            getattr(day, 'cs_suggested_tp_pct', 4.0))
                        return (direction, day.cs_confidence, stop, tp, 'CS')
                return None
            return fn
        fn = make_lowconf2()
        trades = simulate_trades(signals, fn, 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n > 0 else 0
        new_t = n - 402
        bl = min(t.pnl for t in trades) if trades else 0
        if new_t > 0 and wr >= 99.5:
            print(f"    h>={hm:.2f} confl>={cm:.2f}: {n:4d} trades (+{new_t:3d}), {wr:5.1f}% WR, BL=${bl:+,.0f}")

# ══════════════════════════════════════════════════════════
# FRONTIER 5: HOLD days where CS is close to a signal
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FRONTIER 5: Near-signal HOLD days")
print("="*70)

# Check HOLD days with high health and confluence
hold_days = [d for d in signals if d.cs_action == 'HOLD' and d.date not in cv_dates]
print(f"Total HOLD days (not in CV): {len(hold_days)}")

# Profile by health and confluence
high_health_hold = [d for d in hold_days if d.cs_channel_health >= 0.50]
high_confl_hold = [d for d in hold_days if d.cs_confluence_score >= 0.80]
both = [d for d in hold_days if d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 0.80]
print(f"  h>=0.50: {len(high_health_hold)}")
print(f"  confl>=0.80: {len(high_confl_hold)}")
print(f"  both: {len(both)}")

# What's the confidence distribution of HOLD days?
hold_confs = [d.cs_confidence for d in hold_days]
print(f"  Confidence range: {min(hold_confs):.3f} to {max(hold_confs):.3f}")
print(f"  conf>=0.30: {sum(1 for c in hold_confs if c >= 0.30)}")
print(f"  conf>=0.40: {sum(1 for c in hold_confs if c >= 0.40)}")

# ══════════════════════════════════════════════════════════
# FRONTIER 6: V5 bounce with higher trail power
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FRONTIER 6: Trail power sweep for individual V5 losers")
print("="*70)

# Try removing V5 filters entirely but with higher trail power
for power in [6, 7, 8, 10, 12, 15, 20]:
    def make_v5_all(p=power):
        def fn(day: DaySignals):
            result = cu_fn(day)
            if result is not None:
                return result
            if day.v5_take_bounce:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    fn = make_v5_all()
    trades = simulate_trades(signals, fn, 'test', cooldown=0, trail_power=power)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n > 0 else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 else ""
    print(f"  power={power:2d} ALL V5: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ══════════════════════════════════════════════════════════
# FRONTIER 7: Direction-specific relaxation
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FRONTIER 7: BUY-only vs SELL-only relaxation")
print("="*70)

# Profile CV trades by direction
cv_buys = [t for t in cv_trades if t.direction == 'BUY']
cv_sells = [t for t in cv_trades if t.direction == 'SELL']
print(f"CV BUYs: {len(cv_buys)}, all wins: {all(t.pnl > 0 for t in cv_buys)}")
print(f"CV SELLs: {len(cv_sells)}, all wins: {all(t.pnl > 0 for t in cv_sells)}")

# Try adding more BUYs with relaxed filters
for h_min in [0.30, 0.35, 0.38]:
    for confl_min in [0.80, 0.85, 0.90]:
        def make_buy_relax(hm=h_min, cm=confl_min):
            def fn(day: DaySignals):
                result = cv_fn(day)
                if result is not None:
                    return result
                # Extra BUYs with relaxed thresholds
                if day.cs_action == 'BUY' and day.cs_confidence >= 0.10:
                    if day.cs_channel_health >= hm and day.cs_confluence_score >= cm:
                        stop, tp = _floor_stop_tp(
                            getattr(day, 'cs_suggested_stop_pct', 2.0),
                            getattr(day, 'cs_suggested_tp_pct', 4.0))
                        return ('BUY', day.cs_confidence, stop, tp, 'CS')
                return None
            return fn
        fn = make_buy_relax()
        trades = simulate_trades(signals, fn, 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n > 0 else 0
        new_t = n - 402
        bl = min(t.pnl for t in trades) if trades else 0
        if new_t > 0:
            print(f"  BUY h>={hm:.2f} confl>={cm:.2f}: {n:4d} trades (+{new_t:3d}), {wr:5.1f}% WR, BL=${bl:+,.0f}")

# Try adding more SELLs
for h_min in [0.30, 0.35, 0.38]:
    for confl_min in [0.80, 0.85, 0.90]:
        def make_sell_relax(hm=h_min, cm=confl_min):
            def fn(day: DaySignals):
                result = cv_fn(day)
                if result is not None:
                    return result
                if day.cs_action == 'SELL' and day.cs_confidence >= 0.10:
                    if day.cs_channel_health >= hm and day.cs_confluence_score >= cm:
                        stop, tp = _floor_stop_tp(
                            getattr(day, 'cs_suggested_stop_pct', 2.0),
                            getattr(day, 'cs_suggested_tp_pct', 4.0))
                        return ('SELL', day.cs_confidence, stop, tp, 'CS')
                return None
            return fn
        fn = make_sell_relax()
        trades = simulate_trades(signals, fn, 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n > 0 else 0
        new_t = n - 402
        bl = min(t.pnl for t in trades) if trades else 0
        if new_t > 0:
            print(f"  SELL h>={hm:.2f} confl>={cm:.2f}: {n:4d} trades (+{new_t:3d}), {wr:5.1f}% WR, BL=${bl:+,.0f}")

print("\nDone")
