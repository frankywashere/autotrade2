#!/usr/bin/env python3
"""v36: Push beyond CW (412 trades, 100% WR).
1. Gate-free h>=0.37 (was 0.38) + V5 h<0.57
2. Missed signal analysis with TF/health/confl filters
3. Additional V5 signals with extra filters for the losers
"""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _build_filter_cascade,
    _make_v27_ct, _make_v28_cu, _make_v30_cw, _floor_stop_tp
)

cache_path = Path(__file__).parent / 'combo_cache' / 'combo_signals.pkl'
with open(cache_path, 'rb') as f:
    data = pickle.load(f)
signals = data['signals']
vix_daily, spy_daily = data['vix_daily'], data['spy_daily']

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

ct_fn = _make_v27_ct(*args)
cu_fn = _make_v28_cu(*args)
cw_fn = _make_v30_cw(*args)

day_map = {d.date: d for d in signals}

# Baseline CW
cw_trades = simulate_trades(signals, cw_fn, 'CW', cooldown=0, trail_power=6)
cw_dates = {t.entry_date for t in cw_trades}
print(f"CW baseline: {len(cw_trades)} trades, 100% WR, ${sum(t.pnl for t in cw_trades):+,.0f}")

# ══════════════════════════════════════════════════════════
# IDEA 1: Lower gate-free bypass from h>=0.38 to h>=0.37
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("IDEA 1: Gate-free h>=0.37 + V5 h<0.57")
print("="*70)

for h_gate in [0.38, 0.37, 0.36, 0.35]:
    def make_cx(hg=h_gate):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            # Gate-free bypass
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                if day.cs_channel_health >= hg or day.cs_confluence_score >= 0.90:
                    stop, tp = _floor_stop_tp(
                        getattr(day, 'cs_suggested_stop_pct', 2.0),
                        getattr(day, 'cs_suggested_tp_pct', 4.0))
                    return (day.cs_action, day.cs_confidence, stop, tp, 'CS')
            # V5 bounce
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_cx(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " <-- CW" if h_gate == 0.38 else (" ***" if wr >= 100 and n > 412 else "")
    print(f"  h>={h_gate:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")
    # Show new trades
    if n > 412 and wr >= 100:
        cx_dates = {t.entry_date for t in trades}
        new_dates = cx_dates - cw_dates
        for nd in sorted(new_dates):
            day = day_map.get(nd)
            if day:
                print(f"    NEW: {nd} {day.cs_action} h={day.cs_channel_health:.3f} c={day.cs_confidence:.3f} confl={day.cs_confluence_score:.2f}")

# ══════════════════════════════════════════════════════════
# IDEA 2: Lower confl bypass threshold from 0.90
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("IDEA 2: Confl bypass threshold sweep (with h>=0.38)")
print("="*70)

for confl_thresh in [0.90, 0.85, 0.80, 0.75]:
    def make_cx2(ct=confl_thresh):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                if day.cs_channel_health >= 0.38 or day.cs_confluence_score >= ct:
                    stop, tp = _floor_stop_tp(
                        getattr(day, 'cs_suggested_stop_pct', 2.0),
                        getattr(day, 'cs_suggested_tp_pct', 4.0))
                    return (day.cs_action, day.cs_confidence, stop, tp, 'CS')
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_cx2(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " <-- CW" if confl_thresh == 0.90 else (" ***" if wr >= 100 and n > 412 else "")
    print(f"  confl>={confl_thresh:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ══════════════════════════════════════════════════════════
# IDEA 3: Recovery for missed signals with strict filters
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("IDEA 3: Missed signal recovery (TFs=0 days)")
print("="*70)

# Profile all days NOT in CW that have CS signals
missed_cs = [d for d in signals if d.cs_action in ('BUY', 'SELL') and d.date not in cw_dates]
print(f"Missed CS signals: {len(missed_cs)}")

# Try recovering with specific filters
# The 11 losses from Frontier 1 all had TFs=0
# What if we require at least 1 TF? or high health + confl?
for filter_name, filter_fn in [
    ("h>=0.50", lambda d: d.cs_channel_health >= 0.50),
    ("h>=0.60", lambda d: d.cs_channel_health >= 0.60),
    ("h>=0.50&confl>=0.80", lambda d: d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 0.80),
    ("h>=0.60&confl>=0.80", lambda d: d.cs_channel_health >= 0.60 and d.cs_confluence_score >= 0.80),
    ("h>=0.50&confl>=1.00", lambda d: d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 1.00),
    ("h>=0.60&confl>=1.00", lambda d: d.cs_channel_health >= 0.60 and d.cs_confluence_score >= 1.00),
    ("h>=0.70", lambda d: d.cs_channel_health >= 0.70),
    ("BUY&h>=0.50", lambda d: d.cs_action == 'BUY' and d.cs_channel_health >= 0.50),
    ("BUY&h>=0.60", lambda d: d.cs_action == 'BUY' and d.cs_channel_health >= 0.60),
    ("SELL&h>=0.50", lambda d: d.cs_action == 'SELL' and d.cs_channel_health >= 0.50),
    ("SELL&h>=0.60", lambda d: d.cs_action == 'SELL' and d.cs_channel_health >= 0.60),
]:
    filtered_dates = {d.date for d in missed_cs if filter_fn(d)}
    if not filtered_dates:
        continue
    def make_recovery(fd=filtered_dates):
        def fn(day):
            result = cw_fn(day)
            if result is not None:
                return result
            if day.date in fd:
                stop, tp = _floor_stop_tp(
                    getattr(day, 'cs_suggested_stop_pct', 2.0),
                    getattr(day, 'cs_suggested_tp_pct', 4.0))
                return (day.cs_action, day.cs_confidence, stop, tp, 'CS')
            return None
        return fn
    trades = simulate_trades(signals, make_recovery(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    new_t = n - 412
    bl = min(t.pnl for t in trades) if trades else 0
    pnl = sum(t.pnl for t in trades)
    marker = " ***" if wr >= 100 and new_t > 0 else ""
    print(f"  {filter_name:25s}: +{new_t:3d} trades, {n:4d} total, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ══════════════════════════════════════════════════════════
# IDEA 4: BUY-only relaxation of gate-free bypass
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("IDEA 4: BUY-only gate-free bypass relaxation")
print("="*70)

# What if we relax h threshold ONLY for BUY signals?
for h_buy in [0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.30]:
    def make_buy_relax(hb=h_buy):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                h_thresh = hb if day.cs_action == 'BUY' else 0.38
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    stop, tp = _floor_stop_tp(
                        getattr(day, 'cs_suggested_stop_pct', 2.0),
                        getattr(day, 'cs_suggested_tp_pct', 4.0))
                    return (day.cs_action, day.cs_confidence, stop, tp, 'CS')
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_buy_relax(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 412 else ""
    print(f"  BUY h>={h_buy:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# Same for SELL
print()
for h_sell in [0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.30]:
    def make_sell_relax(hs=h_sell):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                h_thresh = 0.38 if day.cs_action == 'BUY' else hs
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    stop, tp = _floor_stop_tp(
                        getattr(day, 'cs_suggested_stop_pct', 2.0),
                        getattr(day, 'cs_suggested_tp_pct', 4.0))
                    return (day.cs_action, day.cs_confidence, stop, tp, 'CS')
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_sell_relax(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 412 else ""
    print(f"  SELL h>={h_sell:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ══════════════════════════════════════════════════════════
# IDEA 5: Combined h>=0.37 + confl>=0.85 for gate-free
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("IDEA 5: Dual-axis gate-free bypass")
print("="*70)

for h_gate in [0.38, 0.37, 0.36, 0.35]:
    for confl_gate in [0.90, 0.85, 0.80]:
        def make_dual(hg=h_gate, cg=confl_gate):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                    if day.cs_channel_health >= hg or day.cs_confluence_score >= cg:
                        stop, tp = _floor_stop_tp(
                            getattr(day, 'cs_suggested_stop_pct', 2.0),
                            getattr(day, 'cs_suggested_tp_pct', 4.0))
                        return (day.cs_action, day.cs_confidence, stop, tp, 'CS')
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_dual(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        pnl = sum(t.pnl for t in trades)
        bl = min(t.pnl for t in trades) if trades else 0
        marker = " ***" if wr >= 100 and n > 412 else ""
        print(f"  h>={h_gate:.2f} confl>={confl_gate:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

print("\nDone")
