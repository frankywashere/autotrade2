#!/usr/bin/env python3
"""Fine sweep of SELL gate-free h threshold."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, simulate_trades, _build_filter_cascade,
    _make_v27_ct, MIN_SIGNAL_CONFIDENCE, _floor_stop_tp
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
day_map = {d.date: d for d in signals}

# Fine sweep of SELL h threshold (BUY stays at h>=0.38)
print("SELL gate-free h threshold sweep (BUY h>=0.38, V5 h<0.57):")
best_n = 0
for h_sell_pct in range(38, 19, -1):
    h_sell = h_sell_pct / 100.0
    def make_fn(hs=h_sell):
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
    trades = simulate_trades(signals, make_fn(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > best_n else ""
    if wr >= 100 and n > best_n:
        best_n = n
    print(f"  SELL h>={h_sell:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# Try SELL h>=0.30 + BUY h>=0.37 combined
print("\nCombined BUY/SELL relaxation:")
for h_buy in [0.38, 0.37]:
    for h_sell in [0.38, 0.35, 0.33, 0.31, 0.30, 0.28, 0.25]:
        def make_combo(hb=h_buy, hs=h_sell):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
                    h_thresh = hb if day.cs_action == 'BUY' else hs
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
        trades = simulate_trades(signals, make_combo(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        pnl = sum(t.pnl for t in trades)
        bl = min(t.pnl for t in trades) if trades else 0
        marker = " ***" if wr >= 100 and n > 412 else ""
        if wr >= 99.5 or n > 412:
            print(f"  BUY h>={h_buy:.2f} SELL h>={h_sell:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# Show the new SELL trades at h>=0.30
print("\nNew SELL trades from h>=0.30 (not in CW):")
from v15.validation.combo_backtest import _make_v30_cw
cw_fn = _make_v30_cw(*args)
cw_trades = simulate_trades(signals, cw_fn, 'CW', cooldown=0, trail_power=6)
cw_dates = {t.entry_date for t in cw_trades}

def make_best():
    def fn(day):
        result = ct_fn(day)
        if result is not None:
            return result
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            h_thresh = 0.38 if day.cs_action == 'BUY' else 0.30
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

best_trades = simulate_trades(signals, make_best(), 'best', cooldown=0, trail_power=6)
best_dates = {t.entry_date for t in best_trades}
new_dates = best_dates - cw_dates
print(f"New trade dates: {len(new_dates)}")
for nd in sorted(new_dates):
    t = next((x for x in best_trades if x.entry_date == nd), None)
    day = day_map.get(nd)
    if t and day:
        dd = nd.date() if hasattr(nd, 'date') else nd
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        print(f"  {str(nd)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} h={day.cs_channel_health:.3f} c={day.cs_confidence:.3f} "
              f"confl={day.cs_confluence_score:.2f} VIX={vix_map.get(nd, 0):.1f} {dow}")

# Validate best candidate
print("\nBest candidate validation (BUY h>=0.38 SELL h>=0.30 V5 h<0.57):")
n = len(best_trades)
w = sum(1 for t in best_trades if t.pnl > 0)
print(f"  Full: {n} trades, {w/n*100:.1f}% WR, ${sum(t.pnl for t in best_trades):+,.0f}")

train = [t for t in best_trades if t.entry_date.year <= 2021]
test = [t for t in best_trades if 2022 <= t.entry_date.year <= 2025]
oos = [t for t in best_trades if t.entry_date.year >= 2026]
print(f"  Train: {len(train)}t, {sum(1 for t in train if t.pnl > 0)/len(train)*100:.0f}% WR")
print(f"  Test:  {len(test)}t, {sum(1 for t in test if t.pnl > 0)/len(test)*100:.0f}% WR")
if oos:
    print(f"  2026:  {len(oos)}t, {sum(1 for t in oos if t.pnl > 0)/len(oos)*100:.0f}% WR, ${sum(t.pnl for t in oos):+,.0f}")
wf = sum(1 for yr in range(2017, 2026) for yearly in [[t for t in best_trades if t.entry_date.year == yr]] if yearly and all(t.pnl > 0 for t in yearly))
print(f"  WF: {wf}/9 PASS")

print("\nDone")
