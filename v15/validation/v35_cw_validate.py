#!/usr/bin/env python3
"""Quick CW (h<0.57) validation."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, simulate_trades, _build_filter_cascade,
    _make_v28_cu, _floor_stop_tp
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
cu_fn = _make_v28_cu(*args)
day_map = {d.date: d for d in signals}

def make_cw(h_max=0.57):
    def fn(day):
        r = cu_fn(day)
        if r is not None:
            return r
        if day.v5_take_bounce and day.cs_channel_health < h_max and day.cs_position_score < 0.85:
            return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn

# CW validation
t = simulate_trades(signals, make_cw(), 'CW', cooldown=0, trail_power=6)
n = len(t)
w = sum(1 for x in t if x.pnl > 0)
wr = w / n * 100
pnl = sum(x.pnl for x in t)
bl = min(x.pnl for x in t)
print(f"CW (h<0.57 pos<0.85): {n} trades, {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

train = [x for x in t if x.entry_date.year <= 2021]
test = [x for x in t if 2022 <= x.entry_date.year <= 2025]
oos = [x for x in t if x.entry_date.year >= 2026]

print(f"  Train: {len(train)}t, {sum(1 for x in train if x.pnl > 0)/len(train)*100:.0f}% WR, ${sum(x.pnl for x in train):+,.0f}")
print(f"  Test:  {len(test)}t, {sum(1 for x in test if x.pnl > 0)/len(test)*100:.0f}% WR, ${sum(x.pnl for x in test):+,.0f}")
if oos:
    print(f"  2026:  {len(oos)}t, {sum(1 for x in oos if x.pnl > 0)/len(oos)*100:.0f}% WR, ${sum(x.pnl for x in oos):+,.0f}")
else:
    print("  2026:  0 trades")

# Walk-forward
wf = sum(1 for yr in range(2017, 2026) for yearly in [[x for x in t if x.entry_date.year == yr]] if yearly and all(x.pnl > 0 for x in yearly))
print(f"  WF: {wf}/9 PASS")

# Year-by-year
print("\nYear-by-year:")
for year in range(2016, 2027):
    yearly = [x for x in t if x.entry_date.year == year]
    if not yearly: continue
    yn = len(yearly)
    yw = sum(1 for x in yearly if x.pnl > 0)
    ypnl = sum(x.pnl for x in yearly)
    ybl = min(x.pnl for x in yearly)
    marker = " <-- OOS" if year >= 2026 else ""
    print(f"  {year}: {yn:3d}t {yw}/{yn} wins ${ypnl:+9,.0f} BL=${ybl:+,.0f}{marker}")

# 2026 OOS trades
if oos:
    print("\n2026 OOS trades:")
    for x in sorted(oos, key=lambda x: x.entry_date):
        d = day_map.get(x.entry_date)
        src = "V5" if d and d.v5_take_bounce else "CS"
        print(f"  {str(x.entry_date)[:10]} {x.direction:5s} ${x.pnl:+8,.0f} src={src} h={d.cs_channel_health:.3f if d else 0}")

# Identify the V5 signal that fails at h=0.575
print("\nV5 signals near h=0.57 boundary:")
for d in signals:
    if d.v5_take_bounce and 0.56 <= d.cs_channel_health <= 0.59:
        print(f"  {d.date} h={d.cs_channel_health:.3f} pos={d.cs_position_score:.3f} v5c={d.v5_confidence or 0:.3f}")

print("\nDone")
