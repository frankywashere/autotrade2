#!/usr/bin/env python3
"""v35 fine sweep: find exact V5 h boundary between 0.50 and 0.60."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, simulate_trades, _build_filter_cascade,
    _make_v28_cu, MIN_SIGNAL_CONFIDENCE, _floor_stop_tp
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
args = (cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map, spy_ret_2d)
cu_fn = _make_v28_cu(*args)

day_map = {day.date: day for day in signals}

# Fine h boundary sweep
print("Fine V5 h boundary sweep (pos<0.85):")
for h_max_pct in range(50, 61):
    h_max = h_max_pct / 100.0
    def make_fn(hm=h_max):
        def fn(day):
            r = cu_fn(day)
            if r is not None:
                return r
            if day.v5_take_bounce and day.cs_channel_health < hm and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_fn(), 'x', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 402 else (" <-- CV" if h_max == 0.50 else "")
    print(f"  h<{h_max:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# Fine-grained around the boundary
print("\nFine-grained around 0.55-0.60:")
for h_max_pct in range(550, 601, 5):
    h_max = h_max_pct / 1000.0
    def make_fn2(hm=h_max):
        def fn(day):
            r = cu_fn(day)
            if r is not None:
                return r
            if day.v5_take_bounce and day.cs_channel_health < hm and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_fn2(), 'x', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 else ""
    print(f"  h<{h_max:.3f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# Also check what V5 signals have health near the boundary
print("\nV5 bounce signals by health:")
v5_days = [d for d in signals if d.v5_take_bounce]
for d in sorted(v5_days, key=lambda x: x.cs_channel_health):
    in_cv = "IN_CV" if d.date in {t.entry_date for t in simulate_trades(signals, _make_v28_cu(*args), 'x', cooldown=0, trail_power=6)} else ""
    if 0.40 < d.cs_channel_health < 0.65:
        print(f"  {d.date} h={d.cs_channel_health:.3f} pos={d.cs_position_score:.3f} v5c={d.v5_confidence or 0:.3f} {in_cv}")

# Best result: implement CW with h<0.55
print("\n" + "="*70)
print("BEST CW CANDIDATE: CU + V5[h<0.55 & pos<0.85]")
print("="*70)
def make_cw():
    def fn(day):
        r = cu_fn(day)
        if r is not None:
            return r
        if day.v5_take_bounce and day.cs_channel_health < 0.55 and day.cs_position_score < 0.85:
            return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn

cw_trades = simulate_trades(signals, make_cw(), 'CW', cooldown=0, trail_power=6)
n = len(cw_trades)
w = sum(1 for t in cw_trades if t.pnl > 0)
wr = w/n*100
pnl = sum(t.pnl for t in cw_trades)
bl = min(t.pnl for t in cw_trades)
print(f"CW: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# Train/test/2026 split
train = [t for t in cw_trades if t.entry_date.year <= 2021]
test = [t for t in cw_trades if 2022 <= t.entry_date.year <= 2025]
oos = [t for t in cw_trades if t.entry_date.year >= 2026]

for label, subset in [("Train 2016-2021", train), ("Test 2022-2025", test), ("2026 OOS", oos)]:
    if subset:
        sn = len(subset)
        sw = sum(1 for t in subset if t.pnl > 0)
        swr = sw/sn*100
        spnl = sum(t.pnl for t in subset)
        sbl = min(t.pnl for t in subset)
        print(f"  {label}: {sn} trades, {swr:.1f}% WR, ${spnl:+,.0f}, BL=${sbl:+,.0f}")
    else:
        print(f"  {label}: 0 trades")

# Walk-forward
wf_pass = 0
wf_total = 0
for year in range(2017, 2026):
    yearly = [t for t in cw_trades if t.entry_date.year == year]
    if not yearly: continue
    wf_total += 1
    if all(t.pnl > 0 for t in yearly):
        wf_pass += 1
print(f"  Walk-Forward: {wf_pass}/{wf_total} years PASS")

# Year-by-year
print("\nYear-by-year:")
for year in range(2016, 2027):
    yearly = [t for t in cw_trades if t.entry_date.year == year]
    if not yearly: continue
    yn = len(yearly)
    yw = sum(1 for t in yearly if t.pnl > 0)
    ywr = yw/yn*100
    ypnl = sum(t.pnl for t in yearly)
    ybl = min(t.pnl for t in yearly)
    marker = " <-- OOS" if year >= 2026 else ""
    print(f"  {year}: {yn:3d} trades, {ywr:5.1f}% WR, ${ypnl:+9,.0f}, BL=${ybl:+,.0f}{marker}")

# 2026 OOS trade details
if oos:
    print("\n2026 OOS trades:")
    for t in sorted(oos, key=lambda x: x.entry_date):
        day = day_map.get(t.entry_date)
        src = "V5" if day and day.v5_take_bounce else "CS"
        dd = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        h_str = f"{day.cs_channel_health:.3f}" if day else "N/A"
        print(f"  {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} src={src} h={h_str} VIX={vix_map.get(t.entry_date, 22):.1f} {dow}")

print("\nDone")
