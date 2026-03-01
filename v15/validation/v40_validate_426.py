#!/usr/bin/env python3
"""Quick validation: CY + Wed h>=0.14 = 426 trades."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, simulate_trades, _build_filter_cascade,
    _make_v27_ct, MIN_SIGNAL_CONFIDENCE, _floor_stop_tp,
    _SigProxy, _AnalysisProxy
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

def _tf0_base(day):
    if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
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

def make_cz():
    """CZ: CY + Wed h>=0.14 = 426 trades."""
    def fn(day):
        result = ct_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            h_buy = 0.38
            h_sell = 0.31
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            vix = vix_map.get(day.date, 22)
            spy_d = spy_dist_map.get(day.date, 0)
            # CY conditions: Mon|VIX>25|BUY&SPY<-1% → h=0.22
            relax = dd.weekday() == 0 or vix > 25
            if day.cs_action == 'BUY' and spy_d < -1.0:
                relax = True
            if relax:
                h_buy = min(h_buy, 0.22)
                h_sell = min(h_sell, 0.22)
            # Wednesday → h=0.14
            if dd.weekday() == 2:
                h_buy = min(h_buy, 0.14)
                h_sell = min(h_sell, 0.14)
            h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
            if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                return result
        if day.v5_take_bounce:
            if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn

trades = simulate_trades(signals, make_cz(), 'CZ', cooldown=0, trail_power=6)
n = len(trades)
w = sum(1 for t in trades if t.pnl > 0)
wr = w/n*100 if n else 0
pnl = sum(t.pnl for t in trades)
bl = min(t.pnl for t in trades) if trades else 0
print(f"CZ: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

train = [t for t in trades if t.entry_date.year <= 2021]
test = [t for t in trades if 2022 <= t.entry_date.year <= 2025]
oos = [t for t in trades if t.entry_date.year >= 2026]

nt = len(train); wt = sum(1 for t in train if t.pnl > 0)
ne = len(test); we = sum(1 for t in test if t.pnl > 0)
print(f"Train: {nt}t {wt/nt*100:.0f}% WR ${sum(t.pnl for t in train):+,.0f}")
print(f"Test:  {ne}t {we/ne*100:.0f}% WR ${sum(t.pnl for t in test):+,.0f}")

wf_pass = 0
wf_total = 0
for year in range(2017, 2026):
    yearly = [t for t in trades if t.entry_date.year == year]
    if not yearly: continue
    wf_total += 1
    passed = all(t.pnl > 0 for t in yearly)
    if passed: wf_pass += 1
    yn = len(yearly)
    yw = sum(1 for t in yearly if t.pnl > 0)
    ybl = min(t.pnl for t in yearly)
    print(f"  {year}: {yn:3d}t {yw}/{yn} BL=${ybl:+,.0f} {'PASS' if passed else 'FAIL'}")

if oos:
    no = len(oos); wo = sum(1 for t in oos if t.pnl > 0)
    oos_pass = wo == no
    print(f"2026: {no}t {wo}/{no} ${sum(t.pnl for t in oos):+,.0f} {'PASS' if oos_pass else 'FAIL'}")
else:
    oos_pass = False

all_pass = wt == nt and we == ne and wf_pass == wf_total and oos_pass
print(f"\nOVERALL: {'*** ALL PASS ***' if all_pass else 'FAIL'} ({wf_pass}/{wf_total} WF)")

# New trade vs 425-trade version
print("\nExtra trade (426 vs 425):")
trades_425 = simulate_trades(signals, make_cz.__wrapped__ if hasattr(make_cz, '__wrapped__') else None, 'test', cooldown=0, trail_power=6) if False else None
# Just check what's at h=0.14 boundary
for day in signals:
    dd = day.date.date() if hasattr(day.date, 'date') else day.date
    if dd.weekday() == 2:  # Wednesday
        result = _tf0_base(day)
        if result is not None:
            if day.cs_channel_health >= 0.14 and day.cs_channel_health < 0.16:
                print(f"  Wed boundary: {day.date} h={day.cs_channel_health:.3f} c={day.cs_confidence:.3f} "
                      f"confl={day.cs_confluence_score:.2f} {day.cs_action}")

print("\nDone")
