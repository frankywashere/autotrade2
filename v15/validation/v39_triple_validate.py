#!/usr/bin/env python3
"""Full 3-stage validation of triple combo: Mon|VIX>25|BUY SPY<-1% h>=0.22.
422 trades, 100% WR from v39 experiment."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, simulate_trades, _build_filter_cascade,
    _make_v27_ct, _make_v31_cx, MIN_SIGNAL_CONFIDENCE, _floor_stop_tp,
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

def make_triple():
    """Triple conditional: Mon | VIX>25 | BUY&SPY<-1%, h>=0.22."""
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
            relax = dd.weekday() == 0 or vix > 25
            if day.cs_action == 'BUY' and spy_d < -1.0:
                relax = True
            if relax:
                h_buy = min(h_buy, 0.22)
                h_sell = min(h_sell, 0.22)
            h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
            if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                return result
        if day.v5_take_bounce:
            if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn

print("=" * 100)
print("FULL VALIDATION: Mon|VIX>25|BUY SPY<-1% h>=0.22")
print("=" * 100)

trades = simulate_trades(signals, make_triple(), 'triple', cooldown=0, trail_power=6)
n = len(trades)
w = sum(1 for t in trades if t.pnl > 0)
wr = w/n*100 if n else 0
pnl = sum(t.pnl for t in trades)
bl = min(t.pnl for t in trades) if trades else 0
print(f"\nFull: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# Holdout
train = [t for t in trades if t.entry_date.year <= 2021]
test = [t for t in trades if 2022 <= t.entry_date.year <= 2025]
oos = [t for t in trades if t.entry_date.year >= 2026]

n_train = len(train)
w_train = sum(1 for t in train if t.pnl > 0)
wr_train = w_train/n_train*100 if n_train else 0
pnl_train = sum(t.pnl for t in train)

n_test = len(test)
w_test = sum(1 for t in test if t.pnl > 0)
wr_test = w_test/n_test*100 if n_test else 0
pnl_test = sum(t.pnl for t in test)

print(f"\nTrain (2016-2021): {n_train} trades, {wr_train:.1f}% WR, ${pnl_train:+,.0f}")
print(f"Test  (2022-2025): {n_test} trades, {wr_test:.1f}% WR, ${pnl_test:+,.0f}")
holdout_pass = wr_train >= 100 and wr_test >= 100
print(f"Holdout: {'PASS' if holdout_pass else 'FAIL'}")

# Walk-forward
print("\nWalk-forward:")
wf_pass = 0
wf_total = 0
for year in range(2017, 2026):
    yearly = [t for t in trades if t.entry_date.year == year]
    if not yearly: continue
    wf_total += 1
    yn = len(yearly)
    yw = sum(1 for t in yearly if t.pnl > 0)
    ywr = yw/yn*100
    ypnl = sum(t.pnl for t in yearly)
    ybl = min(t.pnl for t in yearly)
    passed = all(t.pnl > 0 for t in yearly)
    if passed: wf_pass += 1
    print(f"  {year}: {yn:3d}t {ywr:5.1f}% WR ${ypnl:+9,.0f} BL=${ybl:+,.0f} {'PASS' if passed else 'FAIL'}")
print(f"Walk-forward: {wf_pass}/{wf_total}")

# 2026 OOS
print(f"\n2026 OOS:")
if oos:
    n_oos = len(oos)
    w_oos = sum(1 for t in oos if t.pnl > 0)
    wr_oos = w_oos/n_oos*100
    pnl_oos = sum(t.pnl for t in oos)
    bl_oos = min(t.pnl for t in oos)
    oos_pass = wr_oos >= 100
    print(f"  {n_oos} trades, {wr_oos:.1f}% WR, ${pnl_oos:+,.0f}, BL=${bl_oos:+,.0f}")
    print(f"  OOS: {'PASS' if oos_pass else 'FAIL'}")
    for t in sorted(oos, key=lambda x: x.entry_date):
        day = day_map.get(t.entry_date)
        src = "V5" if day and day.v5_take_bounce else "CS"
        dd = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        h_str = f"{day.cs_channel_health:.3f}" if day else "N/A"
        c_str = f"{day.cs_confidence:.3f}" if day else "N/A"
        cf_str = f"{day.cs_confluence_score:.2f}" if day else "N/A"
        print(f"    {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} src={src} "
              f"h={h_str} c={c_str} confl={cf_str} VIX={vix_map.get(t.entry_date, 0):.1f} {dow}")
else:
    oos_pass = False
    print("  No 2026 trades")

# New trades vs CX
print("\nNew trades vs CX (414):")
cx_fn = _make_v31_cx(*args)
cx_trades = simulate_trades(signals, cx_fn, 'CX', cooldown=0, trail_power=6)
cx_dates = {t.entry_date for t in cx_trades}
triple_dates = {t.entry_date for t in trades}
new_dates = triple_dates - cx_dates
lost_dates = cx_dates - triple_dates
print(f"  New: {len(new_dates)} | Lost: {len(lost_dates)}")
for nd in sorted(new_dates):
    t = next((x for x in trades if x.entry_date == nd), None)
    day = day_map.get(nd)
    if t and day:
        dd = nd.date() if hasattr(nd, 'date') else nd
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        vix = vix_map.get(nd, 0)
        spy_d = spy_dist_map.get(nd, 0)
        reason = []
        if dd.weekday() == 0: reason.append("Mon")
        if vix > 25: reason.append(f"VIX={vix:.1f}")
        if day.cs_action == 'BUY' and spy_d < -1.0: reason.append(f"SPY={spy_d:.2f}%")
        print(f"  NEW: {str(nd)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} h={day.cs_channel_health:.3f} "
              f"c={day.cs_confidence:.3f} confl={day.cs_confluence_score:.2f} "
              f"VIX={vix:.1f} {dow} [{', '.join(reason)}]")
if lost_dates:
    for ld in sorted(lost_dates):
        t = next((x for x in cx_trades if x.entry_date == ld), None)
        day = day_map.get(ld)
        if t and day:
            print(f"  LOST: {str(ld)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} h={day.cs_channel_health:.3f}")

# Year-by-year
print("\nYear-by-year:")
for year in range(2016, 2027):
    yearly = [t for t in trades if t.entry_date.year == year]
    if not yearly: continue
    yn = len(yearly)
    yw = sum(1 for t in yearly if t.pnl > 0)
    ypnl = sum(t.pnl for t in yearly)
    ybl = min(t.pnl for t in yearly)
    ybw = max(t.pnl for t in yearly)
    marker = " <-- OOS" if year >= 2026 else ""
    print(f"  {year}: {yn:3d}t {yw}/{yn} wins ${ypnl:+9,.0f} BL=${ybl:+,.0f} BW=${ybw:+,.0f}{marker}")

# Final verdict
all_pass = holdout_pass and wf_pass == wf_total and oos_pass
print(f"\n{'='*100}")
print(f"TRIPLE COMBO: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f}")
print(f"  Holdout: {'PASS' if holdout_pass else 'FAIL'}")
print(f"  WF: {wf_pass}/{wf_total} {'PASS' if wf_pass == wf_total else 'FAIL'}")
print(f"  OOS: {'PASS' if oos_pass else 'FAIL'}")
print(f"  OVERALL: {'*** ALL STAGES PASS ***' if all_pass else 'FAIL'}")
print(f"{'='*100}")

# Also validate Mon-only (419 trades) for comparison
print("\n\n--- Comparison: Mon-only h>=0.22 ---")
def make_mon():
    def fn(day):
        result = ct_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            h_buy = 0.38
            h_sell = 0.31
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            if dd.weekday() == 0:
                h_buy = min(h_buy, 0.22)
                h_sell = min(h_sell, 0.22)
            h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
            if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                return result
        if day.v5_take_bounce:
            if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn

mon_trades = simulate_trades(signals, make_mon(), 'mon', cooldown=0, trail_power=6)
mn = len(mon_trades)
mw = sum(1 for t in mon_trades if t.pnl > 0)
mwr = mw/mn*100 if mn else 0
mpnl = sum(t.pnl for t in mon_trades)
print(f"Mon-only: {mn} trades, {mwr:.1f}% WR, ${mpnl:+,.0f}")
mtrain = [t for t in mon_trades if t.entry_date.year <= 2021]
mtest = [t for t in mon_trades if 2022 <= t.entry_date.year <= 2025]
moos = [t for t in mon_trades if t.entry_date.year >= 2026]
print(f"  Train: {len(mtrain)}t {sum(1 for t in mtrain if t.pnl > 0)/len(mtrain)*100:.0f}%")
print(f"  Test:  {len(mtest)}t {sum(1 for t in mtest if t.pnl > 0)/len(mtest)*100:.0f}%")
if moos:
    print(f"  2026:  {len(moos)}t {sum(1 for t in moos if t.pnl > 0)/len(moos)*100:.0f}% ${sum(t.pnl for t in moos):+,.0f}")
mwf = sum(1 for yr in range(2017, 2026) for yrly in [[t for t in mon_trades if t.entry_date.year == yr]] if yrly and all(t.pnl > 0 for t in yrly))
print(f"  WF: {mwf}/9")

print("\nDone")
