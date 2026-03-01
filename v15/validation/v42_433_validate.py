#!/usr/bin/env python3
"""Full validation: DC = DB + SELL Tue VIX<13 = 433 trades.

Stack:
- CZ (426t)
- + Fri confl>=0.80 & ent>=0.70 (+2t from DA)
- + SELL Thu VIX<15|SPY<-1% (+4t, -1 displaced from DB)
- + SELL Tue VIX<13 (+2t from DC)
= 433t 100% WR

Also explore further beyond 433."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, simulate_trades, _build_filter_cascade,
    _make_v27_ct, _make_v33_cz, MIN_SIGNAL_CONFIDENCE, _floor_stop_tp,
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
cz_fn = _make_v33_cz(*args)

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

def make_dc():
    """DC: CZ + Fri[confl>=0.80&ent>=0.70] + SELL Thu[VIX<15|SPY<-1%] + SELL Tue[VIX<13]."""
    def fn(day):
        result = cz_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            # Friday: confl>=0.80 & ent>=0.70
            if (dd.weekday() == 4 and day.cs_confluence_score >= 0.80 and
                day.cs_entropy_score is not None and day.cs_entropy_score >= 0.70):
                return result
            # SELL Thursday: VIX<15 or SPY<-1%
            if dd.weekday() == 3 and day.cs_action == 'SELL':
                vix_val = vix_map.get(day.date, 22)
                spy_d = spy_dist_map.get(day.date, 0)
                if vix_val < 15 or spy_d < -1.0:
                    return result
            # SELL Tuesday: VIX<13
            if dd.weekday() == 1 and day.cs_action == 'SELL':
                vix_val = vix_map.get(day.date, 22)
                if vix_val < 13:
                    return result
        return None
    return fn

print("="*70)
print("FULL VALIDATION: DC = DB + SELL Tue[VIX<13]")
print("="*70)

trades = simulate_trades(signals, make_dc(), 'DC', cooldown=0, trail_power=6)
n = len(trades)
w = sum(1 for t in trades if t.pnl > 0)
wr = w/n*100 if n else 0
pnl = sum(t.pnl for t in trades)
bl = min(t.pnl for t in trades) if trades else 0

print(f"\nDC: {n}t {wr:.1f}% WR ${pnl:+,.0f} BL=${bl:+,.0f}")

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

# Show new trades vs CZ
cz_trades = simulate_trades(signals, cz_fn, 'CZ', cooldown=0, trail_power=6)
cz_dates = {t.entry_date for t in cz_trades}
new_trades_list = [t for t in trades if t.entry_date not in cz_dates]
lost_trades_list = [t for t in cz_trades if t.entry_date not in {t2.entry_date for t2 in trades}]

print(f"\nNew vs CZ: +{len(new_trades_list)} new, -{len(lost_trades_list)} lost = net +{len(new_trades_list)-len(lost_trades_list)}")
for t in sorted(new_trades_list, key=lambda x: x.entry_date):
    dd = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
    day_sig = next((d for d in signals if d.date == t.entry_date), None)
    h_val = day_sig.cs_channel_health if day_sig else 0
    c_val = day_sig.cs_confluence_score if day_sig else 0
    vix_val = vix_map.get(t.entry_date, 0)
    print(f"  NEW: {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} h={h_val:.3f} confl={c_val:.2f} VIX={vix_val:.1f} {dow}")
if lost_trades_list:
    for t in sorted(lost_trades_list, key=lambda x: x.entry_date):
        dd = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        print(f"  LOST: {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} {dow}")

# ════════════════════════════════════════════════════════
# Push beyond DC (433t)
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Push beyond DC (433t)")
print("="*70)

dc_fn_cached = make_dc()

# BUY Tue conditions
print("\n--- BUY Tue conditions on DC ---")
for cond_label, cond_check in [
    ("VIX<12", lambda d: vix_map.get(d.date, 22) < 12),
    ("VIX<13", lambda d: vix_map.get(d.date, 22) < 13),
    ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
    ("VIX>25", lambda d: vix_map.get(d.date, 22) > 25),
    ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
    ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
    ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
    ("confl>=0.80&ent>=0.95", lambda d: d.cs_confluence_score >= 0.80 and
                                         d.cs_entropy_score is not None and d.cs_entropy_score >= 0.95),
    ("pos<0.90", lambda d: d.cs_position_score < 0.90),
    ("conf>=0.60", lambda d: d.cs_confidence >= 0.60),
]:
    def make_dc_buy_tue(cc=cond_check):
        def fn(day):
            result = dc_fn_cached(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 1 and day.cs_action == 'BUY' and cc(day):
                    return result
            return None
        return fn
    t_trades = simulate_trades(signals, make_dc_buy_tue(), 'test', cooldown=0, trail_power=6)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    if tn > 433:
        tpnl = sum(t.pnl for t in t_trades)
        tbl = min(t.pnl for t in t_trades) if t_trades else 0
        flag = "***" if twr >= 100 else ""
        print(f"  BUY Tue {cond_label}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

# SELL Tue wider VIX
print("\n--- SELL Tue wider VIX on DC ---")
for vix_th in [14, 15, 16, 17, 18]:
    def make_dc_tue_vix(vt=vix_th):
        def fn(day):
            result = dc_fn_cached(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 1 and day.cs_action == 'SELL':
                    vix_val = vix_map.get(day.date, 22)
                    if vix_val < vt:
                        return result
            return None
        return fn
    t_trades = simulate_trades(signals, make_dc_tue_vix(), 'test', cooldown=0, trail_power=6)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    if tn > 433:
        tpnl = sum(t.pnl for t in t_trades)
        tbl = min(t.pnl for t in t_trades) if t_trades else 0
        flag = "***" if twr >= 100 else ""
        print(f"  SELL Tue VIX<{vix_th}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

# SELL Tue other conditions
print("\n--- SELL Tue additional conditions on DC ---")
for cond_label, cond_check in [
    ("SPY<-0.5%", lambda d: spy_dist_map.get(d.date, 0) < -0.5),
    ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
    ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
    ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
    ("ent>=0.90", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.90),
    ("ent>=0.95", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.95),
    ("pos<0.90", lambda d: d.cs_position_score < 0.90),
    ("conf>=0.60", lambda d: d.cs_confidence >= 0.60),
    ("h>=0.30", lambda d: d.cs_channel_health >= 0.30),
]:
    def make_dc_sell_tue2(cc=cond_check):
        def fn(day):
            result = dc_fn_cached(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 1 and day.cs_action == 'SELL' and cc(day):
                    return result
            return None
        return fn
    t_trades = simulate_trades(signals, make_dc_sell_tue2(), 'test', cooldown=0, trail_power=6)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    if tn > 433:
        tpnl = sum(t.pnl for t in t_trades)
        tbl = min(t.pnl for t in t_trades) if t_trades else 0
        flag = "***" if twr >= 100 else ""
        print(f"  SELL Tue {cond_label}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

# BUY Thu conditions on DC
print("\n--- BUY Thu conditions on DC ---")
for cond_label, cond_check in [
    ("VIX<12", lambda d: vix_map.get(d.date, 22) < 12),
    ("VIX<13", lambda d: vix_map.get(d.date, 22) < 13),
    ("VIX>25", lambda d: vix_map.get(d.date, 22) > 25),
    ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
    ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
    ("ent>=0.95", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.95),
    ("pos<0.85", lambda d: d.cs_position_score < 0.85),
]:
    def make_dc_buy_thu(cc=cond_check):
        def fn(day):
            result = dc_fn_cached(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 3 and day.cs_action == 'BUY' and cc(day):
                    return result
            return None
        return fn
    t_trades = simulate_trades(signals, make_dc_buy_thu(), 'test', cooldown=0, trail_power=6)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    if tn > 433:
        tpnl = sum(t.pnl for t in t_trades)
        tbl = min(t.pnl for t in t_trades) if t_trades else 0
        flag = "***" if twr >= 100 else ""
        print(f"  BUY Thu {cond_label}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

# SELL Thu wider conditions on DC
print("\n--- SELL Thu wider on DC ---")
for cond_label, cond_check in [
    ("VIX<16", lambda d: vix_map.get(d.date, 22) < 16),
    ("VIX<17", lambda d: vix_map.get(d.date, 22) < 17),
    ("SPY<-0.5%", lambda d: spy_dist_map.get(d.date, 0) < -0.5),
    ("ent>=0.90", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.90),
    ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
    ("pos<0.90", lambda d: d.cs_position_score < 0.90),
]:
    def make_dc_sell_thu2(cc=cond_check):
        def fn(day):
            result = dc_fn_cached(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 3 and day.cs_action == 'SELL' and cc(day):
                    return result
            return None
        return fn
    t_trades = simulate_trades(signals, make_dc_sell_thu2(), 'test', cooldown=0, trail_power=6)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    if tn > 433:
        tpnl = sum(t.pnl for t in t_trades)
        tbl = min(t.pnl for t in t_trades) if t_trades else 0
        flag = "***" if twr >= 100 else ""
        print(f"  SELL Thu {cond_label}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

# Broader Fri conditions on DC
print("\n--- Broader Fri on DC ---")
for cond_label, cond_check in [
    ("confl>=0.70&ent>=0.70", lambda d: d.cs_confluence_score >= 0.70 and
                                         d.cs_entropy_score is not None and d.cs_entropy_score >= 0.70),
    ("confl>=0.70&ent>=0.90", lambda d: d.cs_confluence_score >= 0.70 and
                                         d.cs_entropy_score is not None and d.cs_entropy_score >= 0.90),
    ("SELL pos<0.90", lambda d: d.cs_action == 'SELL' and d.cs_position_score < 0.90),
    ("SELL VIX<15", lambda d: d.cs_action == 'SELL' and vix_map.get(d.date, 22) < 15),
]:
    def make_dc_fri(cc=cond_check):
        def fn(day):
            result = dc_fn_cached(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 4 and cc(day):
                    return result
            return None
        return fn
    t_trades = simulate_trades(signals, make_dc_fri(), 'test', cooldown=0, trail_power=6)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    if tn > 433:
        tpnl = sum(t.pnl for t in t_trades)
        tbl = min(t.pnl for t in t_trades) if t_trades else 0
        flag = "***" if twr >= 100 else ""
        print(f"  Fri {cond_label}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

print("\nDone")
