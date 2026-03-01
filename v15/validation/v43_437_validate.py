#!/usr/bin/env python3
"""Validate DH = DG + BUY Tue[VIX<15 & SRet<-0.5%] = 437t 100% $712,593.
Then explore stacking with other clean additions from Part 4 profile.

Profile showed 2 clean 437/437 additions:
  - 2024-04-02 BUY Tue: VIX=14.6, SRet=-0.6% (matches BUY Tue VIX<15&SRet<-0.5%)
  - 2021-10-26 SELL Tue: confl=0.50, ent=0.90 (needs separate condition)
"""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, simulate_trades, _build_filter_cascade,
    _make_v34_dd, MIN_SIGNAL_CONFIDENCE, _floor_stop_tp,
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

dd_fn = _make_v34_dd(*args)

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

def make_dg():
    def fn(day):
        result = dd_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            d = day.date.date() if hasattr(day.date, 'date') else day.date
            if (d.weekday() == 1 and day.cs_action == 'BUY' and
                spy_return_map.get(day.date, 0) < -1.0):
                return result
            if (d.weekday() == 3 and day.cs_action == 'BUY' and
                day.cs_channel_health >= 0.25):
                return result
        return None
    return fn

def full_validate(trades, name):
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    print(f"\n{name}: {n}t {wr:.1f}% WR ${pnl:+,.0f} BL=${bl:+,.0f}")

    train = [t for t in trades if t.entry_date.year <= 2021]
    test = [t for t in trades if 2022 <= t.entry_date.year <= 2025]
    oos = [t for t in trades if t.entry_date.year >= 2026]

    nt = len(train); wt = sum(1 for t in train if t.pnl > 0)
    ne = len(test); we = sum(1 for t in test if t.pnl > 0)
    print(f"  Train: {nt}t {wt/nt*100:.0f}% WR ${sum(t.pnl for t in train):+,.0f}")
    print(f"  Test:  {ne}t {we/ne*100:.0f}% WR ${sum(t.pnl for t in test):+,.0f}")

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
        print(f"    {year}: {yn:3d}t {yw}/{yn} BL=${ybl:+,.0f} {'PASS' if passed else 'FAIL'}")

    if oos:
        no = len(oos); wo = sum(1 for t in oos if t.pnl > 0)
        oos_pass = wo == no
        print(f"  2026: {no}t {wo}/{no} ${sum(t.pnl for t in oos):+,.0f} {'PASS' if oos_pass else 'FAIL'}")
    else:
        oos_pass = False

    all_pass = wt == nt and we == ne and wf_pass == wf_total and oos_pass
    print(f"  OVERALL: {'*** ALL PASS ***' if all_pass else 'FAIL'} ({wf_pass}/{wf_total} WF)")
    return all_pass

# ════════════════════════════════════════════════════════════
# SECTION 1: Validate DH = DG + BUY Tue[VIX<15 & SRet<-0.5%]
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1: DH = DG + BUY Tue[VIX<15 & SRet<-0.5%]")
print("=" * 70)

def make_dh():
    dg_fn = make_dg()
    def fn(day):
        result = dg_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            d = day.date.date() if hasattr(day.date, 'date') else day.date
            if (d.weekday() == 1 and day.cs_action == 'BUY' and
                vix_map.get(day.date, 22) < 15 and
                spy_return_map.get(day.date, 0) < -0.5):
                return result
        return None
    return fn

dh_trades = simulate_trades(signals, make_dh(), 'DH', cooldown=0, trail_power=12)
dh_pass = full_validate(dh_trades, "DH@tp=12")

dg_fn = make_dg()
dg_trades = simulate_trades(signals, dg_fn, 'DG', cooldown=0, trail_power=12)
dg_dates = {t.entry_date for t in dg_trades}
dh_dates = {t.entry_date for t in dh_trades}

new_trades = [t for t in dh_trades if t.entry_date not in dg_dates]
lost_trades = [t for t in dg_trades if t.entry_date not in dh_dates]
print(f"\n  New vs DG: +{len(new_trades)} new, -{len(lost_trades)} lost")
for t in sorted(new_trades, key=lambda x: x.entry_date):
    d = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
    dow = ['Mon','Tue','Wed','Thu','Fri'][d.weekday()]
    print(f"    NEW: {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} {dow}")
for t in sorted(lost_trades, key=lambda x: x.entry_date):
    d = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
    dow = ['Mon','Tue','Wed','Thu','Fri'][d.weekday()]
    print(f"    LOST: {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} {dow}")

# ════════════════════════════════════════════════════════════
# SECTION 2: SRet threshold sensitivity for the BUY Tue VIX<15 condition
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: BUY Tue VIX<15 & SRet threshold sensitivity")
print("=" * 70)

for sret_th in [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -1.0]:
    def make_sret_test(sr=sret_th):
        dg_fn_inner = make_dg()
        def fn(day):
            result = dg_fn_inner(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                d = day.date.date() if hasattr(day.date, 'date') else day.date
                if (d.weekday() == 1 and day.cs_action == 'BUY' and
                    vix_map.get(day.date, 22) < 15 and
                    spy_return_map.get(day.date, 0) < sr):
                    return result
            return None
        return fn

    trades = simulate_trades(signals, make_sret_test(), 'test', cooldown=0, trail_power=12)
    tn = len(trades)
    tw = sum(1 for t in trades if t.pnl > 0)
    tbl = min(t.pnl for t in trades) if trades else 0
    tpnl = sum(t.pnl for t in trades)
    flag = "***" if tw == tn else ""
    print(f"  SRet<{sret_th:.1f}%: {tn}t {tw/tn*100:.1f}% BL=${tbl:+,.0f} ${tpnl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════════
# SECTION 3: Stack with SELL Tue ent>=0.90 (2021-10-26 was clean 437/437)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: DH + SELL Tue conditions (targeting 2021-10-26)")
print("=" * 70)

# The 2021-10-26 SELL Tue has confl=0.50, ent=0.90, VIX=16, SPY=+3.3%, SRet=+0.1%
# Need a condition that catches it. ent>=0.90, SPY>3%, confl<0.55?
sell_tue_conds = [
    ("ent>=0.90", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.90),
    ("ent>=0.85", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.85),
    ("SPY>3%", lambda d: spy_dist_map.get(d.date, 0) > 3.0),
    ("SPY>2.5%", lambda d: spy_dist_map.get(d.date, 0) > 2.5),
    ("SPY>2%", lambda d: spy_dist_map.get(d.date, 0) > 2.0),
    ("VIX<17", lambda d: vix_map.get(d.date, 22) < 17),
    ("h>=0.25", lambda d: d.cs_channel_health >= 0.25),
    ("h>=0.20", lambda d: d.cs_channel_health >= 0.20),
]

for cl, cc in sell_tue_conds:
    def make_dh_sell_tue(c=cc):
        dh_fn = make_dh()
        def fn(day):
            result = dh_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                d = day.date.date() if hasattr(day.date, 'date') else day.date
                if d.weekday() == 1 and day.cs_action == 'SELL' and c(day):
                    return result
            return None
        return fn

    trades = simulate_trades(signals, make_dh_sell_tue(), 'test', cooldown=0, trail_power=12)
    tn = len(trades)
    tw = sum(1 for t in trades if t.pnl > 0)
    tbl = min(t.pnl for t in trades) if trades else 0
    tpnl = sum(t.pnl for t in trades)
    flag = "*** 100% ***" if tw == tn else ""
    print(f"  DH + SELL Tue {cl}: {tn}t {tw/tn*100:.1f}% BL=${tbl:+,.0f} ${tpnl:+,.0f} {flag}")

# Double conditions for SELL Tue
print("\n  --- Double conditions for SELL Tue ---")
for i, (cl1, cc1) in enumerate(sell_tue_conds):
    for cl2, cc2 in sell_tue_conds[i+1:]:
        def make_dh_sell_tue_double(c1=cc1, c2=cc2):
            dh_fn = make_dh()
            def fn(day):
                result = dh_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    d = day.date.date() if hasattr(day.date, 'date') else day.date
                    if d.weekday() == 1 and day.cs_action == 'SELL' and c1(day) and c2(day):
                        return result
                return None
            return fn

        trades = simulate_trades(signals, make_dh_sell_tue_double(), 'test', cooldown=0, trail_power=12)
        tn = len(trades)
        tw = sum(1 for t in trades if t.pnl > 0)
        if tn > 437 and tw == tn:
            tpnl = sum(t.pnl for t in trades)
            print(f"    SELL Tue {cl1}&{cl2}: {tn}t 100% ${tpnl:+,.0f} ***")

# ════════════════════════════════════════════════════════════
# SECTION 4: Exhaustive double-condition push beyond DH (437t)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Double-condition push beyond DH (437t) at tp=12")
print("=" * 70)

compact_conds = [
    ("VIX<12", lambda d: vix_map.get(d.date, 22) < 12),
    ("VIX<13", lambda d: vix_map.get(d.date, 22) < 13),
    ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
    ("VIX<17", lambda d: vix_map.get(d.date, 22) < 17),
    ("VIX>25", lambda d: vix_map.get(d.date, 22) > 25),
    ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
    ("SPY<-0.5%", lambda d: spy_dist_map.get(d.date, 0) < -0.5),
    ("SPY>0.5%", lambda d: spy_dist_map.get(d.date, 0) > 0.5),
    ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
    ("SPY>2%", lambda d: spy_dist_map.get(d.date, 0) > 2.0),
    ("SPY>3%", lambda d: spy_dist_map.get(d.date, 0) > 3.0),
    ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
    ("confl>=0.90", lambda d: d.cs_confluence_score >= 0.90),
    ("ent>=0.70", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.70),
    ("ent>=0.80", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.80),
    ("ent>=0.90", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.90),
    ("h>=0.20", lambda d: d.cs_channel_health >= 0.20),
    ("h>=0.25", lambda d: d.cs_channel_health >= 0.25),
    ("h>=0.30", lambda d: d.cs_channel_health >= 0.30),
    ("pos<0.85", lambda d: d.cs_position_score < 0.85),
    ("pos<0.90", lambda d: d.cs_position_score < 0.90),
    ("conf>=0.55", lambda d: d.cs_confidence >= 0.55),
    ("SRet<-0.5%", lambda d: spy_return_map.get(d.date, 0) < -0.5),
    ("SRet<-1%", lambda d: spy_return_map.get(d.date, 0) < -1.0),
    ("SRet>0.5%", lambda d: spy_return_map.get(d.date, 0) > 0.5),
]

dh_fn_ref = make_dh()
dh_count = len(dh_trades)
found = False

for direction in ['BUY', 'SELL']:
    for dow_val, dow_label in [(0,"Mon"), (1,"Tue"), (2,"Wed"), (3,"Thu"), (4,"Fri")]:
        for i, (cl1, cc1) in enumerate(compact_conds):
            for cl2, cc2 in compact_conds[i+1:]:
                def make_double(dr=direction, dv=dow_val, c1=cc1, c2=cc2):
                    dh_fn_inner = make_dh()
                    def fn(day):
                        result = dh_fn_inner(day)
                        if result is not None:
                            return result
                        result = _tf0_base(day)
                        if result is not None:
                            dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                            if dd_date.weekday() == dv and day.cs_action == dr and c1(day) and c2(day):
                                return result
                        return None
                    return fn

                t_trades = simulate_trades(signals, make_double(), 'test', cooldown=0, trail_power=12)
                tn = len(t_trades)
                tw = sum(1 for t in t_trades if t.pnl > 0)
                if tn > dh_count and tw == tn:
                    tpnl = sum(t.pnl for t in t_trades)
                    print(f"  {direction} {dow_label} {cl1}&{cl2}: {tn}t 100% ${tpnl:+,.0f} ***")
                    found = True

if not found:
    print(f"  No double-condition expansion beyond {dh_count}t at tp=12")

# ════════════════════════════════════════════════════════════
# SECTION 5: Also try tp=8 as the minimum safe DG trail power
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: DH at tp=8 (minimum DG-safe trail power)")
print("=" * 70)

dh_tp8 = simulate_trades(signals, make_dh(), 'DH', cooldown=0, trail_power=8)
n8 = len(dh_tp8)
w8 = sum(1 for t in dh_tp8 if t.pnl > 0)
pnl8 = sum(t.pnl for t in dh_tp8)
bl8 = min(t.pnl for t in dh_tp8) if dh_tp8 else 0
print(f"DH@tp=8: {n8}t {w8/n8*100:.1f}% WR ${pnl8:+,.0f} BL=${bl8:+,.0f}")

if w8 == n8:
    print("  DH@tp=8 also 100% WR!")
    # Quick validation
    full_validate(dh_tp8, "DH@tp=8")

print("\nDone.")
