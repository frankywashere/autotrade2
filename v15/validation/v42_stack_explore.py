#!/usr/bin/env python3
"""Stack confirmed 100% WR expansions and check for additive trade gains.

Confirmed hits beyond CZ (426t):
1. Fri confl>=0.80 & ent>=0.70 → 428t (DA)
2. SELL Thu VIX<15 → 428t
3. SELL Fri pos<0.90 → 427t
4. SELL Thu SPY<-1% → 427t

Questions:
- Are these overlapping or additive?
- Can stacking them push to 430+?
- Full validation of best stacked combo
"""
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

cz_trades = simulate_trades(signals, cz_fn, 'CZ', cooldown=0, trail_power=6)
cz_dates = {t.entry_date for t in cz_trades}
print(f"CZ baseline: {len(cz_trades)} trades")

# ════════════════════════════════════════════════════════
# Individual component analysis — what new trades does each add?
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Individual component trade analysis")
print("="*70)

components = [
    ("Fri confl>=0.80&ent>=0.70",
     lambda day, dd: dd.weekday() == 4 and day.cs_confluence_score >= 0.80 and
                     day.cs_entropy_score is not None and day.cs_entropy_score >= 0.70),
    ("SELL Thu VIX<15",
     lambda day, dd: dd.weekday() == 3 and day.cs_action == 'SELL' and vix_map.get(day.date, 22) < 15),
    ("SELL Fri pos<0.90",
     lambda day, dd: dd.weekday() == 4 and day.cs_action == 'SELL' and day.cs_position_score < 0.90),
    ("SELL Thu SPY<-1%",
     lambda day, dd: dd.weekday() == 3 and day.cs_action == 'SELL' and spy_dist_map.get(day.date, 0) < -1.0),
]

for label, check_fn in components:
    def make_comp(cf=check_fn):
        def fn(day):
            result = cz_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if cf(day, dd):
                    return result
            return None
        return fn
    comp_trades = simulate_trades(signals, make_comp(), 'test', cooldown=0, trail_power=6)
    comp_dates = {t.entry_date for t in comp_trades}
    new_dates = comp_dates - cz_dates
    lost_dates = cz_dates - comp_dates

    tn = len(comp_trades)
    tw = sum(1 for t in comp_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    tpnl = sum(t.pnl for t in comp_trades)

    print(f"\n{label}: {tn}t {twr:.1f}% ${tpnl:+,.0f}")
    print(f"  New entry dates: {sorted(str(d)[:10] for d in new_dates)}")
    if lost_dates:
        print(f"  Lost entry dates: {sorted(str(d)[:10] for d in lost_dates)}")
    for t in comp_trades:
        if t.entry_date in new_dates:
            dd = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
            dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
            day_sig = next((d for d in signals if d.date == t.entry_date), None)
            h_val = day_sig.cs_channel_health if day_sig else 0
            c_val = day_sig.cs_confluence_score if day_sig else 0
            ent_val = day_sig.cs_entropy_score if day_sig else 0
            pos_val = day_sig.cs_position_score if day_sig else 0
            vix_val = vix_map.get(t.entry_date, 0)
            print(f"    {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} "
                  f"h={h_val:.3f} confl={c_val:.2f} ent={ent_val:.3f} pos={pos_val:.3f} "
                  f"VIX={vix_val:.1f} {dow}")

# ════════════════════════════════════════════════════════
# Pairwise stacking
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Pairwise stacking")
print("="*70)

pairs = [
    ("DA + SELL_Thu_VIX15",
     lambda day, dd: (dd.weekday() == 4 and day.cs_confluence_score >= 0.80 and
                      day.cs_entropy_score is not None and day.cs_entropy_score >= 0.70) or
                     (dd.weekday() == 3 and day.cs_action == 'SELL' and vix_map.get(day.date, 22) < 15)),
    ("DA + SELL_Fri_pos90",
     lambda day, dd: (dd.weekday() == 4 and day.cs_confluence_score >= 0.80 and
                      day.cs_entropy_score is not None and day.cs_entropy_score >= 0.70) or
                     (dd.weekday() == 4 and day.cs_action == 'SELL' and day.cs_position_score < 0.90)),
    ("DA + SELL_Thu_SPY",
     lambda day, dd: (dd.weekday() == 4 and day.cs_confluence_score >= 0.80 and
                      day.cs_entropy_score is not None and day.cs_entropy_score >= 0.70) or
                     (dd.weekday() == 3 and day.cs_action == 'SELL' and spy_dist_map.get(day.date, 0) < -1.0)),
    ("SELL_Thu_VIX15 + SELL_Fri_pos90",
     lambda day, dd: (dd.weekday() == 3 and day.cs_action == 'SELL' and vix_map.get(day.date, 22) < 15) or
                     (dd.weekday() == 4 and day.cs_action == 'SELL' and day.cs_position_score < 0.90)),
    ("SELL_Thu_VIX15 + SELL_Thu_SPY",
     lambda day, dd: dd.weekday() == 3 and day.cs_action == 'SELL' and
                     (vix_map.get(day.date, 22) < 15 or spy_dist_map.get(day.date, 0) < -1.0)),
]

for label, check_fn in pairs:
    def make_pair(cf=check_fn):
        def fn(day):
            result = cz_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if cf(day, dd):
                    return result
            return None
        return fn
    pair_trades = simulate_trades(signals, make_pair(), 'test', cooldown=0, trail_power=6)
    tn = len(pair_trades)
    tw = sum(1 for t in pair_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    tpnl = sum(t.pnl for t in pair_trades)
    tbl = min(t.pnl for t in pair_trades) if pair_trades else 0
    flag = "***" if twr >= 100 else ""
    print(f"  {label}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Triple stacking
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Triple stacking")
print("="*70)

def make_triple_stack():
    def fn(day):
        result = cz_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            # DA: Fri confl>=0.80&ent>=0.70
            if (dd.weekday() == 4 and day.cs_confluence_score >= 0.80 and
                day.cs_entropy_score is not None and day.cs_entropy_score >= 0.70):
                return result
            # SELL Thu VIX<15
            if dd.weekday() == 3 and day.cs_action == 'SELL' and vix_map.get(day.date, 22) < 15:
                return result
            # SELL Thu SPY<-1%
            if dd.weekday() == 3 and day.cs_action == 'SELL' and spy_dist_map.get(day.date, 0) < -1.0:
                return result
        return None
    return fn

triple_trades = simulate_trades(signals, make_triple_stack(), 'test', cooldown=0, trail_power=6)
tn = len(triple_trades)
tw = sum(1 for t in triple_trades if t.pnl > 0)
twr = tw/tn*100 if tn else 0
tpnl = sum(t.pnl for t in triple_trades)
tbl = min(t.pnl for t in triple_trades) if triple_trades else 0
flag = "***" if twr >= 100 else ""
print(f"  DA+SELL_Thu_VIX15+SELL_Thu_SPY: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

# All 4
def make_quad_stack():
    def fn(day):
        result = cz_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            # DA: Fri confl>=0.80&ent>=0.70
            if (dd.weekday() == 4 and day.cs_confluence_score >= 0.80 and
                day.cs_entropy_score is not None and day.cs_entropy_score >= 0.70):
                return result
            # SELL Thu VIX<15
            if dd.weekday() == 3 and day.cs_action == 'SELL' and vix_map.get(day.date, 22) < 15:
                return result
            # SELL Thu SPY<-1%
            if dd.weekday() == 3 and day.cs_action == 'SELL' and spy_dist_map.get(day.date, 0) < -1.0:
                return result
            # SELL Fri pos<0.90
            if dd.weekday() == 4 and day.cs_action == 'SELL' and day.cs_position_score < 0.90:
                return result
        return None
    return fn

quad_trades = simulate_trades(signals, make_quad_stack(), 'test', cooldown=0, trail_power=6)
tn = len(quad_trades)
tw = sum(1 for t in quad_trades if t.pnl > 0)
twr = tw/tn*100 if tn else 0
tpnl = sum(t.pnl for t in quad_trades)
tbl = min(t.pnl for t in quad_trades) if quad_trades else 0
flag = "***" if twr >= 100 else ""
print(f"  DA+SELL_Thu_VIX15+SELL_Thu_SPY+SELL_Fri_pos90: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# If we found 100% WR stacks, full validate the best one
# ════════════════════════════════════════════════════════
# Best individual: DA (428t), SELL Thu VIX<15 (428t)
# Test DA + SELL Thu VIX<15 full validation if it passes
print("\n" + "="*70)
print("Full validation: DA + SELL Thu VIX<15 (if 100% WR)")
print("="*70)

def make_da_plus_thu():
    def fn(day):
        result = cz_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            # DA: Fri confl>=0.80&ent>=0.70
            if (dd.weekday() == 4 and day.cs_confluence_score >= 0.80 and
                day.cs_entropy_score is not None and day.cs_entropy_score >= 0.70):
                return result
            # SELL Thu VIX<15
            if dd.weekday() == 3 and day.cs_action == 'SELL' and vix_map.get(day.date, 22) < 15:
                return result
        return None
    return fn

trades = simulate_trades(signals, make_da_plus_thu(), 'test', cooldown=0, trail_power=6)
n = len(trades)
w = sum(1 for t in trades if t.pnl > 0)
wr = w/n*100 if n else 0
pnl = sum(t.pnl for t in trades)
bl = min(t.pnl for t in trades) if trades else 0
print(f"DA+Thu: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f}")

if wr >= 100:
    print("\n*** 100% WR — running full validation ***")
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

    # Show new trades
    new_entries = {t.entry_date for t in trades} - cz_dates
    for t in sorted(trades, key=lambda x: x.entry_date):
        if t.entry_date in new_entries:
            dd = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
            dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
            print(f"  NEW: {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} {dow}")

# ════════════════════════════════════════════════════════
# Also explore: what SELL conditions on Thu are safe?
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SELL Thu condition sweep (on DA base)")
print("="*70)

for cond_label, cond_check in [
    ("VIX<12", lambda d: vix_map.get(d.date, 22) < 12),
    ("VIX<13", lambda d: vix_map.get(d.date, 22) < 13),
    ("VIX<14", lambda d: vix_map.get(d.date, 22) < 14),
    ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
    ("VIX<16", lambda d: vix_map.get(d.date, 22) < 16),
    ("VIX<17", lambda d: vix_map.get(d.date, 22) < 17),
    ("VIX<18", lambda d: vix_map.get(d.date, 22) < 18),
    ("SPY>0%", lambda d: spy_dist_map.get(d.date, 0) > 0),
    ("SPY>0.5%", lambda d: spy_dist_map.get(d.date, 0) > 0.5),
    ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
    ("SPY<-0.5%", lambda d: spy_dist_map.get(d.date, 0) < -0.5),
    ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
    ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
    ("confl>=0.70", lambda d: d.cs_confluence_score >= 0.70),
    ("confl>=0.60", lambda d: d.cs_confluence_score >= 0.60),
    ("conf>=0.55", lambda d: d.cs_confidence >= 0.55),
    ("conf>=0.60", lambda d: d.cs_confidence >= 0.60),
    ("pos<0.95", lambda d: d.cs_position_score < 0.95),
    ("pos<0.90", lambda d: d.cs_position_score < 0.90),
    ("ent>=0.90", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.90),
    ("ent>=0.85", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.85),
    ("ent>=0.80", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.80),
]:
    def make_da_sell_thu(cc=cond_check):
        def fn(day):
            result = cz_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                # DA
                if (dd.weekday() == 4 and day.cs_confluence_score >= 0.80 and
                    day.cs_entropy_score is not None and day.cs_entropy_score >= 0.70):
                    return result
                # SELL Thu + condition
                if dd.weekday() == 3 and day.cs_action == 'SELL' and cc(day):
                    return result
            return None
        return fn
    t_trades = simulate_trades(signals, make_da_sell_thu(), 'test', cooldown=0, trail_power=6)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    tpnl = sum(t.pnl for t in t_trades)
    tbl = min(t.pnl for t in t_trades) if t_trades else 0
    flag = "***" if twr >= 100 else ""
    if tn > 428:
        print(f"  SELL Thu {cond_label}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")
    elif twr >= 100 and tn > 426:
        print(f"  SELL Thu {cond_label}: {tn}t {twr:.1f}% ${tpnl:+,.0f} {flag}")

print("\nDone")
