#!/usr/bin/env python3
"""v43 Validation: Push DD (434t) to 435+ via BUY Tue + BUY Thu expansion.

Best paths from v43 experiment:
  1. DD@tp=12 + BUY Tue SRet<-1% = 435t 100% $710,599
  2. DD@tp=12 + BUY Thu h>=0.25 = 435t 100% $709,344
  3. DD@tp=6 + BUY Tue[SRet<-0.5% & h>=0.25 & SPY>0.5%] = 435t 100% $707,878
  4. DD@tp=6 + BUY Thu[ent>=0.70 & h>=0.25] = 435t 100% $707,631

Key question: can we STACK both to get 436t?
Also: validate with holdout/walk-forward/2026 OOS.
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

def make_dd():
    """DD baseline (434t)."""
    def fn(day):
        result = cz_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            if (dd.weekday() == 4 and day.cs_confluence_score >= 0.80 and
                day.cs_entropy_score is not None and day.cs_entropy_score >= 0.70):
                return result
            if dd.weekday() == 3 and day.cs_action == 'SELL':
                vix_val = vix_map.get(day.date, 22)
                spy_d = spy_dist_map.get(day.date, 0)
                if vix_val < 15 or spy_d < -1.0:
                    return result
            if dd.weekday() == 1 and day.cs_action == 'SELL':
                vix_val = vix_map.get(day.date, 22)
                spy_d = spy_dist_map.get(day.date, 0)
                if vix_val < 13 or spy_d < -0.5:
                    return result
        return None
    return fn

def full_validate(trades, name):
    """Run full 3-stage validation and print results."""
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
# SECTION 1: Validate DD@tp=12 baseline
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1: DD@tp=12 baseline validation")
print("=" * 70)

dd_fn = make_dd()
dd_tp12 = simulate_trades(signals, dd_fn, 'DD', cooldown=0, trail_power=12)
full_validate(dd_tp12, "DD@tp=12")
dd_dates = {t.entry_date for t in dd_tp12}

# ════════════════════════════════════════════════════════════
# SECTION 2: DE candidate = DD@tp=12 + BUY Tue SRet<-1%
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: DE = DD@tp=12 + BUY Tue SRet<-1%")
print("=" * 70)

def make_de():
    dd_fn_inner = make_dd()
    def fn(day):
        result = dd_fn_inner(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
            if (dd_date.weekday() == 1 and day.cs_action == 'BUY' and
                spy_return_map.get(day.date, 0) < -1.0):
                return result
        return None
    return fn

de_trades = simulate_trades(signals, make_de(), 'DE', cooldown=0, trail_power=12)
de_pass = full_validate(de_trades, "DE@tp=12")

de_dates = {t.entry_date for t in de_trades}
new_trades = [t for t in de_trades if t.entry_date not in dd_dates]
lost_trades = [t for t in dd_tp12 if t.entry_date not in de_dates]
print(f"\n  New vs DD: +{len(new_trades)} new, -{len(lost_trades)} lost")
for t in sorted(new_trades, key=lambda x: x.entry_date):
    dd_val = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd_val.weekday()]
    print(f"    NEW: {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} {dow}")
for t in sorted(lost_trades, key=lambda x: x.entry_date):
    dd_val = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd_val.weekday()]
    print(f"    LOST: {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} {dow}")

# ════════════════════════════════════════════════════════════
# SECTION 3: DF candidate = DD@tp=12 + BUY Thu h>=0.25
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: DF = DD@tp=12 + BUY Thu h>=0.25")
print("=" * 70)

def make_df():
    dd_fn_inner = make_dd()
    def fn(day):
        result = dd_fn_inner(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
            if (dd_date.weekday() == 3 and day.cs_action == 'BUY' and
                day.cs_channel_health >= 0.25):
                return result
        return None
    return fn

df_trades = simulate_trades(signals, make_df(), 'DF', cooldown=0, trail_power=12)
df_pass = full_validate(df_trades, "DF@tp=12")

df_dates = {t.entry_date for t in df_trades}
new_trades = [t for t in df_trades if t.entry_date not in dd_dates]
lost_trades = [t for t in dd_tp12 if t.entry_date not in df_dates]
print(f"\n  New vs DD: +{len(new_trades)} new, -{len(lost_trades)} lost")
for t in sorted(new_trades, key=lambda x: x.entry_date):
    dd_val = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd_val.weekday()]
    day_sig = next((d for d in signals if d.date == t.entry_date), None)
    h_val = day_sig.cs_channel_health if day_sig else 0
    print(f"    NEW: {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} {dow} h={h_val:.3f}")

# ════════════════════════════════════════════════════════════
# SECTION 4: DG = DD@tp=12 + BUY Tue SRet<-1% + BUY Thu h>=0.25 (STACK)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: DG = DD@tp=12 + BUY Tue SRet<-1% + BUY Thu h>=0.25 (STACKED)")
print("=" * 70)

def make_dg():
    dd_fn_inner = make_dd()
    def fn(day):
        result = dd_fn_inner(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
            # BUY Tuesday: SPY return yesterday < -1%
            if (dd_date.weekday() == 1 and day.cs_action == 'BUY' and
                spy_return_map.get(day.date, 0) < -1.0):
                return result
            # BUY Thursday: h >= 0.25
            if (dd_date.weekday() == 3 and day.cs_action == 'BUY' and
                day.cs_channel_health >= 0.25):
                return result
        return None
    return fn

dg_trades = simulate_trades(signals, make_dg(), 'DG', cooldown=0, trail_power=12)
dg_pass = full_validate(dg_trades, "DG@tp=12")

dg_dates = {t.entry_date for t in dg_trades}
new_trades = [t for t in dg_trades if t.entry_date not in dd_dates]
lost_trades = [t for t in dd_tp12 if t.entry_date not in dg_dates]
print(f"\n  New vs DD: +{len(new_trades)} new, -{len(lost_trades)} lost")
for t in sorted(new_trades, key=lambda x: x.entry_date):
    dd_val = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd_val.weekday()]
    print(f"    NEW: {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} {dow}")
for t in sorted(lost_trades, key=lambda x: x.entry_date):
    dd_val = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd_val.weekday()]
    print(f"    LOST: {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} {dow}")

# ════════════════════════════════════════════════════════════
# SECTION 5: Try triple conditions at tp=6 as alternative
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: Triple-condition at tp=6 — BUY Tue SRet<-0.5% & h>=0.25")
print("=" * 70)

# The v43 sweep showed multiple triple combos work at tp=6.
# Try the simplest: SRet<-0.5% & h>=0.25 (2 conditions, not 3)
for sret_th in [-0.3, -0.5, -0.7, -1.0, -1.5]:
    for h_th in [0.20, 0.25, 0.30, 0.35]:
        def make_buy_tue_2cond(sr=sret_th, ht=h_th):
            dd_fn_inner = make_dd()
            def fn(day):
                result = dd_fn_inner(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                    if (dd_date.weekday() == 1 and day.cs_action == 'BUY' and
                        spy_return_map.get(day.date, 0) < sr and
                        day.cs_channel_health >= ht):
                        return result
                return None
            return fn

        trades = simulate_trades(signals, make_buy_tue_2cond(), 'test', cooldown=0, trail_power=6)
        tn = len(trades)
        tw = sum(1 for t in trades if t.pnl > 0)
        tbl = min(t.pnl for t in trades) if trades else 0
        if tn > 434:
            flag = "***" if tw == tn else ""
            tpnl = sum(t.pnl for t in trades)
            print(f"  BUY Tue SRet<{sret_th}% & h>={h_th}: {tn}t {tw/tn*100:.1f}% BL=${tbl:+,.0f} ${tpnl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════════
# SECTION 6: Try simpler Tue conditions at intermediate trail powers
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: BUY Tue SRet sweep across trail powers")
print("=" * 70)

for sret_th in [-0.5, -0.7, -1.0, -1.5, -2.0]:
    for tp in [6, 8, 10, 12]:
        def make_buy_tue_sret(sr=sret_th):
            dd_fn_inner = make_dd()
            def fn(day):
                result = dd_fn_inner(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                    if (dd_date.weekday() == 1 and day.cs_action == 'BUY' and
                        spy_return_map.get(day.date, 0) < sr):
                        return result
                return None
            return fn

        trades = simulate_trades(signals, make_buy_tue_sret(), 'test', cooldown=0, trail_power=tp)
        tn = len(trades)
        tw = sum(1 for t in trades if t.pnl > 0)
        if tn > 434 and tw == tn:
            tpnl = sum(t.pnl for t in trades)
            print(f"  BUY Tue SRet<{sret_th}% tp={tp}: {tn}t 100% ${tpnl:+,.0f} ***")

# ════════════════════════════════════════════════════════════
# SECTION 7: BUY Thu h sweep across trail powers
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7: BUY Thu h sweep across trail powers")
print("=" * 70)

for h_th in [0.20, 0.22, 0.25, 0.28, 0.30, 0.35]:
    for tp in [6, 8, 10, 12]:
        def make_buy_thu_h(ht=h_th):
            dd_fn_inner = make_dd()
            def fn(day):
                result = dd_fn_inner(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                    if (dd_date.weekday() == 3 and day.cs_action == 'BUY' and
                        day.cs_channel_health >= ht):
                        return result
                return None
            return fn

        trades = simulate_trades(signals, make_buy_thu_h(), 'test', cooldown=0, trail_power=tp)
        tn = len(trades)
        tw = sum(1 for t in trades if t.pnl > 0)
        if tn > 434 and tw == tn:
            tpnl = sum(t.pnl for t in trades)
            print(f"  BUY Thu h>={h_th} tp={tp}: {tn}t 100% ${tpnl:+,.0f} ***")

# ════════════════════════════════════════════════════════════
# SECTION 8: Push beyond DG (stacked) — further expansions at tp=12
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8: Push beyond DG (if DG is 100%)")
print("=" * 70)

if dg_pass:
    dg_count = len(dg_trades)
    dg_dates_set = {t.entry_date for t in dg_trades}

    push_conditions = [
        ("VIX<12", lambda d: vix_map.get(d.date, 22) < 12),
        ("VIX<13", lambda d: vix_map.get(d.date, 22) < 13),
        ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
        ("VIX>25", lambda d: vix_map.get(d.date, 22) > 25),
        ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
        ("SPY<-0.5%", lambda d: spy_dist_map.get(d.date, 0) < -0.5),
        ("SPY>0%", lambda d: spy_dist_map.get(d.date, 0) > 0),
        ("SPY>0.5%", lambda d: spy_dist_map.get(d.date, 0) > 0.5),
        ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
        ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
        ("confl>=0.90", lambda d: d.cs_confluence_score >= 0.90),
        ("ent>=0.70", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.70),
        ("ent>=0.80", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.80),
        ("ent>=0.90", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.90),
        ("h>=0.20", lambda d: d.cs_channel_health >= 0.20),
        ("h>=0.25", lambda d: d.cs_channel_health >= 0.25),
        ("h>=0.30", lambda d: d.cs_channel_health >= 0.30),
        ("h>=0.35", lambda d: d.cs_channel_health >= 0.35),
        ("pos<0.85", lambda d: d.cs_position_score < 0.85),
        ("pos<0.90", lambda d: d.cs_position_score < 0.90),
        ("conf>=0.55", lambda d: d.cs_confidence >= 0.55),
        ("conf>=0.60", lambda d: d.cs_confidence >= 0.60),
        ("SRet<-0.5%", lambda d: spy_return_map.get(d.date, 0) < -0.5),
        ("SRet<-1%", lambda d: spy_return_map.get(d.date, 0) < -1.0),
        ("SRet>0.5%", lambda d: spy_return_map.get(d.date, 0) > 0.5),
    ]

    found_push = False
    for direction in ['BUY', 'SELL']:
        for dow_val, dow_label in [(0,"Mon"), (1,"Tue"), (2,"Wed"), (3,"Thu"), (4,"Fri")]:
            for cond_label, cond_check in push_conditions:
                def make_beyond_dg(dr=direction, dv=dow_val, cc=cond_check):
                    dg_fn_inner = make_dg()
                    def fn(day):
                        result = dg_fn_inner(day)
                        if result is not None:
                            return result
                        result = _tf0_base(day)
                        if result is not None:
                            dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                            if dd_date.weekday() == dv and day.cs_action == dr and cc(day):
                                return result
                        return None
                    return fn

                t_trades = simulate_trades(signals, make_beyond_dg(), 'test', cooldown=0, trail_power=12)
                tn = len(t_trades)
                tw = sum(1 for t in t_trades if t.pnl > 0)
                if tn > dg_count and tw == tn:
                    tpnl = sum(t.pnl for t in t_trades)
                    print(f"  {direction} {dow_label} {cond_label}: {tn}t 100% ${tpnl:+,.0f} ***")
                    found_push = True

    if not found_push:
        print(f"  No single-condition expansion beyond {dg_count}t at tp=12 100% WR")
else:
    print("  DG did not pass, skipping push search")

print("\nDone.")
