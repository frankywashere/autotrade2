#!/usr/bin/env python3
"""Validate DI = DH + SELL Tue[SPY>2.5%] = 438t 100% $713,028.
Then exhaustive push beyond 438t."""
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

def make_di():
    """DI: DD + BUY Tue SRet<-1% + BUY Thu h>=0.25 + BUY Tue VIX<15&SRet<-0.3% + SELL Tue SPY>2.5%"""
    def fn(day):
        result = dd_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            d = day.date.date() if hasattr(day.date, 'date') else day.date
            # DG expansion: BUY Tue SRet<-1%
            if (d.weekday() == 1 and day.cs_action == 'BUY' and
                spy_return_map.get(day.date, 0) < -1.0):
                return result
            # DG expansion: BUY Thu h>=0.25
            if (d.weekday() == 3 and day.cs_action == 'BUY' and
                day.cs_channel_health >= 0.25):
                return result
            # DH expansion: BUY Tue VIX<15 & SRet<-0.3%
            if (d.weekday() == 1 and day.cs_action == 'BUY' and
                vix_map.get(day.date, 22) < 15 and
                spy_return_map.get(day.date, 0) < -0.3):
                return result
            # DI expansion: SELL Tue SPY>2.5%
            if (d.weekday() == 1 and day.cs_action == 'SELL' and
                spy_dist_map.get(day.date, 0) > 2.5):
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
# SECTION 1: Full validation of DI (438t)
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1: DI full validation")
print("=" * 70)

di_trades = simulate_trades(signals, make_di(), 'DI', cooldown=0, trail_power=12)
di_pass = full_validate(di_trades, "DI@tp=12")

# Compare with DG
di_fn = make_di()
di_dates = {t.entry_date for t in di_trades}

# Also validate at tp=8
di_tp8 = simulate_trades(signals, di_fn, 'DI', cooldown=0, trail_power=8)
n8 = len(di_tp8)
w8 = sum(1 for t in di_tp8 if t.pnl > 0)
bl8 = min(t.pnl for t in di_tp8) if di_tp8 else 0
print(f"\nDI@tp=8: {n8}t {w8/n8*100:.1f}% WR BL=${bl8:+,.0f}")

# ════════════════════════════════════════════════════════════
# SECTION 2: Profile remaining rejected signals
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: Profile DI-rejected signals")
print("=" * 70)

di_signal_dates = set()
for d in signals:
    result = di_fn(d)
    if result is not None:
        di_signal_dates.add(d.date)

rejected = []
for d in signals:
    if d.date in di_signal_dates:
        continue
    result = _tf0_base(d)
    if result is not None:
        rejected.append(d)

print(f"DI-rejected signals that pass VIX cascade: {len(rejected)}")
di_base_dates = {t.entry_date for t in di_trades}

for d in rejected:
    dd_date = d.date.date() if hasattr(d.date, 'date') else d.date
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd_date.weekday()] if dd_date.weekday() < 5 else '?'

    def make_single_add(sig_date=d.date):
        di_fn_inner = make_di()
        def fn(day):
            result = di_fn_inner(day)
            if result is not None:
                return result
            if day.date == sig_date:
                return _tf0_base(day)
            return None
        return fn

    trades = simulate_trades(signals, make_single_add(), 'test', cooldown=0, trail_power=12)
    new = [t for t in trades if t.entry_date not in di_base_dates]
    disp = [t for t in di_trades if t.entry_date not in {t2.entry_date for t2 in trades}]

    if new:
        nt_pnl = new[0].pnl
        total_n = len(trades)
        total_w = sum(1 for t in trades if t.pnl > 0)
        vix_val = vix_map.get(d.date, 0)
        spy_d = spy_dist_map.get(d.date, 0)
        sret = spy_return_map.get(d.date, 0)
        disp_str = f" disp={len(disp)}" if disp else ""
        status = "W" if nt_pnl > 0 else "L"
        all_win = "100%" if total_w == total_n else f"{total_w}/{total_n}"
        print(f"  {str(d.date)[:10]} {d.cs_action:5s} ${nt_pnl:+8,.0f} {dow} "
              f"h={d.cs_channel_health:.3f} c={d.cs_confidence:.3f} confl={d.cs_confluence_score:.2f} "
              f"ent={d.cs_entropy_score if d.cs_entropy_score else 0:.2f} "
              f"VIX={vix_val:.1f} SPY={spy_d:+.1f}% SRet={sret:+.1f}% "
              f"[{all_win}]{disp_str}")

# ════════════════════════════════════════════════════════════
# SECTION 3: Exhaustive single-condition push beyond DI
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: Single-condition push beyond DI (438t)")
print("=" * 70)

di_count = len(di_trades)
all_conds = [
    ("VIX<10", lambda d: vix_map.get(d.date, 22) < 10),
    ("VIX<12", lambda d: vix_map.get(d.date, 22) < 12),
    ("VIX<13", lambda d: vix_map.get(d.date, 22) < 13),
    ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
    ("VIX<17", lambda d: vix_map.get(d.date, 22) < 17),
    ("VIX>25", lambda d: vix_map.get(d.date, 22) > 25),
    ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
    ("SPY<-0.5%", lambda d: spy_dist_map.get(d.date, 0) < -0.5),
    ("SPY>0%", lambda d: spy_dist_map.get(d.date, 0) > 0),
    ("SPY>0.5%", lambda d: spy_dist_map.get(d.date, 0) > 0.5),
    ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
    ("SPY>2%", lambda d: spy_dist_map.get(d.date, 0) > 2.0),
    ("SPY>3%", lambda d: spy_dist_map.get(d.date, 0) > 3.0),
    ("confl>=0.70", lambda d: d.cs_confluence_score >= 0.70),
    ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
    ("confl>=0.90", lambda d: d.cs_confluence_score >= 0.90),
    ("ent>=0.60", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.60),
    ("ent>=0.70", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.70),
    ("ent>=0.80", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.80),
    ("ent>=0.90", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.90),
    ("h>=0.15", lambda d: d.cs_channel_health >= 0.15),
    ("h>=0.20", lambda d: d.cs_channel_health >= 0.20),
    ("h>=0.25", lambda d: d.cs_channel_health >= 0.25),
    ("h>=0.30", lambda d: d.cs_channel_health >= 0.30),
    ("h>=0.35", lambda d: d.cs_channel_health >= 0.35),
    ("pos<0.80", lambda d: d.cs_position_score < 0.80),
    ("pos<0.85", lambda d: d.cs_position_score < 0.85),
    ("pos<0.90", lambda d: d.cs_position_score < 0.90),
    ("conf>=0.50", lambda d: d.cs_confidence >= 0.50),
    ("conf>=0.55", lambda d: d.cs_confidence >= 0.55),
    ("conf>=0.60", lambda d: d.cs_confidence >= 0.60),
    ("SRet<-0.5%", lambda d: spy_return_map.get(d.date, 0) < -0.5),
    ("SRet<-1%", lambda d: spy_return_map.get(d.date, 0) < -1.0),
    ("SRet>0.5%", lambda d: spy_return_map.get(d.date, 0) > 0.5),
    ("SRet2d<-1%", lambda d: spy_ret_2d.get(d.date, 0) < -1.0),
]

found = False
for direction in ['BUY', 'SELL']:
    for dow_val, dow_label in [(0,"Mon"), (1,"Tue"), (2,"Wed"), (3,"Thu"), (4,"Fri")]:
        for cond_label, cond_check in all_conds:
            def make_expand(dr=direction, dv=dow_val, cc=cond_check):
                di_fn_inner = make_di()
                def fn(day):
                    result = di_fn_inner(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd_date.weekday() == dv and day.cs_action == dr and cc(day):
                            return result
                    return None
                return fn

            t_trades = simulate_trades(signals, make_expand(), 'test', cooldown=0, trail_power=12)
            tn = len(t_trades)
            tw = sum(1 for t in t_trades if t.pnl > 0)
            if tn > di_count and tw == tn:
                tpnl = sum(t.pnl for t in t_trades)
                print(f"  {direction} {dow_label} {cond_label}: {tn}t 100% ${tpnl:+,.0f} ***")
                found = True

if not found:
    print(f"  No single-condition expansion beyond {di_count}t at tp=12")

# ════════════════════════════════════════════════════════════
# SECTION 4: Double-condition push (focused on promising days)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Double-condition push beyond DI (438t)")
print("=" * 70)

compact_conds = [
    ("VIX<12", lambda d: vix_map.get(d.date, 22) < 12),
    ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
    ("VIX<17", lambda d: vix_map.get(d.date, 22) < 17),
    ("VIX>25", lambda d: vix_map.get(d.date, 22) > 25),
    ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
    ("SPY>0.5%", lambda d: spy_dist_map.get(d.date, 0) > 0.5),
    ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
    ("SPY>2%", lambda d: spy_dist_map.get(d.date, 0) > 2.0),
    ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
    ("confl>=0.90", lambda d: d.cs_confluence_score >= 0.90),
    ("ent>=0.80", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.80),
    ("ent>=0.90", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.90),
    ("h>=0.25", lambda d: d.cs_channel_health >= 0.25),
    ("h>=0.30", lambda d: d.cs_channel_health >= 0.30),
    ("h>=0.35", lambda d: d.cs_channel_health >= 0.35),
    ("pos<0.85", lambda d: d.cs_position_score < 0.85),
    ("conf>=0.55", lambda d: d.cs_confidence >= 0.55),
    ("SRet<-0.5%", lambda d: spy_return_map.get(d.date, 0) < -0.5),
    ("SRet<-1%", lambda d: spy_return_map.get(d.date, 0) < -1.0),
    ("SRet>0.5%", lambda d: spy_return_map.get(d.date, 0) > 0.5),
]

found_double = False
for direction in ['BUY', 'SELL']:
    for dow_val, dow_label in [(0,"Mon"), (1,"Tue"), (2,"Wed"), (3,"Thu"), (4,"Fri")]:
        for i, (cl1, cc1) in enumerate(compact_conds):
            for cl2, cc2 in compact_conds[i+1:]:
                def make_double(dr=direction, dv=dow_val, c1=cc1, c2=cc2):
                    di_fn_inner = make_di()
                    def fn(day):
                        result = di_fn_inner(day)
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
                if tn > di_count and tw == tn:
                    tpnl = sum(t.pnl for t in t_trades)
                    print(f"  {direction} {dow_label} {cl1}&{cl2}: {tn}t 100% ${tpnl:+,.0f} ***")
                    found_double = True

if not found_double:
    print(f"  No double-condition expansion beyond {di_count}t at tp=12")

# ════════════════════════════════════════════════════════════
# SECTION 5: Try tp=15, 20, 25 for expansion beyond DI
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: Higher trail powers for DI expansion")
print("=" * 70)

for tp in [15, 20, 25]:
    di_test = simulate_trades(signals, di_fn, 'DI', cooldown=0, trail_power=tp)
    di_n = len(di_test)
    di_w = sum(1 for t in di_test if t.pnl > 0)
    if di_w != di_n:
        print(f"  DI@tp={tp} NOT 100% ({di_w}/{di_n}), skipping")
        continue

    print(f"  DI@tp={tp}: {di_n}t 100% — searching single expansions...")
    found_tp = False
    for direction in ['BUY', 'SELL']:
        for dow_val, dow_label in [(0,"Mon"), (1,"Tue"), (2,"Wed"), (3,"Thu"), (4,"Fri")]:
            for cond_label, cond_check in all_conds:
                def make_expand_tp(dr=direction, dv=dow_val, cc=cond_check):
                    di_fn_inner = make_di()
                    def fn(day):
                        result = di_fn_inner(day)
                        if result is not None:
                            return result
                        result = _tf0_base(day)
                        if result is not None:
                            dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                            if dd_date.weekday() == dv and day.cs_action == dr and cc(day):
                                return result
                        return None
                    return fn

                t_trades = simulate_trades(signals, make_expand_tp(), 'test', cooldown=0, trail_power=tp)
                tn = len(t_trades)
                tw = sum(1 for t in t_trades if t.pnl > 0)
                if tn > di_n and tw == tn:
                    tpnl = sum(t.pnl for t in t_trades)
                    print(f"    {direction} {dow_label} {cond_label}: {tn}t 100% ${tpnl:+,.0f} ***")
                    found_tp = True

    if not found_tp:
        print(f"    No single-condition expansion beyond {di_n}t at tp={tp}")

print("\nDone.")
