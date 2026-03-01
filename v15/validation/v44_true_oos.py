#!/usr/bin/env python3
"""v44: TRUE out-of-sample validation.

All condition searches use ONLY 2016-2025 data.
Then test winning combos on 2026 as genuine holdout.

Key question: do DG/DH/DI conditions still work when found
without seeing 2026 data?
"""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, simulate_trades, _build_filter_cascade,
    _make_v27_ct, _make_v33_cz, _make_v34_dd, MIN_SIGNAL_CONFIDENCE,
    _floor_stop_tp, _SigProxy, _AnalysisProxy
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

dd_fn_full = _make_v34_dd(*args)
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

# Split signals: training (2016-2025) vs OOS (2026)
signals_train = [s for s in signals if s.date.year <= 2025]
signals_oos = [s for s in signals if s.date.year >= 2026]
print(f"Signals: {len(signals_train)} train (2016-2025), {len(signals_oos)} OOS (2026)")

def full_validate(trades, name, show_oos=True):
    """Validate with proper splits."""
    n = len(trades)
    if n == 0:
        print(f"\n{name}: 0 trades")
        return False
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades)
    print(f"\n{name}: {n}t {wr:.1f}% WR ${pnl:+,.0f} BL=${bl:+,.0f}")

    train = [t for t in trades if t.entry_date.year <= 2021]
    test = [t for t in trades if 2022 <= t.entry_date.year <= 2025]
    oos = [t for t in trades if t.entry_date.year >= 2026]

    if train:
        nt = len(train); wt = sum(1 for t in train if t.pnl > 0)
        print(f"  Train: {nt}t {wt/nt*100:.0f}% WR ${sum(t.pnl for t in train):+,.0f}")
    if test:
        ne = len(test); we = sum(1 for t in test if t.pnl > 0)
        print(f"  Test:  {ne}t {we/ne*100:.0f}% WR ${sum(t.pnl for t in test):+,.0f}")

    wf_pass = 0; wf_total = 0
    for year in range(2017, 2026):
        yearly = [t for t in trades if t.entry_date.year == year]
        if not yearly: continue
        wf_total += 1
        passed = all(t.pnl > 0 for t in yearly)
        if passed: wf_pass += 1
        yn = len(yearly); yw = sum(1 for t in yearly if t.pnl > 0)
        ybl = min(t.pnl for t in yearly)
        print(f"    {year}: {yn:3d}t {yw}/{yn} BL=${ybl:+,.0f} {'PASS' if passed else 'FAIL'}")

    if show_oos and oos:
        no = len(oos); wo = sum(1 for t in oos if t.pnl > 0)
        oos_pass = wo == no
        obl = min(t.pnl for t in oos)
        print(f"  2026 OOS: {no}t {wo}/{no} ${sum(t.pnl for t in oos):+,.0f} BL=${obl:+,.0f} {'PASS' if oos_pass else 'FAIL'}")
        for t in oos:
            print(f"    {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f}")
        return oos_pass and wt == nt and we == ne and wf_pass == wf_total
    return False


# ════════════════════════════════════════════════════════════
# SECTION 1: Re-derive DD from scratch using only 2016-2025 signals
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1: DD on train-only signals (2016-2025)")
print("=" * 70)

# DD was found using full data. But its conditions (Fri entropy, SELL Thu VIX/SPY,
# SELL Tue VIX/SPY) were derived from patterns in 2016-2025 data.
# Let's verify DD produces same train results and then check 2026 truly OOS.

def make_dd():
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

# Run DD on TRAIN ONLY signals to see train-period performance
dd_fn = make_dd()
dd_train = simulate_trades(signals_train, dd_fn, 'DD_train', cooldown=0, trail_power=6)
dd_train_n = len(dd_train)
dd_train_w = sum(1 for t in dd_train if t.pnl > 0)
dd_train_pnl = sum(t.pnl for t in dd_train)
print(f"DD on 2016-2025 only: {dd_train_n}t {dd_train_w/dd_train_n*100:.1f}% WR ${dd_train_pnl:+,.0f}")

# Now run DD on FULL signals (includes 2026) for comparison
dd_full = simulate_trades(signals, dd_fn, 'DD_full', cooldown=0, trail_power=6)
full_validate(dd_full, "DD (full, tp=6)")

# ════════════════════════════════════════════════════════════
# SECTION 2: Re-derive DG/DH/DI expansions on train-only signals
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: Condition search on TRAIN-ONLY (2016-2025)")
print("=" * 70)

# Search for BUY Tue, BUY Thu, SELL Tue conditions using only train signals
dd_train_dates = {t.entry_date for t in dd_train}

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
    ("SPY>2.5%", lambda d: spy_dist_map.get(d.date, 0) > 2.5),
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
    ("SRet<-0.3%", lambda d: spy_return_map.get(d.date, 0) < -0.3),
    ("SRet<-0.5%", lambda d: spy_return_map.get(d.date, 0) < -0.5),
    ("SRet<-1%", lambda d: spy_return_map.get(d.date, 0) < -1.0),
    ("SRet>0.5%", lambda d: spy_return_map.get(d.date, 0) > 0.5),
    ("SRet2d<-1%", lambda d: spy_ret_2d.get(d.date, 0) < -1.0),
]

# DD at different trail powers on TRAIN only
print("\n--- DD trail power sweep (train only) ---")
for tp in [6, 8, 10, 12, 15]:
    trades = simulate_trades(signals_train, dd_fn, 'DD', cooldown=0, trail_power=tp)
    n = len(trades); w = sum(1 for t in trades if t.pnl > 0)
    bl = min(t.pnl for t in trades) if trades else 0
    pnl = sum(t.pnl for t in trades)
    flag = "100%" if w == n else f"FAIL BL=${bl:+,.0f}"
    print(f"  DD@tp={tp:2d} train: {n}t {w/n*100:.1f}% WR ${pnl:+,.0f} {flag}")

# Single-condition search on DD@tp=12 using TRAIN ONLY
print("\n--- Single-condition search on DD@tp=12 (train only) ---")
found_train = []
for direction in ['BUY', 'SELL']:
    for dow_val, dow_label in [(0,"Mon"), (1,"Tue"), (2,"Wed"), (3,"Thu"), (4,"Fri")]:
        for cond_label, cond_check in all_conds:
            def make_expand(dr=direction, dv=dow_val, cc=cond_check):
                def fn(day):
                    result = dd_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd_date.weekday() == dv and day.cs_action == dr and cc(day):
                            return result
                    return None
                return fn

            # Run on TRAIN ONLY signals
            t_trades = simulate_trades(signals_train, make_expand(), 'test', cooldown=0, trail_power=12)
            tn = len(t_trades)
            tw = sum(1 for t in t_trades if t.pnl > 0)
            if tn > dd_train_n and tw == tn:
                tpnl = sum(t.pnl for t in t_trades)
                print(f"  {direction} {dow_label} {cond_label}: {tn}t 100% ${tpnl:+,.0f} *** (train only)")
                found_train.append((direction, dow_label, cond_label, tn, tpnl))

if not found_train:
    print("  No single-condition expansion found on train-only data")

# Double-condition search on DD@tp=12 using TRAIN ONLY (for BUY Tue which was found via double)
print("\n--- Double-condition search BUY Tue on DD@tp=12 (train only) ---")
compact_conds = [
    ("VIX<12", lambda d: vix_map.get(d.date, 22) < 12),
    ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
    ("VIX<17", lambda d: vix_map.get(d.date, 22) < 17),
    ("SPY>0.5%", lambda d: spy_dist_map.get(d.date, 0) > 0.5),
    ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
    ("SPY>2%", lambda d: spy_dist_map.get(d.date, 0) > 2.0),
    ("SPY>2.5%", lambda d: spy_dist_map.get(d.date, 0) > 2.5),
    ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
    ("ent>=0.80", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.80),
    ("ent>=0.90", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.90),
    ("h>=0.25", lambda d: d.cs_channel_health >= 0.25),
    ("h>=0.30", lambda d: d.cs_channel_health >= 0.30),
    ("SRet<-0.3%", lambda d: spy_return_map.get(d.date, 0) < -0.3),
    ("SRet<-0.5%", lambda d: spy_return_map.get(d.date, 0) < -0.5),
    ("SRet<-1%", lambda d: spy_return_map.get(d.date, 0) < -1.0),
    ("SRet>0.5%", lambda d: spy_return_map.get(d.date, 0) > 0.5),
]

found_double_train = []
for direction in ['BUY', 'SELL']:
    for dow_val, dow_label in [(1,"Tue"), (3,"Thu"), (4,"Fri")]:
        for i, (cl1, cc1) in enumerate(compact_conds):
            for cl2, cc2 in compact_conds[i+1:]:
                def make_double(dr=direction, dv=dow_val, c1=cc1, c2=cc2):
                    def fn(day):
                        result = dd_fn(day)
                        if result is not None:
                            return result
                        result = _tf0_base(day)
                        if result is not None:
                            dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                            if dd_date.weekday() == dv and day.cs_action == dr and c1(day) and c2(day):
                                return result
                        return None
                    return fn

                # TRAIN ONLY
                t_trades = simulate_trades(signals_train, make_double(), 'test', cooldown=0, trail_power=12)
                tn = len(t_trades)
                tw = sum(1 for t in t_trades if t.pnl > 0)
                if tn > dd_train_n and tw == tn:
                    tpnl = sum(t.pnl for t in t_trades)
                    print(f"  {direction} {dow_label} {cl1}&{cl2}: {tn}t 100% ${tpnl:+,.0f} *** (train only)")
                    found_double_train.append((direction, dow_label, cl1, cl2, tn, tpnl))

if not found_double_train:
    print("  No double-condition expansion found on train-only data")

# ════════════════════════════════════════════════════════════
# SECTION 3: Build best combo from train-only, then test on 2026
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: Build best combo from train-only findings, test on 2026")
print("=" * 70)

# Reconstruct DG-equivalent from train-only findings
# DG had: BUY Tue SRet<-1% and BUY Thu h>=0.25
# Check if these were found in train-only search above

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

dg_fn = make_dg()
# Train only
dg_train = simulate_trades(signals_train, dg_fn, 'DG_train', cooldown=0, trail_power=12)
dg_train_n = len(dg_train)
dg_train_w = sum(1 for t in dg_train if t.pnl > 0)
print(f"DG train (2016-2025): {dg_train_n}t {dg_train_w/dg_train_n*100:.1f}% WR")

# Now test on FULL including 2026
dg_full = simulate_trades(signals, dg_fn, 'DG_full', cooldown=0, trail_power=12)
full_validate(dg_full, "DG (full, tp=12)")

# DH: DG + BUY Tue VIX<15 & SRet<-0.3%
def make_dh():
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
                spy_return_map.get(day.date, 0) < -0.3):
                return result
        return None
    return fn

dh_fn = make_dh()
dh_train = simulate_trades(signals_train, dh_fn, 'DH_train', cooldown=0, trail_power=12)
dh_train_n = len(dh_train)
dh_train_w = sum(1 for t in dh_train if t.pnl > 0)
print(f"\nDH train (2016-2025): {dh_train_n}t {dh_train_w/dh_train_n*100:.1f}% WR")

dh_full = simulate_trades(signals, dh_fn, 'DH_full', cooldown=0, trail_power=12)
full_validate(dh_full, "DH (full, tp=12)")

# DI: DH + SELL Tue SPY>2.5%
def make_di():
    dh_fn_inner = make_dh()
    def fn(day):
        result = dh_fn_inner(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            d = day.date.date() if hasattr(day.date, 'date') else day.date
            if (d.weekday() == 1 and day.cs_action == 'SELL' and
                spy_dist_map.get(day.date, 0) > 2.5):
                return result
        return None
    return fn

di_fn = make_di()
di_train = simulate_trades(signals_train, di_fn, 'DI_train', cooldown=0, trail_power=12)
di_train_n = len(di_train)
di_train_w = sum(1 for t in di_train if t.pnl > 0)
print(f"\nDI train (2016-2025): {di_train_n}t {di_train_w/di_train_n*100:.1f}% WR")

di_full = simulate_trades(signals, di_fn, 'DI_full', cooldown=0, trail_power=12)
full_validate(di_full, "DI (full, tp=12)")

# ════════════════════════════════════════════════════════════
# SECTION 4: Train-only stacking search
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Stacking search on train-only, then true OOS test")
print("=" * 70)

# Start from DD@tp=12 on train only. Stack all single conditions that work.
# Then test the stacked combo on full (including 2026).
print("Starting from DD@tp=12 train-only baseline...")

# Accumulate working conditions on train only
current_fn = dd_fn
current_name = "DD"
dd_train_base = simulate_trades(signals_train, current_fn, current_name, cooldown=0, trail_power=12)
current_train_n = len(dd_train_base)
print(f"Baseline: {current_train_n}t on train-only")

# Iteratively add best single-condition that increases trades at 100% WR
iteration = 0
while True:
    iteration += 1
    best = None
    best_n = current_train_n
    best_fn = None

    for direction in ['BUY', 'SELL']:
        for dow_val, dow_label in [(0,"Mon"), (1,"Tue"), (2,"Wed"), (3,"Thu"), (4,"Fri")]:
            for cond_label, cond_check in all_conds:
                def make_iter_expand(base=current_fn, dr=direction, dv=dow_val, cc=cond_check):
                    def fn(day):
                        result = base(day)
                        if result is not None:
                            return result
                        result = _tf0_base(day)
                        if result is not None:
                            dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                            if dd_date.weekday() == dv and day.cs_action == dr and cc(day):
                                return result
                        return None
                    return fn

                t_trades = simulate_trades(signals_train, make_iter_expand(), 'test', cooldown=0, trail_power=12)
                tn = len(t_trades)
                tw = sum(1 for t in t_trades if t.pnl > 0)
                if tn > best_n and tw == tn:
                    best_n = tn
                    best = (direction, dow_label, cond_label)
                    # Need to capture the actual function
                    best_fn = make_iter_expand()

    if best is None:
        print(f"\nIteration {iteration}: No more single-condition expansions on train-only")
        break

    direction, dow_label, cond_label = best
    current_fn = best_fn
    current_train_n = best_n
    current_name = f"{current_name}+{direction[0]}{dow_label}{cond_label}"
    print(f"  Iter {iteration}: +{direction} {dow_label} {cond_label} -> {current_train_n}t (train-only)")

# Now test the accumulated combo on FULL signals (2026 is true OOS)
print(f"\n--- Testing accumulated combo on FULL (including 2026 OOS) ---")
final_trades = simulate_trades(signals, current_fn, 'Final', cooldown=0, trail_power=12)
full_validate(final_trades, f"Accumulated combo (tp=12)")

# Also show what conditions were found
print(f"\nFinal combo built from train-only: {current_name}")

print("\nDone.")
