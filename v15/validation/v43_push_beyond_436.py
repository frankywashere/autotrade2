#!/usr/bin/env python3
"""Push beyond DG (436t @ tp=12).

v43 Section 8 found no single-condition expansion beyond 436t at tp=12.
Now try:
  1. Higher trail powers (tp=15, 20) for DG - check if still 100%
  2. Higher trail powers unlocking new single-condition expansions
  3. Double-condition combos on DG at tp=12
  4. Confidence boosting for marginal trades
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
    """DG: DD + BUY Tue SRet<-1% + BUY Thu h>=0.25."""
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

# ════════════════════════════════════════════════════════════
# PART 1: DG at higher trail powers
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: DG at various trail powers")
print("=" * 70)

dg_fn = make_dg()
for tp in range(6, 31):
    trades = simulate_trades(signals, dg_fn, 'DG', cooldown=0, trail_power=tp)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    bl = min(t.pnl for t in trades) if trades else 0
    pnl = sum(t.pnl for t in trades)
    flag = "100%" if w == n else f"FAIL BL=${bl:+,.0f}"
    if n != 436 or w != n:
        print(f"  DG tp={tp:2d}: {n}t {w/n*100:.1f}% WR ${pnl:+,.0f} {flag}")
    else:
        print(f"  DG tp={tp:2d}: {n}t 100% WR ${pnl:+,.0f}")

# Find the highest TP where DG is still 100%
best_dg_tp = 12
for tp in range(12, 31):
    trades = simulate_trades(signals, dg_fn, 'DG', cooldown=0, trail_power=tp)
    if all(t.pnl > 0 for t in trades):
        best_dg_tp = tp
    else:
        break

print(f"\nBest DG trail_power: {best_dg_tp}")

# ════════════════════════════════════════════════════════════
# PART 2: Single-condition expansion at higher trail powers
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: Single-condition expansion beyond DG at higher trail powers")
print("=" * 70)

all_conds = [
    ("VIX<10", lambda d: vix_map.get(d.date, 22) < 10),
    ("VIX<12", lambda d: vix_map.get(d.date, 22) < 12),
    ("VIX<13", lambda d: vix_map.get(d.date, 22) < 13),
    ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
    ("VIX>25", lambda d: vix_map.get(d.date, 22) > 25),
    ("VIX>30", lambda d: vix_map.get(d.date, 22) > 30),
    ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
    ("SPY<-0.5%", lambda d: spy_dist_map.get(d.date, 0) < -0.5),
    ("SPY>0%", lambda d: spy_dist_map.get(d.date, 0) > 0),
    ("SPY>0.5%", lambda d: spy_dist_map.get(d.date, 0) > 0.5),
    ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
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

for test_tp in [15, 20, 25]:
    # First check DG baseline at this TP
    dg_test = simulate_trades(signals, dg_fn, 'DG', cooldown=0, trail_power=test_tp)
    dg_n = len(dg_test)
    dg_w = sum(1 for t in dg_test if t.pnl > 0)
    if dg_w != dg_n:
        print(f"\n  DG@tp={test_tp} NOT 100% ({dg_w}/{dg_n}), skipping")
        continue

    print(f"\n  --- Expansions at tp={test_tp} (DG baseline: {dg_n}t 100%) ---")
    found = False
    for direction in ['BUY', 'SELL']:
        for dow_val, dow_label in [(0,"Mon"), (1,"Tue"), (2,"Wed"), (3,"Thu"), (4,"Fri")]:
            for cond_label, cond_check in all_conds:
                def make_expand(dr=direction, dv=dow_val, cc=cond_check):
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

                t_trades = simulate_trades(signals, make_expand(), 'test', cooldown=0, trail_power=test_tp)
                tn = len(t_trades)
                tw = sum(1 for t in t_trades if t.pnl > 0)
                if tn > dg_n and tw == tn:
                    tpnl = sum(t.pnl for t in t_trades)
                    print(f"    {direction} {dow_label} {cond_label}: {tn}t 100% ${tpnl:+,.0f} ***")
                    found = True

    if not found:
        print(f"    No expansion beyond {dg_n}t at tp={test_tp}")

# ════════════════════════════════════════════════════════════
# PART 3: Double-condition combos on DG at tp=12
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: Double-condition combos beyond DG at tp=12")
print("=" * 70)

# Use a smaller condition set for O(n^2) search
compact_conds = [
    ("VIX<12", lambda d: vix_map.get(d.date, 22) < 12),
    ("VIX<13", lambda d: vix_map.get(d.date, 22) < 13),
    ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
    ("VIX>25", lambda d: vix_map.get(d.date, 22) > 25),
    ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
    ("SPY<-0.5%", lambda d: spy_dist_map.get(d.date, 0) < -0.5),
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
    ("pos<0.85", lambda d: d.cs_position_score < 0.85),
    ("pos<0.90", lambda d: d.cs_position_score < 0.90),
    ("conf>=0.55", lambda d: d.cs_confidence >= 0.55),
    ("SRet<-0.5%", lambda d: spy_return_map.get(d.date, 0) < -0.5),
    ("SRet<-1%", lambda d: spy_return_map.get(d.date, 0) < -1.0),
    ("SRet>0.5%", lambda d: spy_return_map.get(d.date, 0) > 0.5),
]

dg_tp12 = simulate_trades(signals, dg_fn, 'DG', cooldown=0, trail_power=12)
dg_count = len(dg_tp12)

found_double = False
for direction in ['BUY', 'SELL']:
    for dow_val, dow_label in [(0,"Mon"), (1,"Tue"), (2,"Wed"), (3,"Thu"), (4,"Fri")]:
        for i, (cl1, cc1) in enumerate(compact_conds):
            for cl2, cc2 in compact_conds[i+1:]:
                def make_double(dr=direction, dv=dow_val, c1=cc1, c2=cc2):
                    dg_fn_inner = make_dg()
                    def fn(day):
                        result = dg_fn_inner(day)
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
                if tn > dg_count and tw == tn:
                    tpnl = sum(t.pnl for t in t_trades)
                    print(f"  {direction} {dow_label} {cl1}&{cl2}: {tn}t 100% ${tpnl:+,.0f} ***")
                    found_double = True

if not found_double:
    print(f"  No double-condition expansion beyond {dg_count}t at tp=12")

# ════════════════════════════════════════════════════════════
# PART 4: Profile all DG-rejected signals
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: Profile DG-rejected signals that pass VIX cascade")
print("=" * 70)

dg_fn_ref = make_dg()
dg_dates = set()
for d in signals:
    result = dg_fn_ref(d)
    if result is not None:
        dg_dates.add(d.date)

rejected = []
for d in signals:
    if d.date in dg_dates:
        continue
    result = _tf0_base(d)
    if result is not None:
        rejected.append(d)

print(f"DG-rejected signals that pass VIX cascade: {len(rejected)}")

# Simulate each rejected signal individually to see if it's a winner or loser
for d in rejected:
    dd_date = d.date.date() if hasattr(d.date, 'date') else d.date
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd_date.weekday()] if dd_date.weekday() < 5 else '?'

    def make_single_add(sig_date=d.date):
        dg_fn_inner = make_dg()
        def fn(day):
            result = dg_fn_inner(day)
            if result is not None:
                return result
            if day.date == sig_date:
                return _tf0_base(day)
            return None
        return fn

    trades = simulate_trades(signals, make_single_add(), 'test', cooldown=0, trail_power=12)
    # Find the trade for this signal
    dg_base_dates = {t.entry_date for t in dg_tp12}
    new_trades = [t for t in trades if t.entry_date not in dg_base_dates]

    if new_trades:
        nt = new_trades[0]
        vix_val = vix_map.get(d.date, 0)
        spy_d = spy_dist_map.get(d.date, 0)
        sret = spy_return_map.get(d.date, 0)
        total_n = len(trades)
        total_w = sum(1 for t in trades if t.pnl > 0)
        displaced = [t for t in dg_tp12 if t.entry_date not in {t2.entry_date for t2 in trades}]
        disp_str = f" disp={len(displaced)}" if displaced else ""
        print(f"  {str(d.date)[:10]} {d.cs_action:5s} ${nt.pnl:+8,.0f} {dow} "
              f"h={d.cs_channel_health:.3f} c={d.cs_confidence:.3f} confl={d.cs_confluence_score:.2f} "
              f"ent={d.cs_entropy_score if d.cs_entropy_score else 0:.2f} "
              f"VIX={vix_val:.1f} SPY={spy_d:+.1f}% SRet={sret:+.1f}% "
              f"n={total_n} w={total_w}{disp_str}")
    else:
        # Signal generated but no new trade (displaced by cooldown)
        pass

print("\nDone.")
