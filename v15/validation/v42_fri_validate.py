#!/usr/bin/env python3
"""Validate CZ + Friday confl>=0.80 & ent>=0.85 = 428 trades.
Full 3-stage: holdout, walk-forward, 2026 OOS.
Also test:
1. Trail power variations (5-12) on the new combo
2. Whether ent threshold matters (0.70 vs 0.85 gives same count)
3. Stack with other trail powers to help marginal Thu/Tue
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

# ════════════════════════════════════════════════════════
# New combo: CZ + Friday confl>=0.80 (DA candidate)
# ════════════════════════════════════════════════════════

def make_da():
    """DA: CZ + Friday confl>=0.80 & ent>=0.85."""
    def fn(day):
        result = cz_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            if dd.weekday() == 4:  # Friday
                if (day.cs_confluence_score >= 0.80 and
                    day.cs_entropy_score is not None and day.cs_entropy_score >= 0.85):
                    return result
        return None
    return fn

print("=" * 70)
print("VALIDATION: DA = CZ + Friday confl>=0.80 & ent>=0.85")
print("=" * 70)

# Test at multiple trail powers
for tp in [6, 7, 8]:
    trades = simulate_trades(signals, make_da(), 'DA', cooldown=0, trail_power=tp)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    print(f"\nDA trail_power={tp}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f}")

# Full validation at trail_power=6
trades = simulate_trades(signals, make_da(), 'DA', cooldown=0, trail_power=6)
n = len(trades)
w = sum(1 for t in trades if t.pnl > 0)
wr = w/n*100 if n else 0
pnl = sum(t.pnl for t in trades)
bl = min(t.pnl for t in trades) if trades else 0

print(f"\n{'='*70}")
print(f"DA FULL VALIDATION (trail_power=6)")
print(f"{'='*70}")
print(f"Total: {n}t {wr:.1f}% WR ${pnl:+,.0f} BL=${bl:+,.0f}")

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
    print("2026: no trades")

all_pass = wt == nt and we == ne and wf_pass == wf_total and oos_pass
print(f"\nOVERALL: {'*** ALL PASS ***' if all_pass else 'FAIL'} ({wf_pass}/{wf_total} WF)")

# Show new trades (not in CZ)
cz_trades = simulate_trades(signals, cz_fn, 'CZ', cooldown=0, trail_power=6)
cz_entry_dates = {t.entry_date for t in cz_trades}
new_trades = [t for t in trades if t.entry_date not in cz_entry_dates]
lost_trades = [t for t in cz_trades if t.entry_date not in {t2.entry_date for t2 in trades}]

print(f"\nNew trades vs CZ: +{len(new_trades)} new, -{len(lost_trades)} lost")
for t in sorted(new_trades, key=lambda x: x.entry_date):
    dd = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
    day_sig = next((d for d in signals if d.date == t.entry_date), None)
    if day_sig:
        print(f"  NEW: {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} "
              f"h={day_sig.cs_channel_health:.3f} confl={day_sig.cs_confluence_score:.2f} "
              f"ent={day_sig.cs_entropy_score:.3f} {dow}")
    else:
        print(f"  NEW: {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} {dow}")

if lost_trades:
    print("\nLost trades (displaced):")
    for t in sorted(lost_trades, key=lambda x: x.entry_date):
        print(f"  LOST: {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f}")

# ════════════════════════════════════════════════════════
# Try weaker entropy gate (0.70 gives same count per Part 4)
# ════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Entropy threshold sensitivity on DA")
print(f"{'='*70}")

for ent_th in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50]:
    def make_da_ent(et=ent_th):
        def fn(day):
            result = cz_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 4:
                    if (day.cs_confluence_score >= 0.80 and
                        day.cs_entropy_score is not None and day.cs_entropy_score >= et):
                        return result
            return None
        return fn
    t_trades = simulate_trades(signals, make_da_ent(), 'test', cooldown=0, trail_power=6)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    tpnl = sum(t.pnl for t in t_trades)
    tbl = min(t.pnl for t in t_trades) if t_trades else 0
    flag = "***" if twr >= 100 else ""
    print(f"  ent>={ent_th:.2f}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Can we add Thu with trail_power=7 or 8?
# ════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Thu marginal trades at different trail powers")
print(f"{'='*70}")

# Thu best from v42b was h>=0.25&confl>=0.80: 427-428t 99.8% BL=-$3
for tp in [5, 6, 7, 8, 10]:
    # DA + Thu confl>=0.80
    def make_da_thu(trail_p=tp):
        def fn(day):
            result = cz_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                # Friday gate (DA)
                if dd.weekday() == 4:
                    if (day.cs_confluence_score >= 0.80 and
                        day.cs_entropy_score is not None and day.cs_entropy_score >= 0.85):
                        return result
                # Thursday gate (candidate)
                if dd.weekday() == 3:
                    if day.cs_confluence_score >= 0.80:
                        return result
            return None
        return fn
    t_trades = simulate_trades(signals, make_da_thu(), 'test', cooldown=0, trail_power=tp)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    tpnl = sum(t.pnl for t in t_trades)
    tbl = min(t.pnl for t in t_trades) if t_trades else 0
    flag = "***" if twr >= 100 else ""
    print(f"  DA+Thu confl>=0.80 tp={tp}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

    # DA + Thu confl>=0.80 & ent>=0.85
    def make_da_thu_ent(trail_p=tp):
        def fn(day):
            result = cz_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 4:
                    if (day.cs_confluence_score >= 0.80 and
                        day.cs_entropy_score is not None and day.cs_entropy_score >= 0.85):
                        return result
                if dd.weekday() == 3:
                    if (day.cs_confluence_score >= 0.80 and
                        day.cs_entropy_score is not None and day.cs_entropy_score >= 0.85):
                        return result
            return None
        return fn
    t_trades = simulate_trades(signals, make_da_thu_ent(), 'test', cooldown=0, trail_power=tp)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    tpnl = sum(t.pnl for t in t_trades)
    tbl = min(t.pnl for t in t_trades) if t_trades else 0
    flag = "***" if twr >= 100 else ""
    print(f"  DA+Thu confl>=0.80&ent>=0.85 tp={tp}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Can we add Tue with trail_power variation?
# ════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Tue marginal trades at different trail powers")
print(f"{'='*70}")

for tp in [5, 6, 7, 8, 10]:
    # DA + Tue confl>=0.80
    def make_da_tue(trail_p=tp):
        def fn(day):
            result = cz_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 4:
                    if (day.cs_confluence_score >= 0.80 and
                        day.cs_entropy_score is not None and day.cs_entropy_score >= 0.85):
                        return result
                if dd.weekday() == 1:
                    if day.cs_confluence_score >= 0.80:
                        return result
            return None
        return fn
    t_trades = simulate_trades(signals, make_da_tue(), 'test', cooldown=0, trail_power=tp)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    tpnl = sum(t.pnl for t in t_trades)
    tbl = min(t.pnl for t in t_trades) if t_trades else 0
    flag = "***" if twr >= 100 else ""
    print(f"  DA+Tue confl>=0.80 tp={tp}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

    # DA + Tue confl>=0.80 & ent>=0.95
    def make_da_tue_ent(trail_p=tp):
        def fn(day):
            result = cz_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 4:
                    if (day.cs_confluence_score >= 0.80 and
                        day.cs_entropy_score is not None and day.cs_entropy_score >= 0.85):
                        return result
                if dd.weekday() == 1:
                    if (day.cs_confluence_score >= 0.80 and
                        day.cs_entropy_score is not None and day.cs_entropy_score >= 0.95):
                        return result
            return None
        return fn
    t_trades = simulate_trades(signals, make_da_tue_ent(), 'test', cooldown=0, trail_power=tp)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    tpnl = sum(t.pnl for t in t_trades)
    tbl = min(t.pnl for t in t_trades) if t_trades else 0
    flag = "***" if twr >= 100 else ""
    print(f"  DA+Tue confl>=0.80&ent>=0.95 tp={tp}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Can we add ALL THREE (Tue+Thu+Fri) at high trail power?
# ════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ALL THREE: Tue+Thu+Fri confl>=0.80 at different trail powers")
print(f"{'='*70}")

for tp in [5, 6, 7, 8, 10, 12]:
    def make_da_all(trail_p=tp):
        def fn(day):
            result = cz_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() in (1, 3, 4):  # Tue, Thu, Fri
                    if day.cs_confluence_score >= 0.80:
                        return result
            return None
        return fn
    t_trades = simulate_trades(signals, make_da_all(), 'test', cooldown=0, trail_power=tp)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    tpnl = sum(t.pnl for t in t_trades)
    tbl = min(t.pnl for t in t_trades) if t_trades else 0
    flag = "***" if twr >= 100 else ""
    print(f"  TuThFr confl>=0.80 tp={tp:2d}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Remaining v42b experiments: relax base thresholds, confl gate
# ════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Relaxing existing CZ conditions")
print(f"{'='*70}")

for h_buy_val in [0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30]:
    for h_sell_val in [0.30, 0.29, 0.28, 0.27, 0.26, 0.25]:
        def make_relaxed(hb=h_buy_val, hs=h_sell_val):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_b = hb
                    h_s = hs
                    dd = day.date.date() if hasattr(day.date, 'date') else day.date
                    vix_val = vix_map.get(day.date, 22)
                    spy_d = spy_dist_map.get(day.date, 0)
                    relax = dd.weekday() == 0 or vix_val > 25
                    if day.cs_action == 'BUY' and spy_d < -1.0:
                        relax = True
                    if relax:
                        h_b = min(h_b, 0.22)
                        h_s = min(h_s, 0.22)
                    if dd.weekday() == 2:
                        h_b = min(h_b, 0.14)
                        h_s = min(h_s, 0.14)
                    h_thresh = h_b if day.cs_action == 'BUY' else h_s
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        t_trades = simulate_trades(signals, make_relaxed(), 'test', cooldown=0, trail_power=6)
        tn = len(t_trades)
        tw = sum(1 for t in t_trades if t.pnl > 0)
        twr = tw/tn*100 if tn else 0
        if tn > 426 and twr >= 100:
            tpnl = sum(t.pnl for t in t_trades)
            print(f"  BUY h>={h_buy_val:.2f} SELL h>={h_sell_val:.2f}: {tn}t {twr:.1f}% ${tpnl:+,.0f} ***")

print(f"\n{'='*70}")
print("Confluence gate relaxation")
print(f"{'='*70}")

for confl_gate_val in [0.88, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]:
    def make_confl_relax(cg=confl_gate_val):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                h_buy = 0.38
                h_sell = 0.31
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                vix_val = vix_map.get(day.date, 22)
                spy_d = spy_dist_map.get(day.date, 0)
                relax = dd.weekday() == 0 or vix_val > 25
                if day.cs_action == 'BUY' and spy_d < -1.0:
                    relax = True
                if relax:
                    h_buy = min(h_buy, 0.22)
                    h_sell = min(h_sell, 0.22)
                if dd.weekday() == 2:
                    h_buy = min(h_buy, 0.14)
                    h_sell = min(h_sell, 0.14)
                h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= cg:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    t_trades = simulate_trades(signals, make_confl_relax(), 'test', cooldown=0, trail_power=6)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    twr = tw/tn*100 if tn else 0
    tpnl = sum(t.pnl for t in t_trades)
    tbl = min(t.pnl for t in t_trades) if t_trades else 0
    flag = "***" if twr >= 100 else ""
    print(f"  confl>={confl_gate_val:.2f}: {tn}t {twr:.1f}% ${tpnl:+,.0f} BL=${tbl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Direction-specific DOW relaxation (from v42b Part 9)
# ════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Direction-specific DOW relaxation")
print(f"{'='*70}")

for direction in ['BUY', 'SELL']:
    for extra_dow, dow_label in [({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri"),
                                  ({1,3}, "Tue+Thu"), ({3,4}, "Thu+Fri"),
                                  ({1,4}, "Tue+Fri"), ({1,3,4}, "TuThFr")]:
        for h_threshold in [0.30, 0.25, 0.22, 0.20, 0.15, 0.10]:
            def make_dir_dow(dr=direction, ed=extra_dow, ht=h_threshold):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd.weekday() in ed and day.cs_action == dr:
                            if day.cs_channel_health >= ht:
                                return result
                    return None
                return fn
            t_trades = simulate_trades(signals, make_dir_dow(), 'test', cooldown=0, trail_power=6)
            tn = len(t_trades)
            tw = sum(1 for t in t_trades if t.pnl > 0)
            twr = tw/tn*100 if tn else 0
            if tn > 426 and twr >= 100:
                tpnl = sum(t.pnl for t in t_trades)
                print(f"  {direction} {dow_label} h>={h_threshold:.2f}: {tn}t {twr:.1f}% ${tpnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Conditional DOW+Direction+VIX/SPY combos
# ════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Conditional DOW+Direction+VIX/SPY combos")
print(f"{'='*70}")

conditions = [
    ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
    ("VIX<18", lambda d: vix_map.get(d.date, 22) < 18),
    ("VIX>25", lambda d: vix_map.get(d.date, 22) > 25),
    ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
    ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
    ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
    ("conf>=0.60", lambda d: d.cs_confidence >= 0.60),
    ("pos<0.90", lambda d: d.cs_position_score < 0.90),
]

for direction in ['BUY', 'SELL']:
    for extra_dow, dow_label in [({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri")]:
        for cond_label, cond_check in conditions:
            for h_threshold in [0.25, 0.20, 0.15, 0.10]:
                def make_cond_dir(dr=direction, ed=extra_dow, cf=cond_check, ht=h_threshold):
                    def fn(day):
                        result = cz_fn(day)
                        if result is not None:
                            return result
                        result = _tf0_base(day)
                        if result is not None:
                            dd = day.date.date() if hasattr(day.date, 'date') else day.date
                            if dd.weekday() in ed and day.cs_action == dr:
                                if cf(day) and day.cs_channel_health >= ht:
                                    return result
                        return None
                    return fn
                t_trades = simulate_trades(signals, make_cond_dir(), 'test', cooldown=0, trail_power=6)
                tn = len(t_trades)
                tw = sum(1 for t in t_trades if t.pnl > 0)
                twr = tw/tn*100 if tn else 0
                if twr >= 100 and tn > 426:
                    tpnl = sum(t.pnl for t in t_trades)
                    print(f"  {direction} {dow_label} {cond_label} h>={h_threshold:.2f}: {tn}t {twr:.1f}% ${tpnl:+,.0f} ***")

print("\nDone")
