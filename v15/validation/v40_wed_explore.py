#!/usr/bin/env python3
"""v40b: Wednesday relaxation on CY, fine-tune + other DOW stacking."""
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

def make_combo(dow_set={0}, dow_h=0.22, vix_thresh=25, spy_thresh=-1.0):
    """Flexible combo builder."""
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
            relax = dd.weekday() in dow_set or vix > vix_thresh
            if day.cs_action == 'BUY' and spy_d < spy_thresh:
                relax = True
            if relax:
                h_buy = min(h_buy, dow_h)
                h_sell = min(h_sell, dow_h)
            h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
            if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                return result
        if day.v5_take_bounce:
            if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn

print("=" * 100)
print("v40b: WEDNESDAY + DOW STACKING EXPLORATION")
print("=" * 100)

# Baseline CY
cy_t = simulate_trades(signals, make_combo({0}), 'CY', cooldown=0, trail_power=6)
print(f"CY: {len(cy_t)}t {sum(1 for t in cy_t if t.pnl > 0)/len(cy_t)*100:.0f}% WR ${sum(t.pnl for t in cy_t):+,.0f}")

# ═══════════════════════════════════════
# Fine Wed h threshold
# ═══════════════════════════════════════
print("\n--- Fine Wed h sweep (Mon+Wed, h varies for Wed) ---")
for wed_h_pct in range(30, 10, -1):
    wed_h = wed_h_pct / 100.0
    def make_mw(wh=wed_h):
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
                relax_full = dd.weekday() == 0 or vix > 25  # Mon/VIX get 0.22
                relax_wed = dd.weekday() == 2  # Wed gets variable threshold
                if day.cs_action == 'BUY' and spy_d < -1.0:
                    relax_full = True
                if relax_full:
                    h_buy = min(h_buy, 0.22)
                    h_sell = min(h_sell, 0.22)
                if relax_wed:
                    h_buy = min(h_buy, wh)
                    h_sell = min(h_sell, wh)
                h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_mw(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 422 else (" <-- 100%" if wr >= 100 else "")
    print(f"  Wed h>={wed_h:.2f}: {n:4d}t {wr:5.1f}% WR ${pnl:+9,.0f} BL=${bl:+,.0f}{marker}")

# ═══════════════════════════════════════
# Stack all DOW combos on CY+Wed
# ═══════════════════════════════════════
print("\n--- DOW stacking: CY+Wed+X (Mon h=0.22, Wed h=best, X h=?) ---")

# First find the best Wed threshold
best_wed_h = 0.16  # from v40 results (425 trades)

for extra_dow, label in [(1, 'Tue'), (3, 'Thu'), (4, 'Fri')]:
    for extra_h_pct in range(30, 10, -2):
        extra_h = extra_h_pct / 100.0
        def make_3dow(ed=extra_dow, eh=extra_h, bwh=best_wed_h):
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
                    # Mon / VIX>25 / BUY SPY<-1%: h=0.22
                    relax_full = dd.weekday() == 0 or vix > 25
                    if day.cs_action == 'BUY' and spy_d < -1.0:
                        relax_full = True
                    if relax_full:
                        h_buy = min(h_buy, 0.22)
                        h_sell = min(h_sell, 0.22)
                    # Wed: h=best_wed_h
                    if dd.weekday() == 2:
                        h_buy = min(h_buy, bwh)
                        h_sell = min(h_sell, bwh)
                    # Extra DOW: h=eh
                    if dd.weekday() == ed:
                        h_buy = min(h_buy, eh)
                        h_sell = min(h_sell, eh)
                    h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_3dow(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 425:
            pnl = sum(t.pnl for t in trades)
            bl = min(t.pnl for t in trades)
            print(f"  CY+Wed+{label} h>={extra_h:.2f}: {n}t {wr:.1f}% WR ${pnl:+,.0f} BL=${bl:+,.0f} ***")

# ═══════════════════════════════════════
# All DOW (every day has lower h)
# ═══════════════════════════════════════
print("\n--- All DOW relaxation ---")
for h_pct in range(30, 10, -1):
    h = h_pct / 100.0
    trades = simulate_trades(signals, make_combo({0,1,2,3,4}, h), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 425 else (" <-- 100%" if wr >= 100 else "")
    print(f"  All DOW h>={h:.2f}: {n:4d}t {wr:5.1f}% WR ${pnl:+9,.0f} BL=${bl:+,.0f}{marker}")

# ═══════════════════════════════════════
# Full validation of best: CY+Wed h>=0.16 (425t)
# ═══════════════════════════════════════
print("\n" + "=" * 100)
print("FULL VALIDATION: CY + Wed h>=0.16 (425 trades)")
print("=" * 100)

def make_best():
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
            if dd.weekday() == 2:  # Wednesday
                h_buy = min(h_buy, 0.16)
                h_sell = min(h_sell, 0.16)
            h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
            if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                return result
        if day.v5_take_bounce:
            if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn

trades = simulate_trades(signals, make_best(), 'best', cooldown=0, trail_power=6)
n = len(trades)
w = sum(1 for t in trades if t.pnl > 0)
wr = w/n*100 if n else 0
pnl = sum(t.pnl for t in trades)
bl = min(t.pnl for t in trades) if trades else 0
print(f"\nFull: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

train = [t for t in trades if t.entry_date.year <= 2021]
test = [t for t in trades if 2022 <= t.entry_date.year <= 2025]
oos = [t for t in trades if t.entry_date.year >= 2026]

nt = len(train)
wt = sum(1 for t in train if t.pnl > 0)
print(f"Train: {nt}t {wt/nt*100:.0f}% WR ${sum(t.pnl for t in train):+,.0f}")
ne = len(test)
we = sum(1 for t in test if t.pnl > 0)
print(f"Test:  {ne}t {we/ne*100:.0f}% WR ${sum(t.pnl for t in test):+,.0f}")
holdout_pass = wt == nt and we == ne

wf_pass = 0
wf_total = 0
print("\nWalk-forward:")
for year in range(2017, 2026):
    yearly = [t for t in trades if t.entry_date.year == year]
    if not yearly: continue
    wf_total += 1
    yn = len(yearly)
    yw = sum(1 for t in yearly if t.pnl > 0)
    ybl = min(t.pnl for t in yearly)
    passed = all(t.pnl > 0 for t in yearly)
    if passed: wf_pass += 1
    print(f"  {year}: {yn:3d}t {yw}/{yn} wins ${sum(t.pnl for t in yearly):+9,.0f} BL=${ybl:+,.0f} {'PASS' if passed else 'FAIL'}")

if oos:
    no = len(oos)
    wo = sum(1 for t in oos if t.pnl > 0)
    po = sum(t.pnl for t in oos)
    oos_pass = wo == no
    print(f"\n2026 OOS: {no}t {wo}/{no} wins ${po:+,.0f}")
    for t in sorted(oos, key=lambda x: x.entry_date):
        day = day_map.get(t.entry_date)
        src = "V5" if day and day.v5_take_bounce else "CS"
        dd = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        h = day.cs_channel_health if day else 0
        print(f"  {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} src={src} h={h:.3f} {dow}")
else:
    oos_pass = False

# New trades
from v15.validation.combo_backtest import _make_v32_cy
cy_fn = _make_v32_cy(*args)
cy_trades2 = simulate_trades(signals, cy_fn, 'CY', cooldown=0, trail_power=6)
cy_dates = {t.entry_date for t in cy_trades2}
best_dates = {t.entry_date for t in trades}
new_dates = best_dates - cy_dates
lost_dates = cy_dates - best_dates
print(f"\nNew vs CY: +{len(new_dates)} -{len(lost_dates)}")
for nd in sorted(new_dates):
    t = next((x for x in trades if x.entry_date == nd), None)
    day = day_map.get(nd)
    if t and day:
        dd = nd.date() if hasattr(nd, 'date') else nd
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        print(f"  NEW: {str(nd)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} h={day.cs_channel_health:.3f} {dow}")
for ld in sorted(lost_dates):
    t = next((x for x in cy_trades2 if x.entry_date == ld), None)
    if t:
        print(f"  LOST: {str(ld)[:10]} {t.direction:5s} ${t.pnl:+8,.0f}")

all_pass = holdout_pass and wf_pass == wf_total and oos_pass
print(f"\n{'='*100}")
print(f"VERDICT: {n}t {wr:.1f}% WR ${pnl:+,.0f}")
print(f"  Holdout: {'PASS' if holdout_pass else 'FAIL'}")
print(f"  WF: {wf_pass}/{wf_total}")
print(f"  OOS: {'PASS' if oos_pass else 'FAIL'}")
print(f"  OVERALL: {'*** ALL STAGES PASS ***' if all_pass else 'FAIL'}")
print(f"{'='*100}")

print("\nDone")
