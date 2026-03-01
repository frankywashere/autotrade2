#!/usr/bin/env python3
"""v39: Monday conditional relaxation + combined conditions.
Best from v38: Mon h>=0.22 = 419 trades, 100% WR.
Fine-tune Monday threshold, combine with VIX>25 and BUY SPY<-1%."""
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

print("=" * 100)
print("v39: MONDAY RELAXATION + COMBINED CONDITIONS")
print("=" * 100)

# ════════════════════════════════════════════════════════
# Fine-tune Monday h threshold
# ════════════════════════════════════════════════════════
print("\n--- Fine Monday h sweep ---")
for h_pct in range(30, 14, -1):
    h = h_pct / 100.0
    def make_mon(hr=h):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                h_buy = 0.38
                h_sell = 0.31
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 0:  # Monday
                    h_buy = min(h_buy, hr)
                    h_sell = min(h_sell, hr)
                h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_mon(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 414 else (" <-- 100%" if wr >= 100 else "")
    print(f"  Mon h>={h:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ════════════════════════════════════════════════════════
# Monday-only direction-specific
# ════════════════════════════════════════════════════════
print("\n--- Monday BUY-only relaxation ---")
for h_pct in range(30, 14, -2):
    h = h_pct / 100.0
    def make_mon_buy(hr=h):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                h_buy = 0.38
                h_sell = 0.31
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 0 and day.cs_action == 'BUY':
                    h_buy = min(h_buy, hr)
                h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_mon_buy(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 414 else (" <-- 100%" if wr >= 100 else "")
    print(f"  Mon BUY h>={h:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

print("\n--- Monday SELL-only relaxation ---")
for h_pct in range(30, 14, -2):
    h = h_pct / 100.0
    def make_mon_sell(hr=h):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                h_buy = 0.38
                h_sell = 0.31
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 0 and day.cs_action == 'SELL':
                    h_sell = min(h_sell, hr)
                h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_mon_sell(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 414 else (" <-- 100%" if wr >= 100 else "")
    print(f"  Mon SELL h>={h:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ════════════════════════════════════════════════════════
# Combined: Monday + VIX>25
# ════════════════════════════════════════════════════════
print("\n--- Combined: Monday OR VIX>25 relaxation ---")
for h_pct in range(30, 14, -1):
    h = h_pct / 100.0
    def make_combo1(hr=h):
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
                if dd.weekday() == 0 or vix > 25:
                    h_buy = min(h_buy, hr)
                    h_sell = min(h_sell, hr)
                h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_combo1(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 419 else (" <-- 100%" if wr >= 100 else "")
    print(f"  Mon|VIX>25 h>={h:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ════════════════════════════════════════════════════════
# Combined: Monday + BUY SPY<-1%
# ════════════════════════════════════════════════════════
print("\n--- Combined: Monday + BUY SPY<-1% relaxation ---")
for h_pct in range(26, 14, -1):
    h = h_pct / 100.0
    def make_combo2(hr=h):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                h_buy = 0.38
                h_sell = 0.31
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                spy_d = spy_dist_map.get(day.date, 0)
                relax = dd.weekday() == 0  # Monday always relax
                if day.cs_action == 'BUY' and spy_d < -1.0:
                    relax = True  # BUY when SPY crashed
                if relax:
                    h_buy = min(h_buy, hr)
                    h_sell = min(h_sell, hr)
                h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_combo2(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 419 else (" <-- 100%" if wr >= 100 else "")
    print(f"  Mon+BSPY h>={h:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ════════════════════════════════════════════════════════
# Triple combo: Monday + VIX>25 + BUY SPY<-1%
# ════════════════════════════════════════════════════════
print("\n--- Triple: Monday | VIX>25 | BUY SPY<-1% ---")
for h_pct in range(26, 14, -1):
    h = h_pct / 100.0
    def make_triple(hr=h):
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
                    h_buy = min(h_buy, hr)
                    h_sell = min(h_sell, hr)
                h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_triple(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 419 else (" <-- 100%" if wr >= 100 else "")
    print(f"  Mon|VIX25|BSPY h>={h:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ════════════════════════════════════════════════════════
# Full validation of best: Mon h>=0.22 (419t)
# ════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("FULL VALIDATION: Mon h>=0.22 (best candidate)")
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
            if dd.weekday() == 0:  # Monday
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

n_train = len(train)
w_train = sum(1 for t in train if t.pnl > 0)
wr_train = w_train/n_train*100 if n_train else 0
print(f"Train (2016-2021): {n_train} trades, {wr_train:.1f}% WR, ${sum(t.pnl for t in train):+,.0f}")

n_test = len(test)
w_test = sum(1 for t in test if t.pnl > 0)
wr_test = w_test/n_test*100 if n_test else 0
print(f"Test  (2022-2025): {n_test} trades, {wr_test:.1f}% WR, ${sum(t.pnl for t in test):+,.0f}")

holdout_pass = wr_train >= 100 and wr_test >= 100
print(f"Holdout: {'PASS' if holdout_pass else 'FAIL'}")

wf_pass = 0
wf_total = 0
print("\nWalk-forward:")
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

if oos:
    n_oos = len(oos)
    w_oos = sum(1 for t in oos if t.pnl > 0)
    wr_oos = w_oos/n_oos*100
    pnl_oos = sum(t.pnl for t in oos)
    bl_oos = min(t.pnl for t in oos)
    print(f"\n2026 OOS: {n_oos} trades, {wr_oos:.1f}% WR, ${pnl_oos:+,.0f}, BL=${bl_oos:+,.0f}")
    oos_pass = wr_oos >= 100
    print(f"OOS: {'PASS' if oos_pass else 'FAIL'}")
    for t in sorted(oos, key=lambda x: x.entry_date):
        day = day_map.get(t.entry_date)
        src = "V5" if day and day.v5_take_bounce else "CS"
        dd = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        print(f"  {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} src={src} "
              f"h={day.cs_channel_health:.3f if day else 0} {dow}")
else:
    oos_pass = False

# New trades vs CX
print("\nNew trades vs CX:")
from v15.validation.combo_backtest import _make_v31_cx
cx_fn = _make_v31_cx(*args)
cx_trades = simulate_trades(signals, cx_fn, 'CX', cooldown=0, trail_power=6)
cx_dates = {t.entry_date for t in cx_trades}
best_dates = {t.entry_date for t in trades}
new_dates = best_dates - cx_dates
lost_dates = cx_dates - best_dates
for nd in sorted(new_dates):
    t = next((x for x in trades if x.entry_date == nd), None)
    day = day_map.get(nd)
    if t and day:
        dd = nd.date() if hasattr(nd, 'date') else nd
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        print(f"  NEW: {str(nd)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} h={day.cs_channel_health:.3f} "
              f"c={day.cs_confidence:.3f} confl={day.cs_confluence_score:.2f} {dow}")
if lost_dates:
    for ld in sorted(lost_dates):
        t = next((x for x in cx_trades if x.entry_date == ld), None)
        day = day_map.get(ld)
        if t and day:
            print(f"  LOST: {str(ld)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} h={day.cs_channel_health:.3f}")

all_pass = holdout_pass and wf_pass == wf_total and oos_pass
print(f"\n{'='*100}")
print(f"VERDICT: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f}")
print(f"  Holdout: {'PASS' if holdout_pass else 'FAIL'}")
print(f"  WF: {wf_pass}/{wf_total} {'PASS' if wf_pass == wf_total else 'FAIL'}")
print(f"  OOS: {'PASS' if oos_pass else 'FAIL'}")
print(f"  OVERALL: {'*** ALL STAGES PASS ***' if all_pass else 'FAIL'}")
print(f"{'='*100}")

# ════════════════════════════════════════════════════════
# Other DOW exploration
# ════════════════════════════════════════════════════════
print("\n--- Other day-of-week relaxation ---")
for dow_idx, dow_name in [(1, 'Tue'), (2, 'Wed'), (3, 'Thu'), (4, 'Fri')]:
    for h_pct in range(30, 14, -2):
        h = h_pct / 100.0
        def make_other_dow(di=dow_idx, hr=h):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_sell = 0.31
                    dd = day.date.date() if hasattr(day.date, 'date') else day.date
                    if dd.weekday() == di:
                        h_buy = min(h_buy, hr)
                        h_sell = min(h_sell, hr)
                    h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades2 = simulate_trades(signals, make_other_dow(), 'test', cooldown=0, trail_power=6)
        n2 = len(trades2)
        w2 = sum(1 for t in trades2 if t.pnl > 0)
        wr2 = w2/n2*100 if n2 else 0
        if wr2 >= 100 and n2 > 414:
            pnl2 = sum(t.pnl for t in trades2)
            print(f"  {dow_name} h>={h:.2f}: {n2} trades, {wr2:.1f}% WR, ${pnl2:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Add Monday to CX: Mon+other_DOW combined
# ════════════════════════════════════════════════════════
print("\n--- Mon + other DOW combined ---")
for dow_idx, dow_name in [(1, 'Tue'), (2, 'Wed'), (3, 'Thu'), (4, 'Fri')]:
    for h_pct in range(26, 14, -2):
        h = h_pct / 100.0
        def make_mon_plus(di=dow_idx, hr=h):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_sell = 0.31
                    dd = day.date.date() if hasattr(day.date, 'date') else day.date
                    if dd.weekday() == 0:  # Monday always relax to 0.22
                        h_buy = min(h_buy, 0.22)
                        h_sell = min(h_sell, 0.22)
                    elif dd.weekday() == di:  # Other day relax to hr
                        h_buy = min(h_buy, hr)
                        h_sell = min(h_sell, hr)
                    h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades2 = simulate_trades(signals, make_mon_plus(), 'test', cooldown=0, trail_power=6)
        n2 = len(trades2)
        w2 = sum(1 for t in trades2 if t.pnl > 0)
        wr2 = w2/n2*100 if n2 else 0
        if wr2 >= 100 and n2 > 419:
            pnl2 = sum(t.pnl for t in trades2)
            print(f"  Mon+{dow_name} h>={h:.2f}: {n2} trades, {wr2:.1f}% WR, ${pnl2:+,.0f} ***")

# Mon + VIX>25 with different thresholds
print("\n--- Mon h>=0.22 + VIX>25 with different thresholds ---")
for vix_h_pct in range(30, 14, -2):
    vix_h = vix_h_pct / 100.0
    def make_mon_vix(vh=vix_h):
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
                if dd.weekday() == 0:
                    h_buy = min(h_buy, 0.22)
                    h_sell = min(h_sell, 0.22)
                if vix > 25:
                    h_buy = min(h_buy, vh)
                    h_sell = min(h_sell, vh)
                h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades2 = simulate_trades(signals, make_mon_vix(), 'test', cooldown=0, trail_power=6)
    n2 = len(trades2)
    w2 = sum(1 for t in trades2 if t.pnl > 0)
    wr2 = w2/n2*100 if n2 else 0
    pnl2 = sum(t.pnl for t in trades2)
    bl2 = min(t.pnl for t in trades2) if trades2 else 0
    marker = " ***" if wr2 >= 100 and n2 > 419 else (" <-- 100%" if wr2 >= 100 else "")
    print(f"  Mon22+VIX25 h>={vix_h:.2f}: {n2:4d}t {wr2:5.1f}% WR ${pnl2:+9,.0f} BL=${bl2:+,.0f}{marker}")

print("\nDone")
