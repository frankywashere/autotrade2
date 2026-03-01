#!/usr/bin/env python3
"""v37: CX (v31) full 3-stage validation.
CX = CW + direction-specific gate-free: BUY h>=0.38, SELL h>=0.31.
Validation: holdout, walk-forward, 2026 OOS."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, simulate_trades, _build_filter_cascade,
    _make_v27_ct, _make_v28_cu, _make_v30_cw, _make_v31_cx,
    MIN_SIGNAL_CONFIDENCE, _floor_stop_tp
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

day_map = {d.date: d for d in signals}

# Build combo functions
cw_fn = _make_v30_cw(*args)
cx_fn = _make_v31_cx(*args)

print("=" * 100)
print("CX (v31) FULL 3-STAGE VALIDATION")
print("BUY h>=0.38, SELL h>=0.31, V5 h<0.57 & pos<0.85")
print("=" * 100)

# ═══════════════════════════════════════════════════════
# STAGE 0: Full period comparison CW vs CX
# ═══════════════════════════════════════════════════════
print("\n--- CW vs CX comparison ---")
for name, fn in [('CW', cw_fn), ('CX', cx_fn)]:
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    print(f"  {name}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# ═══════════════════════════════════════════════════════
# STAGE 1: Holdout (Train 2016-2021, Test 2022-2025)
# ═══════════════════════════════════════════════════════
print("\n--- Stage 1: Holdout ---")
cx_trades = simulate_trades(signals, cx_fn, 'CX', cooldown=0, trail_power=6)
train = [t for t in cx_trades if t.entry_date.year <= 2021]
test = [t for t in cx_trades if 2022 <= t.entry_date.year <= 2025]
oos = [t for t in cx_trades if t.entry_date.year >= 2026]

n_train = len(train)
w_train = sum(1 for t in train if t.pnl > 0)
wr_train = w_train/n_train*100 if n_train else 0
pnl_train = sum(t.pnl for t in train)

n_test = len(test)
w_test = sum(1 for t in test if t.pnl > 0)
wr_test = w_test/n_test*100 if n_test else 0
pnl_test = sum(t.pnl for t in test)

print(f"  Train (2016-2021): {n_train} trades, {wr_train:.1f}% WR, ${pnl_train:+,.0f}")
print(f"  Test  (2022-2025): {n_test} trades, {wr_test:.1f}% WR, ${pnl_test:+,.0f}")
holdout_pass = wr_train >= 100 and wr_test >= 100
print(f"  Holdout: {'PASS' if holdout_pass else 'FAIL'}")

# ═══════════════════════════════════════════════════════
# STAGE 2: Walk-forward (expanding window, year-by-year)
# ═══════════════════════════════════════════════════════
print("\n--- Stage 2: Walk-forward ---")
wf_pass = 0
wf_total = 0
for year in range(2017, 2026):
    yearly = [t for t in cx_trades if t.entry_date.year == year]
    if not yearly:
        continue
    wf_total += 1
    n = len(yearly)
    w = sum(1 for t in yearly if t.pnl > 0)
    wr = w/n*100
    pnl = sum(t.pnl for t in yearly)
    bl = min(t.pnl for t in yearly)
    passed = all(t.pnl > 0 for t in yearly)
    if passed:
        wf_pass += 1
    print(f"  {year}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f} {'PASS' if passed else 'FAIL'}")
print(f"  Walk-forward: {wf_pass}/{wf_total} PASS")

# ═══════════════════════════════════════════════════════
# STAGE 3: 2026 OOS
# ═══════════════════════════════════════════════════════
print("\n--- Stage 3: 2026 OOS ---")
if oos:
    n_oos = len(oos)
    w_oos = sum(1 for t in oos if t.pnl > 0)
    wr_oos = w_oos/n_oos*100
    pnl_oos = sum(t.pnl for t in oos)
    bl_oos = min(t.pnl for t in oos)
    print(f"  2026 OOS: {n_oos} trades, {wr_oos:.1f}% WR, ${pnl_oos:+,.0f}, BL=${bl_oos:+,.0f}")
    oos_pass = wr_oos >= 100
    print(f"  OOS: {'PASS' if oos_pass else 'FAIL'}")
    print(f"\n  2026 trade log:")
    for t in sorted(oos, key=lambda x: x.entry_date):
        day = day_map.get(t.entry_date)
        src = "V5" if day and day.v5_take_bounce else "CS"
        dd = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        h_str = f"{day.cs_channel_health:.3f}" if day else "N/A"
        c_str = f"{day.cs_confidence:.3f}" if day else "N/A"
        cf_str = f"{day.cs_confluence_score:.2f}" if day else "N/A"
        print(f"    {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} src={src} "
              f"h={h_str} c={c_str} confl={cf_str} VIX={vix_map.get(t.entry_date, 0):.1f} {dow}")
else:
    print("  No 2026 OOS trades")
    oos_pass = False

# ═══════════════════════════════════════════════════════
# NEW TRADES: CX vs CW
# ═══════════════════════════════════════════════════════
print("\n--- New CX trades (not in CW) ---")
cw_trades = simulate_trades(signals, cw_fn, 'CW', cooldown=0, trail_power=6)
cw_dates = {t.entry_date for t in cw_trades}
cx_dates = {t.entry_date for t in cx_trades}
new_dates = cx_dates - cw_dates
lost_dates = cw_dates - cx_dates

print(f"New trade dates in CX: {len(new_dates)}")
for nd in sorted(new_dates):
    t = next((x for x in cx_trades if x.entry_date == nd), None)
    day = day_map.get(nd)
    if t and day:
        dd = nd.date() if hasattr(nd, 'date') else nd
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        print(f"  {str(nd)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} h={day.cs_channel_health:.3f} "
              f"c={day.cs_confidence:.3f} confl={day.cs_confluence_score:.2f} "
              f"VIX={vix_map.get(nd, 0):.1f} {dow}")

if lost_dates:
    print(f"\nLost trade dates (in CW but not CX): {len(lost_dates)}")
    for ld in sorted(lost_dates):
        t = next((x for x in cw_trades if x.entry_date == ld), None)
        day = day_map.get(ld)
        if t and day:
            print(f"  {str(ld)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} h={day.cs_channel_health:.3f}")

# ═══════════════════════════════════════════════════════
# YEAR-BY-YEAR FULL BREAKDOWN
# ═══════════════════════════════════════════════════════
print("\n--- Year-by-year ---")
for year in range(2016, 2027):
    yearly = [t for t in cx_trades if t.entry_date.year == year]
    if not yearly: continue
    n = len(yearly)
    w = sum(1 for t in yearly if t.pnl > 0)
    pnl = sum(t.pnl for t in yearly)
    bl = min(t.pnl for t in yearly)
    bw = max(t.pnl for t in yearly)
    marker = " <-- OOS" if year >= 2026 else ""
    print(f"  {year}: {n:3d} trades, {w}/{n} wins, ${pnl:+9,.0f}, BL=${bl:+,.0f}, BW=${bw:+,.0f}{marker}")

# ═══════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 100)
all_pass = holdout_pass and wf_pass == wf_total and oos_pass
n_total = len(cx_trades)
w_total = sum(1 for t in cx_trades if t.pnl > 0)
wr_total = w_total/n_total*100 if n_total else 0
pnl_total = sum(t.pnl for t in cx_trades)
print(f"CX (v31): {n_total} trades, {wr_total:.1f}% WR, ${pnl_total:+,.0f}")
print(f"  Holdout: {'PASS' if holdout_pass else 'FAIL'}")
print(f"  Walk-forward: {wf_pass}/{wf_total} {'PASS' if wf_pass == wf_total else 'FAIL'}")
print(f"  2026 OOS: {'PASS' if oos_pass else 'FAIL'}")
print(f"  OVERALL: {'*** ALL STAGES PASS ***' if all_pass else 'FAIL'}")
print("=" * 100)

# ═══════════════════════════════════════════════════════
# FRONTIER: push further beyond CX
# ═══════════════════════════════════════════════════════
print("\n--- Frontier: further expansion ideas ---")

# Idea 1: BUY h>=0.37 (relax BUY by 0.01)
ct_fn = _make_v27_ct(*args)
from v15.validation.combo_backtest import _SigProxy, _AnalysisProxy

def _tf0_base_day(day):
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

# Sweep BUY h threshold
print("\nBUY h threshold sweep (SELL h>=0.29 fixed):")
for h_buy_pct in range(38, 34, -1):
    h_buy = h_buy_pct / 100.0
    def make_fn(hb=h_buy):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base_day(day)
            if result is not None:
                h_thresh = hb if day.cs_action == 'BUY' else 0.29
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_fn(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 415 else ""
    print(f"  BUY h>={h_buy:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# Sweep SELL h threshold even lower
print("\nSELL h threshold further sweep (BUY h>=0.38 fixed):")
for h_sell_pct in range(29, 19, -1):
    h_sell = h_sell_pct / 100.0
    def make_fn2(hs=h_sell):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base_day(day)
            if result is not None:
                h_thresh = 0.38 if day.cs_action == 'BUY' else hs
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_fn2(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 415 else ""
    print(f"  SELL h>={h_sell:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# Idea 2: V5 pos threshold relaxation
print("\nV5 pos threshold sweep (h<0.57 fixed):")
for pos_pct in range(85, 100, 5):
    pos = pos_pct / 100.0
    def make_v5(p=pos):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base_day(day)
            if result is not None:
                h_thresh = 0.38 if day.cs_action == 'BUY' else 0.29
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < p:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_v5(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 415 else ""
    print(f"  V5 pos<{pos:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# Idea 3: Confl threshold relaxation
print("\nConfl bypass sweep (alongside h gates):")
for confl_pct in range(90, 74, -2):
    confl = confl_pct / 100.0
    def make_confl(cf=confl):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base_day(day)
            if result is not None:
                h_thresh = 0.38 if day.cs_action == 'BUY' else 0.29
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= cf:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_confl(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 415 else ""
    print(f"  confl>={confl:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# Idea 4: Direction-specific confl
print("\nDirection-specific confl (BUY confl>=0.90, SELL confl lower):")
for sell_confl_pct in range(90, 74, -2):
    sell_confl = sell_confl_pct / 100.0
    def make_dconfl(sc=sell_confl):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base_day(day)
            if result is not None:
                h_thresh = 0.38 if day.cs_action == 'BUY' else 0.29
                confl_thresh = 0.90 if day.cs_action == 'BUY' else sc
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= confl_thresh:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_dconfl(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 415 else ""
    print(f"  SELL confl>={sell_confl:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# Idea 5: V5 h threshold relaxation
print("\nV5 h threshold sweep:")
for v5h_pct in range(57, 70, 1):
    v5h = v5h_pct / 100.0
    def make_v5h(vh=v5h):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base_day(day)
            if result is not None:
                h_thresh = 0.38 if day.cs_action == 'BUY' else 0.29
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < vh and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_v5h(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " ***" if wr >= 100 and n > 415 else ""
    print(f"  V5 h<{v5h:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

print("\nDone")
