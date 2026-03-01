#!/usr/bin/env python3
"""v32: Refine V5 bounce filter for CU integration.
Best from v31: h<0.50 & pos<0.80 = 388 trades, 100% WR.
Goals:
1. Fine-tune h and pos boundaries
2. Train/test split validation
3. Profile all new V5 trades
4. Test combining with gap recovery (even if gap alone fails 100%)
5. Compare top V5 filter candidates"""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _build_filter_cascade,
    _make_v28_cu, _SigProxy, _AnalysisProxy, _floor_stop_tp
)

cache_path = Path(__file__).parent / 'combo_cache' / 'combo_signals.pkl'
with open(cache_path, 'rb') as f:
    data = pickle.load(f)
signals = data['signals']
vix_daily = data.get('vix_daily')
spy_daily = data.get('spy_daily')

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

cu_fn = _make_v28_cu(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50,
                      vix_map, spy_return_map, spy_ret_2d)
cu_trades = simulate_trades(signals, cu_fn, 'CU', cooldown=0, trail_power=6)
cu_dates = {t.entry_date for t in cu_trades}
day_map = {day.date: day for day in signals}
print(f"CU baseline: {len(cu_trades)} trades, {sum(1 for t in cu_trades if t.pnl>0)/len(cu_trades)*100:.1f}% WR, ${sum(t.pnl for t in cu_trades):+,.0f}")

def make_cu_v5(v5_check=None):
    def fn(day):
        result = cu_fn(day)
        if result is not None: return result
        if day.v5_take_bounce:
            if v5_check is None or v5_check(day):
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn

def run_v5(name, check, detail=False):
    fn = make_cu_v5(check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = sorted([t for t in trades if t.entry_date not in cu_dates], key=lambda x: x.entry_date)
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:45s}: {n:3d} trades (+{len(new):2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")
    if detail and new:
        for t in new:
            day = day_map.get(t.entry_date)
            if day:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
                train = "TRAIN" if t.entry_date.year <= 2021 else "TEST"
                pnl_marker = " <-- LOSS" if t.pnl <= 0 else ""
                print(f"    {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+7,.0f} "
                      f"v5c={day.v5_confidence or 0:.3f} h={day.cs_channel_health:.3f} "
                      f"pos={day.cs_position_score:.2f} confl={day.cs_confluence_score:.2f} "
                      f"VIX={vix_map.get(day.date,22):.1f} {dow} {train}{pnl_marker}")
    return trades, new

# ── SECTION 1: Fine h boundary for V5 ──
print("\n=== FINE H BOUNDARY (V5 filter: h<X) ===")
for h in [0.55, 0.54, 0.53, 0.52, 0.51, 0.50, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.40, 0.35, 0.30]:
    def make_h(hh):
        return lambda d: d.cs_channel_health < hh
    run_v5(f'V5 h<{h:.2f}', make_h(h))

# ── SECTION 2: Fine pos boundary with h<0.50 ──
print("\n=== FINE POS BOUNDARY (V5 filter: h<0.50 & pos<X) ===")
for p in [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.30, 0.10]:
    def make_hp(pp):
        return lambda d: d.cs_channel_health < 0.50 and d.cs_position_score < pp
    run_v5(f'V5 h<0.50 & pos<{p:.2f}', make_hp(p))

# ── SECTION 3: Test h<0.51 (slightly wider) ──
print("\n=== H<0.51 VARIANTS ===")
for p in [0.99, 0.90, 0.85, 0.80, 0.70, 0.50]:
    def make_hp51(pp):
        return lambda d: d.cs_channel_health < 0.51 and d.cs_position_score < pp
    run_v5(f'V5 h<0.51 & pos<{p:.2f}', make_hp51(p))

# ── SECTION 4: Best candidate detail + train/test ──
print("\n=== BEST CANDIDATE: h<0.50 & pos<0.80 DETAIL ===")
best_check = lambda d: d.cs_channel_health < 0.50 and d.cs_position_score < 0.80
trades_best, new_best = run_v5('BEST: h<0.50 & pos<0.80', best_check, detail=True)

print("\n=== TRAIN/TEST SPLIT ===")
for label, subset in [('Train 2016-2021', [t for t in trades_best if t.entry_date.year <= 2021]),
                       ('Test 2022-2025', [t for t in trades_best if t.entry_date.year > 2021])]:
    n = len(subset)
    w = sum(1 for t in subset if t.pnl > 0)
    wr = w / n * 100 if n > 0 else 0
    pnl = sum(t.pnl for t in subset)
    bl = min(t.pnl for t in subset) if subset else 0
    new = len([t for t in subset if t.entry_date not in cu_dates])
    print(f"  {label}: {n} trades (+{new} V5), {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# ── SECTION 5: Alternative best candidates for comparison ──
print("\n=== ALTERNATIVE CANDIDATES ===")
candidates = [
    ('A: h<0.50 & pos<0.80', lambda d: d.cs_channel_health < 0.50 and d.cs_position_score < 0.80),
    ('B: v5c>=0.80', lambda d: (d.v5_confidence or 0) >= 0.80),
    ('C: h<0.50 & pos<0.50', lambda d: d.cs_channel_health < 0.50 and d.cs_position_score < 0.50),
    ('D: h<0.50', lambda d: d.cs_channel_health < 0.50),
    ('E: h>=0.30 & SRet>=0', lambda d: d.cs_channel_health >= 0.30 and spy_return_map.get(d.date, 0) >= 0),
    ('F: notFri & SRet>=0', lambda d: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 4 and spy_return_map.get(d.date, 0) >= 0),
    ('G: h>=0.30 & h<0.50', lambda d: d.cs_channel_health >= 0.30 and d.cs_channel_health < 0.50),
    ('H: notFri & h<0.50', lambda d: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 4 and d.cs_channel_health < 0.50),
    ('I: h<0.50 & TF>=2 & SRet>-0.5', lambda d: d.cs_channel_health < 0.50 and _count_tf_confirming(d, 'BUY') >= 2 and spy_return_map.get(d.date, 0) > -0.5),
    ('J: h>=0.40 & TF>=2', lambda d: d.cs_channel_health >= 0.40 and _count_tf_confirming(d, 'BUY') >= 2),
]

print(f"  {'Candidate':45s} {'Trades':>6} {'New':>4} {'WR':>6} {'PnL':>10} {'BL':>8}")
for name, check in candidates:
    fn = make_cu_v5(check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cu_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:45s} {n:6d} {new:4d} {wr:5.1f}% ${pnl:+9,.0f} ${bl:+7,.0f}{marker}")

# ── SECTION 6: Detail on candidate A (best) for all new trades ──
print("\n=== CANDIDATE A DETAIL (ALL V5 TRADES) ===")
best_fn = make_cu_v5(best_check)
best_trades2 = simulate_trades(signals, best_fn, 'A', cooldown=0, trail_power=6)
v5_trades = [t for t in best_trades2 if t.entry_date not in cu_dates]
print(f"V5 trades: {len(v5_trades)}")
for t in sorted(v5_trades, key=lambda x: x.entry_date):
    day = day_map.get(t.entry_date)
    if day:
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        train = "TRAIN" if t.entry_date.year <= 2021 else "TEST"
        print(f"  {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+7,.0f} "
              f"v5c={day.v5_confidence or 0:.3f} h={day.cs_channel_health:.3f} "
              f"pos={day.cs_position_score:.2f} confl={day.cs_confluence_score:.2f} "
              f"cs={day.cs_action} VIX={vix_map.get(day.date,22):.1f} {dow} {train}")

# ── SECTION 7: Trail power sweep on best V5 filter ──
print("\n=== TRAIL POWER SWEEP (h<0.50 & pos<0.80) ===")
for power in [4, 5, 6, 7, 8, 10, 12, 15, 20]:
    fn = make_cu_v5(best_check)
    trades = simulate_trades(signals, fn, f'p{power}', cooldown=0, trail_power=power)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    print(f"  power={power:2d}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# ── SECTION 8: Sharpe / MaxDD comparison ──
print("\n=== FINAL COMPARISON TABLE ===")
combos = [
    ('CU (v28 baseline)', cu_fn),
    ('CU + V5(h<0.50&pos<0.80)', make_cu_v5(best_check)),
    ('CU + V5(v5c>=0.80)', make_cu_v5(lambda d: (d.v5_confidence or 0) >= 0.80)),
    ('CU + V5(raw, no filter)', make_cu_v5()),
]

print(f"  {'Name':40s} {'Trades':>6} {'WR':>6} {'PnL':>10} {'Sharpe':>7} {'BL':>8} {'New':>4}")
for name, fn in combos:
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades)
    returns = [t.pnl for t in trades]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252/10) if np.std(returns) > 0 else 0
    new = len([t for t in trades if t.entry_date not in cu_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:40s} {n:6d} {wr:5.1f}% ${pnl:+9,.0f} {sharpe:7.2f} ${bl:+7,.0f} {new:4d}{marker}")

# ── SECTION 9: V5 with different stop widths on best filter ──
print("\n=== V5 STOP WIDTH ON BEST FILTER ===")
for stop, tp in [(1.0, 2.0), (1.5, 3.0), (2.0, 4.0), (2.5, 5.0), (3.0, 6.0), (4.0, 8.0)]:
    def make_v5_stops(s, t, check):
        def fn(day):
            result = cu_fn(day)
            if result is not None: return result
            if day.v5_take_bounce and check(day):
                return ('BUY', day.v5_confidence or 0.60, s, t, 'V5')
            return None
        return fn
    fn = make_v5_stops(stop, tp, best_check)
    trades = simulate_trades(signals, fn, f's{stop}/t{tp}', cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cu_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  stop={stop}%/tp={tp}%: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

# ── SECTION 10: V5 with conf boost on best filter ──
print("\n=== V5 CONF BOOST ON BEST FILTER ===")
for bc in [0.60, 0.70, 0.80, 0.90, 0.95]:
    def make_v5_bc(boost_conf, check):
        def fn(day):
            result = cu_fn(day)
            if result is not None: return result
            if day.v5_take_bounce and check(day):
                return ('BUY', boost_conf, 2.0, 4.0, 'V5')
            return None
        return fn
    fn = make_v5_bc(bc, best_check)
    trades = simulate_trades(signals, fn, f'bc={bc}', cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    marker = " ***" if wr >= 100 else ""
    print(f"  conf={bc}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

print("\nDone")
