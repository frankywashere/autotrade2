#!/usr/bin/env python3
"""v29b: Profile confl100 trades + combine with other bypasses.
confl100 = 349 trades, 100% WR, $594K (bypasses CP gates entirely!).
Also: h50&confl80 = 347, BUY h50&SPY>0 = 347.
Can we combine for even more?"""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _build_filter_cascade,
    _make_v27_ct, _SigProxy, _AnalysisProxy, _floor_stop_tp
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

ct_fn = _make_v27_ct(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50,
                      vix_map, spy_return_map, spy_ret_2d)
ct_trades = simulate_trades(signals, ct_fn, 'CT', cooldown=0, trail_power=6)
ct_dates = {t.entry_date for t in ct_trades}
day_map = {day.date: day for day in signals}
print(f"CT baseline: {len(ct_trades)} trades, {sum(1 for t in ct_trades if t.pnl>0)/len(ct_trades)*100:.1f}% WR, ${sum(t.pnl for t in ct_trades):+,.0f}")

# TF0 base (no CP gates)
def make_all_cs_base():
    def fn(day):
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            return None
        sig = _SigProxy(day)
        ana = _AnalysisProxy(day.cs_tf_states)
        ok, adj, _ = cascade_vix.evaluate(sig, ana, feature_vec=None, bar_datetime=day.date,
                                           higher_tf_data=None, spy_df=None, vix_df=None)
        if not ok or adj < MIN_SIGNAL_CONFIDENCE: return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, adj, s, t, 'CS')
    return fn

all_base_fn = make_all_cs_base()

def make_ct_plus_bypass(checks):
    """CT + bypass recovery for base signals that fail ALL existing gates."""
    def fn(day):
        result = ct_fn(day)
        if result is not None: return result
        result = all_base_fn(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        for check in checks:
            if check(day, action, conf):
                return result
        return None
    return fn

def run(name, checks, detail=False):
    fn = make_ct_plus_bypass(checks)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = sorted([t for t in trades if t.entry_date not in ct_dates], key=lambda x: x.entry_date)
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:50s}: {n:3d} trades (+{len(new):2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")
    if detail and new:
        for t in new:
            day = day_map.get(t.entry_date)
            if day:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
                train = "TRAIN" if t.entry_date.year <= 2021 else "TEST"
                print(f"    {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+7,.0f} "
                      f"h={day.cs_channel_health:.3f} conf={day.cs_confidence:.3f} "
                      f"confl={day.cs_confluence_score:.2f} pos={day.cs_position_score:.2f} "
                      f"VIX={vix_map.get(day.date,22):.1f} {dow} {train}")
    return trades, new

# ── SECTION 1: Profile confl100 ──
print("\n=== CONFL100 PROFILING ===")
run('confl>=1.0', [lambda d, a, c: d.cs_confluence_score >= 1.0], detail=True)
run('confl>=0.98', [lambda d, a, c: d.cs_confluence_score >= 0.98], detail=True)
run('confl>=0.95', [lambda d, a, c: d.cs_confluence_score >= 0.95], detail=True)

# ── SECTION 2: Profile h50&confl80 and BUY h50&SPY>0 ──
print("\n=== OTHER BYPASS PROFILING ===")
run('h50&confl80', [lambda d, a, c: d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 0.80], detail=True)
run('BUY h50&SPY>0', [lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.50 and spy_dist_map.get(d.date, -999) >= 0], detail=True)
run('h50&confl90', [lambda d, a, c: d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 0.90], detail=True)

# ── SECTION 3: Combined ──
print("\n=== COMBINED BYPASS ===")
checks_confl100 = lambda d, a, c: d.cs_confluence_score >= 1.0
checks_h50_confl80 = lambda d, a, c: d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 0.80
checks_buy_h50_spy = lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.50 and spy_dist_map.get(d.date, -999) >= 0
checks_h50_confl90 = lambda d, a, c: d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 0.90
checks_h60_confl70 = lambda d, a, c: d.cs_channel_health >= 0.60 and d.cs_confluence_score >= 0.70
checks_h50_c80 = lambda d, a, c: d.cs_channel_health >= 0.50 and c >= 0.80

combos = [
    ('confl100 + h50confl80', [checks_confl100, checks_h50_confl80]),
    ('confl100 + BUYh50SPY', [checks_confl100, checks_buy_h50_spy]),
    ('confl100 + h50confl90', [checks_confl100, checks_h50_confl90]),
    ('confl100 + h60confl70', [checks_confl100, checks_h60_confl70]),
    ('confl100 + h50c80', [checks_confl100, checks_h50_c80]),
    ('confl100 + h50confl80 + BUYh50SPY', [checks_confl100, checks_h50_confl80, checks_buy_h50_spy]),
    ('confl100 + h50confl80 + h60confl70', [checks_confl100, checks_h50_confl80, checks_h60_confl70]),
]

for name, checks in combos:
    run(name, checks)

# ── SECTION 4: Fine-grained confl sweep for bypass ──
print("\n=== FINE CONFL SWEEP (BYPASS, no CP gates) ===")
for ct in [1.00, 0.98, 0.95, 0.92, 0.90, 0.88, 0.85, 0.83, 0.80]:
    def make_check(threshold):
        return lambda d, a, c: d.cs_confluence_score >= threshold
    run(f'confl>={ct:.2f} bypass', [make_check(ct)])

# ── SECTION 5: Health sweep for bypass ──
print("\n=== HEALTH SWEEP (BYPASS, no CP gates) ===")
for h in [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]:
    def make_check(hh):
        return lambda d, a, c: d.cs_channel_health >= hh
    run(f'h>={h:.2f} bypass', [make_check(h)])

# ── SECTION 6: Health + confluence 2D sweep ──
print("\n=== 2D SWEEP: HEALTH x CONFLUENCE ===")
for h in [0.60, 0.55, 0.50, 0.45, 0.40]:
    for cf in [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
        def make_check(hh, cc):
            return lambda d, a, c: d.cs_channel_health >= hh and d.cs_confluence_score >= cc
        fn = make_ct_plus_bypass([make_check(h, cf)])
        trades = simulate_trades(signals, fn, f'h{h}cf{cf}', cooldown=0, trail_power=6)
        n = len(trades)
        if n <= 345: continue
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / n * 100
        bl = min(t.pnl for t in trades)
        pnl = sum(t.pnl for t in trades)
        new = len([t for t in trades if t.entry_date not in ct_dates])
        marker = " ***" if wr >= 100 else ""
        if wr >= 100:
            print(f"  h>={h:.2f} & cf>={cf:.2f}: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}{marker}")

# ── SECTION 7: BUY-only and SELL-only bypass ──
print("\n=== DIRECTIONAL BYPASS ===")
buy_bypasses = [
    ('BUY confl100', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 1.0),
    ('BUY h50', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.50),
    ('BUY h60', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.60),
    ('BUY h50&confl70', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 0.70),
    ('BUY h40&confl90', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.40 and d.cs_confluence_score >= 0.90),
    ('BUY h40&confl100', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.40 and d.cs_confluence_score >= 1.0),
]
for name, check in buy_bypasses:
    run(name, [check])

sell_bypasses = [
    ('SELL confl100', lambda d, a, c: a == 'SELL' and d.cs_confluence_score >= 1.0),
    ('SELL h50', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.50),
    ('SELL h60', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.60),
    ('SELL h50&confl80', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 0.80),
    ('SELL h50&pos100', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.50 and d.cs_position_score >= 1.0),
    ('SELL h40&confl100', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.40 and d.cs_confluence_score >= 1.0),
]
for name, check in sell_bypasses:
    run(name, [check])

# ── SECTION 8: Best combo detail + train/test ──
print("\n=== BEST COMBO CANDIDATES ===")
# Run the top combos with detail
run('BEST-A: confl100', [checks_confl100], detail=True)

# Test best with different trail powers
print("\n=== TRAIL POWER SWEEP ON CONFL100 ===")
fn_best = make_ct_plus_bypass([checks_confl100])
for power in [5, 6, 7, 8, 10, 12]:
    trades = simulate_trades(signals, fn_best, f'p{power}', cooldown=0, trail_power=power)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    print(f"  power={power:2d}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# Train/test split
print("\n=== TRAIN/TEST SPLIT ===")
trades_best = simulate_trades(signals, fn_best, 'best', cooldown=0, trail_power=6)
for label, subset in [('Train 2016-2021', [t for t in trades_best if t.entry_date.year <= 2021]),
                       ('Test 2022-2025', [t for t in trades_best if t.entry_date.year > 2021])]:
    n = len(subset)
    w = sum(1 for t in subset if t.pnl > 0)
    wr = w / n * 100 if n > 0 else 0
    pnl = sum(t.pnl for t in subset)
    bl = min(t.pnl for t in subset) if subset else 0
    print(f"  {label}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

print("\nDone")
