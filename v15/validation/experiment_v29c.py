#!/usr/bin/env python3
"""v29c: h>=0.40 bypass = 358 trades, 100% WR, $610K.
confl>=0.90 bypass = 353 trades, 100% WR, $600K.
Test: h40 OR confl90 combined, profile new trades, fine-tune."""
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
    def fn(day):
        result = ct_fn(day)
        if result is not None: return result
        result = all_base_fn(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        for check in checks:
            if check(day, action, conf): return result
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

# ── SECTION 1: Profile h>=0.40 bypass trades ──
print("\n=== H>=0.40 BYPASS PROFILING ===")
run('h>=0.40 bypass', [lambda d, a, c: d.cs_channel_health >= 0.40], detail=True)

# ── SECTION 2: h40 OR confl90 ──
print("\n=== H40 OR CONFL90 COMBINATIONS ===")
combos = [
    ('h40 OR confl90', [lambda d, a, c: d.cs_channel_health >= 0.40 or d.cs_confluence_score >= 0.90]),
    ('h40 OR confl88', [lambda d, a, c: d.cs_channel_health >= 0.40 or d.cs_confluence_score >= 0.88]),
    ('h40 OR confl85', [lambda d, a, c: d.cs_channel_health >= 0.40 or d.cs_confluence_score >= 0.85]),
    ('h40 OR confl100', [lambda d, a, c: d.cs_channel_health >= 0.40 or d.cs_confluence_score >= 1.0]),
    ('h42 OR confl90', [lambda d, a, c: d.cs_channel_health >= 0.42 or d.cs_confluence_score >= 0.90]),
    ('h38 OR confl90', [lambda d, a, c: d.cs_channel_health >= 0.38 or d.cs_confluence_score >= 0.90]),
    ('h35 OR confl90', [lambda d, a, c: d.cs_channel_health >= 0.35 or d.cs_confluence_score >= 0.90]),
]

for name, checks in combos:
    run(name, checks)

# Best combo with detail
print("\n=== BEST: h40 OR confl90 DETAIL ===")
run('h40|cf90 detail', [lambda d, a, c: d.cs_channel_health >= 0.40 or d.cs_confluence_score >= 0.90], detail=True)

# ── SECTION 3: Even more combinations ──
print("\n=== TRIPLE OR BYPASS ===")
triples = [
    ('h40|cf90|BUYh50SPY', [lambda d, a, c: d.cs_channel_health >= 0.40 or d.cs_confluence_score >= 0.90 or (a == 'BUY' and d.cs_channel_health >= 0.50 and spy_dist_map.get(d.date, -999) >= 0)]),
    ('h40|cf90|h50cf80', [lambda d, a, c: d.cs_channel_health >= 0.40 or d.cs_confluence_score >= 0.90 or (d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 0.80)]),
    ('h40|cf90|BUYh50', [lambda d, a, c: d.cs_channel_health >= 0.40 or d.cs_confluence_score >= 0.90 or (a == 'BUY' and d.cs_channel_health >= 0.50)]),
    ('h38|cf90', [lambda d, a, c: d.cs_channel_health >= 0.38 or d.cs_confluence_score >= 0.90]),
    ('h35|cf90', [lambda d, a, c: d.cs_channel_health >= 0.35 or d.cs_confluence_score >= 0.90]),
    ('h35|cf88', [lambda d, a, c: d.cs_channel_health >= 0.35 or d.cs_confluence_score >= 0.88]),
]

for name, checks in triples:
    run(name, checks)

# ── SECTION 4: Fine health sweep near boundary ──
print("\n=== FINE HEALTH BOUNDARY ===")
for h in [0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35]:
    def make_h(hh):
        return lambda d, a, c: d.cs_channel_health >= hh
    fn = make_ct_plus_bypass([make_h(h)])
    trades = simulate_trades(signals, fn, f'h{h}', cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in ct_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  h>={h:.2f}: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

# ── SECTION 5: Fine health + confl OR boundary ──
print("\n=== FINE h OR cf BOUNDARY ===")
for h in [0.42, 0.40, 0.38]:
    for cf in [0.92, 0.90, 0.88]:
        def make_or(hh, cc):
            return lambda d, a, c: d.cs_channel_health >= hh or d.cs_confluence_score >= cc
        fn = make_ct_plus_bypass([make_or(h, cf)])
        trades = simulate_trades(signals, fn, f'h{h}|cf{cf}', cooldown=0, trail_power=6)
        n = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / n * 100
        bl = min(t.pnl for t in trades)
        pnl = sum(t.pnl for t in trades)
        new = len([t for t in trades if t.entry_date not in ct_dates])
        marker = " ***" if wr >= 100 else ""
        print(f"  h>={h:.2f}|cf>={cf:.2f}: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

# ── SECTION 6: Trail power sweep on best ──
print("\n=== TRAIL POWER SWEEP ON h40|cf90 ===")
fn_best = make_ct_plus_bypass([lambda d, a, c: d.cs_channel_health >= 0.40 or d.cs_confluence_score >= 0.90])
for power in [5, 6, 7, 8, 10, 12, 15, 20]:
    trades = simulate_trades(signals, fn_best, f'p{power}', cooldown=0, trail_power=power)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    print(f"  power={power:2d}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# ── SECTION 7: Train/test split ──
print("\n=== TRAIN/TEST SPLIT ON h40|cf90 ===")
best_trades = simulate_trades(signals, fn_best, 'best', cooldown=0, trail_power=6)
for label, subset in [('Train 2016-2021', [t for t in best_trades if t.entry_date.year <= 2021]),
                       ('Test 2022-2025', [t for t in best_trades if t.entry_date.year > 2021])]:
    n = len(subset)
    w = sum(1 for t in subset if t.pnl > 0)
    wr = w / n * 100 if n > 0 else 0
    pnl = sum(t.pnl for t in subset)
    bl = min(t.pnl for t in subset) if subset else 0
    print(f"  {label}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# ── SECTION 8: What about h>=0.40 bypass on ALL trades (not just TF0) ──
# i.e., bypass CP gates AND TF gates for h>=0.40
print("\n=== h>=0.40 FULL BYPASS (no TF gates, no CP gates) ===")
# This would mean: any CS BUY/SELL signal with h>=0.40 that passes VIX cascade
fn_full_bypass = make_ct_plus_bypass([lambda d, a, c: d.cs_channel_health >= 0.40])
trades_full = simulate_trades(signals, fn_full_bypass, 'full_bypass', cooldown=0, trail_power=6)
n = len(trades_full)
wins = sum(1 for t in trades_full if t.pnl > 0)
wr = wins / n * 100
pnl = sum(t.pnl for t in trades_full)
bl = min(t.pnl for t in trades_full)
print(f"  h>=0.40 full bypass: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")
# (This IS the same as h>=0.40 bypass since all_base_fn is already TF0)
# But let's confirm the number matches

# ── SECTION 9: Comparison table ──
print("\n=== COMPARISON TABLE ===")
comparisons = [
    ('CT (v27)', ct_fn),
    ('CT + h40 bypass', make_ct_plus_bypass([lambda d, a, c: d.cs_channel_health >= 0.40])),
    ('CT + cf90 bypass', make_ct_plus_bypass([lambda d, a, c: d.cs_confluence_score >= 0.90])),
    ('CT + h40|cf90', make_ct_plus_bypass([lambda d, a, c: d.cs_channel_health >= 0.40 or d.cs_confluence_score >= 0.90])),
    ('CT + h38|cf90', make_ct_plus_bypass([lambda d, a, c: d.cs_channel_health >= 0.38 or d.cs_confluence_score >= 0.90])),
]

print(f"  {'Name':30s} {'Trades':>6} {'WR':>6} {'PnL':>10} {'Sharpe':>7} {'BL':>8}")
for name, fn in comparisons:
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades)
    # Approx Sharpe
    returns = [t.pnl for t in trades]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252/10) if np.std(returns) > 0 else 0
    print(f"  {name:30s} {n:6d} {wr:5.1f}% ${pnl:+9,.0f} {sharpe:7.2f} ${bl:+7,.0f}")

print("\nDone")
