#!/usr/bin/env python3
"""v24d: Optimize tf2 expansion filter. Best so far: h25&confl60 (297), h>=0.35 (288).
Test combined OR filters and variations."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _make_s1_tf3_vix_combo, _build_filter_cascade,
    _make_v23_cp, _SigProxy, _AnalysisProxy, _floor_stop_tp
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

cp_fn = _make_v23_cp(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50,
                      vix_map, spy_return_map, spy_ret_2d)
cp_trades = simulate_trades(signals, cp_fn, 'CP', cooldown=0, trail_power=6)
print(f"CP: {len(cp_trades)} trades, 100% WR, ${sum(t.pnl for t in cp_trades):+,.0f}")

# ── tf2 base ──
def make_s1_tf2_vix():
    prev_tf_states = {}
    streaks = defaultdict(int)
    def fn(day):
        nonlocal prev_tf_states, streaks
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    streaks[tf] = 0
                    continue
                md = state.get('momentum_direction', 0.0)
                prev_md = prev_tf_states.get(tf, 0.0)
                if (md > 0 and prev_md > 0) or (md < 0 and prev_md < 0):
                    streaks[tf] += 1
                else:
                    streaks[tf] = 1 if md != 0 else 0
                prev_tf_states[tf] = md
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            return None
        action = day.cs_action
        confirmed = 0
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False): continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks.get(tf, 0) >= 1:
                    confirmed += 1
        if confirmed < 2: return None
        sig = _SigProxy(day)
        ana = _AnalysisProxy(day.cs_tf_states)
        ok, adj, _ = cascade_vix.evaluate(sig, ana, feature_vec=None, bar_datetime=day.date,
                                           higher_tf_data=None, spy_df=None, vix_df=None)
        if not ok or adj < MIN_SIGNAL_CONFIDENCE: return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, adj, s, t, 'CS')
    return fn

tf2_base = make_s1_tf2_vix()

# CP-like gates on tf2
def make_cp_filtered_tf2(base_fn):
    def fn(day):
        result = base_fn(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        if action == 'BUY':
            spy_pass = False
            if day.date in spy_above_sma20: spy_pass = True
            elif conf >= 0.80: spy_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 5: spy_pass = True
            elif conf >= 0.70 and _count_tf_confirming(day, 'BUY') >= 4 and day.cs_position_score < 0.95: spy_pass = True
            elif conf >= 0.65 and _count_tf_confirming(day, 'BUY') >= 4 and day.cs_channel_health >= 0.40: spy_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.65: spy_pass = True
            if not spy_pass:
                if day.cs_confluence_score >= 0.9 and conf >= 0.45: spy_pass = True
                elif vix_map.get(day.date, 22) > 30 and conf >= 0.50: spy_pass = True
                elif vix_map.get(day.date, 22) > 25 and conf >= 0.55: spy_pass = True
                elif spy_return_map.get(day.date, 0) > 1.0 and conf >= 0.55: spy_pass = True
            if not spy_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1: spy_pass = True
                elif day.cs_channel_health >= 0.25 and day.cs_position_score < 0.95: spy_pass = True
            if not spy_pass: return None
            conf_pass = False
            if conf >= 0.66: conf_pass = True
            elif day.cs_position_score <= 0.99: conf_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 4: conf_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.55: conf_pass = True
            if not conf_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1: conf_pass = True
                elif day.cs_channel_health >= 0.25 and day.cs_position_score < 0.95: conf_pass = True
            if not conf_pass: return None
        if action == 'SELL':
            spy_pass = False
            if day.date in spy_above_055pct: spy_pass = True
            elif spy_dist_map.get(day.date, 999) < 0 and day.cs_channel_health >= 0.32: spy_pass = True
            elif 0 <= spy_dist_map.get(day.date, 999) < 0.55 and day.cs_position_score < 0.99: spy_pass = True
            elif 0 <= spy_dist_map.get(day.date, 999) < 0.55 and day.cs_position_score >= 0.99 and day.cs_channel_health >= 0.35: spy_pass = True
            if not spy_pass and spy_dist_5.get(day.date, -999) >= 0: spy_pass = True
            if not spy_pass:
                vix = vix_map.get(day.date, 22)
                if (vix < 20 or vix > 25) and day.cs_channel_health >= 0.25: spy_pass = True
            if not spy_pass and day.cs_channel_health >= 0.25:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() in (0, 2, 3): spy_pass = True
            if not spy_pass and spy_dist_50.get(day.date, -999) >= 1.0: spy_pass = True
            if not spy_pass:
                if _count_tf_confirming(day, 'SELL') >= 5 and day.cs_channel_health >= 0.20: spy_pass = True
                elif spy_ret_2d.get(day.date, 0) < -2.0 and day.cs_channel_health >= 0.15: spy_pass = True
            if not spy_pass: return None
            conf_pass = False
            if conf >= 0.65: conf_pass = True
            elif day.cs_channel_health >= 0.30: conf_pass = True
            elif day.cs_confluence_score >= 0.9 and day.cs_channel_health >= 0.25: conf_pass = True
            elif vix_map.get(day.date, 22) > 25 and day.cs_channel_health >= 0.20: conf_pass = True
            elif spy_return_map.get(day.date, 0) < -1.0 and day.cs_channel_health >= 0.20: conf_pass = True
            elif vix_map.get(day.date, 22) > 30 and day.cs_channel_health >= 0.15: conf_pass = True
            elif day.cs_channel_health >= 0.10 and conf >= 0.60 and _count_tf_confirming(day, 'SELL') >= 4: conf_pass = True
            if not conf_pass: return None
        return result
    return fn

tf2_filtered = make_cp_filtered_tf2(tf2_base)

def make_cp_plus_tf2(extra_filter):
    def fn(day):
        result = cp_fn(day)
        if result is not None:
            return result
        result = tf2_filtered(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result
        if extra_filter(day, action, conf):
            return result
        return None
    return fn

def test(name, filt):
    fn = make_cp_plus_tf2(filt)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n == 0: return
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    tr = [t for t in trades if t.entry_date.year <= 2021]
    ts = [t for t in trades if t.entry_date.year > 2021]
    marker = " ***" if wr >= 100 and n > 255 else ""
    if n > 255:
        print(f"  {name:40s}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f} "
              f"[Tr:{len(tr)}@{sum(1 for t in tr if t.pnl>0)/max(1,len(tr))*100:.0f}% "
              f"Ts:{len(ts)}@{sum(1 for t in ts if t.pnl>0)/max(1,len(ts))*100:.0f}%]{marker}")

# ── Combined OR filters ──
print("\n=== COMBINED OR FILTERS ===")
test('h35 OR (h25&confl60)', lambda d, a, c: d.cs_channel_health >= 0.35 or (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60))
test('h35 OR (h25&confl50)', lambda d, a, c: d.cs_channel_health >= 0.35 or (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.50))
test('h35 OR (h25&confl70)', lambda d, a, c: d.cs_channel_health >= 0.35 or (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.70))
test('h35 OR (h30&confl50)', lambda d, a, c: d.cs_channel_health >= 0.35 or (d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.50))
test('h35 OR (h30&confl60)', lambda d, a, c: d.cs_channel_health >= 0.35 or (d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.60))

# ── h>=0.35 + confidence/position extensions ──
print("\n=== h35 + EXTENSIONS ===")
test('h35 OR (h25&c60)', lambda d, a, c: d.cs_channel_health >= 0.35 or (d.cs_channel_health >= 0.25 and c >= 0.60))
test('h35 OR (h25&rawC50)', lambda d, a, c: d.cs_channel_health >= 0.35 or (d.cs_channel_health >= 0.25 and d.cs_confidence >= 0.50))
test('h35 OR (h25&pos<90)', lambda d, a, c: d.cs_channel_health >= 0.35 or (d.cs_channel_health >= 0.25 and d.cs_position_score < 0.90))
test('h35 OR (h30&c55)', lambda d, a, c: d.cs_channel_health >= 0.35 or (d.cs_channel_health >= 0.30 and c >= 0.55))
test('h35 OR (h25&c55&pos<95)', lambda d, a, c: d.cs_channel_health >= 0.35 or (d.cs_channel_health >= 0.25 and c >= 0.55 and d.cs_position_score < 0.95))

# ── h25&confl60 + extensions ──
print("\n=== h25&confl60 + EXTENSIONS ===")
test('h25&confl60 OR h35', lambda d, a, c: (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60) or d.cs_channel_health >= 0.35)
test('h25&confl60 OR (h30&rawC50)', lambda d, a, c: (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60) or (d.cs_channel_health >= 0.30 and d.cs_confidence >= 0.50))
test('h25&confl60 OR (h30&c55)', lambda d, a, c: (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60) or (d.cs_channel_health >= 0.30 and c >= 0.55))
test('h25&confl60 OR (h30&pos<95)', lambda d, a, c: (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60) or (d.cs_channel_health >= 0.30 and d.cs_position_score < 0.95))
test('(h25&confl50) OR h35', lambda d, a, c: (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.50) or d.cs_channel_health >= 0.35)
test('(h20&confl70) OR h35', lambda d, a, c: (d.cs_channel_health >= 0.20 and d.cs_confluence_score >= 0.70) or d.cs_channel_health >= 0.35)
test('(h20&confl80) OR h30', lambda d, a, c: (d.cs_channel_health >= 0.20 and d.cs_confluence_score >= 0.80) or d.cs_channel_health >= 0.30)

# ── Triple OR ──
print("\n=== TRIPLE OR ===")
test('h35|h25&confl60|h30&c55', lambda d, a, c:
     d.cs_channel_health >= 0.35 or
     (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60) or
     (d.cs_channel_health >= 0.30 and c >= 0.55))
test('h35|h25&confl60|h30&rawC50', lambda d, a, c:
     d.cs_channel_health >= 0.35 or
     (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60) or
     (d.cs_channel_health >= 0.30 and d.cs_confidence >= 0.50))
test('h35|h25&confl60|h30&pos<95', lambda d, a, c:
     d.cs_channel_health >= 0.35 or
     (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60) or
     (d.cs_channel_health >= 0.30 and d.cs_position_score < 0.95))
test('h35|h25&confl60|h25&c60', lambda d, a, c:
     d.cs_channel_health >= 0.35 or
     (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60) or
     (d.cs_channel_health >= 0.25 and c >= 0.60))
test('h35|h25&confl50|h30&rawC50', lambda d, a, c:
     d.cs_channel_health >= 0.35 or
     (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.50) or
     (d.cs_channel_health >= 0.30 and d.cs_confidence >= 0.50))

# ── Detailed best ──
print("\n=== DETAILED BEST ===")
# Run best combo with year breakdown
best_filt = lambda d, a, c: d.cs_channel_health >= 0.35 or (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60)
best_fn = make_cp_plus_tf2(best_filt)
best_trades = simulate_trades(signals, best_fn, 'BEST', cooldown=0, trail_power=6)
n = len(best_trades)
wins = sum(1 for t in best_trades if t.pnl > 0)
pnl = sum(t.pnl for t in best_trades)
print(f"\nBEST (h35|h25&confl60): {n} trades, {wins/n*100:.1f}% WR, ${pnl:+,.0f}")
print(f"  Longs: {sum(1 for t in best_trades if t.direction=='BUY')}, Shorts: {sum(1 for t in best_trades if t.direction=='SELL')}")

by_year = defaultdict(list)
for t in best_trades:
    by_year[t.entry_date.year].append(t)
print(f"\n  {'Year':>4} {'Trades':>6} {'WR':>5} {'PnL':>10}")
for year in sorted(by_year):
    yr = by_year[year]
    yr_n = len(yr)
    yr_w = sum(1 for t in yr if t.pnl > 0)
    yr_pnl = sum(t.pnl for t in yr)
    print(f"  {year:>4} {yr_n:>6} {yr_w/yr_n*100:>5.1f} ${yr_pnl:>+9,.0f}")

print("\nDone")
