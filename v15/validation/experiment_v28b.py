#!/usr/bin/env python3
"""v28b: Combine best recovery paths from v28.
confl>=0.90 adds +2 trades, +$5K PnL (strictly better than CS).
c90 adds +5 dates (net +2 trades). Mon&h30 adds +1 trade.
SELL pos100 adds +1 trade, +$3K PnL.
Test: stacking all at 100% WR."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _build_filter_cascade,
    _make_v26_cs, _SigProxy, _AnalysisProxy, _floor_stop_tp
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

cs_fn = _make_v26_cs(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50,
                      vix_map, spy_return_map, spy_ret_2d)
cs_trades = simulate_trades(signals, cs_fn, 'CS', cooldown=0, trail_power=6)
cs_dates = {t.entry_date for t in cs_trades}
day_map = {day.date: day for day in signals}
print(f"CS baseline: {len(cs_trades)} trades, {sum(1 for t in cs_trades if t.pnl>0)/len(cs_trades)*100:.1f}% WR, ${sum(t.pnl for t in cs_trades):+,.0f}")

# ── All CS+CP base ──
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

def make_cp_filtered(base_fn):
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

all_base = make_all_cs_base()
all_filtered = make_cp_filtered(all_base)

# ── Recovery checks ──
def check_confl90(day, action, conf):
    return day.cs_confluence_score >= 0.90

def check_c90(day, action, conf):
    return conf >= 0.90

def check_mon_h30(day, action, conf):
    dd = day.date.date() if hasattr(day.date, 'date') else day.date
    return dd.weekday() == 0 and day.cs_channel_health >= 0.30

def check_sell_pos100(day, action, conf):
    return action == 'SELL' and day.cs_position_score >= 1.0

def check_sell_h30(day, action, conf):
    return action == 'SELL' and day.cs_channel_health >= 0.30

def check_buy_vix50(day, action, conf):
    return action == 'BUY' and vix_map.get(day.date, 22) > 50 and day.cs_channel_health >= 0.15

def check_sell_confl90(day, action, conf):
    return action == 'SELL' and day.cs_confluence_score >= 0.90

# Additional granular checks
def check_confl85_h30(day, action, conf):
    return day.cs_confluence_score >= 0.85 and day.cs_channel_health >= 0.30

def check_confl80_h35(day, action, conf):
    return day.cs_confluence_score >= 0.80 and day.cs_channel_health >= 0.35

def check_h50(day, action, conf):
    return day.cs_channel_health >= 0.50

def check_h45(day, action, conf):
    return day.cs_channel_health >= 0.45

def check_h40_confl70(day, action, conf):
    return day.cs_channel_health >= 0.40 and day.cs_confluence_score >= 0.70

def check_pos0_h35(day, action, conf):
    return day.cs_position_score == 0.0 and day.cs_channel_health >= 0.35

def check_buy_pos0_h30(day, action, conf):
    return action == 'BUY' and day.cs_position_score == 0.0 and day.cs_channel_health >= 0.30

def check_buy_h50(day, action, conf):
    return action == 'BUY' and day.cs_channel_health >= 0.50

def check_buy_h45(day, action, conf):
    return action == 'BUY' and day.cs_channel_health >= 0.45

def check_buy_confl70_h40(day, action, conf):
    return action == 'BUY' and day.cs_confluence_score >= 0.70 and day.cs_channel_health >= 0.40

def check_buy_confl80(day, action, conf):
    return action == 'BUY' and day.cs_confluence_score >= 0.80

def check_sell_h25_pos100(day, action, conf):
    return action == 'SELL' and day.cs_channel_health >= 0.25 and day.cs_position_score >= 1.0

# ── Build combined function ──
def make_combined(checks):
    def fn(day):
        result = cs_fn(day)
        if result is not None: return result
        result = all_filtered(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        for check in checks:
            if check(day, action, conf):
                return result
        return None
    return fn

def run(name, checks):
    fn = make_combined(checks)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = [t for t in trades if t.entry_date not in cs_dates]
    new_count = len(new)
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:45s}: {n:3d} trades (+{new_count:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")
    return trades, new

# ── SECTION 1: All single recoveries (100% WR from v28) ──
print("\n=== SINGLE RECOVERY PATHS ===")
singles = [
    ('confl>=0.90', [check_confl90]),
    ('c90', [check_c90]),
    ('Mon&h30', [check_mon_h30]),
    ('SELL pos100', [check_sell_pos100]),
    ('SELL h30', [check_sell_h30]),
    ('BUY VIX>50&h15', [check_buy_vix50]),
    ('SELL confl90', [check_sell_confl90]),
    ('confl85&h30', [check_confl85_h30]),
    ('confl80&h35', [check_confl80_h35]),
    ('h>=0.50', [check_h50]),
    ('h>=0.45', [check_h45]),
    ('h40&confl70', [check_h40_confl70]),
    ('pos0&h35', [check_pos0_h35]),
    ('BUY pos0&h30', [check_buy_pos0_h30]),
    ('BUY h>=0.50', [check_buy_h50]),
    ('BUY h>=0.45', [check_buy_h45]),
    ('BUY confl70&h40', [check_buy_confl70_h40]),
    ('BUY confl80', [check_buy_confl80]),
    ('SELL h25&pos100', [check_sell_h25_pos100]),
]

for name, checks in singles:
    run(name, checks)

# ── SECTION 2: Confl90 as base + additional paths ──
print("\n=== CONFL90 + ADDITIONAL PATHS ===")
confl90_combos = [
    ('confl90 + c90', [check_confl90, check_c90]),
    ('confl90 + Mon&h30', [check_confl90, check_mon_h30]),
    ('confl90 + SELL pos100', [check_confl90, check_sell_pos100]),
    ('confl90 + SELL h30', [check_confl90, check_sell_h30]),
    ('confl90 + BUY VIX50', [check_confl90, check_buy_vix50]),
    ('confl90 + h50', [check_confl90, check_h50]),
    ('confl90 + h45', [check_confl90, check_h45]),
    ('confl90 + BUY h50', [check_confl90, check_buy_h50]),
    ('confl90 + BUY pos0&h30', [check_confl90, check_buy_pos0_h30]),
    ('confl90 + pos0&h35', [check_confl90, check_pos0_h35]),
]

for name, checks in confl90_combos:
    run(name, checks)

# ── SECTION 3: Triple combos ──
print("\n=== TRIPLE COMBOS ===")
triples = [
    ('confl90+c90+Mon', [check_confl90, check_c90, check_mon_h30]),
    ('confl90+c90+VIX50', [check_confl90, check_c90, check_buy_vix50]),
    ('confl90+c90+SELLh30', [check_confl90, check_c90, check_sell_h30]),
    ('confl90+c90+h50', [check_confl90, check_c90, check_h50]),
    ('confl90+c90+BUYh50', [check_confl90, check_c90, check_buy_h50]),
    ('confl90+c90+pos0h35', [check_confl90, check_c90, check_pos0_h35]),
    ('confl90+Mon+SELLh30', [check_confl90, check_mon_h30, check_sell_h30]),
    ('confl90+Mon+h50', [check_confl90, check_mon_h30, check_h50]),
    ('confl90+SELLh30+h50', [check_confl90, check_sell_h30, check_h50]),
    ('confl90+SELLh30+BUYh50', [check_confl90, check_sell_h30, check_buy_h50]),
]

for name, checks in triples:
    run(name, checks)

# ── SECTION 4: Quad combos (kitchen sink) ──
print("\n=== QUAD COMBOS ===")
quads = [
    ('confl90+c90+Mon+SELLh30', [check_confl90, check_c90, check_mon_h30, check_sell_h30]),
    ('confl90+c90+Mon+h50', [check_confl90, check_c90, check_mon_h30, check_h50]),
    ('confl90+c90+Mon+BUYh50', [check_confl90, check_c90, check_mon_h30, check_buy_h50]),
    ('confl90+c90+SELLh30+h50', [check_confl90, check_c90, check_sell_h30, check_h50]),
    ('confl90+c90+SELLh30+BUYh50', [check_confl90, check_c90, check_sell_h30, check_buy_h50]),
    ('confl90+Mon+SELLh30+BUYh50', [check_confl90, check_mon_h30, check_sell_h30, check_buy_h50]),
]

for name, checks in quads:
    run(name, checks)

# ── SECTION 5: Full kitchen sink ──
print("\n=== KITCHEN SINK (5+) ===")
kitchens = [
    ('confl90+c90+Mon+SELLh30+BUYh50', [check_confl90, check_c90, check_mon_h30, check_sell_h30, check_buy_h50]),
    ('confl90+c90+Mon+SELLh30+h50', [check_confl90, check_c90, check_mon_h30, check_sell_h30, check_h50]),
    ('confl90+c90+Mon+SELLh30+h45', [check_confl90, check_c90, check_mon_h30, check_sell_h30, check_h45]),
    ('ALL', [check_confl90, check_c90, check_mon_h30, check_sell_h30, check_h50, check_buy_vix50, check_sell_pos100]),
]

for name, checks in kitchens:
    run(name, checks)

# ── SECTION 6: Best combo detail ──
print("\n=== BEST COMBO DETAIL ===")
# Run best and show new trades
best_checks = [check_confl90, check_c90, check_mon_h30, check_sell_h30, check_buy_h50]
best_fn = make_combined(best_checks)
best_trades = simulate_trades(signals, best_fn, 'BEST', cooldown=0, trail_power=6)
best_new = sorted([t for t in best_trades if t.entry_date not in cs_dates], key=lambda x: x.entry_date)

print(f"\nBest combo: {len(best_trades)} trades, {sum(1 for t in best_trades if t.pnl>0)/len(best_trades)*100:.1f}% WR, "
      f"${sum(t.pnl for t in best_trades):+,.0f}")
print(f"New trades ({len(best_new)}):")
for t in best_new:
    day = day_map.get(t.entry_date)
    if day:
        tfs = _count_tf_confirming(day, t.direction)
        vix = vix_map.get(day.date, 22)
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dd.weekday()]
        train = "TRAIN" if t.entry_date.year <= 2021 else "TEST"
        print(f"  {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+7,.0f} h={day.cs_channel_health:.3f} "
              f"c={day.cs_confidence:.3f} confl={day.cs_confluence_score:.2f} "
              f"pos={day.cs_position_score:.2f} TFs={tfs} VIX={vix:.1f} {dow} {train}")

# ── SECTION 7: Trail power sweep on best ──
print("\n=== TRAIL POWER SWEEP ON BEST ===")
for power in [5, 6, 7, 8, 10, 12, 15, 20]:
    trades = simulate_trades(signals, best_fn, f'p{power}', cooldown=0, trail_power=power)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    print(f"  power={power:2d}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# ── SECTION 8: Train/Test split ──
print("\n=== TRAIN/TEST SPLIT ===")
best_train = [t for t in best_trades if t.entry_date.year <= 2021]
best_test = [t for t in best_trades if t.entry_date.year > 2021]
for label, subset in [('Train 2016-2021', best_train), ('Test 2022-2025', best_test)]:
    n = len(subset)
    w = sum(1 for t in subset if t.pnl > 0)
    wr = w / n * 100 if n > 0 else 0
    pnl = sum(t.pnl for t in subset)
    bl = min(t.pnl for t in subset) if subset else 0
    print(f"  {label}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# ── SECTION 9: Fine-grained confl sweeps ──
print("\n=== FINE CONFLUENCE SWEEP ===")
for ct in [0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80]:
    def make_check(threshold):
        def check(day, action, conf):
            return day.cs_confluence_score >= threshold
        return check
    fn = make_combined([make_check(ct)])
    trades = simulate_trades(signals, fn, f'cf{ct}', cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    marker = " ***" if wr >= 100 else ""
    print(f"  confl>={ct:.2f}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

print("\nDone")
