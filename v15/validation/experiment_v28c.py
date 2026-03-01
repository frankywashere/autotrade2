#!/usr/bin/env python3
"""v28c: Final combo optimization. Test BUY confl80 stacking + profile all new trades.
Best from v28b: confl90+BUY_VIX50 = 342 trades, $588K, 100% WR."""
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

def run(name, checks, detail=False):
    fn = make_combined(checks)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = sorted([t for t in trades if t.entry_date not in cs_dates], key=lambda x: x.entry_date)
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

# Recovery checks
def check_confl90(d, a, c): return d.cs_confluence_score >= 0.90
def check_confl88(d, a, c): return d.cs_confluence_score >= 0.88
def check_buy_confl80(d, a, c): return a == 'BUY' and d.cs_confluence_score >= 0.80
def check_buy_confl85(d, a, c): return a == 'BUY' and d.cs_confluence_score >= 0.85
def check_buy_confl75(d, a, c): return a == 'BUY' and d.cs_confluence_score >= 0.75
def check_buy_confl70(d, a, c): return a == 'BUY' and d.cs_confluence_score >= 0.70
def check_buy_vix50(d, a, c): return a == 'BUY' and vix_map.get(d.date, 22) > 50 and d.cs_channel_health >= 0.15
def check_buy_vix30(d, a, c): return a == 'BUY' and vix_map.get(d.date, 22) > 30 and d.cs_channel_health >= 0.15
def check_c90(d, a, c): return c >= 0.90
def check_c85(d, a, c): return c >= 0.85
def check_mon_h30(d, a, c):
    dd = d.date.date() if hasattr(d.date, 'date') else d.date
    return dd.weekday() == 0 and d.cs_channel_health >= 0.30
def check_sell_h30(d, a, c): return a == 'SELL' and d.cs_channel_health >= 0.30
def check_sell_confl90(d, a, c): return a == 'SELL' and d.cs_confluence_score >= 0.90

# ── SECTION 1: Profile BUY confl80 new trades ──
print("\n=== BUY CONFL80 PROFILING ===")
run('BUY confl>=0.80', [check_buy_confl80], detail=True)
run('BUY confl>=0.85', [check_buy_confl85], detail=True)
run('BUY confl>=0.75', [check_buy_confl75], detail=True)
run('BUY confl>=0.70', [check_buy_confl70], detail=True)

# ── SECTION 2: BUY confl80 + SELL confl90 combos ──
print("\n=== BUY CONFL80 + SELL CONFL90 COMBOS ===")
run('SELL confl90 + BUY confl80', [check_sell_confl90, check_buy_confl80], detail=True)
run('SELL confl90 + BUY confl85', [check_sell_confl90, check_buy_confl85], detail=True)
run('SELL confl90 + BUY confl75', [check_sell_confl90, check_buy_confl75], detail=True)
run('confl90 + BUY confl80', [check_confl90, check_buy_confl80], detail=True)

# ── SECTION 3: Best PnL combo + BUY confl80 ──
print("\n=== CONFL90 + VIX50 + BUY CONFL80 ===")
run('confl90+VIX50+BUYconfl80', [check_confl90, check_buy_vix50, check_buy_confl80], detail=True)
run('confl90+VIX50+BUYconfl85', [check_confl90, check_buy_vix50, check_buy_confl85], detail=True)

# ── SECTION 4: Max trade combos with BUY confl80 ──
print("\n=== MAX TRADE COMBOS ===")
run('confl90+c90+BUYconfl80+Mon', [check_confl90, check_c90, check_buy_confl80, check_mon_h30])
run('confl90+c90+BUYconfl80+SELLh30', [check_confl90, check_c90, check_buy_confl80, check_sell_h30])
run('confl90+c90+BUYconfl80+Mon+SELLh30', [check_confl90, check_c90, check_buy_confl80, check_mon_h30, check_sell_h30])
run('confl90+BUYconfl80+Mon+SELLh30', [check_confl90, check_buy_confl80, check_mon_h30, check_sell_h30])
run('confl90+BUYconfl80+VIX50+Mon+SELLh30', [check_confl90, check_buy_confl80, check_buy_vix50, check_mon_h30, check_sell_h30])

# ── SECTION 5: Simplest high-value combos (for CT definition) ──
print("\n=== CANDIDATE CT DEFINITIONS ===")
print("  --- PnL-maximizing ---")
run('CT-A: confl90+VIX50', [check_confl90, check_buy_vix50], detail=True)
print("  --- Balanced ---")
run('CT-B: confl90+BUYconfl80', [check_confl90, check_buy_confl80], detail=True)
print("  --- Trade-maximizing ---")
run('CT-C: confl90+c90+Mon+SELLh30', [check_confl90, check_c90, check_mon_h30, check_sell_h30])

# ── SECTION 6: Even more BUY recovery ──
print("\n=== BUY RECOVERY DEEP DIVE ===")
# What about BUY with high health + various confluence levels?
for h in [0.55, 0.50, 0.45, 0.40, 0.35]:
    def make_check(hh):
        return lambda d, a, c: a == 'BUY' and d.cs_channel_health >= hh
    run(f'BUY h>={h}', [check_confl90, make_check(h)])

# BUY with pos near 0 (oversold)
for pos_max in [0.20, 0.30, 0.40, 0.50]:
    def make_check(pp):
        return lambda d, a, c: a == 'BUY' and d.cs_position_score <= pp and d.cs_channel_health >= 0.30
    run(f'BUY pos<={pos_max}&h30', [check_confl90, make_check(pos_max)])

# ── SECTION 7: VIX>30 BUY recovery ──
print("\n=== VIX RECOVERY ===")
run('confl90 + BUY VIX>30&h15', [check_confl90, check_buy_vix30], detail=True)
run('confl90 + BUY VIX>50&h15', [check_confl90, check_buy_vix50], detail=True)

# ── SECTION 8: Confl88 boundary test ──
print("\n=== CONFL BOUNDARY (88 vs 90) ===")
run('confl88', [check_confl88], detail=True)
run('confl88+BUYconfl80', [check_confl88, check_buy_confl80], detail=True)
run('confl88+VIX50', [check_confl88, check_buy_vix50], detail=True)

print("\nDone")
