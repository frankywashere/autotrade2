#!/usr/bin/env python3
"""Quick test: CN = CM + longConf[confl90&c55] + shConf[h10&c60&TF4]"""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming,
    simulate_trades, _make_s1_tf3_vix_combo, _build_filter_cascade,
    _make_v20_grand
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

base_fn = _make_s1_tf3_vix_combo(cascade_vix)

# CM baseline
cm_fn = _make_v20_grand(cascade_vix, spy_above_sma20, spy_above_055pct,
                         spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map)
cm_trades = simulate_trades(signals, cm_fn, 'CM', cooldown=0, trail_power=6)
n = len(cm_trades)
print(f"CM: {n} trades, {sum(1 for t in cm_trades if t.pnl>0)/n*100:.1f}% WR, ${sum(t.pnl for t in cm_trades):+,.0f}")

# CN = CM + longConf[confl90&c55] + shConf[h10&c60&TF4]
def cn_fn(day):
    result = base_fn(day)
    if result is None: return None
    action, conf, s, t, src = result
    if action == 'BUY':
        if day.date in spy_above_sma20: pass
        elif conf >= 0.80: pass
        elif _count_tf_confirming(day, 'BUY') >= 5: pass
        elif conf >= 0.70 and _count_tf_confirming(day, 'BUY') >= 4 and day.cs_position_score < 0.95: pass
        elif conf >= 0.65 and _count_tf_confirming(day, 'BUY') >= 4 and day.cs_channel_health >= 0.40: pass
        elif day.cs_confluence_score >= 0.9 and conf >= 0.65: pass
        else: return None
        if conf >= 0.66: pass
        elif day.cs_position_score <= 0.99: pass
        elif _count_tf_confirming(day, 'BUY') >= 4: pass
        elif day.cs_confluence_score >= 0.9 and conf >= 0.55: pass  # NEW: longConf
        else: return None
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
            if dd.weekday() in (0, 3): spy_pass = True
        if not spy_pass and spy_dist_50.get(day.date, -999) >= 1.0: spy_pass = True
        if not spy_pass: return None
        if conf >= 0.65: pass
        elif day.cs_channel_health >= 0.30: pass
        elif day.cs_confluence_score >= 0.9 and day.cs_channel_health >= 0.25: pass
        elif vix_map.get(day.date, 22) > 25 and day.cs_channel_health >= 0.20: pass
        elif spy_return_map.get(day.date, 0) < -1.0 and day.cs_channel_health >= 0.20: pass
        elif vix_map.get(day.date, 22) > 30 and day.cs_channel_health >= 0.15: pass
        elif day.cs_channel_health >= 0.10 and conf >= 0.60 and _count_tf_confirming(day, 'SELL') >= 4: pass  # NEW: shConf
        else: return None
    return result

cn_trades = simulate_trades(signals, cn_fn, 'CN', cooldown=0, trail_power=6)
n = len(cn_trades)
wins = sum(1 for t in cn_trades if t.pnl > 0)
pnl = sum(t.pnl for t in cn_trades)
bl = min(t.pnl for t in cn_trades) if cn_trades else 0
tr = [t for t in cn_trades if t.entry_date.year <= 2021]
ts = [t for t in cn_trades if t.entry_date.year > 2021]
print(f"CN: {n} trades, {wins/n*100:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")
print(f"  Train: {len(tr)} @ {sum(1 for t in tr if t.pnl>0)/len(tr)*100:.0f}%  Test: {len(ts)} @ {sum(1 for t in ts if t.pnl>0)/len(ts)*100:.0f}%")
print(f"  Longs: {sum(1 for t in cn_trades if t.direction=='BUY')}, Shorts: {sum(1 for t in cn_trades if t.direction=='SELL')}")
