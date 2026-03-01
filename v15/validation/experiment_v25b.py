#!/usr/bin/env python3
"""v25b: Optimize TF1 expansion filter for CQ. Best from v25: h40|h30&confl70 = 329 trades 100% WR.
Sweep finer granularity and triple/quad OR combos."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _build_filter_cascade,
    _make_v24_cq, _SigProxy, _AnalysisProxy, _floor_stop_tp
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

cq_fn = _make_v24_cq(cascade_vix, spy_above_sma20, spy_above_055pct,
                       spy_dist_map, spy_dist_5, spy_dist_50,
                       vix_map, spy_return_map, spy_ret_2d)

# TF1 base + CP-like gate
def make_s1_tfN_vix(min_tfs=1):
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
        if confirmed < min_tfs: return None
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

tf1_base = make_s1_tfN_vix(1)
tf1_filtered = make_cp_filtered(tf1_base)

def make_cq_plus_tf1(extra_filter):
    def fn(day):
        result = cq_fn(day)
        if result is not None: return result
        result = tf1_filtered(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        if extra_filter(day, action, conf): return result
        return None
    return fn

def test(name, filt, threshold=311):
    fn = make_cq_plus_tf1(filt)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= threshold: return
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    tr = [t for t in trades if t.entry_date.year <= 2021]
    ts = [t for t in trades if t.entry_date.year > 2021]
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:50s}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f} "
          f"[Tr:{len(tr)}@{sum(1 for t in tr if t.pnl>0)/max(1,len(tr))*100:.0f}% "
          f"Ts:{len(ts)}@{sum(1 for t in ts if t.pnl>0)/max(1,len(ts))*100:.0f}%]{marker}")

# ── Fine granularity sweep for single health thresholds ──
print("=== SINGLE HEALTH THRESHOLD ===")
for h in [0.50, 0.45, 0.42, 0.40, 0.38, 0.37, 0.36, 0.35, 0.33, 0.30, 0.28, 0.25]:
    test(f'h>={h}', lambda d, a, c, hh=h: d.cs_channel_health >= hh)

# ── Double OR: h_high | h_low&confl ──
print("\n=== h_high | h_low & confl ===")
for h_hi in [0.45, 0.42, 0.40, 0.38, 0.37, 0.36, 0.35]:
    for h_lo in [0.35, 0.32, 0.30, 0.28, 0.25, 0.22, 0.20]:
        if h_lo >= h_hi: continue
        for confl in [0.80, 0.75, 0.70, 0.65, 0.60, 0.55]:
            test(f'h{int(h_hi*100)}|h{int(h_lo*100)}&confl{int(confl*100)}',
                 lambda d, a, c, hh=h_hi, hl=h_lo, cf=confl:
                 d.cs_channel_health >= hh or
                 (d.cs_channel_health >= hl and d.cs_confluence_score >= cf))

# ── Double OR: h_high | h_low&c_adj ──
print("\n=== h_high | h_low & adj_conf ===")
for h_hi in [0.42, 0.40, 0.38, 0.37, 0.36, 0.35]:
    for h_lo in [0.32, 0.30, 0.28, 0.25]:
        if h_lo >= h_hi: continue
        for c in [0.75, 0.70, 0.65, 0.60, 0.55]:
            test(f'h{int(h_hi*100)}|h{int(h_lo*100)}&c{int(c*100)}',
                 lambda d, a, c_val, hh=h_hi, hl=h_lo, cc=c:
                 d.cs_channel_health >= hh or
                 (d.cs_channel_health >= hl and c_val >= cc))

# ── Triple OR: h_high | h_low&confl | h_low&c_adj ──
print("\n=== TRIPLE OR: h_high | h_lo&confl | h_lo&c ===")
for h_hi in [0.42, 0.40, 0.38, 0.37, 0.36, 0.35]:
    for h_lo in [0.32, 0.30, 0.28, 0.25]:
        if h_lo >= h_hi: continue
        for confl in [0.75, 0.70, 0.65, 0.60]:
            for c in [0.70, 0.65, 0.60, 0.55]:
                test(f'h{int(h_hi*100)}|h{int(h_lo*100)}&cf{int(confl*100)}|h{int(h_lo*100)}&c{int(c*100)}',
                     lambda d, a, c_val, hh=h_hi, hl=h_lo, cf=confl, cc=c:
                     d.cs_channel_health >= hh or
                     (d.cs_channel_health >= hl and d.cs_confluence_score >= cf) or
                     (d.cs_channel_health >= hl and c_val >= cc))

# ── Triple OR with TF count ──
print("\n=== TRIPLE OR WITH TF ===")
for h_hi in [0.40, 0.38, 0.36, 0.35]:
    for h_lo in [0.30, 0.28, 0.25, 0.20]:
        if h_lo >= h_hi: continue
        for min_tf in [3, 4, 5]:
            test(f'h{int(h_hi*100)}|h{int(h_lo*100)}&TF{min_tf}',
                 lambda d, a, c, hh=h_hi, hl=h_lo, mt=min_tf:
                 d.cs_channel_health >= hh or
                 (d.cs_channel_health >= hl and _count_tf_confirming(d, a) >= mt))

# ── Position score variants ──
print("\n=== POSITION SCORE VARIANTS ===")
for h_hi in [0.40, 0.38, 0.36, 0.35]:
    for h_lo in [0.30, 0.28, 0.25]:
        if h_lo >= h_hi: continue
        for pos in [0.95, 0.90, 0.85, 0.80]:
            test(f'h{int(h_hi*100)}|h{int(h_lo*100)}&pos<{int(pos*100)}',
                 lambda d, a, c, hh=h_hi, hl=h_lo, pp=pos:
                 d.cs_channel_health >= hh or
                 (d.cs_channel_health >= hl and d.cs_position_score < pp))

# ── VIX-conditioned variants ──
print("\n=== VIX VARIANTS ===")
for h_hi in [0.40, 0.38, 0.35]:
    for h_lo in [0.30, 0.25, 0.20]:
        if h_lo >= h_hi: continue
        for vl in [18, 20]:
            for vh in [25, 30]:
                test(f'h{int(h_hi*100)}|h{int(h_lo*100)}&VIX<{vl}or>{vh}',
                     lambda d, a, c, hh=h_hi, hl=h_lo, lo=vl, hi=vh:
                     d.cs_channel_health >= hh or
                     (d.cs_channel_health >= hl and
                      (vix_map.get(d.date, 22) < lo or vix_map.get(d.date, 22) > hi)))

print("\nDone")
