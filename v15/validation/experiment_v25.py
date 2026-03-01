#!/usr/bin/env python3
"""v25: Profile remaining trades after CQ (311). Test TF1 expansion and
further relaxation of the TF2 triple OR filter."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _make_s1_tf3_vix_combo, _build_filter_cascade,
    _make_v23_cp, _make_v24_cq, _SigProxy, _AnalysisProxy, _floor_stop_tp
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
spy_ret_3d = {}
for i in range(3, len(spy_close)):
    spy_ret_3d[spy_daily.index[i]] = (spy_close[i]-spy_close[i-3])/spy_close[i-3]*100

# Build CQ baseline
cq_fn = _make_v24_cq(cascade_vix, spy_above_sma20, spy_above_055pct,
                       spy_dist_map, spy_dist_5, spy_dist_50,
                       vix_map, spy_return_map, spy_ret_2d)
cq_trades = simulate_trades(signals, cq_fn, 'CQ', cooldown=0, trail_power=6)
print(f"CQ: {len(cq_trades)} trades, {sum(1 for t in cq_trades if t.pnl>0)/len(cq_trades)*100:.1f}% WR, ${sum(t.pnl for t in cq_trades):+,.0f}")

# Build raw tf1 and tf2 bases (for comparison)
def make_s1_tfN_vix(min_tfs=3):
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

# CP-like gate applied to any base
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

# ── Profile remaining tf2 trades not in CQ ──
print("\n=== TF2 TRADES FILTERED BY CQ ===")
tf2_base = make_s1_tfN_vix(2)
tf2_filtered = make_cp_filtered(tf2_base)
tf2_trades = simulate_trades(signals, tf2_filtered, 'tf2_filt', cooldown=0, trail_power=6)
cq_dates = {t.entry_date for t in cq_trades}
tf2_not_in_cq = [t for t in tf2_trades if t.entry_date not in cq_dates]
print(f"TF2 filtered: {len(tf2_trades)} total, {len(tf2_not_in_cq)} not in CQ")
if tf2_not_in_cq:
    print(f"  {sum(1 for t in tf2_not_in_cq if t.pnl>0)}W / {sum(1 for t in tf2_not_in_cq if t.pnl<=0)}L")
    day_map = {day.date: day for day in signals}
    print(f"\n  {'Date':12} {'Dir':5} {'PnL':>8} {'Conf':>5} {'Health':>6} {'Pos':>5} "
          f"{'Confl':>5} {'TFs':>3} {'SPY20':>6} {'VIX':>5} {'SRet':>6} {'DOW':>4} {'Train':>5}")
    for t in sorted(tf2_not_in_cq, key=lambda x: x.entry_date):
        day = day_map.get(t.entry_date)
        if not day: continue
        result = tf2_base(day)
        if result is None: continue
        action, conf, s_pct, t_pct, src = result
        tfs = _count_tf_confirming(day, action)
        spy_d20 = spy_dist_map.get(day.date, 999)
        vix = vix_map.get(day.date, 22)
        sret = spy_return_map.get(day.date, 0)
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dd.weekday()]
        train = "TRAIN" if t.entry_date.year <= 2021 else "TEST"
        print(f"  {str(day.date)[:10]:12} {action:5} ${t.pnl:>+7,.0f} {conf:5.3f} {day.cs_channel_health:6.3f} "
              f"{day.cs_position_score:5.2f} {day.cs_confluence_score:5.2f} {tfs:3d} {spy_d20:+6.2f} {vix:5.1f} {sret:+6.2f} {dow_name:>4} {train:>5}")

# ── Test TF1 base ──
print("\n=== TF1 BASE (RAW) ===")
tf1_base = make_s1_tfN_vix(1)
tf1_trades_raw = simulate_trades(signals, tf1_base, 'tf1_raw', cooldown=0, trail_power=6)
print(f"TF1 raw: {len(tf1_trades_raw)} trades, {sum(1 for t in tf1_trades_raw if t.pnl>0)/max(1,len(tf1_trades_raw))*100:.1f}% WR, ${sum(t.pnl for t in tf1_trades_raw):+,.0f}")

tf1_filtered = make_cp_filtered(tf1_base)
tf1_trades_filt = simulate_trades(signals, tf1_filtered, 'tf1_filt', cooldown=0, trail_power=6)
print(f"TF1 filtered: {len(tf1_trades_filt)} trades, {sum(1 for t in tf1_trades_filt if t.pnl>0)/max(1,len(tf1_trades_filt))*100:.1f}% WR, ${sum(t.pnl for t in tf1_trades_filt):+,.0f}")

tf1_not_in_cq = [t for t in tf1_trades_filt if t.entry_date not in cq_dates]
print(f"TF1 new (not in CQ): {len(tf1_not_in_cq)} trades ({sum(1 for t in tf1_not_in_cq if t.pnl>0)}W/{sum(1 for t in tf1_not_in_cq if t.pnl<=0)}L)")

if tf1_not_in_cq:
    print(f"\n  {'Date':12} {'Dir':5} {'PnL':>8} {'Conf':>5} {'Health':>6} {'Pos':>5} "
          f"{'Confl':>5} {'TFs':>3} {'SPY20':>6} {'VIX':>5} {'SRet':>6} {'DOW':>4} {'Train':>5}")
    for t in sorted(tf1_not_in_cq, key=lambda x: x.entry_date):
        day = day_map.get(t.entry_date)
        if not day: continue
        result = tf1_base(day)
        if result is None: continue
        action, conf, s_pct, t_pct, src = result
        tfs = _count_tf_confirming(day, action)
        spy_d20 = spy_dist_map.get(day.date, 999)
        vix = vix_map.get(day.date, 22)
        sret = spy_return_map.get(day.date, 0)
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dd.weekday()]
        train = "TRAIN" if t.entry_date.year <= 2021 else "TEST"
        print(f"  {str(day.date)[:10]:12} {action:5} ${t.pnl:>+7,.0f} {conf:5.3f} {day.cs_channel_health:6.3f} "
              f"{day.cs_position_score:5.2f} {day.cs_confluence_score:5.2f} {tfs:3d} {spy_d20:+6.2f} {vix:5.1f} {sret:+6.2f} {dow_name:>4} {train:>5}")

# ── Test CQ + TF1 with various filters ──
print("\n=== CQ + TF1 EXPANSION TESTS ===")
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

tests = [
    ('h50', lambda d, a, c: d.cs_channel_health >= 0.50),
    ('h45', lambda d, a, c: d.cs_channel_health >= 0.45),
    ('h40', lambda d, a, c: d.cs_channel_health >= 0.40),
    ('h35', lambda d, a, c: d.cs_channel_health >= 0.35),
    ('h35&confl70', lambda d, a, c: d.cs_channel_health >= 0.35 and d.cs_confluence_score >= 0.70),
    ('h35&c65', lambda d, a, c: d.cs_channel_health >= 0.35 and c >= 0.65),
    ('h40&confl60', lambda d, a, c: d.cs_channel_health >= 0.40 and d.cs_confluence_score >= 0.60),
    ('h40&c60', lambda d, a, c: d.cs_channel_health >= 0.40 and c >= 0.60),
    ('h45&confl50', lambda d, a, c: d.cs_channel_health >= 0.45 and d.cs_confluence_score >= 0.50),
    ('h50|h40&confl70', lambda d, a, c: d.cs_channel_health >= 0.50 or (d.cs_channel_health >= 0.40 and d.cs_confluence_score >= 0.70)),
    ('h50|h40&c65', lambda d, a, c: d.cs_channel_health >= 0.50 or (d.cs_channel_health >= 0.40 and c >= 0.65)),
    ('h50|h35&confl80', lambda d, a, c: d.cs_channel_health >= 0.50 or (d.cs_channel_health >= 0.35 and d.cs_confluence_score >= 0.80)),
    ('h45|h35&confl70', lambda d, a, c: d.cs_channel_health >= 0.45 or (d.cs_channel_health >= 0.35 and d.cs_confluence_score >= 0.70)),
    ('h45|h35&c65', lambda d, a, c: d.cs_channel_health >= 0.45 or (d.cs_channel_health >= 0.35 and c >= 0.65)),
    ('h45|h35&confl60', lambda d, a, c: d.cs_channel_health >= 0.45 or (d.cs_channel_health >= 0.35 and d.cs_confluence_score >= 0.60)),
    ('h40|h30&confl70', lambda d, a, c: d.cs_channel_health >= 0.40 or (d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.70)),
    ('h40|h30&c65', lambda d, a, c: d.cs_channel_health >= 0.40 or (d.cs_channel_health >= 0.30 and c >= 0.65)),
]

for name, filt in tests:
    fn = make_cq_plus_tf1(filt)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 311: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    tr = [t for t in trades if t.entry_date.year <= 2021]
    ts = [t for t in trades if t.entry_date.year > 2021]
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f} "
          f"[Tr:{len(tr)}@{sum(1 for t in tr if t.pnl>0)/max(1,len(tr))*100:.0f}% "
          f"Ts:{len(ts)}@{sum(1 for t in ts if t.pnl>0)/max(1,len(ts))*100:.0f}%]{marker}")

# ── Also test further TF2 relaxation (broader triple OR) ──
print("\n=== BROADER TF2 FILTER ON CQ ===")
# CQ currently uses h35|h25&confl60|h25&c60, test broader versions
cp_fn2 = _make_v23_cp(cascade_vix, spy_above_sma20, spy_above_055pct,
                       spy_dist_map, spy_dist_5, spy_dist_50,
                       vix_map, spy_return_map, spy_ret_2d)
tf2_base2 = make_s1_tfN_vix(2)
tf2_filtered2 = make_cp_filtered(tf2_base2)

def make_cp_plus_tf2_broad(extra_filter):
    def fn(day):
        result = cp_fn2(day)
        if result is not None: return result
        result = tf2_filtered2(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        if extra_filter(day, action, conf): return result
        return None
    return fn

broad_tests = [
    ('h30|h20&confl70|h20&c65', lambda d, a, c:
     d.cs_channel_health >= 0.30 or
     (d.cs_channel_health >= 0.20 and d.cs_confluence_score >= 0.70) or
     (d.cs_channel_health >= 0.20 and c >= 0.65)),
    ('h35|h20&confl70|h20&c65', lambda d, a, c:
     d.cs_channel_health >= 0.35 or
     (d.cs_channel_health >= 0.20 and d.cs_confluence_score >= 0.70) or
     (d.cs_channel_health >= 0.20 and c >= 0.65)),
    ('h35|h25&confl50|h25&c55', lambda d, a, c:
     d.cs_channel_health >= 0.35 or
     (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.50) or
     (d.cs_channel_health >= 0.25 and c >= 0.55)),
    ('h30|h25&confl60|h25&c60', lambda d, a, c:
     d.cs_channel_health >= 0.30 or
     (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60) or
     (d.cs_channel_health >= 0.25 and c >= 0.60)),
    ('h35|h25&confl60|h25&c55', lambda d, a, c:
     d.cs_channel_health >= 0.35 or
     (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60) or
     (d.cs_channel_health >= 0.25 and c >= 0.55)),
    ('h35|h20&confl80|h20&c70', lambda d, a, c:
     d.cs_channel_health >= 0.35 or
     (d.cs_channel_health >= 0.20 and d.cs_confluence_score >= 0.80) or
     (d.cs_channel_health >= 0.20 and c >= 0.70)),
    ('h35|h25&confl60|h25&c60|h20&confl80', lambda d, a, c:
     d.cs_channel_health >= 0.35 or
     (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60) or
     (d.cs_channel_health >= 0.25 and c >= 0.60) or
     (d.cs_channel_health >= 0.20 and d.cs_confluence_score >= 0.80)),
    ('h35|h25&confl60|h25&c60|TF4&h20', lambda d, a, c:
     d.cs_channel_health >= 0.35 or
     (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60) or
     (d.cs_channel_health >= 0.25 and c >= 0.60) or
     (_count_tf_confirming(d, a) >= 4 and d.cs_channel_health >= 0.20)),
    ('h35|h25&confl60|h25&c60|pos<80&h20', lambda d, a, c:
     d.cs_channel_health >= 0.35 or
     (d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60) or
     (d.cs_channel_health >= 0.25 and c >= 0.60) or
     (d.cs_position_score < 0.80 and d.cs_channel_health >= 0.20)),
]

for name, filt in broad_tests:
    fn = make_cp_plus_tf2_broad(filt)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 311: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    tr = [t for t in trades if t.entry_date.year <= 2021]
    ts = [t for t in trades if t.entry_date.year > 2021]
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:45s}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f} "
          f"[Tr:{len(tr)}@{sum(1 for t in tr if t.pnl>0)/max(1,len(tr))*100:.0f}% "
          f"Ts:{len(ts)}@{sum(1 for t in ts if t.pnl>0)/max(1,len(ts))*100:.0f}%]{marker}")

# ── Trail power sweep on CQ ──
print("\n=== TRAIL POWER SWEEP ON CQ ===")
for power in [4, 5, 6, 7, 8, 10, 12]:
    cq_fn_p = _make_v24_cq(cascade_vix, spy_above_sma20, spy_above_055pct,
                             spy_dist_map, spy_dist_5, spy_dist_50,
                             vix_map, spy_return_map, spy_ret_2d)
    trades = simulate_trades(signals, cq_fn_p, f'CQ_p{power}', cooldown=0, trail_power=power)
    n = len(trades)
    if n == 0: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    print(f"  power={power:2d}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

print("\nDone")
