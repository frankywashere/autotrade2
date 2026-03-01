#!/usr/bin/env python3
"""v26: Profile remaining trades after CR (331). Test TF0 (no TF requirement),
profile the loss trades at each TF level, and search for any remaining safe recovery."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _build_filter_cascade,
    _make_v25_cr, _SigProxy, _AnalysisProxy, _floor_stop_tp
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

cr_fn = _make_v25_cr(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50,
                      vix_map, spy_return_map, spy_ret_2d)
cr_trades = simulate_trades(signals, cr_fn, 'CR', cooldown=0, trail_power=6)
print(f"CR: {len(cr_trades)} trades, {sum(1 for t in cr_trades if t.pnl>0)/len(cr_trades)*100:.1f}% WR, ${sum(t.pnl for t in cr_trades):+,.0f}")

day_map = {day.date: day for day in signals}
cr_dates = {t.entry_date for t in cr_trades}

# ── TF0 base: no TF confirmation, just CS signal + VIX cascade ──
print("\n=== TF0 BASE (NO TF REQUIREMENT) ===")
def make_tf0_base():
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

tf0_base = make_tf0_base()
tf0_filtered = make_cp_filtered(tf0_base)
tf0_trades = simulate_trades(signals, tf0_filtered, 'tf0_filt', cooldown=0, trail_power=6)
tf0_not_in_cr = [t for t in tf0_trades if t.entry_date not in cr_dates]
print(f"TF0 filtered: {len(tf0_trades)} trades ({sum(1 for t in tf0_trades if t.pnl>0)}W/{sum(1 for t in tf0_trades if t.pnl<=0)}L)")
print(f"TF0 new (not in CR): {len(tf0_not_in_cr)} trades ({sum(1 for t in tf0_not_in_cr if t.pnl>0)}W/{sum(1 for t in tf0_not_in_cr if t.pnl<=0)}L)")

if tf0_not_in_cr:
    print(f"\n  {'Date':12} {'Dir':5} {'PnL':>8} {'Conf':>5} {'Health':>6} {'Pos':>5} "
          f"{'Confl':>5} {'TFs':>3} {'SPY20':>6} {'VIX':>5} {'SRet':>6} {'DOW':>4} {'Train':>5}")
    for t in sorted(tf0_not_in_cr, key=lambda x: x.entry_date):
        day = day_map.get(t.entry_date)
        if not day: continue
        tfs = _count_tf_confirming(day, t.direction)
        spy_d20 = spy_dist_map.get(day.date, 999)
        vix = vix_map.get(day.date, 22)
        sret = spy_return_map.get(day.date, 0)
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dd.weekday()]
        train = "TRAIN" if t.entry_date.year <= 2021 else "TEST"
        print(f"  {str(day.date)[:10]:12} {t.direction:5} ${t.pnl:>+7,.0f} {day.cs_confidence:5.3f} {day.cs_channel_health:6.3f} "
              f"{day.cs_position_score:5.2f} {day.cs_confluence_score:5.2f} {tfs:3d} {spy_d20:+6.2f} {vix:5.1f} {sret:+6.2f} {dow_name:>4} {train:>5}")

# ── Test CR + TF0 with filters ──
print("\n=== CR + TF0 EXPANSION TESTS ===")
def make_cr_plus_tf0(extra_filter):
    def fn(day):
        result = cr_fn(day)
        if result is not None: return result
        result = tf0_filtered(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        if extra_filter(day, action, conf): return result
        return None
    return fn

tests = [
    ('h60', lambda d, a, c: d.cs_channel_health >= 0.60),
    ('h55', lambda d, a, c: d.cs_channel_health >= 0.55),
    ('h50', lambda d, a, c: d.cs_channel_health >= 0.50),
    ('h45', lambda d, a, c: d.cs_channel_health >= 0.45),
    ('h40', lambda d, a, c: d.cs_channel_health >= 0.40),
    ('h35', lambda d, a, c: d.cs_channel_health >= 0.35),
    ('h50&confl70', lambda d, a, c: d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 0.70),
    ('h45&confl70', lambda d, a, c: d.cs_channel_health >= 0.45 and d.cs_confluence_score >= 0.70),
    ('h40&confl70', lambda d, a, c: d.cs_channel_health >= 0.40 and d.cs_confluence_score >= 0.70),
    ('h40&confl80', lambda d, a, c: d.cs_channel_health >= 0.40 and d.cs_confluence_score >= 0.80),
    ('h50|h40&confl70', lambda d, a, c: d.cs_channel_health >= 0.50 or (d.cs_channel_health >= 0.40 and d.cs_confluence_score >= 0.70)),
    ('h50|h40&c70', lambda d, a, c: d.cs_channel_health >= 0.50 or (d.cs_channel_health >= 0.40 and c >= 0.70)),
    ('h50|h35&confl80', lambda d, a, c: d.cs_channel_health >= 0.50 or (d.cs_channel_health >= 0.35 and d.cs_confluence_score >= 0.80)),
    ('h45|h35&confl70', lambda d, a, c: d.cs_channel_health >= 0.45 or (d.cs_channel_health >= 0.35 and d.cs_confluence_score >= 0.70)),
    ('h45|h35&c70', lambda d, a, c: d.cs_channel_health >= 0.45 or (d.cs_channel_health >= 0.35 and c >= 0.70)),
    ('h40|h30&confl80', lambda d, a, c: d.cs_channel_health >= 0.40 or (d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.80)),
    ('h40|h30&c75', lambda d, a, c: d.cs_channel_health >= 0.40 or (d.cs_channel_health >= 0.30 and c >= 0.75)),
    ('c80', lambda d, a, c: c >= 0.80),
    ('c85', lambda d, a, c: c >= 0.85),
    ('c90', lambda d, a, c: c >= 0.90),
    ('confl90&h35', lambda d, a, c: d.cs_confluence_score >= 0.90 and d.cs_channel_health >= 0.35),
    ('confl90&c60', lambda d, a, c: d.cs_confluence_score >= 0.90 and c >= 0.60),
]

for name, filt in tests:
    fn = make_cr_plus_tf0(filt)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 331: continue
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

# ── Profile TF1 trades that CR rejected ──
print("\n=== TF1 TRADES REJECTED BY CR ===")
from v15.validation.combo_backtest import _make_s1_tf3_vix_combo
base_fn = _make_s1_tf3_vix_combo(cascade_vix)  # tf3 base for reference

# Build a tf1 base with streak tracking
def make_tf1_base():
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
        if confirmed < 1: return None
        sig = _SigProxy(day)
        ana = _AnalysisProxy(day.cs_tf_states)
        ok, adj, _ = cascade_vix.evaluate(sig, ana, feature_vec=None, bar_datetime=day.date,
                                           higher_tf_data=None, spy_df=None, vix_df=None)
        if not ok or adj < MIN_SIGNAL_CONFIDENCE: return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, adj, s, t, 'CS')
    return fn

tf1_base = make_tf1_base()
tf1_filtered = make_cp_filtered(tf1_base)
tf1_trades = simulate_trades(signals, tf1_filtered, 'tf1_filt', cooldown=0, trail_power=6)
tf1_not_in_cr = [t for t in tf1_trades if t.entry_date not in cr_dates]
print(f"TF1 filtered total: {len(tf1_trades)}, rejected by CR: {len(tf1_not_in_cr)} "
      f"({sum(1 for t in tf1_not_in_cr if t.pnl>0)}W/{sum(1 for t in tf1_not_in_cr if t.pnl<=0)}L)")

if tf1_not_in_cr:
    print(f"\n  {'Date':12} {'Dir':5} {'PnL':>8} {'Conf':>5} {'Health':>6} {'Pos':>5} "
          f"{'Confl':>5} {'TFs':>3} {'SPY20':>6} {'VIX':>5} {'SRet':>6} {'DOW':>4} {'Train':>5}")
    for t in sorted(tf1_not_in_cr, key=lambda x: x.entry_date):
        day = day_map.get(t.entry_date)
        if not day: continue
        tfs = _count_tf_confirming(day, t.direction)
        spy_d20 = spy_dist_map.get(day.date, 999)
        vix = vix_map.get(day.date, 22)
        sret = spy_return_map.get(day.date, 0)
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dd.weekday()]
        train = "TRAIN" if t.entry_date.year <= 2021 else "TEST"
        print(f"  {str(day.date)[:10]:12} {t.direction:5} ${t.pnl:>+7,.0f} {day.cs_confidence:5.3f} {day.cs_channel_health:6.3f} "
              f"{day.cs_position_score:5.2f} {day.cs_confluence_score:5.2f} {tfs:3d} {spy_d20:+6.2f} {vix:5.1f} {sret:+6.2f} {dow_name:>4} {train:>5}")

# ── Recovery experiments for TF1 rejected trades ──
print("\n=== TF1 RECOVERY EXPERIMENTS ===")
def make_cr_plus_tf1_recovery(check):
    def fn(day):
        result = cr_fn(day)
        if result is not None: return result
        result = tf1_filtered(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        if check(day, action, conf): return result
        return None
    return fn

recoveries = [
    # Lower health thresholds for BUY-only
    ('BUY h30&confl70', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.70),
    ('BUY h25&confl80', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.80),
    ('BUY h30&c70', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.30 and c >= 0.70),
    ('BUY h25&c75', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.25 and c >= 0.75),
    ('BUY h30&pos<85', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.30 and d.cs_position_score < 0.85),
    ('BUY h25&pos<80', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.25 and d.cs_position_score < 0.80),
    ('BUY h20&confl90', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.20 and d.cs_confluence_score >= 0.90),
    ('BUY c80&h20', lambda d, a, c: a == 'BUY' and c >= 0.80 and d.cs_channel_health >= 0.20),
    ('BUY c90', lambda d, a, c: a == 'BUY' and c >= 0.90),
    # Lower health thresholds for SELL-only
    ('SELL h30&confl70', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.70),
    ('SELL h25&confl80', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.80),
    ('SELL h30&c70', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.30 and c >= 0.70),
    ('SELL h25&c75', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.25 and c >= 0.75),
    ('SELL c80&h20', lambda d, a, c: a == 'SELL' and c >= 0.80 and d.cs_channel_health >= 0.20),
    ('SELL c90', lambda d, a, c: a == 'SELL' and c >= 0.90),
    # Day-of-week specific
    ('Mon&h30', lambda d, a, c: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 0 and d.cs_channel_health >= 0.30),
    ('Wed&h30', lambda d, a, c: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 2 and d.cs_channel_health >= 0.30),
    ('Thu&h30', lambda d, a, c: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 3 and d.cs_channel_health >= 0.30),
    ('Fri&h30', lambda d, a, c: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 4 and d.cs_channel_health >= 0.30),
    # VIX specific
    ('VIX<18&h25', lambda d, a, c: vix_map.get(d.date, 22) < 18 and d.cs_channel_health >= 0.25),
    ('VIX>25&h25', lambda d, a, c: vix_map.get(d.date, 22) > 25 and d.cs_channel_health >= 0.25),
    ('VIX>30&h20', lambda d, a, c: vix_map.get(d.date, 22) > 30 and d.cs_channel_health >= 0.20),
    # Combined
    ('h30&notTue', lambda d, a, c: d.cs_channel_health >= 0.30 and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 1),
    ('h25&notTue&confl70', lambda d, a, c: d.cs_channel_health >= 0.25 and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 1 and d.cs_confluence_score >= 0.70),
]

for name, check in recoveries:
    fn = make_cr_plus_tf1_recovery(check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 331: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

# ── Theoretical maximum: all CS signals with CP gates ──
print("\n=== THEORETICAL MAX: ALL CS + CP GATES ===")
all_cs_filt = make_cp_filtered(make_tf0_base())
all_trades = simulate_trades(signals, all_cs_filt, 'all_cs', cooldown=0, trail_power=6)
print(f"ALL CS + CP gates: {len(all_trades)} trades, "
      f"{sum(1 for t in all_trades if t.pnl>0)/max(1,len(all_trades))*100:.1f}% WR, "
      f"${sum(t.pnl for t in all_trades):+,.0f}")

# ── Trail power sweep on CR ──
print("\n=== TRAIL POWER SWEEP ON CR ===")
for power in [5, 6, 7, 8, 10, 12]:
    cr_fn_p = _make_v25_cr(cascade_vix, spy_above_sma20, spy_above_055pct,
                             spy_dist_map, spy_dist_5, spy_dist_50,
                             vix_map, spy_return_map, spy_ret_2d)
    trades = simulate_trades(signals, cr_fn_p, f'CR_p{power}', cooldown=0, trail_power=power)
    n = len(trades)
    if n == 0: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    print(f"  power={power:2d}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

print("\nDone")
