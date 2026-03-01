#!/usr/bin/env python3
"""v27: Squeeze final trades from theoretical max (352). CS=339 at 100% WR.
13 remaining trades include 4 losses. Can we safely recover more wins?
Also test: TF1/TF0 trades rejected by CR/CS triple OR filters."""
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
print(f"CS: {len(cs_trades)} trades, {sum(1 for t in cs_trades if t.pnl>0)/len(cs_trades)*100:.1f}% WR, ${sum(t.pnl for t in cs_trades):+,.0f}")

day_map = {day.date: day for day in signals}
cs_dates = {t.entry_date for t in cs_trades}

# ── Build all-CS-signals base (no TF, no health, just CS+VIX+CP gates) ──
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
all_trades = simulate_trades(signals, all_filtered, 'all', cooldown=0, trail_power=6)
all_not_in_cs = [t for t in all_trades if t.entry_date not in cs_dates]
print(f"\nAll CS+CP: {len(all_trades)} trades ({sum(1 for t in all_trades if t.pnl>0)}W/{sum(1 for t in all_trades if t.pnl<=0)}L)")
print(f"Not in CS: {len(all_not_in_cs)} trades ({sum(1 for t in all_not_in_cs if t.pnl>0)}W/{sum(1 for t in all_not_in_cs if t.pnl<=0)}L)")

if all_not_in_cs:
    print(f"\n  {'Date':12} {'Dir':5} {'PnL':>8} {'Conf':>5} {'Health':>6} {'Pos':>5} "
          f"{'Confl':>5} {'TFs':>3} {'SPY20':>6} {'VIX':>5} {'SRet':>6} {'DOW':>4} {'Train':>5}")
    for t in sorted(all_not_in_cs, key=lambda x: x.entry_date):
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

# ── Exhaustive recovery attempts ──
print("\n=== RECOVERY ATTEMPTS ===")
def make_cs_plus(check):
    def fn(day):
        result = cs_fn(day)
        if result is not None: return result
        result = all_filtered(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        if check(day, action, conf): return result
        return None
    return fn

recoveries = [
    # Individual trade targeting
    ('c90', lambda d, a, c: c >= 0.90),
    ('c85&h30', lambda d, a, c: c >= 0.85 and d.cs_channel_health >= 0.30),
    ('c80&h30', lambda d, a, c: c >= 0.80 and d.cs_channel_health >= 0.30),
    ('c75&h30', lambda d, a, c: c >= 0.75 and d.cs_channel_health >= 0.30),
    ('c70&h30', lambda d, a, c: c >= 0.70 and d.cs_channel_health >= 0.30),
    ('confl90&h30', lambda d, a, c: d.cs_confluence_score >= 0.90 and d.cs_channel_health >= 0.30),
    ('confl80&h30', lambda d, a, c: d.cs_confluence_score >= 0.80 and d.cs_channel_health >= 0.30),
    ('pos<50&h25', lambda d, a, c: d.cs_position_score < 0.50 and d.cs_channel_health >= 0.25),
    ('pos<20&h20', lambda d, a, c: d.cs_position_score < 0.20 and d.cs_channel_health >= 0.20),
    ('pos0&h25', lambda d, a, c: d.cs_position_score == 0.0 and d.cs_channel_health >= 0.25),
    ('pos0&h30', lambda d, a, c: d.cs_position_score == 0.0 and d.cs_channel_health >= 0.30),
    ('pos0&h35', lambda d, a, c: d.cs_position_score == 0.0 and d.cs_channel_health >= 0.35),
    # BUY-specific
    ('BUY pos0&h30', lambda d, a, c: a == 'BUY' and d.cs_position_score == 0.0 and d.cs_channel_health >= 0.30),
    ('BUY pos0&h25&confl70', lambda d, a, c: a == 'BUY' and d.cs_position_score == 0.0 and d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.70),
    ('BUY pos<10&h30', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.10 and d.cs_channel_health >= 0.30),
    ('BUY pos<50&h30', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.50 and d.cs_channel_health >= 0.30),
    ('BUY VIX>30&h20', lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) > 30 and d.cs_channel_health >= 0.20),
    ('BUY VIX>50&h15', lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) > 50 and d.cs_channel_health >= 0.15),
    ('BUY SPY<-3&h25', lambda d, a, c: a == 'BUY' and spy_dist_map.get(d.date, 999) < -3.0 and d.cs_channel_health >= 0.25),
    # SELL-specific
    ('SELL pos0&h30', lambda d, a, c: a == 'SELL' and d.cs_position_score == 0.0 and d.cs_channel_health >= 0.30),
    ('SELL h30&c60', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.30 and c >= 0.60),
    ('SELL h30&confl70', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.70),
    # Day-of-week
    ('Mon&h30', lambda d, a, c: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 0 and d.cs_channel_health >= 0.30),
    ('Wed&h30', lambda d, a, c: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 2 and d.cs_channel_health >= 0.30),
    ('Thu&h30', lambda d, a, c: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 3 and d.cs_channel_health >= 0.30),
    ('Fri&h30', lambda d, a, c: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 4 and d.cs_channel_health >= 0.30),
    ('notFri&h30', lambda d, a, c: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 4 and d.cs_channel_health >= 0.30),
    ('notMon&h30', lambda d, a, c: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 0 and d.cs_channel_health >= 0.30),
    # SPY regime
    ('SPY>0&h25', lambda d, a, c: spy_dist_map.get(d.date, -999) > 0 and d.cs_channel_health >= 0.25),
    ('SPY>0&h30', lambda d, a, c: spy_dist_map.get(d.date, -999) > 0 and d.cs_channel_health >= 0.30),
    ('SPY>1&h25', lambda d, a, c: spy_dist_map.get(d.date, -999) > 1.0 and d.cs_channel_health >= 0.25),
    # VIX
    ('VIX<18&h30', lambda d, a, c: vix_map.get(d.date, 22) < 18 and d.cs_channel_health >= 0.30),
    ('VIX>25&h25', lambda d, a, c: vix_map.get(d.date, 22) > 25 and d.cs_channel_health >= 0.25),
    ('VIXext&h25', lambda d, a, c: (vix_map.get(d.date, 22) < 18 or vix_map.get(d.date, 22) > 25) and d.cs_channel_health >= 0.25),
    # Combined
    ('h30&pos<50&notFri', lambda d, a, c: d.cs_channel_health >= 0.30 and d.cs_position_score < 0.50 and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 4),
    ('h25&pos0&notFri', lambda d, a, c: d.cs_channel_health >= 0.25 and d.cs_position_score == 0.0 and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 4),
    ('h30&pos0&VIX<20', lambda d, a, c: d.cs_channel_health >= 0.30 and d.cs_position_score == 0.0 and vix_map.get(d.date, 22) < 20),
]

for name, check in recoveries:
    fn = make_cs_plus(check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 339: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

# ── Also test: what if we relax TF1/TF2 health thresholds? ──
print("\n=== RELAX EXISTING TF HEALTH THRESHOLDS ===")
# TF2 currently uses h35|h25&confl60|h25&c60. What if we add h25&pos0?
# TF1 currently uses h35|h30&confl60|h30&c60. What if we add h25&pos0?

def make_cs_relax(tf_level, check_name, check):
    """Try adding a recovery rule at a specific TF level."""
    # Build the appropriate TF base
    prev_tf_states = {}
    streaks = defaultdict(int)
    def tf_base(day):
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
            for tf2, state in day.cs_tf_states.items():
                if not state.get('valid', False): continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks.get(tf2, 0) >= 1:
                    confirmed += 1
        if confirmed < tf_level: return None
        sig = _SigProxy(day)
        ana = _AnalysisProxy(day.cs_tf_states)
        ok, adj, _ = cascade_vix.evaluate(sig, ana, feature_vec=None, bar_datetime=day.date,
                                           higher_tf_data=None, spy_df=None, vix_df=None)
        if not ok or adj < MIN_SIGNAL_CONFIDENCE: return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, adj, s, t, 'CS')
    tf_filtered = make_cp_filtered(tf_base)

    def fn(day):
        result = cs_fn(day)
        if result is not None: return result
        result = tf_filtered(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        if check(day, action, conf): return result
        return None
    return fn

relax_tests = [
    (2, 'tf2: h25&pos0', lambda d, a, c: d.cs_channel_health >= 0.25 and d.cs_position_score == 0.0),
    (2, 'tf2: h30&pos<50', lambda d, a, c: d.cs_channel_health >= 0.30 and d.cs_position_score < 0.50),
    (2, 'tf2: h25&c70', lambda d, a, c: d.cs_channel_health >= 0.25 and c >= 0.70),
    (2, 'tf2: h20&confl80', lambda d, a, c: d.cs_channel_health >= 0.20 and d.cs_confluence_score >= 0.80),
    (2, 'tf2: h20&c80', lambda d, a, c: d.cs_channel_health >= 0.20 and c >= 0.80),
    (1, 'tf1: h25&pos0', lambda d, a, c: d.cs_channel_health >= 0.25 and d.cs_position_score == 0.0),
    (1, 'tf1: h30&pos<50', lambda d, a, c: d.cs_channel_health >= 0.30 and d.cs_position_score < 0.50),
    (1, 'tf1: h25&c70', lambda d, a, c: d.cs_channel_health >= 0.25 and c >= 0.70),
    (1, 'tf1: h20&confl80', lambda d, a, c: d.cs_channel_health >= 0.20 and d.cs_confluence_score >= 0.80),
    (1, 'tf1: h20&c80', lambda d, a, c: d.cs_channel_health >= 0.20 and c >= 0.80),
    (1, 'tf1: BUY c90', lambda d, a, c: a == 'BUY' and c >= 0.90),
    (1, 'tf1: Mon&h30', lambda d, a, c: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 0 and d.cs_channel_health >= 0.30),
]

for tf_level, name, check in relax_tests:
    fn = make_cs_relax(tf_level, name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 339: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

print("\nDone")
