#!/usr/bin/env python3
"""v29: Beyond TF expansion. CT=345 trades, 100% WR. Theoretical max=352 with CP gates.
Explore entirely new approaches:
1. Relax CP gates themselves (not just TF/health filters)
2. Alternative signals: CS HOLD with extreme indicators
3. Trail/stop variations per-trade
4. Multi-day signal persistence (signal fires N days in a row)
5. Cross-signal features (V5 bounce + CS signal overlap)
6. Time-of-year seasonality
7. Earnings proximity
"""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _build_filter_cascade,
    _make_v27_ct, _make_v26_cs, _SigProxy, _AnalysisProxy, _floor_stop_tp
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

# ── SECTION 1: How many CS signals total? (including HOLD) ──
print("\n=== CS SIGNAL CENSUS ===")
total_days = len(signals)
buy_days = sum(1 for d in signals if d.cs_action == 'BUY')
sell_days = sum(1 for d in signals if d.cs_action == 'SELL')
hold_days = sum(1 for d in signals if d.cs_action == 'HOLD')
none_days = sum(1 for d in signals if d.cs_action not in ('BUY', 'SELL', 'HOLD'))
print(f"Total: {total_days}, BUY: {buy_days}, SELL: {sell_days}, HOLD: {hold_days}, None: {none_days}")

# BUY/SELL with various confidence thresholds
for thresh in [0.0, 0.10, 0.20, 0.30, 0.45]:
    n = sum(1 for d in signals if d.cs_action in ('BUY', 'SELL') and d.cs_confidence >= thresh)
    print(f"  BUY|SELL conf>={thresh}: {n}")

# ── SECTION 2: Expand beyond CP gates — what if we relax SPY/CONF gates? ──
print("\n=== BEYOND CP GATES: RELAX SPY GATE ===")
# Currently CP has complex SPY gate. What if we use simpler alternatives?

def make_all_cs_base():
    """TF0 base: just CS signal + VIX cascade."""
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

# How many TF0 signals pass VIX cascade but NOT CP gates?
all_base_trades = simulate_trades(signals, all_base_fn, 'allBase', cooldown=0, trail_power=6)
not_in_ct = [t for t in all_base_trades if t.entry_date not in ct_dates]
print(f"\nAll TF0 base (no CP gates): {len(all_base_trades)} trades "
      f"({sum(1 for t in all_base_trades if t.pnl>0)}W/{sum(1 for t in all_base_trades if t.pnl<=0)}L)")
print(f"Not in CT: {len(not_in_ct)} trades "
      f"({sum(1 for t in not_in_ct if t.pnl>0)}W/{sum(1 for t in not_in_ct if t.pnl<=0)}L)")

# Profile the gap trades (base - CT)
if not_in_ct:
    wins = [t for t in not_in_ct if t.pnl > 0]
    losses = [t for t in not_in_ct if t.pnl <= 0]
    print(f"\n  Win range: ${min(t.pnl for t in wins):+,.0f} to ${max(t.pnl for t in wins):+,.0f}" if wins else "")
    print(f"  Loss range: ${max(t.pnl for t in losses):+,.0f} to ${min(t.pnl for t in losses):+,.0f}" if losses else "")

    # What features distinguish wins from losses?
    print(f"\n  {'Date':12} {'Dir':5} {'PnL':>8} {'Conf':>5} {'Health':>6} {'Pos':>5} "
          f"{'Confl':>5} {'TFs':>3} {'SPY20':>6} {'VIX':>5} {'SRet':>6} {'DOW':>4}")
    for t in sorted(not_in_ct, key=lambda x: x.pnl)[:15]:  # worst 15
        day = day_map.get(t.entry_date)
        if not day: continue
        tfs = _count_tf_confirming(day, t.direction)
        spy_d20 = spy_dist_map.get(day.date, 999)
        vix = vix_map.get(day.date, 22)
        sret = spy_return_map.get(day.date, 0)
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        print(f"  {str(day.date)[:10]:12} {t.direction:5} ${t.pnl:>+7,.0f} {day.cs_confidence:5.3f} {day.cs_channel_health:6.3f} "
              f"{day.cs_position_score:5.2f} {day.cs_confluence_score:5.2f} {tfs:3d} {spy_d20:+6.2f} {vix:5.1f} {sret:+6.2f} {dow:>4}")
    print("  ...")
    for t in sorted(not_in_ct, key=lambda x: x.pnl)[-5:]:  # best 5
        day = day_map.get(t.entry_date)
        if not day: continue
        tfs = _count_tf_confirming(day, t.direction)
        spy_d20 = spy_dist_map.get(day.date, 999)
        vix = vix_map.get(day.date, 22)
        sret = spy_return_map.get(day.date, 0)
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        print(f"  {str(day.date)[:10]:12} {t.direction:5} ${t.pnl:>+7,.0f} {day.cs_confidence:5.3f} {day.cs_channel_health:6.3f} "
              f"{day.cs_position_score:5.2f} {day.cs_confluence_score:5.2f} {tfs:3d} {spy_d20:+6.2f} {vix:5.1f} {sret:+6.2f} {dow:>4}")

# ── SECTION 3: Simplified gate alternatives ──
print("\n=== SIMPLIFIED GATE ALTERNATIVES ===")

def make_simplified_gate(base_fn, check):
    """CT + simplified gate recovery for base signals that fail CP gates."""
    def fn(day):
        result = ct_fn(day)
        if result is not None: return result
        result = base_fn(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        if check(day, action, conf): return result
        return None
    return fn

# Try recovering base signals (that fail CP gates) with strong filters
simplified_gates = [
    ('h50&confl90', lambda d, a, c: d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 0.90),
    ('h50&confl80', lambda d, a, c: d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 0.80),
    ('h60&confl70', lambda d, a, c: d.cs_channel_health >= 0.60 and d.cs_confluence_score >= 0.70),
    ('h50&c80', lambda d, a, c: d.cs_channel_health >= 0.50 and c >= 0.80),
    ('h60&c70', lambda d, a, c: d.cs_channel_health >= 0.60 and c >= 0.70),
    ('h50&pos0', lambda d, a, c: d.cs_channel_health >= 0.50 and d.cs_position_score == 0.0),
    ('h40&pos0&confl80', lambda d, a, c: d.cs_channel_health >= 0.40 and d.cs_position_score == 0.0 and d.cs_confluence_score >= 0.80),
    ('BUY h50&SPY>0', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.50 and spy_dist_map.get(d.date, -999) >= 0),
    ('SELL h50&SPY>055', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.50 and d.date in spy_above_055pct),
    ('c90&h20', lambda d, a, c: c >= 0.90 and d.cs_channel_health >= 0.20),
    ('c85&h30', lambda d, a, c: c >= 0.85 and d.cs_channel_health >= 0.30),
    ('confl100', lambda d, a, c: d.cs_confluence_score >= 1.0),
    ('confl100&h20', lambda d, a, c: d.cs_confluence_score >= 1.0 and d.cs_channel_health >= 0.20),
]

for name, check in simplified_gates:
    fn = make_simplified_gate(all_base_fn, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 345: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in ct_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

# ── SECTION 4: Signal persistence (multi-day signals) ──
print("\n=== SIGNAL PERSISTENCE ===")
# For each day with a CS signal, check if the SAME direction signal fired
# on previous days. Persistent signals may be higher quality.

persistence_map = {}
prev_action = None
streak = 0
for day in signals:
    if day.cs_action in ('BUY', 'SELL'):
        if day.cs_action == prev_action:
            streak += 1
        else:
            streak = 1
        prev_action = day.cs_action
    else:
        streak = 0
        prev_action = None
    persistence_map[day.date] = streak

# Profile: do persistent signals (streak >= 2) have higher WR?
print("\n  Persistence distribution in gap trades:")
for min_streak in [1, 2, 3, 4, 5]:
    matching = [t for t in not_in_ct if persistence_map.get(t.entry_date, 0) >= min_streak]
    if not matching: continue
    w = sum(1 for t in matching if t.pnl > 0)
    l = sum(1 for t in matching if t.pnl <= 0)
    print(f"  streak>={min_streak}: {w}W/{l}L ({w/(w+l)*100:.1f}% WR)")

# Test: CT + persistent signals (streak >= 2) that bypass CP gates
for min_streak in [2, 3, 4]:
    def make_check(ms):
        def check(d, a, c):
            return persistence_map.get(d.date, 0) >= ms and d.cs_channel_health >= 0.30
        return check
    fn = make_simplified_gate(all_base_fn, make_check(min_streak))
    trades = simulate_trades(signals, fn, f'persist{min_streak}', cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 345: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in ct_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  streak>={min_streak}&h30: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

# ── SECTION 5: Month-of-year seasonality ──
print("\n=== MONTH SEASONALITY ===")
# Profile gap trades by month
month_dist = defaultdict(lambda: {'win': 0, 'loss': 0})
for t in not_in_ct:
    m = t.entry_date.month
    if t.pnl > 0: month_dist[m]['win'] += 1
    else: month_dist[m]['loss'] += 1

for m in sorted(month_dist.keys()):
    w, l = month_dist[m]['win'], month_dist[m]['loss']
    print(f"  Month {m:2d}: {w}W/{l}L")

# CT trades by month
ct_month_dist = defaultdict(lambda: {'win': 0, 'loss': 0, 'pnl': 0})
for t in ct_trades:
    m = t.entry_date.month
    ct_month_dist[m]['win'] += 1
    ct_month_dist[m]['pnl'] += t.pnl

print("\n  CT trades per month:")
for m in range(1, 13):
    d = ct_month_dist[m]
    print(f"  Month {m:2d}: {d['win']} trades, ${d['pnl']:+,.0f}")

# ── SECTION 6: Lower MIN_SIGNAL_CONFIDENCE ──
print("\n=== LOWER MIN_SIGNAL_CONFIDENCE ===")
# Currently MIN_SIGNAL_CONFIDENCE = 0.45. What if we lower it?
for min_conf in [0.40, 0.35, 0.30, 0.25, 0.20, 0.10]:
    def make_wider_base(mc):
        def fn(day):
            if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < mc:
                return None
            sig = _SigProxy(day)
            ana = _AnalysisProxy(day.cs_tf_states)
            ok, adj, _ = cascade_vix.evaluate(sig, ana, feature_vec=None, bar_datetime=day.date,
                                               higher_tf_data=None, spy_df=None, vix_df=None)
            if not ok or adj < mc: return None
            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return (day.cs_action, adj, s, t, 'CS')
        return fn

    wider = make_wider_base(min_conf)
    wider_trades = simulate_trades(signals, wider, f'mc{min_conf}', cooldown=0, trail_power=6)
    n = len(wider_trades)
    wins = sum(1 for t in wider_trades if t.pnl > 0)
    wr = wins / n * 100 if n > 0 else 0
    print(f"  min_conf={min_conf:.2f}: {n:4d} trades, {wr:5.1f}% WR")

# ── SECTION 7: V5 bounce signal overlap ──
print("\n=== V5 BOUNCE SIGNAL OVERLAP ===")
v5_dates = set()
for day in signals:
    if day.v5_take_bounce:
        v5_dates.add(day.date)
print(f"V5 bounce dates: {len(v5_dates)}")

# How many V5 dates are NOT in CT?
v5_not_in_ct = v5_dates - ct_dates
print(f"V5 dates not in CT: {len(v5_not_in_ct)}")

# What if we add V5 signals as override?
def make_ct_plus_v5():
    def fn(day):
        result = ct_fn(day)
        if result is not None: return result
        if day.v5_take_bounce:
            return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn

v5_fn = make_ct_plus_v5()
v5_trades = simulate_trades(signals, v5_fn, 'CT+V5', cooldown=0, trail_power=6)
n = len(v5_trades)
wins = sum(1 for t in v5_trades if t.pnl > 0)
wr = wins / n * 100
bl = min(t.pnl for t in v5_trades)
pnl = sum(t.pnl for t in v5_trades)
new_v5 = [t for t in v5_trades if t.entry_date not in ct_dates]
print(f"CT+V5: {n} trades (+{len(new_v5)}), {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")
if new_v5:
    for t in sorted(new_v5, key=lambda x: x.entry_date):
        day = day_map.get(t.entry_date)
        if day:
            print(f"  {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+7,.0f}")

# ── SECTION 8: What's left after CT? Remaining gap profile ──
print("\n=== REMAINING GAP: CT vs THEORETICAL MAX ===")
# All CP-gated TF0 trades (theoretical max with CP gates)
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

cp_trades = simulate_trades(signals, make_cp_filtered(all_base_fn), 'CP_all', cooldown=0, trail_power=6)
cp_not_ct = [t for t in cp_trades if t.entry_date not in ct_dates]
print(f"CP-gated TF0 (theoretical max): {len(cp_trades)} trades")
print(f"Not in CT: {len(cp_not_ct)} ({sum(1 for t in cp_not_ct if t.pnl>0)}W/{sum(1 for t in cp_not_ct if t.pnl<=0)}L)")

# Also: how many base signals fail CP gates entirely?
base_dates = {t.entry_date for t in all_base_trades}
cp_dates = {t.entry_date for t in cp_trades}
failed_cp = base_dates - cp_dates
print(f"\nBase signals that FAIL CP gates entirely: {len(failed_cp)}")

# Profile those that fail CP gates but are wins
failed_cp_trades = [t for t in all_base_trades if t.entry_date in failed_cp]
failed_wins = [t for t in failed_cp_trades if t.pnl > 0]
failed_losses = [t for t in failed_cp_trades if t.pnl <= 0]
print(f"  Wins: {len(failed_wins)}, Losses: {len(failed_losses)}")
if failed_wins:
    print(f"  Win PnL: ${sum(t.pnl for t in failed_wins):+,.0f}")
if failed_losses:
    print(f"  Loss PnL: ${sum(t.pnl for t in failed_losses):+,.0f}")

# Can we recover any of these?
for check_name, check in [
    ('h60&confl90', lambda d, a, c: d.cs_channel_health >= 0.60 and d.cs_confluence_score >= 0.90),
    ('h50&confl100', lambda d, a, c: d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 1.0),
    ('h50&c90', lambda d, a, c: d.cs_channel_health >= 0.50 and c >= 0.90),
    ('BUY h60&SPY>0', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.60 and spy_dist_map.get(d.date, -999) >= 0),
    ('BUY h50&VIX<18', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.50 and vix_map.get(d.date, 22) < 18),
    ('persist3&h40', lambda d, a, c: persistence_map.get(d.date, 0) >= 3 and d.cs_channel_health >= 0.40),
]:
    fn = make_simplified_gate(all_base_fn, check)
    trades = simulate_trades(signals, fn, check_name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 345: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in ct_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  {check_name:30s}: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

print("\nDone")
