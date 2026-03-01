#!/usr/bin/env python3
"""v23: Profile remaining ~15 filtered trades after CO (248).
AI baseline = 269 (263 at 100%). ~15 trades left to recover.
Deep dive with aggressive multi-condition OR recovery."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming,
    simulate_trades, _make_s1_tf3_vix_combo, _build_filter_cascade,
    _make_v22_co
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

base_fn = _make_s1_tf3_vix_combo(cascade_vix)
co_fn = _make_v22_co(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50,
                      vix_map, spy_return_map, spy_ret_2d)

# Run both
base_trades = simulate_trades(signals, base_fn, 'base', cooldown=0, trail_power=6)
co_trades = simulate_trades(signals, co_fn, 'CO', cooldown=0, trail_power=6)

co_dates = {t.entry_date for t in co_trades}
filtered_out = [t for t in base_trades if t.entry_date not in co_dates]

print(f"Base (s1_tf3_vix): {len(base_trades)} trades")
print(f"CO: {len(co_trades)} trades")
print(f"Filtered out: {len(filtered_out)} trades")
print(f"  Winners: {sum(1 for t in filtered_out if t.pnl > 0)}")
print(f"  Losers: {sum(1 for t in filtered_out if t.pnl <= 0)}")
print()

# Build day lookup
day_map = {day.date: day for day in signals}

# Profile each filtered trade in extreme detail
print("=" * 160)
print(f"{'Date':12} {'Dir':5} {'PnL':>8} {'Conf':>5} {'Health':>6} {'Pos':>5} {'Confl':>5} {'TFs':>3} "
      f"{'SPY%':>6} {'SMA5':>6} {'SMA50':>6} {'VIX':>5} {'SRet':>6} {'SR2d':>6} {'SR3d':>6} {'DOW':>4}")
print("=" * 160)

for t in sorted(filtered_out, key=lambda x: x.entry_date):
    day = day_map.get(t.entry_date)
    if day is None: continue
    result = base_fn(day)
    if result is None: continue
    action, conf, s_pct, t_pct, src = result
    tfs = _count_tf_confirming(day, action)
    spy_d20 = spy_dist_map.get(day.date, 999)
    spy_d5 = spy_dist_5.get(day.date, -999)
    spy_d50 = spy_dist_50.get(day.date, -999)
    vix = vix_map.get(day.date, 22)
    sret = spy_return_map.get(day.date, 0)
    sr2d = spy_ret_2d.get(day.date, 0)
    sr3d = spy_ret_3d.get(day.date, 0)
    dd = day.date.date() if hasattr(day.date, 'date') else day.date
    dow = dd.weekday()
    dow_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]

    print(f"{str(day.date)[:10]:12} {action:5} ${t.pnl:>+7,.0f} {conf:5.3f} {day.cs_channel_health:6.3f} "
          f"{day.cs_position_score:5.2f} {day.cs_confluence_score:5.2f} {tfs:3d} {spy_d20:+6.2f} {spy_d5:+6.2f} "
          f"{spy_d50:+6.2f} {vix:5.1f} {sret:+6.2f} {sr2d:+6.2f} {sr3d:+6.2f} {dow_name:>4}")

print()

# ── Aggressive recovery experiments ──
# For each remaining trade, figure out what UNIQUE condition could let it through

def make_co_plus(name, check):
    def fn(day):
        result = co_fn(day)
        if result is not None:
            return result
        result2 = base_fn(day)
        if result2 is None:
            return None
        action, conf, s_pct, t_pct, src = result2
        if check(day, action, conf):
            return result2
        return None
    return fn

print("=" * 80)
print("AGGRESSIVE RECOVERY EXPERIMENTS")
print("=" * 80)

# ── A: Very low thresholds with multiple guards ──
print("\n--- A: Multi-guard ultra-low thresholds ---")
a_experiments = [
    # LONG: any remaining BUY with multiple safety checks
    ('A: BUY c60&h30&TF3', lambda d, a, c: a == 'BUY' and c >= 0.60 and d.cs_channel_health >= 0.30 and _count_tf_confirming(d, 'BUY') >= 3),
    ('A: BUY c55&h40&TF3', lambda d, a, c: a == 'BUY' and c >= 0.55 and d.cs_channel_health >= 0.40 and _count_tf_confirming(d, 'BUY') >= 3),
    ('A: BUY c50&h50&TF3', lambda d, a, c: a == 'BUY' and c >= 0.50 and d.cs_channel_health >= 0.50 and _count_tf_confirming(d, 'BUY') >= 3),
    ('A: BUY c55&h30&pos<95', lambda d, a, c: a == 'BUY' and c >= 0.55 and d.cs_channel_health >= 0.30 and d.cs_position_score < 0.95),
    ('A: BUY c60&h20&pos<90', lambda d, a, c: a == 'BUY' and c >= 0.60 and d.cs_channel_health >= 0.20 and d.cs_position_score < 0.90),
    ('A: BUY c70&h15', lambda d, a, c: a == 'BUY' and c >= 0.70 and d.cs_channel_health >= 0.15),
    ('A: BUY c75&h10', lambda d, a, c: a == 'BUY' and c >= 0.75 and d.cs_channel_health >= 0.10),
    ('A: BUY c80&any', lambda d, a, c: a == 'BUY' and c >= 0.80),
    ('A: BUY c85', lambda d, a, c: a == 'BUY' and c >= 0.85),
    # SHORT: any remaining SELL
    ('A: SELL c65&h25&TF3', lambda d, a, c: a == 'SELL' and c >= 0.65 and d.cs_channel_health >= 0.25 and _count_tf_confirming(d, 'SELL') >= 3),
    ('A: SELL c70&h20&TF3', lambda d, a, c: a == 'SELL' and c >= 0.70 and d.cs_channel_health >= 0.20 and _count_tf_confirming(d, 'SELL') >= 3),
    ('A: SELL c75&h15&TF3', lambda d, a, c: a == 'SELL' and c >= 0.75 and d.cs_channel_health >= 0.15 and _count_tf_confirming(d, 'SELL') >= 3),
    ('A: SELL c80&h10', lambda d, a, c: a == 'SELL' and c >= 0.80 and d.cs_channel_health >= 0.10),
    ('A: SELL c80', lambda d, a, c: a == 'SELL' and c >= 0.80),
    ('A: SELL c85', lambda d, a, c: a == 'SELL' and c >= 0.85),
    ('A: SELL h40&TF3', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.40 and _count_tf_confirming(d, 'SELL') >= 3),
    ('A: SELL h35&c55&TF4', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.35 and c >= 0.55 and _count_tf_confirming(d, 'SELL') >= 4),
]

for name, check in a_experiments:
    fn = make_co_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 248: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

# ── B: VIX-conditioned recovery ──
print("\n--- B: VIX-conditioned ---")
b_experiments = [
    ('B: BUY VIX>20&c60', lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) > 20 and c >= 0.60),
    ('B: BUY VIX>22&c55', lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) > 22 and c >= 0.55),
    ('B: BUY VIX<15&c60', lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) < 15 and c >= 0.60),
    ('B: BUY VIX<13&c55', lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) < 13 and c >= 0.55),
    ('B: SELL VIX>28&h15', lambda d, a, c: a == 'SELL' and vix_map.get(d.date, 22) > 28 and d.cs_channel_health >= 0.15),
    ('B: SELL VIX<16&h20', lambda d, a, c: a == 'SELL' and vix_map.get(d.date, 22) < 16 and d.cs_channel_health >= 0.20),
    ('B: SELL VIX<14&h15', lambda d, a, c: a == 'SELL' and vix_map.get(d.date, 22) < 14 and d.cs_channel_health >= 0.15),
    ('B: SELL VIX20-25&TF4&c65', lambda d, a, c: a == 'SELL' and 20 <= vix_map.get(d.date, 22) <= 25 and _count_tf_confirming(d, 'SELL') >= 4 and c >= 0.65),
    ('B: SELL VIX20-25&h30&c60', lambda d, a, c: a == 'SELL' and 20 <= vix_map.get(d.date, 22) <= 25 and d.cs_channel_health >= 0.30 and c >= 0.60),
    ('B: SELL VIX20-25&h35&c55', lambda d, a, c: a == 'SELL' and 20 <= vix_map.get(d.date, 22) <= 25 and d.cs_channel_health >= 0.35 and c >= 0.55),
    ('B: SELL VIX20-25&h40', lambda d, a, c: a == 'SELL' and 20 <= vix_map.get(d.date, 22) <= 25 and d.cs_channel_health >= 0.40),
]

for name, check in b_experiments:
    fn = make_co_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 248: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

# ── C: SPY-conditioned recovery ──
print("\n--- C: SPY-conditioned ---")
c_experiments = [
    ('C: BUY SPYd20>2&c55', lambda d, a, c: a == 'BUY' and spy_dist_map.get(d.date, -999) > 2.0 and c >= 0.55),
    ('C: BUY SPYd20>1.5&c55', lambda d, a, c: a == 'BUY' and spy_dist_map.get(d.date, -999) > 1.5 and c >= 0.55),
    ('C: BUY SPYd5>0.5&c60', lambda d, a, c: a == 'BUY' and spy_dist_5.get(d.date, -999) > 0.5 and c >= 0.60),
    ('C: BUY SPYd50>3&c50', lambda d, a, c: a == 'BUY' and spy_dist_50.get(d.date, -999) > 3.0 and c >= 0.50),
    ('C: SELL SPYd20<-3&h20', lambda d, a, c: a == 'SELL' and spy_dist_map.get(d.date, 999) < -3.0 and d.cs_channel_health >= 0.20),
    ('C: SELL SPYd20<-2&h25', lambda d, a, c: a == 'SELL' and spy_dist_map.get(d.date, 999) < -2.0 and d.cs_channel_health >= 0.25),
    ('C: SELL SPYd5<-1&h20', lambda d, a, c: a == 'SELL' and spy_dist_5.get(d.date, 999) < -1.0 and d.cs_channel_health >= 0.20),
    ('C: SELL SPYd50<-2&h20', lambda d, a, c: a == 'SELL' and spy_dist_50.get(d.date, 999) < -2.0 and d.cs_channel_health >= 0.20),
    ('C: BUY SPYd20<-1&c70', lambda d, a, c: a == 'BUY' and spy_dist_map.get(d.date, -999) < -1.0 and c >= 0.70),
    ('C: BUY SPYd20<-2&c60', lambda d, a, c: a == 'BUY' and spy_dist_map.get(d.date, -999) < -2.0 and c >= 0.60),
]

for name, check in c_experiments:
    fn = make_co_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 248: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

# ── D: Day-of-week specific ──
print("\n--- D: Day-of-week ---")
d_experiments = [
    ('D: BUY Mon&c55', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 0 and c >= 0.55),
    ('D: BUY Tue&c55', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and c >= 0.55),
    ('D: BUY Wed&c55', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 2 and c >= 0.55),
    ('D: BUY Thu&c55', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 3 and c >= 0.55),
    ('D: BUY Fri&c55', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 4 and c >= 0.55),
    ('D: SELL Tue&h20', lambda d, a, c: a == 'SELL' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and d.cs_channel_health >= 0.20),
    ('D: SELL Fri&h20', lambda d, a, c: a == 'SELL' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 4 and d.cs_channel_health >= 0.20),
    ('D: SELL Tue&c60', lambda d, a, c: a == 'SELL' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and c >= 0.60),
    ('D: SELL Fri&c60', lambda d, a, c: a == 'SELL' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 4 and c >= 0.60),
]

for name, check in d_experiments:
    fn = make_co_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 248: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

# ── E: Extreme confluence/position combos ──
print("\n--- E: Extreme signal quality ---")
e_experiments = [
    ('E: BUY confl80&c50&TF3', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.80 and c >= 0.50 and _count_tf_confirming(d, 'BUY') >= 3),
    ('E: BUY confl75&c55&TF3', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.75 and c >= 0.55 and _count_tf_confirming(d, 'BUY') >= 3),
    ('E: BUY confl70&c60&TF3', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.70 and c >= 0.60 and _count_tf_confirming(d, 'BUY') >= 3),
    ('E: SELL confl90&c55', lambda d, a, c: a == 'SELL' and d.cs_confluence_score >= 0.90 and c >= 0.55),
    ('E: SELL confl85&c60', lambda d, a, c: a == 'SELL' and d.cs_confluence_score >= 0.85 and c >= 0.60),
    ('E: SELL confl80&c65', lambda d, a, c: a == 'SELL' and d.cs_confluence_score >= 0.80 and c >= 0.65),
    ('E: BUY pos<80&h30&c50', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.80 and d.cs_channel_health >= 0.30 and c >= 0.50),
    ('E: BUY pos<70&h20&c50', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.70 and d.cs_channel_health >= 0.20 and c >= 0.50),
    ('E: BUY pos<90&h25&c55', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.90 and d.cs_channel_health >= 0.25 and c >= 0.55),
    ('E: SELL pos>99&c55', lambda d, a, c: a == 'SELL' and d.cs_position_score >= 0.99 and c >= 0.55),
    ('E: SELL pos>95&c60&TF3', lambda d, a, c: a == 'SELL' and d.cs_position_score >= 0.95 and c >= 0.60 and _count_tf_confirming(d, 'SELL') >= 3),
]

for name, check in e_experiments:
    fn = make_co_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 248: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

# ── F: Momentum persistence (multi-day SPY) ──
print("\n--- F: SPY momentum persistence ---")
f_experiments = [
    ('F: BUY SR3d>2&c50', lambda d, a, c: a == 'BUY' and spy_ret_3d.get(d.date, 0) > 2.0 and c >= 0.50),
    ('F: BUY SR3d>1&c55', lambda d, a, c: a == 'BUY' and spy_ret_3d.get(d.date, 0) > 1.0 and c >= 0.55),
    ('F: SELL SR3d<-2&h15', lambda d, a, c: a == 'SELL' and spy_ret_3d.get(d.date, 0) < -2.0 and d.cs_channel_health >= 0.15),
    ('F: SELL SR3d<-1.5&h20', lambda d, a, c: a == 'SELL' and spy_ret_3d.get(d.date, 0) < -1.5 and d.cs_channel_health >= 0.20),
    ('F: SELL SR2d<-1&h25', lambda d, a, c: a == 'SELL' and spy_ret_2d.get(d.date, 0) < -1.0 and d.cs_channel_health >= 0.25),
    ('F: BUY SR>0.5&c60&h20', lambda d, a, c: a == 'BUY' and spy_return_map.get(d.date, 0) > 0.5 and c >= 0.60 and d.cs_channel_health >= 0.20),
]

for name, check in f_experiments:
    fn = make_co_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 248: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

# ── G: Combined multi-axis recovery ──
print("\n--- G: Combined recovery ---")
# Try combining ANY check that individually adds at least 1 trade at 100%
# Build a list of successful individual checks and combine them

print("\nDone")
