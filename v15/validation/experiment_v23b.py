#!/usr/bin/env python3
"""v23b: Combine top v23 100% WR recovery axes on CO (248).
All recoveries are LONG trades. Top: c55&h30&pos<95 (+3), Wed&c55 (+3), pos<90&h25&c55 (+2)."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations
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

base_fn = _make_s1_tf3_vix_combo(cascade_vix)
co_fn = _make_v22_co(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50,
                      vix_map, spy_return_map, spy_ret_2d)

co_trades = simulate_trades(signals, co_fn, 'CO', cooldown=0, trail_power=6)
print(f"CO baseline: {len(co_trades)} trades, {sum(1 for t in co_trades if t.pnl>0)/len(co_trades)*100:.1f}% WR")

RECOVERIES = {
    'c55_h30_pos95': lambda d, a, c: a == 'BUY' and c >= 0.55 and d.cs_channel_health >= 0.30 and d.cs_position_score < 0.95,
    'Wed_c55': lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 2 and c >= 0.55,
    'pos90_h25_c55': lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.90 and d.cs_channel_health >= 0.25 and c >= 0.55,
    'VIX20_c60': lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) > 20 and c >= 0.60,
    'SPYdn2_c60': lambda d, a, c: a == 'BUY' and spy_dist_map.get(d.date, -999) < -2.0 and c >= 0.60,
    'Mon_c55': lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 0 and c >= 0.55,
    'Thu_c55': lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 3 and c >= 0.55,
    'Fri_c55': lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 4 and c >= 0.55,
    # Simpler: all days except Tuesday at c55
    'notTue_c55': lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 1 and c >= 0.55,
    # Even simpler: h>=0.25 & c>=0.55 & pos<0.95
    'h25_c55_pos95': lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.25 and c >= 0.55 and d.cs_position_score < 0.95,
    # New combos not tested before
    'h20_c60_pos90': lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.20 and c >= 0.60 and d.cs_position_score < 0.90,
    'h25_c55_TF4': lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.25 and c >= 0.55 and _count_tf_confirming(d, 'BUY') >= 4,
    'h20_c60_TF4': lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.20 and c >= 0.60 and _count_tf_confirming(d, 'BUY') >= 4,
    'h20_c55_TF5': lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.20 and c >= 0.55 and _count_tf_confirming(d, 'BUY') >= 5,
    'c55_notTue_h20': lambda d, a, c: a == 'BUY' and c >= 0.55 and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 1 and d.cs_channel_health >= 0.20,
}

def make_combo(keys):
    checks = [RECOVERIES[k] for k in keys]
    def fn(day):
        result = co_fn(day)
        if result is not None:
            return result
        result2 = base_fn(day)
        if result2 is None:
            return None
        action, conf, s_pct, t_pct, src = result2
        for check in checks:
            if check(day, action, conf):
                return result2
        return None
    return fn

def test_combo(name, keys):
    fn = make_combo(keys)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n == 0: return n
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    tr = [t for t in trades if t.entry_date.year <= 2021]
    ts = [t for t in trades if t.entry_date.year > 2021]
    marker = " ***" if wr >= 100 and n > 248 else ""
    print(f"  {name:45s}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f} "
          f"[Tr:{len(tr)}@{sum(1 for t in tr if t.pnl>0)/len(tr)*100:.0f}% Ts:{len(ts)}@{sum(1 for t in ts if t.pnl>0)/len(ts)*100:.0f}%]{marker}")
    return n

# ── Individual tests ──
print("\n=== INDIVIDUAL ===")
for key in RECOVERIES:
    test_combo(key, [key])

# ── Top pairs ──
print("\n=== PAIRWISE (top 3 individual gainers) ===")
top3 = ['c55_h30_pos95', 'Wed_c55', 'pos90_h25_c55']
for pair in combinations(top3, 2):
    test_combo('+'.join(pair), list(pair))

# ── All top 3 ──
print("\n=== ALL TOP 3 ===")
test_combo('+'.join(top3), top3)

# ── Top 3 + day-specific ──
print("\n=== TOP 3 + DAY ===")
test_combo('top3+Mon', top3 + ['Mon_c55'])
test_combo('top3+Thu', top3 + ['Thu_c55'])
test_combo('top3+Fri', top3 + ['Fri_c55'])
test_combo('top3+VIX20', top3 + ['VIX20_c60'])
test_combo('top3+SPYdn2', top3 + ['SPYdn2_c60'])
test_combo('top3+Mon+Thu+Fri', top3 + ['Mon_c55', 'Thu_c55', 'Fri_c55'])
test_combo('top3+Mon+Thu+Fri+VIX20+SPYdn2', top3 + ['Mon_c55', 'Thu_c55', 'Fri_c55', 'VIX20_c60', 'SPYdn2_c60'])

# ── Simpler universal rules ──
print("\n=== SIMPLER UNIVERSAL RULES ===")
test_combo('notTue_c55', ['notTue_c55'])
test_combo('h25_c55_pos95', ['h25_c55_pos95'])
test_combo('h20_c60_pos90', ['h20_c60_pos90'])
test_combo('h25_c55_TF4', ['h25_c55_TF4'])
test_combo('h20_c60_TF4', ['h20_c60_TF4'])
test_combo('h20_c55_TF5', ['h20_c55_TF5'])
test_combo('c55_notTue_h20', ['c55_notTue_h20'])

# ── Combined best universal + day ──
print("\n=== BEST COMBOS ===")
test_combo('notTue_c55+h25_c55_pos95', ['notTue_c55', 'h25_c55_pos95'])
test_combo('notTue_c55+h25_c55_TF4', ['notTue_c55', 'h25_c55_TF4'])
test_combo('c55_notTue_h20+h25_c55_pos95', ['c55_notTue_h20', 'h25_c55_pos95'])
test_combo('notTue_c55+h20_c60_TF4', ['notTue_c55', 'h20_c60_TF4'])
test_combo('h25_c55_pos95+h20_c60_TF4', ['h25_c55_pos95', 'h20_c60_TF4'])
test_combo('h25_c55_pos95+h25_c55_TF4', ['h25_c55_pos95', 'h25_c55_TF4'])

# ── The "just let them all through" test ──
print("\n=== MAXIMUM RECOVERY ===")
all_keys = list(RECOVERIES.keys())
test_combo('ALL', all_keys)

# ── Detailed analysis of best combo ──
print("\n=== DETAILED BEST ===")
# Run best combo and show year-by-year
from collections import defaultdict
best_keys = ['notTue_c55']  # try the simplest first
best_fn = make_combo(best_keys)
best_trades = simulate_trades(signals, best_fn, 'BEST', cooldown=0, trail_power=6)
n = len(best_trades)
wins = sum(1 for t in best_trades if t.pnl > 0)
print(f"\nnotTue_c55: {n} trades, {wins/n*100:.1f}% WR, ${sum(t.pnl for t in best_trades):+,.0f}")

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
