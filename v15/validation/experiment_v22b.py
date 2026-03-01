#!/usr/bin/env python3
"""v22b: Combine top v22 100% WR discoveries on top of CN (238 trades).
Top individual: LC:confl90&c45 (+4), VIX:BUY>30&c50 (+4), VIX:BUY>25&c55 (+3),
L:confl90&c50 (+2), S_SPY:Wed&h25 (+1), S_SPY:TF5&h20 (+1), SC:h05&c60&TF5 (+1),
Mom:SRet2d<-2&h15 (+1), Mom:BUY SRet>1&c55 (+1)"""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming,
    simulate_trades, _make_s1_tf3_vix_combo, _build_filter_cascade,
    _make_v21_cn
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
cn_fn = _make_v21_cn(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map)

# CN baseline
cn_trades = simulate_trades(signals, cn_fn, 'CN', cooldown=0, trail_power=6)
print(f"CN baseline: {len(cn_trades)} trades, {sum(1 for t in cn_trades if t.pnl>0)/len(cn_trades)*100:.1f}% WR")

# ── Define individual recovery checks ──
# Each returns True if the trade should be recovered
RECOVERIES = {
    'LC_confl90_c45': lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.9 and c >= 0.45,
    'VIX_BUY_30_c50': lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) > 30 and c >= 0.50,
    'VIX_BUY_25_c55': lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) > 25 and c >= 0.55,
    'L_confl90_c50': lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.9 and c >= 0.50,
    'S_SPY_Wed_h25': lambda d, a, c: a == 'SELL' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 2 and d.cs_channel_health >= 0.25,
    'S_SPY_TF5_h20': lambda d, a, c: a == 'SELL' and _count_tf_confirming(d, 'SELL') >= 5 and d.cs_channel_health >= 0.20,
    'SC_h05_c60_TF5': lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.05 and c >= 0.60 and _count_tf_confirming(d, 'SELL') >= 5,
    'Mom_SRet2d_n2_h15': lambda d, a, c: a == 'SELL' and spy_ret_2d.get(d.date, 0) < -2.0 and d.cs_channel_health >= 0.15,
    'Mom_BUY_SRet_p1_c55': lambda d, a, c: a == 'BUY' and spy_return_map.get(d.date, 0) > 1.0 and c >= 0.55,
}

def make_combo(recovery_keys):
    """Build a function that extends CN with selected recovery checks."""
    checks = [RECOVERIES[k] for k in recovery_keys]
    def fn(day):
        result = cn_fn(day)
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
    if n == 0: return
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    tr = [t for t in trades if t.entry_date.year <= 2021]
    ts = [t for t in trades if t.entry_date.year > 2021]
    marker = " ***" if wr >= 100 and n > 238 else ""
    print(f"  {name:40s}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f} "
          f"[Tr:{len(tr)}@{sum(1 for t in tr if t.pnl>0)/len(tr)*100:.0f}% Ts:{len(ts)}@{sum(1 for t in ts if t.pnl>0)/len(ts)*100:.0f}%]{marker}")

# ── Individual tests ──
print("\n=== INDIVIDUAL RECOVERIES ===")
for key in RECOVERIES:
    test_combo(key, [key])

# ── Pairwise: top 4 biggest gains ──
print("\n=== PAIRWISE (top gainers) ===")
top4 = ['LC_confl90_c45', 'VIX_BUY_30_c50', 'VIX_BUY_25_c55', 'L_confl90_c50']
from itertools import combinations
for pair in combinations(top4, 2):
    test_combo('+'.join(pair), list(pair))

# ── Triple: top 3 ──
print("\n=== TRIPLE ===")
for triple in combinations(top4, 3):
    test_combo('+'.join(triple), list(triple))

# ── All top 4 ──
print("\n=== ALL TOP 4 ===")
test_combo('+'.join(top4), top4)

# ── Now add short-side recoveries to best long combo ──
print("\n=== TOP LONG + SHORT RECOVERIES ===")
# Note: LC_confl90_c45 subsumes L_confl90_c50 (lower threshold)
# And VIX_BUY_25_c55 subsumes VIX_BUY_30_c50 (lower VIX threshold)
# So the real unique long axes are: LC_confl90_c45 + VIX_BUY_25_c55
long_base = ['LC_confl90_c45', 'VIX_BUY_25_c55']
short_recoveries = ['S_SPY_Wed_h25', 'S_SPY_TF5_h20', 'SC_h05_c60_TF5', 'Mom_SRet2d_n2_h15']

# Long base alone
test_combo('LONG_BASE', long_base)

# Add each short recovery individually
for sr in short_recoveries:
    test_combo(f'LONG_BASE+{sr}', long_base + [sr])

# Add pairwise short
for pair in combinations(short_recoveries, 2):
    test_combo(f'LONG_BASE+{"+".join(pair)}', long_base + list(pair))

# Add all short
test_combo('LONG_BASE+ALL_SHORT', long_base + short_recoveries)

# ── Add Mom_BUY to the mix ──
print("\n=== FULL STACK ===")
all_long = ['LC_confl90_c45', 'VIX_BUY_25_c55', 'Mom_BUY_SRet_p1_c55']
test_combo('ALL_LONG', all_long)
test_combo('ALL_LONG+ALL_SHORT', all_long + short_recoveries)

# ── Ultimate: everything ──
print("\n=== EVERYTHING ===")
all_keys = list(RECOVERIES.keys())
test_combo('ALL_9', all_keys)

# Try without the most dangerous ones
# Remove VIX_BUY_25_c55 (subsumed by 30) to check
test_combo('ALL_minus_VIX25', [k for k in all_keys if k != 'VIX_BUY_25_c55'])
test_combo('ALL_minus_L_confl90', [k for k in all_keys if k != 'L_confl90_c50'])

# ── Best combo detailed analysis ──
print("\n=== BEST COMBO ANALYSIS ===")
# Run the best combo and show train/test split by year
best_keys = all_long + short_recoveries  # or all_keys, depending on results
best_fn = make_combo(best_keys)
best_trades = simulate_trades(signals, best_fn, 'BEST', cooldown=0, trail_power=6)
n = len(best_trades)
wins = sum(1 for t in best_trades if t.pnl > 0)
print(f"\nBEST: {n} trades, {wins/n*100:.1f}% WR, ${sum(t.pnl for t in best_trades):+,.0f}")
print(f"  BL=${min(t.pnl for t in best_trades):+,.0f}")
print(f"  Longs: {sum(1 for t in best_trades if t.direction=='BUY')}, Shorts: {sum(1 for t in best_trades if t.direction=='SELL')}")

# Year-by-year
from collections import defaultdict
by_year = defaultdict(list)
for t in best_trades:
    by_year[t.entry_date.year].append(t)
print(f"\n  {'Year':>4} {'Trades':>6} {'WR':>5} {'PnL':>10}")
for year in sorted(by_year):
    yr_trades = by_year[year]
    yr_n = len(yr_trades)
    yr_w = sum(1 for t in yr_trades if t.pnl > 0)
    yr_pnl = sum(t.pnl for t in yr_trades)
    print(f"  {year:>4} {yr_n:>6} {yr_w/yr_n*100:>5.1f} ${yr_pnl:>+9,.0f}")

print("\nDone")
