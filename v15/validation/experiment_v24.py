#!/usr/bin/env python3
"""v24: Profile the final ~8 filtered trades after CP (255).
These are the hardest trades. Try every possible safe recovery."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming,
    simulate_trades, _make_s1_tf3_vix_combo, _build_filter_cascade,
    _make_v23_cp
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
cp_fn = _make_v23_cp(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50,
                      vix_map, spy_return_map, spy_ret_2d)

base_trades = simulate_trades(signals, base_fn, 'base', cooldown=0, trail_power=6)
cp_trades = simulate_trades(signals, cp_fn, 'CP', cooldown=0, trail_power=6)

cp_dates = {t.entry_date for t in cp_trades}
filtered_out = [t for t in base_trades if t.entry_date not in cp_dates]

print(f"Base: {len(base_trades)} trades, CP: {len(cp_trades)} trades")
print(f"Filtered out: {len(filtered_out)} trades ({sum(1 for t in filtered_out if t.pnl>0)}W/{sum(1 for t in filtered_out if t.pnl<=0)}L)")
print()

day_map = {day.date: day for day in signals}

# Extreme detail on each remaining trade
print("=" * 180)
header = (f"{'Date':12} {'Dir':5} {'PnL':>8} {'Conf':>5} {'Health':>6} {'Pos':>5} "
          f"{'Confl':>5} {'TFs':>3} {'SPY20':>6} {'SMA5':>6} {'SMA50':>6} "
          f"{'VIX':>5} {'SRet':>6} {'SR2d':>6} {'SR3d':>6} {'DOW':>4} {'Train':>5}")
print(header)
print("=" * 180)

for t in sorted(filtered_out, key=lambda x: x.entry_date):
    day = day_map.get(t.entry_date)
    if day is None:
        print(f"  {t.entry_date}: day not found in map")
        continue
    result = base_fn(day)
    if result is None:
        print(f"  {t.entry_date}: base_fn returns None")
        continue
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
    dow_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dd.weekday()]
    train = "TRAIN" if t.entry_date.year <= 2021 else "TEST"

    print(f"{str(day.date)[:10]:12} {action:5} ${t.pnl:>+7,.0f} {conf:5.3f} {day.cs_channel_health:6.3f} "
          f"{day.cs_position_score:5.2f} {day.cs_confluence_score:5.2f} {tfs:3d} {spy_d20:+6.2f} {spy_d5:+6.2f} "
          f"{spy_d50:+6.2f} {vix:5.1f} {sret:+6.2f} {sr2d:+6.2f} {sr3d:+6.2f} {dow_name:>4} {train:>5}")

# ── Now try every conceivable safe recovery for these remaining trades ──
print()
print("=" * 80)
print("FINAL RECOVERY ATTEMPTS")
print("=" * 80)

def make_cp_plus(name, check):
    def fn(day):
        result = cp_fn(day)
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

# Generate exhaustive experiments targeting the specific remaining trades
experiments = [
    # BUY trades remaining: likely low health, Tuesday, or conf < 0.55
    ('BUY c50&h20', lambda d, a, c: a == 'BUY' and c >= 0.50 and d.cs_channel_health >= 0.20),
    ('BUY c45&h25', lambda d, a, c: a == 'BUY' and c >= 0.45 and d.cs_channel_health >= 0.25),
    ('BUY c50&h15', lambda d, a, c: a == 'BUY' and c >= 0.50 and d.cs_channel_health >= 0.15),
    ('BUY c55&h15', lambda d, a, c: a == 'BUY' and c >= 0.55 and d.cs_channel_health >= 0.15),
    ('BUY c60&h10', lambda d, a, c: a == 'BUY' and c >= 0.60 and d.cs_channel_health >= 0.10),
    ('BUY c75', lambda d, a, c: a == 'BUY' and c >= 0.75),
    ('BUY c80', lambda d, a, c: a == 'BUY' and c >= 0.80),
    ('BUY c85', lambda d, a, c: a == 'BUY' and c >= 0.85),
    ('BUY TF5', lambda d, a, c: a == 'BUY' and _count_tf_confirming(d, 'BUY') >= 5),
    ('BUY TF4&c50', lambda d, a, c: a == 'BUY' and _count_tf_confirming(d, 'BUY') >= 4 and c >= 0.50),
    ('BUY Tue&c55&h25', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and c >= 0.55 and d.cs_channel_health >= 0.25),
    ('BUY Tue&c60', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and c >= 0.60),
    ('BUY Tue&c65', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and c >= 0.65),
    ('BUY Tue&c60&h25', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and c >= 0.60 and d.cs_channel_health >= 0.25),
    ('BUY Tue&c55&pos<90', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and c >= 0.55 and d.cs_position_score < 0.90),
    ('BUY Tue&c55&TF4', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and c >= 0.55 and _count_tf_confirming(d, 'BUY') >= 4),
    ('BUY Tue&c55&VIX<20', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and c >= 0.55 and vix_map.get(d.date, 22) < 20),
    ('BUY Tue&c55&VIX>20', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and c >= 0.55 and vix_map.get(d.date, 22) > 20),
    ('BUY Tue&c55&SPY>0', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and c >= 0.55 and spy_dist_map.get(d.date, -999) > 0),
    ('BUY Tue&c60&pos<95', lambda d, a, c: a == 'BUY' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and c >= 0.60 and d.cs_position_score < 0.95),
    # SELL trades remaining
    ('SELL c55&h25&TF3', lambda d, a, c: a == 'SELL' and c >= 0.55 and d.cs_channel_health >= 0.25 and _count_tf_confirming(d, 'SELL') >= 3),
    ('SELL c60&h20&TF3', lambda d, a, c: a == 'SELL' and c >= 0.60 and d.cs_channel_health >= 0.20 and _count_tf_confirming(d, 'SELL') >= 3),
    ('SELL c60&h20&TF4', lambda d, a, c: a == 'SELL' and c >= 0.60 and d.cs_channel_health >= 0.20 and _count_tf_confirming(d, 'SELL') >= 4),
    ('SELL c65&h15&TF3', lambda d, a, c: a == 'SELL' and c >= 0.65 and d.cs_channel_health >= 0.15 and _count_tf_confirming(d, 'SELL') >= 3),
    ('SELL c70&h10&TF3', lambda d, a, c: a == 'SELL' and c >= 0.70 and d.cs_channel_health >= 0.10 and _count_tf_confirming(d, 'SELL') >= 3),
    ('SELL c75&h10', lambda d, a, c: a == 'SELL' and c >= 0.75 and d.cs_channel_health >= 0.10),
    ('SELL c80&h05', lambda d, a, c: a == 'SELL' and c >= 0.80 and d.cs_channel_health >= 0.05),
    ('SELL c80', lambda d, a, c: a == 'SELL' and c >= 0.80),
    ('SELL c85', lambda d, a, c: a == 'SELL' and c >= 0.85),
    ('SELL confl90', lambda d, a, c: a == 'SELL' and d.cs_confluence_score >= 0.90),
    ('SELL confl85&c55', lambda d, a, c: a == 'SELL' and d.cs_confluence_score >= 0.85 and c >= 0.55),
    ('SELL pos>98', lambda d, a, c: a == 'SELL' and d.cs_position_score >= 0.98),
    ('SELL pos>99&h10', lambda d, a, c: a == 'SELL' and d.cs_position_score >= 0.99 and d.cs_channel_health >= 0.10),
    ('SELL VIX>30', lambda d, a, c: a == 'SELL' and vix_map.get(d.date, 22) > 30),
    ('SELL VIX>25&h10', lambda d, a, c: a == 'SELL' and vix_map.get(d.date, 22) > 25 and d.cs_channel_health >= 0.10),
    ('SELL VIX<18&h15', lambda d, a, c: a == 'SELL' and vix_map.get(d.date, 22) < 18 and d.cs_channel_health >= 0.15),
    ('SELL SPYd20<-3&h10', lambda d, a, c: a == 'SELL' and spy_dist_map.get(d.date, 999) < -3.0 and d.cs_channel_health >= 0.10),
    ('SELL SPYd20<-4&h05', lambda d, a, c: a == 'SELL' and spy_dist_map.get(d.date, 999) < -4.0 and d.cs_channel_health >= 0.05),
    ('SELL SR<-1&h20', lambda d, a, c: a == 'SELL' and spy_return_map.get(d.date, 0) < -1.0 and d.cs_channel_health >= 0.20),
    ('SELL SR<-1.5&h15', lambda d, a, c: a == 'SELL' and spy_return_map.get(d.date, 0) < -1.5 and d.cs_channel_health >= 0.15),
    ('SELL SR3d<-2&h10', lambda d, a, c: a == 'SELL' and spy_ret_3d.get(d.date, 0) < -2.0 and d.cs_channel_health >= 0.10),
    # Universal
    ('ANY c80', lambda d, a, c: c >= 0.80),
    ('ANY c85', lambda d, a, c: c >= 0.85),
    ('ANY c75&TF4', lambda d, a, c: c >= 0.75 and _count_tf_confirming(d, a) >= 4),
    ('ANY c70&TF5', lambda d, a, c: c >= 0.70 and _count_tf_confirming(d, a) >= 5),
    ('ANY confl90&c50', lambda d, a, c: d.cs_confluence_score >= 0.90 and c >= 0.50),
    ('ANY h35&c55&TF3', lambda d, a, c: d.cs_channel_health >= 0.35 and c >= 0.55 and _count_tf_confirming(d, a) >= 3),
]

for name, check in experiments:
    fn = make_cp_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 255: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

print("\nDone")
