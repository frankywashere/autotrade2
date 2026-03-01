#!/usr/bin/env python3
"""v30: Explore V5 bounce integration with CU.
CT+V5 = 389 trades, 99.5% WR (2 losses: -$31, -$6).
Can we filter V5 to 100% WR? Also explore remaining CS base gap."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _build_filter_cascade,
    _make_v28_cu, _SigProxy, _AnalysisProxy, _floor_stop_tp
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

cu_fn = _make_v28_cu(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50,
                      vix_map, spy_return_map, spy_ret_2d)
cu_trades = simulate_trades(signals, cu_fn, 'CU', cooldown=0, trail_power=6)
cu_dates = {t.entry_date for t in cu_trades}
day_map = {day.date: day for day in signals}
print(f"CU baseline: {len(cu_trades)} trades, {sum(1 for t in cu_trades if t.pnl>0)/len(cu_trades)*100:.1f}% WR, ${sum(t.pnl for t in cu_trades):+,.0f}")

# ── SECTION 1: V5 signal analysis ──
print("\n=== V5 BOUNCE SIGNAL ANALYSIS ===")
v5_signals = [(day.date, day) for day in signals if day.v5_take_bounce]
print(f"Total V5 signals: {len(v5_signals)}")

# V5 signals not in CU
v5_not_cu = [(dt, day) for dt, day in v5_signals if dt not in cu_dates]
print(f"V5 not in CU: {len(v5_not_cu)}")

# Profile V5 signals
print(f"\n  {'Date':12} {'V5conf':>6} {'CSact':>6} {'CSconf':>7} {'Health':>6} {'Pos':>5} {'Confl':>5} {'VIX':>5} {'SPY20':>6} {'DOW':>4}")
for dt, day in v5_signals:
    dd = dt.date() if hasattr(dt, 'date') else dt
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
    spy_d20 = spy_dist_map.get(dt, 999)
    vix = vix_map.get(dt, 22)
    in_cu = "CU" if dt in cu_dates else "  "
    print(f"  {str(dt)[:10]:12} {day.v5_confidence or 0:.3f} {day.cs_action:>6} {day.cs_confidence:7.3f} "
          f"{day.cs_channel_health:6.3f} {day.cs_position_score:5.2f} {day.cs_confluence_score:5.2f} "
          f"{vix:5.1f} {spy_d20:+6.2f} {dow:>4} {in_cu}")

# ── SECTION 2: CU + V5 integration ──
print("\n=== CU + V5 INTEGRATION ===")

def make_cu_plus_v5(v5_check=None):
    def fn(day):
        result = cu_fn(day)
        if result is not None: return result
        if day.v5_take_bounce:
            if v5_check is None or v5_check(day):
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn

# Raw V5 (no filter)
fn = make_cu_plus_v5()
trades = simulate_trades(signals, fn, 'CU+V5', cooldown=0, trail_power=6)
n = len(trades)
wins = sum(1 for t in trades if t.pnl > 0)
wr = wins / n * 100
bl = min(t.pnl for t in trades)
pnl = sum(t.pnl for t in trades)
new = [t for t in trades if t.entry_date not in cu_dates]
print(f"CU+V5 (raw): {n} trades (+{len(new)}), {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")
# Show the new V5 trades
for t in sorted(new, key=lambda x: x.pnl)[:5]:
    print(f"  {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+7,.0f}")

# Identify the losers
v5_losers = [t for t in new if t.pnl <= 0]
print(f"\nV5 losses ({len(v5_losers)}):")
for t in v5_losers:
    day = day_map.get(t.entry_date)
    if day:
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        print(f"  {str(t.entry_date)[:10]} ${t.pnl:+,.0f} v5c={day.v5_confidence or 0:.3f} "
              f"cs={day.cs_action} h={day.cs_channel_health:.3f} "
              f"pos={day.cs_position_score:.2f} confl={day.cs_confluence_score:.2f} "
              f"VIX={vix_map.get(day.date,22):.1f} SPY={spy_dist_map.get(day.date,999):+.2f} {dow}")

# ── SECTION 3: Filter V5 to 100% WR ──
print("\n=== V5 FILTERS FOR 100% WR ===")
v5_filters = [
    ('V5 SPY>0', lambda d: spy_dist_map.get(d.date, -999) >= 0),
    ('V5 VIX<20', lambda d: vix_map.get(d.date, 22) < 20),
    ('V5 VIX<25', lambda d: vix_map.get(d.date, 22) < 25),
    ('V5 h>=0.30', lambda d: d.cs_channel_health >= 0.30),
    ('V5 h>=0.20', lambda d: d.cs_channel_health >= 0.20),
    ('V5 notMon', lambda d: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 0),
    ('V5 notFri', lambda d: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 4),
    ('V5 notTue', lambda d: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 1),
    ('V5 pos<0.50', lambda d: d.cs_position_score < 0.50),
    ('V5 pos<0.30', lambda d: d.cs_position_score < 0.30),
    ('V5 confl>=0.50', lambda d: d.cs_confluence_score >= 0.50),
    ('V5 confl>=0.70', lambda d: d.cs_confluence_score >= 0.70),
    ('V5 v5conf>=0.60', lambda d: (d.v5_confidence or 0) >= 0.60),
    ('V5 v5conf>=0.50', lambda d: (d.v5_confidence or 0) >= 0.50),
    ('V5 v5conf>=0.55', lambda d: (d.v5_confidence or 0) >= 0.55),
    ('V5 cs=BUY', lambda d: d.cs_action == 'BUY'),
    ('V5 cs=BUY|HOLD', lambda d: d.cs_action in ('BUY', 'HOLD')),
    ('V5 cs!=SELL', lambda d: d.cs_action != 'SELL'),
    ('V5 SPY>0&h>=0.20', lambda d: spy_dist_map.get(d.date, -999) >= 0 and d.cs_channel_health >= 0.20),
    ('V5 SPY>0&notMon', lambda d: spy_dist_map.get(d.date, -999) >= 0 and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 0),
]

for name, check in v5_filters:
    fn = make_cu_plus_v5(check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 364: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cu_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:30s}: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}{marker}")

# ── SECTION 4: V5 with different trail powers ──
print("\n=== V5 TRAIL POWER ===")
fn_v5 = make_cu_plus_v5()
for power in [4, 5, 6, 7, 8, 10, 12, 15]:
    trades = simulate_trades(signals, fn_v5, f'p{power}', cooldown=0, trail_power=power)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    print(f"  power={power:2d}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# ── SECTION 5: V5 with wider stops ──
print("\n=== V5 WIDER STOPS ===")
for stop, tp in [(2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0)]:
    def make_v5_wider(s, t):
        def fn(day):
            result = cu_fn(day)
            if result is not None: return result
            if day.v5_take_bounce:
                return ('BUY', day.v5_confidence or 0.60, s, t, 'V5')
            return None
        return fn
    fn = make_v5_wider(stop, tp)
    trades = simulate_trades(signals, fn, f's{stop}', cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cu_dates])
    print(f"  stop={stop}%/tp={tp}%: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# ── SECTION 6: Remaining CS base gap analysis ──
print("\n=== REMAINING CS BASE GAP ===")
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

all_base = make_all_cs_base()
all_trades = simulate_trades(signals, all_base, 'allBase', cooldown=0, trail_power=6)
gap = [t for t in all_trades if t.entry_date not in cu_dates]
print(f"All base: {len(all_trades)} trades ({sum(1 for t in all_trades if t.pnl>0)}W/{sum(1 for t in all_trades if t.pnl<=0)}L)")
print(f"Not in CU: {len(gap)} trades ({sum(1 for t in gap if t.pnl>0)}W/{sum(1 for t in gap if t.pnl<=0)}L)")

if gap:
    print(f"\n  {'Date':12} {'Dir':5} {'PnL':>8} {'Health':>6} {'Pos':>5} {'Confl':>5} {'VIX':>5}")
    for t in sorted(gap, key=lambda x: x.pnl):
        day = day_map.get(t.entry_date)
        if day:
            vix = vix_map.get(day.date, 22)
            print(f"  {str(t.entry_date)[:10]:12} {t.direction:5} ${t.pnl:>+7,.0f} "
                  f"{day.cs_channel_health:6.3f} {day.cs_position_score:5.2f} {day.cs_confluence_score:5.2f} {vix:5.1f}")

print("\nDone")
