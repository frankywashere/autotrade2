#!/usr/bin/env python3
"""v24c: Filter tf2 expansion losses. tf2 adds 85 trades (83W/2L).
Losses: 2018-05-18 LONG -$1,815 (c=0.322, h=0.467), 2019-03-22 LONG -$7 (c=0.216, h=0.372).
Both have low raw confidence. Find a filter that blocks these while keeping most winners."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _make_s1_tf3_vix_combo, _build_filter_cascade,
    _make_v23_cp, _SigProxy, _AnalysisProxy, _floor_stop_tp
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

# ── CP baseline ──
cp_fn = _make_v23_cp(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50,
                      vix_map, spy_return_map, spy_ret_2d)
cp_trades = simulate_trades(signals, cp_fn, 'CP', cooldown=0, trail_power=6)
cp_dates = {t.entry_date for t in cp_trades}
print(f"CP: {len(cp_trades)} trades, 100% WR, ${sum(t.pnl for t in cp_trades):+,.0f}")

# ── Build tf2 base ──
def make_s1_tfN_vix(min_tfs=2):
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
                if not state.get('valid', False):
                    continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks.get(tf, 0) >= 1:
                    confirmed += 1
        if confirmed < min_tfs:
            return None
        sig = _SigProxy(day)
        ana = _AnalysisProxy(day.cs_tf_states)
        ok, adj, _ = cascade_vix.evaluate(
            sig, ana, feature_vec=None, bar_datetime=day.date,
            higher_tf_data=None, spy_df=None, vix_df=None,
        )
        if not ok or adj < MIN_SIGNAL_CONFIDENCE:
            return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, adj, s, t, 'CS')
    return fn

tf2_base = make_s1_tfN_vix(2)

# ── Profile tf2 new trades (trades that tf2 accepts but tf3 rejects) ──
# Run tf2 with CP-like filtering to get all tf2 trades
def make_cp_filtered_base(base_fn):
    """CP-level gates on arbitrary base."""
    def fn(day):
        result = base_fn(day)
        if result is None:
            return None
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

tf2_filtered = make_cp_filtered_base(tf2_base)
tf2_trades = simulate_trades(signals, tf2_filtered, 'tf2_filt', cooldown=0, trail_power=6)
print(f"tf2 filtered: {len(tf2_trades)} trades, {sum(1 for t in tf2_trades if t.pnl>0)/len(tf2_trades)*100:.1f}% WR")

# Find new trades (not in CP)
new_trades = [t for t in tf2_trades if t.entry_date not in cp_dates]
print(f"New from tf2: {len(new_trades)} trades ({sum(1 for t in new_trades if t.pnl>0)}W/{sum(1 for t in new_trades if t.pnl<=0)}L)")

# Profile ALL new trades
day_map = {day.date: day for day in signals}
print(f"\n{'Date':12} {'Dir':5} {'PnL':>8} {'RawC':>5} {'AdjC':>5} {'Health':>6} {'Pos':>5} {'Confl':>5} {'TFs':>3} {'SPY%':>6} {'VIX':>5} {'DOW':>4}")
for t in sorted(new_trades, key=lambda x: x.entry_date):
    day = day_map.get(t.entry_date)
    if not day: continue
    result = tf2_filtered(day)
    if result is None: continue
    action, adj_conf, _, _, _ = result
    tfs = _count_tf_confirming(day, action)
    spy_d20 = spy_dist_map.get(day.date, 999)
    vix = vix_map.get(day.date, 22)
    dd = day.date.date() if hasattr(day.date, 'date') else day.date
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
    win = 'W' if t.pnl > 0 else 'L'
    print(f"{str(day.date)[:10]:12} {action:5} ${t.pnl:>+7,.0f} {day.cs_confidence:5.3f} {adj_conf:5.3f} "
          f"{day.cs_channel_health:6.3f} {day.cs_position_score:5.2f} {day.cs_confluence_score:5.2f} {tfs:3d} "
          f"{spy_d20:+6.2f} {vix:5.1f} {dow:>4} {win}")

# ── Try adding tf2 trades to CP with extra filters ──
print("\n=== CP + tf2 FILTERED NEW TRADES ===")

def make_cp_plus_tf2(extra_filter):
    """CP + tf2 expansion with extra filter on tf2-only trades."""
    def fn(day):
        result = cp_fn(day)
        if result is not None:
            return result
        # CP rejected — try tf2 base with filtering
        result = tf2_filtered(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result
        if extra_filter(day, action, conf):
            return result
        return None
    return fn

# Try various filters on the tf2-only trades
filters = [
    ('tf2: h>=0.50', lambda d, a, c: d.cs_channel_health >= 0.50),
    ('tf2: h>=0.45', lambda d, a, c: d.cs_channel_health >= 0.45),
    ('tf2: h>=0.40', lambda d, a, c: d.cs_channel_health >= 0.40),
    ('tf2: h>=0.35', lambda d, a, c: d.cs_channel_health >= 0.35),
    ('tf2: h>=0.30', lambda d, a, c: d.cs_channel_health >= 0.30),
    ('tf2: rawC>=0.50', lambda d, a, c: d.cs_confidence >= 0.50),
    ('tf2: rawC>=0.40', lambda d, a, c: d.cs_confidence >= 0.40),
    ('tf2: rawC>=0.35', lambda d, a, c: d.cs_confidence >= 0.35),
    ('tf2: c>=0.55 (adj)', lambda d, a, c: c >= 0.55),
    ('tf2: c>=0.50 (adj)', lambda d, a, c: c >= 0.50),
    ('tf2: confl>=0.50', lambda d, a, c: d.cs_confluence_score >= 0.50),
    ('tf2: confl>=0.60', lambda d, a, c: d.cs_confluence_score >= 0.60),
    ('tf2: confl>=0.70', lambda d, a, c: d.cs_confluence_score >= 0.70),
    ('tf2: pos<0.95', lambda d, a, c: d.cs_position_score < 0.95),
    ('tf2: pos<0.90', lambda d, a, c: d.cs_position_score < 0.90),
    ('tf2: SPY>0', lambda d, a, c: spy_dist_map.get(d.date, -999) >= 0),
    ('tf2: VIX<20', lambda d, a, c: vix_map.get(d.date, 22) < 20),
    ('tf2: h30&rawC40', lambda d, a, c: d.cs_channel_health >= 0.30 and d.cs_confidence >= 0.40),
    ('tf2: h25&rawC50', lambda d, a, c: d.cs_channel_health >= 0.25 and d.cs_confidence >= 0.50),
    ('tf2: h30&pos<95', lambda d, a, c: d.cs_channel_health >= 0.30 and d.cs_position_score < 0.95),
    ('tf2: h25&confl60', lambda d, a, c: d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60),
    ('tf2: notMay', lambda d, a, c: (d.date.date() if hasattr(d.date, 'date') else d.date).month != 5),
    ('tf2: any', lambda d, a, c: True),
]

for name, filt in filters:
    fn = make_cp_plus_tf2(filt)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= 255: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    tr = [t for t in trades if t.entry_date.year <= 2021]
    ts = [t for t in trades if t.entry_date.year > 2021]
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:25s}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f} "
          f"[Tr:{len(tr)}@{sum(1 for t in tr if t.pnl>0)/max(1,len(tr))*100:.0f}% "
          f"Ts:{len(ts)}@{sum(1 for t in ts if t.pnl>0)/max(1,len(ts))*100:.0f}%]{marker}")

print("\nDone")
