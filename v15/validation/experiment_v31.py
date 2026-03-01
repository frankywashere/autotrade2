#!/usr/bin/env python3
"""v31: Two frontiers to expand CU (364, 100% WR):
1. V5 bounce: CU+V5 = 407 trades, 99.5% WR. 2 losses (-$31, -$6).
   Try multi-condition filters to eliminate BOTH losses.
2. Base gap: 54 trades not in CU (42W/12L). Profile losers deeply,
   find multi-condition separators."""
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

# ══════════════════════════════════════════════════════════
# FRONTIER 1: V5 BOUNCE LOSS ELIMINATION
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FRONTIER 1: V5 BOUNCE LOSS ELIMINATION")
print("="*70)

# V5 losses from v30:
# 2017-10-23: -$31, v5c=0.660, cs=HOLD, h=0.506, pos=?, confl=?, VIX=?, Mon
# 2019-04-26: -$6, v5c=0.975, cs=HOLD, h=0.280, pos=?, confl=?, VIX=?, Fri

# First, deep profile ALL V5 signals
print("\n=== V5 SIGNAL DEEP PROFILE ===")
v5_days = [(day.date, day) for day in signals if day.v5_take_bounce]
print(f"Total V5 signals: {len(v5_days)}")

# Run CU+V5 raw to identify the 2 losers
def make_cu_v5(v5_check=None):
    def fn(day):
        result = cu_fn(day)
        if result is not None: return result
        if day.v5_take_bounce:
            if v5_check is None or v5_check(day):
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn

raw_fn = make_cu_v5()
raw_trades = simulate_trades(signals, raw_fn, 'CU+V5', cooldown=0, trail_power=6)
v5_new = [t for t in raw_trades if t.entry_date not in cu_dates]
v5_losers = [t for t in v5_new if t.pnl <= 0]
v5_winners = [t for t in v5_new if t.pnl > 0]
print(f"CU+V5 raw: {len(raw_trades)} trades (+{len(v5_new)} new), {sum(1 for t in raw_trades if t.pnl>0)/len(raw_trades)*100:.1f}% WR")
print(f"V5 new: {len(v5_winners)}W/{len(v5_losers)}L")

# Deep profile losers
print(f"\n--- V5 LOSERS ---")
for t in v5_losers:
    day = day_map.get(t.entry_date)
    if day:
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        spy_d5 = spy_dist_5.get(day.date, 999)
        spy_d20 = spy_dist_map.get(day.date, 999)
        spy_d50 = spy_dist_50.get(day.date, 999)
        vix = vix_map.get(day.date, 22)
        spy_ret = spy_return_map.get(day.date, 0)
        spy_r2d = spy_ret_2d.get(day.date, 0)
        tf_count = _count_tf_confirming(day, 'BUY')
        print(f"  {str(t.entry_date)[:10]} ${t.pnl:+,.0f} v5c={day.v5_confidence or 0:.3f} "
              f"cs={day.cs_action} h={day.cs_channel_health:.3f} "
              f"pos={day.cs_position_score:.2f} confl={day.cs_confluence_score:.2f} "
              f"VIX={vix:.1f} SPY5={spy_d5:+.2f} SPY20={spy_d20:+.2f} SPY50={spy_d50:+.2f} "
              f"SRet={spy_ret:+.2f} SRet2d={spy_r2d:+.2f} TFs={tf_count} {dow}")

# Deep profile a sample of winners for comparison
print(f"\n--- V5 WINNERS (sorted by PnL) ---")
for t in sorted(v5_winners, key=lambda x: x.pnl)[:10]:
    day = day_map.get(t.entry_date)
    if day:
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        vix = vix_map.get(day.date, 22)
        spy_d20 = spy_dist_map.get(day.date, 999)
        tf_count = _count_tf_confirming(day, 'BUY')
        print(f"  {str(t.entry_date)[:10]} ${t.pnl:+,.0f} v5c={day.v5_confidence or 0:.3f} "
              f"cs={day.cs_action} h={day.cs_channel_health:.3f} "
              f"pos={day.cs_position_score:.2f} confl={day.cs_confluence_score:.2f} "
              f"VIX={vix:.1f} SPY20={spy_d20:+.2f} TFs={tf_count} {dow}")

# ── Multi-condition V5 filters ──
print("\n=== MULTI-CONDITION V5 FILTERS ===")

# Build feature vectors for V5 signals
v5_features = {}
for dt, day in v5_days:
    dd = dt.date() if hasattr(dt, 'date') else dt
    v5_features[dt] = {
        'v5c': day.v5_confidence or 0,
        'cs_action': day.cs_action,
        'h': day.cs_channel_health,
        'pos': day.cs_position_score,
        'confl': day.cs_confluence_score,
        'vix': vix_map.get(dt, 22),
        'spy5': spy_dist_5.get(dt, 0),
        'spy20': spy_dist_map.get(dt, 0),
        'spy50': spy_dist_50.get(dt, 0),
        'sret': spy_return_map.get(dt, 0),
        'sret2d': spy_ret_2d.get(dt, 0),
        'dow': dd.weekday(),
        'tf_count': _count_tf_confirming(day, 'BUY'),
        'conf': day.cs_confidence,
    }

# Systematic multi-condition filter search
filters = {
    'notFri': lambda d: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 4,
    'notMon': lambda d: (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() != 0,
    'h>=0.30': lambda d: d.cs_channel_health >= 0.30,
    'h>=0.35': lambda d: d.cs_channel_health >= 0.35,
    'h>=0.40': lambda d: d.cs_channel_health >= 0.40,
    'h<0.50': lambda d: d.cs_channel_health < 0.50,
    'pos<0.50': lambda d: d.cs_position_score < 0.50,
    'pos<0.80': lambda d: d.cs_position_score < 0.80,
    'pos<0.30': lambda d: d.cs_position_score < 0.30,
    'confl>=0.50': lambda d: d.cs_confluence_score >= 0.50,
    'confl>=0.70': lambda d: d.cs_confluence_score >= 0.70,
    'VIX<20': lambda d: vix_map.get(d.date, 22) < 20,
    'VIX<25': lambda d: vix_map.get(d.date, 22) < 25,
    'VIX<30': lambda d: vix_map.get(d.date, 22) < 30,
    'SPY20>=0': lambda d: spy_dist_map.get(d.date, -999) >= 0,
    'SPY5>=0': lambda d: spy_dist_5.get(d.date, -999) >= 0,
    'v5c>=0.60': lambda d: (d.v5_confidence or 0) >= 0.60,
    'v5c>=0.70': lambda d: (d.v5_confidence or 0) >= 0.70,
    'v5c>=0.80': lambda d: (d.v5_confidence or 0) >= 0.80,
    'TF>=1': lambda d: _count_tf_confirming(d, 'BUY') >= 1,
    'TF>=2': lambda d: _count_tf_confirming(d, 'BUY') >= 2,
    'cs!=SELL': lambda d: d.cs_action != 'SELL',
    'SRet>=0': lambda d: spy_return_map.get(d.date, 0) >= 0,
    'SRet>-0.5': lambda d: spy_return_map.get(d.date, 0) > -0.5,
}

# Try all pairs of filters
filter_names = list(filters.keys())
print(f"\nTesting {len(filter_names)} single filters and {len(filter_names)*(len(filter_names)-1)//2} pairs...")

# Single filters first
single_results = []
for name, filt in filters.items():
    fn = make_cu_v5(filt)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    new = len([t for t in trades if t.entry_date not in cu_dates])
    bl = min(t.pnl for t in trades)
    if n > len(cu_trades):
        single_results.append((name, n, new, wr, bl))

single_results.sort(key=lambda x: (-x[3], -x[1]))
print(f"\n--- Single V5 filters (100% WR only) ---")
for name, n, new, wr, bl in single_results:
    if wr >= 100:
        print(f"  {name:20s}: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, BL=${bl:+,.0f} ***")
print(f"\n--- Single V5 filters (>99% WR, not 100%) ---")
for name, n, new, wr, bl in single_results:
    if 99 <= wr < 100:
        print(f"  {name:20s}: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, BL=${bl:+,.0f}")

# Now test pairs
pair_results = []
for i in range(len(filter_names)):
    for j in range(i+1, len(filter_names)):
        n1, n2 = filter_names[i], filter_names[j]
        f1, f2 = filters[n1], filters[n2]
        def make_pair(ff1, ff2):
            return lambda d: ff1(d) and ff2(d)
        fn = make_cu_v5(make_pair(f1, f2))
        trades = simulate_trades(signals, fn, f'{n1}&{n2}', cooldown=0, trail_power=6)
        n = len(trades)
        if n <= len(cu_trades): continue
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / n * 100
        new = len([t for t in trades if t.entry_date not in cu_dates])
        bl = min(t.pnl for t in trades)
        if wr >= 100:
            pair_results.append((f'{n1} & {n2}', n, new, wr, bl))

pair_results.sort(key=lambda x: -x[1])
print(f"\n--- Pair V5 filters (100% WR) - top 20 by trade count ---")
for name, n, new, wr, bl in pair_results[:20]:
    print(f"  {name:40s}: {n:3d} trades (+{new:2d}), BL=${bl:+,.0f} ***")

# Try best triples
if pair_results:
    print(f"\n=== TRIPLE V5 FILTERS (extending best pairs) ===")
    # Take top 5 pairs and add each remaining filter
    top_pairs = pair_results[:5]
    triple_results = []
    for pair_name, _, _, _, _ in top_pairs:
        parts = pair_name.split(' & ')
        f1, f2 = filters[parts[0]], filters[parts[1]]
        for n3 in filter_names:
            if n3 in parts: continue
            f3 = filters[n3]
            def make_triple(ff1, ff2, ff3):
                return lambda d: ff1(d) and ff2(d) and ff3(d)
            fn = make_cu_v5(make_triple(f1, f2, f3))
            trades = simulate_trades(signals, fn, f'triple', cooldown=0, trail_power=6)
            n = len(trades)
            if n <= len(cu_trades): continue
            wins = sum(1 for t in trades if t.pnl > 0)
            wr = wins / n * 100
            new = len([t for t in trades if t.entry_date not in cu_dates])
            bl = min(t.pnl for t in trades)
            if wr >= 100:
                triple_results.append((f'{pair_name} & {n3}', n, new, wr, bl))

    triple_results.sort(key=lambda x: -x[1])
    seen = set()
    print(f"--- Triple V5 filters (100% WR) - top 15 by trade count ---")
    for name, n, new, wr, bl in triple_results[:15]:
        if n not in seen or True:  # show all
            print(f"  {name:55s}: {n:3d} trades (+{new:2d}), BL=${bl:+,.0f} ***")
            seen.add(n)

# ── V5 with different stops/trails ──
print("\n=== V5 STOP/TRAIL VARIATIONS ===")
# Maybe wider stops help the tiny losses
for stop, tp in [(2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (2.5, 5.0), (1.5, 3.0)]:
    def make_v5_stops(s, t):
        def fn(day):
            result = cu_fn(day)
            if result is not None: return result
            if day.v5_take_bounce:
                return ('BUY', day.v5_confidence or 0.60, s, t, 'V5')
            return None
        return fn
    fn = make_v5_stops(stop, tp)
    trades = simulate_trades(signals, fn, f's{stop}/t{tp}', cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    new = len([t for t in trades if t.entry_date not in cu_dates])
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    print(f"  stop={stop}%/tp={tp}%: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# V5 with higher trail power (tighter trail = exit faster)
for power in [6, 8, 10, 12, 15, 20]:
    fn = make_cu_v5()
    trades = simulate_trades(signals, fn, f'p{power}', cooldown=0, trail_power=power)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cu_dates])
    print(f"  trail_power={power:2d}: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")


# ══════════════════════════════════════════════════════════
# FRONTIER 2: BASE GAP RECOVERY
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FRONTIER 2: BASE GAP RECOVERY")
print("="*70)

# Get all base signals that fail everything
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
gap_wins = [t for t in gap if t.pnl > 0]
gap_losses = [t for t in gap if t.pnl <= 0]
print(f"All base: {len(all_trades)} trades ({sum(1 for t in all_trades if t.pnl>0)}W/{sum(1 for t in all_trades if t.pnl<=0)}L)")
print(f"Not in CU: {len(gap)} trades ({len(gap_wins)}W/{len(gap_losses)}L)")

# Deep profile gap trades
print(f"\n--- GAP LOSSES ({len(gap_losses)}) ---")
for t in sorted(gap_losses, key=lambda x: x.pnl):
    day = day_map.get(t.entry_date)
    if day:
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        vix = vix_map.get(day.date, 22)
        spy_d5 = spy_dist_5.get(day.date, 0)
        spy_d20 = spy_dist_map.get(day.date, 0)
        spy_d50 = spy_dist_50.get(day.date, 0)
        sret = spy_return_map.get(day.date, 0)
        tf_buy = _count_tf_confirming(day, 'BUY')
        tf_sell = _count_tf_confirming(day, 'SELL')
        tf = tf_buy if day.cs_action == 'BUY' else tf_sell
        train = "TRAIN" if t.entry_date.year <= 2021 else "TEST"
        print(f"  {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+7,.0f} "
              f"h={day.cs_channel_health:.3f} conf={day.cs_confidence:.3f} "
              f"confl={day.cs_confluence_score:.2f} pos={day.cs_position_score:.2f} "
              f"VIX={vix:5.1f} SPY5={spy_d5:+.2f} SPY20={spy_d20:+.2f} SPY50={spy_d50:+.2f} "
              f"SRet={sret:+.2f} TFs={tf} {dow} {train}")

print(f"\n--- GAP WINNERS ({len(gap_wins)}) ---")
for t in sorted(gap_wins, key=lambda x: x.pnl):
    day = day_map.get(t.entry_date)
    if day:
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        vix = vix_map.get(day.date, 22)
        spy_d20 = spy_dist_map.get(day.date, 0)
        tf_buy = _count_tf_confirming(day, 'BUY')
        tf_sell = _count_tf_confirming(day, 'SELL')
        tf = tf_buy if day.cs_action == 'BUY' else tf_sell
        train = "TRAIN" if t.entry_date.year <= 2021 else "TEST"
        print(f"  {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+7,.0f} "
              f"h={day.cs_channel_health:.3f} conf={day.cs_confidence:.3f} "
              f"confl={day.cs_confluence_score:.2f} pos={day.cs_position_score:.2f} "
              f"VIX={vix:5.1f} SPY20={spy_d20:+.2f} TFs={tf} {dow} {train}")

# ── Statistical comparison: gap wins vs gap losses ──
print("\n=== STATISTICAL SEPARATION ===")
win_feats = defaultdict(list)
loss_feats = defaultdict(list)
for t in gap:
    day = day_map.get(t.entry_date)
    if not day: continue
    dd = day.date.date() if hasattr(day.date, 'date') else day.date
    feats = {
        'h': day.cs_channel_health,
        'conf': day.cs_confidence,
        'confl': day.cs_confluence_score,
        'pos': day.cs_position_score,
        'vix': vix_map.get(day.date, 22),
        'spy5': spy_dist_5.get(day.date, 0),
        'spy20': spy_dist_map.get(day.date, 0),
        'spy50': spy_dist_50.get(day.date, 0),
        'sret': spy_return_map.get(day.date, 0),
        'sret2d': spy_ret_2d.get(day.date, 0),
        'dow': dd.weekday(),
        'tf': _count_tf_confirming(day, day.cs_action),
    }
    target = win_feats if t.pnl > 0 else loss_feats
    for k, v in feats.items():
        target[k].append(v)

print(f"{'Feature':10s} {'Win mean':>10} {'Win med':>10} {'Loss mean':>10} {'Loss med':>10} {'Sep?':>5}")
for feat in ['h', 'conf', 'confl', 'pos', 'vix', 'spy5', 'spy20', 'spy50', 'sret', 'sret2d', 'dow', 'tf']:
    wm = np.mean(win_feats[feat]) if win_feats[feat] else 0
    wmed = np.median(win_feats[feat]) if win_feats[feat] else 0
    lm = np.mean(loss_feats[feat]) if loss_feats[feat] else 0
    lmed = np.median(loss_feats[feat]) if loss_feats[feat] else 0
    sep = "YES" if abs(wm - lm) > 0.1 * max(abs(wm), abs(lm), 0.01) else "no"
    print(f"  {feat:10s} {wm:10.3f} {wmed:10.3f} {lm:10.3f} {lmed:10.3f} {sep:>5}")

# ── Try multi-condition base gap filters ──
print("\n=== BASE GAP MULTI-CONDITION FILTERS ===")

def make_cu_plus_gap(gap_check):
    """CU + filtered gap recovery."""
    def fn(day):
        result = cu_fn(day)
        if result is not None: return result
        result = all_base(day)
        if result is None: return None
        action, conf, s_pct, t_pct, src = result
        if gap_check(day, action, conf):
            return result
        return None
    return fn

# Try various gap recovery conditions
gap_filters = [
    # Direction-specific
    ('BUY h>=0.30', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.30),
    ('BUY h>=0.25', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.25),
    ('BUY h>=0.20', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.20),
    ('BUY h>=0.30&confl>=0.50', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.50),
    ('BUY h>=0.30&confl>=0.60', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.60),
    ('BUY h>=0.30&confl>=0.70', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.70),
    ('BUY h>=0.25&confl>=0.70', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.70),
    ('BUY h>=0.25&confl>=0.80', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.80),
    ('BUY confl>=0.70', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.70),
    ('BUY confl>=0.75', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.75),
    ('BUY VIX<20&h>=0.25', lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) < 20 and d.cs_channel_health >= 0.25),
    ('BUY SPY20>=0&h>=0.25', lambda d, a, c: a == 'BUY' and spy_dist_map.get(d.date, -999) >= 0 and d.cs_channel_health >= 0.25),
    ('BUY SPY5>=0&h>=0.25', lambda d, a, c: a == 'BUY' and spy_dist_5.get(d.date, -999) >= 0 and d.cs_channel_health >= 0.25),
    ('SELL h>=0.30', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.30),
    ('SELL h>=0.25', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.25),
    ('SELL h>=0.30&confl>=0.50', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.50),
    ('SELL h>=0.30&confl>=0.60', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.60),
    ('SELL h>=0.30&pos>=0.80', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.30 and d.cs_position_score >= 0.80),
    ('SELL h>=0.25&confl>=0.70', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.70),
    # Both directions
    ('h>=0.35&confl>=0.50', lambda d, a, c: d.cs_channel_health >= 0.35 and d.cs_confluence_score >= 0.50),
    ('h>=0.30&confl>=0.60', lambda d, a, c: d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.60),
    ('h>=0.30&confl>=0.70', lambda d, a, c: d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.70),
    ('h>=0.25&confl>=0.80', lambda d, a, c: d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.80),
    ('h>=0.35&c>=0.60', lambda d, a, c: d.cs_channel_health >= 0.35 and c >= 0.60),
    ('h>=0.30&c>=0.65', lambda d, a, c: d.cs_channel_health >= 0.30 and c >= 0.65),
    ('h>=0.30&c>=0.70', lambda d, a, c: d.cs_channel_health >= 0.30 and c >= 0.70),
    # Combined with VIX
    ('h>=0.30&VIX<20', lambda d, a, c: d.cs_channel_health >= 0.30 and vix_map.get(d.date, 22) < 20),
    ('h>=0.30&VIX<25', lambda d, a, c: d.cs_channel_health >= 0.30 and vix_map.get(d.date, 22) < 25),
    # Combined with SPY
    ('h>=0.30&SPY20>=0', lambda d, a, c: d.cs_channel_health >= 0.30 and spy_dist_map.get(d.date, -999) >= 0),
    ('h>=0.25&SPY20>=0', lambda d, a, c: d.cs_channel_health >= 0.25 and spy_dist_map.get(d.date, -999) >= 0),
    # Triple conditions
    ('BUY h>=0.25&confl>=0.60&SPY>=0', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.60 and spy_dist_map.get(d.date, -999) >= 0),
    ('BUY h>=0.25&c>=0.60&VIX<25', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.25 and c >= 0.60 and vix_map.get(d.date, 22) < 25),
    ('h>=0.30&confl>=0.50&SPY>=0', lambda d, a, c: d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.50 and spy_dist_map.get(d.date, -999) >= 0),
    ('h>=0.30&confl>=0.50&VIX<25', lambda d, a, c: d.cs_channel_health >= 0.30 and d.cs_confluence_score >= 0.50 and vix_map.get(d.date, 22) < 25),
    ('h>=0.35&confl>=0.60&c>=0.55', lambda d, a, c: d.cs_channel_health >= 0.35 and d.cs_confluence_score >= 0.60 and c >= 0.55),
    # Broader: any direction, weaker conditions
    ('confl>=0.85', lambda d, a, c: d.cs_confluence_score >= 0.85),
    ('confl>=0.80&h>=0.30', lambda d, a, c: d.cs_confluence_score >= 0.80 and d.cs_channel_health >= 0.30),
    ('confl>=0.80&h>=0.25', lambda d, a, c: d.cs_confluence_score >= 0.80 and d.cs_channel_health >= 0.25),
    ('c>=0.70&h>=0.30', lambda d, a, c: c >= 0.70 and d.cs_channel_health >= 0.30),
    ('c>=0.65&h>=0.35', lambda d, a, c: c >= 0.65 and d.cs_channel_health >= 0.35),
]

print(f"{'Filter':45s} {'Trades':>6} {'New':>4} {'WR':>6} {'PnL':>10} {'BL':>8}")
for name, check in gap_filters:
    fn = make_cu_plus_gap(check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= len(cu_trades): continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cu_dates])
    marker = " ***" if wr >= 100 else ""
    print(f"  {name:45s} {n:6d} {new:4d} {wr:5.1f}% ${pnl:+9,.0f} ${bl:+7,.0f}{marker}")

# ── Show detail for any 100% WR gap filters ──
print("\n=== DETAIL: BEST GAP RECOVERY (100% WR) ===")
best_gap_100 = []
for name, check in gap_filters:
    fn = make_cu_plus_gap(check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    if n <= len(cu_trades): continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    if wr >= 100:
        new_trades = [t for t in trades if t.entry_date not in cu_dates]
        best_gap_100.append((name, n, new_trades))

best_gap_100.sort(key=lambda x: -x[1])
for name, n, new_trades in best_gap_100[:5]:
    pnl = sum(t.pnl for t in new_trades)
    print(f"\n  {name}: {n} trades (+{len(new_trades)}), new PnL=${pnl:+,.0f}")
    for t in sorted(new_trades, key=lambda x: x.entry_date):
        day = day_map.get(t.entry_date)
        if day:
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
            train = "TRAIN" if t.entry_date.year <= 2021 else "TEST"
            print(f"    {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+7,.0f} "
                  f"h={day.cs_channel_health:.3f} confl={day.cs_confluence_score:.2f} "
                  f"conf={day.cs_confidence:.3f} pos={day.cs_position_score:.2f} "
                  f"VIX={vix_map.get(day.date,22):.1f} {dow} {train}")

# ══════════════════════════════════════════════════════════
# FRONTIER 3: COMBINED — CU + V5 + GAP RECOVERY
# ══════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FRONTIER 3: COMBINED CU + V5 + GAP")
print("="*70)

# If we find good V5 filters AND gap recovery, combine them
# For now, test: CU + best V5 filter + best gap recovery

# Also test: what if we use V5 with wider stop to avoid the tiny losses?
print("\n=== CU + V5(wider stops) ===")
for stop, tp in [(3.0, 6.0), (4.0, 8.0), (5.0, 10.0)]:
    def make_cu_v5_wide(s, t):
        def fn(day):
            result = cu_fn(day)
            if result is not None: return result
            if day.v5_take_bounce:
                return ('BUY', day.v5_confidence or 0.60, s, t, 'V5')
            return None
        return fn
    fn = make_cu_v5_wide(stop, tp)
    trades = simulate_trades(signals, fn, f'CU+V5 s{stop}/t{tp}', cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cu_dates])
    losses = [t for t in trades if t.entry_date not in cu_dates and t.pnl <= 0]
    print(f"  s{stop}/t{tp}: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}, losses={len(losses)}")
    for lt in losses:
        print(f"    LOSS: {str(lt.entry_date)[:10]} ${lt.pnl:+,.0f}")

# V5 with higher confidence for entry (boost confidence to reduce trail)
print("\n=== CU + V5 WITH BOOSTED CONFIDENCE ===")
for boost_conf in [0.70, 0.80, 0.90, 0.95]:
    def make_cu_v5_conf(bc):
        def fn(day):
            result = cu_fn(day)
            if result is not None: return result
            if day.v5_take_bounce:
                return ('BUY', bc, 2.0, 4.0, 'V5')
            return None
        return fn
    fn = make_cu_v5_conf(boost_conf)
    trades = simulate_trades(signals, fn, f'V5c={boost_conf}', cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    new = len([t for t in trades if t.entry_date not in cu_dates])
    losses = [t for t in trades if t.entry_date not in cu_dates and t.pnl <= 0]
    print(f"  conf={boost_conf}: {n:3d} trades (+{new:2d}), {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}, losses={len(losses)}")

print("\nDone")
