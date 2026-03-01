#!/usr/bin/env python3
"""v22: Profile remaining ~25 filtered trades. CN=238, AI=269 (263 at 100%).
Deep dive into what blocks the remaining trades and try new recovery axes."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming,
    simulate_trades, _make_s1_tf3_vix_combo, _build_filter_cascade,
    _make_v20_grand, _make_v21_cn
)

# ── Load data ──
cache_path = Path(__file__).parent / 'combo_cache' / 'combo_signals.pkl'
with open(cache_path, 'rb') as f:
    data = pickle.load(f)
signals = data['signals']
vix_daily = data.get('vix_daily')
spy_daily = data.get('spy_daily')

cascade_vix = _build_filter_cascade(vix=True)
cascade_vix.precompute_vix_cooldown(vix_daily)

# ── Precompute all maps ──
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

# ── Get base and CN filter functions ──
base_fn = _make_s1_tf3_vix_combo(cascade_vix)
cn_fn = _make_v21_cn(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50, vix_map, spy_return_map)

# ── Identify trades CN filters out ──
# Run base (s1_tf3_vix) to get all base-passing trades
base_trades = simulate_trades(signals, base_fn, 'base', cooldown=0, trail_power=6)

# Run CN
cn_trades = simulate_trades(signals, cn_fn, 'CN', cooldown=0, trail_power=6)

# Find base trades NOT in CN (by entry date)
cn_dates = {t.entry_date for t in cn_trades}
filtered_out = [t for t in base_trades if t.entry_date not in cn_dates]

print(f"Base (s1_tf3_vix): {len(base_trades)} trades")
print(f"CN: {len(cn_trades)} trades")
print(f"Filtered out: {len(filtered_out)} trades")
print(f"  Winners: {sum(1 for t in filtered_out if t.pnl > 0)}")
print(f"  Losers: {sum(1 for t in filtered_out if t.pnl <= 0)}")
print()

# ── Profile each filtered trade ──
# Build a lookup from date to DaySignals
day_map = {day.date: day for day in signals}

print("=" * 120)
print(f"{'Date':12} {'Dir':5} {'PnL':>8} {'Conf':>5} {'Health':>6} {'Pos':>5} {'Confl':>5} {'TFs':>3} {'SPY%':>6} {'SMA5':>6} {'SMA50':>6} {'VIX':>5} {'SRet':>6} {'DOW':>4} {'Why Blocked'}")
print("=" * 120)

for t in sorted(filtered_out, key=lambda x: x.entry_date):
    day = day_map.get(t.entry_date)
    if day is None:
        continue

    result = base_fn(day)
    if result is None:
        continue
    action, conf, s_pct, t_pct, src = result

    tfs = _count_tf_confirming(day, action)
    spy_d20 = spy_dist_map.get(day.date, 999)
    spy_d5 = spy_dist_5.get(day.date, -999)
    spy_d50 = spy_dist_50.get(day.date, -999)
    vix = vix_map.get(day.date, 22)
    sret = spy_return_map.get(day.date, 0)
    dd = day.date.date() if hasattr(day.date, 'date') else day.date
    dow = dd.weekday()

    # Determine WHY it was blocked
    reasons = []
    if action == 'BUY':
        # Check LONG SPY gate
        spy_pass = False
        if day.date in spy_above_sma20: spy_pass = True
        elif conf >= 0.80: spy_pass = True
        elif tfs >= 5: spy_pass = True
        elif conf >= 0.70 and tfs >= 4 and day.cs_position_score < 0.95: spy_pass = True
        elif conf >= 0.65 and tfs >= 4 and day.cs_channel_health >= 0.40: spy_pass = True
        elif day.cs_confluence_score >= 0.9 and conf >= 0.65: spy_pass = True
        if not spy_pass:
            reasons.append(f'L_SPY(sma20={day.date in spy_above_sma20})')

        # Check LONG CONF gate
        conf_pass = False
        if conf >= 0.66: conf_pass = True
        elif day.cs_position_score <= 0.99: conf_pass = True
        elif tfs >= 4: conf_pass = True
        elif day.cs_confluence_score >= 0.9 and conf >= 0.55: conf_pass = True
        if not conf_pass:
            reasons.append(f'L_CONF(c={conf:.2f},pos={day.cs_position_score:.2f})')

    if action == 'SELL':
        # Check SHORT SPY gate
        spy_pass = False
        if day.date in spy_above_055pct: spy_pass = True
        elif spy_d20 < 0 and day.cs_channel_health >= 0.32: spy_pass = True
        elif 0 <= spy_d20 < 0.55 and day.cs_position_score < 0.99: spy_pass = True
        elif 0 <= spy_d20 < 0.55 and day.cs_position_score >= 0.99 and day.cs_channel_health >= 0.35: spy_pass = True
        if not spy_pass and spy_d5 >= 0: spy_pass = True
        if not spy_pass:
            if (vix < 20 or vix > 25) and day.cs_channel_health >= 0.25: spy_pass = True
        if not spy_pass and day.cs_channel_health >= 0.25:
            if dow in (0, 3): spy_pass = True
        if not spy_pass and spy_d50 >= 1.0: spy_pass = True
        if not spy_pass:
            reasons.append(f'S_SPY(d20={spy_d20:.2f},d5={spy_d5:.2f})')

        # Check SHORT CONF gate
        conf_pass = False
        if conf >= 0.65: conf_pass = True
        elif day.cs_channel_health >= 0.30: conf_pass = True
        elif day.cs_confluence_score >= 0.9 and day.cs_channel_health >= 0.25: conf_pass = True
        elif vix > 25 and day.cs_channel_health >= 0.20: conf_pass = True
        elif sret < -1.0 and day.cs_channel_health >= 0.20: conf_pass = True
        elif vix > 30 and day.cs_channel_health >= 0.15: conf_pass = True
        elif day.cs_channel_health >= 0.10 and conf >= 0.60 and tfs >= 4: conf_pass = True
        if not conf_pass:
            reasons.append(f'S_CONF(h={day.cs_channel_health:.2f},c={conf:.2f})')

    why = ' + '.join(reasons) if reasons else '???'

    print(f"{str(day.date)[:10]:12} {action:5} ${t.pnl:>+7,.0f} {conf:5.3f} {day.cs_channel_health:6.3f} "
          f"{day.cs_position_score:5.2f} {day.cs_confluence_score:5.2f} {tfs:3d} {spy_d20:+6.2f} {spy_d5:+6.2f} "
          f"{spy_d50:+6.2f} {vix:5.1f} {sret:+6.2f} {dow:4d} {why}")

print()
print("=" * 80)
print("SUMMARY BY BLOCK REASON")
print("=" * 80)

# Count by reason category
reason_counts = defaultdict(lambda: {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0})
for t in filtered_out:
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
    dd = day.date.date() if hasattr(day.date, 'date') else day.date
    dow = dd.weekday()

    reasons = []
    if action == 'BUY':
        spy_pass = False
        if day.date in spy_above_sma20: spy_pass = True
        elif conf >= 0.80: spy_pass = True
        elif tfs >= 5: spy_pass = True
        elif conf >= 0.70 and tfs >= 4 and day.cs_position_score < 0.95: spy_pass = True
        elif conf >= 0.65 and tfs >= 4 and day.cs_channel_health >= 0.40: spy_pass = True
        elif day.cs_confluence_score >= 0.9 and conf >= 0.65: spy_pass = True
        if not spy_pass: reasons.append('L_SPY')

        conf_pass = False
        if conf >= 0.66: conf_pass = True
        elif day.cs_position_score <= 0.99: conf_pass = True
        elif tfs >= 4: conf_pass = True
        elif day.cs_confluence_score >= 0.9 and conf >= 0.55: conf_pass = True
        if not conf_pass: reasons.append('L_CONF')

    if action == 'SELL':
        spy_pass = False
        if day.date in spy_above_055pct: spy_pass = True
        elif spy_d20 < 0 and day.cs_channel_health >= 0.32: spy_pass = True
        elif 0 <= spy_d20 < 0.55 and day.cs_position_score < 0.99: spy_pass = True
        elif 0 <= spy_d20 < 0.55 and day.cs_position_score >= 0.99 and day.cs_channel_health >= 0.35: spy_pass = True
        if not spy_pass and spy_d5 >= 0: spy_pass = True
        if not spy_pass:
            if (vix < 20 or vix > 25) and day.cs_channel_health >= 0.25: spy_pass = True
        if not spy_pass and day.cs_channel_health >= 0.25:
            if dow in (0, 3): spy_pass = True
        if not spy_pass and spy_d50 >= 1.0: spy_pass = True
        if not spy_pass: reasons.append('S_SPY')

        conf_pass = False
        if conf >= 0.65: conf_pass = True
        elif day.cs_channel_health >= 0.30: conf_pass = True
        elif day.cs_confluence_score >= 0.9 and day.cs_channel_health >= 0.25: conf_pass = True
        elif vix > 25 and day.cs_channel_health >= 0.20: conf_pass = True
        elif sret < -1.0 and day.cs_channel_health >= 0.20: conf_pass = True
        elif vix > 30 and day.cs_channel_health >= 0.15: conf_pass = True
        elif day.cs_channel_health >= 0.10 and conf >= 0.60 and tfs >= 4: conf_pass = True
        if not conf_pass: reasons.append('S_CONF')

    key = '+'.join(reasons) if reasons else '???'
    rc = reason_counts[key]
    rc['count'] += 1
    rc['wins'] += 1 if t.pnl > 0 else 0
    rc['losses'] += 1 if t.pnl <= 0 else 0
    rc['pnl'] += t.pnl

for reason, stats in sorted(reason_counts.items(), key=lambda x: -x[1]['count']):
    wr = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
    print(f"  {reason:20s}: {stats['count']:3d} trades ({stats['wins']}W/{stats['losses']}L = {wr:.0f}% WR, ${stats['pnl']:+,.0f})")

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT SECTION: Try new recovery axes
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("RECOVERY EXPERIMENTS")
print("=" * 80)

# Helper: build CN + extra recovery
def make_cn_plus(extra_fn_name, extra_check):
    """Return a function that extends CN with an extra recovery check."""
    def fn(day):
        result = cn_fn(day)
        if result is not None:
            return result
        # CN rejected this — try extra recovery
        result2 = base_fn(day)
        if result2 is None:
            return None
        action, conf, s_pct, t_pct, src = result2
        if extra_check(day, action, conf):
            return result2
        return None
    return fn

# ── Experiment 1: LONG recovery with lower conf thresholds ──
print("\n--- Exp 1: LONG SPY recovery with various gates ---")
long_experiments = [
    ('L: c60&TF4&h50', lambda d, a, c: a == 'BUY' and c >= 0.60 and _count_tf_confirming(d, 'BUY') >= 4 and d.cs_channel_health >= 0.50),
    ('L: c60&TF5', lambda d, a, c: a == 'BUY' and c >= 0.60 and _count_tf_confirming(d, 'BUY') >= 5),
    ('L: confl90&c50', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.9 and c >= 0.50),
    ('L: confl85&c60', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.85 and c >= 0.60),
    ('L: confl85&c55', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.85 and c >= 0.55),
    ('L: c55&TF5&h40', lambda d, a, c: a == 'BUY' and c >= 0.55 and _count_tf_confirming(d, 'BUY') >= 5 and d.cs_channel_health >= 0.40),
    ('L: c50&TF5&pos<90', lambda d, a, c: a == 'BUY' and c >= 0.50 and _count_tf_confirming(d, 'BUY') >= 5 and d.cs_position_score < 0.90),
    ('L: pos<80&c55', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.80 and c >= 0.55),
    ('L: pos<70&c50', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.70 and c >= 0.50),
    ('L: h50&c55&pos<95', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.50 and c >= 0.55 and d.cs_position_score < 0.95),
]

for name, check in long_experiments:
    fn = make_cn_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100 if n > 0 else 0
    bl = min(t.pnl for t in trades) if trades else 0
    if n > 238:
        marker = " *** NEW" if wr >= 100 else ""
        print(f"  {name:25s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

# ── Experiment 2: SHORT SPY recovery with new gates ──
print("\n--- Exp 2: SHORT SPY recovery with new gates ---")
short_spy_experiments = [
    ('S_SPY: VIX20-25&h30', lambda d, a, c: a == 'SELL' and 20 <= vix_map.get(d.date, 22) <= 25 and d.cs_channel_health >= 0.30),
    ('S_SPY: VIX20-25&h35', lambda d, a, c: a == 'SELL' and 20 <= vix_map.get(d.date, 22) <= 25 and d.cs_channel_health >= 0.35),
    ('S_SPY: VIX20-25&h40', lambda d, a, c: a == 'SELL' and 20 <= vix_map.get(d.date, 22) <= 25 and d.cs_channel_health >= 0.40),
    ('S_SPY: VIX20-25&c70', lambda d, a, c: a == 'SELL' and 20 <= vix_map.get(d.date, 22) <= 25 and c >= 0.70),
    ('S_SPY: VIX20-25&c70&TF4', lambda d, a, c: a == 'SELL' and 20 <= vix_map.get(d.date, 22) <= 25 and c >= 0.70 and _count_tf_confirming(d, 'SELL') >= 4),
    ('S_SPY: Tue&h30', lambda d, a, c: a == 'SELL' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and d.cs_channel_health >= 0.30),
    ('S_SPY: Wed&h30', lambda d, a, c: a == 'SELL' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 2 and d.cs_channel_health >= 0.30),
    ('S_SPY: Fri&h30', lambda d, a, c: a == 'SELL' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 4 and d.cs_channel_health >= 0.30),
    ('S_SPY: Tue&h25', lambda d, a, c: a == 'SELL' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 1 and d.cs_channel_health >= 0.25),
    ('S_SPY: Wed&h25', lambda d, a, c: a == 'SELL' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 2 and d.cs_channel_health >= 0.25),
    ('S_SPY: Fri&h25', lambda d, a, c: a == 'SELL' and (d.date.date() if hasattr(d.date, 'date') else d.date).weekday() == 4 and d.cs_channel_health >= 0.25),
    ('S_SPY: SMA10>=0', lambda d, a, c: a == 'SELL'),  # placeholder - need SMA10
    ('S_SPY: confl90&h25', lambda d, a, c: a == 'SELL' and d.cs_confluence_score >= 0.9 and d.cs_channel_health >= 0.25),
    ('S_SPY: confl85&h25', lambda d, a, c: a == 'SELL' and d.cs_confluence_score >= 0.85 and d.cs_channel_health >= 0.25),
    ('S_SPY: c70&h20', lambda d, a, c: a == 'SELL' and c >= 0.70 and d.cs_channel_health >= 0.20),
    ('S_SPY: c75&h15', lambda d, a, c: a == 'SELL' and c >= 0.75 and d.cs_channel_health >= 0.15),
    ('S_SPY: TF5&h20', lambda d, a, c: a == 'SELL' and _count_tf_confirming(d, 'SELL') >= 5 and d.cs_channel_health >= 0.20),
    ('S_SPY: TF4&c65&h20', lambda d, a, c: a == 'SELL' and _count_tf_confirming(d, 'SELL') >= 4 and c >= 0.65 and d.cs_channel_health >= 0.20),
    ('S_SPY: pos>95&h25', lambda d, a, c: a == 'SELL' and d.cs_position_score >= 0.95 and d.cs_channel_health >= 0.25),
]

for name, check in short_spy_experiments:
    if 'SMA10' in name:
        continue  # skip placeholder
    fn = make_cn_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100 if n > 0 else 0
    bl = min(t.pnl for t in trades) if trades else 0
    if n > 238:
        marker = " *** NEW" if wr >= 100 else ""
        print(f"  {name:25s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

# ── Experiment 3: SHORT CONF recovery ──
print("\n--- Exp 3: SHORT CONF recovery ---")
short_conf_experiments = [
    ('SC: h05&c65&TF4', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.05 and c >= 0.65 and _count_tf_confirming(d, 'SELL') >= 4),
    ('SC: h05&c70&TF3', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.05 and c >= 0.70 and _count_tf_confirming(d, 'SELL') >= 3),
    ('SC: h15&c55&TF5', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.15 and c >= 0.55 and _count_tf_confirming(d, 'SELL') >= 5),
    ('SC: h05&c60&TF5', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.05 and c >= 0.60 and _count_tf_confirming(d, 'SELL') >= 5),
    ('SC: h15&c60&confl90', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.15 and c >= 0.60 and d.cs_confluence_score >= 0.9),
    ('SC: h20&c55&TF4', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.20 and c >= 0.55 and _count_tf_confirming(d, 'SELL') >= 4),
    ('SC: h10&c55&TF5', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.10 and c >= 0.55 and _count_tf_confirming(d, 'SELL') >= 5),
    ('SC: h25&confl85', lambda d, a, c: a == 'SELL' and d.cs_channel_health >= 0.25 and d.cs_confluence_score >= 0.85),
    ('SC: pos>98&h10&c55', lambda d, a, c: a == 'SELL' and d.cs_position_score >= 0.98 and d.cs_channel_health >= 0.10 and c >= 0.55),
]

for name, check in short_conf_experiments:
    fn = make_cn_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100 if n > 0 else 0
    bl = min(t.pnl for t in trades) if trades else 0
    if n > 238:
        marker = " *** NEW" if wr >= 100 else ""
        print(f"  {name:25s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

# ── Experiment 4: LONG CONF recovery ──
print("\n--- Exp 4: LONG CONF recovery ---")
long_conf_experiments = [
    ('LC: confl85&c50', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.85 and c >= 0.50),
    ('LC: confl80&c55', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.80 and c >= 0.55),
    ('LC: confl90&c45', lambda d, a, c: a == 'BUY' and d.cs_confluence_score >= 0.9 and c >= 0.45),
    ('LC: pos<80&c50&TF3', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.80 and c >= 0.50 and _count_tf_confirming(d, 'BUY') >= 3),
    ('LC: pos<70', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.70),
    ('LC: h50&c50&TF4', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.50 and c >= 0.50 and _count_tf_confirming(d, 'BUY') >= 4),
    ('LC: h40&c55&pos<90', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.40 and c >= 0.55 and d.cs_position_score < 0.90),
    ('LC: c60&pos<95&TF3', lambda d, a, c: a == 'BUY' and c >= 0.60 and d.cs_position_score < 0.95 and _count_tf_confirming(d, 'BUY') >= 3),
]

for name, check in long_conf_experiments:
    fn = make_cn_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100 if n > 0 else 0
    bl = min(t.pnl for t in trades) if trades else 0
    if n > 238:
        marker = " *** NEW" if wr >= 100 else ""
        print(f"  {name:25s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

# ── Experiment 5: Cross-gate recovery (SPY check passes but CONF fails, or vice versa) ──
print("\n--- Exp 5: Targeted gate-specific recovery ---")
# For trades blocked by S_CONF only (SPY gate passed): lower conf requirements
# For trades blocked by S_SPY only (CONF gate passed): expand SPY paths
# For trades blocked by L_SPY only: expand long SPY paths
# For trades blocked by L_CONF only: expand long conf paths

# Try letting through trades where BOTH gates are close to passing
cross_experiments = [
    ('X: SELL any_h&c60&TF5', lambda d, a, c: a == 'SELL' and c >= 0.60 and _count_tf_confirming(d, 'SELL') >= 5),
    ('X: SELL c70&TF4', lambda d, a, c: a == 'SELL' and c >= 0.70 and _count_tf_confirming(d, 'SELL') >= 4),
    ('X: SELL c75&TF3', lambda d, a, c: a == 'SELL' and c >= 0.75 and _count_tf_confirming(d, 'SELL') >= 3),
    ('X: SELL c80', lambda d, a, c: a == 'SELL' and c >= 0.80),
    ('X: BUY c60&TF4&h40', lambda d, a, c: a == 'BUY' and c >= 0.60 and _count_tf_confirming(d, 'BUY') >= 4 and d.cs_channel_health >= 0.40),
    ('X: BUY c55&TF5', lambda d, a, c: a == 'BUY' and c >= 0.55 and _count_tf_confirming(d, 'BUY') >= 5),
    ('X: BUY h50&confl80', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.50 and d.cs_confluence_score >= 0.80),
    ('X: BUY h60&c50', lambda d, a, c: a == 'BUY' and d.cs_channel_health >= 0.60 and c >= 0.50),
]

for name, check in cross_experiments:
    fn = make_cn_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100 if n > 0 else 0
    bl = min(t.pnl for t in trades) if trades else 0
    if n > 238:
        marker = " *** NEW" if wr >= 100 else ""
        print(f"  {name:25s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

# ── Experiment 6: VIX-regime specific recovery ──
print("\n--- Exp 6: VIX-regime specific recovery ---")
vix_experiments = [
    ('VIX: SELL VIX<18&h20', lambda d, a, c: a == 'SELL' and vix_map.get(d.date, 22) < 18 and d.cs_channel_health >= 0.20),
    ('VIX: SELL VIX<15&h15', lambda d, a, c: a == 'SELL' and vix_map.get(d.date, 22) < 15 and d.cs_channel_health >= 0.15),
    ('VIX: SELL VIX>30&h10', lambda d, a, c: a == 'SELL' and vix_map.get(d.date, 22) > 30 and d.cs_channel_health >= 0.10),
    ('VIX: SELL VIX>35&h05', lambda d, a, c: a == 'SELL' and vix_map.get(d.date, 22) > 35 and d.cs_channel_health >= 0.05),
    ('VIX: BUY VIX>25&c55', lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) > 25 and c >= 0.55),
    ('VIX: BUY VIX>30&c50', lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) > 30 and c >= 0.50),
    ('VIX: BUY VIX<15&c55', lambda d, a, c: a == 'BUY' and vix_map.get(d.date, 22) < 15 and c >= 0.55),
]

for name, check in vix_experiments:
    fn = make_cn_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100 if n > 0 else 0
    bl = min(t.pnl for t in trades) if trades else 0
    if n > 238:
        marker = " *** NEW" if wr >= 100 else ""
        print(f"  {name:25s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

# ── Experiment 7: SPY return momentum recovery ──
print("\n--- Exp 7: SPY momentum recovery ---")
# 2-day and 3-day SPY momentum
spy_ret_2d = {}
spy_ret_3d = {}
for i in range(2, len(spy_close)):
    spy_ret_2d[spy_daily.index[i]] = (spy_close[i]-spy_close[i-2])/spy_close[i-2]*100
for i in range(3, len(spy_close)):
    spy_ret_3d[spy_daily.index[i]] = (spy_close[i]-spy_close[i-3])/spy_close[i-3]*100

mom_experiments = [
    ('Mom: SELL SRet2d<-1&h20', lambda d, a, c: a == 'SELL' and spy_ret_2d.get(d.date, 0) < -1.0 and d.cs_channel_health >= 0.20),
    ('Mom: SELL SRet3d<-1.5&h20', lambda d, a, c: a == 'SELL' and spy_ret_3d.get(d.date, 0) < -1.5 and d.cs_channel_health >= 0.20),
    ('Mom: SELL SRet2d<-2&h15', lambda d, a, c: a == 'SELL' and spy_ret_2d.get(d.date, 0) < -2.0 and d.cs_channel_health >= 0.15),
    ('Mom: BUY SRet>1&c55', lambda d, a, c: a == 'BUY' and spy_return_map.get(d.date, 0) > 1.0 and c >= 0.55),
    ('Mom: BUY SRet2d>1&c55', lambda d, a, c: a == 'BUY' and spy_ret_2d.get(d.date, 0) > 1.0 and c >= 0.55),
    ('Mom: BUY SRet3d>1.5&c50', lambda d, a, c: a == 'BUY' and spy_ret_3d.get(d.date, 0) > 1.5 and c >= 0.50),
]

for name, check in mom_experiments:
    fn = make_cn_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100 if n > 0 else 0
    bl = min(t.pnl for t in trades) if trades else 0
    if n > 238:
        marker = " *** NEW" if wr >= 100 else ""
        print(f"  {name:25s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

# ── Experiment 8: TSLA-specific gates ──
print("\n--- Exp 8: Multi-signal quality gates ---")
# Use energy_ratio, timing_score, and other DaySignals fields
quality_experiments = [
    ('Q: SELL pos>98&c55', lambda d, a, c: a == 'SELL' and d.cs_position_score >= 0.98 and c >= 0.55),
    ('Q: BUY pos<50&c45', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.50 and c >= 0.45),
    ('Q: BUY pos<60&c50', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.60 and c >= 0.50),
    ('Q: BUY pos<40', lambda d, a, c: a == 'BUY' and d.cs_position_score < 0.40),
]

for name, check in quality_experiments:
    fn = make_cn_plus(name, check)
    trades = simulate_trades(signals, fn, name, cooldown=0, trail_power=6)
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100 if n > 0 else 0
    bl = min(t.pnl for t in trades) if trades else 0
    if n > 238:
        marker = " *** NEW" if wr >= 100 else ""
        print(f"  {name:25s}: {n:3d} trades, {wr:5.1f}% WR, BL=${bl:+,.0f}{marker}")

print("\n✓ v22 experiments complete")
