#!/usr/bin/env python3
"""v42b: Continue v42 Parts 4-10 (fixed variable capture bugs) + new ideas.

New ideas based on v42 Part 1-3 findings:
- Only 38 rejected signals, 18 winners, 12 losers
- Untapped features (energy/timing/entropy) don't discriminate
- Try: intraday range, gap ratios, previous-day context, per-direction DOW"""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, simulate_trades, _build_filter_cascade,
    _make_v27_ct, _make_v33_cz, MIN_SIGNAL_CONFIDENCE, _floor_stop_tp,
    _SigProxy, _AnalysisProxy
)

cache_path = Path(__file__).parent / 'combo_cache' / 'combo_signals.pkl'
with open(cache_path, 'rb') as f:
    data = pickle.load(f)
signals = data['signals']
vix_daily, spy_daily = data['vix_daily'], data['spy_daily']

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

args = (cascade_vix, spy_above_sma20, spy_above_055pct,
        spy_dist_map, spy_dist_5, spy_dist_50,
        vix_map, spy_return_map, spy_ret_2d)

ct_fn = _make_v27_ct(*args)
cz_fn = _make_v33_cz(*args)

def _tf0_base(day):
    if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
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

cz_trades = simulate_trades(signals, cz_fn, 'CZ', cooldown=0, trail_power=6)
print(f"CZ baseline: {len(cz_trades)} trades, "
      f"{sum(1 for t in cz_trades if t.pnl > 0)/len(cz_trades)*100:.1f}% WR")

# Precompute daily maps
daily_df = data.get('daily_df')
day_range_map = {}  # date -> intraday range pct
day_gap_map = {}    # date -> gap pct (open vs prev close)
prev_close_map = {}
if daily_df is not None:
    for i in range(len(daily_df)):
        idx = daily_df.index[i]
        row = daily_df.iloc[i]
        if row['close'] > 0:
            day_range_map[idx] = (row['high'] - row['low']) / row['close'] * 100
        if i > 0:
            prev_c = daily_df.iloc[i-1]['close']
            if prev_c > 0:
                day_gap_map[idx] = (row['open'] - prev_c) / prev_c * 100
                prev_close_map[idx] = prev_c

# Build signal date -> day index map for prev-day lookups
sig_by_date = {d.date: d for d in signals}
sorted_dates = sorted(sig_by_date.keys())
date_to_idx = {d: i for i, d in enumerate(sorted_dates)}

# ════════════════════════════════════════════════════════
# Part 4 (fixed): Combined feature AND-gates
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 4: Triple AND-gates for Tue/Thu/Fri (FIXED)")
print("="*70)

for extra_dow, dow_label in [({1,3,4}, "TuThFr"), ({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri")]:
    print(f"\n--- {dow_label}: h + confl + entropy ---")
    for h_th in [0.25, 0.20, 0.15, 0.10]:
        for confl_th in [0.80, 0.70, 0.60, 0.50]:
            for ent_th in [0.95, 0.90, 0.85, 0.80, 0.70]:
                def make_triple(ed=extra_dow, ht=h_th, ct=confl_th, entt=ent_th):
                    def fn(day):
                        result = cz_fn(day)
                        if result is not None:
                            return result
                        result = _tf0_base(day)
                        if result is not None:
                            dd = day.date.date() if hasattr(day.date, 'date') else day.date
                            if dd.weekday() in ed:
                                if (day.cs_channel_health >= ht and
                                    day.cs_confluence_score >= ct and
                                    day.cs_entropy_score is not None and day.cs_entropy_score >= entt):
                                    return result
                        return None
                    return fn
                trades = simulate_trades(signals, make_triple(), 'test', cooldown=0, trail_power=6)
                n = len(trades)
                w = sum(1 for t in trades if t.pnl > 0)
                wr = w/n*100 if n else 0
                if wr >= 100 and n > 426:
                    pnl = sum(t.pnl for t in trades)
                    print(f"  h>={h_th:.2f}&confl>={confl_th:.2f}&ent>={ent_th:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} ***")
                elif n > 426 and wr >= 99.5:
                    pnl = sum(t.pnl for t in trades)
                    bl = min(t.pnl for t in trades)
                    print(f"  h>={h_th:.2f}&confl>={confl_th:.2f}&ent>={ent_th:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f}")

# ════════════════════════════════════════════════════════
# Part 5: V5 expansion
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 5: V5 filter relaxation")
print("="*70)

for v5h_max in [0.60, 0.65, 0.70, 0.80, 1.0]:
    for v5p_max in [0.85, 0.90, 0.95, 1.0]:
        def make_v5_relax(hm=v5h_max, pm=v5p_max):
            def fn(day):
                result = cz_fn(day)
                if result is not None:
                    return result
                # Try wider V5 (cz_fn already checks V5 with h<0.57 & pos<0.85)
                if day.v5_take_bounce:
                    if day.cs_channel_health < hm and day.cs_position_score < pm:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_v5_relax(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if n > 426:
            pnl = sum(t.pnl for t in trades)
            bl = min(t.pnl for t in trades)
            flag = "***" if wr >= 100 else ""
            print(f"  V5 h<{v5h_max:.2f} pos<{v5p_max:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Part 6: Trail power variation
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 6: Trail power variation on CZ")
print("="*70)

for tp in [4, 5, 7, 8, 10, 12]:
    trades = simulate_trades(signals, cz_fn, 'CZ', cooldown=0, trail_power=tp)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    print(f"  trail_power={tp:2d}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f}")

# ════════════════════════════════════════════════════════
# Part 7: Intraday range filter for Tue/Thu/Fri
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 7: Intraday range filter for unsafe days")
print("="*70)

if day_range_map:
    for extra_dow, dow_label in [({1,3,4}, "TuThFr"), ({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri")]:
        print(f"\n--- {dow_label} ---")
        # Low range = less volatile = safer?
        for range_max in [8.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0]:
            def make_range_gate(ed=extra_dow, rm=range_max):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd.weekday() in ed:
                            dr = day_range_map.get(day.date, 99)
                            if dr <= rm:
                                return result
                    return None
                return fn
            trades = simulate_trades(signals, make_range_gate(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if n > 426:
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                flag = "***" if wr >= 100 else ""
                print(f"  range<={rm:.1f}%: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

        # High range = strong move = more reliable?
        for range_min in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
            def make_range_gate2(ed=extra_dow, rmn=range_min):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd.weekday() in ed:
                            dr = day_range_map.get(day.date, 0)
                            if dr >= rmn:
                                return result
                    return None
                return fn
            trades = simulate_trades(signals, make_range_gate2(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if n > 426:
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                flag = "***" if wr >= 100 else ""
                print(f"  range>={rmn:.1f}%: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")
else:
    print("daily_df not in cache, skipping")

# ════════════════════════════════════════════════════════
# Part 8: Gap filter for Tue/Thu/Fri
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 8: Gap filter for unsafe days")
print("="*70)

if day_gap_map:
    for extra_dow, dow_label in [({1,3,4}, "TuThFr"), ({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri")]:
        print(f"\n--- {dow_label} ---")
        # Small gap = normal open = safer?
        for gap_max in [3.0, 2.0, 1.5, 1.0, 0.5]:
            def make_gap_gate(ed=extra_dow, gm=gap_max):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd.weekday() in ed:
                            gap = abs(day_gap_map.get(day.date, 99))
                            if gap <= gm:
                                return result
                    return None
                return fn
            trades = simulate_trades(signals, make_gap_gate(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if n > 426:
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                flag = "***" if wr >= 100 else ""
                print(f"  |gap|<={gm:.1f}%: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")
else:
    print("day_gap_map empty, skipping")

# ════════════════════════════════════════════════════════
# Part 9: Direction-specific DOW relaxation
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 9: Direction-specific DOW relaxation")
print("="*70)

# Maybe BUY is safe on certain Tue/Thu/Fri but SELL isn't (or vice versa)
for direction in ['BUY', 'SELL']:
    for extra_dow, dow_label in [({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri"),
                                  ({1,3}, "Tue+Thu"), ({3,4}, "Thu+Fri"),
                                  ({1,4}, "Tue+Fri"), ({1,3,4}, "TuThFr")]:
        for h_th in [0.30, 0.25, 0.22, 0.20, 0.15, 0.10]:
            def make_dir_dow(dr=direction, ed=extra_dow, ht=h_th):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd.weekday() in ed and day.cs_action == dr:
                            if day.cs_channel_health >= ht:
                                return result
                    return None
                return fn
            trades = simulate_trades(signals, make_dir_dow(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if n > 426:
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                flag = "***" if wr >= 100 else ""
                print(f"  {direction} {dow_label} h>={h_th:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Part 10: Cross-day confirmation (prev day also had signal)
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 10: Cross-day confirmation (prev day also had signal)")
print("="*70)

for extra_dow, dow_label in [({1,3,4}, "TuThFr"), ({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri")]:
    print(f"\n--- {dow_label} ---")
    # Require that previous day also had a BUY/SELL signal (clustering)
    def make_cluster(ed=extra_dow):
        def fn(day):
            result = cz_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() in ed:
                    idx = date_to_idx.get(day.date)
                    if idx is not None and idx > 0:
                        prev_day = sig_by_date[sorted_dates[idx - 1]]
                        if prev_day.cs_action in ('BUY', 'SELL'):
                            return result
            return None
        return fn
    trades = simulate_trades(signals, make_cluster(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    if n > 426:
        pnl = sum(t.pnl for t in trades)
        bl = min(t.pnl for t in trades)
        flag = "***" if wr >= 100 else ""
        print(f"  prev_signal: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

    # Require same direction as previous day
    def make_same_dir(ed=extra_dow):
        def fn(day):
            result = cz_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() in ed:
                    idx = date_to_idx.get(day.date)
                    if idx is not None and idx > 0:
                        prev_day = sig_by_date[sorted_dates[idx - 1]]
                        if prev_day.cs_action == day.cs_action:
                            return result
            return None
        return fn
    trades = simulate_trades(signals, make_same_dir(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    if n > 426:
        pnl = sum(t.pnl for t in trades)
        bl = min(t.pnl for t in trades)
        flag = "***" if wr >= 100 else ""
        print(f"  same_dir: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Part 11: Per-loser analysis — what filter blocks each specific loser?
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 11: Per-loser analysis — can we block specific losers?")
print("="*70)

# Known losers from v42: 12 specific dates
# Let's try adding ALL rejected signals and see which ones actually lose
def make_all_rejected():
    def fn(day):
        result = cz_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            return result
        return None
    return fn

all_trades = simulate_trades(signals, make_all_rejected(), 'test', cooldown=0, trail_power=6)
cz_dates_set = {t.entry_date for t in cz_trades}
new_trades = [t for t in all_trades if t.entry_date not in cz_dates_set]
new_winners = [t for t in new_trades if t.pnl > 0]
new_losers = [t for t in new_trades if t.pnl <= 0]

print(f"\nAdding ALL rejected signals:")
print(f"  Total: {len(all_trades)}t ({len(new_trades)} new)")
n = len(all_trades)
w = sum(1 for t in all_trades if t.pnl > 0)
wr = w/n*100 if n else 0
pnl = sum(t.pnl for t in all_trades)
bl = min(t.pnl for t in all_trades) if all_trades else 0
print(f"  All: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f}")
print(f"  New winners: {len(new_winners)}, New losers: {len(new_losers)}")

if new_losers:
    print("\nActual new losers when ALL added (with displacement effects):")
    for t in sorted(new_losers, key=lambda x: x.pnl):
        # Find corresponding signal day
        sig_day = None
        for day in signals:
            if cz_fn(day) is None and _tf0_base(day) is not None:
                # Check if this day could produce this trade
                pass
        dd = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        print(f"  {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} {dow}")

# Also check: what if we exclude JUST the worst losers' dates?
if new_losers:
    loser_dates = {t.entry_date for t in new_losers}
    print(f"\nLoser entry dates to block: {sorted(str(d)[:10] for d in loser_dates)}")

    # Try blocking just those dates
    def make_block_losers(blocked=loser_dates):
        def fn(day):
            result = cz_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                # Check if this would produce a trade on a blocked date
                # We can't easily predict entry date from signal date,
                # but we can still try
                return result
            return None
        return fn

# ════════════════════════════════════════════════════════
# Part 12: Relaxing EXISTING CZ conditions (wider catch)
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 12: Relaxing existing CZ conditions")
print("="*70)

# What if we lower the base h thresholds (for ALL days)?
for h_buy in [0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30]:
    for h_sell in [0.30, 0.29, 0.28, 0.27, 0.26, 0.25]:
        def make_relaxed(hb=h_buy, hs=h_sell):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_b = hb
                    h_s = hs
                    dd = day.date.date() if hasattr(day.date, 'date') else day.date
                    vix_val = vix_map.get(day.date, 22)
                    spy_d = spy_dist_map.get(day.date, 0)
                    relax = dd.weekday() == 0 or vix_val > 25
                    if day.cs_action == 'BUY' and spy_d < -1.0:
                        relax = True
                    if relax:
                        h_b = min(h_b, 0.22)
                        h_s = min(h_s, 0.22)
                    if dd.weekday() == 2:
                        h_b = min(h_b, 0.14)
                        h_s = min(h_s, 0.14)
                    h_thresh = h_b if day.cs_action == 'BUY' else h_s
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_relaxed(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if n > 426 and wr >= 100:
            pnl = sum(t.pnl for t in trades)
            print(f"  BUY h>={h_buy:.2f} SELL h>={h_sell:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Part 13: Lower confluence gate (currently 0.90)
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 13: Confluence gate relaxation")
print("="*70)

for confl_gate in [0.88, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]:
    def make_confl_relax(cg=confl_gate):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                h_buy = 0.38
                h_sell = 0.31
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                vix_val = vix_map.get(day.date, 22)
                spy_d = spy_dist_map.get(day.date, 0)
                relax = dd.weekday() == 0 or vix_val > 25
                if day.cs_action == 'BUY' and spy_d < -1.0:
                    relax = True
                if relax:
                    h_buy = min(h_buy, 0.22)
                    h_sell = min(h_sell, 0.22)
                if dd.weekday() == 2:
                    h_buy = min(h_buy, 0.14)
                    h_sell = min(h_sell, 0.14)
                h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= cg:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_confl_relax(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    flag = "***" if wr >= 100 else ""
    print(f"  confl>={confl_gate:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Part 14: Conditional DOW with direction + VIX/SPY combo
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 14: Conditional DOW+Direction+VIX/SPY combos")
print("="*70)

conditions = [
    ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
    ("VIX<18", lambda d: vix_map.get(d.date, 22) < 18),
    ("VIX>25", lambda d: vix_map.get(d.date, 22) > 25),
    ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
    ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
    ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
    ("conf>=0.60", lambda d: d.cs_confidence >= 0.60),
    ("pos<0.90", lambda d: d.cs_position_score < 0.90),
]

for direction in ['BUY', 'SELL']:
    for extra_dow, dow_label in [({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri")]:
        for cond_label, cond_fn_check in conditions:
            for h_th in [0.25, 0.20, 0.15, 0.10]:
                def make_cond_dir_dow(dr=direction, ed=extra_dow, cf=cond_fn_check, ht=h_th):
                    def fn(day):
                        result = cz_fn(day)
                        if result is not None:
                            return result
                        result = _tf0_base(day)
                        if result is not None:
                            dd = day.date.date() if hasattr(day.date, 'date') else day.date
                            if dd.weekday() in ed and day.cs_action == dr:
                                if cf(day) and day.cs_channel_health >= ht:
                                    return result
                        return None
                    return fn
                trades = simulate_trades(signals, make_cond_dir_dow(), 'test', cooldown=0, trail_power=6)
                n = len(trades)
                w = sum(1 for t in trades if t.pnl > 0)
                wr = w/n*100 if n else 0
                if wr >= 100 and n > 426:
                    pnl = sum(t.pnl for t in trades)
                    print(f"  {direction} {dow_label} {cond_label} h>={h_th:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} ***")

print("\nDone")
