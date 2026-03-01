#!/usr/bin/env python3
"""v42: Exploit untapped DaySignals fields to push beyond CZ (426 trades).

Untapped fields never used in any gate condition:
- cs_energy_score: energy/momentum metric
- cs_entropy_score: disorder/volatility metric
- cs_timing_score: timing quality metric
- cs_signal_type: signal classification ('bounce', etc.)
- cs_primary_tf: which TF generated signal
- cs_stop_pct / cs_tp_pct: suggested stops (only used for sim, not filtering)

Strategy: First analyze feature distributions across CZ winners vs rejected signals,
then try using untapped features as additional gates for unsafe days."""
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
day_map = {d.date: d for d in signals}

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
cz_dates = {t.entry_date for t in cz_trades}
# Also get signal dates (day before entry)
cz_signal_dates = set()
date_list = [d.date for d in signals]
for t in cz_trades:
    idx = date_list.index(t.entry_date) if t.entry_date in date_list else -1
    if idx > 0:
        cz_signal_dates.add(date_list[idx - 1])
    # Also add entry_date itself as signal might fire on same day
    cz_signal_dates.add(t.entry_date)

print(f"CZ baseline: {len(cz_trades)} trades, "
      f"{sum(1 for t in cz_trades if t.pnl > 0)/len(cz_trades)*100:.1f}% WR")

# ════════════════════════════════════════════════════════
# Part 1: Feature distribution analysis
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 1: Feature distributions — CZ trades vs rejected signals")
print("="*70)

# Categorize all signals
cz_features = []  # Features of signals that became CZ trades
rejected_features = []  # Signals that pass _tf0_base but NOT in CZ

for day in signals:
    result = cz_fn(day)
    is_cz = result is not None

    tf0_result = _tf0_base(day)
    passes_cascade = tf0_result is not None

    features = {
        'date': day.date,
        'action': day.cs_action,
        'energy': day.cs_energy_score,
        'entropy': day.cs_entropy_score,
        'timing': day.cs_timing_score,
        'health': day.cs_channel_health,
        'confidence': day.cs_confidence,
        'confluence': day.cs_confluence_score,
        'position': day.cs_position_score,
        'signal_type': day.cs_signal_type,
        'primary_tf': day.cs_primary_tf,
        'stop_pct': day.cs_stop_pct,
        'tp_pct': day.cs_tp_pct,
    }

    if is_cz:
        cz_features.append(features)
    elif passes_cascade:
        rejected_features.append(features)

print(f"\nCZ signal days: {len(cz_features)}")
print(f"Rejected (pass cascade, fail CZ gates): {len(rejected_features)}")

for field in ['energy', 'entropy', 'timing', 'health', 'confidence', 'confluence', 'position', 'stop_pct', 'tp_pct']:
    cz_vals = [f[field] for f in cz_features if f[field] is not None]
    rej_vals = [f[field] for f in rejected_features if f[field] is not None]
    if cz_vals and rej_vals:
        print(f"\n  {field}:")
        print(f"    CZ:  mean={np.mean(cz_vals):.4f} std={np.std(cz_vals):.4f} "
              f"min={np.min(cz_vals):.4f} max={np.max(cz_vals):.4f}")
        print(f"    Rej: mean={np.mean(rej_vals):.4f} std={np.std(rej_vals):.4f} "
              f"min={np.min(rej_vals):.4f} max={np.max(rej_vals):.4f}")
    elif cz_vals:
        print(f"\n  {field}: CZ only ({len(cz_vals)} vals), no rejected")

# Signal type distribution
print(f"\n  signal_type:")
cz_types = Counter(f['signal_type'] for f in cz_features)
rej_types = Counter(f['signal_type'] for f in rejected_features)
print(f"    CZ:  {dict(cz_types)}")
print(f"    Rej: {dict(rej_types)}")

# Primary TF distribution
print(f"\n  primary_tf:")
cz_tfs = Counter(f['primary_tf'] for f in cz_features)
rej_tfs = Counter(f['primary_tf'] for f in rejected_features)
print(f"    CZ:  {dict(cz_tfs)}")
print(f"    Rej: {dict(rej_tfs)}")

# ════════════════════════════════════════════════════════
# Part 2: Fix missed signals analysis (use signal date not entry date)
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 2: Fixed missed signals analysis")
print("="*70)

# For each rejected signal, simulate adding it to CZ and check if it wins
missed_winners = []
missed_losers = []
for day in signals:
    # Skip if CZ already fires on this day
    result_cz = cz_fn(day)
    if result_cz is not None:
        continue
    result = _tf0_base(day)
    if result is None:
        continue

    # Create a function that adds this one trade to CZ
    signal_date = day.date
    signal_result = result
    def make_single(sd=signal_date, sr=signal_result):
        def fn(d):
            if d.date == sd:
                return sr
            return cz_fn(d)
        return fn

    test_trades = simulate_trades(signals, make_single(), 'test', cooldown=0, trail_power=6)
    # Find the trade that entered AFTER this signal date
    # With cooldown=0, the trade enters next available day
    for t in test_trades:
        if t.entry_date not in cz_dates:
            # This is a new trade not in CZ
            if t.pnl > 0:
                missed_winners.append((day, t))
            else:
                missed_losers.append((day, t))
            break

print(f"Missed winners: {len(missed_winners)}")
print(f"Missed losers: {len(missed_losers)}")

if missed_losers:
    print("\nMissed losers (why they can't be added):")
    for day, trade in sorted(missed_losers, key=lambda x: x[1].pnl)[:20]:
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        vix_val = vix_map.get(day.date, 0)
        spy_d = spy_dist_map.get(day.date, 0)
        print(f"  {str(day.date)[:10]} {day.cs_action:4s} ${trade.pnl:+8,.0f} "
              f"h={day.cs_channel_health:.3f} c={day.cs_confidence:.3f} "
              f"confl={day.cs_confluence_score:.2f} e={day.cs_energy_score:.3f} "
              f"ent={day.cs_entropy_score:.3f} tim={day.cs_timing_score:.3f} "
              f"pos={day.cs_position_score:.3f} VIX={vix_val:.1f} SPY={spy_d:+.2f}% {dow}")

if missed_winners:
    print(f"\nMissed winners ({len(missed_winners)} total), showing all:")
    for day, trade in sorted(missed_winners, key=lambda x: -x[1].pnl)[:30]:
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        vix_val = vix_map.get(day.date, 0)
        spy_d = spy_dist_map.get(day.date, 0)
        print(f"  {str(day.date)[:10]} {day.cs_action:4s} ${trade.pnl:+8,.0f} "
              f"h={day.cs_channel_health:.3f} c={day.cs_confidence:.3f} "
              f"confl={day.cs_confluence_score:.2f} e={day.cs_energy_score:.3f} "
              f"ent={day.cs_entropy_score:.3f} tim={day.cs_timing_score:.3f} "
              f"pos={day.cs_position_score:.3f} VIX={vix_val:.1f} SPY={spy_d:+.2f}% {dow}")

# ════════════════════════════════════════════════════════
# Part 3: Untapped feature gates on Tue/Thu/Fri
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 3: Untapped feature gates for unsafe days (Tue/Thu/Fri)")
print("="*70)

# Try energy, entropy, timing as additional gates on Tue/Thu/Fri
for extra_dow, dow_label in [({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri"),
                              ({1,3}, "Tue+Thu"), ({3,4}, "Thu+Fri"),
                              ({1,3,4}, "TuThFr")]:
    print(f"\n--- {dow_label} ---")
    for field_name, get_field in [
        ("energy", lambda d: d.cs_energy_score),
        ("entropy", lambda d: d.cs_entropy_score),
        ("timing", lambda d: d.cs_timing_score),
        ("stop_pct", lambda d: d.cs_stop_pct),
        ("tp_pct", lambda d: d.cs_tp_pct),
    ]:
        for thresh in [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
                       0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
            def make_feat_gate(ed=extra_dow, gf=get_field, ft=thresh):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd.weekday() in ed:
                            val = gf(day)
                            if val is not None and val >= ft:
                                return result
                    return None
                return fn
            trades = simulate_trades(signals, make_feat_gate(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if n > 426:
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                flag = "***" if wr >= 100 else ""
                print(f"  {field_name}>={thresh:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Part 4: Combined untapped feature gates
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 4: Combined untapped feature AND-gates for Tue/Thu/Fri")
print("="*70)

for extra_dow, dow_label in [({1,3,4}, "TuThFr"), ({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri")]:
    print(f"\n--- {dow_label} ---")
    for e_th in [0.80, 0.70, 0.60, 0.50, 0.40]:
        for t_th in [0.80, 0.70, 0.60, 0.50, 0.40]:
            def make_et_gate(ed=extra_dow, et=e_th, tt=t_th):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd.weekday() in ed:
                            if (day.cs_energy_score is not None and day.cs_energy_score >= et and
                                day.cs_timing_score is not None and day.cs_timing_score >= tt):
                                return result
                    return None
                return fn
            trades = simulate_trades(signals, make_et_gate(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if n > 426:
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                flag = "***" if wr >= 100 else ""
                print(f"  e>={et:.2f}&t>={tt:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

    # Energy + entropy
    for e_th in [0.80, 0.70, 0.60, 0.50, 0.40]:
        for ent_th in [0.80, 0.70, 0.60, 0.50, 0.40]:
            def make_ee_gate(ed=extra_dow, et=e_th, entt=ent_th):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd.weekday() in ed:
                            if (day.cs_energy_score is not None and day.cs_energy_score >= et and
                                day.cs_entropy_score is not None and day.cs_entropy_score >= entt):
                                return result
                    return None
                return fn
            trades = simulate_trades(signals, make_ee_gate(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if n > 426:
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                flag = "***" if wr >= 100 else ""
                print(f"  e>={et:.2f}&ent>={entt:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Part 5: Signal type and primary TF specific gates
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 5: Signal type / primary TF gates for unsafe days")
print("="*70)

# Collect unique signal types and primary TFs
all_sig_types = set()
all_pri_tfs = set()
for day in signals:
    if day.cs_signal_type: all_sig_types.add(day.cs_signal_type)
    if day.cs_primary_tf: all_pri_tfs.add(day.cs_primary_tf)
print(f"Signal types: {sorted(all_sig_types)}")
print(f"Primary TFs: {sorted(all_pri_tfs)}")

for extra_dow, dow_label in [({1,3,4}, "TuThFr"), ({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri")]:
    print(f"\n--- {dow_label} by signal_type ---")
    for sig_type in sorted(all_sig_types):
        for h_th in [0.30, 0.25, 0.20, 0.15, 0.10]:
            def make_type_gate(ed=extra_dow, st=sig_type, ht=h_th):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd.weekday() in ed:
                            if day.cs_signal_type == st and day.cs_channel_health >= ht:
                                return result
                    return None
                return fn
            trades = simulate_trades(signals, make_type_gate(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if n > 426:
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                flag = "***" if wr >= 100 else ""
                print(f"  type={st} h>={ht:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

    print(f"\n--- {dow_label} by primary_tf ---")
    for pri_tf in sorted(all_pri_tfs):
        for h_th in [0.30, 0.25, 0.20, 0.15, 0.10]:
            def make_tf_gate(ed=extra_dow, pt=pri_tf, ht=h_th):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd.weekday() in ed:
                            if day.cs_primary_tf == pt and day.cs_channel_health >= ht:
                                return result
                    return None
                return fn
            trades = simulate_trades(signals, make_tf_gate(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if n > 426:
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                flag = "***" if wr >= 100 else ""
                print(f"  tf={pt} h>={ht:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Part 6: Stop/TP ratio as quality filter
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 6: Stop/TP ratio filter for unsafe days")
print("="*70)

for extra_dow, dow_label in [({1,3,4}, "TuThFr"), ({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri")]:
    print(f"\n--- {dow_label} ---")
    # R:R ratio (tp/stop) as quality gate
    for rr_min in [3.0, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0]:
        for h_th in [0.25, 0.20, 0.15, 0.10]:
            def make_rr_gate(ed=extra_dow, rr=rr_min, ht=h_th):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd.weekday() in ed:
                            stop = day.cs_stop_pct if day.cs_stop_pct else 2.0
                            tp = day.cs_tp_pct if day.cs_tp_pct else 4.0
                            rr_ratio = tp / stop if stop > 0 else 0
                            if rr_ratio >= rr and day.cs_channel_health >= ht:
                                return result
                    return None
                return fn
            trades = simulate_trades(signals, make_rr_gate(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if n > 426:
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                flag = "***" if wr >= 100 else ""
                print(f"  R:R>={rr:.1f} h>={ht:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Part 7: Position score as quality filter (inverted — low pos = deep in channel)
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 7: Position score filter for unsafe days")
print("="*70)

for extra_dow, dow_label in [({1,3,4}, "TuThFr"), ({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri")]:
    print(f"\n--- {dow_label} ---")
    for pos_max in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        for h_th in [0.25, 0.20, 0.15, 0.10]:
            def make_pos_gate(ed=extra_dow, pm=pos_max, ht=h_th):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd.weekday() in ed:
                            if day.cs_position_score <= pm and day.cs_channel_health >= ht:
                                return result
                    return None
                return fn
            trades = simulate_trades(signals, make_pos_gate(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if n > 426:
                pnl = sum(t.pnl for t in trades)
                bl = min(t.pnl for t in trades)
                flag = "***" if wr >= 100 else ""
                print(f"  pos<={pm:.2f} h>={ht:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Part 8: Multi-feature AND gates (3 features at once)
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 8: Triple AND-gates for Tue/Thu/Fri")
print("="*70)

for extra_dow, dow_label in [({1,3,4}, "TuThFr"), ({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri")]:
    print(f"\n--- {dow_label}: h + confl + energy/timing ---")
    for h_th in [0.25, 0.20, 0.15, 0.10]:
        for confl_th in [0.80, 0.70, 0.60, 0.50]:
            for e_th in [0.70, 0.60, 0.50, 0.40]:
                def make_triple(ed=extra_dow, ht=h_th, ct=confl_th, et=e_th):
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
                                    day.cs_energy_score is not None and day.cs_energy_score >= et):
                                    return result
                        return None
                    return fn
                trades = simulate_trades(signals, make_triple(), 'test', cooldown=0, trail_power=6)
                n = len(trades)
                w = sum(1 for t in trades if t.pnl > 0)
                wr = w/n*100 if n else 0
                if wr >= 100 and n > 426:
                    pnl = sum(t.pnl for t in trades)
                    print(f"  h>={ht:.2f}&confl>={ct:.2f}&e>={et:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} ***")

    print(f"\n--- {dow_label}: h + confl + timing ---")
    for h_th in [0.25, 0.20, 0.15, 0.10]:
        for confl_th in [0.80, 0.70, 0.60, 0.50]:
            for t_th in [0.70, 0.60, 0.50, 0.40]:
                def make_triple_t(ed=extra_dow, ht=h_th, ct=confl_th, tt=t_th):
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
                                    day.cs_timing_score is not None and day.cs_timing_score >= tt):
                                    return result
                        return None
                    return fn
                trades = simulate_trades(signals, make_triple_t(), 'test', cooldown=0, trail_power=6)
                n = len(trades)
                w = sum(1 for t in trades if t.pnl > 0)
                wr = w/n*100 if n else 0
                if wr >= 100 and n > 426:
                    pnl = sum(t.pnl for t in trades)
                    print(f"  h>={ht:.2f}&confl>={ct:.2f}&t>={tt:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Part 9: V5 expansion — try relaxing V5 filters
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 9: V5 filter relaxation")
print("="*70)

for v5h_max in [0.60, 0.65, 0.70, 0.80, 1.0]:
    for v5p_max in [0.85, 0.90, 0.95, 1.0]:
        def make_v5_relax(hm=v5h_max, pm=v5p_max):
            def fn(day):
                result = cz_fn(day)
                if result is not None:
                    return result
                # Try wider V5
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
            print(f"  V5 h<{hm:.2f} pos<{pm:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════
# Part 10: Trail power variation per condition
# ════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 10: Trail power variation")
print("="*70)

for tp in [4, 5, 7, 8, 10]:
    trades = simulate_trades(signals, cz_fn, 'CZ', cooldown=0, trail_power=tp)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    print(f"  trail_power={tp}: {n}t {wr:.1f}% ${pnl:+,.0f} BL=${bl:+,.0f}")

print("\nDone")
