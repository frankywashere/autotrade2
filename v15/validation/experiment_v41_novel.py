#!/usr/bin/env python3
"""v41: Novel approaches beyond CZ (426 trades, 100% WR).
Try entirely different expansion axes:
1. Conditional Tue/Thu/Fri with multiple guards
2. Trail power variation per condition
3. HOLD-day recovery (days with CS=HOLD but high quality indicators)
4. V5 SELL signal (V5 is currently BUY only)
5. Confl gate with h-AND gates
6. Per-year adaptive thresholds (walk-forward style)"""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
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
print(f"CZ baseline: {len(cz_trades)} trades, "
      f"{sum(1 for t in cz_trades if t.pnl > 0)/len(cz_trades)*100:.1f}% WR")

# ════════════════════════════════════════════════════════
# Idea 1: Tue/Thu/Fri with multi-guard (h AND confl AND conf)
# ════════════════════════════════════════════════════════
print("\n--- Idea 1: Guarded Tue/Thu/Fri on CZ ---")
for extra_dows, label in [({1}, "Tue"), ({3}, "Thu"), ({4}, "Fri"),
                           ({1, 3}, "Tue+Thu"), ({3, 4}, "Thu+Fri"),
                           ({1, 4}, "Tue+Fri"), ({1, 3, 4}, "TuThFr")]:
    for h_pct in range(30, 14, -2):
        h = h_pct / 100.0
        for confl_pct in [50, 60, 70, 80]:
            confl = confl_pct / 100.0
            def make_guarded(ed=extra_dows, hr=h, cf=confl):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    # CZ rejected — try guarded Tue/Thu/Fri
                    result = _tf0_base(day)
                    if result is not None:
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        if dd.weekday() in ed:
                            if (day.cs_channel_health >= hr and
                                day.cs_confluence_score >= cf):
                                return result
                    return None
                return fn
            trades = simulate_trades(signals, make_guarded(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if wr >= 100 and n > 426:
                pnl = sum(t.pnl for t in trades)
                print(f"  {label} h>={h:.2f} confl>={confl:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 2: Confidence-guarded recovery on non-Mon/Wed days
# ════════════════════════════════════════════════════════
print("\n--- Idea 2: Confidence-guarded non-Mon/Wed recovery ---")
for c_pct in [0.50, 0.55, 0.60, 0.65, 0.70]:
    for h_pct in range(30, 14, -2):
        h = h_pct / 100.0
        def make_conf_guard(cv=c_pct, hr=h):
            def fn(day):
                result = cz_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    if day.cs_confidence >= cv and day.cs_channel_health >= hr:
                        return result
                return None
            return fn
        trades = simulate_trades(signals, make_conf_guard(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 426:
            pnl = sum(t.pnl for t in trades)
            print(f"  c>={cv:.2f} h>={h:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 3: VIX-low + conf guard
# ════════════════════════════════════════════════════════
print("\n--- Idea 3: VIX<15 + high-conf recovery ---")
for c_pct in [0.45, 0.50, 0.55, 0.60]:
    for h_pct in range(30, 10, -2):
        h = h_pct / 100.0
        def make_vix_conf(cv=c_pct, hr=h):
            def fn(day):
                result = cz_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    vix = vix_map.get(day.date, 22)
                    if vix < 15 and day.cs_confidence >= cv and day.cs_channel_health >= hr:
                        return result
                return None
            return fn
        trades = simulate_trades(signals, make_vix_conf(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 426:
            pnl = sum(t.pnl for t in trades)
            print(f"  VIX<15 c>={cv:.2f} h>={h:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 4: SPY momentum recovery (SPY trending strongly)
# ════════════════════════════════════════════════════════
print("\n--- Idea 4: SPY momentum recovery ---")
for spy_cond, spy_label in [
    (lambda d: spy_dist_50.get(d.date, 0) > 3.0, "SPY50>3%"),
    (lambda d: spy_dist_50.get(d.date, 0) < -3.0, "SPY50<-3%"),
    (lambda d: abs(spy_return_map.get(d.date, 0)) > 1.5, "|SRet|>1.5%"),
    (lambda d: spy_dist_5.get(d.date, 0) > 1.0, "SPY5>1%"),
    (lambda d: spy_dist_5.get(d.date, 0) < -1.0, "SPY5<-1%"),
]:
    for h_pct in range(30, 10, -2):
        h = h_pct / 100.0
        def make_spy(sc=spy_cond, hr=h):
            def fn(day):
                result = cz_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    if sc(day) and day.cs_channel_health >= hr:
                        return result
                return None
            return fn
        trades = simulate_trades(signals, make_spy(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 426:
            pnl = sum(t.pnl for t in trades)
            print(f"  {spy_label} h>={h:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 5: Direction-specific conditional (BUY-only or SELL-only)
# ════════════════════════════════════════════════════════
print("\n--- Idea 5: Direction-specific recovery on CZ ---")
for direction in ['BUY', 'SELL']:
    for cond, label in [
        (lambda d: vix_map.get(d.date, 22) < 15, "VIX<15"),
        (lambda d: vix_map.get(d.date, 22) > 30, "VIX>30"),
        (lambda d: d.cs_confidence >= 0.60, "c>=0.60"),
        (lambda d: d.cs_confluence_score >= 0.80, "confl>=0.80"),
        (lambda d: spy_dist_map.get(d.date, 0) > 2.0, "SPY>2%"),
        (lambda d: spy_dist_map.get(d.date, 0) < -2.0, "SPY<-2%"),
    ]:
        for h_pct in range(30, 10, -4):
            h = h_pct / 100.0
            def make_dir(dr=direction, cd=cond, hr=h):
                def fn(day):
                    result = cz_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        if day.cs_action == dr and cd(day) and day.cs_channel_health >= hr:
                            return result
                    return None
                return fn
            trades = simulate_trades(signals, make_dir(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if wr >= 100 and n > 426:
                pnl = sum(t.pnl for t in trades)
                print(f"  {dr} {label} h>={h:.2f}: {n}t {wr:.1f}% ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 6: Analyze missed signals — what features correlate with winners?
# ════════════════════════════════════════════════════════
print("\n--- Idea 6: Missed signals analysis ---")
missed_winners = []
missed_losers = []
for day in signals:
    if day.date in cz_dates:
        continue
    result = _tf0_base(day)
    if result is None:
        continue
    # Simulate this one trade
    temp_fn_result = result
    def make_single(r=temp_fn_result):
        def fn(d):
            if d.date == day.date:
                return r
            return cz_fn(d)
        return fn
    # Just check if a standalone trade on this day would win
    # Use the simplest simulation
    test_trades = simulate_trades(signals, make_single(), 'test', cooldown=0, trail_power=6)
    this_trade = next((t for t in test_trades if t.entry_date == day.date), None)
    if this_trade:
        if this_trade.pnl > 0:
            missed_winners.append((day, this_trade))
        else:
            missed_losers.append((day, this_trade))

print(f"Missed winners: {len(missed_winners)}, Missed losers: {len(missed_losers)}")
# Show missed losers to understand why they fail
if missed_losers:
    print("\nMissed losers (why they can't be added):")
    for day, trade in sorted(missed_losers, key=lambda x: x[1].pnl)[:15]:
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        vix = vix_map.get(day.date, 0)
        spy_d = spy_dist_map.get(day.date, 0)
        print(f"  {str(day.date)[:10]} {day.cs_action:4s} ${trade.pnl:+8,.0f} h={day.cs_channel_health:.3f} "
              f"c={day.cs_confidence:.3f} confl={day.cs_confluence_score:.2f} "
              f"pos={day.cs_position_score:.3f} VIX={vix:.1f} SPY={spy_d:+.2f}% {dow}")

# Show missed winners with favorable properties
if missed_winners:
    print(f"\nMissed winners ({len(missed_winners)} total), top 20:")
    for day, trade in sorted(missed_winners, key=lambda x: -x[1].pnl)[:20]:
        dd = day.date.date() if hasattr(day.date, 'date') else day.date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        vix = vix_map.get(day.date, 0)
        print(f"  {str(day.date)[:10]} {day.cs_action:4s} ${trade.pnl:+8,.0f} h={day.cs_channel_health:.3f} "
              f"c={day.cs_confidence:.3f} confl={day.cs_confluence_score:.2f} {dow} VIX={vix:.1f}")

print("\nDone")
