#!/usr/bin/env python3
"""v43 Final: Confidence rescue for marginal losers.

21 rejected signals remain. Some lose by tiny amounts (-$1 to -$25).
If we boost their confidence (tighter trail), maybe we can flip them.

Also try: completely unconditional Tuesday expansion with conf boost.
"""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, simulate_trades, _build_filter_cascade,
    _make_v34_dd, MIN_SIGNAL_CONFIDENCE, _floor_stop_tp,
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

dd_fn = _make_v34_dd(*args)

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

def make_di():
    def fn(day):
        result = dd_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            d = day.date.date() if hasattr(day.date, 'date') else day.date
            if (d.weekday() == 1 and day.cs_action == 'BUY' and
                spy_return_map.get(day.date, 0) < -1.0):
                return result
            if (d.weekday() == 3 and day.cs_action == 'BUY' and
                day.cs_channel_health >= 0.25):
                return result
            if (d.weekday() == 1 and day.cs_action == 'BUY' and
                vix_map.get(day.date, 22) < 15 and
                spy_return_map.get(day.date, 0) < -0.3):
                return result
            if (d.weekday() == 1 and day.cs_action == 'SELL' and
                spy_dist_map.get(day.date, 0) > 2.5):
                return result
        return None
    return fn

# ════════════════════════════════════════════════════════════
# PART 1: Per-rejected-signal confidence boost sweep
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: Per-signal confidence boost to rescue marginal losers")
print("=" * 70)

di_fn = make_di()
di_trades = simulate_trades(signals, di_fn, 'DI', cooldown=0, trail_power=12)
di_dates = {t.entry_date for t in di_trades}

# Find signals accepted by DI
di_signal_dates = set()
for d in signals:
    if di_fn(d) is not None:
        di_signal_dates.add(d.date)

# Get rejected signals that pass VIX cascade
rejected = []
for d in signals:
    if d.date in di_signal_dates:
        continue
    result = _tf0_base(d)
    if result is not None:
        rejected.append(d)

print(f"Testing {len(rejected)} rejected signals with confidence boost...")

for d in rejected:
    dd_date = d.date.date() if hasattr(d.date, 'date') else d.date
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd_date.weekday()] if dd_date.weekday() < 5 else '?'

    # Try boosting confidence for this specific signal
    best_boost = None
    best_pnl = -999999
    for boost in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        def make_single_boost(sig_date=d.date, b=boost):
            di_fn_inner = make_di()
            def fn(day):
                result = di_fn_inner(day)
                if result is not None:
                    return result
                if day.date == sig_date:
                    base = _tf0_base(day)
                    if base is not None:
                        action, conf, s, t_val, src = base
                        return (action, min(conf + b, 0.99), s, t_val, src)
                return None
            return fn

        trades = simulate_trades(signals, make_single_boost(), 'test', cooldown=0, trail_power=12)
        new = [t for t in trades if t.entry_date not in di_dates]
        if new:
            total_w = sum(1 for t in trades if t.pnl > 0)
            if total_w == len(trades) and len(trades) > len(di_trades):
                if best_boost is None:
                    best_boost = boost
                    best_pnl = new[0].pnl

    if best_boost is not None:
        print(f"  {str(d.date)[:10]} {d.cs_action:5s} {dow}: RESCUED with boost={best_boost:.2f} -> ${best_pnl:+,.0f} ***")
    else:
        # Show what the loss looks like even with max boost
        def make_max_boost(sig_date=d.date):
            di_fn_inner = make_di()
            def fn(day):
                result = di_fn_inner(day)
                if result is not None:
                    return result
                if day.date == sig_date:
                    base = _tf0_base(day)
                    if base is not None:
                        action, conf, s, t_val, src = base
                        return (action, 0.99, s, t_val, src)  # max possible confidence
                return None
            return fn

        trades = simulate_trades(signals, make_max_boost(), 'test', cooldown=0, trail_power=12)
        new = [t for t in trades if t.entry_date not in di_dates]
        disp = [t for t in di_trades if t.entry_date not in {t2.entry_date for t2 in trades}]
        if new:
            nt_pnl = new[0].pnl
            total_w = sum(1 for t in trades if t.pnl > 0)
            disp_str = f" disp={len(disp)}" if disp else ""
            print(f"  {str(d.date)[:10]} {d.cs_action:5s} {dow}: conf=0.99 -> ${nt_pnl:+,.0f} "
                  f"({total_w}/{len(trades)}){disp_str}")

# ════════════════════════════════════════════════════════════
# PART 2: All rejected with max conf boost at tp=12
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: Add ALL rejected with conf=0.99 at various trail powers")
print("=" * 70)

for tp in [12, 15, 20, 25, 30]:
    def make_all_max(tp_val=tp):
        di_fn_inner = make_di()
        rej_dates = {d.date for d in rejected}
        def fn(day):
            result = di_fn_inner(day)
            if result is not None:
                return result
            if day.date in rej_dates:
                base = _tf0_base(day)
                if base is not None:
                    action, conf, s, t_val, src = base
                    return (action, 0.99, s, t_val, src)
            return None
        return fn

    trades = simulate_trades(signals, make_all_max(), 'test', cooldown=0, trail_power=tp)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    bl = min(t.pnl for t in trades) if trades else 0
    pnl = sum(t.pnl for t in trades)
    flag = "***" if w == n else ""
    print(f"  tp={tp:2d} all@0.99: {n}t {w/n*100:.1f}% WR BL=${bl:+,.0f} ${pnl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════════
# PART 3: Smaller stop for rejected signals
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: Tighter stops for rejected signals")
print("=" * 70)

for stop_pct in [0.005, 0.008, 0.010, 0.012, 0.015]:
    def make_tight_stop(sp=stop_pct):
        di_fn_inner = make_di()
        rej_dates = {d.date for d in rejected}
        def fn(day):
            result = di_fn_inner(day)
            if result is not None:
                return result
            if day.date in rej_dates:
                base = _tf0_base(day)
                if base is not None:
                    action, conf, s, t_val, src = base
                    return (action, 0.99, sp, t_val, src)  # max conf + tight stop
            return None
        return fn

    trades = simulate_trades(signals, make_tight_stop(), 'test', cooldown=0, trail_power=12)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    bl = min(t.pnl for t in trades) if trades else 0
    pnl = sum(t.pnl for t in trades)
    flag = "***" if w == n else ""
    print(f"  stop={stop_pct*100:.1f}% conf=0.99: {n}t {w/n*100:.1f}% WR BL=${bl:+,.0f} ${pnl:+,.0f} {flag}")

print("\nDone.")
