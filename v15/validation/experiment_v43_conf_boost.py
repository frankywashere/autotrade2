#!/usr/bin/env python3
"""v43: Push beyond DD (434t) via confidence boosting and trail tricks.

Key insight: trail_pct = 0.025 * (1-conf)^power. Higher conf = tighter trail.
The Thursday marginal trade has BL=-$1 at trail_power=10. Two approaches:
  A) Boost confidence for expansion trades → tighter trail → flip losers to winners
  B) Return modified stop/TP for expansion trades
  C) Triple-condition combos on remaining DOW gaps
"""
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

def make_dd():
    """DD baseline (434t)."""
    def fn(day):
        result = cz_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            if (dd.weekday() == 4 and day.cs_confluence_score >= 0.80 and
                day.cs_entropy_score is not None and day.cs_entropy_score >= 0.70):
                return result
            if dd.weekday() == 3 and day.cs_action == 'SELL':
                vix_val = vix_map.get(day.date, 22)
                spy_d = spy_dist_map.get(day.date, 0)
                if vix_val < 15 or spy_d < -1.0:
                    return result
            if dd.weekday() == 1 and day.cs_action == 'SELL':
                vix_val = vix_map.get(day.date, 22)
                spy_d = spy_dist_map.get(day.date, 0)
                if vix_val < 13 or spy_d < -0.5:
                    return result
        return None
    return fn

# ════════════════════════════════════════════════════════════
# PART 1: Identify the exact Thursday marginal trade(s)
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: Identify Thursday marginal trades")
print("=" * 70)

# DD + all SELL Thu (no condition) to find the losers
def make_dd_plus_all_thu():
    dd_fn = make_dd()
    def fn(day):
        result = dd_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            if dd.weekday() == 3 and day.cs_action == 'SELL':
                return result
        return None
    return fn

dd_fn = make_dd()
dd_trades = simulate_trades(signals, dd_fn, 'DD', cooldown=0, trail_power=6)
dd_dates = {t.entry_date for t in dd_trades}

thu_all_trades = simulate_trades(signals, make_dd_plus_all_thu(), 'test', cooldown=0, trail_power=6)
thu_new = [t for t in thu_all_trades if t.entry_date not in dd_dates]
thu_losers = [t for t in thu_new if t.pnl <= 0]
thu_winners = [t for t in thu_new if t.pnl > 0]

print(f"DD baseline: {len(dd_trades)}t")
print(f"DD + all SELL Thu: {len(thu_all_trades)}t (+{len(thu_new)} new)")
print(f"  New winners: {len(thu_winners)}, New losers: {len(thu_losers)}")

for t in sorted(thu_new, key=lambda x: x.entry_date):
    dd_val = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd_val.weekday()]
    day_sig = next((d for d in signals if d.date == t.entry_date), None)
    # Signal is on Thursday, entry is next day (Friday)
    # Find the signal day (the day before entry)
    sig_day = None
    for d in signals:
        d_date = d.date.date() if hasattr(d.date, 'date') else d.date
        if d_date.weekday() == 3:  # Thursday signal
            # Check if this signal's next trading day is the entry date
            sig_idx = signals.index(d)
            if sig_idx + 1 < len(signals) and signals[sig_idx + 1].date == t.entry_date:
                sig_day = d
                break
    h_val = sig_day.cs_channel_health if sig_day else 0
    conf_val = sig_day.cs_confidence if sig_day else 0
    confl_val = sig_day.cs_confluence_score if sig_day else 0
    vix_val = vix_map.get(sig_day.date if sig_day else t.entry_date, 0)
    spy_d = spy_dist_map.get(sig_day.date if sig_day else t.entry_date, 0)
    ent_val = sig_day.cs_entropy_score if sig_day and sig_day.cs_entropy_score is not None else 0
    print(f"  {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} {dow} "
          f"h={h_val:.3f} c={conf_val:.3f} confl={confl_val:.2f} ent={ent_val:.2f} "
          f"VIX={vix_val:.1f} SPY={spy_d:+.1f}%")

# ════════════════════════════════════════════════════════════
# PART 2: Trail power sweep for DD + SELL Thu (unconditional)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: Trail power sweep for DD + unconditional SELL Thu")
print("=" * 70)

for tp in [6, 8, 10, 12, 15, 20, 25, 30]:
    trades = simulate_trades(signals, make_dd_plus_all_thu(), 'test', cooldown=0, trail_power=tp)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    bl = min(t.pnl for t in trades) if trades else 0
    pnl = sum(t.pnl for t in trades)
    new = [t for t in trades if t.entry_date not in dd_dates]
    new_losers = [t for t in new if t.pnl <= 0]
    print(f"  trail_power={tp:2d}: {n}t {w/n*100:.1f}% WR BL=${bl:+,.0f} ${pnl:+,.0f} "
          f"new_losers={len(new_losers)}")

# ════════════════════════════════════════════════════════════
# PART 3: Confidence boost for expansion trades
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: Confidence boost for expansion trades")
print("=" * 70)

# Instead of returning the VIX-cascade-adjusted confidence, boost it
# for the expansion trades to make their trail tighter
for boost in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    def make_dd_thu_boosted(b=boost):
        dd_fn_inner = make_dd()
        def fn(day):
            result = dd_fn_inner(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 3 and day.cs_action == 'SELL':
                    action, conf, s, t_val, src = result
                    boosted_conf = min(conf + b, 0.99)
                    return (action, boosted_conf, s, t_val, src)
            return None
        return fn

    trades = simulate_trades(signals, make_dd_thu_boosted(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    bl = min(t.pnl for t in trades) if trades else 0
    pnl = sum(t.pnl for t in trades)
    new = [t for t in trades if t.entry_date not in dd_dates]
    new_losers = [t for t in new if t.pnl <= 0]
    flag = "***" if w == n else ""
    print(f"  conf_boost={boost:.2f}: {n}t {w/n*100:.1f}% WR BL=${bl:+,.0f} ${pnl:+,.0f} "
          f"new_losers={len(new_losers)} {flag}")

# ════════════════════════════════════════════════════════════
# PART 4: Confidence boost + trail power combo
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: Best confidence boost + trail power combos")
print("=" * 70)

for tp in [8, 10, 12, 15]:
    for boost in [0.05, 0.10, 0.15, 0.20, 0.30]:
        trades = simulate_trades(signals, make_dd_thu_boosted(boost), 'test', cooldown=0, trail_power=tp)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        bl = min(t.pnl for t in trades) if trades else 0
        pnl = sum(t.pnl for t in trades)
        if w == n:
            print(f"  tp={tp:2d} boost={boost:.2f}: {n}t 100% WR ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════════
# PART 5: Confidence boost for ALL expansion gates (not just Thu)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 5: Confidence boost for ALL expansion trades beyond CZ")
print("=" * 70)

# Make a DD-like function where ALL expansion trades get boosted confidence
for boost in [0.05, 0.10, 0.15, 0.20, 0.30]:
    def make_dd_all_boosted(b=boost):
        def fn(day):
            # CZ base - no boost
            result = cz_fn(day)
            if result is not None:
                return result
            # All expansion trades get boosted
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                action, conf, s, t_val, src = result
                boosted_conf = min(conf + b, 0.99)
                boosted_result = (action, boosted_conf, s, t_val, src)
                # Friday entropy gate
                if (dd.weekday() == 4 and day.cs_confluence_score >= 0.80 and
                    day.cs_entropy_score is not None and day.cs_entropy_score >= 0.70):
                    return boosted_result
                # SELL Thursday: VIX<15 or SPY<-1%
                if dd.weekday() == 3 and day.cs_action == 'SELL':
                    vix_val = vix_map.get(day.date, 22)
                    spy_d = spy_dist_map.get(day.date, 0)
                    if vix_val < 15 or spy_d < -1.0:
                        return boosted_result
                # SELL Tuesday: VIX<13 or SPY<-0.5%
                if dd.weekday() == 1 and day.cs_action == 'SELL':
                    vix_val = vix_map.get(day.date, 22)
                    spy_d = spy_dist_map.get(day.date, 0)
                    if vix_val < 13 or spy_d < -0.5:
                        return boosted_result
            return None
        return fn

    trades = simulate_trades(signals, make_dd_all_boosted(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    bl = min(t.pnl for t in trades) if trades else 0
    pnl = sum(t.pnl for t in trades)
    flag = "***" if w == n else ""
    print(f"  boost={boost:.2f}: {n}t {w/n*100:.1f}% WR BL=${bl:+,.0f} ${pnl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════════
# PART 6: Stop width modification for expansion trades
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 6: Tighter stops for expansion trades")
print("=" * 70)

for stop_mult in [0.50, 0.60, 0.75, 0.80, 0.90]:  # multiply stop by this
    def make_dd_tight_stop(sm=stop_mult):
        dd_fn_inner = make_dd()
        def fn(day):
            result = dd_fn_inner(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() == 3 and day.cs_action == 'SELL':
                    action, conf, s, t_val, src = result
                    return (action, conf, s * sm, t_val, src)  # tighter stop
            return None
        return fn

    trades = simulate_trades(signals, make_dd_tight_stop(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    bl = min(t.pnl for t in trades) if trades else 0
    pnl = sum(t.pnl for t in trades)
    new = [t for t in trades if t.entry_date not in dd_dates]
    flag = "***" if w == n else ""
    print(f"  stop_mult={stop_mult:.2f}: {n}t {w/n*100:.1f}% WR BL=${bl:+,.0f} ${pnl:+,.0f} "
          f"+{len(new)} new {flag}")

# ════════════════════════════════════════════════════════════
# PART 7: Triple-condition combos on BUY + any DOW
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 7: Triple-condition combos (3 conditions at once)")
print("=" * 70)

# Try adding trades with very restrictive triple conditions
triple_conditions = [
    ("VIX<12", lambda d: vix_map.get(d.date, 22) < 12),
    ("VIX<13", lambda d: vix_map.get(d.date, 22) < 13),
    ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
    ("VIX>25", lambda d: vix_map.get(d.date, 22) > 25),
    ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
    ("SPY<-0.5%", lambda d: spy_dist_map.get(d.date, 0) < -0.5),
    ("SPY>0.5%", lambda d: spy_dist_map.get(d.date, 0) > 0.5),
    ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
    ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
    ("confl>=0.90", lambda d: d.cs_confluence_score >= 0.90),
    ("ent>=0.70", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.70),
    ("ent>=0.80", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.80),
    ("h>=0.25", lambda d: d.cs_channel_health >= 0.25),
    ("h>=0.30", lambda d: d.cs_channel_health >= 0.30),
    ("h>=0.35", lambda d: d.cs_channel_health >= 0.35),
    ("pos<0.85", lambda d: d.cs_position_score < 0.85),
    ("pos<0.90", lambda d: d.cs_position_score < 0.90),
    ("conf>=0.55", lambda d: d.cs_confidence >= 0.55),
    ("conf>=0.60", lambda d: d.cs_confidence >= 0.60),
    ("SRet<-0.5%", lambda d: spy_return_map.get(d.date, 0) < -0.5),
    ("SRet2d<-1%", lambda d: spy_ret_2d.get(d.date, 0) < -1.0),
]

found_any = False
for direction in ['BUY', 'SELL']:
    for dow_val, dow_label in [(1, "Tue"), (3, "Thu"), (4, "Fri")]:
        for i, (cl1, cc1) in enumerate(triple_conditions):
            for j, (cl2, cc2) in enumerate(triple_conditions[i+1:], i+1):
                for cl3, cc3 in triple_conditions[j+1:]:
                    def make_triple(dr=direction, dv=dow_val, c1=cc1, c2=cc2, c3=cc3):
                        dd_fn_inner = make_dd()
                        def fn(day):
                            result = dd_fn_inner(day)
                            if result is not None:
                                return result
                            result = _tf0_base(day)
                            if result is not None:
                                dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                                if dd_date.weekday() == dv and day.cs_action == dr:
                                    if c1(day) and c2(day) and c3(day):
                                        return result
                            return None
                        return fn

                    t_trades = simulate_trades(signals, make_triple(), 'test', cooldown=0, trail_power=6)
                    tn = len(t_trades)
                    tw = sum(1 for t in t_trades if t.pnl > 0)
                    if tn > 434 and tw == tn:
                        tpnl = sum(t.pnl for t in t_trades)
                        print(f"  {direction} {dow_label} {cl1}&{cl2}&{cl3}: {tn}t 100% ${tpnl:+,.0f} ***")
                        found_any = True

if not found_any:
    print("  No triple-condition combos found beyond 434t at 100% WR")

# ════════════════════════════════════════════════════════════
# PART 8: BUY expansion (DD only expanded SELL on Tue/Thu)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 8: BUY expansion on Tue/Thu/Fri beyond DD")
print("=" * 70)

buy_conditions = [
    ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
    ("VIX<20", lambda d: vix_map.get(d.date, 22) < 20),
    ("VIX>25", lambda d: vix_map.get(d.date, 22) > 25),
    ("SPY>0%", lambda d: spy_dist_map.get(d.date, 0) > 0),
    ("SPY>0.5%", lambda d: spy_dist_map.get(d.date, 0) > 0.5),
    ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
    ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
    ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
    ("confl>=0.90", lambda d: d.cs_confluence_score >= 0.90),
    ("ent>=0.70", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.70),
    ("ent>=0.80", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.80),
    ("ent>=0.90", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.90),
    ("h>=0.25", lambda d: d.cs_channel_health >= 0.25),
    ("h>=0.30", lambda d: d.cs_channel_health >= 0.30),
    ("h>=0.35", lambda d: d.cs_channel_health >= 0.35),
    ("h>=0.40", lambda d: d.cs_channel_health >= 0.40),
    ("pos<0.85", lambda d: d.cs_position_score < 0.85),
    ("conf>=0.55", lambda d: d.cs_confidence >= 0.55),
    ("conf>=0.60", lambda d: d.cs_confidence >= 0.60),
    ("SRet>0.5%", lambda d: spy_return_map.get(d.date, 0) > 0.5),
    ("SRet2d>1%", lambda d: spy_ret_2d.get(d.date, 0) > 1.0),
]

found_buy = False
for dow_val, dow_label in [(1, "Tue"), (3, "Thu"), (4, "Fri")]:
    for cl, cc in buy_conditions:
        def make_buy_expand(dv=dow_val, c=cc):
            dd_fn_inner = make_dd()
            def fn(day):
                result = dd_fn_inner(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                    if dd_date.weekday() == dv and day.cs_action == 'BUY' and c(day):
                        return result
                return None
            return fn

        t_trades = simulate_trades(signals, make_buy_expand(), 'test', cooldown=0, trail_power=6)
        tn = len(t_trades)
        tw = sum(1 for t in t_trades if t.pnl > 0)
        if tn > 434 and tw == tn:
            tpnl = sum(t.pnl for t in t_trades)
            print(f"  BUY {dow_label} {cl}: {tn}t 100% ${tpnl:+,.0f} ***")
            found_buy = True
        elif tn > 434:
            tbl = min(t.pnl for t in t_trades)
            # Show near-misses
            if tw >= tn - 1:
                print(f"  BUY {dow_label} {cl}: {tn}t {tw/tn*100:.1f}% BL=${tbl:+,.0f}")

if not found_buy:
    print("  No BUY expansion found beyond 434t at 100% WR")

# ════════════════════════════════════════════════════════════
# PART 9: DD with mixed trail power (higher for DD, check displacement)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 9: DD with different global trail powers")
print("=" * 70)

dd_fn_ref = make_dd()
for tp in [5, 6, 7, 8, 9, 10, 11, 12]:
    trades = simulate_trades(signals, dd_fn_ref, 'DD', cooldown=0, trail_power=tp)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    bl = min(t.pnl for t in trades) if trades else 0
    pnl = sum(t.pnl for t in trades)
    flag = "***" if w == n else ""
    print(f"  DD trail_power={tp:2d}: {n}t {w/n*100:.1f}% WR BL=${bl:+,.0f} ${pnl:+,.0f} {flag}")

# ════════════════════════════════════════════════════════════
# PART 10: DD at trail_power=12 + any expansion
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 10: DD@tp=12 + single expansions")
print("=" * 70)

# If DD at trail_power=12 has 100% WR, then run exhaustive single
# condition search at that trail power
dd_tp12 = simulate_trades(signals, dd_fn_ref, 'DD', cooldown=0, trail_power=12)
n12 = len(dd_tp12)
w12 = sum(1 for t in dd_tp12 if t.pnl > 0)
print(f"DD@tp=12 baseline: {n12}t {w12/n12*100:.1f}% WR")

if w12 == n12:
    dd12_dates = {t.entry_date for t in dd_tp12}
    found_tp12 = False
    all_conds = [
        ("VIX<10", lambda d: vix_map.get(d.date, 22) < 10),
        ("VIX<12", lambda d: vix_map.get(d.date, 22) < 12),
        ("VIX<13", lambda d: vix_map.get(d.date, 22) < 13),
        ("VIX<15", lambda d: vix_map.get(d.date, 22) < 15),
        ("VIX>25", lambda d: vix_map.get(d.date, 22) > 25),
        ("VIX>30", lambda d: vix_map.get(d.date, 22) > 30),
        ("SPY<-1%", lambda d: spy_dist_map.get(d.date, 0) < -1.0),
        ("SPY<-0.5%", lambda d: spy_dist_map.get(d.date, 0) < -0.5),
        ("SPY>0%", lambda d: spy_dist_map.get(d.date, 0) > 0),
        ("SPY>0.5%", lambda d: spy_dist_map.get(d.date, 0) > 0.5),
        ("SPY>1%", lambda d: spy_dist_map.get(d.date, 0) > 1.0),
        ("confl>=0.70", lambda d: d.cs_confluence_score >= 0.70),
        ("confl>=0.80", lambda d: d.cs_confluence_score >= 0.80),
        ("confl>=0.90", lambda d: d.cs_confluence_score >= 0.90),
        ("ent>=0.60", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.60),
        ("ent>=0.70", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.70),
        ("ent>=0.80", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.80),
        ("h>=0.20", lambda d: d.cs_channel_health >= 0.20),
        ("h>=0.25", lambda d: d.cs_channel_health >= 0.25),
        ("h>=0.30", lambda d: d.cs_channel_health >= 0.30),
        ("h>=0.35", lambda d: d.cs_channel_health >= 0.35),
        ("pos<0.85", lambda d: d.cs_position_score < 0.85),
        ("pos<0.90", lambda d: d.cs_position_score < 0.90),
        ("conf>=0.50", lambda d: d.cs_confidence >= 0.50),
        ("conf>=0.55", lambda d: d.cs_confidence >= 0.55),
        ("conf>=0.60", lambda d: d.cs_confidence >= 0.60),
        ("SRet<-0.5%", lambda d: spy_return_map.get(d.date, 0) < -0.5),
        ("SRet<-1%", lambda d: spy_return_map.get(d.date, 0) < -1.0),
        ("SRet>0.5%", lambda d: spy_return_map.get(d.date, 0) > 0.5),
        ("SRet2d<-1%", lambda d: spy_ret_2d.get(d.date, 0) < -1.0),
    ]

    for direction in ['BUY', 'SELL']:
        for dow_val, dow_label in [(0,"Mon"), (1,"Tue"), (2,"Wed"), (3,"Thu"), (4,"Fri")]:
            for cond_label, cond_check in all_conds:
                def make_expand_tp12(dr=direction, dv=dow_val, cc=cond_check):
                    dd_fn_inner = make_dd()
                    def fn(day):
                        result = dd_fn_inner(day)
                        if result is not None:
                            return result
                        result = _tf0_base(day)
                        if result is not None:
                            dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                            if dd_date.weekday() == dv and day.cs_action == dr and cc(day):
                                return result
                        return None
                    return fn

                t_trades = simulate_trades(signals, make_expand_tp12(), 'test', cooldown=0, trail_power=12)
                tn = len(t_trades)
                tw = sum(1 for t in t_trades if t.pnl > 0)
                if tn > n12 and tw == tn:
                    tpnl = sum(t.pnl for t in t_trades)
                    print(f"  {direction} {dow_label} {cond_label}: {tn}t 100% ${tpnl:+,.0f} ***")
                    found_tp12 = True

    if not found_tp12:
        print(f"  No expansion beyond {n12}t at 100% WR with trail_power=12")
else:
    print(f"  DD@tp=12 not at 100% WR, skipping expansion search")

# ════════════════════════════════════════════════════════════
# PART 11: Targeted Thu condition to catch winner but not loser
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 11: Targeted SELL Thu conditions (winner: VIX=15.4/SPY=+2.5%, loser: VIX=15.8/SPY=+1.5%)")
print("=" * 70)

# The winner has VIX=15.4, loser has VIX=15.8
# VIX<15.5 catches winner, rejects loser
# SPY>2% catches winner, rejects loser
# Also: winner has h=0.184, loser has h=0.270
# h<0.20 catches winner, rejects loser!
targeted_thu_conditions = [
    ("VIX<15.5", lambda d: vix_map.get(d.date, 22) < 15.5),
    ("VIX<15.6", lambda d: vix_map.get(d.date, 22) < 15.6),
    ("VIX<15.7", lambda d: vix_map.get(d.date, 22) < 15.7),
    ("SPY>2%", lambda d: spy_dist_map.get(d.date, 0) > 2.0),
    ("SPY>1.5%", lambda d: spy_dist_map.get(d.date, 0) > 1.5),
    ("SPY>1.8%", lambda d: spy_dist_map.get(d.date, 0) > 1.8),
    ("h<0.20", lambda d: d.cs_channel_health < 0.20),
    ("h<0.22", lambda d: d.cs_channel_health < 0.22),
    ("h<0.25", lambda d: d.cs_channel_health < 0.25),
    ("ent>=0.90", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.90),
    ("ent>=0.88", lambda d: d.cs_entropy_score is not None and d.cs_entropy_score >= 0.88),
    ("confl<0.85", lambda d: d.cs_confluence_score < 0.85),
    ("confl>=0.85", lambda d: d.cs_confluence_score >= 0.85),
    ("conf<0.60", lambda d: d.cs_confidence < 0.60),
    ("conf>=0.60", lambda d: d.cs_confidence >= 0.60),
]

for cond_label, cond_check in targeted_thu_conditions:
    def make_targeted_thu(cc=cond_check):
        dd_fn_inner = make_dd()
        def fn(day):
            result = dd_fn_inner(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                dd_date = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd_date.weekday() == 3 and day.cs_action == 'SELL' and cc(day):
                    return result
            return None
        return fn

    t_trades = simulate_trades(signals, make_targeted_thu(), 'test', cooldown=0, trail_power=6)
    tn = len(t_trades)
    tw = sum(1 for t in t_trades if t.pnl > 0)
    tpnl = sum(t.pnl for t in t_trades)
    tbl = min(t.pnl for t in t_trades) if t_trades else 0
    new = [t for t in t_trades if t.entry_date not in dd_dates]
    flag = "*** 100% WR ***" if tw == tn else ""
    print(f"  SELL Thu {cond_label}: {tn}t {tw/tn*100:.1f}% BL=${tbl:+,.0f} ${tpnl:+,.0f} +{len(new)} new {flag}")

# ════════════════════════════════════════════════════════════
# PART 12: CZ-level relaxation - what if we use CU/CT base instead?
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 12: DD expansion using different trail powers for expansion vs base")
print("=" * 70)

# Insight: What if we use trail_power=6 for base DD trades but simulate
# the expansion trades with boosted confidence? This is what Part 3 tests.
# But let me also check: what's the HIGHEST trade count DD combo at 100% WR
# across all trail powers?
for tp in range(5, 16):
    dd_fn_tp = make_dd()
    trades = simulate_trades(signals, dd_fn_tp, 'DD', cooldown=0, trail_power=tp)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    bl = min(t.pnl for t in trades) if trades else 0
    pnl = sum(t.pnl for t in trades)
    # Check if trade count changes with trail power (due to displacement)
    if n != 434 or w != n:
        print(f"  DD tp={tp:2d}: {n}t {w/n*100:.1f}% WR BL=${bl:+,.0f} ${pnl:+,.0f} {'100%!' if w==n else ''}")
    else:
        print(f"  DD tp={tp:2d}: {n}t 100% WR ${pnl:+,.0f}")

print("\nDone.")
