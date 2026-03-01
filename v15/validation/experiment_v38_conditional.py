#!/usr/bin/env python3
"""v38: Conditional relaxation beyond CX (414 trades, 100% WR).
Try lowering h thresholds under specific market conditions:
- VIX ranges, SPY distance from SMA, day-of-week, confidence bands.
All using proper VIX cascade (matching CW/CU architecture)."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, simulate_trades, _build_filter_cascade,
    _make_v27_ct, MIN_SIGNAL_CONFIDENCE, _floor_stop_tp,
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
day_map = {d.date: d for d in signals}

def _tf0_base(day):
    """TF0: CS signal + VIX cascade."""
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

def make_cx():
    """CX baseline: BUY h>=0.38, SELL h>=0.31, V5 h<0.57 pos<0.85."""
    def fn(day):
        result = ct_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            h_thresh = 0.38 if day.cs_action == 'BUY' else 0.31
            if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                return result
        if day.v5_take_bounce:
            if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn

print("=" * 100)
print("v38: CONDITIONAL RELAXATION BEYOND CX (414 trades)")
print("=" * 100)

# Baseline
cx_trades = simulate_trades(signals, make_cx(), 'CX', cooldown=0, trail_power=6)
print(f"\nCX baseline: {len(cx_trades)} trades, "
      f"{sum(1 for t in cx_trades if t.pnl > 0)/len(cx_trades)*100:.1f}% WR, "
      f"${sum(t.pnl for t in cx_trades):+,.0f}")
cx_dates = {t.entry_date for t in cx_trades}

# ════════════════════════════════════════════════════════
# First: Identify ALL missed signals and their properties
# ════════════════════════════════════════════════════════
print("\n--- Missed signals analysis ---")
missed = []
for day in signals:
    if day.date in cx_dates:
        continue
    result = _tf0_base(day)
    if result is not None:
        # This signal passes VIX cascade but fails the h/confl gate
        h_thresh = 0.38 if day.cs_action == 'BUY' else 0.31
        if day.cs_channel_health < h_thresh and day.cs_confluence_score < 0.90:
            missed.append(day)

print(f"Missed VIX-passed signals that fail h/confl gate: {len(missed)}")

# Analyze properties
for day in missed[:20]:
    dd = day.date.date() if hasattr(day.date, 'date') else day.date
    dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
    vix = vix_map.get(day.date, 0)
    spy_d = spy_dist_map.get(day.date, 0)
    spy_d5 = spy_dist_5.get(day.date, 0)
    spy_ret = spy_return_map.get(day.date, 0)
    print(f"  {str(day.date)[:10]} {day.cs_action:4s} h={day.cs_channel_health:.3f} "
          f"c={day.cs_confidence:.3f} confl={day.cs_confluence_score:.2f} "
          f"pos={day.cs_position_score:.3f} VIX={vix:.1f} SPY20={spy_d:+.2f}% "
          f"SPY5={spy_d5:+.2f}% SRet={spy_ret:+.2f}% {dow}")

# ════════════════════════════════════════════════════════
# Idea 1: VIX-conditioned relaxation
# Lower h threshold when VIX is in "safe" ranges
# ════════════════════════════════════════════════════════
print("\n--- Idea 1: VIX-conditioned h relaxation ---")
for vix_cond, vix_label in [
    (lambda v: v < 15, "VIX<15"),
    (lambda v: v > 25, "VIX>25"),
    (lambda v: v < 20, "VIX<20"),
    (lambda v: v > 20, "VIX>20"),
    (lambda v: v < 15 or v > 25, "VIX<15|>25"),
]:
    for h_relax_pct in range(30, 20, -2):
        h_relax = h_relax_pct / 100.0
        def make_vix_relax(vc=vix_cond, hr=h_relax):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_sell = 0.31
                    # VIX-conditioned relaxation
                    vix = vix_map.get(day.date, 22)
                    if vc(vix):
                        h_buy = min(h_buy, hr)
                        h_sell = min(h_sell, hr)
                    h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_vix_relax(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 414:
            pnl = sum(t.pnl for t in trades)
            print(f"  {vix_label} h>={h_relax:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 2: SPY distance-conditioned relaxation
# ════════════════════════════════════════════════════════
print("\n--- Idea 2: SPY distance-conditioned h relaxation ---")
for spy_cond, spy_label in [
    (lambda d: d > 0, "SPY>SMA20"),
    (lambda d: d > 0.55, "SPY>0.55%"),
    (lambda d: d > 1.0, "SPY>1.0%"),
    (lambda d: d < 0, "SPY<SMA20"),
    (lambda d: d < -1.0, "SPY<-1.0%"),
]:
    for h_relax_pct in range(30, 20, -2):
        h_relax = h_relax_pct / 100.0
        def make_spy_relax(sc=spy_cond, hr=h_relax):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_sell = 0.31
                    spy_d = spy_dist_map.get(day.date, 0)
                    if sc(spy_d):
                        h_buy = min(h_buy, hr)
                        h_sell = min(h_sell, hr)
                    h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_spy_relax(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 414:
            pnl = sum(t.pnl for t in trades)
            print(f"  {spy_label} h>={h_relax:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 3: Day-of-week conditioned relaxation
# ════════════════════════════════════════════════════════
print("\n--- Idea 3: Day-of-week conditioned h relaxation ---")
for dow_set, dow_label in [
    ({0}, "Mon"),
    ({4}, "Fri"),
    ({0, 4}, "Mon+Fri"),
    ({1, 2, 3}, "Tue-Thu"),
    ({0, 2, 4}, "MWF"),
    ({3, 4}, "Thu+Fri"),
]:
    for h_relax_pct in range(30, 20, -2):
        h_relax = h_relax_pct / 100.0
        def make_dow_relax(ds=dow_set, hr=h_relax):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_sell = 0.31
                    dd = day.date.date() if hasattr(day.date, 'date') else day.date
                    if dd.weekday() in ds:
                        h_buy = min(h_buy, hr)
                        h_sell = min(h_sell, hr)
                    h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_dow_relax(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 414:
            pnl = sum(t.pnl for t in trades)
            print(f"  {dow_label} h>={h_relax:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 4: Direction-specific conditional relaxation
# Only relax SELL or BUY under conditions
# ════════════════════════════════════════════════════════
print("\n--- Idea 4: SELL-only VIX-conditioned relaxation ---")
for vix_cond, vix_label in [
    (lambda v: v < 15, "VIX<15"),
    (lambda v: v > 25, "VIX>25"),
    (lambda v: v < 20, "VIX<20"),
    (lambda v: v < 15 or v > 25, "VIX<15|>25"),
]:
    for h_sell_pct in range(30, 19, -1):
        h_sell = h_sell_pct / 100.0
        def make_sell_vix(vc=vix_cond, hs=h_sell):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_s = 0.31
                    if day.cs_action == 'SELL':
                        vix = vix_map.get(day.date, 22)
                        if vc(vix):
                            h_s = min(h_s, hs)
                    h_thresh = h_buy if day.cs_action == 'BUY' else h_s
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_sell_vix(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 414:
            pnl = sum(t.pnl for t in trades)
            print(f"  SELL {vix_label} h>={h_sell:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 5: BUY-only conditioned relaxation
# ════════════════════════════════════════════════════════
print("\n--- Idea 5: BUY-only conditioned relaxation ---")
for cond, label in [
    (lambda d: vix_map.get(d.date, 22) > 25, "VIX>25"),
    (lambda d: vix_map.get(d.date, 22) < 15, "VIX<15"),
    (lambda d: spy_dist_map.get(d.date, 0) < -1.0, "SPY<-1%"),
    (lambda d: spy_dist_map.get(d.date, 0) > 1.0, "SPY>1%"),
    (lambda d: d.cs_confidence >= 0.60, "c>=0.60"),
    (lambda d: d.cs_confidence >= 0.50, "c>=0.50"),
    (lambda d: d.cs_position_score >= 0.90, "pos>=0.90"),
    (lambda d: d.cs_position_score < 0.30, "pos<0.30"),
]:
    for h_buy_pct in range(37, 19, -2):
        h_buy = h_buy_pct / 100.0
        def make_buy_cond(c=cond, hb=h_buy):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_b = 0.38
                    if day.cs_action == 'BUY' and c(day):
                        h_b = min(h_b, hb)
                    h_thresh = h_b if day.cs_action == 'BUY' else 0.31
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_buy_cond(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 414:
            pnl = sum(t.pnl for t in trades)
            print(f"  BUY {label} h>={h_buy:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 6: Confl bypass with conditions
# ════════════════════════════════════════════════════════
print("\n--- Idea 6: Conditional confl bypass ---")
for cond, label in [
    (lambda d: vix_map.get(d.date, 22) < 15, "VIX<15"),
    (lambda d: vix_map.get(d.date, 22) > 25, "VIX>25"),
    (lambda d: spy_dist_map.get(d.date, 0) > 0, "SPY>SMA20"),
    (lambda d: d.cs_confidence >= 0.50, "c>=0.50"),
    (lambda d: d.cs_position_score >= 0.85, "pos>=0.85"),
]:
    for confl_pct in range(88, 59, -4):
        confl = confl_pct / 100.0
        def make_confl_cond(c=cond, cf=confl):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_thresh = 0.38 if day.cs_action == 'BUY' else 0.31
                    confl_thresh = 0.90
                    if c(day):
                        confl_thresh = min(confl_thresh, cf)
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= confl_thresh:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_confl_cond(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 414:
            pnl = sum(t.pnl for t in trades)
            print(f"  {label} confl>={confl:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 7: Multi-condition recovery (h AND conf AND confl)
# ════════════════════════════════════════════════════════
print("\n--- Idea 7: Triple-gate recovery (h + conf + confl) ---")
for h_pct in range(20, 38, 3):
    h = h_pct / 100.0
    for c_pct in range(50, 85, 5):
        c = c_pct / 100.0
        for cf_pct in range(60, 90, 5):
            cf = cf_pct / 100.0
            def make_triple(hv=h, cv=c, cfv=cf):
                def fn(day):
                    result = ct_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        h_thresh = 0.38 if day.cs_action == 'BUY' else 0.31
                        if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                            return result
                        # Triple-gate recovery
                        if (day.cs_channel_health >= hv and
                            day.cs_confidence >= cv and
                            day.cs_confluence_score >= cfv):
                            return result
                    if day.v5_take_bounce:
                        if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                            return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                    return None
                return fn
            trades = simulate_trades(signals, make_triple(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if wr >= 100 and n > 414:
                pnl = sum(t.pnl for t in trades)
                print(f"  h>={hv:.2f} c>={cv:.2f} confl>={cfv:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 8: Quadruple gate (h + conf + confl + pos)
# ════════════════════════════════════════════════════════
print("\n--- Idea 8: Quad-gate recovery (h + conf + confl + pos) ---")
for h_pct in range(20, 38, 4):
    h = h_pct / 100.0
    for c_pct in [0.50, 0.55, 0.60, 0.65, 0.70]:
        for cf_pct in [0.60, 0.70, 0.80]:
            for pos_pct in [0.80, 0.85, 0.90, 0.95]:
                def make_quad(hv=h, cv=c_pct, cfv=cf_pct, pv=pos_pct):
                    def fn(day):
                        result = ct_fn(day)
                        if result is not None:
                            return result
                        result = _tf0_base(day)
                        if result is not None:
                            h_thresh = 0.38 if day.cs_action == 'BUY' else 0.31
                            if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                                return result
                            if (day.cs_channel_health >= hv and
                                day.cs_confidence >= cv and
                                day.cs_confluence_score >= cfv and
                                day.cs_position_score >= pv):
                                return result
                        if day.v5_take_bounce:
                            if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                        return None
                    return fn
                trades = simulate_trades(signals, make_quad(), 'test', cooldown=0, trail_power=6)
                n = len(trades)
                w = sum(1 for t in trades if t.pnl > 0)
                wr = w/n*100 if n else 0
                if wr >= 100 and n > 414:
                    pnl = sum(t.pnl for t in trades)
                    print(f"  h>={hv:.2f} c>={cv:.2f} confl>={cfv:.2f} pos>={pv:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 9: SPY 2-day return conditioned
# ════════════════════════════════════════════════════════
print("\n--- Idea 9: SPY 2-day return conditioned relaxation ---")
for spy_cond, spy_label in [
    (lambda d: spy_ret_2d.get(d.date, 0) < -1.0, "SPY2d<-1%"),
    (lambda d: spy_ret_2d.get(d.date, 0) > 1.0, "SPY2d>1%"),
    (lambda d: spy_return_map.get(d.date, 0) < -0.5, "SRet<-0.5%"),
    (lambda d: spy_return_map.get(d.date, 0) > 0.5, "SRet>0.5%"),
    (lambda d: spy_dist_5.get(d.date, 0) > 0, "SPY5>0"),
    (lambda d: spy_dist_50.get(d.date, 0) > 1.0, "SPY50>1%"),
]:
    for h_pct in range(30, 19, -2):
        h = h_pct / 100.0
        def make_spy2d(sc=spy_cond, hr=h):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_sell = 0.31
                    if sc(day):
                        h_buy = min(h_buy, hr)
                        h_sell = min(h_sell, hr)
                    h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_spy2d(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 414:
            pnl = sum(t.pnl for t in trades)
            print(f"  {spy_label} h>={h:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 10: V5 boundary expansion with conditions
# ════════════════════════════════════════════════════════
print("\n--- Idea 10: V5 conditional expansion ---")
for cond, label in [
    (lambda d: vix_map.get(d.date, 22) > 20, "VIX>20"),
    (lambda d: vix_map.get(d.date, 22) < 18, "VIX<18"),
    (lambda d: spy_dist_map.get(d.date, 0) < -0.5, "SPY<-0.5%"),
    (lambda d: d.cs_confidence >= 0.50, "c>=0.50"),
    (lambda d: d.cs_confluence_score >= 0.60, "confl>=0.60"),
]:
    for v5h_pct in range(58, 75, 2):
        v5h = v5h_pct / 100.0
        def make_v5_cond(c=cond, vh=v5h):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_thresh = 0.38 if day.cs_action == 'BUY' else 0.31
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    h_max = vh if c(day) else 0.57
                    if day.cs_channel_health < h_max and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_v5_cond(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 414:
            pnl = sum(t.pnl for t in trades)
            print(f"  V5 {label} h<{vh:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

print("\nDone")
