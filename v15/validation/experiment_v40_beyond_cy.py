#!/usr/bin/env python3
"""v40: Push beyond CY (422 trades, 100% WR).
Explore: other DOW relaxation stacked on CY, more conditional combos,
V5 expansion with conditions, SELL confl relaxation with conditions."""
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

def make_cy():
    """CY baseline: Mon|VIX>25|BUY SPY<-1% h>=0.22."""
    def fn(day):
        result = ct_fn(day)
        if result is not None:
            return result
        result = _tf0_base(day)
        if result is not None:
            h_buy = 0.38
            h_sell = 0.31
            dd = day.date.date() if hasattr(day.date, 'date') else day.date
            vix = vix_map.get(day.date, 22)
            spy_d = spy_dist_map.get(day.date, 0)
            relax = dd.weekday() == 0 or vix > 25
            if day.cs_action == 'BUY' and spy_d < -1.0:
                relax = True
            if relax:
                h_buy = min(h_buy, 0.22)
                h_sell = min(h_sell, 0.22)
            h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
            if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                return result
        if day.v5_take_bounce:
            if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
        return None
    return fn

cy_trades = simulate_trades(signals, make_cy(), 'CY', cooldown=0, trail_power=6)
print(f"CY baseline: {len(cy_trades)} trades, "
      f"{sum(1 for t in cy_trades if t.pnl > 0)/len(cy_trades)*100:.1f}% WR, "
      f"${sum(t.pnl for t in cy_trades):+,.0f}")

# ════════════════════════════════════════════════════════
# Idea 1: Other DOW relaxation stacked on CY
# ════════════════════════════════════════════════════════
print("\n--- Idea 1: Add other DOW to CY ---")
for extra_dows, label in [
    ({4}, "Fri"),
    ({3}, "Thu"),
    ({1}, "Tue"),
    ({2}, "Wed"),
    ({3, 4}, "Thu+Fri"),
    ({1, 4}, "Tue+Fri"),
    ({1, 3}, "Tue+Thu"),
]:
    for h_pct in range(30, 14, -2):
        h = h_pct / 100.0
        def make_fn(ed=extra_dows, hr=h):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_sell = 0.31
                    dd = day.date.date() if hasattr(day.date, 'date') else day.date
                    vix = vix_map.get(day.date, 22)
                    spy_d = spy_dist_map.get(day.date, 0)
                    # CY conditions (Mon|VIX>25|BUY SPY<-1%)
                    relax = dd.weekday() == 0 or vix > 25
                    if day.cs_action == 'BUY' and spy_d < -1.0:
                        relax = True
                    # Extra DOW conditions
                    if dd.weekday() in ed:
                        relax = True
                    if relax:
                        h_buy = min(h_buy, 0.22)
                        h_sell = min(h_sell, 0.22)
                    # Different h for extra DOW (might need different threshold)
                    if dd.weekday() in ed and not (dd.weekday() == 0 or vix > 25):
                        h_buy = min(0.38, hr)
                        h_sell = min(0.31, hr)
                    h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_fn(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 422:
            pnl = sum(t.pnl for t in trades)
            bl = min(t.pnl for t in trades)
            print(f"  CY+{label} h>={h:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 2: Relax CY h threshold further on specific conditions
# ════════════════════════════════════════════════════════
print("\n--- Idea 2: CY with different relax thresholds per condition ---")
for mon_h_pct in range(22, 14, -2):
    mon_h = mon_h_pct / 100.0
    for vix_h_pct in range(22, 14, -2):
        vix_h = vix_h_pct / 100.0
        def make_diff(mh=mon_h, vh=vix_h):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_sell = 0.31
                    dd = day.date.date() if hasattr(day.date, 'date') else day.date
                    vix = vix_map.get(day.date, 22)
                    spy_d = spy_dist_map.get(day.date, 0)
                    if dd.weekday() == 0:
                        h_buy = min(h_buy, mh)
                        h_sell = min(h_sell, mh)
                    if vix > 25:
                        h_buy = min(h_buy, vh)
                        h_sell = min(h_sell, vh)
                    if day.cs_action == 'BUY' and spy_d < -1.0:
                        h_buy = min(h_buy, 0.22)
                    h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_diff(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 422:
            pnl = sum(t.pnl for t in trades)
            print(f"  Mon h>={mh:.2f} VIX h>={vh:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 3: SPY dist conditions (other thresholds)
# ════════════════════════════════════════════════════════
print("\n--- Idea 3: Additional SPY conditions on CY ---")
for spy_cond, spy_label in [
    (lambda d, a: a == 'SELL' and spy_dist_map.get(d.date, 0) > 1.0, "SELL&SPY>1%"),
    (lambda d, a: a == 'BUY' and spy_dist_map.get(d.date, 0) < -0.5, "BUY&SPY<-0.5%"),
    (lambda d, a: spy_ret_2d.get(d.date, 0) < -1.0, "SPY2d<-1%"),
    (lambda d, a: spy_ret_2d.get(d.date, 0) > 1.0, "SPY2d>1%"),
    (lambda d, a: spy_return_map.get(d.date, 0) < -1.0, "SRet<-1%"),
    (lambda d, a: a == 'SELL' and spy_dist_5.get(d.date, 0) > 0.5, "SELL&SPY5>0.5%"),
    (lambda d, a: a == 'BUY' and spy_dist_5.get(d.date, 0) < -0.5, "BUY&SPY5<-0.5%"),
    (lambda d, a: spy_dist_50.get(d.date, 0) > 2.0, "SPY50>2%"),
    (lambda d, a: spy_dist_50.get(d.date, 0) < -2.0, "SPY50<-2%"),
]:
    for h_pct in range(30, 14, -2):
        h = h_pct / 100.0
        def make_spy(sc=spy_cond, hr=h):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_sell = 0.31
                    dd = day.date.date() if hasattr(day.date, 'date') else day.date
                    vix = vix_map.get(day.date, 22)
                    spy_d = spy_dist_map.get(day.date, 0)
                    # CY base conditions
                    relax = dd.weekday() == 0 or vix > 25
                    if day.cs_action == 'BUY' and spy_d < -1.0:
                        relax = True
                    # Extra SPY condition
                    if sc(day, day.cs_action):
                        relax = True
                    if relax:
                        h_buy = min(h_buy, 0.22)
                        h_sell = min(h_sell, 0.22)
                    h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_spy(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 422:
            pnl = sum(t.pnl for t in trades)
            print(f"  CY+{spy_label} h>={h:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 4: Confidence-gated low-h recovery on CY
# ════════════════════════════════════════════════════════
print("\n--- Idea 4: Confidence-gated recovery on CY ---")
for c_pct in [0.50, 0.55, 0.60, 0.65, 0.70]:
    for h_pct in range(30, 14, -2):
        h = h_pct / 100.0
        def make_conf(cv=c_pct, hr=h):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_sell = 0.31
                    dd = day.date.date() if hasattr(day.date, 'date') else day.date
                    vix = vix_map.get(day.date, 22)
                    spy_d = spy_dist_map.get(day.date, 0)
                    relax = dd.weekday() == 0 or vix > 25
                    if day.cs_action == 'BUY' and spy_d < -1.0:
                        relax = True
                    if relax:
                        h_buy = min(h_buy, 0.22)
                        h_sell = min(h_sell, 0.22)
                    # Additional: confidence-gated recovery
                    if day.cs_confidence >= cv:
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
        trades = simulate_trades(signals, make_conf(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 422:
            pnl = sum(t.pnl for t in trades)
            print(f"  c>={cv:.2f} h>={h:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 5: VIX threshold variations
# ════════════════════════════════════════════════════════
print("\n--- Idea 5: VIX threshold sweep on CY ---")
for vix_thresh in [20, 22, 24, 26, 28, 30]:
    for h_pct in range(22, 14, -2):
        h = h_pct / 100.0
        def make_vix(vt=vix_thresh, hr=h):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_sell = 0.31
                    dd = day.date.date() if hasattr(day.date, 'date') else day.date
                    vix = vix_map.get(day.date, 22)
                    spy_d = spy_dist_map.get(day.date, 0)
                    relax = dd.weekday() == 0 or vix > vt
                    if day.cs_action == 'BUY' and spy_d < -1.0:
                        relax = True
                    if relax:
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
        trades = simulate_trades(signals, make_vix(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 422:
            pnl = sum(t.pnl for t in trades)
            print(f"  VIX>{vt} h>={h:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 6: V5 conditional expansion on CY
# ════════════════════════════════════════════════════════
print("\n--- Idea 6: V5 expansion with conditions on CY ---")
for cond, label in [
    (lambda d: vix_map.get(d.date, 22) > 20, "VIX>20"),
    (lambda d: vix_map.get(d.date, 22) > 25, "VIX>25"),
    (lambda d: spy_dist_map.get(d.date, 0) < -0.5, "SPY<-0.5%"),
    (lambda d: d.cs_confidence >= 0.50, "c>=0.50"),
]:
    for v5h_pct in range(58, 75, 2):
        v5h = v5h_pct / 100.0
        for v5pos_pct in [85, 90, 95]:
            v5pos = v5pos_pct / 100.0
            def make_v5(c=cond, vh=v5h, vp=v5pos):
                def fn(day):
                    result = ct_fn(day)
                    if result is not None:
                        return result
                    result = _tf0_base(day)
                    if result is not None:
                        h_buy = 0.38
                        h_sell = 0.31
                        dd = day.date.date() if hasattr(day.date, 'date') else day.date
                        vix = vix_map.get(day.date, 22)
                        spy_d = spy_dist_map.get(day.date, 0)
                        relax = dd.weekday() == 0 or vix > 25
                        if day.cs_action == 'BUY' and spy_d < -1.0:
                            relax = True
                        if relax:
                            h_buy = min(h_buy, 0.22)
                            h_sell = min(h_sell, 0.22)
                        h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                        if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                            return result
                    if day.v5_take_bounce:
                        h_max = vh if c(day) else 0.57
                        pos_max = vp if c(day) else 0.85
                        if day.cs_channel_health < h_max and day.cs_position_score < pos_max:
                            return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                    return None
                return fn
            trades = simulate_trades(signals, make_v5(), 'test', cooldown=0, trail_power=6)
            n = len(trades)
            w = sum(1 for t in trades if t.pnl > 0)
            wr = w/n*100 if n else 0
            if wr >= 100 and n > 422:
                pnl = sum(t.pnl for t in trades)
                print(f"  V5 {label} h<{vh:.2f} pos<{vp:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 7: Confl threshold with direction+condition
# ════════════════════════════════════════════════════════
print("\n--- Idea 7: Conditional confl bypass on CY ---")
for cond, label in [
    (lambda d: vix_map.get(d.date, 22) > 25, "VIX>25"),
    (lambda d: d.date.date().weekday() == 0 if hasattr(d.date, 'date') else d.date.weekday() == 0, "Mon"),
    (lambda d: d.cs_confidence >= 0.50, "c>=0.50"),
    (lambda d: d.cs_confidence >= 0.60, "c>=0.60"),
]:
    for confl_pct in range(88, 59, -4):
        confl = confl_pct / 100.0
        def make_confl(c=cond, cf=confl):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_sell = 0.31
                    dd = day.date.date() if hasattr(day.date, 'date') else day.date
                    vix = vix_map.get(day.date, 22)
                    spy_d = spy_dist_map.get(day.date, 0)
                    relax = dd.weekday() == 0 or vix > 25
                    if day.cs_action == 'BUY' and spy_d < -1.0:
                        relax = True
                    if relax:
                        h_buy = min(h_buy, 0.22)
                        h_sell = min(h_sell, 0.22)
                    confl_thresh = 0.90
                    if c(day):
                        confl_thresh = min(confl_thresh, cf)
                    h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= confl_thresh:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_confl(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 422:
            pnl = sum(t.pnl for t in trades)
            print(f"  {label} confl>={confl:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Idea 8: Position-score conditioned relaxation
# ════════════════════════════════════════════════════════
print("\n--- Idea 8: Position-score conditioned on CY ---")
for pos_cond, pos_label in [
    (lambda d: d.cs_position_score >= 0.90, "pos>=0.90"),
    (lambda d: d.cs_position_score >= 0.85, "pos>=0.85"),
    (lambda d: d.cs_position_score < 0.30, "pos<0.30"),
    (lambda d: d.cs_position_score < 0.20, "pos<0.20"),
]:
    for h_pct in range(30, 14, -2):
        h = h_pct / 100.0
        def make_pos(pc=pos_cond, hr=h):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_buy = 0.38
                    h_sell = 0.31
                    dd = day.date.date() if hasattr(day.date, 'date') else day.date
                    vix = vix_map.get(day.date, 22)
                    spy_d = spy_dist_map.get(day.date, 0)
                    relax = dd.weekday() == 0 or vix > 25
                    if day.cs_action == 'BUY' and spy_d < -1.0:
                        relax = True
                    if pc(day):
                        relax = True
                    if relax:
                        h_buy = min(h_buy, 0.22)
                        h_sell = min(h_sell, 0.22)
                    h_thresh = h_buy if day.cs_action == 'BUY' else h_sell
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_pos(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 422:
            pnl = sum(t.pnl for t in trades)
            print(f"  CY+{pos_label} h>={h:.2f}: {n} trades, {wr:.1f}% WR, ${pnl:+,.0f} ***")

print("\nDone")
