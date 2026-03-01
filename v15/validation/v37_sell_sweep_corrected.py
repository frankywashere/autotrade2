#!/usr/bin/env python3
"""Corrected SELL h sweep using proper VIX cascade (matching CW/CU architecture).
The v36_sell_sweep.py used raw confidence without VIX cascade — results were misleading."""
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
    """TF0: CS signal + VIX cascade (matching CU/CW architecture)."""
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

# ════════════════════════════════════════════════════════
# Sweep 1: SELL h threshold (BUY h>=0.38 fixed, VIX cascade)
# ════════════════════════════════════════════════════════
print("CORRECTED SELL h sweep (with VIX cascade, BUY h>=0.38):")
print("=" * 80)

for h_sell_pct in range(38, 19, -1):
    h_sell = h_sell_pct / 100.0
    def make_fn(hs=h_sell):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            # Gate-free bypass WITH VIX cascade
            result = _tf0_base(day)
            if result is not None:
                h_thresh = 0.38 if day.cs_action == 'BUY' else hs
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            # V5 bounce
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_fn(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " *** BEST" if wr >= 100 and n > 412 else (" <-- 100%" if wr >= 100 else "")
    print(f"  SELL h>={h_sell:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ════════════════════════════════════════════════════════
# Sweep 2: BUY h threshold (SELL h>=0.38, VIX cascade)
# ════════════════════════════════════════════════════════
print("\nCORRECTED BUY h sweep (with VIX cascade, SELL h>=0.38):")
print("=" * 80)

for h_buy_pct in range(38, 19, -1):
    h_buy = h_buy_pct / 100.0
    def make_fn2(hb=h_buy):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                h_thresh = hb if day.cs_action == 'BUY' else 0.38
                if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                    return result
            if day.v5_take_bounce:
                if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                    return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
            return None
        return fn
    trades = simulate_trades(signals, make_fn2(), 'test', cooldown=0, trail_power=6)
    n = len(trades)
    w = sum(1 for t in trades if t.pnl > 0)
    wr = w/n*100 if n else 0
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " *** BEST" if wr >= 100 and n > 412 else (" <-- 100%" if wr >= 100 else "")
    print(f"  BUY h>={h_buy:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ════════════════════════════════════════════════════════
# Sweep 3: Combined BUY + SELL relaxation (safe zone only)
# ════════════════════════════════════════════════════════
print("\nCombined BUY/SELL relaxation (100% WR only):")
print("=" * 80)

best_combo = (0.38, 0.38, 412)
for h_buy_pct in range(38, 29, -1):
    h_buy = h_buy_pct / 100.0
    for h_sell_pct in range(38, 29, -1):
        h_sell = h_sell_pct / 100.0
        def make_combo(hb=h_buy, hs=h_sell):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    h_thresh = hb if day.cs_action == 'BUY' else hs
                    if day.cs_channel_health >= h_thresh or day.cs_confluence_score >= 0.90:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_combo(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 412:
            pnl = sum(t.pnl for t in trades)
            bl = min(t.pnl for t in trades) if trades else 0
            if n > best_combo[2]:
                best_combo = (h_buy, h_sell, n)
            print(f"  BUY h>={h_buy:.2f} SELL h>={h_sell:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f} ***")

if best_combo[2] > 412:
    print(f"\nBest 100% WR combo: BUY h>={best_combo[0]:.2f} SELL h>={best_combo[1]:.2f}: {best_combo[2]} trades")
else:
    print(f"\nNo improvement over CW (412 trades)")

# ════════════════════════════════════════════════════════
# Sweep 4: Confl bypass relaxation
# ════════════════════════════════════════════════════════
print("\nConfl bypass sweep (h>=0.38 for both, VIX cascade):")
print("=" * 80)

for confl_pct in range(90, 49, -2):
    confl = confl_pct / 100.0
    def make_confl(cf=confl):
        def fn(day):
            result = ct_fn(day)
            if result is not None:
                return result
            result = _tf0_base(day)
            if result is not None:
                if day.cs_channel_health >= 0.38 or day.cs_confluence_score >= cf:
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
    pnl = sum(t.pnl for t in trades)
    bl = min(t.pnl for t in trades) if trades else 0
    marker = " *** BEST" if wr >= 100 and n > 412 else (" <-- 100%" if wr >= 100 else "")
    print(f"  confl>={confl:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}{marker}")

# ════════════════════════════════════════════════════════
# Sweep 5: Direction-specific confl
# ════════════════════════════════════════════════════════
print("\nDirection-specific confl sweep (h>=0.38 for both):")
print("=" * 80)

for buy_confl_pct in [90, 88, 86, 84, 82, 80]:
    for sell_confl_pct in range(90, 49, -4):
        buy_confl = buy_confl_pct / 100.0
        sell_confl = sell_confl_pct / 100.0
        def make_dconfl(bc=buy_confl, sc=sell_confl):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    confl_thresh = bc if day.cs_action == 'BUY' else sc
                    if day.cs_channel_health >= 0.38 or day.cs_confluence_score >= confl_thresh:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_dconfl(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 412:
            pnl = sum(t.pnl for t in trades)
            bl = min(t.pnl for t in trades) if trades else 0
            print(f"  BUY confl>={buy_confl:.2f} SELL confl>={sell_confl:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Sweep 6: h lower bound + confl crossover
# ════════════════════════════════════════════════════════
print("\nCross-axis: h + confl combined gates:")
print("=" * 80)
for h_pct in range(38, 19, -2):
    h = h_pct / 100.0
    for confl_pct in range(90, 49, -4):
        confl = confl_pct / 100.0
        def make_cross(hv=h, cf=confl):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    if day.cs_channel_health >= hv or day.cs_confluence_score >= cf:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_cross(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 412:
            pnl = sum(t.pnl for t in trades)
            bl = min(t.pnl for t in trades)
            print(f"  h>={hv:.2f} confl>={cf:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Sweep 7: AND-gate (h AND confl) instead of OR
# ════════════════════════════════════════════════════════
print("\nAND-gate: h AND confl thresholds:")
print("=" * 80)
for h_pct in range(20, 39, 2):
    h = h_pct / 100.0
    for confl_pct in range(50, 92, 4):
        confl = confl_pct / 100.0
        def make_and(hv=h, cf=confl):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    # OR-gate at current levels PLUS AND-gate at lower levels
                    if day.cs_channel_health >= 0.38 or day.cs_confluence_score >= 0.90:
                        return result
                    # Additional AND-gate recovery
                    if day.cs_channel_health >= hv and day.cs_confluence_score >= cf:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_and(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 412:
            pnl = sum(t.pnl for t in trades)
            bl = min(t.pnl for t in trades)
            print(f"  h>={hv:.2f} AND confl>={cf:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Sweep 8: Confidence-gated recovery
# ════════════════════════════════════════════════════════
print("\nConfidence-gated recovery (bypass at lower h if high confidence):")
print("=" * 80)
for h_pct in range(20, 38, 2):
    h = h_pct / 100.0
    for c_pct in range(50, 90, 5):
        c = c_pct / 100.0
        def make_conf_gate(hv=h, cv=c):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    if day.cs_channel_health >= 0.38 or day.cs_confluence_score >= 0.90:
                        return result
                    # Confidence-gated recovery
                    if day.cs_channel_health >= hv and day.cs_confidence >= cv:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_conf_gate(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 412:
            pnl = sum(t.pnl for t in trades)
            bl = min(t.pnl for t in trades)
            print(f"  h>={hv:.2f} AND c>={cv:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f} ***")

# ════════════════════════════════════════════════════════
# Sweep 9: Position-score gated recovery
# ════════════════════════════════════════════════════════
print("\nPosition-score gated recovery:")
print("=" * 80)
for h_pct in range(20, 38, 2):
    h = h_pct / 100.0
    for pos_pct in range(50, 100, 5):
        pos = pos_pct / 100.0
        def make_pos_gate(hv=h, pv=pos):
            def fn(day):
                result = ct_fn(day)
                if result is not None:
                    return result
                result = _tf0_base(day)
                if result is not None:
                    if day.cs_channel_health >= 0.38 or day.cs_confluence_score >= 0.90:
                        return result
                    # Position-gated recovery
                    if day.cs_channel_health >= hv and day.cs_position_score >= pv:
                        return result
                if day.v5_take_bounce:
                    if day.cs_channel_health < 0.57 and day.cs_position_score < 0.85:
                        return ('BUY', day.v5_confidence or 0.60, 2.0, 4.0, 'V5')
                return None
            return fn
        trades = simulate_trades(signals, make_pos_gate(), 'test', cooldown=0, trail_power=6)
        n = len(trades)
        w = sum(1 for t in trades if t.pnl > 0)
        wr = w/n*100 if n else 0
        if wr >= 100 and n > 412:
            pnl = sum(t.pnl for t in trades)
            bl = min(t.pnl for t in trades)
            print(f"  h>={hv:.2f} AND pos>={pv:.2f}: {n:4d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f} ***")

print("\nDone")
