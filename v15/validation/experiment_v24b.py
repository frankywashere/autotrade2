#!/usr/bin/env python3
"""v24b: Test alternate base functions and trail power sweep.
Use the existing s1_tf3_vix infrastructure but with modified TF thresholds."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _make_s1_tf3_vix_combo, _build_filter_cascade,
    _make_v23_cp, _SigProxy, _AnalysisProxy, _floor_stop_tp
)
from collections import defaultdict

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

cp_fn = _make_v23_cp(cascade_vix, spy_above_sma20, spy_above_055pct,
                      spy_dist_map, spy_dist_5, spy_dist_50,
                      vix_map, spy_return_map, spy_ret_2d)
cp_trades = simulate_trades(signals, cp_fn, 'CP', cooldown=0, trail_power=6)
print(f"CP baseline: {len(cp_trades)} trades, {sum(1 for t in cp_trades if t.pnl>0)/len(cp_trades)*100:.1f}% WR, ${sum(t.pnl for t in cp_trades):+,.0f}")

# ── Build s1_tf_vix variant with configurable min_tfs ──
def make_s1_tfN_vix(min_tfs=3, min_conf=MIN_SIGNAL_CONFIDENCE):
    """Like s1_tf3_vix but with configurable TF threshold and min confidence."""
    prev_tf_states = {}
    streaks = defaultdict(int)

    def fn(day):
        nonlocal prev_tf_states, streaks
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    streaks[tf] = 0
                    continue
                md = state.get('momentum_direction', 0.0)
                prev_md = prev_tf_states.get(tf, 0.0)
                if (md > 0 and prev_md > 0) or (md < 0 and prev_md < 0):
                    streaks[tf] += 1
                else:
                    streaks[tf] = 1 if md != 0 else 0
                prev_tf_states[tf] = md
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < min_conf:
            return None
        action = day.cs_action
        confirmed = 0
        if day.cs_tf_states:
            for tf, state in day.cs_tf_states.items():
                if not state.get('valid', False):
                    continue
                md = state.get('momentum_direction', 0.0)
                aligned = (action == 'BUY' and md > 0) or (action == 'SELL' and md < 0)
                if aligned and streaks.get(tf, 0) >= 1:
                    confirmed += 1
        if confirmed < min_tfs:
            return None
        sig = _SigProxy(day)
        ana = _AnalysisProxy(day.cs_tf_states)
        ok, adj, _ = cascade_vix.evaluate(
            sig, ana, feature_vec=None, bar_datetime=day.date,
            higher_tf_data=None, spy_df=None, vix_df=None,
        )
        if not ok or adj < min_conf:
            return None
        s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
        return (day.cs_action, adj, s, t, 'CS')
    return fn

# ── Test alternate bases raw (no CP filters) ──
print("\n=== RAW ALTERNATE BASES ===")
for min_tfs in [1, 2, 3, 4, 5]:
    for min_conf in [0.35, 0.40, 0.45, 0.50]:
        base = make_s1_tfN_vix(min_tfs, min_conf)
        trades = simulate_trades(signals, base, f'tf{min_tfs}_c{int(min_conf*100)}', cooldown=0, trail_power=6)
        n = len(trades)
        if n == 0: continue
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / n * 100
        bl = min(t.pnl for t in trades)
        pnl = sum(t.pnl for t in trades)
        print(f"  tf>={min_tfs} conf>={min_conf}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

# ── Apply CP-level gates to expanded bases ──
print("\n=== CP-LIKE FILTERING ON ALTERNATE BASES ===")

def make_cp_filtered(alt_base_fn):
    """Apply CP-level gates to any base function."""
    def fn(day):
        result = alt_base_fn(day)
        if result is None:
            return None
        action, conf, s_pct, t_pct, src = result

        if action == 'BUY':
            # LONG SPY gate + CO/v23 overrides
            spy_pass = False
            if day.date in spy_above_sma20: spy_pass = True
            elif conf >= 0.80: spy_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 5: spy_pass = True
            elif conf >= 0.70 and _count_tf_confirming(day, 'BUY') >= 4 and day.cs_position_score < 0.95: spy_pass = True
            elif conf >= 0.65 and _count_tf_confirming(day, 'BUY') >= 4 and day.cs_channel_health >= 0.40: spy_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.65: spy_pass = True
            # CO LONG overrides
            if not spy_pass:
                if day.cs_confluence_score >= 0.9 and conf >= 0.45: spy_pass = True
                elif vix_map.get(day.date, 22) > 30 and conf >= 0.50: spy_pass = True
                elif vix_map.get(day.date, 22) > 25 and conf >= 0.55: spy_pass = True
                elif spy_return_map.get(day.date, 0) > 1.0 and conf >= 0.55: spy_pass = True
            # v23 BUY recovery
            if not spy_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1: spy_pass = True
                elif day.cs_channel_health >= 0.25 and day.cs_position_score < 0.95: spy_pass = True
            if not spy_pass:
                return None

            # LONG CONF gate + v23
            conf_pass = False
            if conf >= 0.66: conf_pass = True
            elif day.cs_position_score <= 0.99: conf_pass = True
            elif _count_tf_confirming(day, 'BUY') >= 4: conf_pass = True
            elif day.cs_confluence_score >= 0.9 and conf >= 0.55: conf_pass = True
            if not conf_pass and conf >= 0.55:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() != 1: conf_pass = True
                elif day.cs_channel_health >= 0.25 and day.cs_position_score < 0.95: conf_pass = True
            if not conf_pass:
                return None

        if action == 'SELL':
            # SHORT SPY gate (full CM+CO)
            spy_pass = False
            if day.date in spy_above_055pct: spy_pass = True
            elif spy_dist_map.get(day.date, 999) < 0 and day.cs_channel_health >= 0.32: spy_pass = True
            elif 0 <= spy_dist_map.get(day.date, 999) < 0.55 and day.cs_position_score < 0.99: spy_pass = True
            elif 0 <= spy_dist_map.get(day.date, 999) < 0.55 and day.cs_position_score >= 0.99 and day.cs_channel_health >= 0.35: spy_pass = True
            if not spy_pass and spy_dist_5.get(day.date, -999) >= 0: spy_pass = True
            if not spy_pass:
                vix = vix_map.get(day.date, 22)
                if (vix < 20 or vix > 25) and day.cs_channel_health >= 0.25: spy_pass = True
            if not spy_pass and day.cs_channel_health >= 0.25:
                dd = day.date.date() if hasattr(day.date, 'date') else day.date
                if dd.weekday() in (0, 2, 3): spy_pass = True
            if not spy_pass and spy_dist_50.get(day.date, -999) >= 1.0: spy_pass = True
            # CO short overrides
            if not spy_pass:
                if _count_tf_confirming(day, 'SELL') >= 5 and day.cs_channel_health >= 0.20: spy_pass = True
                elif spy_ret_2d.get(day.date, 0) < -2.0 and day.cs_channel_health >= 0.15: spy_pass = True
            if not spy_pass:
                return None

            # SHORT CONF gate
            conf_pass = False
            if conf >= 0.65: conf_pass = True
            elif day.cs_channel_health >= 0.30: conf_pass = True
            elif day.cs_confluence_score >= 0.9 and day.cs_channel_health >= 0.25: conf_pass = True
            elif vix_map.get(day.date, 22) > 25 and day.cs_channel_health >= 0.20: conf_pass = True
            elif spy_return_map.get(day.date, 0) < -1.0 and day.cs_channel_health >= 0.20: conf_pass = True
            elif vix_map.get(day.date, 22) > 30 and day.cs_channel_health >= 0.15: conf_pass = True
            elif day.cs_channel_health >= 0.10 and conf >= 0.60 and _count_tf_confirming(day, 'SELL') >= 4: conf_pass = True
            if not conf_pass:
                return None

        return result
    return fn

for min_tfs in [1, 2, 3, 4]:
    for min_conf in [0.35, 0.40, 0.45]:
        base = make_s1_tfN_vix(min_tfs, min_conf)
        filt = make_cp_filtered(base)
        trades = simulate_trades(signals, filt, f'filt_tf{min_tfs}_c{int(min_conf*100)}', cooldown=0, trail_power=6)
        n = len(trades)
        if n == 0: continue
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / n * 100
        bl = min(t.pnl for t in trades)
        pnl = sum(t.pnl for t in trades)
        tr = [t for t in trades if t.entry_date.year <= 2021]
        ts = [t for t in trades if t.entry_date.year > 2021]
        marker = " ***" if wr >= 100 and n > 255 else ""
        print(f"  tf>={min_tfs} c>={min_conf}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f} "
              f"[{len(tr)}@{sum(1 for t in tr if t.pnl>0)/max(1,len(tr))*100:.0f}%/{len(ts)}@{sum(1 for t in ts if t.pnl>0)/max(1,len(ts))*100:.0f}%]{marker}")

# ── Find new trades from expanded base not in CP ──
print("\n=== NEW TRADES FROM EXPANDED BASES ===")
for min_tfs in [1, 2]:
    for min_conf in [0.35, 0.40, 0.45]:
        base = make_s1_tfN_vix(min_tfs, min_conf)
        filt = make_cp_filtered(base)
        alt_trades = simulate_trades(signals, filt, f'alt_tf{min_tfs}_c{int(min_conf*100)}', cooldown=0, trail_power=6)
        cp_dates_set = {t.entry_date for t in cp_trades}
        new_trades = [t for t in alt_trades if t.entry_date not in cp_dates_set]
        if not new_trades: continue
        new_w = sum(1 for t in new_trades if t.pnl > 0)
        new_l = len(new_trades) - new_w
        print(f"\n  tf>={min_tfs} c>={min_conf}: {len(new_trades)} new trades ({new_w}W/{new_l}L)")
        for t in sorted(new_trades, key=lambda x: x.entry_date)[:15]:
            day = {d.date: d for d in signals}.get(t.entry_date)
            if day:
                tfs = _count_tf_confirming(day, t.direction)
                print(f"    {t.entry_date.strftime('%Y-%m-%d'):10} {t.direction:5} ${t.pnl:>+7,.0f} c={day.cs_confidence:.3f} h={day.cs_channel_health:.3f} tfs={tfs}")
            else:
                print(f"    {t.entry_date.strftime('%Y-%m-%d'):10} {t.direction:5} ${t.pnl:>+7,.0f}")

# ── Trail power sweep ──
print("\n=== TRAIL POWER SWEEP ON CP ===")
for power in [4, 5, 6, 7, 8, 10, 12]:
    trades = simulate_trades(signals, cp_fn, f'CP_p{power}', cooldown=0, trail_power=power)
    n = len(trades)
    if n == 0: continue
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    bl = min(t.pnl for t in trades)
    pnl = sum(t.pnl for t in trades)
    print(f"  power={power:2d}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+,.0f}, BL=${bl:+,.0f}")

print("\nDone")
