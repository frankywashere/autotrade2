#!/usr/bin/env python3
"""v34: 3-stage validation for ALL combos (CH through CV).
Stage 1: Holdout — Train on 2016-2021, test on 2022-2025
Stage 2: Walk-forward — expanding window, 2017-2025
Stage 3: 2026 OOS — true out-of-sample on 2026 data

Per user request: run 2026 OOS on ALL previous combos and update tracking."""
import pickle, sys, os, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v15.validation.combo_backtest import (
    DaySignals, Trade, _count_tf_confirming, MIN_SIGNAL_CONFIDENCE,
    simulate_trades, _build_filter_cascade, _SigProxy, _AnalysisProxy, _floor_stop_tp,
    _make_v16_champion_combo, _make_v16_safe_combo,
    _make_v17_grand_champion, _make_v18_squeeze, _make_v19_grand,
    _make_v20_grand, _make_v21_cn, _make_v22_co, _make_v23_cp,
    _make_v24_cq, _make_v25_cr, _make_v26_cs, _make_v27_ct,
    _make_v28_cu, _make_v29_cv, _make_v30_cw, _make_v31_cx,
)

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

args_v22 = (cascade_vix, spy_above_sma20, spy_above_055pct,
            spy_dist_map, spy_dist_5, spy_dist_50,
            vix_map, spy_return_map, spy_ret_2d)  # 9 args: v22+
args_v20 = (cascade_vix, spy_above_sma20, spy_above_055pct,
            spy_dist_map, spy_dist_5, spy_dist_50,
            vix_map, spy_return_map)  # 8 args: v20-v21
args_v17 = (cascade_vix, spy_above_sma20, spy_above_055pct, spy_dist_map)  # 4 args: v17-v19
args_v16 = (cascade_vix, spy_above_sma20, spy_above_055pct)  # 3 args: v16

# Build all combos (different signatures for different versions)
combos = [
    ('CH', _make_v16_champion_combo(*args_v16)),
    ('CI', _make_v16_safe_combo(*args_v16)),
    ('CJ', _make_v17_grand_champion(*args_v17)),
    ('CK', _make_v18_squeeze(*args_v17)),
    ('CL', _make_v19_grand(*args_v17)),
    ('CM', _make_v20_grand(*args_v20)),
    ('CN', _make_v21_cn(*args_v20)),
    ('CO', _make_v22_co(*args_v22)),
    ('CP', _make_v23_cp(*args_v22)),
    ('CQ', _make_v24_cq(*args_v22)),
    ('CR', _make_v25_cr(*args_v22)),
    ('CS', _make_v26_cs(*args_v22)),
    ('CT', _make_v27_ct(*args_v22)),
    ('CU', _make_v28_cu(*args_v22)),
    ('CV', _make_v29_cv(*args_v22)),
    ('CW', _make_v30_cw(*args_v22)),
]

day_map = {day.date: day for day in signals}

# Check 2026 signal coverage
signals_2026 = [day for day in signals if day.date.year >= 2026]
cs_buy_2026 = [d for d in signals_2026 if d.cs_action == 'BUY']
cs_sell_2026 = [d for d in signals_2026 if d.cs_action == 'SELL']
v5_2026 = [d for d in signals_2026 if d.v5_take_bounce]
print(f"2026 signals: {len(signals_2026)} days, {len(cs_buy_2026)} CS BUY, {len(cs_sell_2026)} CS SELL, {len(v5_2026)} V5 bounce")
for d in signals_2026[:5]:
    print(f"  {d.date}: CS={d.cs_action} conf={d.cs_confidence:.3f} h={d.cs_channel_health:.3f} "
          f"pos={d.cs_position_score:.3f} confl={d.cs_confluence_score:.2f} "
          f"V5={d.v5_take_bounce} v5c={d.v5_confidence or 0:.3f}")

# ══════════════════════════════════════════════════════════
# SUMMARY TABLE: All combos, all stages
# ══════════════════════════════════════════════════════════
print("\n" + "="*120)
print("COMPREHENSIVE 3-STAGE VALIDATION: ALL COMBOS (CH-CV)")
print("="*120)

header = (f"{'Combo':5s} | {'Full':>6} {'WR':>5} {'PnL':>10} | "
          f"{'Train':>5} {'WR':>5} {'PnL':>10} | "
          f"{'Test':>5} {'WR':>5} {'PnL':>10} | "
          f"{'WF':>5} | "
          f"{'2026':>4} {'WR':>5} {'PnL':>10} {'BL':>8} | Status")
print(header)
print("-"*120)

results = {}

for combo_name, combo_fn in combos:
    trades = simulate_trades(signals, combo_fn, combo_name, cooldown=0, trail_power=6)

    # Full period
    n_full = len(trades)
    w_full = sum(1 for t in trades if t.pnl > 0)
    wr_full = w_full/n_full*100 if n_full else 0
    pnl_full = sum(t.pnl for t in trades)

    # Train (2016-2021)
    train = [t for t in trades if t.entry_date.year <= 2021]
    n_train = len(train)
    w_train = sum(1 for t in train if t.pnl > 0)
    wr_train = w_train/n_train*100 if n_train else 0
    pnl_train = sum(t.pnl for t in train)

    # Test (2022-2025)
    test = [t for t in trades if 2022 <= t.entry_date.year <= 2025]
    n_test = len(test)
    w_test = sum(1 for t in test if t.pnl > 0)
    wr_test = w_test/n_test*100 if n_test else 0
    pnl_test = sum(t.pnl for t in test)

    # Walk-forward
    wf_pass = 0
    wf_total = 0
    for test_year in range(2017, 2026):
        yearly = [t for t in trades if t.entry_date.year == test_year]
        if not yearly: continue
        wf_total += 1
        if all(t.pnl > 0 for t in yearly):
            wf_pass += 1

    # 2026 OOS
    oos = [t for t in trades if t.entry_date.year >= 2026]
    n_oos = len(oos)
    w_oos = sum(1 for t in oos if t.pnl > 0)
    wr_oos = w_oos/n_oos*100 if n_oos else 0
    pnl_oos = sum(t.pnl for t in oos)
    bl_oos = min(t.pnl for t in oos) if oos else 0

    # Status
    status_parts = []
    if wr_full >= 100: status_parts.append("FULL")
    if wr_train >= 100: status_parts.append("TRAIN")
    if wr_test >= 100: status_parts.append("TEST")
    if wf_pass == wf_total: status_parts.append("WF")
    if n_oos > 0 and wr_oos >= 100: status_parts.append("OOS")
    elif n_oos == 0: status_parts.append("OOS:N/A")

    if n_oos > 0:
        oos_str = f"{n_oos:4d} {wr_oos:4.0f}% ${pnl_oos:+9,.0f} ${bl_oos:+7,.0f}"
    else:
        oos_str = "  -- no trades --         "

    print(f"{combo_name:5s} | {n_full:5d} {wr_full:4.0f}% ${pnl_full:+9,.0f} | "
          f"{n_train:5d} {wr_train:4.0f}% ${pnl_train:+9,.0f} | "
          f"{n_test:5d} {wr_test:4.0f}% ${pnl_test:+9,.0f} | "
          f"{wf_pass}/{wf_total:2d} | "
          f"{oos_str} | {' '.join(status_parts)}")

    results[combo_name] = {
        'trades': n_full, 'wr': wr_full, 'pnl': pnl_full,
        'train_t': n_train, 'train_wr': wr_train, 'train_pnl': pnl_train,
        'test_t': n_test, 'test_wr': wr_test, 'test_pnl': pnl_test,
        'wf': f"{wf_pass}/{wf_total}",
        'oos_t': n_oos, 'oos_wr': wr_oos, 'oos_pnl': pnl_oos, 'oos_bl': bl_oos,
    }

# ══════════════════════════════════════════════════════════
# DETAILED 2026 OOS TRADE LOG
# ══════════════════════════════════════════════════════════
print("\n" + "="*120)
print("DETAILED 2026 OOS TRADE LOG")
print("="*120)

for combo_name, combo_fn in combos:
    trades = simulate_trades(signals, combo_fn, combo_name, cooldown=0, trail_power=6)
    oos = [t for t in trades if t.entry_date.year >= 2026]
    if not oos:
        continue
    print(f"\n--- {combo_name}: {len(oos)} OOS trades ---")
    for t in sorted(oos, key=lambda x: x.entry_date):
        day = day_map.get(t.entry_date)
        src = "V5" if day and day.v5_take_bounce else "CS"
        dd = t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date
        dow = ['Mon','Tue','Wed','Thu','Fri'][dd.weekday()] if dd.weekday() < 5 else '???'
        h_str = f"{day.cs_channel_health:.3f}" if day else "N/A"
        c_str = f"{day.cs_confidence:.3f}" if day else "N/A"
        cf_str = f"{day.cs_confluence_score:.2f}" if day else "N/A"
        pos_str = f"{day.cs_position_score:.3f}" if day else "N/A"
        print(f"  {str(t.entry_date)[:10]} {t.direction:5s} ${t.pnl:+8,.0f} src={src} "
              f"h={h_str} c={c_str} pos={pos_str} confl={cf_str} "
              f"VIX={vix_map.get(t.entry_date, 22):.1f} {dow}")

# ══════════════════════════════════════════════════════════
# YEAR-BY-YEAR FOR CV (champion)
# ══════════════════════════════════════════════════════════
print("\n" + "="*120)
print("CV (v29) YEAR-BY-YEAR BREAKDOWN")
print("="*120)
cv_trades = simulate_trades(signals, combos[-1][1], 'CV', cooldown=0, trail_power=6)
for year in range(2016, 2027):
    yearly = [t for t in cv_trades if t.entry_date.year == year]
    if not yearly: continue
    n = len(yearly)
    w = sum(1 for t in yearly if t.pnl > 0)
    wr = w/n*100
    pnl = sum(t.pnl for t in yearly)
    bl = min(t.pnl for t in yearly)
    bw = max(t.pnl for t in yearly)
    marker = " <-- OOS" if year >= 2026 else ""
    print(f"  {year}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+9,.0f}, BL=${bl:+,.0f}, BW=${bw:+,.0f}{marker}")

# V5 component in 2026
print("\n--- V5 component in CV (2026 OOS) ---")
cu_trades = simulate_trades(signals, combos[-2][1], 'CU', cooldown=0, trail_power=6)
cu_dates = {t.entry_date for t in cu_trades}
v5_only = [t for t in cv_trades if t.entry_date not in cu_dates]
v5_oos = [t for t in v5_only if t.entry_date.year >= 2026]
if v5_oos:
    n = len(v5_oos)
    w = sum(1 for t in v5_oos if t.pnl > 0)
    pnl = sum(t.pnl for t in v5_oos)
    print(f"  V5 2026 OOS: {n} trades, {w}/{n} wins, ${pnl:+,.0f}")
    for t in v5_oos:
        day = day_map.get(t.entry_date)
        print(f"    {str(t.entry_date)[:10]} ${t.pnl:+,.0f} h={day.cs_channel_health:.3f} pos={day.cs_position_score:.3f}")
else:
    print("  No V5 2026 OOS trades")

print("\nDone")
