#!/usr/bin/env python3
"""
c13a Full Signal Validation — All 3 Signal Types

Tests CS-5TF, CS-DW (daily+weekly only), and Intraday (FD Enhanced-Union)
with identical validation methodology:
  1. Per-Year      — $100K fresh capital each year (2016–2026)
  2. Holdout       — Train ≤2021, Test 2022–2025
  3. Walk-Forward  — Rolling 5yr→1yr (6 windows)
  4. 2026 OOS      — Standalone, trade-by-trade log

Usage:
    python -m v15.validation.c13a_full_validation
    python -m v15.validation.c13a_full_validation --tsla data/TSLAMin.txt
    python -m v15.validation.c13a_full_validation --phase2-only
"""

import argparse
import json
import os
import pickle
import sys
import time
import datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ---------------------------------------------------------------------------
# Import from combo_backtest (CS-5TF machinery)
# ---------------------------------------------------------------------------
from v15.validation.combo_backtest import (
    DaySignals,
    Trade,
    simulate_trades,
    phase1_precompute,
    report_combo,
    _build_filter_cascade,
    _SigProxy,
    _AnalysisProxy,
    _floor_stop_tp,
    _apply_costs,
    MIN_SIGNAL_CONFIDENCE,
    CAPITAL,
    DEFAULT_STOP_PCT,
    DEFAULT_TP_PCT,
    TRAILING_STOP_BASE,
    MAX_HOLD_DAYS,
    COOLDOWN_DAYS,
    SLIPPAGE_PCT,
    COMMISSION_PER_SHARE,
    CACHE_DIR,
    CACHE_FILE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SETTINGS = {
    'cs_5tf': {
        'name': 'CS-5TF (DI v36 champion)',
        'stop_pct': 0.02, 'tp_pct': 0.04,
        'trail_formula': '0.025 * (1 - conf)^8',
        'trail_base': 0.025, 'trail_power': 8,
        'max_hold_days': 10, 'cooldown': 0,
        'capital': 100_000, 'slippage_pct': 0.0001,
        'commission_per_share': 0.005,
    },
    'cs_dw': {
        'name': 'CS-DW (daily+weekly only)',
        'stop_pct': 0.02, 'tp_pct': 0.04,
        'trail_formula': '0.025 * (1 - conf)^8',
        'trail_base': 0.025, 'trail_power': 8,
        'max_hold_days': 10, 'cooldown': 0,
        'capital': 100_000, 'slippage_pct': 0.0001,
        'commission_per_share': 0.005,
    },
    'intraday': {
        'name': 'Intraday FD Enhanced-Union',
        'stop_pct': 0.008, 'tp_pct': 0.020,
        'trail_formula': '0.006 * (1 - conf)^6',
        'trail_base': 0.006, 'trail_power': 6,
        'pm_window': '13:00-15:25 ET', 'eod_close': True,
        'capital': 100_000, 'slippage_pct': 0.0002,
        'commission_per_share': 0.005,
        'long_only': True,
    },
    'surfer_ml': {
        'name': 'Surfer ML (26 models, realistic mode)',
        'resolution': '5-min bars',
        'stop': 'ATR-based, ML-adjusted (EL/IS tightening)',
        'tp': 'CS suggested, bounce TP widened 1.3x',
        'trail': 'profit-tier based (surfer_backtest)',
        'sizing': 'risk-normalized + ML signal quality + win-streak ramp',
        'capital': 100_000, 'max_leverage': 4.0,
        'slippage_bps': 3.0, 'commission_per_share': 0.005,
        'models': '26 pkl/pt models in surfer_models/',
    },
}

WALK_FORWARD_WINDOWS = [
    ('2016-2020', '2021'),
    ('2017-2021', '2022'),
    ('2018-2022', '2023'),
    ('2019-2023', '2024'),
    ('2020-2024', '2025'),
    ('2021-2025', '2026'),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metrics(trades: List[Trade]) -> dict:
    """Compute standard metrics from a list of Trade objects."""
    n = len(trades)
    if n == 0:
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'sharpe': 0, 'max_dd_pct': 0,
                'avg_win': 0, 'avg_loss': 0, 'biggest_loss': 0}
    pnls = np.array([t.pnl for t in trades])
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100
    total = float(pnls.sum())
    avg_w = float(np.mean([p for p in pnls if p > 0])) if wins > 0 else 0
    avg_l = float(np.mean([p for p in pnls if p <= 0])) if wins < n else 0
    bl = float(pnls.min())
    avg_hold = float(np.mean([t.hold_days for t in trades]))
    sharpe = (float(pnls.mean() / pnls.std() * np.sqrt(252 / max(avg_hold, 1)))
              if pnls.std() > 0 else 0.0)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    mdd = float(dd.max()) / CAPITAL * 100 if len(dd) > 0 else 0
    return {
        'trades': n, 'wins': wins, 'win_rate': round(wr, 1),
        'total_pnl': round(total), 'avg_win': round(avg_w),
        'avg_loss': round(avg_l), 'biggest_loss': round(bl),
        'sharpe': round(sharpe, 2), 'max_dd_pct': round(mdd, 1),
        'avg_hold_days': round(avg_hold, 1),
    }


def _compute_metrics_intraday(trades: list) -> dict:
    """Compute metrics from intraday trade tuples.
    Tuple: (entry_time, exit_time, entry_px, exit_px, conf, shares, pnl, hold_bars, exit_reason, name)
    """
    n = len(trades)
    if n == 0:
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'sharpe': 0, 'max_dd_pct': 0,
                'avg_win': 0, 'avg_loss': 0, 'biggest_loss': 0}
    pnls = np.array([t[6] for t in trades])
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100
    total = float(pnls.sum())
    avg_w = float(np.mean([p for p in pnls if p > 0])) if wins > 0 else 0
    avg_l = float(np.mean([p for p in pnls if p <= 0])) if wins < n else 0
    bl = float(pnls.min())
    sharpe = (float(pnls.mean() / pnls.std() * np.sqrt(252))
              if pnls.std() > 0 else 0.0)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    mdd = float(dd.max()) / CAPITAL * 100 if len(dd) > 0 else 0
    return {
        'trades': n, 'wins': wins, 'win_rate': round(wr, 1),
        'total_pnl': round(total), 'avg_win': round(avg_w),
        'avg_loss': round(avg_l), 'biggest_loss': round(bl),
        'sharpe': round(sharpe, 2), 'max_dd_pct': round(mdd, 1),
    }


def _print_year_table(year_results: dict, label: str):
    """Print a per-year summary table."""
    print(f"\n{'='*75}")
    print(f"  {label} — Per-Year Results")
    print(f"{'='*75}")
    print(f"  {'Year':>6} {'Trades':>7} {'WR%':>6} {'PnL':>11} {'Sharpe':>7} {'MaxDD%':>7} {'BigLoss':>9}")
    print(f"  {'-'*60}")
    cum = 0
    for yr in sorted(year_results.keys()):
        m = year_results[yr]
        cum += m['total_pnl']
        print(f"  {yr:>6} {m['trades']:>7} {m['win_rate']:>5.1f}% ${m['total_pnl']:>+9,} "
              f"{m['sharpe']:>7.2f} {m['max_dd_pct']:>6.1f}% ${m['biggest_loss']:>+7,}")
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':>6} {'':>7} {'':>6} ${cum:>+9,}")


def _print_holdout(train_m: dict, test_m: dict):
    """Print holdout split results."""
    print(f"\n  --- Holdout Split ---")
    print(f"  {'Split':<18} {'Trades':>7} {'WR%':>6} {'PnL':>11} {'Sharpe':>7}")
    print(f"  {'-'*50}")
    for label, m in [('Train (≤2021)', train_m), ('Test (2022-2025)', test_m)]:
        print(f"  {label:<18} {m['trades']:>7} {m['win_rate']:>5.1f}% ${m['total_pnl']:>+9,} {m['sharpe']:>7.2f}")


def _print_walkforward(wf_results: list):
    """Print walk-forward results."""
    print(f"\n  --- Walk-Forward (5yr train → 1yr test) ---")
    print(f"  {'Window':<16} {'Train Trades':>12} {'Train WR':>9} {'Test Trades':>12} {'Test WR':>8} {'Test PnL':>10}")
    print(f"  {'-'*70}")
    for w in wf_results:
        print(f"  {w['train_period']+'→'+w['test_year']:<16} {w['train']['trades']:>12} "
              f"{w['train']['win_rate']:>8.1f}% {w['test']['trades']:>12} "
              f"{w['test']['win_rate']:>7.1f}% ${w['test']['total_pnl']:>+8,}")


def _print_2026_trades(trades, signal_type: str):
    """Print trade-by-trade log for 2026."""
    print(f"\n  --- 2026 OOS Trade Log ---")
    if not trades:
        print("  No 2026 trades.")
        return
    if signal_type == 'intraday':
        print(f"  {'#':>3} {'Entry':>20} {'Exit':>20} {'EntryPx':>9} {'ExitPx':>9} {'Conf':>5} {'Shares':>7} {'PnL':>9} {'Reason'}")
        print(f"  {'-'*100}")
        for i, t in enumerate(trades, 1):
            print(f"  {i:>3} {str(t[0]):>20} {str(t[1]):>20} ${t[2]:>7.2f} ${t[3]:>7.2f} "
                  f"{t[4]:>5.2f} {t[5]:>7} ${t[6]:>+7,.0f} {t[8]}")
    else:
        print(f"  {'#':>3} {'Entry':>12} {'Exit':>12} {'Dir':<6} {'EntryPx':>9} {'ExitPx':>9} "
              f"{'Conf':>5} {'PnL':>9} {'Reason'}")
        print(f"  {'-'*85}")
        for i, t in enumerate(trades, 1):
            print(f"  {i:>3} {str(t.entry_date.date()):>12} {str(t.exit_date.date()):>12} "
                  f"{t.direction:<6} ${t.entry_price:>7.2f} ${t.exit_price:>7.2f} "
                  f"{t.confidence:>5.2f} ${t.pnl:>+7,.0f} {t.exit_reason}")


# ---------------------------------------------------------------------------
# A. CS-5TF (DI v36 champion combo)
# ---------------------------------------------------------------------------

def _build_di_combo(signals, daily_df, spy_daily, vix_daily):
    """Build the DI v36 combo function, reproducing phase2_run_combos setup."""
    from v15.validation.combo_backtest import (
        _make_v36_di,
    )

    cascade_vix = _build_filter_cascade(vix=True)
    cascade_vix.precompute_vix_cooldown(vix_daily)

    spy_above_sma20 = set()
    spy_above_055pct = set()
    spy_dist_map = {}
    spy_dist_5 = {}
    spy_dist_50 = {}
    vix_map = {}
    spy_return_map = {}
    spy_ret_2d = {}

    if spy_daily is not None and len(spy_daily) > 20:
        spy_close = spy_daily['close'].values.astype(float)
        spy_sma20 = pd.Series(spy_close).rolling(20).mean().values
        for i in range(20, len(spy_close)):
            if spy_close[i] > spy_sma20[i]:
                spy_above_sma20.add(spy_daily.index[i])
            if spy_sma20[i] > 0:
                dist_pct = (spy_close[i] - spy_sma20[i]) / spy_sma20[i] * 100
                if dist_pct >= 0.55:
                    spy_above_055pct.add(spy_daily.index[i])
                spy_dist_map[spy_daily.index[i]] = dist_pct

        for win, dist_dict in [(5, spy_dist_5), (50, spy_dist_50)]:
            if len(spy_daily) > win:
                sma = pd.Series(spy_close).rolling(win).mean().values
                for i in range(win, len(spy_close)):
                    if sma[i] > 0:
                        dist_dict[spy_daily.index[i]] = (spy_close[i] - sma[i]) / sma[i] * 100

    if vix_daily is not None:
        for idx, row in vix_daily.iterrows():
            vix_map[idx] = row['close']

    if spy_daily is not None:
        spy_close_arr = spy_daily['close'].values.astype(float)
        for i in range(1, len(spy_close_arr)):
            spy_return_map[spy_daily.index[i]] = (spy_close_arr[i] - spy_close_arr[i-1]) / spy_close_arr[i-1] * 100
        for i in range(2, len(spy_close_arr)):
            spy_ret_2d[spy_daily.index[i]] = (spy_close_arr[i] - spy_close_arr[i-2]) / spy_close_arr[i-2] * 100

    fn = _make_v36_di(cascade_vix, spy_above_sma20, spy_above_055pct,
                       spy_dist_map, spy_dist_5, spy_dist_50,
                       vix_map, spy_return_map, spy_ret_2d)
    return fn


def validate_cs_5tf(signals, daily_df, spy_daily, vix_daily):
    """Validate CS-5TF (DI v36) with all 4 validation methods."""
    print("\n" + "#"*75)
    print("  A. CS-5TF — DI v36 Champion Combo")
    print("#"*75)

    combo_fn = _build_di_combo(signals, daily_df, spy_daily, vix_daily)
    tp = SETTINGS['cs_5tf']['trail_power']

    # 1. Per-year
    year_results = {}
    for yr in range(2016, 2027):
        yr_signals = [s for s in signals if s.date.year == yr]
        if not yr_signals:
            continue
        trades = simulate_trades(yr_signals, combo_fn, f'CS-5TF-{yr}', cooldown=0, trail_power=tp)
        year_results[yr] = _compute_metrics(trades)
    _print_year_table(year_results, 'CS-5TF (DI v36)')

    # 2. Holdout
    train_signals = [s for s in signals if s.date.year <= 2021]
    test_signals = [s for s in signals if 2022 <= s.date.year <= 2025]
    train_trades = simulate_trades(train_signals, combo_fn, 'CS-5TF-train', cooldown=0, trail_power=tp)
    test_trades = simulate_trades(test_signals, combo_fn, 'CS-5TF-test', cooldown=0, trail_power=tp)
    _print_holdout(_compute_metrics(train_trades), _compute_metrics(test_trades))

    # 3. Walk-forward
    wf_results = []
    for train_range, test_yr in WALK_FORWARD_WINDOWS:
        ty_start, ty_end = [int(y) for y in train_range.split('-')]
        tr_sigs = [s for s in signals if ty_start <= s.date.year <= ty_end]
        te_sigs = [s for s in signals if s.date.year == int(test_yr)]
        tr_trades = simulate_trades(tr_sigs, combo_fn, f'WF-train-{train_range}', cooldown=0, trail_power=tp)
        te_trades = simulate_trades(te_sigs, combo_fn, f'WF-test-{test_yr}', cooldown=0, trail_power=tp)
        wf_results.append({
            'train_period': train_range, 'test_year': test_yr,
            'train': _compute_metrics(tr_trades), 'test': _compute_metrics(te_trades),
        })
    _print_walkforward(wf_results)

    # 4. 2026 OOS
    oos_signals = [s for s in signals if s.date.year >= 2026]
    oos_trades = simulate_trades(oos_signals, combo_fn, 'CS-5TF-2026', cooldown=0, trail_power=tp)
    _print_2026_trades(oos_trades, 'daily')

    # All trades combined for summary
    all_trades = simulate_trades(signals, combo_fn, 'CS-5TF-all', cooldown=0, trail_power=tp)
    all_metrics = _compute_metrics(all_trades)
    print(f"\n  ALL PERIODS: {all_metrics['trades']} trades, "
          f"{all_metrics['win_rate']}% WR, ${all_metrics['total_pnl']:+,}, "
          f"Sharpe={all_metrics['sharpe']}")

    return {
        'signal_type': 'CS-5TF',
        'settings': SETTINGS['cs_5tf'],
        'per_year': year_results,
        'holdout': {'train': _compute_metrics(train_trades), 'test': _compute_metrics(test_trades)},
        'walk_forward': wf_results,
        'oos_2026': _compute_metrics(oos_trades),
        'all': all_metrics,
    }


# ---------------------------------------------------------------------------
# B. CS-DW (Daily + Weekly only)
# ---------------------------------------------------------------------------

def _phase1_precompute_dw(tsla_min_path, start, end):
    """Phase 1 with only daily + weekly TFs (no 5min/1h/4h)."""
    from v15.validation.tf_state_backtest import load_all_tfs
    from v15.data.native_tf import fetch_native_tf
    from v15.core.channel_surfer import prepare_multi_tf_analysis

    # Load all TFs but only pass daily+weekly to channel surfer
    tf_data = load_all_tfs(tsla_min_path, start, end)
    daily_df = tf_data['daily']

    # Load SPY + VIX
    spy_daily = fetch_native_tf('SPY', 'daily', start, end)
    spy_daily.columns = [c.lower() for c in spy_daily.columns]
    spy_daily.index = pd.to_datetime(spy_daily.index).tz_localize(None)

    try:
        vix_daily = fetch_native_tf('^VIX', 'daily', start, end)
        vix_daily.columns = [c.lower() for c in vix_daily.columns]
        vix_daily.index = pd.to_datetime(vix_daily.index).tz_localize(None)
    except Exception:
        vix_daily = None

    trading_dates = daily_df.index
    signals = []
    n = len(trading_dates)
    warmup = 260
    t0 = time.time()

    print(f"\n  Phase 1 (DW): Computing signals for {n - warmup:,} trading days (daily+weekly only)...")

    # Build DW-only TF data
    dw_tf_data = {k: v for k, v in tf_data.items() if k in ('daily', 'weekly')}

    for i in range(warmup, n):
        date = trading_dates[i]
        if i % 200 == 0:
            elapsed = time.time() - t0
            pct = (i - warmup) / max(n - warmup, 1) * 100
            print(f"    {i}/{n} ({pct:.0f}%) -- {elapsed:.0f}s")

        row = DaySignals(date=date)
        row.day_open = float(daily_df['open'].iloc[i])
        row.day_high = float(daily_df['high'].iloc[i])
        row.day_low = float(daily_df['low'].iloc[i])
        row.day_close = float(daily_df['close'].iloc[i])

        try:
            # Build native_data with only daily + weekly
            native_slice = {}
            for tf, df in dw_tf_data.items():
                if df is None or len(df) == 0:
                    continue
                mask = df.index <= date + pd.Timedelta(days=1)
                sliced = df.loc[mask]
                if len(sliced) >= 15:
                    native_slice[tf] = sliced

            if native_slice:
                analysis = prepare_multi_tf_analysis(native_data={'TSLA': native_slice})
                sig = analysis.signal
                row.cs_action = sig.action
                row.cs_confidence = sig.confidence
                row.cs_stop_pct = sig.suggested_stop_pct
                row.cs_tp_pct = sig.suggested_tp_pct
                row.cs_signal_type = sig.signal_type
                row.cs_primary_tf = sig.primary_tf
                row.cs_reason = sig.reason
                row.cs_position_score = sig.position_score
                row.cs_energy_score = sig.energy_score
                row.cs_entropy_score = sig.entropy_score
                row.cs_confluence_score = sig.confluence_score
                row.cs_timing_score = sig.timing_score
                row.cs_channel_health = sig.channel_health
                tf_states_slim = {}
                for tf, st in (analysis.tf_states or {}).items():
                    tf_states_slim[tf] = {
                        'valid': getattr(st, 'valid', False),
                        'momentum_direction': getattr(st, 'momentum_direction', 0.0),
                        'momentum_is_turning': getattr(st, 'momentum_is_turning', False),
                    }
                row.cs_tf_states = tf_states_slim
        except Exception as e:
            if i % 500 == 0:
                print(f"      CS-DW error on {date.date()}: {e}")

        signals.append(row)

    elapsed = time.time() - t0
    print(f"  Phase 1 (DW) done: {len(signals):,} days in {elapsed:.0f}s")
    return signals, daily_df, spy_daily, vix_daily


def _make_cs_dw_baseline():
    """CS-DW baseline: any CS BUY/SELL above min confidence."""
    def fn(day: DaySignals):
        if day.cs_action not in ('BUY', 'SELL') or day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
            return None
        s = max(day.cs_stop_pct, DEFAULT_STOP_PCT)
        t = max(day.cs_tp_pct, DEFAULT_TP_PCT)
        return (day.cs_action, day.cs_confidence, s, t, 'CS')
    return fn


def validate_cs_dw(signals_dw, daily_df, spy_daily, vix_daily):
    """Validate CS-DW with baseline (no combo filters)."""
    print("\n" + "#"*75)
    print("  B. CS-DW — Daily + Weekly Only (Baseline)")
    print("#"*75)

    combo_fn = _make_cs_dw_baseline()
    tp = SETTINGS['cs_dw']['trail_power']

    # 1. Per-year
    year_results = {}
    for yr in range(2016, 2027):
        yr_signals = [s for s in signals_dw if s.date.year == yr]
        if not yr_signals:
            continue
        trades = simulate_trades(yr_signals, combo_fn, f'CS-DW-{yr}', cooldown=0, trail_power=tp)
        year_results[yr] = _compute_metrics(trades)
    _print_year_table(year_results, 'CS-DW (baseline)')

    # 2. Holdout
    train_signals = [s for s in signals_dw if s.date.year <= 2021]
    test_signals = [s for s in signals_dw if 2022 <= s.date.year <= 2025]
    train_trades = simulate_trades(train_signals, combo_fn, 'CS-DW-train', cooldown=0, trail_power=tp)
    test_trades = simulate_trades(test_signals, combo_fn, 'CS-DW-test', cooldown=0, trail_power=tp)
    _print_holdout(_compute_metrics(train_trades), _compute_metrics(test_trades))

    # 3. Walk-forward
    wf_results = []
    for train_range, test_yr in WALK_FORWARD_WINDOWS:
        ty_start, ty_end = [int(y) for y in train_range.split('-')]
        tr_sigs = [s for s in signals_dw if ty_start <= s.date.year <= ty_end]
        te_sigs = [s for s in signals_dw if s.date.year == int(test_yr)]
        tr_trades = simulate_trades(tr_sigs, combo_fn, f'WF-DW-train-{train_range}', cooldown=0, trail_power=tp)
        te_trades = simulate_trades(te_sigs, combo_fn, f'WF-DW-test-{test_yr}', cooldown=0, trail_power=tp)
        wf_results.append({
            'train_period': train_range, 'test_year': test_yr,
            'train': _compute_metrics(tr_trades), 'test': _compute_metrics(te_trades),
        })
    _print_walkforward(wf_results)

    # 4. 2026 OOS
    oos_signals = [s for s in signals_dw if s.date.year >= 2026]
    oos_trades = simulate_trades(oos_signals, combo_fn, 'CS-DW-2026', cooldown=0, trail_power=tp)
    _print_2026_trades(oos_trades, 'daily')

    all_trades = simulate_trades(signals_dw, combo_fn, 'CS-DW-all', cooldown=0, trail_power=tp)
    all_metrics = _compute_metrics(all_trades)
    print(f"\n  ALL PERIODS: {all_metrics['trades']} trades, "
          f"{all_metrics['win_rate']}% WR, ${all_metrics['total_pnl']:+,}, "
          f"Sharpe={all_metrics['sharpe']}")

    return {
        'signal_type': 'CS-DW',
        'settings': SETTINGS['cs_dw'],
        'per_year': year_results,
        'holdout': {'train': _compute_metrics(train_trades), 'test': _compute_metrics(test_trades)},
        'walk_forward': wf_results,
        'oos_2026': _compute_metrics(oos_trades),
        'all': all_metrics,
    }


# ---------------------------------------------------------------------------
# C. Intraday (FD Enhanced-Union)
# ---------------------------------------------------------------------------

def _load_intraday_data(tsla_min_path):
    """Load and build intraday features from 1-min data."""
    from v15.validation.intraday_v14b_janfeb import load_1min, build_features, precompute_all
    df1m = load_1min(tsla_min_path)
    features = build_features(df1m)
    f5m = features['5m']
    precomp = precompute_all(features, f5m)
    return f5m, precomp


def _simulate_intraday_year(f5m, precomp, year):
    """Run intraday sim for a single year."""
    from v15.validation.intraday_v14b_janfeb import sig_union_enh, simulate_fixed
    mask = np.array([t.year == year for t in f5m.index])
    if not mask.any():
        return []
    f5m_yr = f5m.loc[mask]
    # Need to recompute precomp arrays for the subset indices
    # Instead, use the full arrays but filter trades by year
    pw = {'stop': 0.008, 'tp': 0.020, 'd_min': 0.20, 'h1_min': 0.15, 'f5_thresh': 0.35,
          'div_thresh': 0.20, 'vwap_thresh': -0.10, 'min_vol_ratio': 0.8}
    trades = simulate_fixed(f5m, sig_union_enh, f'Intraday-{year}', pw, precomp,
                            tb=0.006, tp=6, cd=0, mtd=30,
                            base_capital=100_000.0, max_capital=100_000.0,
                            conf_size=False,
                            tod_start=dt.time(13, 0), tod_end=dt.time(15, 25))
    # Filter to just this year
    return [t for t in trades if t[0].year == year]


def _simulate_intraday_range(f5m, precomp, year_start, year_end):
    """Run intraday sim and return trades within year range."""
    from v15.validation.intraday_v14b_janfeb import sig_union_enh, simulate_fixed
    pw = {'stop': 0.008, 'tp': 0.020, 'd_min': 0.20, 'h1_min': 0.15, 'f5_thresh': 0.35,
          'div_thresh': 0.20, 'vwap_thresh': -0.10, 'min_vol_ratio': 0.8}
    trades = simulate_fixed(f5m, sig_union_enh, f'Intraday-{year_start}-{year_end}', pw, precomp,
                            tb=0.006, tp=6, cd=0, mtd=30,
                            base_capital=100_000.0, max_capital=100_000.0,
                            conf_size=False,
                            tod_start=dt.time(13, 0), tod_end=dt.time(15, 25))
    return [t for t in trades if year_start <= t[0].year <= year_end]


def validate_intraday(tsla_min_path):
    """Validate intraday FD Enhanced-Union."""
    print("\n" + "#"*75)
    print("  C. Intraday — FD Enhanced-Union (PM 13:00-15:25, Long-Only)")
    print("#"*75)

    print("\n  Loading intraday data and features...")
    f5m, precomp = _load_intraday_data(tsla_min_path)

    # Run full simulation once, then filter by year
    from v15.validation.intraday_v14b_janfeb import sig_union_enh, simulate_fixed
    pw = {'stop': 0.008, 'tp': 0.020, 'd_min': 0.20, 'h1_min': 0.15, 'f5_thresh': 0.35,
          'div_thresh': 0.20, 'vwap_thresh': -0.10, 'min_vol_ratio': 0.8}

    print("  Running full intraday simulation...")
    t0 = time.time()
    all_trades = simulate_fixed(f5m, sig_union_enh, 'Intraday-full', pw, precomp,
                                tb=0.006, tp=6, cd=0, mtd=30,
                                base_capital=100_000.0, max_capital=100_000.0,
                                conf_size=False,
                                tod_start=dt.time(13, 0), tod_end=dt.time(15, 25))
    print(f"  Done: {len(all_trades)} trades in {time.time()-t0:.1f}s")

    # 1. Per-year
    year_results = {}
    for yr in range(2016, 2027):
        yr_trades = [t for t in all_trades if t[0].year == yr]
        if not yr_trades:
            continue
        year_results[yr] = _compute_metrics_intraday(yr_trades)
    _print_year_table(year_results, 'Intraday FD Enh-Union')

    # 2. Holdout
    train_trades = [t for t in all_trades if t[0].year <= 2021]
    test_trades = [t for t in all_trades if 2022 <= t[0].year <= 2025]
    _print_holdout(_compute_metrics_intraday(train_trades), _compute_metrics_intraday(test_trades))

    # 3. Walk-forward
    wf_results = []
    for train_range, test_yr in WALK_FORWARD_WINDOWS:
        ty_start, ty_end = [int(y) for y in train_range.split('-')]
        tr_trades = [t for t in all_trades if ty_start <= t[0].year <= ty_end]
        te_trades = [t for t in all_trades if t[0].year == int(test_yr)]
        wf_results.append({
            'train_period': train_range, 'test_year': test_yr,
            'train': _compute_metrics_intraday(tr_trades),
            'test': _compute_metrics_intraday(te_trades),
        })
    _print_walkforward(wf_results)

    # 4. 2026 OOS
    oos_trades = [t for t in all_trades if t[0].year >= 2026]
    _print_2026_trades(oos_trades, 'intraday')

    all_metrics = _compute_metrics_intraday(all_trades)
    print(f"\n  ALL PERIODS: {all_metrics['trades']} trades, "
          f"{all_metrics['win_rate']}% WR, ${all_metrics['total_pnl']:+,}, "
          f"Sharpe={all_metrics['sharpe']}")

    return {
        'signal_type': 'Intraday',
        'settings': SETTINGS['intraday'],
        'per_year': year_results,
        'holdout': {
            'train': _compute_metrics_intraday(train_trades),
            'test': _compute_metrics_intraday(test_trades),
        },
        'walk_forward': wf_results,
        'oos_2026': _compute_metrics_intraday(oos_trades),
        'all': all_metrics,
    }


# ---------------------------------------------------------------------------
# D. Surfer ML (5-min bars, 26 models, realistic mode)
# ---------------------------------------------------------------------------

def _compute_metrics_surfer(trades) -> dict:
    """Compute metrics from surfer_backtest Trade objects."""
    n = len(trades)
    if n == 0:
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'sharpe': 0, 'max_dd_pct': 0,
                'avg_win': 0, 'avg_loss': 0, 'biggest_loss': 0}
    pnls = np.array([t.pnl for t in trades])
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100
    total = float(pnls.sum())
    avg_w = float(np.mean([p for p in pnls if p > 0])) if wins > 0 else 0
    avg_l = float(np.mean([p for p in pnls if p <= 0])) if wins < n else 0
    bl = float(pnls.min())
    avg_hold = float(np.mean([t.hold_bars for t in trades]))
    # Annualize assuming ~78 bars/day
    bars_per_day = 78
    trades_per_year = 252 * bars_per_day / max(avg_hold, 1)
    sharpe = (float(pnls.mean() / pnls.std() * np.sqrt(trades_per_year))
              if pnls.std() > 0 else 0.0)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    mdd = float(dd.max()) / CAPITAL * 100 if len(dd) > 0 else 0
    return {
        'trades': n, 'wins': wins, 'win_rate': round(wr, 1),
        'total_pnl': round(total), 'avg_win': round(avg_w),
        'avg_loss': round(avg_l), 'biggest_loss': round(bl),
        'sharpe': round(sharpe, 2), 'max_dd_pct': round(mdd, 1),
        'avg_hold_bars': round(avg_hold, 1),
    }


def _load_surfer_data(tsla_min_path, start, end):
    """Load 1-min data and resample to 5-min + higher TFs for surfer_backtest."""
    from v15.validation.intraday_v14b_janfeb import load_1min

    MKT_OPEN = dt.time(9, 30)
    MKT_CLOSE = dt.time(16, 0)

    print("  Loading 1-min data...")
    df1m = load_1min(tsla_min_path)

    # Resample to 5-min
    print("  Resampling to 5-min...")
    tsla_5m = df1m.resample('5min').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()
    print(f"  5-min bars: {len(tsla_5m):,}")

    # Higher TFs
    print("  Building higher TFs...")
    higher_tf = {}
    higher_tf['1h'] = df1m.resample('1h').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()
    higher_tf['4h'] = df1m.resample('4h').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()
    higher_tf['daily'] = df1m.resample('1D').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()
    higher_tf['weekly'] = df1m.resample('W-FRI').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()

    for tf, df in higher_tf.items():
        print(f"    {tf}: {len(df):,} bars")

    # SPY + VIX (daily, via yfinance — needed for ML correlation features)
    from v15.data.native_tf import fetch_native_tf
    print("  Loading SPY daily...")
    spy_daily = fetch_native_tf('SPY', 'daily', start, end)
    spy_daily.columns = [c.lower() for c in spy_daily.columns]
    spy_daily.index = pd.to_datetime(spy_daily.index).tz_localize(None)

    print("  Loading VIX daily...")
    try:
        vix_daily = fetch_native_tf('^VIX', 'daily', start, end)
        vix_daily.columns = [c.lower() for c in vix_daily.columns]
        vix_daily.index = pd.to_datetime(vix_daily.index).tz_localize(None)
    except Exception:
        vix_daily = None

    return tsla_5m, higher_tf, spy_daily, vix_daily


def _load_ml_model():
    """Load the best ML model for surfer_backtest."""
    model_path = Path('surfer_models/best_model.pt')
    if not model_path.exists():
        print(f"  WARNING: {model_path} not found, running without ML model")
        return None
    try:
        from v15.core.surfer_ml import GBTModel
        import torch
        model = torch.load(str(model_path), map_location='cpu', weights_only=False)
        print(f"  ML model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"  WARNING: Failed to load ML model: {e}")
        return None


def _load_sq_model():
    """Load signal quality model for ML position sizing."""
    for name in ('signal_quality_model_c10_arch2.pkl',
                 'signal_quality_model_tuned.pkl',
                 'signal_quality_model.pkl'):
        p = Path('v15/validation') / name
        if p.exists():
            try:
                with open(p, 'rb') as f:
                    model = pickle.load(f)
                print(f"  Signal quality model loaded: {name}")
                return model
            except Exception as e:
                print(f"  WARNING: Failed to load {name}: {e}")
    print("  WARNING: No signal quality model found")
    return None


def _run_surfer_for_years(tsla_5m, higher_tf, spy_daily, vix_daily,
                          ml_model, sq_model, year_start, year_end):
    """Run surfer_backtest for a year range, return trades in that range."""
    from v15.core.surfer_backtest import run_backtest

    # Slice 5-min data to year range (with warmup: include 60 days before)
    range_start = pd.Timestamp(f'{year_start}-01-01') - pd.Timedelta(days=90)
    range_end = pd.Timestamp(f'{year_end}-12-31 23:59:59')
    mask = (tsla_5m.index >= range_start) & (tsla_5m.index <= range_end)
    tsla_slice = tsla_5m.loc[mask]

    if len(tsla_slice) < 100:
        return []

    # Slice higher TFs similarly (wider warmup)
    htf_warmup = pd.Timestamp(f'{year_start}-01-01') - pd.Timedelta(days=365)
    htf_slice = {}
    for tf, df in higher_tf.items():
        htf_slice[tf] = df.loc[(df.index >= htf_warmup) & (df.index <= range_end)]

    # ML sizing function
    ml_size_fn = None
    if sq_model is not None:
        def ml_size_fn(quality_score):
            # Scale 0.5x to 2x based on quality score
            return max(0.5, min(2.0, quality_score * 2.0))

    metrics, trades, equity_curve = run_backtest(
        days=0,  # Ignored when tsla_df provided
        eval_interval=3,
        max_hold_bars=60,
        position_size=10000.0,
        min_confidence=0.01,
        use_multi_tf=True,
        ml_model=ml_model,
        tsla_df=tsla_slice,
        higher_tf_dict=htf_slice,
        spy_df_input=spy_daily,
        vix_df_input=vix_daily,
        realistic=True,
        slippage_bps=3.0,
        commission_per_share=0.005,
        max_leverage=4.0,
        initial_capital=100_000.0,
        signal_quality_model=sq_model,
        ml_size_fn=ml_size_fn,
    )

    # Filter trades to only the requested year range
    filtered = []
    for t in trades:
        if t.entry_time:
            try:
                entry_dt = pd.Timestamp(t.entry_time)
                if year_start <= entry_dt.year <= year_end:
                    filtered.append(t)
            except (ValueError, TypeError):
                pass
    return filtered


def validate_surfer_ml(tsla_min_path, start, end):
    """Validate surfer_backtest with full ML stack in realistic mode."""
    print("\n" + "#"*75)
    print("  D. Surfer ML — 26 Models, Realistic Mode, 5-Min Bars")
    print("#"*75)

    print("\n  Loading data...")
    tsla_5m, higher_tf, spy_daily, vix_daily = _load_surfer_data(tsla_min_path, start, end)

    print("\n  Loading models...")
    ml_model = _load_ml_model()
    sq_model = _load_sq_model()

    # 1. Per-year ($100K fresh each year)
    year_results = {}
    for yr in range(2016, 2027):
        print(f"\n  Running year {yr}...")
        trades = _run_surfer_for_years(tsla_5m, higher_tf, spy_daily, vix_daily,
                                        ml_model, sq_model, yr, yr)
        if trades:
            year_results[yr] = _compute_metrics_surfer(trades)
            m = year_results[yr]
            print(f"    {yr}: {m['trades']} trades, {m['win_rate']}% WR, ${m['total_pnl']:+,}")
        else:
            print(f"    {yr}: 0 trades")
    _print_year_table(year_results, 'Surfer ML (realistic)')

    # 2. Holdout
    print("\n  Running holdout (train ≤2021, test 2022-2025)...")
    train_trades = _run_surfer_for_years(tsla_5m, higher_tf, spy_daily, vix_daily,
                                          ml_model, sq_model, 2016, 2021)
    test_trades = _run_surfer_for_years(tsla_5m, higher_tf, spy_daily, vix_daily,
                                         ml_model, sq_model, 2022, 2025)
    _print_holdout(_compute_metrics_surfer(train_trades), _compute_metrics_surfer(test_trades))

    # 3. Walk-forward
    print("\n  Running walk-forward...")
    wf_results = []
    for train_range, test_yr in WALK_FORWARD_WINDOWS:
        ty_start, ty_end = [int(y) for y in train_range.split('-')]
        print(f"    {train_range} → {test_yr}...")
        tr_trades = _run_surfer_for_years(tsla_5m, higher_tf, spy_daily, vix_daily,
                                           ml_model, sq_model, ty_start, ty_end)
        te_trades = _run_surfer_for_years(tsla_5m, higher_tf, spy_daily, vix_daily,
                                           ml_model, sq_model, int(test_yr), int(test_yr))
        wf_results.append({
            'train_period': train_range, 'test_year': test_yr,
            'train': _compute_metrics_surfer(tr_trades),
            'test': _compute_metrics_surfer(te_trades),
        })
    _print_walkforward(wf_results)

    # 4. 2026 OOS
    print("\n  Running 2026 OOS...")
    oos_trades = _run_surfer_for_years(tsla_5m, higher_tf, spy_daily, vix_daily,
                                        ml_model, sq_model, 2026, 2026)
    # Print trade log for 2026
    print(f"\n  --- 2026 OOS Trade Log ---")
    if oos_trades:
        print(f"  {'#':>3} {'Entry':>20} {'Dir':<5} {'EntryPx':>9} {'ExitPx':>9} "
              f"{'Conf':>5} {'Size':>9} {'PnL':>9} {'Type':<8} {'Reason'}")
        print(f"  {'-'*100}")
        for i, t in enumerate(oos_trades, 1):
            print(f"  {i:>3} {t.entry_time[:19]:>20} {t.direction:<5} ${t.entry_price:>7.2f} "
                  f"${t.exit_price:>7.2f} {t.confidence:>5.2f} ${t.trade_size:>7,.0f} "
                  f"${t.pnl:>+7,.0f} {t.signal_type:<8} {t.exit_reason}")
    else:
        print("  No 2026 trades.")

    # All-years combined
    all_trades = []
    for yr in range(2016, 2027):
        all_trades.extend(_run_surfer_for_years(tsla_5m, higher_tf, spy_daily, vix_daily,
                                                  ml_model, sq_model, yr, yr))
    all_metrics = _compute_metrics_surfer(all_trades) if all_trades else {
        'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'sharpe': 0, 'max_dd_pct': 0,
        'avg_win': 0, 'avg_loss': 0, 'biggest_loss': 0}
    print(f"\n  ALL PERIODS: {all_metrics['trades']} trades, "
          f"{all_metrics['win_rate']}% WR, ${all_metrics['total_pnl']:+,}, "
          f"Sharpe={all_metrics['sharpe']}")

    return {
        'signal_type': 'Surfer ML',
        'settings': SETTINGS['surfer_ml'],
        'per_year': year_results,
        'holdout': {'train': _compute_metrics_surfer(train_trades),
                    'test': _compute_metrics_surfer(test_trades)},
        'walk_forward': wf_results,
        'oos_2026': _compute_metrics_surfer(oos_trades),
        'all': all_metrics,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_comparison(results: dict):
    """Print side-by-side comparison of all signal types."""
    print("\n" + "="*80)
    print("  CROSS-SIGNAL COMPARISON")
    print("="*80)
    print(f"  {'Metric':<25} ", end='')
    for key in results:
        print(f"{results[key]['signal_type']:>18}", end='')
    print()
    print(f"  {'-'*25} ", end='')
    for _ in results:
        print(f"{'─'*18}", end='')
    print()

    metrics_to_show = [
        ('All — Trades', lambda r: str(r['all']['trades'])),
        ('All — Win Rate', lambda r: f"{r['all']['win_rate']}%"),
        ('All — Total PnL', lambda r: f"${r['all']['total_pnl']:+,}"),
        ('All — Sharpe', lambda r: f"{r['all']['sharpe']:.2f}"),
        ('All — Max DD%', lambda r: f"{r['all']['max_dd_pct']:.1f}%"),
        ('Holdout Test WR', lambda r: f"{r['holdout']['test']['win_rate']}%"),
        ('Holdout Test PnL', lambda r: f"${r['holdout']['test']['total_pnl']:+,}"),
        ('2026 OOS Trades', lambda r: str(r['oos_2026']['trades'])),
        ('2026 OOS WR', lambda r: f"{r['oos_2026']['win_rate']}%"),
        ('2026 OOS PnL', lambda r: f"${r['oos_2026']['total_pnl']:+,}"),
    ]

    for label, extractor in metrics_to_show:
        print(f"  {label:<25} ", end='')
        for key in results:
            try:
                val = extractor(results[key])
            except (KeyError, TypeError):
                val = 'N/A'
            print(f"{val:>18}", end='')
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='c13a Full Signal Validation')
    parser.add_argument('--tsla', type=str, default=None,
                        help='Path to TSLAMin.txt (1-min data)')
    parser.add_argument('--start', type=str, default='2016-01-01')
    parser.add_argument('--end', type=str, default='2026-12-31')
    parser.add_argument('--phase2-only', action='store_true',
                        help='Skip phase 1, load CS-5TF signals from cache')
    parser.add_argument('--skip-intraday', action='store_true',
                        help='Skip intraday validation (requires 1-min data)')
    parser.add_argument('--skip-dw', action='store_true',
                        help='Skip CS-DW validation (slow phase 1)')
    parser.add_argument('--skip-ml', action='store_true',
                        help='Skip Surfer ML validation (slow, needs models)')
    parser.add_argument('--only-ml', action='store_true',
                        help='Run ONLY Surfer ML validation')
    args = parser.parse_args()

    # Auto-detect TSLAMin.txt
    if args.tsla is None:
        for candidate in ['data/TSLAMin.txt', '../data/TSLAMin.txt',
                          r'C:\AI\x14\data\TSLAMin.txt',
                          os.path.expanduser('~/Desktop/Coding/x14/data/TSLAMin.txt')]:
            if os.path.isfile(candidate):
                args.tsla = candidate
                break

    print("="*75)
    print("  c13a FULL SIGNAL VALIDATION")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  TSLAMin: {args.tsla or 'not found'}")
    print("="*75)

    results = {}

    if args.only_ml:
        # Skip A/B/C, go straight to D
        if args.tsla and os.path.isfile(args.tsla):
            print("\n[D] Running Surfer ML validation ONLY (realistic, 26 models)...")
            results['surfer_ml'] = validate_surfer_ml(args.tsla, args.start, args.end)
        else:
            print("\n[D] Cannot run Surfer ML (no TSLAMin.txt found)")
    else:
        # ── A. CS-5TF ────────────────────────────────────────────────────
        print("\n[A] Loading CS-5TF signals...")
        if args.phase2_only and CACHE_FILE.exists():
            print("  Loading from cache...")
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
            signals = cache['signals']
            daily_df = cache['daily_df']
            spy_daily = cache['spy_daily']
            vix_daily = cache.get('vix_daily')
            print(f"  Loaded {len(signals):,} days from cache")
        else:
            signals, daily_df, spy_daily, vix_daily, _ = phase1_precompute(
                args.tsla, args.start, args.end)

        results['cs_5tf'] = validate_cs_5tf(signals, daily_df, spy_daily, vix_daily)

        # ── B. CS-DW ─────────────────────────────────────────────────────
        if not args.skip_dw:
            print("\n[B] Computing CS-DW signals (daily+weekly only)...")
            signals_dw, daily_df_dw, spy_daily_dw, vix_daily_dw = _phase1_precompute_dw(
                args.tsla, args.start, args.end)
            results['cs_dw'] = validate_cs_dw(signals_dw, daily_df_dw, spy_daily_dw, vix_daily_dw)
        else:
            print("\n[B] Skipping CS-DW (--skip-dw)")

        # ── C. Intraday ──────────────────────────────────────────────────
        if not args.skip_intraday and args.tsla and os.path.isfile(args.tsla):
            print("\n[C] Running intraday validation...")
            results['intraday'] = validate_intraday(args.tsla)
        elif args.skip_intraday:
            print("\n[C] Skipping intraday (--skip-intraday)")
        else:
            print("\n[C] Skipping intraday (no TSLAMin.txt found)")

        # ── D. Surfer ML ──────────────────────────────────────────────────
        if not args.skip_ml:
            if args.tsla and os.path.isfile(args.tsla):
                print("\n[D] Running Surfer ML validation (realistic, 26 models)...")
                results['surfer_ml'] = validate_surfer_ml(args.tsla, args.start, args.end)
            else:
                print("\n[D] Skipping Surfer ML (no TSLAMin.txt found)")
        else:
            print("\n[D] Skipping Surfer ML (--skip-ml)")

    # ── Comparison ────────────────────────────────────────────────────────
    if len(results) > 1:
        print_comparison(results)

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / 'c13a_validation_results.json'
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            v = _convert(obj)
            if v is not obj:
                return v
            return super().default(obj)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")

    print("\n" + "="*75)
    print("  DONE")
    print("="*75)


if __name__ == '__main__':
    main()
