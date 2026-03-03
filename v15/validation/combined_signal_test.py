#!/usr/bin/env python3
"""
Combined Signal Test — All 4 Signal Types Running Together

Runs each backtest engine independently (each uses its own validated engine),
then merges all trades onto a shared timeline to analyze:
  1. Overlap: How many days have trades from multiple signal types?
  2. Conflict: Do signal types ever bet in opposite directions on the same day?
  3. Stacked losses: How many days have losses from 2+ signal types?
  4. Combined P&L: Total equity curve with all 4 running together
  5. Combined metrics: Overall WR, Sharpe, max drawdown

Usage:
    python -m v15.validation.combined_signal_test
    python -m v15.validation.combined_signal_test --tsla data/TSLAMin.txt
"""

import argparse
import os
import sys
import time
import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.combo_backtest import (
    DaySignals, Trade, simulate_trades, phase1_precompute,
    _build_filter_cascade, _make_v36_di,
    CAPITAL, SLIPPAGE_PCT, COMMISSION_PER_SHARE,
)

FLAT_SIZE = 100_000


# ---------------------------------------------------------------------------
# Trade collection helpers
# ---------------------------------------------------------------------------

def _normalize_trade(trade, source: str) -> dict:
    """Convert any trade format to a common dict for comparison."""
    if source == 'intraday':
        # Intraday tuple: (entry_time, exit_time, entry_px, exit_px, conf, shares, pnl, hold_bars, exit_reason, name)
        entry_dt = trade[0]
        exit_dt = trade[1]
        if hasattr(entry_dt, 'date'):
            entry_date = entry_dt.date() if callable(entry_dt.date) else entry_dt.date
        else:
            entry_date = pd.Timestamp(entry_dt).date()
        if hasattr(exit_dt, 'date'):
            exit_date = exit_dt.date() if callable(exit_dt.date) else exit_dt.date
        else:
            exit_date = pd.Timestamp(exit_dt).date()
        return {
            'source': source,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'direction': 'long',  # intraday is long-only
            'pnl': float(trade[6]),
            'confidence': float(trade[4]),
            'entry_price': float(trade[2]),
            'exit_price': float(trade[3]),
        }
    elif source == 'surfer_ml':
        # surfer_backtest returns Trade-like objects with .entry_date, .exit_date, .pnl
        entry_date = trade.get('entry_date', trade.get('entry_time', ''))
        exit_date = trade.get('exit_date', trade.get('exit_time', ''))
        if hasattr(entry_date, 'date'):
            entry_date = entry_date.date() if callable(entry_date.date) else entry_date.date
        elif isinstance(entry_date, str):
            entry_date = pd.Timestamp(entry_date).date()
        if hasattr(exit_date, 'date'):
            exit_date = exit_date.date() if callable(exit_date.date) else exit_date.date
        elif isinstance(exit_date, str):
            exit_date = pd.Timestamp(exit_date).date()
        return {
            'source': source,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'direction': trade.get('direction', 'long'),
            'pnl': float(trade.get('pnl', 0)),
            'confidence': float(trade.get('confidence', 0)),
            'entry_price': float(trade.get('entry_price', 0)),
            'exit_price': float(trade.get('exit_price', 0)),
        }
    else:
        # combo_backtest Trade object
        return {
            'source': source,
            'entry_date': trade.entry_date.date() if hasattr(trade.entry_date, 'date') else trade.entry_date,
            'exit_date': trade.exit_date.date() if hasattr(trade.exit_date, 'date') else trade.exit_date,
            'direction': trade.direction.lower(),
            'pnl': float(trade.pnl),
            'confidence': float(trade.confidence),
            'entry_price': float(trade.entry_price),
            'exit_price': float(trade.exit_price),
        }


# ---------------------------------------------------------------------------
# Run each engine
# ---------------------------------------------------------------------------

def _run_cs_5tf(signals, daily_df, spy_daily, vix_daily):
    """Run CS-5TF (DI v36) and return trades."""
    cascade_vix = _build_filter_cascade(vix=True)
    cascade_vix.precompute_vix_cooldown(vix_daily)

    spy_close = spy_daily['close'].values.astype(float)
    spy_above_sma20 = set()
    spy_above_055pct = set()
    spy_dist_map, spy_dist_5, spy_dist_50 = {}, {}, {}
    vix_map, spy_return_map, spy_ret_2d = {}, {}, {}

    if len(spy_daily) > 20:
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
            sma = pd.Series(spy_close).rolling(win).mean().values
            for i in range(win, len(spy_close)):
                if sma[i] > 0:
                    dist_dict[spy_daily.index[i]] = (spy_close[i] - sma[i]) / sma[i] * 100

    if vix_daily is not None:
        for idx, row in vix_daily.iterrows():
            vix_map[idx] = row['close']

    spy_close_arr = spy_daily['close'].values.astype(float)
    for i in range(1, len(spy_close_arr)):
        spy_return_map[spy_daily.index[i]] = (spy_close_arr[i] - spy_close_arr[i-1]) / spy_close_arr[i-1] * 100
    for i in range(2, len(spy_close_arr)):
        spy_ret_2d[spy_daily.index[i]] = (spy_close_arr[i] - spy_close_arr[i-2]) / spy_close_arr[i-2] * 100

    fn = _make_v36_di(cascade_vix, spy_above_sma20, spy_above_055pct,
                       spy_dist_map, spy_dist_5, spy_dist_50,
                       vix_map, spy_return_map, spy_ret_2d)
    trades = simulate_trades(signals, fn, 'CS-5TF', cooldown=0, trail_power=8)
    return [_normalize_trade(t, 'CS-5TF') for t in trades]


def _run_cs_dw(tsla_min_path, start, end):
    """Run CS-DW and return trades."""
    from v15.validation.c13a_full_validation import _phase1_precompute_dw, validate_cs_dw
    # Use the validation's DW path
    from v15.validation.tf_state_backtest import load_all_tfs
    from v15.data.native_tf import fetch_native_tf
    from v15.core.channel_surfer import prepare_multi_tf_analysis

    tf_data = load_all_tfs(tsla_min_path, start, end)
    daily_df = tf_data['daily']
    spy_daily = fetch_native_tf('SPY', 'daily', start, end)
    spy_daily.columns = [c.lower() for c in spy_daily.columns]
    spy_daily.index = pd.to_datetime(spy_daily.index).tz_localize(None)
    try:
        vix_daily = fetch_native_tf('^VIX', 'daily', start, end)
        vix_daily.columns = [c.lower() for c in vix_daily.columns]
        vix_daily.index = pd.to_datetime(vix_daily.index).tz_localize(None)
    except Exception:
        vix_daily = None

    dw_tf_data = {k: v for k, v in tf_data.items() if k in ('daily', 'weekly')}
    trading_dates = daily_df.index
    signals = []
    warmup = 260

    print("  Computing CS-DW signals...")
    for i in range(warmup, len(trading_dates)):
        date = trading_dates[i]
        row = DaySignals(date=date)
        row.day_open = float(daily_df['open'].iloc[i])
        row.day_high = float(daily_df['high'].iloc[i])
        row.day_low = float(daily_df['low'].iloc[i])
        row.day_close = float(daily_df['close'].iloc[i])
        try:
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
                row.cs_channel_health = sig.channel_health
                row.cs_confluence_score = sig.confluence_score
                tf_states_slim = {}
                for tf, st in (analysis.tf_states or {}).items():
                    tf_states_slim[tf] = {
                        'valid': getattr(st, 'valid', False),
                        'momentum_direction': getattr(st, 'momentum_direction', 0.0),
                        'momentum_is_turning': getattr(st, 'momentum_is_turning', False),
                    }
                row.cs_tf_states = tf_states_slim
        except Exception:
            pass
        signals.append(row)

    # Use DI combo for DW signals too (same filter chain)
    cascade_vix = _build_filter_cascade(vix=True)
    cascade_vix.precompute_vix_cooldown(vix_daily)
    spy_close = spy_daily['close'].values.astype(float)
    spy_above_sma20 = set()
    spy_above_055pct = set()
    spy_dist_map, spy_dist_5, spy_dist_50 = {}, {}, {}
    vix_map, spy_return_map, spy_ret_2d = {}, {}, {}
    if len(spy_daily) > 20:
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
            sma = pd.Series(spy_close).rolling(win).mean().values
            for i in range(win, len(spy_close)):
                if sma[i] > 0:
                    dist_dict[spy_daily.index[i]] = (spy_close[i] - sma[i]) / sma[i] * 100
    if vix_daily is not None:
        for idx, row in vix_daily.iterrows():
            vix_map[idx] = row['close']
    spy_close_arr = spy_daily['close'].values.astype(float)
    for i in range(1, len(spy_close_arr)):
        spy_return_map[spy_daily.index[i]] = (spy_close_arr[i] - spy_close_arr[i-1]) / spy_close_arr[i-1] * 100
    for i in range(2, len(spy_close_arr)):
        spy_ret_2d[spy_daily.index[i]] = (spy_close_arr[i] - spy_close_arr[i-2]) / spy_close_arr[i-2] * 100

    fn = _make_v36_di(cascade_vix, spy_above_sma20, spy_above_055pct,
                       spy_dist_map, spy_dist_5, spy_dist_50,
                       vix_map, spy_return_map, spy_ret_2d)
    trades = simulate_trades(signals, fn, 'CS-DW', cooldown=0, trail_power=8)
    return [_normalize_trade(t, 'CS-DW') for t in trades]


def _run_intraday(tsla_min_path, start, end):
    """Run intraday FD Enhanced-Union and return trades."""
    from v15.validation.intraday_v14b_janfeb import (
        load_1min, simulate_fixed, build_features, precompute_all,
        sig_union_enh,
    )

    print("  Loading 1-min data for intraday...")
    df1 = load_1min(tsla_min_path)
    # Filter to date range before building features
    df1 = df1[(df1.index >= start) & (df1.index <= end)]
    print("  Building features (resampling + channel positions)...")
    features = build_features(df1)
    f5m = features['5m']
    print(f"  5-min bars: {len(f5m):,}")

    precomp = precompute_all(features, f5m)

    pw = {'stop': 0.008, 'tp': 0.020, 'd_min': 0.20, 'h1_min': 0.15, 'f5_thresh': 0.35,
          'div_thresh': 0.20, 'vwap_thresh': -0.10, 'min_vol_ratio': 0.8}
    print("  Running intraday simulation...")
    trades = simulate_fixed(f5m, sig_union_enh, 'Intraday', pw, precomp,
                           tb=0.006, tp=6, cd=0, mtd=30,
                           base_capital=FLAT_SIZE, max_capital=FLAT_SIZE,
                           conf_size=False,
                           tod_start=dt.time(13, 0), tod_end=dt.time(15, 25))
    return [_normalize_trade(t, 'intraday') for t in trades]


def _run_surfer_ml(tsla_min_path, start, end):
    """Run Surfer ML backtest and return trades."""
    import io
    from v15.validation.c13a_full_validation import (
        _load_surfer_data, _load_ml_model, _load_sq_model,
        _run_surfer_for_years,
    )

    print("  Loading Surfer data...")
    tsla_5m, higher_tf, spy_5m, vix_daily = _load_surfer_data(tsla_min_path, start, end)

    print("  Loading ML models...")
    ml_model = _load_ml_model()
    sq_model = _load_sq_model()

    print("  Running Surfer ML backtest (all years, output suppressed)...")
    all_trades = []
    for yr in range(2016, 2027):
        # Suppress verbose surfer_backtest output to avoid disk space issues
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            raw = _run_surfer_for_years(tsla_5m, higher_tf, spy_5m, vix_daily,
                                         ml_model, sq_model, yr, yr)
        finally:
            sys.stdout = old_stdout
        all_trades.extend(raw)
        print(f"    {yr}: {len(raw)} trades", flush=True)

    # Convert Trade dataclass objects to normalized format
    normalized = []
    for t in all_trades:
        entry_time = getattr(t, 'entry_time', '')
        entry_date = pd.Timestamp(entry_time).date() if entry_time else None
        # Estimate exit date from entry + hold_bars (5-min bars)
        hold_bars = getattr(t, 'hold_bars', 0)
        exit_date = entry_date
        if entry_date and hold_bars:
            exit_date = (pd.Timestamp(entry_time) + pd.Timedelta(minutes=hold_bars * 5)).date()
        direction = 'long' if getattr(t, 'direction', '') == 'BUY' else 'short'
        # Use pnl_pct * flat $10K sizing to avoid compounding distortion
        # (backtest compounds equity; raw t.pnl is astronomical)
        flat_pnl = float(getattr(t, 'pnl_pct', 0)) * 10_000.0
        normalized.append({
            'source': 'surfer_ml',
            'entry_date': entry_date,
            'exit_date': exit_date,
            'direction': direction,
            'pnl': flat_pnl,
            'confidence': float(t.confidence),
            'entry_price': float(t.entry_price),
            'exit_price': float(t.exit_price),
        })
    return normalized


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_combined(all_trades: Dict[str, List[dict]]):
    """Analyze interactions between signal types."""
    print("\n" + "=" * 80)
    print("  COMBINED SIGNAL ANALYSIS — All 4 Signal Types Running Together")
    print("=" * 80)

    # 1. Per-source summary
    print("\n  --- Per-Source Summary ---")
    print(f"  {'Source':<15} {'Trades':>7} {'WR%':>6} {'PnL':>12} {'AvgPnL':>8}")
    print(f"  {'-'*55}")
    total_trades = 0
    total_pnl = 0
    all_normalized = []
    for source in ['CS-5TF', 'CS-DW', 'intraday', 'surfer_ml']:
        trades = all_trades.get(source, [])
        n = len(trades)
        if n == 0:
            print(f"  {source:<15} {'0':>7}")
            continue
        pnls = [t['pnl'] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n * 100
        total = sum(pnls)
        avg = total / n
        print(f"  {source:<15} {n:>7} {wr:>5.1f}% ${total:>+10,.0f} ${avg:>+6,.0f}")
        total_trades += n
        total_pnl += total
        all_normalized.extend(trades)

    all_pnls = [t['pnl'] for t in all_normalized]
    combined_wins = sum(1 for p in all_pnls if p > 0)
    combined_wr = combined_wins / len(all_pnls) * 100 if all_pnls else 0
    print(f"  {'-'*55}")
    print(f"  {'COMBINED':<15} {total_trades:>7} {combined_wr:>5.1f}% ${total_pnl:>+10,.0f} "
          f"${total_pnl/max(total_trades,1):>+6,.0f}")

    # 2. Overlap analysis: trades active on the same dates
    print("\n  --- Daily Overlap Analysis ---")
    date_sources = defaultdict(set)  # date -> set of sources with active trades
    for t in all_normalized:
        entry = t['entry_date']
        exit_d = t['exit_date']
        if entry and exit_d:
            current = entry
            while current <= exit_d:
                date_sources[current].add(t['source'])
                current += dt.timedelta(days=1)

    overlap_counts = defaultdict(int)
    for date, sources in date_sources.items():
        overlap_counts[len(sources)] += 1

    for n_sources in sorted(overlap_counts.keys()):
        print(f"  {n_sources} signal type(s) active: {overlap_counts[n_sources]:>5} days")

    # 3. Conflict analysis: same-day entries in opposite directions
    print("\n  --- Direction Conflict Analysis ---")
    entry_date_dir = defaultdict(list)  # date -> [(source, direction)]
    for t in all_normalized:
        entry_date_dir[t['entry_date']].append((t['source'], t['direction']))

    conflicts = 0
    conflict_dates = []
    for date, entries in entry_date_dir.items():
        directions = set(d for _, d in entries)
        if len(directions) > 1:  # both long and short on same day
            conflicts += 1
            if len(conflict_dates) < 10:  # Show first 10
                conflict_dates.append((date, entries))

    print(f"  Days with conflicting directions: {conflicts}")
    if conflict_dates:
        for date, entries in conflict_dates[:5]:
            sources_str = ', '.join(f"{s}={d}" for s, d in entries)
            print(f"    {date}: {sources_str}")
        if conflicts > 5:
            print(f"    ... and {conflicts - 5} more")

    # 4. Stacked loss analysis
    print("\n  --- Stacked Loss Analysis ---")
    loss_dates = defaultdict(list)  # date -> [(source, pnl)]
    for t in all_normalized:
        if t['pnl'] < 0:
            loss_dates[t['exit_date']].append((t['source'], t['pnl']))

    multi_loss_days = {d: v for d, v in loss_dates.items() if len(v) > 1}
    print(f"  Days with losses from 2+ sources: {len(multi_loss_days)}")
    if multi_loss_days:
        for date in sorted(multi_loss_days.keys())[:5]:
            losses = multi_loss_days[date]
            total_loss = sum(p for _, p in losses)
            details = ', '.join(f"{s}=${p:+,.0f}" for s, p in losses)
            print(f"    {date}: {details} (total: ${total_loss:+,.0f})")

    # Worst combined day
    daily_pnl = defaultdict(float)
    for t in all_normalized:
        daily_pnl[t['exit_date']] += t['pnl']
    if daily_pnl:
        worst_date = min(daily_pnl, key=daily_pnl.get)
        best_date = max(daily_pnl, key=daily_pnl.get)
        print(f"\n  Worst combined day: {worst_date} (${daily_pnl[worst_date]:+,.0f})")
        print(f"  Best combined day:  {best_date} (${daily_pnl[best_date]:+,.0f})")

    # 5. Combined equity curve
    print("\n  --- Combined Equity Curve ---")
    sorted_trades = sorted(all_normalized, key=lambda t: (t['exit_date'] or dt.date.min, t['source']))
    equity = FLAT_SIZE
    peak = equity
    max_dd = 0
    yearly_pnl = defaultdict(float)

    for t in sorted_trades:
        equity += t['pnl']
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)
        if t['exit_date']:
            yearly_pnl[t['exit_date'].year] += t['pnl']

    print(f"  Starting capital: ${FLAT_SIZE:,.0f}")
    print(f"  Final equity:     ${equity:,.0f}")
    print(f"  Total P&L:        ${total_pnl:+,.0f}")
    print(f"  Max drawdown:     {max_dd:.1f}%")

    if all_pnls:
        pnl_arr = np.array(all_pnls)
        sharpe = float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(252)) if pnl_arr.std() > 0 else 0
        print(f"  Combined Sharpe:  {sharpe:.2f}")

    print(f"\n  --- Per-Year Combined P&L ---")
    print(f"  {'Year':>6} {'PnL':>12}")
    print(f"  {'-'*20}")
    for yr in sorted(yearly_pnl.keys()):
        print(f"  {yr:>6} ${yearly_pnl[yr]:>+10,.0f}")

    # 6. Per-year breakdown by source
    print(f"\n  --- Per-Year by Source ---")
    sources = ['CS-5TF', 'CS-DW', 'intraday', 'surfer_ml']
    header = f"  {'Year':>6}"
    for s in sources:
        header += f" {s:>14}"
    header += f" {'TOTAL':>12}"
    print(header)
    print(f"  {'-'*75}")

    for yr in sorted(yearly_pnl.keys()):
        row = f"  {yr:>6}"
        yr_total = 0
        for s in sources:
            yr_pnl = sum(t['pnl'] for t in all_trades.get(s, [])
                        if t['exit_date'] and t['exit_date'].year == yr)
            row += f" ${yr_pnl:>+12,.0f}"
            yr_total += yr_pnl
        row += f" ${yr_total:>+10,.0f}"
        print(row)

    return {
        'total_trades': total_trades,
        'combined_wr': round(combined_wr, 1),
        'total_pnl': round(total_pnl),
        'max_dd_pct': round(max_dd, 1),
        'conflict_days': conflicts,
        'multi_loss_days': len(multi_loss_days),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Combined Signal Test')
    parser.add_argument('--tsla', default='data/TSLAMin.txt', help='Path to TSLAMin.txt')
    parser.add_argument('--skip-dw', action='store_true', help='Skip CS-DW (slow Phase 1)')
    parser.add_argument('--skip-ml', action='store_true', help='Skip Surfer ML')
    parser.add_argument('--skip-intraday', action='store_true', help='Skip intraday')
    args = parser.parse_args()

    tsla_path = args.tsla
    for p in [tsla_path, f'../{tsla_path}', os.path.expanduser(f'~/Desktop/Coding/x14/{tsla_path}')]:
        if os.path.isfile(p):
            tsla_path = p
            break

    start = '2016-01-01'
    end = '2026-12-31'

    print("=" * 80)
    print("  COMBINED SIGNAL TEST — All 4 Signal Types")
    print(f"  Period: {start} to {end}")
    print(f"  Data: {tsla_path}")
    print("=" * 80)

    all_trades = {}
    t0 = time.time()

    # --- A. CS-5TF ---
    print("\n[A] Running CS-5TF (DI v36)...")
    try:
        signals, daily_df, spy_daily, vix_daily, _weekly = phase1_precompute(tsla_path, start, end)
        cs5tf_trades = _run_cs_5tf(signals, daily_df, spy_daily, vix_daily)
        all_trades['CS-5TF'] = cs5tf_trades
        print(f"  CS-5TF: {len(cs5tf_trades)} trades in {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"  CS-5TF FAILED: {e}")
        all_trades['CS-5TF'] = []

    # --- B. CS-DW ---
    if not args.skip_dw:
        t1 = time.time()
        print("\n[B] Running CS-DW (daily+weekly)...")
        try:
            csdw_trades = _run_cs_dw(tsla_path, start, end)
            all_trades['CS-DW'] = csdw_trades
            print(f"  CS-DW: {len(csdw_trades)} trades in {time.time()-t1:.0f}s")
        except Exception as e:
            print(f"  CS-DW FAILED: {e}")
            all_trades['CS-DW'] = []
    else:
        print("\n[B] Skipping CS-DW")
        all_trades['CS-DW'] = []

    # --- C. Intraday ---
    if not args.skip_intraday:
        t2 = time.time()
        print("\n[C] Running Intraday (FD Enh-Union)...")
        try:
            intraday_trades = _run_intraday(tsla_path, start, end)
            all_trades['intraday'] = intraday_trades
            print(f"  Intraday: {len(intraday_trades)} trades in {time.time()-t2:.0f}s")
        except Exception as e:
            print(f"  Intraday FAILED: {e}")
            all_trades['intraday'] = []
    else:
        print("\n[C] Skipping Intraday")
        all_trades['intraday'] = []

    # --- D. Surfer ML ---
    if not args.skip_ml:
        t3 = time.time()
        print("\n[D] Running Surfer ML (26 models)...")
        try:
            ml_trades = _run_surfer_ml(tsla_path, start, end)
            all_trades['surfer_ml'] = ml_trades
            print(f"  Surfer ML: {len(ml_trades)} trades in {time.time()-t3:.0f}s")
        except Exception as e:
            print(f"  Surfer ML FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_trades['surfer_ml'] = []
    else:
        print("\n[D] Skipping Surfer ML")
        all_trades['surfer_ml'] = []

    total_time = time.time() - t0
    print(f"\n  All engines completed in {total_time:.0f}s")

    # --- Combined analysis ---
    results = analyze_combined(all_trades)

    print(f"\n{'='*80}")
    print(f"  DONE. Total: {results['total_trades']} trades, "
          f"{results['combined_wr']}% WR, ${results['total_pnl']:+,}")
    print(f"  Max DD: {results['max_dd_pct']}%, "
          f"Conflicts: {results['conflict_days']} days, "
          f"Stacked losses: {results['multi_loss_days']} days")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
