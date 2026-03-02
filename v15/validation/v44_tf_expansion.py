#!/usr/bin/env python3
"""v44: TF Expansion Experiment — Test adding TFs to Channel Surfer analysis.

Phase 1: Pre-detect channels at ALL 12 TFs for every trading day (cached).
Phase 2: For each TF config, reconstruct signals from cached channels
         via analyze_channels() and simulate trades.

Usage:
    # Full run (Phase 1 + Phase 2):
    python v15/validation/v44_tf_expansion.py --tsla data/TSLAMin.txt

    # Phase 2 only (from cache):
    python v15/validation/v44_tf_expansion.py --phase2-only

    # Run specific configs:
    python v15/validation/v44_tf_expansion.py --phase2-only --configs 0,1,5

    # Phase 1 only (build cache):
    python v15/validation/v44_tf_expansion.py --phase1-only
"""

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on path
_proj_root = Path(__file__).resolve().parent.parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

# --- Imports from existing codebase ---

from v15.validation.combo_backtest import (
    DaySignals, Trade, simulate_trades, report_combo,
    _SigProxy, _AnalysisProxy, _build_filter_cascade, _floor_stop_tp,
    MIN_SIGNAL_CONFIDENCE, CAPITAL, TRAIN_END_YEAR,
)
from v15.core.channel_surfer import analyze_channels
from v15.core.channel import detect_channels_multi_window, select_best_channel
from v15.data.native_tf import fetch_native_tf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / 'combo_cache'
CHANNEL_CACHE_FILE = CACHE_DIR / 'tf_expansion_channels.pkl'

# Channel detection windows for ALL 12 timeframes
# (1min/3min new; 5min-monthly from channel_surfer.py TF_WINDOWS)
EXPANDED_TF_WINDOWS = {
    '1min':    [30, 60, 78, 120],
    '3min':    [10, 20, 30, 40, 60],
    '5min':    [10, 15, 20, 30, 40],
    '15min':   [10, 15, 20, 30],
    '30min':   [10, 15, 20, 30],
    '1h':      [10, 20, 30, 40, 50],
    '2h':      [10, 20, 30, 40],
    '3h':      [10, 20, 30],
    '4h':      [10, 20, 30, 40, 50],
    'daily':   [10, 20, 30, 40, 50, 60],
    'weekly':  [10, 20, 30, 40],
    'monthly': [10, 15, 20],
}

ALL_TFS = list(EXPANDED_TF_WINDOWS.keys())

TF_CONFIGS = [
    ('0:Baseline',  ['5min', '1h', '4h', 'daily', 'weekly']),
    ('1:+monthly',  ['5min', '1h', '4h', 'daily', 'weekly', 'monthly']),
    ('2:+15min',    ['5min', '15min', '1h', '4h', 'daily', 'weekly']),
    ('3:+30min',    ['5min', '30min', '1h', '4h', 'daily', 'weekly']),
    ('4:+15m30m',   ['5min', '15min', '30min', '1h', '4h', 'daily', 'weekly']),
    ('5:+2h',       ['5min', '1h', '2h', '4h', 'daily', 'weekly']),
    ('6:+3h',       ['5min', '1h', '3h', '4h', 'daily', 'weekly']),
    ('7:+2h3h',     ['5min', '1h', '2h', '3h', '4h', 'daily', 'weekly']),
    ('8:FullMid',   ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly']),
    ('9:Full10',    ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly']),
    ('10:+3min',    ['3min', '5min', '1h', '4h', 'daily', 'weekly']),
    ('11:+1m3m',    ['1min', '3min', '5min', '1h', '4h', 'daily', 'weekly']),
    ('12:All12',    ALL_TFS),
    ('13:No5min',   ['1h', '4h', 'daily', 'weekly']),
    ('14:DW',       ['daily', 'weekly']),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to a coarser timeframe."""
    return df.resample(rule).agg(
        {'open': 'first', 'high': 'max', 'low': 'min',
         'close': 'last', 'volume': 'sum'}
    ).dropna(subset=['close'])


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex and lowercase column names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df


def load_expanded_tfs(tsla_min_path: str, start: str, end: str) -> dict:
    """Load ALL 12 TFs from 1-min data + yfinance for daily/weekly/monthly.

    Returns: {tf_name: DataFrame} with OHLCV columns for each of 12 TFs.
    """
    # --- Daily, weekly, monthly from yfinance ---
    daily = fetch_native_tf('TSLA', 'daily', start, end)
    daily = _norm_cols(daily)
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    weekly = _resample_ohlcv(daily, 'W-FRI')
    monthly = _resample_ohlcv(daily, 'ME')

    # --- Intraday from 1-min file ---
    print(f"  Reading 1-min data: {tsla_min_path}")
    df1m = pd.read_csv(
        tsla_min_path, sep=';', header=None,
        names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
        parse_dates=['datetime'], date_format='%Y%m%d %H%M%S',
        index_col='datetime',
    )
    df1m = df1m.loc[start:end]
    print(f"  1-min: {len(df1m):,} bars "
          f"({df1m.index[0].date()} -> {df1m.index[-1].date()})")

    # Resample from 1-min
    tf3m  = _resample_ohlcv(df1m, '3min')
    tf5m  = _resample_ohlcv(df1m, '5min')
    tf15m = _resample_ohlcv(df1m, '15min')
    tf30m = _resample_ohlcv(df1m, '30min')
    tf1h  = _resample_ohlcv(df1m, '1h')

    # Resample from 1h
    tf2h = _resample_ohlcv(tf1h, '2h')
    tf3h = _resample_ohlcv(tf1h, '3h')
    tf4h = _resample_ohlcv(tf1h, '4h')

    result = {
        '1min': df1m, '3min': tf3m, '5min': tf5m,
        '15min': tf15m, '30min': tf30m,
        '1h': tf1h, '2h': tf2h, '3h': tf3h, '4h': tf4h,
        'daily': daily, 'weekly': weekly, 'monthly': monthly,
    }

    for tf, df in result.items():
        print(f"    {tf:>8s}: {len(df):>8,} bars")

    return result


# ---------------------------------------------------------------------------
# Index pre-computation for fast slicing
# ---------------------------------------------------------------------------

def _precompute_tf_positions(tf_data: dict, trading_dates) -> dict:
    """For each TF, map each trading date to its end-of-day row position.

    Returns: {tf: {date: end_pos}} so df.iloc[:end_pos] gives data up to date.
    """
    tf_pos = {}
    for tf in ALL_TFS:
        df = tf_data.get(tf)
        if df is None or len(df) == 0:
            tf_pos[tf] = {}
            continue
        idx = df.index
        idx_map = {}
        for date in trading_dates:
            cutoff = date + pd.Timedelta(days=1)
            pos = idx.searchsorted(cutoff, side='right')
            if pos >= 15:
                idx_map[date] = int(pos)
        tf_pos[tf] = idx_map
    return tf_pos


# ---------------------------------------------------------------------------
# Phase 1: Pre-detect channels at all 12 TFs
# ---------------------------------------------------------------------------

def phase1_detect_all_channels(tf_data: dict, daily_df: pd.DataFrame,
                               warmup: int = 260) -> dict:
    """Detect channels at ALL 12 TFs for every trading day.

    Returns: {date: {tf: Channel}} — only includes valid channels.
    """
    trading_dates = daily_df.index[warmup:]
    n_dates = len(trading_dates)

    print(f"\nPhase 1: Detecting channels — {n_dates} days x {len(ALL_TFS)} TFs")
    print(f"  Warmup: {warmup} days, first date: {trading_dates[0].date()}")

    # Pre-compute slice positions
    tf_pos = _precompute_tf_positions(tf_data, trading_dates)

    channel_cache = {}
    t0 = time.time()

    for di, date in enumerate(trading_dates):
        day_channels = {}

        for tf in ALL_TFS:
            pos = tf_pos[tf].get(date)
            if pos is None:
                continue

            df = tf_data[tf]
            sliced = df.iloc[:pos]

            windows = EXPANDED_TF_WINDOWS[tf]
            try:
                multi = detect_channels_multi_window(sliced, windows=windows)
                best_ch, _ = select_best_channel(multi)
            except Exception:
                best_ch = None

            if best_ch is not None and best_ch.valid:
                day_channels[tf] = best_ch

        if day_channels:
            channel_cache[date] = day_channels

        # Progress
        if (di + 1) % 200 == 0 or di == n_dates - 1:
            elapsed = time.time() - t0
            rate = (di + 1) / elapsed
            eta = (n_dates - di - 1) / rate if rate > 0 else 0
            avg_ch = sum(len(v) for v in channel_cache.values()) / max(di + 1, 1)
            print(f"  [{di+1}/{n_dates}] {elapsed:.0f}s elapsed, "
                  f"ETA {eta:.0f}s, avg {avg_ch:.1f} ch/day")

    total = sum(len(v) for v in channel_cache.values())
    print(f"  Phase 1 complete: {total:,} channels in {time.time()-t0:.1f}s")
    return channel_cache


# ---------------------------------------------------------------------------
# Phase 2: Sweep TF configs
# ---------------------------------------------------------------------------

def _build_signals_for_config(
    config_tfs: List[str],
    channel_cache: dict,
    tf_data: dict,
    daily_df: pd.DataFrame,
    trading_dates,
    tf_pos: dict,
) -> List[DaySignals]:
    """Build DaySignals for each trading day using a specific TF config.

    For each day:
      1. Look up cached channels for config's TFs
      2. Slice prices/volumes from pre-extracted arrays
      3. Call analyze_channels() to get composite signal
      4. Build DaySignals
    """
    # Pre-extract close/volume arrays for fast numpy slicing
    tf_closes = {}
    tf_vols = {}
    for tf in config_tfs:
        df = tf_data.get(tf)
        if df is None:
            continue
        tf_closes[tf] = df['close'].values
        tf_vols[tf] = df['volume'].values if 'volume' in df.columns else None

    signals = []

    for date in trading_dates:
        if date not in daily_df.index:
            continue
        didx = daily_df.index.get_loc(date)

        row = DaySignals(date=date)
        row.day_open = float(daily_df['open'].iloc[didx])
        row.day_high = float(daily_df['high'].iloc[didx])
        row.day_low = float(daily_df['low'].iloc[didx])
        row.day_close = float(daily_df['close'].iloc[didx])

        day_ch = channel_cache.get(date, {})

        # Collect analyze_channels inputs from config's TFs
        channels_by_tf = {}
        prices_by_tf = {}
        current_prices = {}
        volumes_by_tf = {}

        for tf in config_tfs:
            ch = day_ch.get(tf)
            if ch is None:
                continue
            pos = tf_pos[tf].get(date)
            if pos is None:
                continue

            channels_by_tf[tf] = ch
            closes = tf_closes.get(tf)
            if closes is not None:
                prices_by_tf[tf] = closes[:pos]
                current_prices[tf] = float(closes[pos - 1])
            vols = tf_vols.get(tf)
            if vols is not None:
                volumes_by_tf[tf] = vols[:pos]

        if not channels_by_tf:
            signals.append(row)
            continue

        # Run the full Channel Surfer analysis pipeline
        try:
            analysis = analyze_channels(
                channels_by_tf, prices_by_tf, current_prices,
                volumes_by_tf=volumes_by_tf or None,
            )
        except Exception:
            signals.append(row)
            continue

        # Extract signal fields into DaySignals
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

        # Store slim TF states for filter proxies
        tf_states_slim = {}
        for tf_name, st in (analysis.tf_states or {}).items():
            tf_states_slim[tf_name] = {
                'valid': getattr(st, 'valid', False),
                'momentum_direction': getattr(st, 'momentum_direction', 0.0),
                'momentum_is_turning': getattr(st, 'momentum_is_turning', False),
            }
        row.cs_tf_states = tf_states_slim

        signals.append(row)

    return signals


def _make_vix_combo(cascade_vix):
    """CS signal + VIX cooldown filter (comparable to DI base filter)."""
    def fn(day):
        if day.cs_action not in ('BUY', 'SELL'):
            return None
        if day.cs_confidence < MIN_SIGNAL_CONFIDENCE:
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
    return fn


def _make_raw_combo():
    """Raw CS signal — no additional filtering."""
    def fn(day):
        if day.cs_action in ('BUY', 'SELL') and day.cs_confidence >= MIN_SIGNAL_CONFIDENCE:
            s, t = _floor_stop_tp(day.cs_stop_pct, day.cs_tp_pct)
            return (day.cs_action, day.cs_confidence, s, t, 'CS')
        return None
    return fn


def phase2_sweep(channel_cache: dict, tf_data: dict, daily_df: pd.DataFrame,
                 vix_daily, configs: list, warmup: int = 260) -> dict:
    """For each TF config, generate signals via analyze_channels and simulate trades."""

    trading_dates = daily_df.index[warmup:]
    tf_pos = _precompute_tf_positions(tf_data, trading_dates)

    # Build VIX cooldown cascade
    cascade_vix = _build_filter_cascade(vix=True)
    try:
        if vix_daily is not None and len(vix_daily) > 0:
            cascade_vix.precompute_vix_cooldown(vix_daily)
    except Exception as e:
        print(f"  WARNING: VIX precompute failed: {e}")

    vix_fn = _make_vix_combo(cascade_vix)
    raw_fn = _make_raw_combo()

    results = {}

    for ci, (cfg_name, cfg_tfs) in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"  [{ci+1}/{len(configs)}] {cfg_name}")
        print(f"  TFs: {cfg_tfs}")
        print(f"{'='*60}")

        # Build DaySignals for this TF config
        t0 = time.time()
        signals = _build_signals_for_config(
            cfg_tfs, channel_cache, tf_data, daily_df, trading_dates, tf_pos,
        )
        sig_t = time.time() - t0

        n_buy = sum(1 for s in signals if s.cs_action == 'BUY')
        n_sell = sum(1 for s in signals if s.cs_action == 'SELL')
        n_hold = len(signals) - n_buy - n_sell
        print(f"  Signals: {n_buy} BUY + {n_sell} SELL + {n_hold} HOLD "
              f"({sig_t:.1f}s)")

        # Simulate trades (cd=0, trail_power=12 to match DI settings)
        t0 = time.time()
        trades_vix = simulate_trades(signals, vix_fn, cfg_name,
                                     cooldown=0, trail_power=12)
        trades_raw = simulate_trades(signals, raw_fn, cfg_name,
                                     cooldown=0, trail_power=12)
        sim_t = time.time() - t0

        print(f"  VIX filter: {len(trades_vix)} trades | "
              f"Raw CS: {len(trades_raw)} trades ({sim_t:.1f}s)")

        report_combo(f"{cfg_name} [VIX]", trades_vix)

        results[cfg_name] = {
            'tfs': cfg_tfs,
            'trades_vix': trades_vix,
            'trades_raw': trades_raw,
            'n_buy': n_buy,
            'n_sell': n_sell,
        }

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

# OOS cutoff: data before this was available during v6-v43 development
OOS_CUTOFF = pd.Timestamp('2025-10-01')


def _split_trades(trades):
    """Split trades into Train (<=2021), Test (2022-Sep2025), OOS (Oct2025+)."""
    train = [t for t in trades if t.entry_date.year <= TRAIN_END_YEAR]
    test = [t for t in trades
            if t.entry_date.year > TRAIN_END_YEAR and t.entry_date < OOS_CUTOFF]
    oos = [t for t in trades if t.entry_date >= OOS_CUTOFF]
    return train, test, oos


def _wr_pnl(trades):
    """Return (count, win_rate%, total_pnl) for a trade list."""
    if not trades:
        return 0, 0.0, 0.0
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    return n, wins / n * 100, sum(t.pnl for t in trades)


def _summary_row(name: str, n_tfs: int, trades: list) -> str:
    """Format one row of the summary table."""
    if not trades:
        return (f"{name:<16} {n_tfs:>3} {'0':>6} {'---':>6} {'---':>10} "
                f"{'---':>7} {'---':>7} {'---':>6} {'---':>6} {'---':>6}")
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    wr = wins / n * 100
    total = sum(t.pnl for t in trades)
    pnls = np.array([t.pnl for t in trades])
    avg_hd = max(np.mean([t.hold_days for t in trades]), 1)
    sharpe = (pnls.mean() / pnls.std() * np.sqrt(252 / avg_hd)
              ) if pnls.std() > 0 else 0
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    mdd = float(dd.max()) / CAPITAL * 100 if len(dd) > 0 else 0
    train, test, oos = _split_trades(trades)
    return (f"{name:<16} {n_tfs:>3} {n:>6} {wr:>5.1f}% ${total:>+9,.0f} "
            f"{sharpe:>7.2f} {mdd:>6.1f}% {len(train):>6} "
            f"{len(test):>6} {len(oos):>6}")


def print_summary(results: dict, configs: list):
    """Print comparison tables for VIX-filtered and raw results."""
    hdr = (f"{'Config':<16} {'TF':>3} {'Trades':>6} {'WR%':>6} {'PnL':>10} "
           f"{'Sharpe':>7} {'MaxDD%':>7} {'Train':>6} {'Test':>6} "
           f"{'OOS':>6}")

    for label, key in [('VIX Filter (cd=0, trail^12)', 'trades_vix'),
                       ('Raw CS (no filter, cd=0, trail^12)', 'trades_raw')]:
        print(f"\n{'='*90}")
        print(f"  TF EXPANSION SUMMARY — {label}")
        print(f"  Train: <=2021 | Test: 2022-Sep2025 | OOS: Oct2025+ (true holdout)")
        print(f"{'='*90}")
        print(hdr)
        print('-' * 90)
        for cfg_name, cfg_tfs in configs:
            r = results.get(cfg_name)
            if r:
                print(_summary_row(cfg_name, len(cfg_tfs), r[key]))

    # OOS detail breakdown
    print(f"\n{'='*80}")
    print(f"  OOS DETAIL (Oct 2025 - Feb 2026) — VIX Filter")
    print(f"{'='*80}")
    print(f"{'Config':<16} {'OOS#':>5} {'WR%':>6} {'OOS PnL':>10} "
          f"{'AvgPnL':>8} {'BigLoss':>9}")
    print('-' * 60)
    for cfg_name, cfg_tfs in configs:
        r = results.get(cfg_name)
        if not r:
            continue
        _, _, oos = _split_trades(r['trades_vix'])
        if not oos:
            print(f"{cfg_name:<16} {'0':>5} {'---':>6} {'---':>10} "
                  f"{'---':>8} {'---':>9}")
            continue
        n, wr, pnl = _wr_pnl(oos)
        avg = pnl / n
        worst = min(t.pnl for t in oos)
        print(f"{cfg_name:<16} {n:>5} {wr:>5.1f}% ${pnl:>+9,.0f} "
              f"${avg:>+7,.0f} ${worst:>+8,.0f}")

    # Signal count comparison
    print(f"\n{'='*60}")
    print(f"  SIGNAL COUNTS PER CONFIG")
    print(f"{'='*60}")
    print(f"{'Config':<16} {'TFs':>3} {'BUY':>6} {'SELL':>6} {'Total':>6}")
    print('-' * 40)
    for cfg_name, cfg_tfs in configs:
        r = results.get(cfg_name)
        if r:
            nb, ns = r['n_buy'], r['n_sell']
            print(f"{cfg_name:<16} {len(cfg_tfs):>3} {nb:>6} {ns:>6} {nb+ns:>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='v44: TF Expansion Experiment')
    parser.add_argument('--tsla', type=str, default=None,
                        help='Path to TSLAMin.txt (1-min data)')
    parser.add_argument('--start', type=str, default='2016-01-01')
    parser.add_argument('--end', type=str, default='2026-02-28')
    parser.add_argument('--phase2-only', action='store_true',
                        help='Skip Phase 1, load channels from cache')
    parser.add_argument('--phase1-only', action='store_true',
                        help='Only run Phase 1 (build channel cache)')
    parser.add_argument('--configs', type=str, default=None,
                        help='Comma-separated config indices (e.g. "0,1,5")')
    parser.add_argument('--warmup', type=int, default=260,
                        help='Warmup days before first signal (default 260)')
    args = parser.parse_args()

    # --- Find TSLAMin.txt ---
    if args.tsla is None:
        for candidate in ['data/TSLAMin.txt', '../data/TSLAMin.txt',
                          '../../data/TSLAMin.txt']:
            if Path(candidate).exists():
                args.tsla = candidate
                break
    if not args.tsla or not Path(args.tsla).exists():
        print("ERROR: TSLAMin.txt not found. Use --tsla <path>")
        sys.exit(1)

    # --- Select configs ---
    configs = TF_CONFIGS
    if args.configs:
        idxs = [int(x.strip()) for x in args.configs.split(',')]
        configs = [TF_CONFIGS[i] for i in idxs if i < len(TF_CONFIGS)]
        print(f"Running {len(configs)} configs: "
              f"{[c[0] for c in configs]}")

    # --- Load all 12 TFs ---
    print("Loading expanded TF data...")
    t0 = time.time()
    tf_data = load_expanded_tfs(args.tsla, args.start, args.end)
    daily_df = tf_data['daily']
    print(f"  Data loaded in {time.time()-t0:.1f}s\n")

    # --- Load SPY + VIX ---
    print("Loading SPY + VIX...")
    try:
        spy_daily = fetch_native_tf('SPY', 'daily', args.start, args.end)
        spy_daily = _norm_cols(spy_daily)
        spy_daily.index = pd.to_datetime(spy_daily.index).tz_localize(None)
    except Exception as e:
        print(f"  WARNING: SPY load failed: {e}")
        spy_daily = None

    try:
        vix_daily = fetch_native_tf('^VIX', 'daily', args.start, args.end)
        vix_daily = _norm_cols(vix_daily)
        vix_daily.index = pd.to_datetime(vix_daily.index).tz_localize(None)
    except Exception as e:
        print(f"  WARNING: VIX load failed: {e}")
        vix_daily = None

    # --- Phase 1: Pre-detect all channels ---
    if args.phase2_only and CHANNEL_CACHE_FILE.exists():
        print(f"\nLoading channel cache: {CHANNEL_CACHE_FILE}")
        with open(CHANNEL_CACHE_FILE, 'rb') as f:
            channel_cache = pickle.load(f)
        total_ch = sum(len(v) for v in channel_cache.values())
        print(f"  {total_ch:,} channels for {len(channel_cache)} days")
    else:
        channel_cache = phase1_detect_all_channels(
            tf_data, daily_df, warmup=args.warmup,
        )
        # Save cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving channel cache -> {CHANNEL_CACHE_FILE}")
        with open(CHANNEL_CACHE_FILE, 'wb') as f:
            pickle.dump(channel_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("  Saved.")

    if args.phase1_only:
        print("\nPhase 1 only — done.")
        return

    # --- Phase 2: Sweep all configs ---
    results = phase2_sweep(
        channel_cache, tf_data, daily_df, vix_daily, configs,
        warmup=args.warmup,
    )

    # --- Summary ---
    print_summary(results, configs)


if __name__ == '__main__':
    main()
