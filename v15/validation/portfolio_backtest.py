#!/usr/bin/env python3
"""
Multi-System Portfolio Backtest

Combines 4 proven TSLA trading systems and finds optimal capital allocation:
  - 5-min Channel Surfer  (~1,050 trades/yr, Sharpe=2.09)
  - 1h Channel Surfer     (~900 trades/yr, Sharpe=2.19)
  - 4h Channel Surfer     (~275 trades/yr, Sharpe=1.27)
  - Swing S1041            (~2-3 trades/yr, WR=100%)

Phases:
  1. Independent baseline  -- run each system on $100K, cache results
  2. Allocation grid search -- find optimal split (instant, arithmetic)
  3. Walk-forward           -- validate top allocations on rolling windows
  4. Overlap analysis       -- concurrent positions, peak exposure

Usage:
    python3 -m v15.validation.portfolio_backtest \
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt

    python3 -m v15.validation.portfolio_backtest --phase 2  # instant if cached
    python3 -m v15.validation.portfolio_backtest --force-rerun --phase 1
"""

import argparse
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CACHE_DIR = Path.home() / '.x14' / 'portfolio_cache'
SYSTEMS = ['5min', '1h', '4h', 'swing']


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Trade
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedTrade:
    """Normalized trade record across all systems."""
    system: str             # '5min', '1h', '4h', 'swing'
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl: float              # USD P&L at $100K capital
    direction: str           # 'long' or 'short'


@dataclass
class SystemYearResult:
    """Cached per-system, per-year result."""
    system: str
    year: int
    trades: List[UnifiedTrade]
    total_pnl: float
    total_trades: int
    win_rate: float
    daily_pnl: pd.Series    # DatetimeIndex -> daily P&L


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading -- single shared loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_data(tsla_path: str, spy_path: str):
    """Load 1-min data once, resample to all needed TFs."""
    from v15.core.historical_data import load_minute_data, resample_to_tf

    print("Loading TSLA 1-min data...")
    tsla_1min = load_minute_data(tsla_path)
    print(f"  TSLA 1-min: {len(tsla_1min):,} bars ({tsla_1min.index[0].date()} to {tsla_1min.index[-1].date()})")

    spy_1min = None
    if spy_path and os.path.isfile(spy_path):
        print("Loading SPY 1-min data...")
        spy_1min = load_minute_data(spy_path)
        print(f"  SPY  1-min: {len(spy_1min):,} bars")

    print("Resampling to all timeframes...")
    tsla_5min = resample_to_tf(tsla_1min, '5min')
    tsla_1h   = resample_to_tf(tsla_1min, '1h')
    tsla_4h   = resample_to_tf(tsla_1min, '4h')
    tsla_daily  = resample_to_tf(tsla_1min, '1D')
    tsla_weekly = resample_to_tf(tsla_1min, '1W')

    spy_5min = resample_to_tf(spy_1min, '5min') if spy_1min is not None else None
    spy_1h   = resample_to_tf(spy_1min, '1h')   if spy_1min is not None else None
    spy_4h   = resample_to_tf(spy_1min, '4h')   if spy_1min is not None else None
    spy_daily  = resample_to_tf(spy_1min, '1D')  if spy_1min is not None else None

    print(f"  5min: {len(tsla_5min):,}  1h: {len(tsla_1h):,}  4h: {len(tsla_4h):,}  "
          f"daily: {len(tsla_daily):,}  weekly: {len(tsla_weekly):,}")

    return {
        'tsla_1min': tsla_1min,
        'tsla_5min': tsla_5min,
        'tsla_1h': tsla_1h,
        'tsla_4h': tsla_4h,
        'tsla_daily': tsla_daily,
        'tsla_weekly': tsla_weekly,
        'spy_1min': spy_1min,
        'spy_5min': spy_5min,
        'spy_1h': spy_1h,
        'spy_4h': spy_4h,
        'spy_daily': spy_daily,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-System Year Runners
# ═══════════════════════════════════════════════════════════════════════════════

def _build_daily_pnl(trades: List[UnifiedTrade], year: int) -> pd.Series:
    """Build a daily P&L series from unified trades for a given year."""
    # Create date range for the year
    dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='B')  # business days
    daily = pd.Series(0.0, index=dates)

    for t in trades:
        # Attribute P&L to exit date
        exit_date = t.exit_time.normalize()
        if exit_date.tzinfo is not None:
            exit_date = exit_date.tz_localize(None)
        # Find nearest business day
        idx = daily.index.searchsorted(exit_date, side='right') - 1
        if 0 <= idx < len(daily):
            daily.iloc[idx] += t.pnl

    return daily


def _normalize_surfer_trades(trades, system: str, tsla_df) -> List[UnifiedTrade]:
    """Convert surfer_backtest.Trade list to UnifiedTrade list."""
    unified = []
    for t in trades:
        # surfer Trade has entry_time (str ISO) and entry_bar/exit_bar (int)
        if t.entry_time:
            entry_ts = pd.Timestamp(t.entry_time)
        elif t.entry_bar < len(tsla_df):
            entry_ts = tsla_df.index[t.entry_bar]
        else:
            continue

        if t.exit_bar < len(tsla_df):
            exit_ts = tsla_df.index[t.exit_bar]
        else:
            exit_ts = entry_ts + pd.Timedelta(hours=1)  # fallback

        direction = 'long' if t.direction == 'BUY' else 'short'
        unified.append(UnifiedTrade(
            system=system,
            entry_time=entry_ts,
            exit_time=exit_ts,
            pnl=t.pnl,
            direction=direction,
        ))
    return unified


def run_5min_year(all_data: dict, year: int, capital: float, vix_df) -> Optional[SystemYearResult]:
    """Run 5-min Channel Surfer for one year."""
    from v15.core.historical_data import prepare_year_data, prepare_backtest_data
    from v15.validation.combined_backtest import run_year_backtest

    # Build the full_data dict that prepare_year_data expects
    full_data = {
        'tsla_5min': all_data['tsla_5min'],
        'higher_tf_data': {
            '1h': all_data['tsla_1h'],
            '4h': all_data['tsla_4h'],
            'daily': all_data['tsla_daily'],
            'weekly': all_data['tsla_weekly'],
        },
        'spy_5min': all_data['spy_5min'],
        'tsla_1min': all_data['tsla_1min'],
    }

    year_data = prepare_year_data(full_data, year)
    if year_data is None:
        return None

    result = run_year_backtest(year_data, year, capital, vix_df)
    if result is None:
        return None

    metrics, trades, equity_curve = result
    unified = _normalize_surfer_trades(trades, '5min', all_data['tsla_5min'])
    daily_pnl = _build_daily_pnl(unified, year)

    return SystemYearResult(
        system='5min', year=year, trades=unified,
        total_pnl=metrics.total_pnl, total_trades=metrics.total_trades,
        win_rate=metrics.win_rate, daily_pnl=daily_pnl,
    )


def _prepare_medium_tf_year(all_data: dict, tf: str, year: int) -> Optional[dict]:
    """Prepare year data for 1h, 4h, or 1d backtest."""
    tsla_key = f'tsla_{tf}'
    spy_key = f'spy_{tf}'
    tsla_tf = all_data[tsla_key]
    spy_tf = all_data.get(spy_key)

    cutoff_start = pd.Timestamp(f'{year - 1}-10-01')
    cutoff_year_end = pd.Timestamp(f'{year}-12-31 23:59:59')

    def _slice(df, start, end):
        if df is None:
            return None
        tz = getattr(df.index, 'tz', None)
        s = start.tz_localize(tz) if tz is not None else start
        e = end.tz_localize(tz) if tz is not None else end
        return df.loc[(df.index >= s) & (df.index <= e)]

    tsla_slice = _slice(tsla_tf, cutoff_start, cutoff_year_end)
    if tsla_slice is None or len(tsla_slice) < 20:
        return None

    spy_slice = _slice(spy_tf, cutoff_start, cutoff_year_end) if spy_tf is not None else None

    # Build higher_tf_dict with all available context TFs.
    # Engine expects '1h' and 'daily' keys -- include even if one matches primary
    # (redundant but harmless channel detection, avoids degraded-analysis warning).
    higher_tf_dict = {}
    if 'tsla_1h' in all_data:
        hourly_slice = _slice(all_data['tsla_1h'], cutoff_start, cutoff_year_end)
        if hourly_slice is not None and len(hourly_slice) > 0:
            higher_tf_dict['1h'] = hourly_slice
    if tf == '1h' and '1h' not in higher_tf_dict:
        higher_tf_dict['1h'] = tsla_slice
    daily_slice = _slice(all_data['tsla_daily'], cutoff_start, cutoff_year_end)
    if daily_slice is not None and len(daily_slice) > 0:
        higher_tf_dict['daily'] = daily_slice
    weekly_slice = _slice(all_data['tsla_weekly'], cutoff_start, cutoff_year_end)
    if weekly_slice is not None and len(weekly_slice) > 0:
        higher_tf_dict['weekly'] = weekly_slice

    return {
        'tsla_tf': tsla_slice,
        'spy_tf': spy_slice,
        'higher_tf_dict': higher_tf_dict,
        'year_start': pd.Timestamp(f'{year}-01-01'),
        'year_end': cutoff_year_end,
    }


def run_medium_tf_year(all_data: dict, tf: str, year: int, capital: float,
                       vix_df) -> Optional[SystemYearResult]:
    """Run 1h or 4h Channel Surfer for one year."""
    from v15.validation.medium_tf_backtest import _run_year, TF_PARAMS

    year_data = _prepare_medium_tf_year(all_data, tf, year)
    if year_data is None:
        return None

    result = _run_year(year_data, tf, capital, vix_df)
    if result is None:
        return None

    metrics, trades, equity_curve = result
    tsla_key = f'tsla_{tf}'
    unified = _normalize_surfer_trades(trades, tf, all_data[tsla_key])
    daily_pnl = _build_daily_pnl(unified, year)

    return SystemYearResult(
        system=tf, year=year, trades=unified,
        total_pnl=metrics.total_pnl, total_trades=metrics.total_trades,
        win_rate=metrics.win_rate, daily_pnl=daily_pnl,
    )


def _normalize_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone info from index for swing engine compatibility."""
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').normalize().tz_localize(None)
    return df


def _resample_weekly_from_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV -> weekly (week ending Friday)."""
    return daily_df.resample('W-FRI').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()


def run_swing_year(all_data: dict, year: int, capital: float,
                   vix_df) -> Optional[SystemYearResult]:
    """Run Swing S1041 for one year."""
    import v15.validation.swing_backtest as swing_mod
    from v15.validation.swing_backtest import run_swing_backtest, sig_s1041_s993_or_s1034

    # Swing expects tz-naive daily data
    tsla_d = _normalize_tz(all_data['tsla_daily'])
    spy_d = _normalize_tz(all_data['spy_daily']) if all_data.get('spy_daily') is not None else None

    # VIX: also normalize
    vix_d = vix_df.copy()
    if vix_d.index.tz is not None:
        vix_d.index = vix_d.index.tz_convert('UTC').normalize().tz_localize(None)

    tsla_w = _resample_weekly_from_daily(tsla_d)
    spy_w = _resample_weekly_from_daily(spy_d) if spy_d is not None else None

    # Monkey-patch MAX_TRADE_USD to match capital allocation
    # Swing uses MAX_TRADE_USD directly for position sizing
    old_max = swing_mod.MAX_TRADE_USD
    swing_mod.MAX_TRADE_USD = capital
    try:
        result = run_swing_backtest(
            tsla=tsla_d, spy=spy_d, vix=vix_d,
            tsla_weekly=tsla_w, spy_weekly=spy_w,
            signal_fn=sig_s1041_s993_or_s1034,
            signal_name='S1041',
            max_hold_days=10,
            stop_pct=0.05,
            start_year=year, end_year=year,
        )
    finally:
        swing_mod.MAX_TRADE_USD = old_max

    # Convert swing Trade -> UnifiedTrade
    unified = []
    for t in result.trades:
        unified.append(UnifiedTrade(
            system='swing',
            entry_time=t.entry_date,
            exit_time=t.exit_date,
            pnl=t.pnl_usd,
            direction='long' if t.direction == 1 else 'short',
        ))

    daily_pnl = _build_daily_pnl(unified, year)

    return SystemYearResult(
        system='swing', year=year, trades=unified,
        total_pnl=result.total_pnl, total_trades=result.n_trades,
        win_rate=result.win_rate, daily_pnl=daily_pnl,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 -- Independent Baseline
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_path(system: str) -> Path:
    return CACHE_DIR / f'{system}_results.pkl'


def _load_cache(system: str) -> Optional[Dict[int, SystemYearResult]]:
    path = _cache_path(system)
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def _save_cache(system: str, results: Dict[int, SystemYearResult]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(system), 'wb') as f:
        pickle.dump(results, f)


def _run_system_all_years(system: str, all_data: dict, years: List[int],
                          capital: float, vix_df) -> Dict[int, SystemYearResult]:
    """Run one system across all years. Used as multiprocessing target."""
    sys_results = {}
    for year in years:
        try:
            if system == '5min':
                result = run_5min_year(all_data, year, capital, vix_df)
            elif system in ('1h', '4h'):
                result = run_medium_tf_year(all_data, system, year, capital, vix_df)
            elif system == 'swing':
                result = run_swing_year(all_data, year, capital, vix_df)
            else:
                result = None
        except Exception as e:
            print(f"    [{system}] {year}: ERROR - {e}")
            result = None

        if result is None:
            print(f"    [{system}] {year}: no data/skipped")
            continue

        sys_results[year] = result
        print(f"    [{system}] {year}: ${result.total_pnl:>10,.0f}  "
              f"{result.total_trades:>4} trades  WR={result.win_rate:.1%}")

    return sys_results


def run_phase1(all_data: dict, years: List[int], capital: float,
               vix_df, force_rerun: bool = False,
               parallel: bool = True) -> Dict[str, Dict[int, SystemYearResult]]:
    """Run all 4 systems across all years. Parallel by default. Cache to disk."""
    print(f"\n{'='*80}")
    print(f"PHASE 1 -- INDEPENDENT BASELINE (capital=${capital:,.0f})")
    print(f"  Systems: {', '.join(SYSTEMS)}")
    print(f"  Years: {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"  Parallel: {parallel}")
    print(f"{'='*80}")

    all_results = {}  # system -> {year: SystemYearResult}
    systems_to_run = []

    # Check caches first
    for system in SYSTEMS:
        if not force_rerun:
            cached = _load_cache(system)
            if cached is not None:
                cached_years = set(cached.keys())
                needed_years = set(years)
                if needed_years.issubset(cached_years):
                    print(f"\n  [{system}] Loaded from cache ({len(cached)} years)")
                    all_results[system] = {y: cached[y] for y in years if y in cached}
                    _print_system_summary(all_results[system], system)
                    continue
        systems_to_run.append(system)

    if not systems_to_run:
        _print_correlation_matrix(all_results, years)
        return all_results

    # Run uncached systems
    if parallel and len(systems_to_run) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"\n  Launching {len(systems_to_run)} systems in parallel: {systems_to_run}")
        t0 = time.time()

        # ProcessPoolExecutor gives each system its own process
        # -- swing's monkey-patch of MAX_TRADE_USD is safe (separate process)
        with ProcessPoolExecutor(max_workers=len(systems_to_run)) as pool:
            futures = {
                pool.submit(_run_system_all_years, system, all_data, years, capital, vix_df): system
                for system in systems_to_run
            }
            for future in as_completed(futures):
                system = futures[future]
                try:
                    sys_results = future.result()
                except Exception as e:
                    print(f"\n  [{system}] FAILED: {e}")
                    sys_results = {}

                all_results[system] = sys_results
                _save_cache(system, sys_results)
                elapsed = time.time() - t0
                print(f"\n  [{system}] Done ({len(sys_results)} years, {elapsed:.1f}s elapsed)")
                _print_system_summary(sys_results, system)

        print(f"\n  All systems complete in {time.time() - t0:.1f}s")
    else:
        # Sequential fallback
        for system in systems_to_run:
            print(f"\n  [{system}] Running {len(years)} years...")
            t_sys = time.time()
            sys_results = _run_system_all_years(system, all_data, years, capital, vix_df)

            all_results[system] = sys_results
            elapsed_sys = time.time() - t_sys
            print(f"    [{system}] Total: {elapsed_sys:.1f}s")

            _save_cache(system, sys_results)
            _print_system_summary(sys_results, system)

    # Correlation matrix
    _print_correlation_matrix(all_results, years)
    return all_results


def _print_system_summary(results: Dict[int, 'SystemYearResult'], system: str):
    """Print aggregate summary for one system."""
    if not results:
        return
    total_pnl = sum(r.total_pnl for r in results.values())
    total_trades = sum(r.total_trades for r in results.values())
    total_wins = sum(int(r.win_rate * r.total_trades + 0.5) for r in results.values())
    wr = total_wins / max(total_trades, 1)
    yr_pnls = [r.total_pnl for r in results.values()]
    sharpe = float(np.mean(yr_pnls) / np.std(yr_pnls)) if len(yr_pnls) >= 2 and np.std(yr_pnls) > 0 else 0.0
    trades_per_yr = total_trades / max(len(results), 1)

    print(f"    Summary: ${total_pnl:,.0f} total  {total_trades:,} trades  WR={wr:.1%}  "
          f"Sharpe={sharpe:.2f}  Trd/yr={trades_per_yr:.0f}")


def _print_correlation_matrix(all_results: Dict[str, Dict[int, 'SystemYearResult']],
                               years: List[int]):
    """Print 4x4 correlation matrix of annual P&L across systems."""
    print(f"\n  {'='*60}")
    print("  ANNUAL P&L CORRELATION MATRIX")
    print(f"  {'='*60}")

    # Build annual P&L series per system
    pnl_df = pd.DataFrame(index=years)
    for system in SYSTEMS:
        results = all_results.get(system, {})
        pnl_df[system] = [results[y].total_pnl if y in results else 0.0 for y in years]

    # Print annual P&L table
    print(f"\n  {'Year':<6}", end='')
    for s in SYSTEMS:
        print(f"  {s:>12}", end='')
    print()
    print(f"  {'-'*6}", end='')
    for _ in SYSTEMS:
        print(f"  {'-'*12}", end='')
    print()

    for year in years:
        print(f"  {year:<6}", end='')
        for s in SYSTEMS:
            pnl = pnl_df.loc[year, s]
            print(f"  ${pnl:>10,.0f}", end='')
        print()

    # Correlation
    corr = pnl_df.corr()
    print(f"\n  {'':8}", end='')
    for s in SYSTEMS:
        print(f"  {s:>8}", end='')
    print()

    for s1 in SYSTEMS:
        print(f"  {s1:8}", end='')
        for s2 in SYSTEMS:
            c = corr.loc[s1, s2]
            print(f"  {c:>8.3f}", end='')
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 -- Allocation Grid Search
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_allocations(step: float = 0.10) -> List[Tuple[float, ...]]:
    """Generate all 4-way allocations summing to 1.0 in `step` increments."""
    steps = int(round(1.0 / step))
    combos = []
    for a in range(steps + 1):
        for b in range(steps + 1 - a):
            for c in range(steps + 1 - a - b):
                d = steps - a - b - c
                combos.append((a * step, b * step, c * step, d * step))
    return combos


@dataclass
class AllocationResult:
    """Result for one allocation combo."""
    weights: Tuple[float, ...]  # (5min, 1h, 4h, swing)
    total_pnl: float
    annual_sharpe: float
    max_drawdown_pct: float
    yr_pnls: List[float]


def run_phase2(all_results: Dict[str, Dict[int, 'SystemYearResult']],
               years: List[int], total_capital: float,
               step: float = 0.10, top_n: int = 10) -> List[AllocationResult]:
    """Grid search over allocation splits. Pure arithmetic, instant."""
    print(f"\n{'='*80}")
    print(f"PHASE 2 -- ALLOCATION GRID SEARCH (total=${total_capital:,.0f}, step={step:.0%})")
    print(f"{'='*80}")

    allocations = _generate_allocations(step)
    print(f"  Testing {len(allocations)} allocation combos...")

    # Precompute per-system combined daily P&L (concatenated across years, at $100K base)
    # Then for each allocation, just multiply by weight and sum -- avoids redundant .add() calls
    sys_daily = {}   # system -> single pd.Series across all years
    sys_yr_pnl = {}  # system -> {year: float}
    for system in SYSTEMS:
        parts = []
        yr_pnl = {}
        for year in years:
            sys_yr = all_results.get(system, {}).get(year)
            if sys_yr is None:
                yr_pnl[year] = 0.0
                continue
            yr_pnl[year] = sys_yr.total_pnl
            parts.append(sys_yr.daily_pnl)
        sys_yr_pnl[system] = yr_pnl
        if parts:
            sys_daily[system] = pd.concat(parts).sort_index()
        else:
            sys_daily[system] = pd.Series(dtype=float)

    # Build a common date index across all systems
    all_dates = set()
    for s in sys_daily.values():
        all_dates.update(s.index)
    all_dates = sorted(all_dates)
    common_idx = pd.DatetimeIndex(all_dates)

    # Reindex each system to common dates
    sys_daily_arr = {}  # system -> np.array aligned to common_idx
    for system in SYSTEMS:
        sys_daily_arr[system] = sys_daily[system].reindex(common_idx, fill_value=0.0).values

    results = []
    for weights in allocations:
        # Combined annual P&L
        yr_pnls = []
        for year in years:
            year_pnl = 0.0
            for i, system in enumerate(SYSTEMS):
                if weights[i] == 0:
                    continue
                scale = (weights[i] * total_capital) / 100_000.0
                year_pnl += sys_yr_pnl[system].get(year, 0.0) * scale
            yr_pnls.append(year_pnl)

        if len(yr_pnls) < 2:
            continue

        total_pnl = sum(yr_pnls)
        mean_pnl = np.mean(yr_pnls)
        std_pnl = np.std(yr_pnls)
        sharpe = float(mean_pnl / std_pnl) if std_pnl > 0 else 0.0

        # Max drawdown from combined daily equity curve (numpy, fast)
        daily_arr = np.zeros(len(common_idx))
        for i, system in enumerate(SYSTEMS):
            if weights[i] == 0:
                continue
            scale = (weights[i] * total_capital) / 100_000.0
            daily_arr += sys_daily_arr[system] * scale

        equity = np.cumsum(daily_arr)
        if len(equity) > 0:
            running_max = np.maximum.accumulate(equity)
            drawdown = equity - running_max
            max_dd_abs = drawdown.min()
            dd_idx = np.argmin(drawdown)
            peak_at_dd = running_max[dd_idx]
            max_dd_pct = max_dd_abs / max(peak_at_dd + total_capital, 1.0)
        else:
            max_dd_pct = 0.0

        results.append(AllocationResult(
            weights=weights,
            total_pnl=total_pnl,
            annual_sharpe=sharpe,
            max_drawdown_pct=max_dd_pct,
            yr_pnls=yr_pnls,
        ))

    # Sort by Sharpe descending
    results.sort(key=lambda r: r.annual_sharpe, reverse=True)

    # Print top N
    print(f"\n  TOP {top_n} ALLOCATIONS BY SHARPE")
    print(f"  {'='*100}")
    header = (f"  {'Rank':<5} {'5min':>5} {'1h':>5} {'4h':>5} {'Swing':>5}"
              f"  {'Total P&L':>12} {'Sharpe':>7} {'MaxDD':>7} {'Avg/Yr':>12} {'Worst Yr':>12}")
    print(header)
    print(f"  {'-'*100}")

    for rank, r in enumerate(results[:top_n], 1):
        avg_yr = np.mean(r.yr_pnls) if r.yr_pnls else 0
        worst_yr = min(r.yr_pnls) if r.yr_pnls else 0
        print(f"  {rank:<5} {r.weights[0]:>4.0%} {r.weights[1]:>4.0%} "
              f"{r.weights[2]:>4.0%} {r.weights[3]:>4.0%}"
              f"  ${r.total_pnl:>10,.0f} {r.annual_sharpe:>7.2f} "
              f"{r.max_drawdown_pct:>7.1%} ${avg_yr:>10,.0f} ${worst_yr:>10,.0f}")

    # Also print equal-weight for reference
    eq_idx = None
    n_sys = len(SYSTEMS)
    eq_w = round(1.0 / n_sys, 2)
    for i, r in enumerate(results):
        if all(abs(w - eq_w) < 0.01 for w in r.weights):
            eq_idx = i
            break
    if eq_idx is not None:
        r = results[eq_idx]
        print(f"\n  Equal-weight (rank #{eq_idx + 1}): "
              f"${r.total_pnl:,.0f}  Sharpe={r.annual_sharpe:.2f}  MaxDD={r.max_drawdown_pct:.1%}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 -- Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase3(all_results: Dict[str, Dict[int, 'SystemYearResult']],
               phase2_results: List['AllocationResult'],
               total_capital: float, top_n: int = 3,
               years: Optional[List[int]] = None):
    """Walk-forward validation for top allocations. Rolling IS 5yr -> OOS 1yr."""
    print(f"\n{'='*80}")
    print(f"PHASE 3 -- WALK-FORWARD VALIDATION (top {top_n} allocations)")
    print(f"{'='*80}")

    # Build windows from available years: IS 5yr -> OOS 1yr
    if years is None:
        # Collect all years with data across any system
        all_years = set()
        for sys_results in all_results.values():
            all_years.update(sys_results.keys())
        years = sorted(all_years)

    min_yr, max_yr = years[0], years[-1]
    windows = [
        (list(range(y, y + 5)), y + 5)
        for y in range(min_yr, max_yr - 4)
        if y + 5 <= max_yr
    ]
    print(f"  Windows: {len(windows)} (IS 5yr -> OOS 1yr)")

    for rank, alloc in enumerate(phase2_results[:top_n], 1):
        weights = alloc.weights
        w_str = '/'.join(f'{w:.0%}' for w in weights)
        print(f"\n  Allocation #{rank}: [{w_str}] (5min/1h/4h/swing)")
        print(f"  {'-'*70}")
        print(f"  {'Window':<18} {'IS P&L':>12} {'IS Sharpe':>10} {'OOS P&L':>12} {'OOS/IS_avg':>10} {'Win?':>5}")
        print(f"  {'-'*70}")

        wins = 0
        total_oos = 0.0

        for is_years, oos_year in windows:
            # IS metrics
            is_pnls = []
            for yr in is_years:
                yr_pnl = 0.0
                for i, system in enumerate(SYSTEMS):
                    if weights[i] == 0:
                        continue
                    sys_yr = all_results.get(system, {}).get(yr)
                    if sys_yr is None:
                        continue
                    scale = (weights[i] * total_capital) / 100_000.0
                    yr_pnl += sys_yr.total_pnl * scale
                is_pnls.append(yr_pnl)

            is_total = sum(is_pnls)
            is_avg = np.mean(is_pnls) if is_pnls else 0
            is_std = np.std(is_pnls) if is_pnls else 1
            is_sharpe = float(is_avg / is_std) if is_std > 0 else 0

            # OOS
            oos_pnl = 0.0
            for i, system in enumerate(SYSTEMS):
                if weights[i] == 0:
                    continue
                sys_yr = all_results.get(system, {}).get(oos_year)
                if sys_yr is None:
                    continue
                scale = (weights[i] * total_capital) / 100_000.0
                oos_pnl += sys_yr.total_pnl * scale

            ratio = oos_pnl / is_avg if is_avg > 0 else 0
            win = oos_pnl > 0
            if win:
                wins += 1
            total_oos += oos_pnl

            label = f"{is_years[0]}-{is_years[-1]}->{oos_year}"
            win_str = "WIN" if win else "LOSS"
            print(f"  {label:<18} ${is_total:>10,.0f} {is_sharpe:>10.2f} "
                  f"${oos_pnl:>10,.0f} {ratio:>10.2f}x {'':>1}{win_str}")

        avg_oos = total_oos / max(len(windows), 1)
        print(f"  {'-'*70}")
        print(f"  Result: {wins}/{len(windows)} wins, total OOS=${total_oos:,.0f}, "
              f"avg OOS=${avg_oos:,.0f}/yr")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 -- Overlap Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase4(all_results: Dict[str, Dict[int, 'SystemYearResult']],
               winning_weights: Tuple[float, ...],
               years: List[int], total_capital: float):
    """Analyze concurrent positions across systems."""
    print(f"\n{'='*80}")
    w_str = '/'.join(f'{w:.0%}' for w in winning_weights)
    print(f"PHASE 4 -- OVERLAP ANALYSIS (allocation: [{w_str}])")
    print(f"{'='*80}")

    # Collect all trades across all systems and years
    all_trades = []
    for i, system in enumerate(SYSTEMS):
        if winning_weights[i] == 0:
            continue
        scale = (winning_weights[i] * total_capital) / 100_000.0
        for year in years:
            sys_yr = all_results.get(system, {}).get(year)
            if sys_yr is None:
                continue
            for t in sys_yr.trades:
                # Strip tz from all timestamps for consistent comparison
                entry = t.entry_time.tz_localize(None) if t.entry_time.tzinfo is not None else t.entry_time
                exit_t = t.exit_time.tz_localize(None) if t.exit_time.tzinfo is not None else t.exit_time
                all_trades.append(UnifiedTrade(
                    system=t.system,
                    entry_time=entry,
                    exit_time=exit_t,
                    pnl=t.pnl * scale,
                    direction=t.direction,
                ))

    if not all_trades:
        print("  No trades to analyze.")
        return

    print(f"  Total trades across all systems: {len(all_trades):,}")

    # Sort by entry time
    all_trades.sort(key=lambda t: t.entry_time)

    # Scan for overlaps using event-based approach
    events = []
    for t in all_trades:
        entry = t.entry_time
        exit_t = t.exit_time
        events.append((entry, +1, t))
        events.append((exit_t, -1, t))

    events.sort(key=lambda e: (e[0], e[1]))  # sort by time, exits before entries at same time

    # Walk through events to find concurrent positions
    active = 0
    max_concurrent = 0
    overlap_minutes = 0
    last_time = None
    overlap_pnl = 0.0
    non_overlap_pnl = 0.0

    # Track which systems are active
    active_systems = set()
    overlap_trades = set()

    concurrent_histogram = {}  # n_concurrent -> count of transitions

    for ts, delta, trade in events:
        if last_time is not None and active >= 2:
            overlap_minutes += (ts - last_time).total_seconds() / 60

        active += delta
        max_concurrent = max(max_concurrent, active)

        if delta == +1:
            active_systems.add(trade.system)
        # Track overlap trades
        if active >= 2 and delta == +1:
            overlap_trades.add(id(trade))

        concurrent_histogram[active] = concurrent_histogram.get(active, 0) + 1
        last_time = ts

    # Classify trades as overlap or non-overlap for P&L
    for t in all_trades:
        if id(t) in overlap_trades:
            overlap_pnl += t.pnl
        else:
            non_overlap_pnl += t.pnl

    # Per-system trade counts
    sys_counts = {}
    sys_pnl = {}
    for t in all_trades:
        sys_counts[t.system] = sys_counts.get(t.system, 0) + 1
        sys_pnl[t.system] = sys_pnl.get(t.system, 0) + t.pnl

    print(f"\n  Per-System Breakdown:")
    print(f"  {'System':<8} {'Trades':>8} {'P&L':>14} {'Avg P&L':>12}")
    print(f"  {'-'*44}")
    for s in SYSTEMS:
        if winning_weights[SYSTEMS.index(s)] == 0:
            continue
        n = sys_counts.get(s, 0)
        pnl = sys_pnl.get(s, 0)
        avg = pnl / max(n, 1)
        print(f"  {s:<8} {n:>8,} ${pnl:>12,.0f} ${avg:>10,.0f}")

    total_pnl = sum(sys_pnl.values())
    print(f"  {'TOTAL':<8} {len(all_trades):>8,} ${total_pnl:>12,.0f} ${total_pnl / max(len(all_trades), 1):>10,.0f}")

    print(f"\n  Overlap Statistics:")
    print(f"    Max concurrent positions: {max_concurrent}")
    print(f"    Overlap time: {overlap_minutes / 60:,.1f} hours ({overlap_minutes / (len(years) * 252 * 6.5 * 60) * 100:.1f}% of market hours)")
    print(f"    Overlap P&L: ${overlap_pnl:,.0f}  ({overlap_pnl / max(total_pnl, 1) * 100:.1f}% of total)")
    print(f"    Non-overlap P&L: ${non_overlap_pnl:,.0f}")

    # Notional exposure at peak
    # Each system uses weight * total_capital for position sizing
    # Max leverage is 4x in surfer, but actual exposure depends on trade count
    active_weights = [w for w in winning_weights if w > 0]
    peak_notional = sum(w * total_capital * 4.0 for w in active_weights)  # theoretical max
    print(f"\n  Peak Notional Exposure (theoretical max, 4x leverage all systems): ${peak_notional:,.0f}")

    # Concurrent position histogram
    print(f"\n  Concurrent Position Histogram:")
    for n_conc in sorted(concurrent_histogram.keys()):
        if n_conc > 0:
            print(f"    {n_conc} positions active: {concurrent_histogram[n_conc]:,} events")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Multi-System Portfolio Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--tsla', type=str, default='data/TSLAMin.txt',
                        help='Path to TSLAMin.txt (default: data/TSLAMin.txt)')
    parser.add_argument('--spy', type=str, default='data/SPYMin.txt',
                        help='Path to SPYMin.txt (default: data/SPYMin.txt)')
    parser.add_argument('--years', type=str, default='2015-2025',
                        help='Year range (default: 2015-2025)')
    parser.add_argument('--total-capital', type=float, default=1_000_000.0,
                        help='Total portfolio capital for allocation (default: $1M)')
    parser.add_argument('--step', type=float, default=0.10,
                        help='Allocation grid step size (default: 0.10)')
    parser.add_argument('--phase', type=str, default='all',
                        help='Phase to run: all, 1, 2, 3, 4 (default: all)')
    parser.add_argument('--force-rerun', action='store_true',
                        help='Ignore pickle cache, re-run all backtests')
    parser.add_argument('--top-n', type=int, default=3,
                        help='Top N allocations for walk-forward (default: 3)')
    parser.add_argument('--baseline-capital', type=float, default=100_000.0,
                        help='Capital per system in Phase 1 (default: $100K)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Run Phase 1 systems sequentially instead of in parallel')

    args = parser.parse_args()

    # Parse years
    if '-' in args.years:
        start, end = args.years.split('-')
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(args.years)]

    phases = args.phase.lower()

    print(f"\n{'#'*80}")
    print(f"# MULTI-SYSTEM PORTFOLIO BACKTEST")
    print(f"# Systems: 5min / 1h / 4h / Swing S1041")
    print(f"# Years: {years[0]}-{years[-1]} | Capital: ${args.total_capital:,.0f}")
    print(f"# Phase: {phases}")
    print(f"{'#'*80}")

    # ── Load data (needed for Phase 1 if not cached) ──
    all_data = None
    need_data = (phases in ('all', '1')) and (args.force_rerun or not _all_cached(years))

    if need_data:
        t0 = time.time()
        all_data = load_all_data(args.tsla, args.spy)
        print(f"  Data loaded in {time.time() - t0:.1f}s")

    # ── Load VIX ──
    vix_df = None
    if need_data:
        try:
            from v15.validation.vix_loader import load_vix_daily
            vix_df = load_vix_daily(start='2014-01-01', end='2025-12-31')
        except Exception as e:
            print(f"  VIX load failed: {e}")

    # ── Phase 1 ──
    all_results = None
    if phases in ('all', '1'):
        all_results = run_phase1(all_data, years, args.baseline_capital, vix_df,
                                  force_rerun=args.force_rerun,
                                  parallel=not args.no_parallel)

    # For phases 2-4, load from cache if not already in memory
    if all_results is None:
        all_results = {}
        for system in SYSTEMS:
            cached = _load_cache(system)
            if cached is not None:
                all_results[system] = {y: cached[y] for y in years if y in cached}
            else:
                print(f"  WARNING: No cache for {system}. Run phase 1 first.")

    # ── Phase 2 ──
    phase2_results = None
    if phases in ('all', '2'):
        phase2_results = run_phase2(all_results, years, args.total_capital,
                                     step=args.step, top_n=10)

    # ── Phase 3 ──
    if phases in ('all', '3'):
        if phase2_results is None:
            # Re-run phase 2 to get rankings
            phase2_results = run_phase2(all_results, years, args.total_capital,
                                         step=args.step, top_n=10)
        run_phase3(all_results, phase2_results, args.total_capital,
                   top_n=args.top_n, years=years)

    # ── Phase 4 ──
    if phases in ('all', '4'):
        if phase2_results is None:
            phase2_results = run_phase2(all_results, years, args.total_capital,
                                         step=args.step, top_n=10)
        # Use top allocation
        if phase2_results:
            winning = phase2_results[0].weights
        else:
            # Equal weight fallback
            winning = tuple(1.0 / len(SYSTEMS) for _ in SYSTEMS)
        run_phase4(all_results, winning, years, args.total_capital)

    # ── Final Summary ──
    print(f"\n{'='*80}")
    print("PORTFOLIO BACKTEST COMPLETE")
    print(f"{'='*80}")


def _all_cached(years: List[int]) -> bool:
    """Check if all systems have cached results for all years."""
    for system in SYSTEMS:
        cached = _load_cache(system)
        if cached is None:
            return False
        if not set(years).issubset(set(cached.keys())):
            return False
    return True


if __name__ == '__main__':
    main()
