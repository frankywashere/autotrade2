#!/usr/bin/env python3
"""
Forward Simulation: runs all 5 systems on recent yfinance data.
Reports every trade per system, then computes portfolio P&L for multiple allocations.

Usage:
    python -m v15.validation.forward_sim --start 2026-02-23 --end 2026-02-26
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SYSTEMS = ['5min', '1h', '4h', '1d', 'swing']

ALLOCATIONS = {
    'Data-driven (0/20/0/80/0)':   (0.00, 0.20, 0.00, 0.80, 0.00),
    'Conservative (0/20/0/65/15)':  (0.00, 0.20, 0.00, 0.65, 0.15),
    'Old rec (35/55/0/10/0)':       (0.35, 0.55, 0.00, 0.10, 0.00),
    '4h included (0/25/25/40/10)':  (0.00, 0.25, 0.25, 0.40, 0.10),
    'Equal (20/20/20/20/20)':       (0.20, 0.20, 0.20, 0.20, 0.20),
}


@dataclass
class SimTrade:
    system: str
    entry_time: str
    exit_time: str
    direction: str
    pnl: float


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _clean_yf(df):
    """Normalize yfinance DataFrame: flatten MultiIndex columns, lowercase."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    # Drop adj close if present (duplicates close in auto_adjust mode)
    if 'adj close' in df.columns:
        df = df.drop(columns=['adj close'])
    return df


def download_data():
    """Download TSLA, SPY, VIX at multiple TFs from yfinance."""
    import yfinance as yf

    print("\nDownloading data from yfinance...")

    # 5min (max 60 days -- plenty for Feb 23-26 + lookback)
    tsla_5min = _clean_yf(yf.download('TSLA', period='60d', interval='5m', progress=False))
    spy_5min  = _clean_yf(yf.download('SPY',  period='60d', interval='5m', progress=False))
    print(f"  TSLA 5min : {len(tsla_5min):,} bars  "
          f"({tsla_5min.index[0]} to {tsla_5min.index[-1]})")

    # 1h (max 730 days)
    tsla_1h = _clean_yf(yf.download('TSLA', period='730d', interval='1h', progress=False))
    spy_1h  = _clean_yf(yf.download('SPY',  period='730d', interval='1h', progress=False))
    print(f"  TSLA 1h   : {len(tsla_1h):,} bars")

    # 4h: resample from 1h
    tsla_4h = tsla_1h.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    spy_4h = spy_1h.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    print(f"  TSLA 4h   : {len(tsla_4h):,} bars (resampled)")

    # Daily (2 years)
    tsla_daily = _clean_yf(yf.download('TSLA', period='2y', interval='1d', progress=False))
    spy_daily  = _clean_yf(yf.download('SPY',  period='2y', interval='1d', progress=False))
    print(f"  TSLA daily: {len(tsla_daily):,} bars")

    # Weekly (5 years -- channel detection needs 30+ bars)
    tsla_weekly = _clean_yf(yf.download('TSLA', period='5y', interval='1wk', progress=False))
    print(f"  TSLA wkly : {len(tsla_weekly):,} bars")

    # VIX
    vix_daily = _clean_yf(yf.download('^VIX', period='2y', interval='1d', progress=False))
    print(f"  VIX       : {len(vix_daily):,} bars")

    return {
        'tsla_5min': tsla_5min, 'spy_5min': spy_5min,
        'tsla_1h': tsla_1h,     'spy_1h': spy_1h,
        'tsla_4h': tsla_4h,     'spy_4h': spy_4h,
        'tsla_daily': tsla_daily, 'spy_daily': spy_daily,
        'tsla_weekly': tsla_weekly,
        'vix_daily': vix_daily,
    }


# ---------------------------------------------------------------------------
# Per-system runners
# ---------------------------------------------------------------------------

def _ts_naive(ts):
    """Strip timezone for comparison."""
    if ts is not None and hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def _filter_sim_window(trades, tsla_df, system, sim_start, sim_end):
    """Convert engine Trade objects to SimTrade, keep only those in window."""
    sim_s = pd.Timestamp(sim_start)
    sim_e = pd.Timestamp(sim_end) + pd.Timedelta(hours=23, minutes=59)

    out = []
    for t in trades:
        if not t.entry_time:
            continue
        entry_ts = _ts_naive(pd.Timestamp(t.entry_time))
        if not (sim_s <= entry_ts <= sim_e):
            continue

        exit_time = ''
        if t.exit_bar < len(tsla_df):
            exit_time = str(tsla_df.index[t.exit_bar])

        out.append(SimTrade(
            system=system,
            entry_time=t.entry_time,
            exit_time=exit_time,
            direction='LONG' if t.direction == 'BUY' else 'SHORT',
            pnl=t.pnl,
        ))
    return out


def run_surfer(data, system, capital, sim_start, sim_end):
    """Run channel surfer backtest for one TF. Returns list of SimTrade."""
    from v15.core.surfer_backtest import run_backtest
    from v15.validation.medium_tf_backtest import TF_PARAMS

    tf_map = {
        '5min': ('tsla_5min', 'spy_5min'),
        '1h':   ('tsla_1h',   'spy_1h'),
        '4h':   ('tsla_4h',   'spy_4h'),
        '1d':   ('tsla_daily', 'spy_daily'),
    }
    tsla_key, spy_key = tf_map[system]
    tsla_df = data[tsla_key]
    spy_df  = data.get(spy_key)

    # Higher TF context -- only TFs ABOVE the primary
    if system == '5min':
        higher_tf = {
            '1h': data['tsla_1h'],
            '4h': data['tsla_4h'],
            'daily': data['tsla_daily'],
            'weekly': data['tsla_weekly'],
        }
        params = dict(eval_interval=3, max_hold_bars=60, bounce_cap=12)
    elif system == '1h':
        higher_tf = {
            'daily': data['tsla_daily'],
            'weekly': data['tsla_weekly'],
        }
        p = TF_PARAMS['1h']
        params = dict(eval_interval=p['eval_interval'],
                      max_hold_bars=p['max_hold_bars'], bounce_cap=p['bounce_cap'])
    elif system == '4h':
        higher_tf = {
            'daily': data['tsla_daily'],
            'weekly': data['tsla_weekly'],
        }
        p = TF_PARAMS['4h']
        params = dict(eval_interval=p['eval_interval'],
                      max_hold_bars=p['max_hold_bars'], bounce_cap=p['bounce_cap'])
    elif system == '1d':
        higher_tf = {
            'weekly': data['tsla_weekly'],
        }
        p = TF_PARAMS['1d']
        params = dict(eval_interval=p['eval_interval'],
                      max_hold_bars=p['max_hold_bars'], bounce_cap=p['bounce_cap'])

    result = run_backtest(
        days=0,
        eval_interval=params['eval_interval'],
        max_hold_bars=params['max_hold_bars'],
        position_size=capital / 10,
        min_confidence=0.45,
        use_multi_tf=True,
        tsla_df=tsla_df,
        higher_tf_dict=higher_tf,
        spy_df_input=spy_df,
        vix_df_input=data['vix_daily'],
        realistic=True,
        slippage_bps=3.0,
        commission_per_share=0.005,
        max_leverage=4.0,
        bounce_cap=params['bounce_cap'],
        max_trade_usd=1_000_000,
    )

    metrics, trades, equity_curve = result
    return _filter_sim_window(trades, tsla_df, system, sim_start, sim_end)


def run_swing(data, capital, sim_start, sim_end):
    """Run swing S1041. Returns list of SimTrade."""
    import v15.validation.swing_backtest as swing_mod
    from v15.validation.swing_backtest import (
        run_swing_backtest, sig_s1041_s993_or_s1034,
    )

    sim_s = pd.Timestamp(sim_start)
    sim_e = pd.Timestamp(sim_end) + pd.Timedelta(hours=23, minutes=59)

    # Swing needs tz-naive daily data
    tsla_d = data['tsla_daily'].copy()
    if tsla_d.index.tz is not None:
        tsla_d.index = tsla_d.index.tz_localize(None)

    spy_d = data['spy_daily'].copy()
    if spy_d.index.tz is not None:
        spy_d.index = spy_d.index.tz_localize(None)

    vix_d = data['vix_daily'].copy()
    if vix_d.index.tz is not None:
        vix_d.index = vix_d.index.tz_localize(None)

    tsla_w = tsla_d.resample('W-FRI').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    spy_w = spy_d.resample('W-FRI').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()

    old_max = swing_mod.MAX_TRADE_USD
    swing_mod.MAX_TRADE_USD = capital
    try:
        result = run_swing_backtest(
            tsla=tsla_d, spy=spy_d, vix=vix_d,
            tsla_weekly=tsla_w, spy_weekly=spy_w,
            signal_fn=sig_s1041_s993_or_s1034,
            signal_name='S1041',
            max_hold_days=10, stop_pct=0.05,
            start_year=2026, end_year=2026,
        )
    finally:
        swing_mod.MAX_TRADE_USD = old_max

    out = []
    for t in result.trades:
        entry = _ts_naive(t.entry_date)
        if sim_s <= entry <= sim_e:
            out.append(SimTrade(
                system='swing',
                entry_time=str(t.entry_date),
                exit_time=str(t.exit_date),
                direction='LONG' if t.direction == 1 else 'SHORT',
                pnl=t.pnl_usd,
            ))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Forward Simulation')
    parser.add_argument('--start', default='2026-02-23')
    parser.add_argument('--end',   default='2026-02-26')
    parser.add_argument('--capital', type=float, default=1_000_000)
    args = parser.parse_args()

    total_capital = args.capital
    baseline = 100_000  # run each system at $100K, scale later

    print(f"\n{'#'*70}")
    print(f"# FORWARD SIMULATION: {args.start} to {args.end}")
    print(f"# Total capital: ${total_capital:,.0f}")
    print(f"# Baseline per system: ${baseline:,.0f} (scale by allocation)")
    print(f"{'#'*70}")

    # ── Download ──
    t0 = time.time()
    data = download_data()
    print(f"  Downloaded in {time.time() - t0:.1f}s")

    # ── Run all systems ──
    all_trades: Dict[str, List[SimTrade]] = {}

    for system in SYSTEMS:
        print(f"\n{'='*70}")
        print(f"  RUNNING: {system.upper()}")
        print(f"{'='*70}")
        t_sys = time.time()
        try:
            if system == 'swing':
                trades = run_swing(data, baseline, args.start, args.end)
            else:
                trades = run_surfer(data, system, baseline, args.start, args.end)
            all_trades[system] = trades
            elapsed = time.time() - t_sys
            print(f"\n  >> {system}: {len(trades)} trades in sim window ({elapsed:.1f}s)")
        except Exception as e:
            print(f"\n  >> {system}: ERROR -- {e}")
            import traceback; traceback.print_exc()
            all_trades[system] = []

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n\n{'#'*70}")
    print(f"# RESULTS: {args.start} to {args.end}")
    print(f"{'#'*70}")

    # ── Per-system trade list ──
    print(f"\n{'='*70}")
    print(f"ALL TRADES BY SYSTEM (baseline ${baseline:,.0f})")
    print(f"{'='*70}")

    for system in SYSTEMS:
        trades = all_trades[system]
        print(f"\n  [{system.upper()}] -- {len(trades)} trades")
        if not trades:
            print(f"    (no trades in window)")
            continue
        total_pnl = sum(t.pnl for t in trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / len(trades) * 100 if trades else 0
        for t in trades:
            print(f"    {t.direction:<5} {t.entry_time}  ->  {t.exit_time}  "
                  f"P&L=${t.pnl:>8,.0f}")
        print(f"    {'---':>5}")
        print(f"    Total: ${total_pnl:>8,.0f}  |  {len(trades)} trades  |  "
              f"WR={wr:.0f}%  |  Avg=${total_pnl / len(trades):>6,.0f}")

    # ── System summary table ──
    print(f"\n{'='*70}")
    print(f"SYSTEM SUMMARY (baseline ${baseline:,.0f})")
    print(f"{'='*70}")
    print(f"  {'System':<8} {'Trades':>7} {'Total P&L':>12} {'WR':>6} {'Avg P&L':>10}")
    print(f"  {'-'*50}")

    for system in SYSTEMS:
        trades = all_trades[system]
        n = len(trades)
        if n == 0:
            print(f"  {system:<8} {'0':>7} {'$0':>12} {'n/a':>6} {'n/a':>10}")
            continue
        total_pnl = sum(t.pnl for t in trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / n * 100
        avg = total_pnl / n
        print(f"  {system:<8} {n:>7} ${total_pnl:>10,.0f} {wr:>5.0f}% ${avg:>8,.0f}")

    # ── Allocation comparison ──
    print(f"\n{'='*70}")
    print(f"PORTFOLIO P&L BY ALLOCATION (${total_capital:,.0f} total)")
    print(f"{'='*70}")
    sys_hdr = '  '.join(f'{s:>5}' for s in SYSTEMS)
    print(f"  {'Allocation':<35} {sys_hdr}   {'Trades':>6} {'P&L':>12} {'ROI':>7}")
    print(f"  {'-'*95}")

    for name, weights in ALLOCATIONS.items():
        portfolio_pnl = 0.0
        portfolio_trades = 0
        for i, system in enumerate(SYSTEMS):
            if weights[i] == 0:
                continue
            scale = (weights[i] * total_capital) / baseline
            for t in all_trades[system]:
                portfolio_pnl += t.pnl * scale
            portfolio_trades += len(all_trades[system])

        roi = portfolio_pnl / total_capital * 100
        w_str = '  '.join(f'{w:>4.0%}' for w in weights)
        print(f"  {name:<35} {w_str}   {portfolio_trades:>6} "
              f"${portfolio_pnl:>10,.0f} {roi:>6.2f}%")

    # ── Per-allocation trade detail ──
    print(f"\n{'='*70}")
    print(f"SCALED TRADES PER ALLOCATION")
    print(f"{'='*70}")

    for name, weights in ALLOCATIONS.items():
        print(f"\n  --- {name} ---")
        has_trades = False
        alloc_pnl = 0.0
        for i, system in enumerate(SYSTEMS):
            if weights[i] == 0:
                continue
            scale = (weights[i] * total_capital) / baseline
            cap_str = f"${weights[i] * total_capital:,.0f}"
            for t in all_trades[system]:
                scaled = t.pnl * scale
                alloc_pnl += scaled
                print(f"    [{system:>5}] {t.direction:<5} {t.entry_time}  "
                      f"P&L=${scaled:>10,.0f}  (capital={cap_str})")
                has_trades = True
        if not has_trades:
            print(f"    (no trades)")
        else:
            print(f"    {'':>8} TOTAL P&L = ${alloc_pnl:>10,.0f}")

    print(f"\n{'='*70}")
    print(f"FORWARD SIMULATION COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
