#!/usr/bin/env python3
"""
Medium-Timeframe Channel Surfer Backtest — 1h / 4h primary TF.

Resamples 1-min TSLA+SPY data to the requested TF and runs the Channel Surfer
physics engine with daily/weekly higher-TF context.  Intended to capture
multi-hour directional moves (e.g. Feb-24-style AM/PM runs) that the 5-min
system's 5-hour max-hold misses.

Usage:
    python3 -m v15.validation.medium_tf_backtest \\
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt --tf 1h

    python3 -m v15.validation.medium_tf_backtest --tf 4h
    python3 -m v15.validation.medium_tf_backtest --tf 1h --oos-year 2025
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# TF-specific parameters
# ---------------------------------------------------------------------------

TF_PARAMS = {
    '1h': dict(
        eval_interval=1,
        max_hold_bars=10,       # 10 h ≈ 1.5 trading days
        bounce_cap=4.0,
        max_trade_usd=1_000_000.0,
        context_tfs=['daily', 'weekly'],
    ),
    '4h': dict(
        eval_interval=1,
        max_hold_bars=5,        # 5 × 4h = 20h ≈ 2.5 trading days
        bounce_cap=4.0,
        max_trade_usd=1_000_000.0,
        context_tfs=['daily', 'weekly'],
    ),
}

# Resample rule passed to resample_to_tf
TF_RESAMPLE = {
    '1h': '1h',
    '4h': '4h',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_and_resample(tsla_path: str, spy_path: Optional[str], tf: str):
    """Return (tsla_tf, spy_tf, daily_tsla, weekly_tsla, daily_spy)."""
    from v15.core.historical_data import load_minute_data, resample_to_tf

    print(f"  Loading TSLA 1-min from {tsla_path} …")
    tsla_1min = load_minute_data(tsla_path)
    print(f"  TSLA 1-min: {len(tsla_1min):,} bars")

    spy_1min = None
    if spy_path and os.path.isfile(spy_path):
        print(f"  Loading SPY 1-min from {spy_path} …")
        spy_1min = load_minute_data(spy_path)
        print(f"  SPY  1-min: {len(spy_1min):,} bars")

    rule = TF_RESAMPLE[tf]
    print(f"  Resampling to {tf} …")
    tsla_tf = resample_to_tf(tsla_1min, rule)
    spy_tf = resample_to_tf(spy_1min, rule) if spy_1min is not None else None
    print(f"  TSLA {tf}: {len(tsla_tf):,} bars")
    if spy_tf is not None:
        print(f"  SPY  {tf}: {len(spy_tf):,} bars")

    daily_tsla = resample_to_tf(tsla_1min, '1D')
    weekly_tsla = resample_to_tf(tsla_1min, '1W')
    daily_spy = resample_to_tf(spy_1min, '1D') if spy_1min is not None else None
    print(f"  TSLA daily: {len(daily_tsla):,}  weekly: {len(weekly_tsla):,}")

    return tsla_tf, spy_tf, daily_tsla, weekly_tsla, daily_spy


def _prepare_year(tsla_tf, spy_tf, daily_tsla, weekly_tsla, year: int,
                  lookback_days: int = 90) -> Optional[dict]:
    """Slice all DFs to `year` with a lookback buffer for channel detection."""
    cutoff_start = pd.Timestamp(f'{year - 1}-10-01')  # ~90 days before year
    cutoff_year_start = pd.Timestamp(f'{year}-01-01')
    cutoff_year_end = pd.Timestamp(f'{year}-12-31 23:59:59')

    def _slice(df, start, end):
        if df is None:
            return None
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask]

    tsla_slice = _slice(tsla_tf, cutoff_start, cutoff_year_end)
    spy_slice = _slice(spy_tf, cutoff_start, cutoff_year_end) if spy_tf is not None else None
    daily_slice = _slice(daily_tsla, cutoff_start, cutoff_year_end)
    weekly_slice = _slice(weekly_tsla, cutoff_start, cutoff_year_end)

    if tsla_slice is None or len(tsla_slice) < 20:
        return None

    # Higher TF dict as expected by run_backtest
    higher_tf_dict = {
        'daily': daily_slice,
        'weekly': weekly_slice,
    }

    return {
        'tsla_tf': tsla_slice,
        'spy_tf': spy_slice,
        'higher_tf_dict': higher_tf_dict,
        'year_start': cutoff_year_start,
        'year_end': cutoff_year_end,
    }


def _run_year(year_data: dict, tf: str, capital: float, vix_df,
              signal_filters=None) -> Optional[tuple]:
    """Run one year's backtest on the medium TF."""
    from v15.core.surfer_backtest import run_backtest

    p = TF_PARAMS[tf]
    tsla_tf = year_data['tsla_tf']
    if len(tsla_tf) < 20:
        return None

    result = run_backtest(
        days=0,
        eval_interval=p['eval_interval'],
        max_hold_bars=p['max_hold_bars'],
        position_size=capital / 10,
        min_confidence=0.45,
        use_multi_tf=True,
        tsla_df=tsla_tf,
        higher_tf_dict=year_data['higher_tf_dict'],
        spy_df_input=year_data.get('spy_tf'),
        vix_df_input=vix_df,
        realistic=True,
        slippage_bps=3.0,
        commission_per_share=0.005,
        max_leverage=4.0,
        bounce_cap=p['bounce_cap'],
        max_trade_usd=p['max_trade_usd'],
        initial_capital=capital,
        capture_features=False,
        signal_filters=signal_filters,
    )

    metrics, trades, equity_curve = result[:3]
    return metrics, trades, equity_curve


def _aggregate(results: dict) -> dict:
    if not results:
        return {}
    total_trades = sum(r[0].total_trades for r in results.values())
    total_wins = sum(r[0].wins for r in results.values())
    total_pnl = sum(r[0].total_pnl for r in results.values())
    gross_profit = sum(r[0].gross_profit for r in results.values())
    gross_loss = sum(r[0].gross_loss for r in results.values())
    max_dd = max(r[0].max_drawdown_pct for r in results.values())
    wr = total_wins / max(total_trades, 1)
    pf = gross_profit / max(abs(gross_loss), 1e-6)
    avg_pnl = total_pnl / max(total_trades, 1)
    yr_pnls = [r[0].total_pnl for r in results.values()]
    sharpe = float(np.mean(yr_pnls) / np.std(yr_pnls)) if len(yr_pnls) >= 2 and np.std(yr_pnls) > 0 else 0.0
    trades_per_yr = total_trades / max(len(results), 1)
    # Average hold (in bars) across all trades
    all_holds = []
    for r in results.values():
        for t in r[1]:
            hold = getattr(t, 'hold_bars', None) or getattr(t, 'bars_held', None)
            if hold is not None:
                all_holds.append(hold)
    avg_hold = float(np.mean(all_holds)) if all_holds else 0.0
    return {
        'trades': total_trades, 'wins': total_wins, 'wr': wr, 'pf': pf,
        'total_pnl': total_pnl, 'avg_pnl': avg_pnl, 'max_dd': max_dd,
        'sharpe': sharpe, 'trades_per_yr': trades_per_yr,
        'gross_profit': gross_profit, 'gross_loss': gross_loss,
        'avg_hold_bars': avg_hold,
    }


def _print_table(rows: list, tf: str):
    """Print comparison table. rows = list of (label, agg_dict)."""
    print(f"\n{'='*130}")
    print(f"MEDIUM TF BACKTEST — {tf.upper()} — COMPARISON TABLE")
    print(f"{'='*130}")
    hdr = (f"{'Config':<16} {'Trades':>7} {'WR':>7} {'PF':>6} {'Total P&L':>14} "
           f"{'Avg/Trade':>10} {'Sharpe':>7} {'Trd/Yr':>7} {'MaxDD':>8} {'AvgHold':>9}")
    print(hdr)
    print('-' * 130)
    baseline_pnl = next((a['total_pnl'] for lbl, a in rows if lbl == 'baseline'), 0)
    for label, agg in rows:
        if not agg:
            print(f"  {label}: no results")
            continue
        delta = agg['total_pnl'] - baseline_pnl
        delta_str = f"({delta:+,.0f})" if label != 'baseline' else ''
        print(f"{label:<16} {agg['trades']:>7,} {agg['wr']:>6.1%} {agg['pf']:>6.2f} "
              f"${agg['total_pnl']:>12,.0f} ${agg['avg_pnl']:>8,.0f} {agg['sharpe']:>7.2f} "
              f"{agg['trades_per_yr']:>7.0f} {agg['max_dd']:>7.1%} "
              f"{agg['avg_hold_bars']:>8.1f}  {delta_str}")
    print(f"{'='*130}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Medium-TF Channel Surfer backtest (1h or 4h primary)')
    parser.add_argument('--tsla', type=str, default='data/TSLAMin.txt')
    parser.add_argument('--spy',  type=str, default='data/SPYMin.txt')
    parser.add_argument('--tf',   type=str, default='1h', choices=['1h', '4h'])
    parser.add_argument('--years', type=str, default='2015-2024')
    parser.add_argument('--oos-year', type=int, default=2025)
    parser.add_argument('--capital', type=float, default=100_000.0)
    args = parser.parse_args()

    parts = args.years.split('-')
    start_year = int(parts[0])
    end_year = int(parts[1]) if len(parts) > 1 else start_year
    is_years = list(range(start_year, end_year + 1))

    print(f"\n{'='*70}")
    print(f"MEDIUM TF BACKTEST — {args.tf.upper()}")
    print(f"IS: {start_year}-{end_year}  OOS: {args.oos_year}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Load & resample
    # ------------------------------------------------------------------
    t0 = time.time()
    tsla_tf, spy_tf, daily_tsla, weekly_tsla, daily_spy = _load_and_resample(
        args.tsla, args.spy if os.path.isfile(args.spy) else None, args.tf)
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # VIX
    vix_df = None
    try:
        from v15.validation.vix_loader import load_vix_daily
        vix_df = load_vix_daily(start='2014-01-01', end='2025-12-31')
        print(f"  VIX loaded: {len(vix_df):,} rows")
    except Exception as e:
        print(f"  VIX load failed: {e}")

    # ------------------------------------------------------------------
    # Build configs: baseline + momentum variants
    # ------------------------------------------------------------------
    context_tfs = TF_PARAMS[args.tf]['context_tfs']

    from v15.core.signal_filters import SignalFilterCascade

    configs = {
        'baseline': None,
        'mtf_conflict': SignalFilterCascade(
            momentum_filter_enabled=True,
            momentum_boost=1.0,           # block only
            momentum_conflict_penalty=0.3,
            momentum_context_tfs=context_tfs,
            momentum_min_tfs=2,
        ),
        'mtf_exhaust': SignalFilterCascade(
            momentum_filter_enabled=True,
            momentum_boost=1.2,
            momentum_conflict_penalty=1.0,  # boost only
            momentum_context_tfs=context_tfs,
            momentum_min_tfs=2,
        ),
        'mtf_full': SignalFilterCascade(
            momentum_filter_enabled=True,
            momentum_boost=1.2,
            momentum_conflict_penalty=0.3,
            momentum_context_tfs=context_tfs,
            momentum_min_tfs=2,
        ),
    }

    # ------------------------------------------------------------------
    # IS grid search
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"IS GRID SEARCH — {len(configs)} configs × {len(is_years)} years")
    print(f"{'='*70}")

    all_is_results = {}

    for cfg_name, cascade in configs.items():
        print(f"\n--- Config: {cfg_name} ---")
        year_results = {}
        t_cfg = time.time()

        for year in is_years:
            year_data = _prepare_year(tsla_tf, spy_tf, daily_tsla, weekly_tsla, year)
            if year_data is None:
                print(f"  {year}: no data, skipping")
                continue

            if cascade is not None:
                for k in cascade.stats:
                    cascade.stats[k] = 0

            t_yr = time.time()
            result = _run_year(year_data, args.tf, args.capital, vix_df, cascade)
            if result is None:
                print(f"  {year}: too few bars, skipping")
                continue

            year_results[year] = result
            m = result[0]
            print(f"  {year}: ${m.total_pnl:>10,.0f}  {m.total_trades:>4} trades  "
                  f"WR={m.win_rate:.1%}  PF={m.profit_factor:.2f}  ({time.time()-t_yr:.1f}s)")

        all_is_results[cfg_name] = year_results
        agg = _aggregate(year_results)
        if agg:
            print(f"  Aggregate: ${agg['total_pnl']:,.0f}  Sharpe={agg['sharpe']:.2f}  "
                  f"Trades/yr={agg['trades_per_yr']:.0f}  AvgHold={agg['avg_hold_bars']:.1f}bars  "
                  f"({time.time()-t_cfg:.1f}s)")
        if cascade is not None:
            print(f"  {cascade.summary()}")

    # IS comparison table
    is_agg_rows = [(name, _aggregate(results)) for name, results in all_is_results.items()]
    _print_table(is_agg_rows, args.tf)

    # ------------------------------------------------------------------
    # OOS validation
    # ------------------------------------------------------------------
    if args.oos_year > 0:
        print(f"\n{'='*70}")
        print(f"OOS VALIDATION — {args.oos_year}")
        print(f"{'='*70}")

        oos_year_data = _prepare_year(tsla_tf, spy_tf, daily_tsla, weekly_tsla, args.oos_year)
        if oos_year_data is None:
            print(f"  {args.oos_year}: no OOS data available")
        else:
            oos_rows = []
            for cfg_name, cascade in configs.items():
                if cascade is not None:
                    for k in cascade.stats:
                        cascade.stats[k] = 0

                t_oos = time.time()
                result = _run_year(oos_year_data, args.tf, args.capital, vix_df, cascade)
                if result is None:
                    print(f"  {cfg_name}: too few bars")
                    continue

                m = result[0]
                print(f"  {cfg_name}: ${m.total_pnl:>10,.0f}  {m.total_trades:>4} trades  "
                      f"WR={m.win_rate:.1%}  PF={m.profit_factor:.2f}  ({time.time()-t_oos:.1f}s)")
                if cascade is not None:
                    print(f"    {cascade.summary()}")

                oos_rows.append((cfg_name, _aggregate({args.oos_year: result})))

            _print_table(oos_rows, f"{args.tf} OOS {args.oos_year}")

    print(f"\n{'='*70}")
    print("MEDIUM TF BACKTEST COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
