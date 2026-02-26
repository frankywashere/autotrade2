#!/usr/bin/env python3
"""
Medium-TF Walk-Forward Validation — 1h / 4h primary TF.

Same rolling-window methodology as walk_forward_filters.py but uses the
medium-timeframe backtest engine (1h or 4h primary TF) from medium_tf_backtest.py.

Rolling windows: 5yr IS → 1yr OOS
  IS 2015-2019 → OOS 2020
  IS 2016-2020 → OOS 2021
  IS 2017-2021 → OOS 2022
  IS 2018-2022 → OOS 2023
  IS 2019-2023 → OOS 2024
  IS 2020-2024 → OOS 2025

For each window runs baseline only (no filter, since MTF exhaust is neutral at
the medium-TF level), aggregates 5 IS years, and reports whether OOS P&L is
consistent with IS quality.

Usage:
    python3 -m v15.validation.medium_tf_walk_forward \\
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt --tf 1h
    python3 -m v15.validation.medium_tf_walk_forward --tf 4h
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TRAIN_YEARS = 5
START_YEAR  = 2015
END_YEAR    = 2025

# Import reusable helpers from medium_tf_backtest
from v15.validation.medium_tf_backtest import (
    _load_and_resample,
    _prepare_year,
    _run_year,
    TF_PARAMS,
)


def _agg_list(results: list) -> dict:
    """Aggregate a list of BacktestMetrics objects (not a dict like _aggregate)."""
    if not results:
        return {}
    total_trades = sum(r.total_trades for r in results)
    total_wins   = sum(r.wins         for r in results)
    total_pnl    = sum(r.total_pnl    for r in results)
    gross_profit = sum(r.gross_profit for r in results)
    gross_loss   = sum(r.gross_loss   for r in results)
    yr_pnls      = [r.total_pnl for r in results]
    sharpe = (float(np.mean(yr_pnls) / np.std(yr_pnls))
              if len(yr_pnls) >= 2 and np.std(yr_pnls) > 0 else 0.0)
    return {
        'trades':       total_trades,
        'wins':         total_wins,
        'wr':           total_wins / max(total_trades, 1),
        'pf':           gross_profit / max(abs(gross_loss), 1e-6),
        'pnl':          total_pnl,
        'sharpe':       sharpe,
        'n_years':      len(results),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Medium-TF walk-forward validation (1h or 4h)')
    parser.add_argument('--tsla',    type=str, default='data/TSLAMin.txt',
                        help='Path to 1-min TSLA data file')
    parser.add_argument('--spy',     type=str, default='data/SPYMin.txt',
                        help='Path to 1-min SPY data file')
    parser.add_argument('--tf',      type=str, default='1h', choices=['1h', '4h', '1d'],
                        help='Primary timeframe: 1h, 4h, or 1d (default: 1h)')
    parser.add_argument('--capital', type=float, default=100_000.0,
                        help='Initial capital per year (default: 100,000)')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"MEDIUM TF WALK-FORWARD VALIDATION — {args.tf.upper()}")
    print(f"Train window: {TRAIN_YEARS}yr IS → 1yr OOS  |  {START_YEAR}-{END_YEAR}")
    print(f"TF params: eval_interval={TF_PARAMS[args.tf]['eval_interval']}, "
          f"max_hold_bars={TF_PARAMS[args.tf]['max_hold_bars']}, "
          f"bounce_cap={TF_PARAMS[args.tf]['bounce_cap']}")
    print(f"{'='*70}")

    # ── Load & resample data ───────────────────────────────────────────────
    print("\nLoading data...")
    t0 = time.time()
    spy_path = args.spy if os.path.isfile(args.spy) else None
    tsla_tf, spy_tf, daily_tsla, weekly_tsla, daily_spy, monthly_tsla = _load_and_resample(
        args.tsla, spy_path, args.tf)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # VIX
    vix_df = None
    try:
        from v15.validation.vix_loader import load_vix_daily
        vix_df = load_vix_daily(start='2014-01-01', end='2025-12-31')
        print(f"  VIX loaded: {len(vix_df):,} rows")
    except Exception as e:
        print(f"  VIX load failed: {e}")

    # ── Build rolling windows ──────────────────────────────────────────────
    all_years = list(range(START_YEAR, END_YEAR + 1))
    windows = []
    for i in range(len(all_years) - TRAIN_YEARS):
        is_yrs = all_years[i : i + TRAIN_YEARS]
        oos_yr = all_years[i + TRAIN_YEARS]
        windows.append((is_yrs, oos_yr))

    print(f"\n{len(windows)} windows:")
    for is_yrs, oos_yr in windows:
        print(f"  IS {is_yrs[0]}-{is_yrs[-1]} → OOS {oos_yr}")

    # ── Cache year data ────────────────────────────────────────────────────
    needed_years = set()
    for is_yrs, oos_yr in windows:
        needed_years.update(is_yrs)
        needed_years.add(oos_yr)

    print(f"\nCaching {len(needed_years)} years of {args.tf} data...")
    year_cache = {}
    for yr in sorted(needed_years):
        yd = _prepare_year(tsla_tf, spy_tf, daily_tsla, weekly_tsla, yr,
                           monthly_tsla=monthly_tsla, primary_tf=args.tf)
        if yd is not None:
            year_cache[yr] = yd
            print(f"  {yr}: {len(yd['tsla_tf']):,} bars ({args.tf})")
        else:
            print(f"  {yr}: no data, skipping")

    # ── Run windows ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RUNNING WALK-FORWARD WINDOWS (baseline only)")
    print(f"{'='*70}")

    window_results = []

    for w_idx, (is_yrs, oos_yr) in enumerate(windows):
        print(f"\n── Window {w_idx+1}: IS {is_yrs[0]}-{is_yrs[-1]} → OOS {oos_yr} ──")
        t_w = time.time()

        # IS: aggregate across all 5 in-sample years
        is_metrics_list = []
        for yr in is_yrs:
            if yr not in year_cache:
                continue
            t_yr = time.time()
            result = _run_year(year_cache[yr], args.tf, args.capital, vix_df,
                               signal_filters=None)
            if result is None:
                print(f"  IS {yr}: too few bars, skipping")
                continue
            m = result[0]
            is_metrics_list.append(m)
            print(f"  IS {yr}: ${m.total_pnl:>10,.0f}  {m.total_trades:>4} trades  "
                  f"WR={m.win_rate:.1%}  ({time.time()-t_yr:.1f}s)")

        is_agg = _agg_list(is_metrics_list)

        # OOS: single year
        oos_result = None
        if oos_yr in year_cache:
            t_oos = time.time()
            result = _run_year(year_cache[oos_yr], args.tf, args.capital, vix_df,
                               signal_filters=None)
            if result is not None:
                oos_result = result[0]
                print(f"  OOS {oos_yr}: ${oos_result.total_pnl:>10,.0f}  "
                      f"{oos_result.total_trades:>4} trades  WR={oos_result.win_rate:.1%}  "
                      f"({time.time()-t_oos:.1f}s)")
            else:
                print(f"  OOS {oos_yr}: too few bars, skipping")
        else:
            print(f"  OOS {oos_yr}: no data")

        # Print window summary
        if is_agg:
            avg_is_pnl = is_agg['pnl'] / max(is_agg['n_years'], 1)
            print(f"\n  IS  aggregate: ${is_agg['pnl']:>12,.0f}  "
                  f"(${avg_is_pnl:>8,.0f}/yr)  WR={is_agg['wr']:.1%}  "
                  f"Sharpe={is_agg['sharpe']:.2f}  trades={is_agg['trades']:,}")

        if oos_result is not None and is_agg:
            avg_is_pnl = is_agg['pnl'] / max(is_agg['n_years'], 1)
            oos_is_ratio = (oos_result.total_pnl / avg_is_pnl
                            if abs(avg_is_pnl) > 1 else float('nan'))
            win_flag = 'WIN ' if oos_result.total_pnl > 0 else 'LOSS'
            print(f"  OOS {oos_yr}:   ${oos_result.total_pnl:>12,.0f}  "
                  f"OOS/IS_avg={oos_is_ratio:.2f}x  [{win_flag}]")
        elif oos_result is None:
            print(f"  OOS {oos_yr}: insufficient data")

        print(f"  Window elapsed: {time.time()-t_w:.0f}s")

        window_results.append({
            'is_yrs':    is_yrs,
            'oos_yr':    oos_yr,
            'is_agg':    is_agg,
            'oos_m':     oos_result,
        })

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD SUMMARY — {args.tf.upper()}")
    print(f"{'='*70}")
    hdr = f"{'Window':<24} {'IS P&L (5yr)':>14} {'IS avg/yr':>12} {'OOS P&L':>12} {'OOS/IS_avg':>12} {'OOS WR':>7} {'OOS Trd':>8} {'Result'}"
    print(hdr)
    print('-' * 105)

    oos_wins       = 0
    total_oos_pnl  = 0
    ratios         = []

    for r in window_results:
        label = f"IS {r['is_yrs'][0]}-{r['is_yrs'][-1]} OOS {r['oos_yr']}"
        is_agg = r['is_agg']
        oos_m  = r['oos_m']

        if not is_agg:
            print(f"  {label:<24} {'N/A':>14}")
            continue

        avg_is = is_agg['pnl'] / max(is_agg['n_years'], 1)

        if oos_m is not None:
            ratio   = oos_m.total_pnl / avg_is if abs(avg_is) > 1 else float('nan')
            win_str = 'WIN ' if oos_m.total_pnl > 0 else 'LOSS'
            if oos_m.total_pnl > 0:
                oos_wins += 1
            total_oos_pnl += oos_m.total_pnl
            ratios.append(ratio)
            print(f"  {label:<24} ${is_agg['pnl']:>12,.0f} ${avg_is:>10,.0f} "
                  f"${oos_m.total_pnl:>10,.0f} {ratio:>11.2f}x "
                  f"{oos_m.win_rate:>6.1%} {oos_m.total_trades:>8,}  [{win_str}]")
        else:
            print(f"  {label:<24} ${is_agg['pnl']:>12,.0f} ${avg_is:>10,.0f} "
                  f"{'N/A':>12} {'N/A':>12} {'N/A':>7} {'N/A':>8}  [SKIP]")

    valid = [r for r in window_results if r['oos_m'] is not None]
    avg_ratio = float(np.mean(ratios)) if ratios else float('nan')
    print('-' * 105)
    print(f"  OOS profitable windows: {oos_wins}/{len(valid)}")
    print(f"  Avg OOS/IS_avg ratio  : {avg_ratio:.2f}x")
    print(f"  Total OOS P&L         : ${total_oos_pnl:,.0f}")
    print(f"\n{'='*70}")
    print(f"MEDIUM TF WALK-FORWARD COMPLETE — {args.tf.upper()}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
