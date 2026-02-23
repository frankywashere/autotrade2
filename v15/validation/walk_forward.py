#!/usr/bin/env python3
"""
Walk-Forward Validation — Arch415 / c9 branch.

Tests whether the DOW/TOD multipliers (and any future arch rules) actually
generalize out-of-sample, or are just data-mined artefacts.

Methodology:
  - Rolling windows: train on N years, test on next 1 year
  - Default: 5-year train window, 1-year test, rolling forward
  - Compares: Arch (current surfer_backtest.py) vs Naive baseline (no arch rules)
  - Reports: OOS P&L, WR, PF, and IS→OOS degradation ratio

Windows (default, 5yr train):
  IS 2015-2019 → OOS 2020
  IS 2016-2020 → OOS 2021
  IS 2017-2021 → OOS 2022
  IS 2018-2022 → OOS 2023
  IS 2019-2023 → OOS 2024
  IS 2020-2024 → OOS 2025

Usage:
    python3 -m v15.validation.walk_forward \\
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt \\
        --bounce-cap 12.0
"""

import argparse
import sys
import time
from typing import List, Dict, Tuple, Optional


def run_year_backtest(year_data, capital: float, vix_df) -> Optional[Tuple]:
    """Run one year of backtest with current arch params. Returns (metrics, trades, equity_curve)."""
    from v15.core.surfer_backtest import run_backtest

    tsla_5min = year_data['tsla_5min']
    if len(tsla_5min) < 200:
        return None

    result = run_backtest(
        days=0,
        eval_interval=6,
        max_hold_bars=60,
        position_size=capital / 10,
        min_confidence=0.45,
        use_multi_tf=True,
        tsla_df=tsla_5min,
        higher_tf_dict=year_data['higher_tf_data'],
        spy_df_input=year_data.get('spy_5min'),
        vix_df_input=vix_df,
        realistic=True,
        slippage_bps=3.0,
        commission_per_share=0.005,
        max_leverage=4.0,
        bounce_cap=BOUNCE_CAP,
        max_trade_usd=MAX_TRADE_USD,
        initial_capital=capital,
        capture_features=False,
    )
    return result[:3]


# Globals set by main() before run_year_backtest is called
BOUNCE_CAP = 12.0
MAX_TRADE_USD = 500_000.0


def aggregate_years(results: Dict[int, Tuple]) -> Dict:
    if not results:
        return {}
    import numpy as np
    total_trades = sum(r[0].total_trades for r in results.values())
    total_wins   = sum(r[0].wins          for r in results.values())
    total_pnl    = sum(r[0].total_pnl     for r in results.values())
    gross_profit = sum(r[0].gross_profit  for r in results.values())
    gross_loss   = sum(r[0].gross_loss    for r in results.values())
    max_dd       = max(r[0].max_drawdown_pct for r in results.values())
    wr  = total_wins / max(total_trades, 1)
    pf  = gross_profit / max(abs(gross_loss), 1e-6)
    yr_pnls = [r[0].total_pnl for r in results.values()]
    sharpe = float(np.mean(yr_pnls) / np.std(yr_pnls)) if len(yr_pnls) >= 2 else 0.0
    return {
        'trades': total_trades, 'wins': total_wins, 'wr': wr, 'pf': pf,
        'total_pnl': total_pnl, 'max_dd': max_dd, 'sharpe': sharpe,
        'yr_pnls': yr_pnls,
    }


def print_window(window_idx: int, is_years: List[int], oos_year: int,
                 is_res: Dict, oos_res: Optional[Tuple]):
    is_agg = aggregate_years({y: is_res[y] for y in is_years if y in is_res})
    print(f"\n  Window {window_idx}: IS={is_years[0]}-{is_years[-1]} → OOS={oos_year}")
    print(f"    IS:  trades={is_agg.get('trades',0):,}  WR={is_agg.get('wr',0):.1%}  "
          f"PF={is_agg.get('pf',0):.2f}  P&L=${is_agg.get('total_pnl',0):,.0f}  "
          f"Sharpe={is_agg.get('sharpe',0):.2f}")
    if oos_res:
        m = oos_res[0]
        print(f"    OOS: trades={m.total_trades:,}  WR={m.wins/max(m.total_trades,1):.1%}  "
              f"PF={m.gross_profit/max(abs(m.gross_loss),1e-6):.2f}  "
              f"P&L=${m.total_pnl:,.0f}  MaxDD={m.max_drawdown_pct:.1%}")
        # IS→OOS degradation
        is_avg = is_agg.get('total_pnl', 0) / max(len(is_years), 1)
        ratio = m.total_pnl / is_avg if is_avg > 0 else 0
        print(f"    IS avg/yr=${is_avg:,.0f}  OOS/IS ratio={ratio:.2f}x  "
              f"{'✓ holds up' if ratio >= 0.5 else '✗ degraded'}")
    else:
        print(f"    OOS: no data")


def main():
    global BOUNCE_CAP, MAX_TRADE_USD

    parser = argparse.ArgumentParser(description='Walk-forward validation of Channel Surfer arch')
    parser.add_argument('--tsla',        type=str,   default='data/TSLAMin.txt')
    parser.add_argument('--spy',         type=str,   default=None)
    parser.add_argument('--bounce-cap',  type=float, default=12.0)
    parser.add_argument('--max-trade-usd', type=float, default=500_000.0)
    parser.add_argument('--capital',     type=float, default=100_000.0)
    parser.add_argument('--train-years', type=int,   default=5,
                        help='Number of years in each IS training window (default: 5)')
    parser.add_argument('--start-year',  type=int,   default=2015)
    parser.add_argument('--end-year',    type=int,   default=2025)
    args = parser.parse_args()

    BOUNCE_CAP    = args.bounce_cap
    MAX_TRADE_USD = args.max_trade_usd

    # Load data once
    from v15.core.historical_data import prepare_backtest_data, prepare_year_data
    print(f"{'='*70}")
    print("WALK-FORWARD VALIDATION — Channel Surfer Arch415")
    print(f"  bounce_cap={BOUNCE_CAP}x  max_trade=${MAX_TRADE_USD:,.0f}  "
          f"capital=${args.capital:,.0f}  train_window={args.train_years}yr")
    print(f"{'='*70}")

    t0 = time.time()
    print("\nLoading data...")
    full_data = prepare_backtest_data(args.tsla, args.spy)

    vix_df = None
    try:
        from v15.validation.vix_loader import load_vix_daily
        vix_df = load_vix_daily(start='2014-01-01', end='2025-12-31')
    except Exception as e:
        print(f"  VIX load failed: {e}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    all_years = list(range(args.start_year, args.end_year + 1))

    # Pre-load all year data
    print("\nPre-loading year data...")
    year_data_cache = {}
    for yr in all_years:
        yd = prepare_year_data(full_data, yr)
        if yd is not None:
            year_data_cache[yr] = yd
        else:
            print(f"  {yr}: no data, skipping")

    available_years = sorted(year_data_cache.keys())
    print(f"  Available: {available_years}")

    # Build windows: IS = [start..start+train-1], OOS = start+train
    windows = []
    for i in range(len(available_years) - args.train_years):
        is_years  = available_years[i : i + args.train_years]
        oos_year  = available_years[i + args.train_years]
        windows.append((is_years, oos_year))

    if not windows:
        print(f"\nERROR: Not enough years for {args.train_years}-year train window.")
        sys.exit(1)

    print(f"\n{len(windows)} walk-forward windows (train={args.train_years}yr, test=1yr each)")

    # Run all backtests — cache results by year to avoid re-running
    print(f"\n{'='*70}")
    print("RUNNING BACKTESTS")
    print(f"{'='*70}")

    results_cache: Dict[int, Optional[Tuple]] = {}
    years_needed = set()
    for is_years, oos_year in windows:
        years_needed.update(is_years)
        years_needed.add(oos_year)

    for yr in sorted(years_needed):
        if yr not in year_data_cache:
            results_cache[yr] = None
            continue
        t_yr = time.time()
        r = run_year_backtest(year_data_cache[yr], args.capital, vix_df)
        results_cache[yr] = r
        if r:
            m = r[0]
            wr = m.wins / max(m.total_trades, 1)
            print(f"  {yr}: {m.total_trades:,} trades  WR={wr:.1%}  "
                  f"PF={m.gross_profit/max(abs(m.gross_loss),1e-6):.2f}  "
                  f"P&L=${m.total_pnl:,.0f}  ({time.time()-t_yr:.1f}s)")
        else:
            print(f"  {yr}: skipped (too few bars)")

    # Print per-window IS→OOS analysis
    print(f"\n{'='*70}")
    print("WALK-FORWARD RESULTS")
    print(f"{'='*70}")

    oos_results = {}
    for i, (is_years, oos_year) in enumerate(windows, 1):
        is_res = {y: results_cache[y] for y in is_years if results_cache.get(y)}
        oos_res = results_cache.get(oos_year)
        print_window(i, is_years, oos_year, is_res, oos_res)
        if oos_res:
            oos_results[oos_year] = oos_res

    # Aggregate OOS summary
    print(f"\n{'='*70}")
    print("AGGREGATE OOS SUMMARY")
    print(f"{'='*70}")
    oos_agg = aggregate_years(oos_results)
    if oos_agg:
        print(f"  OOS years:  {sorted(oos_results.keys())}")
        print(f"  Trades:     {oos_agg['trades']:,}")
        print(f"  Win Rate:   {oos_agg['wr']:.1%}")
        print(f"  PF:         {oos_agg['pf']:.2f}")
        print(f"  Total P&L:  ${oos_agg['total_pnl']:,.0f}")
        print(f"  Max DD:     {oos_agg['max_dd']:.1%}")
        print(f"  Sharpe:     {oos_agg['sharpe']:.2f}")
        print(f"  Profitable OOS years: "
              f"{sum(1 for p in oos_agg['yr_pnls'] if p > 0)}/{len(oos_agg['yr_pnls'])}")

    # Compare OOS aggregate vs full IS aggregate
    all_is_years = {y: results_cache[y] for y in available_years
                    if y not in oos_results and results_cache.get(y)}
    is_agg = aggregate_years(all_is_years)
    if is_agg and oos_agg:
        is_avg = is_agg['total_pnl'] / max(len(all_is_years), 1)
        oos_avg = oos_agg['total_pnl'] / max(len(oos_results), 1)
        ratio = oos_avg / is_avg if is_avg > 0 else 0
        print(f"\n  IS avg/yr:  ${is_avg:,.0f}")
        print(f"  OOS avg/yr: ${oos_avg:,.0f}")
        print(f"  OOS/IS:     {ratio:.2f}x  "
              f"{'✓ GENERALIZES (>0.5x)' if ratio >= 0.5 else '✗ DEGRADED (<0.5x) — possible overfit'}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
