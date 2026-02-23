#!/usr/bin/env python3
"""
Validate ML Position Sizing in Real Backtest Engine.

Runs year-by-year backtests twice on identical data:
  1. Baseline: no ML sizing
  2. ML-sized: signal_quality_model + tier sizing function

Compares against the post-hoc estimate ($16.14M) to check whether
leverage caps (4x, 25% per-trade) erode the simulated gains.

Usage:
    python3 -m v15.validation.validate_sizing \
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt \
        --model v15/validation/signal_quality_model_tuned.pkl
"""

import argparse
import sys
import time
from typing import List

import numpy as np


# Tier sizing function (same as compare_filters.py)
def ml_size_tiers(q: float) -> float:
    """Tier-based position sizing: q>=80 -> 1.3x, q>=60 -> 1.0x, q>=40 -> 0.7x, else 0.4x."""
    if q >= 80:
        return 1.3
    elif q >= 60:
        return 1.0
    elif q >= 40:
        return 0.7
    else:
        return 0.4


def ml_size_uponly(q: float) -> float:
    """Upscale-only sizing: never reduces position (at 97% WR, downscaling destroys winners).
    q>=80 -> 1.5x, q>=60 -> 1.2x, q>=40 -> 1.05x, else 1.0x."""
    if q >= 80:
        return 1.5
    elif q >= 60:
        return 1.2
    elif q >= 40:
        return 1.05
    else:
        return 1.0


def run_year_backtest(year_data, year, capital, vix_df, signal_quality_model=None, ml_size_fn=None,
                      bounce_cap=12.0, max_trade_usd=500_000.0):
    """Run a single year backtest, return (metrics, trades, equity_curve)."""
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
        bounce_cap=bounce_cap,
        max_trade_usd=max_trade_usd,
        initial_capital=capital,
        capture_features=False,
        signal_quality_model=signal_quality_model,
        ml_size_fn=ml_size_fn,
    )

    metrics, trades, equity_curve = result[:3]
    return metrics, trades, equity_curve


def main():
    parser = argparse.ArgumentParser(description='Validate ML position sizing in real backtest')
    parser.add_argument('--tsla', type=str, default='data/TSLAMin.txt')
    parser.add_argument('--spy', type=str, default=None)
    parser.add_argument('--model', type=str, default='v15/validation/signal_quality_model_tuned.pkl')
    parser.add_argument('--years', type=str, default='2015-2025')
    parser.add_argument('--capital', type=float, default=100_000.0)
    parser.add_argument('--bounce-cap', type=float, default=12.0,
                        help='Max exposure cap multiplier for bounce signals (default: 12.0, Arch384)')
    parser.add_argument('--max-trade-usd', type=float, default=1_000_000.0,
                        help='Hard dollar cap per trade, 0=unlimited (default: 1000000, Arch418)')
    args = parser.parse_args()

    # Parse year range
    parts = args.years.split('-')
    start_year = int(parts[0])
    end_year = int(parts[1]) if len(parts) > 1 else start_year
    years = list(range(start_year, end_year + 1))

    # Load signal quality model
    from v15.validation.signal_quality_model import SignalQualityModel
    import os
    if not os.path.isfile(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    sqm = SignalQualityModel.load(args.model)
    print(f"Loaded signal quality model from {args.model}")
    print(f"  CV AUC: {sqm.cv_metrics.get('overall_auc', 'N/A')}")

    # Load data once
    from v15.core.historical_data import prepare_backtest_data, prepare_year_data

    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    t0 = time.time()
    full_data = prepare_backtest_data(args.tsla, args.spy)

    vix_df = None
    try:
        from v15.validation.vix_loader import load_vix_daily
        vix_df = load_vix_daily(start='2014-01-01', end='2025-12-31')
    except Exception as e:
        print(f"  VIX load failed: {e}")
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    # Run three backtests year-by-year: baseline, old tiers, upscale-only
    print(f"\n{'='*80}")
    print("RUNNING BACKTESTS: BASELINE vs OLD-TIERS vs UPSCALE-ONLY")
    print(f"  bounce_cap={args.bounce_cap}x, max_trade_usd=${args.max_trade_usd:,.0f}, capital=${args.capital:,.0f}")
    print(f"  Old tiers:     q>=80→1.3x, q>=60→1.0x, q>=40→0.7x, else→0.4x")
    print(f"  Upscale-only:  q>=80→1.5x, q>=60→1.2x, q>=40→1.05x, else→1.0x (never reduce)")
    print(f"{'='*80}")

    baseline_results = {}
    ml_results = {}
    up_results = {}

    for year in years:
        year_data = prepare_year_data(full_data, year)
        if year_data is None:
            print(f"  {year}: no data, skipping")
            continue

        t_year = time.time()

        # Baseline
        b = run_year_backtest(year_data, year, args.capital, vix_df,
                              bounce_cap=args.bounce_cap, max_trade_usd=args.max_trade_usd)
        if b is None:
            print(f"  {year}: too few bars, skipping")
            continue
        baseline_results[year] = b

        # Old tier sizing
        m = run_year_backtest(year_data, year, args.capital, vix_df,
                              signal_quality_model=sqm, ml_size_fn=ml_size_tiers,
                              bounce_cap=args.bounce_cap, max_trade_usd=args.max_trade_usd)
        ml_results[year] = m

        # Upscale-only sizing
        u = run_year_backtest(year_data, year, args.capital, vix_df,
                              signal_quality_model=sqm, ml_size_fn=ml_size_uponly,
                              bounce_cap=args.bounce_cap, max_trade_usd=args.max_trade_usd)
        up_results[year] = u

        bm = b[0]
        mm = m[0]
        um = u[0]
        elapsed = time.time() - t_year
        print(f"  {year}: base ${bm.total_pnl:,.0f} | "
              f"old-tiers ${mm.total_pnl:,.0f} ({mm.total_pnl - bm.total_pnl:+,.0f}) | "
              f"uponly ${um.total_pnl:,.0f} ({um.total_pnl - bm.total_pnl:+,.0f}) "
              f"({elapsed:.1f}s)")

    # Aggregate metrics
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS (11-year totals)")
    print(f"{'='*80}")

    def aggregate(results):
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
        import numpy as np
        sharpe = float(np.mean(yr_pnls) / np.std(yr_pnls)) if len(yr_pnls) >= 2 else 0.0
        return {
            'trades': total_trades, 'wins': total_wins, 'wr': wr, 'pf': pf,
            'total_pnl': total_pnl, 'avg_pnl': avg_pnl, 'max_dd': max_dd,
            'gross_profit': gross_profit, 'gross_loss': gross_loss, 'sharpe': sharpe,
        }

    b_agg = aggregate(baseline_results)
    m_agg = aggregate(ml_results)
    u_agg = aggregate(up_results)

    fmt = "{:<22s} {:>10s} {:>7s} {:>9s} {:>14s} {:>10s} {:>8s} {:>7s}"
    print(fmt.format('', 'Trades', 'WR', 'PF', 'Total P&L', 'Avg P&L', 'Max DD', 'Sharpe'))
    print('-' * 89)
    for label, agg in [('Baseline', b_agg), ('Old tiers (0.4-1.3x)', m_agg), ('Upscale-only (1.0-1.5x)', u_agg)]:
        if not agg:
            continue
        print(fmt.format(
            label,
            f"{agg['trades']:,}",
            f"{agg['wr']:.1%}",
            f"{agg['pf']:.2f}",
            f"${agg['total_pnl']:,.0f}",
            f"${agg['avg_pnl']:,.0f}",
            f"{agg['max_dd']:.1%}",
            f"{agg['sharpe']:.2f}",
        ))

    # Delta analysis
    print(f"\n{'='*80}")
    print("DELTA vs BASELINE")
    print(f"{'='*80}")
    for label, agg in [('Old tiers', m_agg), ('Upscale-only', u_agg)]:
        if not agg or not b_agg:
            continue
        d = agg['total_pnl'] - b_agg['total_pnl']
        dp = d / max(abs(b_agg['total_pnl']), 1) * 100
        sh_d = agg['sharpe'] - b_agg['sharpe']
        print(f"  {label:<22s}: P&L {d:+,.0f} ({dp:+.1f}%)  Sharpe {sh_d:+.2f}  "
              f"{'✓ BETTER' if d > 0 else '✗ WORSE'}")

    # Per-year breakdown
    print(f"\n{'='*80}")
    print("PER-YEAR BREAKDOWN (all three variants)")
    print(f"{'='*80}")
    all_years = sorted(set(baseline_results.keys()) | set(ml_results.keys()) | set(up_results.keys()))
    yr_fmt = "{:<6s} {:>10s} {:>12s} {:>9s} {:>12s} {:>9s} {:>12s} {:>9s}"
    print(yr_fmt.format('Year', 'Base P&L', 'OldTier P&L', 'OT Delta', 'UpOnly P&L', 'UP Delta', 'Best', 'Lift%'))
    print('-' * 93)
    for year in all_years:
        if year not in baseline_results:
            continue
        bm = baseline_results[year][0]
        mm = ml_results[year][0] if year in ml_results else None
        um = up_results[year][0] if year in up_results else None
        md = (mm.total_pnl - bm.total_pnl) if mm else 0
        ud = (um.total_pnl - bm.total_pnl) if um else 0
        best = 'up-only' if ud > md else ('old-tier' if md > 0 else 'base')
        best_lift = max(md, ud) / max(abs(bm.total_pnl), 1) * 100
        print(yr_fmt.format(
            str(year),
            f"${bm.total_pnl:,.0f}",
            f"${mm.total_pnl:,.0f}" if mm else 'N/A',
            f"{md:+,.0f}" if mm else 'N/A',
            f"${um.total_pnl:,.0f}" if um else 'N/A',
            f"{ud:+,.0f}" if um else 'N/A',
            best,
            f"{best_lift:+.1f}%",
        ))


if __name__ == '__main__':
    main()
