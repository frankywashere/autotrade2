#!/usr/bin/env python3
"""
c9 Monte Carlo — Is the DOW pattern statistically significant?

Collects all bounce trades from 11yr backtest. For each trade, records
the DOW and P&L. Then:

  Null hypothesis: DOW has NO effect — any assignment of multipliers
  to days is equally good.

  Test: Shuffle which DOW each trade occurred on (10,000 times).
  Apply Arch415 DOW multipliers to the shuffled assignments.
  Compare actual total P&L vs shuffled distribution.

  If actual P&L is in top 5% → DOW pattern is significant (p < 0.05).
  If actual P&L is in top 1% → highly significant (p < 0.01).

Also tests: Is the timing (sequencing of wins/losses) significant?
  Shuffle all trade P&Ls randomly, keeping sizes. Compare equity curve
  statistics (Sharpe, max DD) to actual.

Usage:
    python3 -m v15.validation.c9_montecarlo \\
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt
"""
import argparse
import time
import numpy as np
import pandas as pd


DOW_MULTS = {0: 1.35, 1: 1.35, 2: 1.35, 3: 1.45, 4: 1.35}  # Arch415


def apply_dow_mults(pnls: np.ndarray, dows: np.ndarray, sizes: np.ndarray) -> float:
    """Given trade P&Ls, DOW assignments, and base sizes, compute total P&L with DOW multipliers."""
    total = 0.0
    for pnl, dow, size in zip(pnls, dows, sizes):
        mult = DOW_MULTS.get(int(dow), 1.0)
        # Scale P&L by how multiplier changes position size
        # (multiplier increases trade_size; P&L scales proportionally for same % move)
        total += pnl * mult
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsla',    default='data/TSLAMin.txt')
    parser.add_argument('--spy',     default=None)
    parser.add_argument('--bounce-cap', type=float, default=12.0)
    parser.add_argument('--capital', type=float, default=100_000.0)
    parser.add_argument('--n-sim',   type=int,   default=10_000)
    args = parser.parse_args()

    from v15.core.historical_data import prepare_backtest_data, prepare_year_data
    from v15.core.surfer_backtest import run_backtest

    print('=' * 70)
    print(f'c9 MONTE CARLO — DOW pattern significance test (n={args.n_sim:,})')
    print('=' * 70)

    t0 = time.time()
    print('\nLoading data...')
    full_data = prepare_backtest_data(args.tsla, args.spy)
    vix_df = None
    try:
        from v15.validation.vix_loader import load_vix_daily
        vix_df = load_vix_daily(start='2014-01-01', end='2025-12-31')
    except Exception:
        pass
    print(f'  Loaded in {time.time()-t0:.1f}s')

    # Collect all trades
    print('\nRunning 11yr backtest...')
    all_trades = []
    for year in range(2015, 2026):
        yd = prepare_year_data(full_data, year)
        if yd is None:
            continue
        tsla_5min = yd['tsla_5min']
        if len(tsla_5min) < 200:
            continue
        t_yr = time.time()
        result = run_backtest(
            days=0, eval_interval=6, max_hold_bars=60,
            position_size=args.capital / 10, min_confidence=0.45,
            use_multi_tf=True, tsla_df=tsla_5min,
            higher_tf_dict=yd['higher_tf_data'],
            spy_df_input=yd.get('spy_5min'), vix_df_input=vix_df,
            realistic=True, slippage_bps=3.0, commission_per_share=0.005,
            max_leverage=4.0, bounce_cap=args.bounce_cap,
            max_trade_usd=500_000.0, initial_capital=args.capital,
            capture_features=False,
        )
        metrics, trades, _ = result[:3]
        all_trades.extend(trades)
        print(f'  {year}: {metrics.total_trades} trades  P&L=${metrics.total_pnl:,.0f}  ({time.time()-t_yr:.1f}s)')

    print(f'\n  Total: {len(all_trades):,} trades')

    # Split bounce vs break
    bounce_trades = [t for t in all_trades if t.signal_type == 'bounce']
    print(f'  Bounce: {len(bounce_trades):,}  Break: {len(all_trades)-len(bounce_trades):,}')

    # Extract arrays
    pnls   = np.array([t.pnl for t in bounce_trades])
    sizes  = np.array([t.trade_size for t in bounce_trades])
    dows   = np.array([pd.Timestamp(t.entry_time).dayofweek for t in bounce_trades if t.entry_time])

    if len(dows) != len(pnls):
        # Fallback: zip with entry_time filter
        valid = [(t.pnl, t.trade_size, pd.Timestamp(t.entry_time).dayofweek)
                 for t in bounce_trades if t.entry_time]
        pnls  = np.array([v[0] for v in valid])
        sizes = np.array([v[1] for v in valid])
        dows  = np.array([v[2] for v in valid])

    print(f'\n  Bounce trades with DOW: {len(pnls):,}')

    # ─────────────────────────────────────────────────────────
    # TEST 1: DOW shuffle significance
    # ─────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('TEST 1 — DOW SHUFFLE (is Thu ×1.45 justified vs random DOW assignment?)')
    print('='*70)

    # Actual P&L with Arch415 DOW multipliers
    actual_dow_pnl = apply_dow_mults(pnls, dows, sizes)

    # Shuffle DOW assignments N times
    rng = np.random.default_rng(42)
    shuffled_pnls = np.zeros(args.n_sim)
    for i in range(args.n_sim):
        shuffled_dows = rng.permutation(dows)
        shuffled_pnls[i] = apply_dow_mults(pnls, shuffled_dows, sizes)

    pct = (shuffled_pnls < actual_dow_pnl).mean() * 100
    print(f'\n  Actual P&L with Arch415 DOW mults: ${actual_dow_pnl:,.0f}')
    print(f'  Shuffled mean:   ${shuffled_pnls.mean():,.0f}')
    print(f'  Shuffled std:    ${shuffled_pnls.std():,.0f}')
    print(f'  Shuffled 95th:   ${np.percentile(shuffled_pnls, 95):,.0f}')
    print(f'  Shuffled 99th:   ${np.percentile(shuffled_pnls, 99):,.0f}')
    print(f'\n  Actual percentile: {pct:.1f}th  (p-value: {(100-pct)/100:.4f})')
    if pct >= 99:
        print('  ✓✓ HIGHLY SIGNIFICANT (p < 0.01) — DOW patterns are real')
    elif pct >= 95:
        print('  ✓  SIGNIFICANT (p < 0.05) — DOW patterns are real')
    elif pct >= 90:
        print('     MARGINAL (p < 0.10) — DOW patterns may be real')
    else:
        print('  ✗  NOT SIGNIFICANT — DOW patterns may be noise')

    # ─────────────────────────────────────────────────────────
    # TEST 2: P&L sequence shuffle (is timing valuable?)
    # ─────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('TEST 2 — P&L SEQUENCE SHUFFLE (does trade ordering/timing matter?)')
    print('='*70)

    def sharpe_from_pnl(pnl_arr):
        # Group by approximate year (every ~1000 trades) for yearly Sharpe
        chunk = max(len(pnl_arr) // 11, 1)
        yr_pnls = [pnl_arr[i:i+chunk].sum() for i in range(0, len(pnl_arr), chunk)]
        yr_pnls = np.array(yr_pnls)
        return float(yr_pnls.mean() / yr_pnls.std()) if yr_pnls.std() > 0 else 0.0

    def max_dd_from_pnl(pnl_arr, capital=100_000.0):
        equity = capital + np.cumsum(pnl_arr)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        return float(dd.max())

    actual_sharpe = sharpe_from_pnl(pnls)
    actual_dd     = max_dd_from_pnl(pnls)
    actual_total  = pnls.sum()

    shuffled_sharpes = np.zeros(args.n_sim)
    shuffled_dds     = np.zeros(args.n_sim)
    for i in range(args.n_sim):
        sp = rng.permutation(pnls)
        shuffled_sharpes[i] = sharpe_from_pnl(sp)
        shuffled_dds[i]     = max_dd_from_pnl(sp)

    sharpe_pct = (shuffled_sharpes < actual_sharpe).mean() * 100
    dd_pct     = (shuffled_dds > actual_dd).mean() * 100  # actual DD better than this % of shuffled

    print(f'\n  Actual Sharpe: {actual_sharpe:.2f} | Shuffled mean: {shuffled_sharpes.mean():.2f} | '
          f'Actual pct: {sharpe_pct:.0f}th')
    print(f'  Actual MaxDD:  {actual_dd:.2%} | Shuffled mean: {shuffled_dds.mean():.2%} | '
          f'Actual DD better than: {dd_pct:.0f}% of shuffled')

    if sharpe_pct >= 95:
        print('  ✓  Trade ordering adds significant Sharpe — timing matters')
    else:
        print('  →  Sharpe is mostly driven by base edge (WR), not sequencing')
    if dd_pct >= 95:
        print('  ✓  MaxDD is significantly lower than random — drawdown control works')
    else:
        print('  →  MaxDD similar to random — drawdown control is average')

    # ─────────────────────────────────────────────────────────
    # TEST 3: Per-DOW P&L significance (t-test vs overall mean)
    # ─────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('TEST 3 — PER-DOW P&L SIGNIFICANCE (is each day different from average?)')
    print('='*70)
    from scipy import stats as scipy_stats

    overall_mean = pnls.mean()
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    fmt = '  {:<5s} {:>6s} {:>10s} {:>10s} {:>8s} {:>10s}'
    print(fmt.format('Day', 'N', 'Avg P&L', 'vs Overall', 'p-value', 'Significant'))
    for d in range(5):
        mask = dows == d
        d_pnls = pnls[mask]
        if len(d_pnls) < 10:
            continue
        t_stat, p_val = scipy_stats.ttest_1samp(d_pnls, overall_mean)
        diff = d_pnls.mean() - overall_mean
        sig = '✓ YES' if p_val < 0.05 else ('~ marginal' if p_val < 0.10 else 'no')
        print(fmt.format(dow_names[d], str(len(d_pnls)),
                         f'${d_pnls.mean():,.0f}', f'{diff:+,.0f}',
                         f'{p_val:.4f}', sig))

    print(f'\nTotal time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
