"""
A/B test: Surfer ML with soft gate (baseline) vs hard ML gate.

Usage:
    python -m v15.validation.surfer_ml_gate_test

$100K flat per year, no compounding. Per-year + all-years comparison.
"""
import sys
import time
from pathlib import Path

import pandas as pd

# Reuse helpers from c13a validation
from v15.validation.c13a_full_validation import (
    _load_surfer_data,
    _load_ml_model,
    _load_sq_model,
    _compute_metrics_surfer,
)


def _run_surfer_year(tsla_5m, higher_tf, spy_5m, vix_daily,
                     ml_model, sq_model, year, ml_hard_gate=False):
    """Run surfer backtest for a single year with optional hard ML gate."""
    from v15.core.surfer_backtest import run_backtest

    range_start = pd.Timestamp(f'{year}-01-01') - pd.Timedelta(days=90)
    range_end = pd.Timestamp(f'{year}-12-31 23:59:59')
    mask = (tsla_5m.index >= range_start) & (tsla_5m.index <= range_end)
    tsla_slice = tsla_5m.loc[mask]

    if len(tsla_slice) < 100:
        return []

    htf_warmup = pd.Timestamp(f'{year}-01-01') - pd.Timedelta(days=365)
    htf_slice = {}
    for tf, df in higher_tf.items():
        htf_slice[tf] = df.loc[(df.index >= htf_warmup) & (df.index <= range_end)]

    ml_size_fn = None
    if sq_model is not None:
        def ml_size_fn(quality_score):
            return max(0.5, min(2.0, quality_score * 2.0))

    # Suppress stdout/stderr from LightGBM etc. (redirect to /dev/null)
    import os
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)

    try:
        metrics, trades, equity_curve = run_backtest(
            days=0,
            eval_interval=3,
            max_hold_bars=60,
            position_size=10000.0,
            min_confidence=0.01,
            use_multi_tf=True,
            ml_model=ml_model,
            tsla_df=tsla_slice,
            higher_tf_dict=htf_slice,
            spy_df_input=spy_5m,
            vix_df_input=vix_daily,
            realistic=True,
            slippage_bps=3.0,
            commission_per_share=0.005,
            max_leverage=4.0,
            initial_capital=100_000.0,
            signal_quality_model=sq_model,
            ml_size_fn=ml_size_fn,
            ml_hard_gate=ml_hard_gate,
        )
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

    # Filter to requested year
    filtered = []
    for t in trades:
        if t.entry_time:
            try:
                entry_dt = pd.Timestamp(t.entry_time)
                if entry_dt.year == year:
                    filtered.append(t)
            except (ValueError, TypeError):
                pass
    return filtered


def main():
    tsla_min_path = 'data/TSLAMin.txt'
    if not Path(tsla_min_path).exists():
        print(f"ERROR: {tsla_min_path} not found")
        sys.exit(1)

    print("=" * 80)
    print("  SURFER ML GATE A/B TEST")
    print("  $100K flat per year, no compounding")
    print("=" * 80)

    print("\n[1] Loading data...")
    t0 = time.time()
    tsla_5m, higher_tf, spy_5m, vix_daily = _load_surfer_data(tsla_min_path, 2015, 2026)
    print(f"    Data loaded in {time.time()-t0:.1f}s")

    print("\n[2] Loading models...")
    ml_model = _load_ml_model()
    sq_model = _load_sq_model()

    years = list(range(2016, 2027))

    # --- Run A: Baseline (soft gate, ml_hard_gate=False) ---
    print("\n[3] Running BASELINE (soft gate = current behavior)...")
    baseline = {}
    baseline_trades = {}
    for yr in years:
        sys.stdout.write(f"    {yr}...")
        sys.stdout.flush()
        trades = _run_surfer_year(tsla_5m, higher_tf, spy_5m, vix_daily,
                                  ml_model, sq_model, yr, ml_hard_gate=False)
        baseline_trades[yr] = trades
        if trades:
            baseline[yr] = _compute_metrics_surfer(trades)
            m = baseline[yr]
            print(f" {m['trades']} trades, {m['win_rate']}% WR, ${m['total_pnl']:+,}")
        else:
            print(" 0 trades")

    # --- Run B: Hard gate (ml_hard_gate=True) ---
    print("\n[4] Running HARD GATE (skip if ML disagrees)...")
    gated = {}
    gated_trades = {}
    for yr in years:
        sys.stdout.write(f"    {yr}...")
        sys.stdout.flush()
        trades = _run_surfer_year(tsla_5m, higher_tf, spy_5m, vix_daily,
                                  ml_model, sq_model, yr, ml_hard_gate=True)
        gated_trades[yr] = trades
        if trades:
            gated[yr] = _compute_metrics_surfer(trades)
            m = gated[yr]
            print(f" {m['trades']} trades, {m['win_rate']}% WR, ${m['total_pnl']:+,}")
        else:
            print(" 0 trades")

    # --- Comparison table ---
    print("\n" + "=" * 80)
    print("  COMPARISON: Baseline (soft) vs Hard ML Gate")
    print("=" * 80)
    print(f"  {'Year':>6}  {'--- Baseline ---':^28}  {'--- Hard Gate ---':^28}  {'Delta':^12}")
    print(f"  {'':>6}  {'Trades':>6} {'WR%':>6} {'P&L':>12}  {'Trades':>6} {'WR%':>6} {'P&L':>12}  {'P&L':>12}")
    print(f"  {'-'*6}  {'-'*6} {'-'*6} {'-'*12}  {'-'*6} {'-'*6} {'-'*12}  {'-'*12}")

    tot_b_trades = tot_g_trades = 0
    tot_b_pnl = tot_g_pnl = 0
    tot_b_wins = tot_g_wins = 0

    for yr in years:
        b = baseline.get(yr, {'trades': 0, 'win_rate': 0, 'total_pnl': 0})
        g = gated.get(yr, {'trades': 0, 'win_rate': 0, 'total_pnl': 0})
        delta = g['total_pnl'] - b['total_pnl']
        marker = ' <<<' if delta > 1000 else (' >>>' if delta < -1000 else '')
        print(f"  {yr:>6}  {b['trades']:>6} {b['win_rate']:>5.1f}% ${b['total_pnl']:>+10,}  "
              f"{g['trades']:>6} {g['win_rate']:>5.1f}% ${g['total_pnl']:>+10,}  ${delta:>+10,}{marker}")
        tot_b_trades += b['trades']
        tot_g_trades += g['trades']
        tot_b_pnl += b['total_pnl']
        tot_g_pnl += g['total_pnl']

    # All trades combined for WR
    all_b = []
    all_g = []
    for yr in years:
        all_b.extend(baseline_trades.get(yr, []))
        all_g.extend(gated_trades.get(yr, []))

    b_wr = sum(1 for t in all_b if t.pnl_pct >= 0) / max(len(all_b), 1) * 100
    g_wr = sum(1 for t in all_g if t.pnl_pct >= 0) / max(len(all_g), 1) * 100

    print(f"  {'-'*6}  {'-'*6} {'-'*6} {'-'*12}  {'-'*6} {'-'*6} {'-'*12}  {'-'*12}")
    delta_total = tot_g_pnl - tot_b_pnl
    print(f"  {'TOTAL':>6}  {tot_b_trades:>6} {b_wr:>5.1f}% ${tot_b_pnl:>+10,}  "
          f"{tot_g_trades:>6} {g_wr:>5.1f}% ${tot_g_pnl:>+10,}  ${delta_total:>+10,}")

    # Verdict
    print(f"\n  Hard gate {'IMPROVES' if tot_g_pnl > tot_b_pnl else 'HURTS'} total P&L "
          f"by ${abs(delta_total):+,}")
    print(f"  Hard gate {'IMPROVES' if g_wr > b_wr else 'HURTS'} win rate "
          f"by {abs(g_wr - b_wr):+.1f}%")
    print(f"  Hard gate removes {tot_b_trades - tot_g_trades} trades "
          f"({(tot_b_trades - tot_g_trades)/max(tot_b_trades,1)*100:.1f}%)")


if __name__ == '__main__':
    main()
