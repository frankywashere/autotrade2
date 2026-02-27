#!/usr/bin/env python3
"""
c9 Break Signal Energy Analysis — testing the counter-indicator hypothesis.

Observation: when price hits channel top, all SELL bounce signals fire AND
the "energy state" fires a BUY breakout (energy_ratio > 1.2). The breakout
fails and price crashes harder than expected.

Hypothesis: high energy_score on break signals is ANTI-correlated with
success (already noted in channel_surfer.py line 1011 as "anti-correlated,
minimal weight"). When combined with opposing bounce confluence, it predicts
a failed breakout.

Analyses:
  1. Break signal P&L bucketed by energy_score quartile
  2. Break signal P&L by (energy_score, confidence) interaction
  3. Break skip rule: skip if energy_score > threshold
     → show P&L impact of skipping high-energy break signals

Usage:
    python3 -m v15.validation.c9_break_energy \
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt
"""
import argparse
import time
from typing import List, Optional
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsla', default='data/TSLAMin.txt')
    parser.add_argument('--spy',  default=None)
    parser.add_argument('--bounce-cap', type=float, default=12.0)
    parser.add_argument('--capital',    type=float, default=100_000.0)
    args = parser.parse_args()

    from v15.core.historical_data import prepare_backtest_data, prepare_year_data
    from v15.core.surfer_backtest import run_backtest

    print('=' * 70)
    print('c9 BREAK SIGNAL ENERGY ANALYSIS — counter-indicator test')
    print('=' * 70)

    t0 = time.time()
    print('\nLoading data...')
    full_data = prepare_backtest_data(args.tsla, args.spy)

    vix_df = None
    try:
        from v15.validation.vix_loader import load_vix_daily
        vix_df = load_vix_daily(start='2014-01-01', end='2025-12-31')
        print(f'  VIX: {len(vix_df)} daily bars')
    except Exception as e:
        print(f'  VIX unavailable: {e}')

    print(f'  Loaded in {time.time()-t0:.1f}s')

    # Collect all trades across 11 years WITH signal metadata
    all_break_trades = []
    all_bounce_trades = []

    print('\nRunning 11yr backtest with capture_features=True...')
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
            capture_features=True,
        )
        if len(result) < 5:
            continue
        metrics, trades, equity_curve, trade_features, trade_signals = result

        for trade, sig_data in zip(trades, trade_signals):
            rec = {
                'year': year,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'win': 1 if trade.pnl > 0 else 0,
                'signal_type': trade.signal_type,
                'direction': trade.direction,
                'confidence': trade.confidence,
                'energy_score': sig_data.get('energy_score', 0.0),
                'position_score': sig_data.get('position_score', 0.0),
                'entropy_score': sig_data.get('entropy_score', 0.0),
                'confluence_score': sig_data.get('confluence_score', 0.0),
                'timing_score': sig_data.get('timing_score', 0.0),
                'channel_health': sig_data.get('channel_health', 0.0),
            }
            if trade.signal_type == 'break':
                all_break_trades.append(rec)
            elif trade.signal_type == 'bounce':
                all_bounce_trades.append(rec)

        wr = metrics.wins / max(metrics.total_trades, 1)
        print(f'  {year}: {metrics.total_trades} trades  WR={wr:.1%}  '
              f'P&L=${metrics.total_pnl:,.0f}  ({time.time()-t_yr:.1f}s)')

    df_brk = pd.DataFrame(all_break_trades)
    df_bnc = pd.DataFrame(all_bounce_trades)

    print(f'\n  Break trades: {len(df_brk):,}  Bounce trades: {len(df_bnc):,}')

    if len(df_brk) == 0:
        print('\nNo break trades found — cannot analyse.')
        return

    # =========================================================================
    print('\n' + '='*70)
    print('ANALYSIS 1 — BREAK SIGNAL ENERGY SCORE vs P&L')
    print('='*70)
    print('\nBreak trade P&L by energy_score quartile (energy = anti-correlated?)')
    # energy_score = min(1.0, energy_ratio/2.0) → saturates at 1.0
    # Use explicit bins since many values hit the 1.0 ceiling
    df_brk['energy_q'] = pd.cut(df_brk['energy_score'],
                                 bins=[0, 0.25, 0.50, 0.75, 1.01],
                                 labels=['Q1 <0.25', 'Q2 0.25-0.5', 'Q3 0.5-0.75', 'Q4 >0.75'])
    overall_avg_brk = df_brk['pnl'].mean()
    fmt = '  {:<15s} {:>6s} {:>7s} {:>10s} {:>10s} {:>10s}'
    print(fmt.format('Energy bucket', 'N', 'WR', 'Avg P&L', 'Total P&L', 'vs avg'))
    print('  ' + '-'*60)
    for bucket, grp in df_brk.groupby('energy_q', observed=True):
        n = len(grp)
        wr = grp['win'].mean()
        avg = grp['pnl'].mean()
        total = grp['pnl'].sum()
        ratio = avg / overall_avg_brk if overall_avg_brk != 0 else 1.0
        print(fmt.format(str(bucket), str(n), f'{wr:.1%}', f'${avg:,.0f}',
                         f'${total:,.0f}', f'{ratio:.2f}x'))
    print(f'  {"─"*60}')
    print(f'  Overall break avg: ${overall_avg_brk:,.0f}/trade, WR={df_brk["win"].mean():.1%}')

    # =========================================================================
    print('\n' + '='*70)
    print('ANALYSIS 2 — BREAK ENERGY × CONFIDENCE INTERACTION')
    print('='*70)
    print('\nEnergy HIGH (>0.5) vs LOW (<=0.5) split by confidence:')
    df_brk['energy_hi'] = (df_brk['energy_score'] > 0.5)
    df_brk['conf_hi'] = (df_brk['confidence'] > 0.65)
    fmt2 = '  {:<30s} {:>6s} {:>7s} {:>10s} {:>10s}'
    print(fmt2.format('Bucket', 'N', 'WR', 'Avg P&L', 'Total P&L'))
    print('  ' + '-'*60)
    combos = [
        ('Low energy + Low conf', False, False),
        ('Low energy + High conf', False, True),
        ('High energy + Low conf', True, False),
        ('High energy + High conf', True, True),
    ]
    for label, e_hi, c_hi in combos:
        grp = df_brk[(df_brk['energy_hi'] == e_hi) & (df_brk['conf_hi'] == c_hi)]
        if len(grp) == 0:
            continue
        n = len(grp)
        wr = grp['win'].mean()
        avg = grp['pnl'].mean()
        total = grp['pnl'].sum()
        print(fmt2.format(label, str(n), f'{wr:.1%}', f'${avg:,.0f}', f'${total:,.0f}'))

    # =========================================================================
    print('\n' + '='*70)
    print('ANALYSIS 3 — BREAK SKIP RULE: SKIP HIGH-ENERGY BREAKS')
    print('='*70)
    print('\nImpact of skipping break signals above energy_score threshold:')
    baseline_total = df_brk['pnl'].sum()
    fmt3 = '  {:<15s} {:>8s} {:>7s} {:>12s} {:>10s} {:>12s}'
    print(fmt3.format('Skip threshold', 'Skipped', 'WR(skip)', 'Skip P&L',
                      'Remaining', 'Remaining P&L'))
    print('  ' + '-'*70)
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        skip_mask = df_brk['energy_score'] > thresh
        kept_mask = ~skip_mask
        skip_grp = df_brk[skip_mask]
        kept_grp = df_brk[kept_mask]
        skip_wr = skip_grp['win'].mean() if len(skip_grp) > 0 else 0.0
        skip_pnl = skip_grp['pnl'].sum()
        kept_pnl = kept_grp['pnl'].sum()
        delta = kept_pnl - baseline_total
        print(fmt3.format(
            f'>{thresh:.1f}',
            str(len(skip_grp)),
            f'{skip_wr:.1%}',
            f'${skip_pnl:,.0f}',
            str(len(kept_grp)),
            f'${kept_pnl:,.0f} ({delta:+,.0f})'
        ))
    print(f'  Baseline (no skip): {len(df_brk)} trades, ${baseline_total:,.0f}')

    # =========================================================================
    print('\n' + '='*70)
    print('ANALYSIS 4 — POSITION SCORE vs P&L FOR BREAK SIGNALS')
    print('='*70)
    print('\nHigher position_score = price closer to channel boundary:')
    df_brk['pos_bucket'] = pd.cut(df_brk['position_score'],
                                   bins=[0, 0.5, 0.7, 0.85, 1.0],
                                   labels=['<0.5 (middle)', '0.5-0.7', '0.7-0.85', '>0.85 (boundary)'])
    fmt4 = '  {:<18s} {:>6s} {:>7s} {:>10s} {:>10s}'
    print(fmt4.format('Position bucket', 'N', 'WR', 'Avg P&L', 'Total P&L'))
    print('  ' + '-'*55)
    for bucket, grp in df_brk.groupby('pos_bucket', observed=True):
        if len(grp) == 0:
            continue
        print(fmt4.format(str(bucket), str(len(grp)), f'{grp["win"].mean():.1%}',
                          f'${grp["pnl"].mean():,.0f}', f'${grp["pnl"].sum():,.0f}'))

    # =========================================================================
    print('\n' + '='*70)
    print('ANALYSIS 5 — YEAR-BY-YEAR BREAK P&L (context: are breaks worth it?)')
    print('='*70)
    total_pnl_all = df_brk['pnl'].sum() + df_bnc['pnl'].sum()
    brk_frac = df_brk['pnl'].sum() / total_pnl_all * 100 if total_pnl_all > 0 else 0
    print(f'\n  Break P&L share of total: {brk_frac:.1f}%')
    print(f'  Break trades: {len(df_brk):,} ({len(df_brk)/(len(df_brk)+len(df_bnc)):.0%} of all trades)')
    fmt5 = '  {:<6s} {:>6s} {:>7s} {:>10s} {:>10s}'
    print(fmt5.format('Year', 'N', 'WR', 'Avg P&L', 'Total P&L'))
    print('  ' + '-'*45)
    for yr, grp in df_brk.groupby('year'):
        print(fmt5.format(str(yr), str(len(grp)), f'{grp["win"].mean():.1%}',
                          f'${grp["pnl"].mean():,.0f}', f'${grp["pnl"].sum():,.0f}'))

    print(f'\nTotal time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
