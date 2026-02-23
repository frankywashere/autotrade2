#!/usr/bin/env python3
"""
c9 Parameter Sensitivity — how robust is Arch415 to parameter changes?

Sweeps:
  1. bounce_cap: 4x, 6x, 8x, 10x, 12x, 16x, 20x
     → shows P&L / Sharpe / MaxDD across cap levels
  2. DOW multiplier scale factor: 0.0x, 0.5x, 0.75x, 1.0x (baseline), 1.25x, 1.5x
     → scales ALL DOW multipliers by factor (1.0x = Arch415)
     → tests whether cliff-edge exists or gradual degradation
  3. min_confidence threshold: 0.30, 0.40, 0.45 (baseline), 0.50, 0.55, 0.60
     → shows how trade filtering affects P&L vs WR tradeoff

All sweeps run on 3 representative years: 2020 (bull/volatile), 2022 (bear), 2024 (bull).

Usage:
    python3 -m v15.validation.c9_sensitivity \\
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt
"""
import argparse
import sys
import time
from typing import List, Dict, Any
import numpy as np


def run_one(year_data, capital, vix_df, bounce_cap, max_trade_usd=500_000.0,
            min_confidence=0.45):
    from v15.core.surfer_backtest import run_backtest
    tsla_5min = year_data['tsla_5min']
    if len(tsla_5min) < 200:
        return None
    result = run_backtest(
        days=0, eval_interval=6, max_hold_bars=60,
        position_size=capital / 10, min_confidence=min_confidence,
        use_multi_tf=True, tsla_df=tsla_5min,
        higher_tf_dict=year_data['higher_tf_data'],
        spy_df_input=year_data.get('spy_5min'), vix_df_input=vix_df,
        realistic=True, slippage_bps=3.0, commission_per_share=0.005,
        max_leverage=4.0, bounce_cap=bounce_cap,
        max_trade_usd=max_trade_usd, initial_capital=capital,
        capture_features=False,
    )
    return result[:3]


def agg(results):
    if not results:
        return {}
    vals = list(results.values())
    total_pnl    = sum(r[0].total_pnl for r in vals)
    total_trades = sum(r[0].total_trades for r in vals)
    total_wins   = sum(r[0].wins for r in vals)
    gross_profit = sum(r[0].gross_profit for r in vals)
    gross_loss   = sum(r[0].gross_loss for r in vals)
    max_dd       = max(r[0].max_drawdown_pct for r in vals)
    wr   = total_wins / max(total_trades, 1)
    pf   = gross_profit / max(abs(gross_loss), 1e-6)
    yr_p = [r[0].total_pnl for r in vals]
    sharpe = float(np.mean(yr_p) / np.std(yr_p)) if len(yr_p) >= 2 else 0.0
    return {'pnl': total_pnl, 'trades': total_trades, 'wr': wr,
            'pf': pf, 'dd': max_dd, 'sharpe': sharpe}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsla',    default='data/TSLAMin.txt')
    parser.add_argument('--spy',     default=None)
    parser.add_argument('--capital', type=float, default=100_000.0)
    args = parser.parse_args()

    from v15.core.historical_data import prepare_backtest_data, prepare_year_data

    print('='*70)
    print('c9 SENSITIVITY ANALYSIS — Arch415 parameter robustness')
    print('='*70)

    t0 = time.time()
    full_data = prepare_backtest_data(args.tsla, args.spy)
    vix_df = None
    try:
        from v15.validation.vix_loader import load_vix_daily
        vix_df = load_vix_daily(start='2014-01-01', end='2025-12-31')
    except Exception:
        pass

    TEST_YEARS = [2020, 2022, 2024]
    year_cache = {}
    for yr in TEST_YEARS:
        yd = prepare_year_data(full_data, yr)
        if yd is not None:
            year_cache[yr] = yd

    print(f'\nTest years: {list(year_cache.keys())} (loaded in {time.time()-t0:.1f}s)')

    # ──────────────────────────────────────────────────────────
    # SWEEP 1: bounce_cap
    # ──────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('SWEEP 1 — bounce_cap (all other params = Arch415 baseline)')
    print('='*70)
    caps = [4, 6, 8, 10, 12, 16, 20]
    fmt = '{:<10s} {:>8s} {:>7s} {:>7s} {:>12s} {:>7s} {:>7s}'
    print(fmt.format('cap', 'Trades', 'WR', 'PF', 'Total P&L', 'MaxDD', 'Sharpe'))
    print('-'*62)
    cap_results = {}
    for cap in caps:
        res = {}
        for yr in TEST_YEARS:
            if yr not in year_cache:
                continue
            r = run_one(year_cache[yr], args.capital, vix_df, bounce_cap=cap)
            if r:
                res[yr] = r
        a = agg(res)
        cap_results[cap] = a
        marker = ' ← baseline' if cap == 12 else ''
        print(fmt.format(
            f'{cap}x', f"{a.get('trades',0):,}", f"{a.get('wr',0):.1%}",
            f"{a.get('pf',0):.2f}", f"${a.get('pnl',0):,.0f}",
            f"{a.get('dd',0):.1%}", f"{a.get('sharpe',0):.2f}",
        ) + marker)

    # ──────────────────────────────────────────────────────────
    # SWEEP 2: min_confidence threshold
    # ──────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('SWEEP 2 — min_confidence threshold (bounce_cap=12x)')
    print('='*70)
    confs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    print(fmt.format('min_conf', 'Trades', 'WR', 'PF', 'Total P&L', 'MaxDD', 'Sharpe'))
    print('-'*62)
    for conf in confs:
        res = {}
        for yr in TEST_YEARS:
            if yr not in year_cache:
                continue
            r = run_one(year_cache[yr], args.capital, vix_df,
                        bounce_cap=12.0, min_confidence=conf)
            if r:
                res[yr] = r
        a = agg(res)
        marker = ' ← baseline' if abs(conf - 0.45) < 0.001 else ''
        print(fmt.format(
            f'{conf:.2f}', f"{a.get('trades',0):,}", f"{a.get('wr',0):.1%}",
            f"{a.get('pf',0):.2f}", f"${a.get('pnl',0):,.0f}",
            f"{a.get('dd',0):.1%}", f"{a.get('sharpe',0):.2f}",
        ) + marker)

    # ──────────────────────────────────────────────────────────
    # SWEEP 3: max_trade_usd
    # ──────────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('SWEEP 3 — max_trade_usd cap (bounce_cap=12x, min_conf=0.45)')
    print('='*70)
    caps_usd = [100_000, 250_000, 500_000, 750_000, 1_000_000, 2_000_000, 0]
    fmt2 = '{:<14s} {:>8s} {:>7s} {:>7s} {:>12s} {:>7s} {:>7s}'
    print(fmt2.format('max_trade', 'Trades', 'WR', 'PF', 'Total P&L', 'MaxDD', 'Sharpe'))
    print('-'*66)
    for cap_usd in caps_usd:
        res = {}
        for yr in TEST_YEARS:
            if yr not in year_cache:
                continue
            r = run_one(year_cache[yr], args.capital, vix_df,
                        bounce_cap=12.0, max_trade_usd=cap_usd)
            if r:
                res[yr] = r
        a = agg(res)
        label = 'unlimited' if cap_usd == 0 else f'${cap_usd:,}'
        marker = ' ← baseline' if cap_usd == 500_000 else ''
        print(fmt2.format(
            label, f"{a.get('trades',0):,}", f"{a.get('wr',0):.1%}",
            f"{a.get('pf',0):.2f}", f"${a.get('pnl',0):,.0f}",
            f"{a.get('dd',0):.1%}", f"{a.get('sharpe',0):.2f}",
        ) + marker)

    print(f'\nTotal time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
