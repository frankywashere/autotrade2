#!/usr/bin/env python3
"""
Walk-Forward Filter Validation — compare sq50_bp vs baseline across rolling windows.

Methodology:
  Rolling windows: 5yr IS → 1yr OOS
  IS 2015-2019 → OOS 2020
  IS 2016-2020 → OOS 2021
  IS 2017-2021 → OOS 2022
  IS 2018-2022 → OOS 2023
  IS 2019-2023 → OOS 2024
  IS 2020-2024 → OOS 2025

For each window runs baseline + sq50_bp (the IS winner), reports whether
the IS advantage transfers OOS.

Usage:
    python3 -m v15.validation.walk_forward_filters \
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt
"""

import argparse
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TRAIN_YEARS = 5
START_YEAR  = 2015
END_YEAR    = 2025


def build_cascade(sq_gate=0.50, break_pred=True, swing=False):
    from v15.core.signal_filters import SignalFilterCascade
    return SignalFilterCascade(
        sq_gate_threshold=sq_gate,
        break_predictor_enabled=break_pred,
        swing_regime_enabled=swing,
        swing_boost=1.2,
        break_penalty=0.5,
    )


def run_year(year_data, capital, vix_df, signal_filters=None,
             bounce_cap=12.0, max_trade_usd=1_000_000.0):
    from v15.core.surfer_backtest import run_backtest
    tsla_5min = year_data['tsla_5min']
    if len(tsla_5min) < 200:
        return None
    result = run_backtest(
        days=0, eval_interval=6, max_hold_bars=60,
        position_size=capital / 10, min_confidence=0.45, use_multi_tf=True,
        tsla_df=tsla_5min, higher_tf_dict=year_data['higher_tf_data'],
        spy_df_input=year_data.get('spy_5min'), vix_df_input=vix_df,
        realistic=True, slippage_bps=3.0, commission_per_share=0.005,
        max_leverage=4.0, bounce_cap=bounce_cap, max_trade_usd=max_trade_usd,
        initial_capital=capital, capture_features=False,
        signal_filters=signal_filters,
    )
    return result[0]  # metrics only


def agg(results):
    if not results:
        return {}
    total_trades = sum(r.total_trades for r in results)
    total_wins   = sum(r.wins         for r in results)
    total_pnl    = sum(r.total_pnl    for r in results)
    gross_profit = sum(r.gross_profit for r in results)
    gross_loss   = sum(r.gross_loss   for r in results)
    pnls = [r.total_pnl for r in results]
    sharpe = float(np.mean(pnls) / np.std(pnls)) if len(pnls) >= 2 and np.std(pnls) > 0 else 0.0
    return {
        'trades': total_trades,
        'wr':     total_wins / max(total_trades, 1),
        'pf':     gross_profit / max(abs(gross_loss), 1e-6),
        'pnl':    total_pnl,
        'sharpe': sharpe,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsla', default='data/TSLAMin.txt')
    parser.add_argument('--spy',  default='data/SPYMin.txt')
    parser.add_argument('--capital',       type=float, default=100_000.0)
    parser.add_argument('--bounce-cap',    type=float, default=12.0)
    parser.add_argument('--max-trade-usd', type=float, default=1_000_000.0)
    args = parser.parse_args()

    from v15.core.historical_data import prepare_backtest_data, prepare_year_data, resample_to_tf, load_minute_data
    from v15.validation.vix_loader import load_vix_daily

    print(f"\n{'='*70}")
    print("WALK-FORWARD FILTER VALIDATION — sq50_bp vs baseline")
    print(f"Train window: {TRAIN_YEARS}yr IS → 1yr OOS  |  {START_YEAR}-{END_YEAR}")
    print(f"{'='*70}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading data...")
    t0 = time.time()
    full_data = prepare_backtest_data(args.tsla, args.spy)

    if 'weekly' not in full_data['higher_tf_data']:
        tsla_1min = full_data.get('tsla_1min')
        if tsla_1min is not None:
            full_data['higher_tf_data']['weekly'] = resample_to_tf(tsla_1min, '1W')

    daily_spy = None
    if args.spy and os.path.isfile(args.spy):
        spy_1min  = load_minute_data(args.spy)
        daily_spy = resample_to_tf(spy_1min, '1D')

    vix_df = load_vix_daily(start='2014-01-01', end='2025-12-31')
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Build rolling windows ─────────────────────────────────────────────
    all_years = list(range(START_YEAR, END_YEAR + 1))
    windows = []
    for i in range(len(all_years) - TRAIN_YEARS):
        is_yrs  = all_years[i : i + TRAIN_YEARS]
        oos_yr  = all_years[i + TRAIN_YEARS]
        windows.append((is_yrs, oos_yr))

    print(f"\n{len(windows)} windows:")
    for is_yrs, oos_yr in windows:
        print(f"  IS {is_yrs[0]}-{is_yrs[-1]} → OOS {oos_yr}")

    # ── Cache year data ───────────────────────────────────────────────────
    needed_years = set()
    for is_yrs, oos_yr in windows:
        needed_years.update(is_yrs)
        needed_years.add(oos_yr)

    print(f"\nCaching {len(needed_years)} years of data...")
    year_cache = {}
    for yr in sorted(needed_years):
        yd = prepare_year_data(full_data, yr)
        if yd:
            year_cache[yr] = yd
            print(f"  {yr}: {len(yd['tsla_5min']):,} bars")

    # ── Run windows ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RUNNING WINDOWS")
    print(f"{'='*70}")

    window_results = []

    for w_idx, (is_yrs, oos_yr) in enumerate(windows):
        print(f"\n── Window {w_idx+1}: IS {is_yrs[0]}-{is_yrs[-1]} → OOS {oos_yr} ──")
        t_w = time.time()

        # Build fresh cascade for this window
        cascade = build_cascade()
        daily_tsla = full_data['higher_tf_data'].get('daily')
        weekly_tsla = full_data['higher_tf_data'].get('weekly')

        # IS runs
        base_is, filt_is = [], []
        for yr in is_yrs:
            if yr not in year_cache:
                continue

            # Baseline
            if cascade is not None:
                for k in cascade.stats:
                    cascade.stats[k] = 0

            m_base = run_year(year_cache[yr], args.capital, vix_df,
                              signal_filters=None,
                              bounce_cap=args.bounce_cap,
                              max_trade_usd=args.max_trade_usd)
            if m_base:
                base_is.append(m_base)

            # Filtered
            if cascade is not None:
                for k in cascade.stats:
                    cascade.stats[k] = 0

            m_filt = run_year(year_cache[yr], args.capital, vix_df,
                              signal_filters=cascade,
                              bounce_cap=args.bounce_cap,
                              max_trade_usd=args.max_trade_usd)
            if m_filt:
                filt_is.append(m_filt)

        is_base = agg(base_is)
        is_filt = agg(filt_is)

        # OOS runs
        if cascade is not None:
            for k in cascade.stats:
                cascade.stats[k] = 0

        m_oos_base = run_year(year_cache.get(oos_yr), args.capital, vix_df,
                              signal_filters=None,
                              bounce_cap=args.bounce_cap,
                              max_trade_usd=args.max_trade_usd)

        fresh_cascade = build_cascade()
        m_oos_filt = run_year(year_cache.get(oos_yr), args.capital, vix_df,
                              signal_filters=fresh_cascade,
                              bounce_cap=args.bounce_cap,
                              max_trade_usd=args.max_trade_usd)

        oos_base = m_oos_base
        oos_filt = m_oos_filt

        # Print window summary
        print(f"  IS  baseline : ${is_base['pnl']:>12,.0f}  WR={is_base['wr']:.1%}  Sharpe={is_base['sharpe']:.2f}  trades={is_base['trades']:,}")
        print(f"  IS  sq50_bp  : ${is_filt['pnl']:>12,.0f}  WR={is_filt['wr']:.1%}  Sharpe={is_filt['sharpe']:.2f}  trades={is_filt['trades']:,}  delta=${is_filt['pnl']-is_base['pnl']:+,.0f}")
        if oos_base and oos_filt:
            delta_oos = oos_filt.total_pnl - oos_base.total_pnl
            wins = 'WIN ' if delta_oos >= 0 else 'LOSS'
            print(f"  OOS baseline : ${oos_base.total_pnl:>12,.0f}  WR={oos_base.win_rate:.1%}  trades={oos_base.total_trades:,}")
            print(f"  OOS sq50_bp  : ${oos_filt.total_pnl:>12,.0f}  WR={oos_filt.win_rate:.1%}  trades={oos_filt.total_trades:,}  delta=${delta_oos:+,.0f}  [{wins}]")

            # IS→OOS transfer ratio
            is_delta  = is_filt['pnl'] - is_base['pnl']
            ratio = delta_oos / is_delta if abs(is_delta) > 1 else float('nan')
            print(f"  IS→OOS transfer: {ratio:.2f}x  (1.0 = perfect, 0 = no transfer, <0 = reversal)")
        else:
            print(f"  OOS: insufficient data for {oos_yr}")

        print(f"  Window elapsed: {time.time()-t_w:.0f}s")

        window_results.append({
            'is_yrs': is_yrs, 'oos_yr': oos_yr,
            'is_base': is_base, 'is_filt': is_filt,
            'oos_base': oos_base, 'oos_filt': oos_filt,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*70}")
    print(f"{'Window':<22} {'IS delta':>12} {'OOS delta':>12} {'Transfer':>10} {'Result'}")
    print('-' * 65)

    oos_wins = 0
    total_oos_delta = 0
    for r in window_results:
        is_delta = r['is_filt']['pnl'] - r['is_base']['pnl']
        if r['oos_base'] and r['oos_filt']:
            oos_delta = r['oos_filt'].total_pnl - r['oos_base'].total_pnl
            ratio = oos_delta / is_delta if abs(is_delta) > 1 else float('nan')
            result = 'WIN ' if oos_delta >= 0 else 'LOSS'
            if oos_delta >= 0:
                oos_wins += 1
            total_oos_delta += oos_delta
            label = f"IS {r['is_yrs'][0]}-{r['is_yrs'][-1]} OOS {r['oos_yr']}"
            print(f"  {label:<20} {is_delta:>+12,.0f} {oos_delta:>+12,.0f} {ratio:>10.2f}x  [{result}]")
        else:
            label = f"IS {r['is_yrs'][0]}-{r['is_yrs'][-1]} OOS {r['oos_yr']}"
            print(f"  {label:<20} {is_delta:>+12,.0f} {'N/A':>12} {'N/A':>10}  [SKIP]")

    valid = [r for r in window_results if r['oos_base'] and r['oos_filt']]
    print('-' * 65)
    print(f"  OOS wins: {oos_wins}/{len(valid)}  Total OOS delta: ${total_oos_delta:+,.0f}")
    print(f"\n{'='*70}")
    print("WALK-FORWARD COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
