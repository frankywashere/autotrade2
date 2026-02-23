#!/usr/bin/env python3
"""
c9 Regime Analysis — what market conditions make trades better or worse?

Runs 11yr backtest, enriches each trade with VIX level, TSLA RSI-14,
SPY 5-bar trend, DOW, and hour-of-day at entry. Analyses:

  1. DOW breakdown — confirm Thu > others, find optimal multipliers
  2. VIX regime — low/mid/high VIX bucket P&L
  3. RSI at entry — overbought/oversold effect on bounce success
  4. SPY alignment — does SPY trend direction matter?
  5. Combined VIX × DOW — is Thu 1.45x justified in all VIX regimes?

Results inform whether Arch417 (regime-based sizing) is worth implementing.

Usage:
    python3 -m v15.validation.c9_regime_analysis \\
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt
"""
import argparse
import time
from typing import List, Optional
import numpy as np
import pandas as pd


def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = deltas[deltas > 0].sum() / period
    losses = (-deltas[deltas < 0]).sum() / period
    if losses < 1e-10:
        return 100.0
    return 100.0 - (100.0 / (1.0 + gains / losses))


def get_vix_at(entry_time, vix_df) -> Optional[float]:
    if vix_df is None or len(vix_df) == 0:
        return None
    try:
        ts = pd.Timestamp(entry_time)
        if vix_df.index.tz is not None:
            ts = ts.tz_localize(vix_df.index.tz) if ts.tzinfo is None else ts.tz_convert(vix_df.index.tz)
        else:
            ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
        avail = vix_df[vix_df.index <= ts]
        if len(avail) == 0:
            return None
        return float(avail['close'].iloc[-1])
    except Exception:
        return None


def get_spy_return_at(entry_time, spy_df, lookback: int = 5) -> Optional[float]:
    """SPY 5-bar return up to (not including) entry bar."""
    if spy_df is None or len(spy_df) == 0:
        return None
    try:
        ts = pd.Timestamp(entry_time)
        if spy_df.index.tz is not None:
            ts = ts.tz_localize(spy_df.index.tz) if ts.tzinfo is None else ts.tz_convert(spy_df.index.tz)
        else:
            ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
        avail = spy_df[spy_df.index <= ts]
        if len(avail) < lookback + 1:
            return None
        c = avail['close'].values
        return float((c[-1] - c[-(lookback + 1)]) / c[-(lookback + 1)])
    except Exception:
        return None


def get_tsla_rsi_at(entry_time, tsla_df, period: int = 14) -> Optional[float]:
    try:
        ts = pd.Timestamp(entry_time)
        if tsla_df.index.tz is not None:
            ts = ts.tz_localize(tsla_df.index.tz) if ts.tzinfo is None else ts.tz_convert(tsla_df.index.tz)
        else:
            ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
        avail = tsla_df[tsla_df.index <= ts]
        if len(avail) < period + 1:
            return None
        return compute_rsi(avail['close'].values, period)
    except Exception:
        return None


def bucket_report(trades_df: pd.DataFrame, bucket_col: str, value_col: str = 'pnl',
                  label: str = '') -> None:
    print(f"\n  {'─'*60}")
    print(f"  {label or bucket_col}")
    print(f"  {'─'*60}")
    fmt = "  {:<20s} {:>6s} {:>7s} {:>10s} {:>10s} {:>10s}"
    print(fmt.format('Bucket', 'N', 'WR', 'Avg P&L', 'Total P&L', 'Optimal mult'))
    print(f"  {'─'*60}")
    overall_avg = trades_df[value_col].mean()
    for bucket, grp in trades_df.groupby(bucket_col, observed=True):
        n = len(grp)
        wr = (grp[value_col] > 0).mean()
        avg = grp[value_col].mean()
        total = grp[value_col].sum()
        mult = avg / overall_avg if overall_avg > 0 else 1.0
        print(fmt.format(str(bucket), str(n), f"{wr:.1%}", f"${avg:,.0f}",
                         f"${total:,.0f}", f"{mult:.2f}x"))
    print(f"  {'─'*60}")
    print(f"  Overall avg: ${overall_avg:,.0f}/trade")


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
    print('c9 REGIME ANALYSIS — enriching trades with VIX / RSI / SPY / DOW')
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

    # Collect all trades across 11 years
    all_trades = []
    all_tsla_dfs = {}

    print('\nRunning 11yr backtest to collect trade records...')
    for year in range(2015, 2026):
        yd = prepare_year_data(full_data, year)
        if yd is None:
            continue
        tsla_5min = yd['tsla_5min']
        if len(tsla_5min) < 200:
            continue
        all_tsla_dfs[year] = tsla_5min

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
        for t in trades:
            t.year = year  # tag with year
        all_trades.extend(trades)
        wr = metrics.wins / max(metrics.total_trades, 1)
        print(f'  {year}: {metrics.total_trades} trades  WR={wr:.1%}  '
              f'P&L=${metrics.total_pnl:,.0f}  ({time.time()-t_yr:.1f}s)')

    print(f'\n  Total trades collected: {len(all_trades):,}')
    print(f'  Enriching with VIX / RSI / SPY...')

    # Build combined 11yr TSLA dataframe for RSI lookups
    tsla_all = pd.concat(all_tsla_dfs.values()).sort_index()
    spy_all = None
    if full_data.get('spy_5min') is not None:
        spy_all = full_data['spy_5min']
    elif full_data.get('spy_df') is not None:
        spy_all = full_data['spy_df']

    # Enrich trades
    records = []
    for t in all_trades:
        if not t.entry_time:
            continue
        ts = pd.Timestamp(t.entry_time)
        dow = ts.dayofweek                  # 0=Mon ... 6=Sun
        if dow > 4:
            continue  # skip weekend bars (pre/post-market edge cases)
        utc_hour = ts.hour if ts.tzinfo is None else ts.tz_convert('UTC').hour
        et_hour = (utc_hour - 5) % 24

        vix = get_vix_at(t.entry_time, vix_df)
        rsi = get_tsla_rsi_at(t.entry_time, tsla_all)
        spy_ret = get_spy_return_at(t.entry_time, spy_all)

        # SPY alignment: +1 = SPY trending same direction as trade, -1 = opposite
        spy_aligned = None
        if spy_ret is not None:
            if t.direction == 'BUY':
                spy_aligned = 1 if spy_ret > 0 else -1
            else:
                spy_aligned = 1 if spy_ret < 0 else -1

        records.append({
            'year': getattr(t, 'year', 0),
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'win': 1 if t.pnl > 0 else 0,
            'signal_type': t.signal_type,
            'direction': t.direction,
            'dow': dow,
            'dow_name': ['Mon','Tue','Wed','Thu','Fri'][dow],
            'et_hour': et_hour,
            'vix': vix,
            'rsi': rsi,
            'spy_ret': spy_ret,
            'spy_aligned': spy_aligned,
            'trade_size': t.trade_size,
        })

    df = pd.DataFrame(records)
    bounce_df = df[df['signal_type'] == 'bounce'].copy()
    print(f'  Bounce trades: {len(bounce_df):,}  Break trades: {len(df[df["signal_type"]=="break"]):,}')

    # ==========================================================================
    print('\n' + '='*70)
    print('ANALYSIS 1 — DOW EFFECT (bounce trades)')
    print('='*70)
    dow_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    bounce_df['dow_name'] = pd.Categorical(bounce_df['dow_name'], categories=dow_order, ordered=True)
    bucket_report(bounce_df, 'dow_name', label='Day of week → avg P&L and implied optimal multiplier')

    # ==========================================================================
    print('\n' + '='*70)
    print('ANALYSIS 2 — VIX REGIME (bounce trades)')
    print('='*70)
    vix_bounce = bounce_df.dropna(subset=['vix']).copy()
    if len(vix_bounce) > 0:
        vix_bounce['vix_bucket'] = pd.cut(
            vix_bounce['vix'],
            bins=[0, 15, 20, 25, 30, 100],
            labels=['<15 (ultra-low)', '15-20 (low)', '20-25 (mid)', '25-30 (high)', '>30 (fear)']
        )
        bucket_report(vix_bounce, 'vix_bucket', label='VIX level at entry → optimal multiplier for each regime')
        print(f'\n  VIX coverage: {len(vix_bounce):,}/{len(bounce_df):,} trades ({len(vix_bounce)/len(bounce_df):.0%})')

    # ==========================================================================
    print('\n' + '='*70)
    print('ANALYSIS 3 — RSI AT ENTRY (bounce trades)')
    print('='*70)
    rsi_bounce = bounce_df.dropna(subset=['rsi']).copy()
    if len(rsi_bounce) > 0:
        rsi_bounce['rsi_bucket'] = pd.cut(
            rsi_bounce['rsi'],
            bins=[0, 30, 40, 50, 60, 70, 100],
            labels=['<30 (oversold)', '30-40', '40-50', '50-60', '60-70', '>70 (overbought)']
        )
        bucket_report(rsi_bounce, 'rsi_bucket', label='TSLA RSI-14 at entry → trade quality')

    # ==========================================================================
    print('\n' + '='*70)
    print('ANALYSIS 4 — SPY ALIGNMENT (bounce trades)')
    print('='*70)
    spy_bounce = bounce_df.dropna(subset=['spy_aligned']).copy()
    if len(spy_bounce) > 0:
        spy_bounce['spy_label'] = spy_bounce['spy_aligned'].map({1: 'Aligned (SPY with signal)', -1: 'Counter (SPY vs signal)'})
        bucket_report(spy_bounce, 'spy_label', label='SPY 5-bar trend alignment vs signal direction')

    # ==========================================================================
    print('\n' + '='*70)
    print('ANALYSIS 5 — VIX × DOW INTERACTION (bounce trades)')
    print('='*70)
    if len(vix_bounce) > 0:
        vix_bounce['vix_simple'] = pd.cut(
            vix_bounce['vix'],
            bins=[0, 20, 30, 100],
            labels=['Low VIX (<20)', 'Mid VIX (20-30)', 'High VIX (>30)']
        )
        print('\n  P&L by DOW × VIX regime (bounce):')
        pivot = vix_bounce.pivot_table(values='pnl', index='dow_name', columns='vix_simple', aggfunc='mean')
        print(pivot.to_string())
        print('\n  Trade counts:')
        cnt = vix_bounce.pivot_table(values='pnl', index='dow_name', columns='vix_simple', aggfunc='count')
        print(cnt.to_string())

    # ==========================================================================
    print('\n' + '='*70)
    print('ANALYSIS 6 — HOUR OF DAY (bounce trades)')
    print('='*70)
    hr_bounce = bounce_df[bounce_df['et_hour'].between(8, 16)].copy()
    hr_bounce['hour_label'] = hr_bounce['et_hour'].apply(lambda h: f'{h}:00 ET')
    bucket_report(hr_bounce, 'hour_label', label='Hour of entry (ET) → optimal TOD multiplier')

    # ==========================================================================
    print('\n' + '='*70)
    print('SUMMARY — RECOMMENDED ARCH417 CANDIDATES')
    print('='*70)
    overall_avg = bounce_df['pnl'].mean()

    if len(vix_bounce) > 100:
        vix_bounce['vix_simple'] = pd.cut(
            vix_bounce['vix'],
            bins=[0, 20, 30, 100],
            labels=['Low (<20)', 'Mid (20-30)', 'High (>30)']
        )
        for bucket, grp in vix_bounce.groupby('vix_simple', observed=True):
            mult = grp['pnl'].mean() / overall_avg if overall_avg > 0 else 1.0
            print(f'  VIX {bucket}: avg=${grp["pnl"].mean():,.0f}  n={len(grp)}  '
                  f'implied_mult={mult:.2f}x  '
                  f'{"→ BOOST candidate" if mult > 1.1 else "→ REDUCE candidate" if mult < 0.9 else "→ neutral"}')

    if len(spy_bounce) > 100:
        for aligned, grp in spy_bounce.groupby('spy_aligned'):
            label = 'aligned' if aligned == 1 else 'counter'
            mult = grp['pnl'].mean() / overall_avg if overall_avg > 0 else 1.0
            print(f'  SPY {label}: avg=${grp["pnl"].mean():,.0f}  n={len(grp)}  '
                  f'implied_mult={mult:.2f}x  '
                  f'{"→ BOOST candidate" if mult > 1.1 else "→ REDUCE candidate" if mult < 0.9 else "→ neutral"}')

    print(f'\nTotal time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
