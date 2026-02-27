#!/usr/bin/env python3
"""
Out-of-sample validation: Train on 2016-2021, Test on 2022-2025.
Tests whether the refined bounce signals generalize beyond the tuning period.
"""
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.tf_state_backtest import (
    load_all_tfs, compute_daily_states, _norm_cols,
    _lin_reg_channel, _momentum_is_turning_up,
)
from v15.validation.bounce_timing import _compute_rsi
from v15.data.native_tf import fetch_native_tf
from v15.features.utils import calc_macd, calc_stochastic


def _lookup_weekly(series, date):
    idx = series.index.searchsorted(date, side='right') - 1
    return float(series.iloc[idx]) if 0 <= idx < len(series) else np.nan

def _lookup_daily(series, date):
    if date in series.index:
        return float(series.loc[date])
    idx = series.index.searchsorted(date, side='right') - 1
    if 0 <= idx < len(series):
        return float(series.iloc[idx])
    return np.nan


def main():
    start, end = '2015-01-01', '2026-12-31'

    print("Loading data...")
    tf_data = load_all_tfs('data/TSLAMin.txt', start, end)
    daily_df = tf_data['daily']
    weekly_df = tf_data['weekly']

    # Indicators
    print("Computing indicators...")
    spy = _norm_cols(fetch_native_tf('SPY', 'daily', start, end))
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_rsi = _compute_rsi(spy['close'], 14)

    tsla_rsi_d = _compute_rsi(daily_df['close'], 14)
    tsla_rsi_w = _compute_rsi(weekly_df['close'], 14)
    tsla_rsi_w_sma = tsla_rsi_w.rolling(14, min_periods=1).mean()

    d_close = daily_df['close'].values
    _, _, macd_hist_d = calc_macd(d_close)
    macd_hist_d_s = pd.Series(macd_hist_d, index=daily_df.index)

    stoch_k, stoch_d = calc_stochastic(
        daily_df['high'].values, daily_df['low'].values, d_close
    )
    stoch_k_s = pd.Series(stoch_k, index=daily_df.index)
    stoch_d_s = pd.Series(stoch_d, index=daily_df.index)

    daily_df = daily_df.copy()
    daily_df['rolling_max_20'] = daily_df['high'].rolling(20, min_periods=1).max()
    daily_df['dd_from_peak'] = (daily_df['close'] / daily_df['rolling_max_20'] - 1) * 100
    daily_df['daily_return'] = daily_df['close'].pct_change() * 100

    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates, warmup_bars=260)
    state_by_date = {r['date']: r for r in state_rows}

    def get_snapshot(date):
        row = state_by_date.get(date)
        if not row:
            return None
        d_state = row.get('daily')
        w_state = row.get('weekly')
        if not d_state or not w_state:
            return None
        di = daily_df.index.get_loc(date)

        pos_by_window = {}
        for w in [35, 45, 50, 60, 70]:
            if di >= w + 10:
                prices = daily_df['close'].iloc[di - w - 10:di + 1].values.astype(float)
                pos, r2, width = _lin_reg_channel(prices[-w:])
                pos_by_window[w] = pos

        spy_val = np.nan
        idx = spy_rsi.index.searchsorted(date)
        if 0 < idx <= len(spy_rsi):
            spy_val = float(spy_rsi.iloc[idx - 1])

        tw_rsi = _lookup_weekly(tsla_rsi_w, date)
        tw_sma = _lookup_weekly(tsla_rsi_w_sma, date)
        td_rsi = _lookup_daily(tsla_rsi_d, date)
        macd_h = _lookup_daily(macd_hist_d_s, date)
        sk = _lookup_daily(stoch_k_s, date)
        sd = _lookup_daily(stoch_d_s, date)
        dd_peak = daily_df.loc[date, 'dd_from_peak'] if date in daily_df.index else np.nan
        day_ret = daily_df.loc[date, 'daily_return'] if date in daily_df.index else np.nan

        return {
            'date': date,
            'close': daily_df.loc[date, 'close'],
            'd_pos_60': d_state['pos_pct'],
            'w_pos': w_state['pos_pct'],
            'd_turning': d_state['is_turning'],
            'w_turning': w_state['is_turning'],
            'pos_by_window': pos_by_window,
            'spy_rsi': spy_val,
            'tsla_rsi_d': td_rsi,
            'tsla_rsi_w': tw_rsi,
            'tsla_rsi_w_sma': tw_sma,
            'rsi_w_above_sma': tw_rsi > tw_sma if not np.isnan(tw_rsi) and not np.isnan(tw_sma) else False,
            'macd_hist_d': macd_h,
            'stoch_k': sk,
            'stoch_d': sd,
            'stoch_cross': sk > sd if not np.isnan(sk) and not np.isnan(sd) else False,
            'dd_from_peak': dd_peak,
            'daily_return': day_ret,
        }

    # ── Signal definitions ────────────────────────────────────────────────────
    SIGNALS = [
        ('S0: Baseline W=60',
         lambda s: s['d_pos_60'] < 0.35 and s['w_pos'] < 0.35),

        ('S2: W=70 baseline',
         lambda s: s['pos_by_window'].get(70, 1.0) < 0.35 and s['w_pos'] < 0.35),

        ('S4: W=45 + w_turning',
         lambda s: s['pos_by_window'].get(45, 1.0) < 0.35 and s['w_pos'] < 0.35 and s['w_turning']),

        ('S5: W=70 + w_turning',
         lambda s: s['pos_by_window'].get(70, 1.0) < 0.35 and s['w_pos'] < 0.35 and s['w_turning']),

        ('S7: w_turn + stoch K>D',
         lambda s: s['d_pos_60'] < 0.35 and s['w_pos'] < 0.35 and s['w_turning'] and s['stoch_cross']),

        ('S8: w_turn + stoch_K<20',
         lambda s: s['d_pos_60'] < 0.35 and s['w_pos'] < 0.35 and s['w_turning'] and s['stoch_k'] < 20),

        ('S9: w_turn + MACD loosening',
         lambda s: s['d_pos_60'] < 0.35 and s['w_pos'] < 0.35 and s['w_turning'] and -1 < s['macd_hist_d'] < 0),

        ('S10: dd<-10% + w_turn + rsi_w<50',
         lambda s: s['d_pos_60'] < 0.35 and s['w_pos'] < 0.35 and s['dd_from_peak'] < -10 and s['w_turning'] and s['tsla_rsi_w'] < 50),

        ('S12: dd<-10% + rsi_w<50 (no turn)',
         lambda s: s['d_pos_60'] < 0.35 and s['w_pos'] < 0.35 and s['dd_from_peak'] < -10 and s['tsla_rsi_w'] < 50),

        ('S13: daily_ret<-3% + rsi_d<30',
         lambda s: s['d_pos_60'] < 0.35 and s['w_pos'] < 0.35 and s['daily_return'] < -3 and s['tsla_rsi_d'] < 30),

        ('S14: dd<-10% + rsi_w 40-50',
         lambda s: s['d_pos_60'] < 0.35 and s['w_pos'] < 0.35 and s['dd_from_peak'] < -10 and 40 <= s['tsla_rsi_w'] < 50),

        ('S17: w_pos<0.02 + d_pos<0.35',
         lambda s: s['w_pos'] < 0.02 and s['d_pos_60'] < 0.35),

        ('S19: w_pos<0.05 + dd<-8% + (wt OR rsi)',
         lambda s: s['w_pos'] < 0.05 and s['d_pos_60'] < 0.35 and s['dd_from_peak'] < -8 and (s['w_turning'] or 40 <= s['tsla_rsi_w'] < 50)),
    ]

    # ── Backtest engine ───────────────────────────────────────────────────────
    def run_backtest(signal_fn, signal_name, max_hold=10, stop_pct=0.20,
                     capital=100_000, cooldown=10, start_year=2016, end_year=2025):
        trades = []
        in_trade = False
        entry_idx = None
        entry_price = None
        last_signal_date = None

        dates = daily_df.index
        for i, date in enumerate(dates):
            yr = date.year
            if yr < start_year or yr > end_year:
                continue
            if in_trade:
                hold = i - entry_idx
                stop_price = entry_price * (1.0 - stop_pct)
                low = daily_df.iloc[i]['low']
                hit_stop = low <= stop_price
                if hold >= max_hold or hit_stop:
                    exit_price = stop_price if hit_stop else daily_df.iloc[i]['close']
                    pnl = (exit_price / entry_price - 1.0) * capital
                    trades.append({
                        'entry_date': dates[entry_idx],
                        'exit_date': date,
                        'entry': entry_price,
                        'exit': exit_price,
                        'hold': hold,
                        'pnl': pnl,
                        'pnl_pct': (exit_price / entry_price - 1.0) * 100,
                        'year': dates[entry_idx].year,
                        'stop': hit_stop,
                    })
                    in_trade = False
            if in_trade:
                continue
            if last_signal_date and (date - last_signal_date).days < cooldown:
                continue
            snap = get_snapshot(date)
            if snap is None:
                continue
            try:
                fires = signal_fn(snap)
            except Exception:
                fires = False
            if not fires:
                continue
            if i + 1 >= len(dates):
                continue
            next_date = dates[i + 1]
            if next_date.year < start_year or next_date.year > end_year:
                continue
            in_trade = True
            entry_idx = i + 1
            entry_price = daily_df.iloc[i + 1]['open']
            last_signal_date = date

        if in_trade and entry_idx is not None:
            last_i = len(dates) - 1
            exit_price = daily_df.iloc[last_i]['close']
            pnl = (exit_price / entry_price - 1.0) * capital
            trades.append({
                'entry_date': dates[entry_idx],
                'exit_date': dates[last_i],
                'entry': entry_price,
                'exit': exit_price,
                'hold': last_i - entry_idx,
                'pnl': pnl,
                'pnl_pct': (exit_price / entry_price - 1.0) * 100,
                'year': dates[entry_idx].year,
                'stop': False,
            })

        n = len(trades)
        if n == 0:
            return {'name': signal_name, 'n': 0, 'wr': 0, 'pnl': 0, 'avg': 0, 'pf': 0, 'trades': []}
        wins = [t for t in trades if t['pnl'] > 0]
        gross_win = sum(t['pnl'] for t in wins)
        gross_los = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
        total_pnl = sum(t['pnl'] for t in trades)
        return {
            'name': signal_name, 'n': n,
            'wr': len(wins) / n, 'pnl': total_pnl, 'avg': total_pnl / n,
            'pf': gross_win / max(gross_los, 1e-6),
            'stops': sum(1 for t in trades if t['stop']),
            'trades': trades,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Run train / test / full
    # ══════════════════════════════════════════════════════════════════════════
    periods = [
        ('TRAIN (2016-2021)', 2016, 2021),
        ('TEST  (2022-2025)', 2022, 2025),
        ('FULL  (2016-2025)', 2016, 2025),
    ]

    all_results = {}
    for period_name, sy, ey in periods:
        results = []
        for name, fn in SIGNALS:
            r = run_backtest(fn, name, start_year=sy, end_year=ey)
            results.append(r)
        all_results[period_name] = results

    # ── Print comparison ──────────────────────────────────────────────────────
    print(f"\n{'='*140}")
    print("OUT-OF-SAMPLE VALIDATION: TRAIN (2016-2021) vs TEST (2022-2025)")
    print(f"Entry: next-day open  |  Hold: 10d  |  Stop: 20%  |  Capital: $100K/trade  |  Cooldown: 10d")
    print(f"{'='*140}")

    # Header
    print(f"\n  {'Signal':<40}", end='')
    for pn, _, _ in periods:
        print(f" | {'n':>3} {'WR':>5} {'P&L':>10} {'avg':>7} {'PF':>5}", end='')
    print(f" | {'OOS':>5}")
    print(f"  {'─'*135}")

    for i, (name, fn) in enumerate(SIGNALS):
        print(f"  {name:<40}", end='')
        train_r = all_results[periods[0][0]][i]
        test_r = all_results[periods[1][0]][i]
        full_r = all_results[periods[2][0]][i]

        for r in [train_r, test_r, full_r]:
            if r['n'] == 0:
                print(f" | {'0':>3} {'--':>5} {'--':>10} {'--':>7} {'--':>5}", end='')
            else:
                print(f" | {r['n']:>3} {r['wr']:>4.0%} ${r['pnl']:>8,.0f} ${r['avg']:>5,.0f} {r['pf']:>5.2f}", end='')

        # OOS verdict
        if train_r['n'] >= 2 and test_r['n'] >= 2:
            train_wr = train_r['wr']
            test_wr = test_r['wr']
            train_avg = train_r['avg']
            test_avg = test_r['avg']
            if test_wr >= 0.50 and test_avg > 0:
                verdict = 'PASS' if test_wr >= train_wr * 0.7 else 'WEAK'
            else:
                verdict = 'FAIL'
        elif test_r['n'] >= 1 and test_r['pnl'] > 0:
            verdict = 'OK*'
        elif test_r['n'] == 0:
            verdict = 'N/A'
        else:
            verdict = 'FAIL'
        print(f" | {verdict:>5}")

    # ── Per-year breakdown for key signals ────────────────────────────────────
    key_signals = ['S0: Baseline W=60', 'S2: W=70 baseline', 'S4: W=45 + w_turning',
                   'S5: W=70 + w_turning', 'S10: dd<-10% + w_turn + rsi_w<50',
                   'S19: w_pos<0.05 + dd<-8% + (wt OR rsi)']

    print(f"\n{'='*140}")
    print("PER-YEAR BREAKDOWN — KEY SIGNALS")
    print(f"{'='*140}")

    full_results = all_results[periods[2][0]]
    for i, (name, fn) in enumerate(SIGNALS):
        if name not in key_signals:
            continue
        r = full_results[i]
        if r['n'] == 0:
            continue
        print(f"\n  [{name}]  n={r['n']}  WR={r['wr']:.0%}  P&L=${r['pnl']:,.0f}  PF={r['pf']:.2f}")

        by_year = defaultdict(list)
        for t in r['trades']:
            by_year[t['year']].append(t)

        print(f"    {'Year':<6} {'n':>3} {'WR':>5} {'P&L':>10} {'avg':>8}  {'Period':>7}")
        print(f"    {'─'*50}")
        for yr in sorted(by_year):
            yr_trades = by_year[yr]
            yr_pnl = sum(t['pnl'] for t in yr_trades)
            yr_wr = sum(1 for t in yr_trades if t['pnl'] > 0) / len(yr_trades) if yr_trades else 0
            period = 'TRAIN' if yr <= 2021 else 'TEST'
            print(f"    {yr:<6} {len(yr_trades):>3} {yr_wr:>4.0%} ${yr_pnl:>8,.0f} ${yr_pnl/len(yr_trades):>6,.0f}  {period:>7}")

    # ── Trade-level detail for test period ────────────────────────────────────
    print(f"\n{'='*140}")
    print("TEST PERIOD TRADES (2022-2025) — KEY SIGNALS")
    print(f"{'='*140}")

    for i, (name, fn) in enumerate(SIGNALS):
        if name not in key_signals:
            continue
        test_r = all_results[periods[1][0]][i]
        if test_r['n'] == 0:
            continue
        print(f"\n  [{name}]  test n={test_r['n']}  WR={test_r['wr']:.0%}  P&L=${test_r['pnl']:,.0f}")
        print(f"  {'Entry':<12} {'Exit':<12} {'Entry$':>8} {'Exit$':>8} {'Hold':>4} {'P&L':>10} {'%':>7} {'Stop':>5}")
        print(f"  {'─'*75}")
        for t in test_r['trades']:
            print(f"  {t['entry_date'].strftime('%Y-%m-%d'):<12} "
                  f"{t['exit_date'].strftime('%Y-%m-%d'):<12} "
                  f"${t['entry']:>7.1f} ${t['exit']:>7.1f} "
                  f"{t['hold']:>4}d ${t['pnl']:>9,.0f} "
                  f"{t['pnl_pct']:>+6.1f}% {'STOP' if t['stop'] else '':>5}")

    print(f"\n{'='*140}")
    print("OOS VALIDATION COMPLETE")
    print(f"{'='*140}")


if __name__ == '__main__':
    main()
