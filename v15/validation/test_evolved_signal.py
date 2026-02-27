#!/usr/bin/env python3
"""
Quick test: run the OE-evolved bounce signal on recent dates
and show what trades it would've made, with forward returns.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.tf_state_backtest import (
    load_all_tfs, compute_daily_states, _norm_cols,
)
from v15.validation.bounce_timing import (
    _compute_rsi, _load_spy_rsi, _build_hourly_series, measure_forward,
    TARGETS, _fmt_hours,
)

# Import evolved signal
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openevolve_bounce'))
from best_program import evaluate_bounce_signal

# Also import seed for comparison
from initial_program import evaluate_bounce_signal as seed_signal


def main():
    start, end = '2015-01-01', '2026-12-31'

    print("Loading data...")
    tf_data = load_all_tfs('data/TSLAMin.txt', start, end)
    daily_df = tf_data['daily']
    spy_rsi = _load_spy_rsi(start, end)
    hourly = _build_hourly_series(tf_data)

    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates, warmup_bars=260)

    # Focus on Feb 2026
    feb_rows = [r for r in state_rows
                if r['date'] >= pd.Timestamp('2026-02-01')
                and r['date'] <= pd.Timestamp('2026-02-28')]

    print(f"\n{'='*90}")
    print(f"EVOLVED SIGNAL TEST — Feb 2026 ({len(feb_rows)} trading days)")
    print(f"{'='*90}")

    # Table header
    print(f"\n{'Date':<12} {'Close':>7} {'D_pos':>5} {'W_pos':>5} {'M_pos':>5} "
          f"{'SPY':>5} | {'Evol':>6} {'Conf':>5} {'Dly':>3} | "
          f"{'Seed':>6} {'Conf':>5} | "
          + ''.join(f'+{int(t*100)}%  ' for t in TARGETS) + f'{'MaxDD':>6}')
    print('-' * 120)

    for row in feb_rows:
        date = row['date']
        states = {}
        for tf in ['5min', '1h', '4h', 'daily', 'weekly', 'monthly']:
            s = row.get(tf)
            if s:
                states[tf] = s

        # SPY RSI
        idx = spy_rsi.index.searchsorted(date)
        rsi_val = float(spy_rsi.iloc[idx - 1]) if 0 < idx <= len(spy_rsi) else 50.0

        # Run both signals
        evol = evaluate_bounce_signal(states, rsi_val)
        seed = seed_signal(states, rsi_val)

        # Get price
        if date not in daily_df.index:
            di = daily_df.index.searchsorted(date)
            if di >= len(daily_df):
                continue
            date = daily_df.index[di]
        close = daily_df.loc[date, 'close']

        # Pos_pct
        d_pos = states.get('daily', {}).get('pos_pct', np.nan)
        w_pos = states.get('weekly', {}).get('pos_pct', np.nan)
        m_pos = states.get('monthly', {}).get('pos_pct', np.nan)
        d_mt = '*' if states.get('daily', {}).get('is_turning') else ' '
        w_mt = '*' if states.get('weekly', {}).get('is_turning') else ' '
        m_mt = '*' if states.get('monthly', {}).get('is_turning') else ' '

        # Forward returns (for all days, to see context)
        fwd = measure_forward(close, date, daily_df, hourly)

        evol_tag = 'LONG' if evol['take_bounce'] else '  -- '
        seed_tag = 'LONG' if seed['take_bounce'] else '  -- '

        fwd_str = ''
        for t in TARGETS:
            fwd_str += f'{_fmt_hours(fwd["hours_to_target"][t]):>5}  '

        line = (f"{date.strftime('%Y-%m-%d'):<12} ${close:>6,.0f} "
                f"{d_pos:>4.2f}{d_mt} {w_pos:>4.2f}{w_mt} {m_pos:>4.2f}{m_mt} "
                f"{rsi_val:>5.1f} | "
                f"{evol_tag:>6} {evol['confidence']:>5.2f} {evol['delay_hours']:>3} | "
                f"{seed_tag:>6} {seed['confidence']:>5.2f} | "
                f"{fwd_str}"
                f"{fwd['max_drawdown']:>+5.1%}")
        print(line)

    # Summary of signal dates
    print(f"\n{'='*90}")
    print("EVOLVED SIGNAL TRADES:")
    print(f"{'='*90}")
    for row in feb_rows:
        date = row['date']
        states = {tf: row.get(tf) for tf in ['5min', '1h', '4h', 'daily', 'weekly', 'monthly'] if row.get(tf)}
        idx = spy_rsi.index.searchsorted(date)
        rsi_val = float(spy_rsi.iloc[idx - 1]) if 0 < idx <= len(spy_rsi) else 50.0
        evol = evaluate_bounce_signal(states, rsi_val)
        if evol['take_bounce']:
            if date not in daily_df.index:
                di = daily_df.index.searchsorted(date)
                if di >= len(daily_df):
                    continue
                date = daily_df.index[di]
            close = daily_df.loc[date, 'close']
            fwd = measure_forward(close, date, daily_df, hourly)
            pos_size = 100_000 * evol['confidence']
            # Compute 10-day return
            next_day = date + pd.Timedelta(days=1)
            end_d = date + pd.Timedelta(days=20)
            fwd_daily = daily_df.loc[next_day:end_d].head(10)
            fwd_ret = (fwd_daily.iloc[-1]['close'] / close - 1.0) if len(fwd_daily) > 0 else 0.0
            pnl = pos_size * fwd_ret

            print(f"\n  {date.strftime('%Y-%m-%d')} close=${close:,.0f}")
            print(f"    Confidence: {evol['confidence']:.2f}  Delay: {evol['delay_hours']}h")
            print(f"    Position:   ${pos_size:,.0f}")
            print(f"    10-day fwd: {fwd_ret:+.1%}  P&L: ${pnl:+,.0f}")
            print(f"    Time to +1%: {_fmt_hours(fwd['hours_to_target'][0.01])}  "
                  f"+2%: {_fmt_hours(fwd['hours_to_target'][0.02])}  "
                  f"+3%: {_fmt_hours(fwd['hours_to_target'][0.03])}  "
                  f"+5%: {_fmt_hours(fwd['hours_to_target'][0.05])}")
            print(f"    Max DD before bounce: {fwd['max_drawdown']:+.1%}")


if __name__ == '__main__':
    main()
