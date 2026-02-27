#!/usr/bin/env python3
"""
Compare all 3 evolved signals (V2, V3, V4) on Feb 23-26 2026.
Shows what trades each would have made and forward returns.
"""
import os, sys, importlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.tf_state_backtest import (
    load_all_tfs, compute_daily_states, _norm_cols,
)
from v15.validation.bounce_timing import _compute_rsi

from v15.data.native_tf import fetch_native_tf

# ── Load signal functions ────────────────────────────────────────────────────
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openevolve_bounce')

def _load_signal(filename):
    path = os.path.join(base, filename)
    spec = importlib.util.spec_from_file_location(filename.replace('.py',''), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluate_bounce_signal

sig_v2 = _load_signal('best_program_v2.py')
sig_v3 = _load_signal('best_program_v3.py')
sig_v4 = _load_signal('best_program_v4.py')


def main():
    start, end = '2015-01-01', '2026-12-31'

    print("Loading data...")
    tf_data = load_all_tfs('data/TSLAMin.txt', start, end)
    daily_df = tf_data['daily']
    weekly_df = tf_data['weekly']

    # SPY RSI
    spy = _norm_cols(fetch_native_tf('SPY', 'daily', start, end))
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_rsi = _compute_rsi(spy['close'], 14)

    # TSLA weekly RSI + SMA
    tsla_rsi_w = _compute_rsi(weekly_df['close'], 14)
    tsla_rsi_sma = tsla_rsi_w.rolling(14, min_periods=1).mean()
    tsla_52w_sma = daily_df['close'].rolling(252, min_periods=60).mean()

    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates, warmup_bars=260)

    # Focus on Feb 23-26
    target_dates = [pd.Timestamp('2026-02-23'), pd.Timestamp('2026-02-24'),
                    pd.Timestamp('2026-02-25'), pd.Timestamp('2026-02-26')]

    feb_rows = [r for r in state_rows if r['date'] in target_dates]

    # Measure forward returns
    def fwd_return(date, days=5):
        """Simple forward return over N trading days."""
        idx = daily_df.index.get_loc(date)
        if idx + days >= len(daily_df):
            return None, None
        ref = daily_df.iloc[idx]['close']
        end_price = daily_df.iloc[idx + days]['close']
        # Also get max drawdown (worst intraday low)
        fwd_slice = daily_df.iloc[idx+1:idx+days+1]
        worst_low = fwd_slice['low'].min()
        return (end_price / ref - 1) * 100, (worst_low / ref - 1) * 100

    def _lookup_weekly(series, date):
        idx = series.index.searchsorted(date, side='right') - 1
        return float(series.iloc[idx]) if 0 <= idx < len(series) else np.nan

    print(f"\n{'='*100}")
    print(f"EVOLVED SIGNAL COMPARISON — Feb 23-26, 2026")
    print(f"{'='*100}")

    for row in feb_rows:
        date = row['date']
        close = daily_df.loc[date, 'close']

        # Build states dict
        states = {}
        for tf in ['5min', '1h', '4h', 'daily', 'weekly', 'monthly']:
            s = row.get(tf)
            if s:
                states[tf] = s

        # Get indicators
        idx = spy_rsi.index.searchsorted(date)
        spy_val = float(spy_rsi.iloc[idx - 1]) if 0 < idx <= len(spy_rsi) else 50.0
        tw_rsi = _lookup_weekly(tsla_rsi_w, date)
        tw_sma = _lookup_weekly(tsla_rsi_sma, date)

        if date in tsla_52w_sma.index:
            sma_val = tsla_52w_sma.loc[date]
            dist_52w = (close - sma_val) / sma_val if sma_val > 0 else 0.0
        else:
            dist_52w = 0.0

        # Forward returns
        fwd5, dd5 = fwd_return(date, 5)
        fwd10, dd10 = fwd_return(date, 10)

        d = states.get('daily', {})
        w = states.get('weekly', {})
        m = states.get('monthly', {})

        print(f"\n{'─'*100}")
        print(f"  {date.strftime('%a %b %d')}  |  TSLA ${close:.2f}  |  "
              f"D_pos={d.get('pos_pct',0):.2f}  W_pos={w.get('pos_pct',0):.2f}  "
              f"M_pos={m.get('pos_pct',0):.2f}")
        print(f"  SPY_RSI={spy_val:.1f}  |  TSLA_RSI_W={tw_rsi:.1f}  |  "
              f"RSI_SMA={tw_sma:.1f}  |  Dist_52w={dist_52w:+.1%}")
        print(f"  D_turning={d.get('is_turning',False)}  W_turning={w.get('is_turning',False)}")
        if fwd5 is not None:
            print(f"  Forward 5d: {fwd5:+.1f}%  (max DD: {dd5:+.1f}%)  |  "
                  f"Forward 10d: {fwd10:+.1f}%  (max DD: {dd10:+.1f}%)")
        print()

        # V2 — channel-only (no TSLA RSI inputs)
        r2 = sig_v2(states, spy_val)
        fire2 = r2.get('take_bounce', False)
        conf2 = r2.get('confidence', 0)
        delay2 = r2.get('delay_hours', 0)

        # V3 — TSLA RSI
        r3 = sig_v3(states, spy_val,
                     tsla_rsi_w=tw_rsi if not np.isnan(tw_rsi) else 50.0,
                     tsla_rsi_sma=tw_sma if not np.isnan(tw_sma) else 50.0,
                     dist_52w_sma=float(dist_52w))
        fire3 = r3.get('take_bounce', False)
        conf3 = r3.get('confidence', 0)
        delay3 = r3.get('delay_hours', 0)

        # V4 — directional
        r4 = sig_v4(states, spy_val,
                     tsla_rsi_w=tw_rsi if not np.isnan(tw_rsi) else 50.0,
                     tsla_rsi_sma=tw_sma if not np.isnan(tw_sma) else 50.0,
                     dist_52w_sma=float(dist_52w))
        dir4 = r4.get('direction', 'none')
        conf4 = r4.get('confidence', 0)
        delay4 = r4.get('delay_hours', 0)

        # Display
        status2 = f"{'LONG' if fire2 else 'NO TRADE':<10} conf={conf2:.2f}  delay={delay2}h"
        status3 = f"{'LONG' if fire3 else 'NO TRADE':<10} conf={conf3:.2f}  delay={delay3}h"
        status4 = f"{dir4.upper():<10} conf={conf4:.2f}  delay={delay4}h"

        print(f"  V2 (channel):    {status2}")
        print(f"  V3 (TSLA RSI):   {status3}")
        print(f"  V4 (direction):  {status4}")

    print(f"\n{'='*100}")
    print("LEGEND: Forward returns are from close on signal date.")
    print("        Positive = price went up, Negative = price went down.")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
