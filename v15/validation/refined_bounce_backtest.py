#!/usr/bin/env python3
"""
Refined bounce signal backtest — consolidates all findings from the deep studies.

Tests refined signal variants as proper backtests with PnL, and specifically
shows what would have happened Feb 23-26, 2026.

Key findings incorporated:
  - W=45 or W=70 daily channel outperforms W=60
  - w_turning is the strongest factor
  - MACD histogram loosening (-1 to 0) is strong
  - dd_from_peak < -10% adds predictive value beyond RSI
  - Weekly RSI 40-50 is the sweet spot
  - Stochastic K < 20 + w_turning works well
  - "Recovering" zone (pos 0.15-0.35) beats "at bottom" (<0.15)
"""
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.tf_state_backtest import (
    load_all_tfs, compute_daily_states, _norm_cols,
    _lin_reg_channel, _momentum_is_turning_up, TF_WINDOWS,
)
from v15.validation.bounce_timing import _compute_rsi
from v15.data.native_tf import fetch_native_tf
from v15.features.utils import (
    calc_macd, calc_bollinger_bands, calc_stochastic, calc_atr,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

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

    # ── Precompute all indicators ─────────────────────────────────────────────
    print("Computing indicators...")

    # SPY RSI
    spy = _norm_cols(fetch_native_tf('SPY', 'daily', start, end))
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_rsi = _compute_rsi(spy['close'], 14)

    # TSLA RSI
    tsla_rsi_d = _compute_rsi(daily_df['close'], 14)
    tsla_rsi_w = _compute_rsi(weekly_df['close'], 14)
    tsla_rsi_w_sma = tsla_rsi_w.rolling(14, min_periods=1).mean()

    # MACD (daily)
    d_close = daily_df['close'].values
    _, _, macd_hist_d = calc_macd(d_close)
    macd_hist_d_s = pd.Series(macd_hist_d, index=daily_df.index)

    # Stochastic (daily)
    stoch_k, stoch_d = calc_stochastic(
        daily_df['high'].values, daily_df['low'].values, d_close
    )
    stoch_k_s = pd.Series(stoch_k, index=daily_df.index)
    stoch_d_s = pd.Series(stoch_d, index=daily_df.index)

    # Drawdown from 20-day peak
    daily_df = daily_df.copy()
    daily_df['rolling_max_20'] = daily_df['high'].rolling(20, min_periods=1).max()
    daily_df['dd_from_peak'] = (daily_df['close'] / daily_df['rolling_max_20'] - 1) * 100
    daily_df['daily_return'] = daily_df['close'].pct_change() * 100

    # TF states
    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates, warmup_bars=260)
    state_by_date = {r['date']: r for r in state_rows}

    # ── Build enriched daily snapshot for each trading date ────────────────────
    print("Building enriched snapshots...")

    def get_snapshot(date):
        """Get all indicators for a date."""
        row = state_by_date.get(date)
        if not row:
            return None
        d_state = row.get('daily')
        w_state = row.get('weekly')
        if not d_state or not w_state:
            return None

        di = daily_df.index.get_loc(date)

        # Multi-window daily pos_pct
        pos_by_window = {}
        turning_by_window = {}
        for w in [35, 45, 50, 60, 70]:
            if di >= w + 10:
                prices = daily_df['close'].iloc[di - w - 10:di + 1].values.astype(float)
                pos, r2, width = _lin_reg_channel(prices[-w:])
                turn = _momentum_is_turning_up(prices, lookback=min(10, len(prices) - 2))
                pos_by_window[w] = pos
                turning_by_window[w] = turn

        # Indicators
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
            'd_state': d_state,
            'w_state': w_state,
            'm_state': row.get('monthly'),
            'd_pos_60': d_state['pos_pct'],
            'w_pos': w_state['pos_pct'],
            'd_turning': d_state['is_turning'],
            'w_turning': w_state['is_turning'],
            'pos_by_window': pos_by_window,
            'turning_by_window': turning_by_window,
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

    # ── Define refined signals ────────────────────────────────────────────────

    SIGNALS = []

    # --- Baseline (current system) ---
    def sig_baseline(snap):
        """Current: d_pos<0.35 + w_pos<0.35 (W=60)."""
        return snap['d_pos_60'] < 0.35 and snap['w_pos'] < 0.35
    SIGNALS.append(('S0: Baseline (W=60, d+w<0.35)', sig_baseline))

    # --- Window variants ---
    def _make_window_sig(w, threshold=0.35):
        def sig(snap):
            d_pos = snap['pos_by_window'].get(w, 1.0)
            return d_pos < threshold and snap['w_pos'] < threshold
        return sig

    SIGNALS.append(('S1: W=45, d+w<0.35', _make_window_sig(45)))
    SIGNALS.append(('S2: W=70, d+w<0.35', _make_window_sig(70)))

    # --- w_turning required ---
    def sig_w_turn_w60(snap):
        return snap['d_pos_60'] < 0.35 and snap['w_pos'] < 0.35 and snap['w_turning']
    SIGNALS.append(('S3: W=60 + w_turning', sig_w_turn_w60))

    def sig_w_turn_w45(snap):
        d_pos = snap['pos_by_window'].get(45, 1.0)
        return d_pos < 0.35 and snap['w_pos'] < 0.35 and snap['w_turning']
    SIGNALS.append(('S4: W=45 + w_turning', sig_w_turn_w45))

    def sig_w_turn_w70(snap):
        d_pos = snap['pos_by_window'].get(70, 1.0)
        return d_pos < 0.35 and snap['w_pos'] < 0.35 and snap['w_turning']
    SIGNALS.append(('S5: W=70 + w_turning', sig_w_turn_w70))

    # --- Best combos from studies ---
    def sig_wt_rsi_w_4050(snap):
        """w_turning + weekly RSI 40-50."""
        d_pos = snap['pos_by_window'].get(45, snap['d_pos_60'])
        return (d_pos < 0.35 and snap['w_pos'] < 0.35 and
                snap['w_turning'] and 40 <= snap['tsla_rsi_w'] < 50)
    SIGNALS.append(('S6: W=45 + w_turn + rsi_w 40-50', sig_wt_rsi_w_4050))

    def sig_wt_stoch_cross(snap):
        """w_turning + stochastic K > D."""
        return (snap['d_pos_60'] < 0.35 and snap['w_pos'] < 0.35 and
                snap['w_turning'] and snap['stoch_cross'])
    SIGNALS.append(('S7: w_turn + stoch K>D', sig_wt_stoch_cross))

    def sig_wt_bb_low(snap):
        """w_turning + stochastic K < 20 (very oversold)."""
        return (snap['d_pos_60'] < 0.35 and snap['w_pos'] < 0.35 and
                snap['w_turning'] and snap['stoch_k'] < 20)
    SIGNALS.append(('S8: w_turn + stoch_K<20', sig_wt_bb_low))

    def sig_macd_loosening(snap):
        """MACD hist -1 to 0 (loosening) + w_turning."""
        return (snap['d_pos_60'] < 0.35 and snap['w_pos'] < 0.35 and
                snap['w_turning'] and -1 < snap['macd_hist_d'] < 0)
    SIGNALS.append(('S9: w_turn + MACD loosening', sig_macd_loosening))

    def sig_dd_peak_wt(snap):
        """DD from peak > -10% + w_turning + rsi_w < 50."""
        return (snap['d_pos_60'] < 0.35 and snap['w_pos'] < 0.35 and
                snap['dd_from_peak'] < -10 and snap['w_turning'] and
                snap['tsla_rsi_w'] < 50)
    SIGNALS.append(('S10: dd_peak<-10% + w_turn + rsi_w<50', sig_dd_peak_wt))

    def sig_dd_peak_any_turn(snap):
        """DD from peak > -10% + any turning + rsi_w < 50."""
        return (snap['d_pos_60'] < 0.35 and snap['w_pos'] < 0.35 and
                snap['dd_from_peak'] < -10 and
                (snap['w_turning'] or snap['d_turning']) and
                snap['tsla_rsi_w'] < 50)
    SIGNALS.append(('S11: dd_peak<-10% + any_turn + rsi_w<50', sig_dd_peak_any_turn))

    def sig_dd_peak_no_turn(snap):
        """DD from peak > -10% + rsi_w < 50 (no turning required)."""
        return (snap['d_pos_60'] < 0.35 and snap['w_pos'] < 0.35 and
                snap['dd_from_peak'] < -10 and snap['tsla_rsi_w'] < 50)
    SIGNALS.append(('S12: dd_peak<-10% + rsi_w<50 (no turn)', sig_dd_peak_no_turn))

    def sig_daily_drop_rsi(snap):
        """Daily drop < -3% + rsi_d < 30."""
        return (snap['d_pos_60'] < 0.35 and snap['w_pos'] < 0.35 and
                snap['daily_return'] < -3 and snap['tsla_rsi_d'] < 30)
    SIGNALS.append(('S13: daily_ret<-3% + rsi_d<30', sig_daily_drop_rsi))

    def sig_dd_peak_rsi_w_4050(snap):
        """DD from peak > -10% + rsi_w 40-50 (the sweet spot zone)."""
        return (snap['d_pos_60'] < 0.35 and snap['w_pos'] < 0.35 and
                snap['dd_from_peak'] < -10 and
                40 <= snap['tsla_rsi_w'] < 50)
    SIGNALS.append(('S14: dd_peak<-10% + rsi_w 40-50', sig_dd_peak_rsi_w_4050))

    # --- Relaxed signals (would catch Feb 23) ---
    def sig_relaxed_dd(snap):
        """DD from peak > -10% + w_pos < 0.10 (extremely oversold weekly)."""
        return (snap['d_pos_60'] < 0.35 and snap['w_pos'] < 0.10 and
                snap['dd_from_peak'] < -10)
    SIGNALS.append(('S15: dd_peak<-10% + w_pos<0.10', sig_relaxed_dd))

    def sig_relaxed_dd_rsi(snap):
        """DD from peak > -8% + w_pos < 0.05 + rsi_w 40-50."""
        return (snap['w_pos'] < 0.05 and
                snap['dd_from_peak'] < -8 and
                40 <= snap['tsla_rsi_w'] < 50)
    SIGNALS.append(('S16: w_pos<0.05 + dd<-8% + rsi_w 40-50', sig_relaxed_dd_rsi))

    def sig_extreme_weekly_oversold(snap):
        """w_pos == 0 (at absolute bottom of weekly channel)."""
        return snap['w_pos'] < 0.02 and snap['d_pos_60'] < 0.35
    SIGNALS.append(('S17: w_pos<0.02 + d_pos<0.35', sig_extreme_weekly_oversold))

    def sig_w0_rsi_sweet(snap):
        """w_pos near 0 + weekly RSI in sweet spot."""
        return (snap['w_pos'] < 0.05 and snap['d_pos_60'] < 0.35 and
                40 <= snap['tsla_rsi_w'] < 52)
    SIGNALS.append(('S18: w_pos<0.05 + rsi_w 40-52', sig_w0_rsi_sweet))

    def sig_w0_dd_turn_or_rsi(snap):
        """w_pos near 0 + dd_peak<-8% + (w_turning OR rsi_w 40-50)."""
        return (snap['w_pos'] < 0.05 and snap['d_pos_60'] < 0.35 and
                snap['dd_from_peak'] < -8 and
                (snap['w_turning'] or 40 <= snap['tsla_rsi_w'] < 50))
    SIGNALS.append(('S19: w_pos<0.05 + dd<-8% + (wt OR rsi)', sig_w0_dd_turn_or_rsi))

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

            # Exit check
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

            # Cooldown
            if last_signal_date and (date - last_signal_date).days < cooldown:
                continue

            # Signal check
            snap = get_snapshot(date)
            if snap is None:
                continue

            try:
                fires = signal_fn(snap)
            except Exception:
                fires = False

            if not fires:
                continue

            # Enter next day open
            if i + 1 >= len(dates):
                continue
            next_date = dates[i + 1]
            if next_date.year < start_year or next_date.year > end_year:
                continue

            in_trade = True
            entry_idx = i + 1
            entry_price = daily_df.iloc[i + 1]['open']
            last_signal_date = date

        # Close open trade
        if in_trade and entry_idx is not None:
            exit_price = daily_df.iloc[-1]['close']
            pnl = (exit_price / entry_price - 1.0) * capital
            trades.append({
                'entry_date': dates[entry_idx],
                'exit_date': dates[-1],
                'entry': entry_price,
                'exit': exit_price,
                'hold': len(dates) - 1 - entry_idx,
                'pnl': pnl,
                'pnl_pct': (exit_price / entry_price - 1.0) * 100,
                'year': dates[entry_idx].year,
                'stop': False,
            })

        n = len(trades)
        if n == 0:
            return {'name': signal_name, 'n': 0, 'wr': 0, 'pnl': 0, 'avg': 0,
                    'pf': 0, 'trades': []}

        wins = [t for t in trades if t['pnl'] > 0]
        gross_win = sum(t['pnl'] for t in wins)
        gross_los = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
        total_pnl = sum(t['pnl'] for t in trades)

        return {
            'name': signal_name,
            'n': n,
            'wr': len(wins) / n,
            'pnl': total_pnl,
            'avg': total_pnl / n,
            'pf': gross_win / max(gross_los, 1e-6),
            'stops': sum(1 for t in trades if t['stop']),
            'trades': trades,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: Run all signal backtests
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print("REFINED BOUNCE SIGNAL BACKTEST")
    print(f"Period: 2016-2025 (IS)  |  Entry: next-day open  |  Hold: 10d  |  Stop: 20%  |  Capital: $100K/trade")
    print(f"{'='*130}")

    results = []
    for name, fn in SIGNALS:
        r = run_backtest(fn, name)
        results.append(r)

    # Summary table
    print(f"\n  {'Signal':<50} {'n':>4} {'WR':>5} {'P&L':>12} {'avg/tr':>8} {'PF':>5} {'stops':>6}")
    print(f"  {'─'*95}")

    results_sorted = sorted(results, key=lambda r: r['pnl'], reverse=True)
    for r in results_sorted:
        if r['n'] == 0:
            print(f"  {r['name']:<50} {'0':>4}")
            continue
        print(f"  {r['name']:<50} {r['n']:>4} {r['wr']:>4.0%} "
              f"${r['pnl']:>10,.0f} ${r['avg']:>6,.0f} {r['pf']:>5.2f} {r['stops']:>6}")

    # Per-year for top 5
    top5 = [r for r in results_sorted if r['n'] >= 3][:5]
    if top5:
        print(f"\n{'='*130}")
        print("PER-YEAR BREAKDOWN — TOP 5")
        print(f"{'='*130}")
        for r in top5:
            print(f"\n  [{r['name']}]  n={r['n']}  WR={r['wr']:.0%}  P&L=${r['pnl']:,.0f}")
            by_year = defaultdict(list)
            for t in r['trades']:
                by_year[t['year']].append(t)
            for yr in sorted(by_year):
                yr_trades = by_year[yr]
                yr_pnl = sum(t['pnl'] for t in yr_trades)
                yr_wr = sum(1 for t in yr_trades if t['pnl'] > 0) / len(yr_trades)
                print(f"    {yr}: {len(yr_trades):>2} trades  "
                      f"WR={yr_wr:.0%}  P&L=${yr_pnl:>8,.0f}")

    # Trade list for top 3
    top3 = [r for r in results_sorted if r['n'] >= 3][:3]
    if top3:
        print(f"\n{'='*130}")
        print("TRADE LIST — TOP 3 SIGNALS")
        print(f"{'='*130}")
        for r in top3:
            print(f"\n  [{r['name']}]")
            print(f"  {'Entry Date':<12} {'Exit Date':<12} {'Entry':>8} {'Exit':>8} "
                  f"{'Hold':>4} {'P&L':>10} {'P&L%':>7} {'Stop':>5}")
            print(f"  {'─'*75}")
            for t in r['trades']:
                print(f"  {t['entry_date'].strftime('%Y-%m-%d'):<12} "
                      f"{t['exit_date'].strftime('%Y-%m-%d'):<12} "
                      f"${t['entry']:>7.1f} ${t['exit']:>7.1f} "
                      f"{t['hold']:>4}d ${t['pnl']:>9,.0f} "
                      f"{t['pnl_pct']:>+6.1f}% {'STOP' if t['stop'] else '':>5}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: Feb 23-26 2026 deep dive
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print("FEB 23-26, 2026 — WHAT EACH SIGNAL WOULD DO")
    print(f"{'='*130}")

    target_dates = [pd.Timestamp('2026-02-20'), pd.Timestamp('2026-02-23'),
                    pd.Timestamp('2026-02-24'), pd.Timestamp('2026-02-25'),
                    pd.Timestamp('2026-02-26'), pd.Timestamp('2026-02-27')]

    # OHLC context
    print(f"\n  TSLA Price Context:")
    for d in target_dates:
        if d in daily_df.index:
            row = daily_df.loc[d]
            ret = daily_df.loc[d, 'daily_return'] if 'daily_return' in daily_df.columns else 0
            dd = daily_df.loc[d, 'dd_from_peak'] if 'dd_from_peak' in daily_df.columns else 0
            print(f"    {d.strftime('%a %b %d')}  O={row['open']:>7.2f}  H={row['high']:>7.2f}  "
                  f"L={row['low']:>7.2f}  C={row['close']:>7.2f}  "
                  f"ret={ret:>+5.1f}%  dd_peak={dd:>+5.1f}%")

    # Signal evaluation per date
    for d in target_dates:
        if d not in daily_df.index:
            continue
        snap = get_snapshot(d)
        if snap is None:
            continue

        print(f"\n  {'─'*120}")
        print(f"  {d.strftime('%a %b %d')}  TSLA ${snap['close']:.2f}")
        print(f"    D_pos(60)={snap['d_pos_60']:.3f}  W_pos={snap['w_pos']:.3f}  "
              f"D_turn={snap['d_turning']}  W_turn={snap['w_turning']}")
        print(f"    RSI_d={snap['tsla_rsi_d']:.1f}  RSI_w={snap['tsla_rsi_w']:.1f}  "
              f"RSI_w_sma={snap['tsla_rsi_w_sma']:.1f}  SPY={snap['spy_rsi']:.1f}")
        print(f"    MACD_hist={snap['macd_hist_d']:+.2f}  "
              f"Stoch_K={snap['stoch_k']:.1f}  Stoch_D={snap['stoch_d']:.1f}  "
              f"K>D={snap['stoch_cross']}")
        print(f"    DD_from_peak={snap['dd_from_peak']:+.1f}%  "
              f"Daily_ret={snap['daily_return']:+.1f}%")

        # Multi-window pos_pct
        window_str = "  ".join(f"W{w}={p:.3f}" for w, p in sorted(snap['pos_by_window'].items()))
        print(f"    Multi-window: {window_str}")

        # Which signals fire?
        print(f"\n    Signal evaluations:")
        for name, fn in SIGNALS:
            try:
                fires = fn(snap)
            except Exception:
                fires = False
            marker = 'FIRE' if fires else '    '
            print(f"      [{marker}] {name}")

        # If we entered next day
        di = daily_df.index.get_loc(d)
        if di + 1 < len(daily_df):
            entry = daily_df.iloc[di + 1]['open']
            print(f"\n    If entered next-day open: ${entry:.2f}")
            for fwd_days in [1, 2, 3, 5]:
                if di + 1 + fwd_days < len(daily_df):
                    fwd_close = daily_df.iloc[di + 1 + fwd_days]['close']
                    fwd_peak = daily_df.iloc[di+2:di+2+fwd_days]['high'].max()
                    fwd_dd = daily_df.iloc[di+2:di+2+fwd_days]['low'].min()
                    print(f"      {fwd_days}d: close={fwd_close:.2f} ({(fwd_close/entry-1)*100:+.1f}%)  "
                          f"peak={fwd_peak:.2f} ({(fwd_peak/entry-1)*100:+.1f}%)  "
                          f"trough={fwd_dd:.2f} ({(fwd_dd/entry-1)*100:+.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3: Compare hold periods for top signals
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print("HOLD PERIOD SENSITIVITY — TOP 5 SIGNALS")
    print(f"{'='*130}")

    top_signal_names = [r['name'] for r in results_sorted if r['n'] >= 3][:5]
    top_signal_fns = {name: fn for name, fn in SIGNALS if name in top_signal_names}

    hold_periods = [5, 7, 10, 15, 20]
    print(f"\n  {'Signal':<50} " + "  ".join(f"h={h:>2}d" for h in hold_periods))
    print(f"  {'─'*100}")

    for name in top_signal_names:
        fn = top_signal_fns.get(name)
        if not fn:
            continue
        cells = []
        for h in hold_periods:
            r = run_backtest(fn, name, max_hold=h)
            if r['n'] > 0:
                cells.append(f"${r['pnl']:>8,.0f} ({r['wr']:.0%})")
            else:
                cells.append(f"{'--':>15}")
        print(f"  {name:<50} " + "  ".join(cells))

    print(f"\n{'='*130}")
    print("BACKTEST COMPLETE")
    print(f"{'='*130}")


if __name__ == '__main__':
    main()
