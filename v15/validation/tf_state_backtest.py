#!/usr/bin/env python3
"""
Multi-TF Dashboard Signal Backtest

Tests the market insights panel signals historically across all 5 TFs:
5min, 1h, 4h, daily, weekly.

Signal components computed per-bar (matching dashboard logic):
  MT   = momentum_is_turning  (sell-off decelerating — 2nd derivative turns +)
  ATB  = at_channel_bottom    (position < 0.20 in linear regression channel)
  NB   = near_bottom          (position < 0.35)
  STR  = channel stress       (energy_ratio > 2.5)
  CON  = bullish consensus    (N of 5 TFs with position < 0.35)

Combinations tested (all 5 TFs):
  1.  weekly_MT
  2.  weekly_MT + 1h_ATB
  3.  weekly_MT + (1h_ATB OR 4h_ATB)
  4.  weekly_MT + consensus_3plus
  5.  weekly_MT + consensus_4plus
  6.  consensus_4plus
  7.  consensus_5 (all 5 TFs bullish)
  8.  (weekly_MT OR daily_MT) + consensus_3plus
  9.  3plus_TFs_MT (3+ TFs simultaneously turning)
  10. weekly_MT + daily_MT + 1h_NB
  11. weekly_MT + daily_MT
  12. 5min_MT + 1h_ATB + weekly_NB
  13. weekly_MT + 1h_STR
  14. weekly_MT + 4h_ATB
  15. daily_MT + 1h_ATB
  16. weekly_NB + daily_MT + consensus_3plus
  17. ALL: weekly_MT + daily_MT + 1h_ATB + consensus_4plus

Entry:  next-day open after signal fires (no re-entry while in trade)
Exit:   max_hold_days OR stop_pct stop loss
Capital: $100K per trade (constant position size for comparability)

Usage:
    python3 -m v15.validation.tf_state_backtest
    python3 -m v15.validation.tf_state_backtest --tsla data/TSLAMin.txt --spy data/SPYMin.txt
    python3 -m v15.validation.tf_state_backtest --hold 7 --stop 0.15
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── Channel window sizes per TF ─────────────────────────────────────────────
TF_WINDOWS = {
    '5min':  78,   # ~2 full trading days of 5-min bars
    '1h':    60,   # 60 hours ≈ 12 trading days
    '4h':    30,   # 30 × 4h bars ≈ 6 weeks
    'daily': 60,   # 60 trading days ≈ 3 months
    'weekly': 50,  # 50 weeks ≈ 1 year
}

# ── TF state helpers ─────────────────────────────────────────────────────────

def _lin_reg_channel(prices: np.ndarray):
    """Linear regression channel.  Returns (pos_pct, r2, width)."""
    n = len(prices)
    if n < 10:
        return 0.5, 0.0, 0.0
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, prices, 1)
    trend = slope * x + intercept
    residuals = prices - trend
    std = residuals.std()
    if std == 0:
        return 0.5, 1.0, 0.0
    lower = trend[-1] - 2.0 * std
    upper = trend[-1] + 2.0 * std
    width = upper - lower
    pos_pct = float(np.clip((prices[-1] - lower) / width, 0.0, 1.0))
    var_total = prices.var()
    r2 = float(1.0 - residuals.var() / var_total) if var_total > 0 else 0.0
    return pos_pct, max(r2, 0.0), width


def _momentum_is_turning_up(prices: np.ndarray, lookback: int = 10) -> bool:
    """
    True when sell-off is exhausting: price falling but deceleration detected.
    Matches dashboard logic: (momentum < 0) AND (acceleration > 0).
    """
    if len(prices) < lookback + 2:
        return False
    p = prices[-lookback:].astype(float)
    roc = np.diff(p) / (p[:-1] + 1e-10)          # % changes
    if len(roc) < 2:
        return False
    accel = np.diff(roc)                           # 2nd derivative
    current_mom = roc[-1]
    current_accel = accel[-1]
    return bool(current_mom < 0 and current_accel > 0)


def _compute_tf_state(prices: np.ndarray, channel_window: int):
    """
    Compute key TF state metrics matching the dashboard market insights panel.

    Returns dict with:
      pos_pct      - 0 = lower boundary, 1 = upper boundary
      r2           - channel fit quality
      is_turning   - momentum_is_turning (sell-off exhaustion)
      energy_ratio - total_energy / binding_energy proxy
      at_bottom    - pos_pct < 0.20
      near_bottom  - pos_pct < 0.35
      at_top       - pos_pct > 0.80
      stressed     - energy_ratio > 2.5
    """
    if len(prices) < channel_window:
        return None
    window = prices[-channel_window:]
    pos_pct, r2, width = _lin_reg_channel(window)
    is_turning = _momentum_is_turning_up(prices, lookback=min(10, len(prices) - 2))

    # Energy proxy matching dashboard physics:
    # potential = distance from channel center (0 at center, 1 at boundary)
    # kinetic   = speed of recent price moves relative to channel width
    potential = abs(pos_pct - 0.5) * 2.0
    p7 = prices[-7:].astype(float)
    recent_returns = np.abs(np.diff(p7) / (p7[:-1] + 1e-10))
    kinetic = float(recent_returns.mean()) * 20.0  # scale: ~0.002/bar → ~0.04
    binding = max(r2, 0.05)                         # channel fit = binding strength
    energy_ratio = (potential + kinetic) / binding

    return {
        'pos_pct':      pos_pct,
        'r2':           r2,
        'is_turning':   is_turning,
        'energy_ratio': energy_ratio,
        'at_bottom':    pos_pct < 0.20,
        'near_bottom':  pos_pct < 0.35,
        'at_top':       pos_pct > 0.80,
        'stressed':     energy_ratio > 2.5,
    }


# ── Signal combination definitions ──────────────────────────────────────────

SIGNALS = [
    # ── A) Weekly/daily-based (macro first, intraday confirmation) ────────────

    ('A1 weekly_MT',
     lambda s: _mt(s, 'weekly')),

    ('A2 weekly_MT + 1h_ATB',
     lambda s: _mt(s, 'weekly') and s['1h'] and s['1h']['at_bottom']),

    ('A3 weekly_MT + (1h OR 4h)_ATB',
     lambda s: _mt(s, 'weekly') and (
         (s['1h'] and s['1h']['at_bottom']) or
         (s['4h'] and s['4h']['at_bottom']))),

    ('A4 weekly_MT + consensus_3+',
     lambda s: _mt(s, 'weekly') and _count_near_bottom(s) >= 3),

    ('A5 weekly_MT + consensus_4+',
     lambda s: _mt(s, 'weekly') and _count_near_bottom(s) >= 4),

    ('A6 weekly_MT + daily_MT',
     lambda s: _mt(s, 'weekly') and _mt(s, 'daily')),

    ('A7 weekly_MT + daily_MT + 1h_NB',
     lambda s: _mt(s, 'weekly') and _mt(s, 'daily') and
               s['1h'] and s['1h']['near_bottom']),

    ('A8 weekly_MT + daily_MT + 1h_ATB',
     lambda s: _mt(s, 'weekly') and _mt(s, 'daily') and
               s['1h'] and s['1h']['at_bottom']),

    ('A9 weekly_MT + 4h_ATB',
     lambda s: _mt(s, 'weekly') and s['4h'] and s['4h']['at_bottom']),

    ('A10 weekly_MT + 1h_STR',
     lambda s: _mt(s, 'weekly') and s['1h'] and s['1h']['stressed']),

    # ── B) 1h-as-base (mid-TF primary, higher TFs confirm, lower confirms) ───
    # "1h is at bottom/turning" + weekly/daily/4h confirming the macro regime

    ('B1 1h_MT',
     lambda s: _mt(s, '1h')),

    ('B2 1h_MT + weekly_NB',
     lambda s: _mt(s, '1h') and s['weekly'] and s['weekly']['near_bottom']),

    ('B3 1h_MT + weekly_NB + daily_NB',
     lambda s: _mt(s, '1h') and
               s['weekly'] and s['weekly']['near_bottom'] and
               s['daily']  and s['daily']['near_bottom']),

    ('B4 1h_ATB + weekly_MT',
     lambda s: s['1h'] and s['1h']['at_bottom'] and _mt(s, 'weekly')),

    ('B5 1h_ATB + weekly_NB + 4h_NB',
     lambda s: s['1h'] and s['1h']['at_bottom'] and
               s['weekly'] and s['weekly']['near_bottom'] and
               s['4h']     and s['4h']['near_bottom']),

    ('B6 1h_ATB + consensus_3+ (excl 5min)',
     lambda s: s['1h'] and s['1h']['at_bottom'] and
               _count_near_bottom_tfs(s, ['1h','4h','daily','weekly']) >= 3),

    ('B7 1h_MT + 4h_MT + weekly_NB',
     lambda s: _mt(s, '1h') and _mt(s, '4h') and
               s['weekly'] and s['weekly']['near_bottom']),

    ('B8 1h_MT + daily_MT + weekly_NB',
     lambda s: _mt(s, '1h') and _mt(s, 'daily') and
               s['weekly'] and s['weekly']['near_bottom']),

    # ── C) 5min-as-base (intraday trigger, higher TFs confirm direction) ──────
    # "5min turning up at bottom" + 1h/4h/daily/weekly all agree macro is low

    ('C1 5min_MT',
     lambda s: _mt(s, '5min')),

    ('C2 5min_MT + 1h_NB',
     lambda s: _mt(s, '5min') and s['1h'] and s['1h']['near_bottom']),

    ('C3 5min_MT + 1h_ATB',
     lambda s: _mt(s, '5min') and s['1h'] and s['1h']['at_bottom']),

    ('C4 5min_MT + 1h_ATB + 4h_NB',
     lambda s: _mt(s, '5min') and
               s['1h'] and s['1h']['at_bottom'] and
               s['4h'] and s['4h']['near_bottom']),

    ('C5 5min_MT + 1h_ATB + weekly_NB',
     lambda s: _mt(s, '5min') and
               s['1h'] and s['1h']['at_bottom'] and
               s['weekly'] and s['weekly']['near_bottom']),

    ('C6 5min_MT + 1h_ATB + 4h_NB + weekly_NB',
     lambda s: _mt(s, '5min') and
               s['1h'] and s['1h']['at_bottom'] and
               s['4h'] and s['4h']['near_bottom'] and
               s['weekly'] and s['weekly']['near_bottom']),

    ('C7 5min_MT + consensus_3+ (all 5)',
     lambda s: _mt(s, '5min') and _count_near_bottom(s) >= 3),

    ('C8 5min_MT + 1h_MT + weekly_NB',
     lambda s: _mt(s, '5min') and _mt(s, '1h') and
               s['weekly'] and s['weekly']['near_bottom']),

    ('C9 5min_ATB + 1h_ATB + weekly_MT',
     lambda s: s['5min'] and s['5min']['at_bottom'] and
               s['1h'] and s['1h']['at_bottom'] and
               _mt(s, 'weekly')),

    # ── D) Consensus-based (TF agreement regardless of which fires first) ────

    ('D1 consensus_3+ (any 3 of 5)',
     lambda s: _count_near_bottom(s) >= 3),

    ('D2 consensus_4+',
     lambda s: _count_near_bottom(s) >= 4),

    ('D3 consensus_5 (all 5 TFs bullish)',
     lambda s: _count_near_bottom(s) >= 5),

    ('D4 3+ TFs MT simultaneously',
     lambda s: sum(1 for tf in ['5min','1h','4h','daily','weekly']
                   if _mt(s, tf)) >= 3),

    ('D5 consensus_4+ + any_MT',
     lambda s: _count_near_bottom(s) >= 4 and
               any(_mt(s, tf) for tf in ['5min','1h','4h','daily','weekly'])),

    ('D6 (weekly OR daily)_MT + consensus_3+',
     lambda s: (_mt(s, 'weekly') or _mt(s, 'daily')) and
               _count_near_bottom(s) >= 3),

    # ── E) "Today's signal" — what fired Feb 26 2026 ─────────────────────────
    # weekly_MT + 5TF consensus + 1h/4h at bottom + 5min stressed

    ('E1 wMT+con5 (today exact)',
     lambda s: _mt(s, 'weekly') and _count_near_bottom(s) >= 5),

    ('E2 wMT+dMT+1hATB+con4+',
     lambda s: _mt(s, 'weekly') and _mt(s, 'daily') and
               s['1h'] and s['1h']['at_bottom'] and
               _count_near_bottom(s) >= 4),

    ('E3 5minMT+1hATB+wMT+con4+',
     lambda s: _mt(s, '5min') and
               s['1h'] and s['1h']['at_bottom'] and
               _mt(s, 'weekly') and _count_near_bottom(s) >= 4),
]


def _mt(states, tf):
    return bool(states.get(tf) and states[tf]['is_turning'])


def _count_near_bottom(states):
    return sum(1 for tf in ['5min', '1h', '4h', 'daily', 'weekly']
               if states.get(tf) and states[tf]['near_bottom'])


def _count_near_bottom_tfs(states, tfs):
    return sum(1 for tf in tfs if states.get(tf) and states[tf]['near_bottom'])


# ── Data loading ─────────────────────────────────────────────────────────────

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg(
        {'open': 'first', 'high': 'max', 'low': 'min',
         'close': 'last', 'volume': 'sum'}
    ).dropna(subset=['close'])


def load_all_tfs(tsla_min_path: str | None, start: str, end: str):
    """
    Load TSLA data resampled to all 5 TFs.
    If 1-min file not available, falls back to yfinance for 1h/4h/5min.
    Always fetches daily & weekly natively (full history).
    """
    print("Loading data...")

    # ── Daily + weekly from native_tf (full history) ─────────────────────────
    from v15.data.native_tf import fetch_native_tf
    daily = _norm_cols(fetch_native_tf('TSLA', 'daily', start, end))
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    weekly = _resample_ohlcv(daily, 'W-FRI')
    print(f"  daily: {len(daily):,} bars  weekly: {len(weekly):,} bars")

    # ── Intraday TFs ─────────────────────────────────────────────────────────
    if tsla_min_path and os.path.isfile(tsla_min_path):
        print(f"  Loading 1-min from {tsla_min_path}...")
        # Format: YYYYMMDD HHMMSS;open;high;low;close;volume (no header, semicolon-delimited)
        df1m = pd.read_csv(
            tsla_min_path, sep=';',
            header=None,
            names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
            parse_dates=['datetime'],
            date_format='%Y%m%d %H%M%S',
            index_col='datetime',
        )
        df1m.index = pd.to_datetime(df1m.index).tz_localize(None)
        df1m = df1m[start:end]
        tf5m = _resample_ohlcv(df1m, '5min')
        tf1h = _resample_ohlcv(df1m, '1h')
        tf4h = _resample_ohlcv(df1m, '4h')
        print(f"  5min: {len(tf5m):,}  1h: {len(tf1h):,}  4h: {len(tf4h):,}")
    else:
        print("  No 1-min file — fetching intraday from yfinance (limited history)...")
        import yfinance as yf
        df5m = _norm_cols(yf.download('TSLA', period='60d',  interval='5m',  progress=False))
        df1h = _norm_cols(yf.download('TSLA', period='730d', interval='1h',  progress=False))
        df5m.index = pd.to_datetime(df5m.index).tz_localize(None)
        df1h.index = pd.to_datetime(df1h.index).tz_localize(None)
        tf5m = df5m
        tf1h = df1h
        tf4h = _resample_ohlcv(df1h, '4h')
        print(f"  NOTE: yfinance 5min limited to last 60d, 1h to last 2yr")
        print(f"  5min: {len(tf5m):,}  1h: {len(tf1h):,}  4h: {len(tf4h):,}")

    return {
        '5min':  tf5m,
        '1h':    tf1h,
        '4h':    tf4h,
        'daily': daily,
        'weekly': weekly,
    }


# ── Per-day TF state computation ─────────────────────────────────────────────

def _last_bar_prices(tf_df: pd.DataFrame, date: pd.Timestamp,
                     channel_window: int) -> np.ndarray | None:
    """
    Extract close prices for the TF up to (and including) `date`.
    Returns None if not enough data.
    """
    idx = tf_df.index.searchsorted(date + pd.Timedelta(days=1), side='left') - 1
    if idx < channel_window:
        return None
    return tf_df['close'].iloc[max(0, idx - channel_window - 10): idx + 1].values.astype(float)


def compute_daily_states(tf_data: dict, trading_dates: pd.DatetimeIndex,
                         warmup_bars: int = 260) -> list:
    """
    For each trading day, compute TF states across all 5 TFs.
    Returns list of dicts: {'date': ..., '5min': state, '1h': state, ...}
    """
    print(f"\nComputing TF states for {len(trading_dates):,} trading days...")
    t0 = time.time()
    rows = []

    for i, date in enumerate(trading_dates):
        if i < warmup_bars:
            continue
        if i % 500 == 0:
            print(f"  {i}/{len(trading_dates)} ({time.time()-t0:.0f}s)...")

        states = {}
        for tf, window in TF_WINDOWS.items():
            df = tf_data.get(tf)
            if df is None or len(df) == 0:
                continue
            prices = _last_bar_prices(df, date, window)
            if prices is None or len(prices) < window:
                continue
            state = _compute_tf_state(prices, window)
            if state is not None:
                states[tf] = state

        rows.append({'date': date, **states})

    print(f"  Done in {time.time()-t0:.0f}s — {len(rows):,} rows with TF states")
    return rows


# ── Backtest engine ──────────────────────────────────────────────────────────

def run_backtest(daily_df: pd.DataFrame, state_rows: list,
                 signal_fn, signal_name: str,
                 capital: float = 100_000.0,
                 max_hold_days: int = 10,
                 stop_pct: float = 0.20,
                 start_year: int = 2015,
                 end_year: int = 2024) -> dict:
    """
    Backtest a signal defined by signal_fn(states_dict) → bool.
    Entry at next-day open, exit at max_hold_days or stop.
    """
    dates = daily_df.index
    opens  = daily_df['open'].values.astype(float)
    closes = daily_df['close'].values.astype(float)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    trades = []
    in_trade = False
    entry_idx = None
    entry_price = None

    for row in state_rows:
        date = row['date']
        yr = date.year
        if yr < start_year or yr > end_year:
            continue

        # Check exit first
        if in_trade:
            di = date_to_idx.get(date)
            if di is None:
                continue
            hold = di - entry_idx
            stop_price = entry_price * (1.0 - stop_pct)
            # Check stop: use today's low if available
            low = daily_df['low'].values[di] if 'low' in daily_df.columns else closes[di]
            hit_stop = low <= stop_price
            if hold >= max_hold_days or hit_stop:
                exit_price = stop_price if hit_stop else closes[di]
                pnl = (exit_price / entry_price - 1.0) * capital
                trades.append({
                    'entry_date': dates[entry_idx],
                    'exit_date':  date,
                    'entry':      entry_price,
                    'exit':       exit_price,
                    'hold':       hold,
                    'pnl':        pnl,
                    'year':       dates[entry_idx].year,
                    'stop':       hit_stop,
                })
                in_trade = False

        if in_trade:
            continue

        # Check signal
        states = {tf: row.get(tf) for tf in ['5min', '1h', '4h', 'daily', 'weekly']}
        try:
            fires = signal_fn(states)
        except Exception:
            fires = False

        if not fires:
            continue

        # Enter at next-day open
        di = date_to_idx.get(date)
        if di is None or di + 1 >= len(dates):
            continue
        next_di = di + 1
        if dates[next_di].year < start_year or dates[next_di].year > end_year:
            continue

        in_trade = True
        entry_idx = next_di
        entry_price = opens[next_di]

    # Close any open trade at last bar
    if in_trade and entry_idx is not None:
        last_di = len(dates) - 1
        exit_price = closes[last_di]
        pnl = (exit_price / entry_price - 1.0) * capital
        trades.append({
            'entry_date': dates[entry_idx],
            'exit_date':  dates[last_di],
            'entry':      entry_price,
            'exit':       exit_price,
            'hold':       last_di - entry_idx,
            'pnl':        pnl,
            'year':       dates[entry_idx].year,
            'stop':       False,
        })

    n  = len(trades)
    if n == 0:
        return {'name': signal_name, 'n': 0, 'wr': 0.0, 'pnl': 0.0,
                'avg': 0.0, 'pf': 0.0, 'trades': []}

    wins      = [t for t in trades if t['pnl'] > 0]
    gross_win = sum(t['pnl'] for t in wins)
    gross_los = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    total_pnl = sum(t['pnl'] for t in trades)

    return {
        'name':   signal_name,
        'n':      n,
        'wr':     len(wins) / n,
        'pnl':    total_pnl,
        'avg':    total_pnl / n,
        'pf':     gross_win / max(gross_los, 1e-6),
        'stops':  sum(1 for t in trades if t['stop']),
        'trades': trades,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Multi-TF dashboard signal backtest (all 5 TFs)')
    parser.add_argument('--tsla', type=str, default='data/TSLAMin.txt',
                        help='Path to 1-min TSLA data (optional — falls back to yfinance)')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end',   type=str, default='2025-12-31')
    parser.add_argument('--hold',  type=int, default=10, help='Max hold days (default: 10)')
    parser.add_argument('--stop',  type=float, default=0.20, help='Stop loss pct (default: 0.20)')
    parser.add_argument('--capital', type=float, default=100_000.0)
    parser.add_argument('--start-year', type=int, default=2015, dest='start_year')
    parser.add_argument('--end-year',   type=int, default=2024, dest='end_year')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("MULTI-TF DASHBOARD SIGNAL BACKTEST")
    print(f"TFs: 5min, 1h, 4h, daily, weekly")
    print(f"Hold: {args.hold}d  Stop: {args.stop:.0%}  Capital: ${args.capital:,.0f}")
    print(f"IS period: {args.start_year}-{args.end_year}")
    print(f"{'='*70}")

    # Load data
    tf_data = load_all_tfs(args.tsla, args.start, args.end)
    daily_df = tf_data['daily']

    # Compute daily TF states
    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates)

    # Run all signals across multiple hold periods
    hold_periods = [args.hold, 20, 30] if args.hold == 10 else [args.hold]

    all_results = {}
    for hold in hold_periods:
        print(f"\n{'='*70}")
        print(f"RUNNING {len(SIGNALS)} SIGNAL COMBINATIONS  [hold={hold}d]")
        print(f"{'='*70}")

        results = []
        for name, fn in SIGNALS:
            r = run_backtest(
                daily_df, state_rows, fn, name,
                capital=args.capital,
                max_hold_days=hold,
                stop_pct=args.stop,
                start_year=args.start_year,
                end_year=args.end_year,
            )
            r['hold'] = hold
            results.append(r)
            status = (f"n={r['n']:>3}  WR={r['wr']:.0%}  P&L=${r['pnl']:>10,.0f}"
                      f"  avg=${r['avg']:>7,.0f}  PF={r['pf']:.2f}")
            print(f"  {name:<46} {status}")
        all_results[hold] = results

    # Use default hold for summary/breakdown
    results = all_results[args.hold]

    # Summary table
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    hdr = f"{'Signal':<44} {'n':>4} {'WR':>5} {'P&L':>12} {'avg/tr':>8} {'PF':>5} {'stops':>6}"
    print(hdr)
    print('-' * 85)

    results_sorted = sorted(results, key=lambda r: r['pnl'], reverse=True)
    for r in results_sorted:
        if r['n'] == 0:
            print(f"  {r['name']:<44} {'0':>4} {'--':>5} {'$0':>12} {'$0':>8} {'--':>5} {'--':>6}")
        else:
            print(f"  {r['name']:<44} {r['n']:>4} {r['wr']:>4.0%} "
                  f"${r['pnl']:>10,.0f} ${r['avg']:>6,.0f} {r['pf']:>5.2f} "
                  f"{r['stops']:>6}")

    # Hold-period comparison for best signals
    if len(hold_periods) > 1:
        print(f"\n{'='*70}")
        print("HOLD-PERIOD SENSITIVITY — TOP 10 SIGNALS (by hold={} P&L)".format(args.hold))
        print(f"{'='*70}")
        top10_names = [r['name'] for r in results_sorted[:10] if r['n'] >= 3]
        hdr2 = f"{'Signal':<46} " + "  ".join(f"h={h:>2}d P&L" for h in hold_periods)
        print(hdr2)
        print('-' * (46 + 14 * len(hold_periods)))
        for name in top10_names:
            row_strs = []
            for h in hold_periods:
                match = next((r for r in all_results[h] if r['name'] == name), None)
                if match and match['n'] > 0:
                    row_strs.append(f"${match['pnl']:>9,.0f}")
                else:
                    row_strs.append(f"{'--':>10}")
            print(f"  {name:<46} " + "  ".join(row_strs))

    # Per-year breakdown for top 3
    top3 = [r for r in results_sorted if r['n'] >= 3][:3]
    if top3:
        print(f"\n{'='*70}")
        print("PER-YEAR BREAKDOWN — TOP 3 SIGNALS")
        print(f"{'='*70}")
        for r in top3:
            print(f"\n  [{r['name']}]  n={r['n']}  WR={r['wr']:.0%}  P&L=${r['pnl']:,.0f}")
            by_year = defaultdict(list)
            for t in r['trades']:
                by_year[t['year']].append(t)
            for yr in sorted(by_year):
                yr_trades = by_year[yr]
                yr_pnl = sum(t['pnl'] for t in yr_trades)
                yr_wr  = sum(1 for t in yr_trades if t['pnl'] > 0) / len(yr_trades)
                print(f"    {yr}: {len(yr_trades):>2} trades  "
                      f"WR={yr_wr:.0%}  P&L=${yr_pnl:>8,.0f}")

    # Today's conditions summary
    if state_rows:
        last = state_rows[-1]
        print(f"\n{'='*70}")
        print(f"CURRENT CONDITIONS ({last['date'].strftime('%Y-%m-%d')})")
        print(f"{'='*70}")
        for tf in ['5min', '1h', '4h', 'daily', 'weekly']:
            s = last.get(tf)
            if s is None:
                print(f"  {tf:<8}  no data")
                continue
            turning = 'MT=YES' if s['is_turning'] else 'MT=no '
            pos_str = f"pos={s['pos_pct']:.2f}"
            zone = ('ATB' if s['at_bottom'] else
                    ('NB ' if s['near_bottom'] else
                     ('TOP' if s['at_top'] else 'mid')))
            energy = f"E={s['energy_ratio']:.1f}x"
            stress = 'STR' if s['stressed'] else '   '
            print(f"  {tf:<8}  {turning}  {pos_str}  [{zone}]  {energy} {stress}")

        # Which signals fire today?
        today_states = {tf: last.get(tf) for tf in ['5min', '1h', '4h', 'daily', 'weekly']}
        firing = [name for name, fn in SIGNALS
                  if _safe_call(fn, today_states)]
        print(f"\n  Signals firing today: {len(firing)}")
        for name in firing:
            print(f"    ✓ {name}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


def _safe_call(fn, states):
    try:
        return bool(fn(states))
    except Exception:
        return False


if __name__ == '__main__':
    main()
