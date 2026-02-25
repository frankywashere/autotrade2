#!/usr/bin/env python3
"""
Paper Replay — Show what trades the system would have made yesterday and today.

Fetches recent 5-min data from yfinance (last 60 days for channel context),
runs the backtest with and without the winning filter cascade, then reports
every trade that entered OR exited in the last N trading days.

For each trade shows:
  - Entry: timestamp, direction, signal type, confidence, timeframe
  - Exit:  timestamp, exit reason, P&L ($), P&L (%)
  - Filter decisions: what the cascade did and why

Usage:
    python3 -m v15.validation.paper_replay
    python3 -m v15.validation.paper_replay --config all_50 --days-back 2
    python3 -m v15.validation.paper_replay --days-back 3 --config sq50
    python3 -m v15.validation.paper_replay --days-back 5 --show-blocked
"""

import argparse
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Reuse the config definitions from combined_backtest
CONFIGS = {
    'baseline':   {'sq_gate': 0.0,  'break_pred': False, 'swing': False},
    'sq50':       {'sq_gate': 0.50, 'break_pred': False, 'swing': False},
    'sq55':       {'sq_gate': 0.55, 'break_pred': False, 'swing': False},
    'bp_only':    {'sq_gate': 0.0,  'break_pred': True,  'swing': False},
    'swing_only': {'sq_gate': 0.0,  'break_pred': False, 'swing': True},
    'sq50_bp':    {'sq_gate': 0.50, 'break_pred': True,  'swing': False},
    'sq50_swing': {'sq_gate': 0.50, 'break_pred': False, 'swing': True},
    'all_50':     {'sq_gate': 0.50, 'break_pred': True,  'swing': True},
    'all_55':     {'sq_gate': 0.55, 'break_pred': True,  'swing': True},
}


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize MultiIndex columns from yfinance to lowercase flat names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df


def fetch_recent_data():
    """Fetch all data needed from yfinance. Returns dict of DataFrames."""
    import yfinance as yf

    print("Fetching recent market data from yfinance...")

    # 5-min TSLA + SPY (60-day max from yfinance)
    tsla_5m = _norm_cols(yf.download('TSLA', period='60d', interval='5m', progress=False))
    spy_5m  = _norm_cols(yf.download('SPY',  period='60d', interval='5m', progress=False))
    print(f"  TSLA 5-min: {len(tsla_5m)} bars  ({tsla_5m.index[0].date()} → {tsla_5m.index[-1].date()})")
    print(f"  SPY  5-min: {len(spy_5m)} bars")

    # Resample 5-min to higher TFs for channel detection
    def resample(df, rule):
        return df.resample(rule).agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna(subset=['close'])

    higher_tf = {
        '1h':    resample(tsla_5m, '1h'),
        '4h':    resample(tsla_5m, '4h'),
        'daily': resample(tsla_5m, '1D'),
    }
    print(f"  1h: {len(higher_tf['1h'])} bars  |  4h: {len(higher_tf['4h'])} bars  |  daily: {len(higher_tf['daily'])} bars")

    # Daily TSLA (5yr) — for swing regime weekly channel detection
    tsla_daily_long = _norm_cols(yf.download('TSLA', period='5y', interval='1d', progress=False))
    tsla_weekly     = _norm_cols(yf.download('TSLA', period='5y', interval='1wk', progress=False))
    spy_daily_long  = _norm_cols(yf.download('SPY',  period='5y', interval='1d', progress=False))
    print(f"  TSLA daily(5yr): {len(tsla_daily_long)} bars  |  weekly: {len(tsla_weekly)} bars")

    # VIX daily
    vix_daily = _norm_cols(yf.download('^VIX', period='2y', interval='1d', progress=False))
    print(f"  VIX daily: {len(vix_daily)} bars")

    return {
        'tsla_5m':        tsla_5m,
        'spy_5m':         spy_5m,
        'higher_tf':      higher_tf,
        'tsla_daily_long': tsla_daily_long,
        'tsla_weekly':    tsla_weekly,
        'spy_daily_long': spy_daily_long,
        'vix_daily':      vix_daily,
    }


def get_last_n_trading_dates(tsla_5m: pd.DataFrame, n: int) -> List[str]:
    """Return the last n unique trading dates from the 5-min index."""
    dates = sorted({str(ts.date()) for ts in tsla_5m.index})
    return dates[-n:]


def run_replay(data: dict, cfg: dict, cascade=None) -> tuple:
    """Run backtest on full 60-day window, return (trades, tsla_index)."""
    from v15.core.surfer_backtest import run_backtest

    # Precompute swing regime on full historical daily data
    if cascade is not None and cascade.swing_regime_enabled:
        cascade.precompute_swing_regime(
            data['tsla_daily_long'],
            data['spy_daily_long'],
            data['vix_daily'],
            data['tsla_weekly'],
        )

    result = run_backtest(
        days=0,
        eval_interval=6,
        max_hold_bars=60,
        position_size=10_000.0,
        min_confidence=0.45,
        use_multi_tf=True,
        tsla_df=data['tsla_5m'],
        higher_tf_dict=data['higher_tf'],
        spy_df_input=data['spy_5m'],
        vix_df_input=data['vix_daily'],
        realistic=True,
        slippage_bps=3.0,
        commission_per_share=0.005,
        max_leverage=4.0,
        bounce_cap=12.0,
        max_trade_usd=1_000_000.0,
        initial_capital=100_000.0,
        signal_filters=cascade,
    )

    metrics, trades, equity_curve = result[:3]
    return trades, data['tsla_5m'].index


def format_ts(ts) -> str:
    """Format a pandas Timestamp for display (date + time ET)."""
    try:
        # Convert to ET if timezone-aware
        if ts.tzinfo is not None:
            ts = ts.tz_convert('US/Eastern')
        return ts.strftime('%Y-%m-%d %H:%M ET')
    except Exception:
        return str(ts)


def bar_to_ts(tsla_index, bar_idx: int):
    """Look up timestamp for a bar index, safely."""
    if bar_idx < len(tsla_index):
        return tsla_index[bar_idx]
    return tsla_index[-1]


def filter_trades_to_window(trades, tsla_index, report_dates: List[str]):
    """
    Return trades where entry OR exit falls within the report_dates window.
    Includes trades that entered before but closed during the window.
    """
    date_set = set(report_dates)
    result = []
    for t in trades:
        entry_ts = bar_to_ts(tsla_index, t.entry_bar)
        exit_ts  = bar_to_ts(tsla_index, t.exit_bar)
        entry_date = str(entry_ts.date()) if hasattr(entry_ts, 'date') else str(entry_ts)[:10]
        exit_date  = str(exit_ts.date())  if hasattr(exit_ts, 'date')  else str(exit_ts)[:10]
        if entry_date in date_set or exit_date in date_set:
            result.append((t, entry_ts, exit_ts))
    return result


def print_trade_list(trade_tuples, label: str, cascade=None):
    """Print a formatted list of trades with entry/exit details."""
    if not trade_tuples:
        print(f"  (no {label} trades in window)")
        return

    # Build lookup from entry datetime → filter log entry (if cascade provided)
    filter_lookup = {}
    if cascade is not None:
        for log_entry in cascade.eval_log:
            key = str(log_entry['bar_datetime'])[:16]  # Match to minute precision
            filter_lookup[key] = log_entry

    print(f"\n  {'─'*110}")
    for trade, entry_ts, exit_ts in sorted(trade_tuples, key=lambda x: x[1]):
        t = trade
        direction_str = 'LONG ' if t.direction == 'BUY' else 'SHORT'
        sig_type_str  = 'bounce' if t.signal_type == 'bounce' else 'break '
        pnl_sign      = '+' if t.pnl >= 0 else ''
        win_loss      = 'WIN ' if t.pnl > 0 else 'LOSS'

        print(f"\n  [{win_loss}] {direction_str} {sig_type_str.upper()} — {t.primary_tf}")
        print(f"    Entry:  {format_ts(entry_ts)}  |  conf={t.confidence:.3f}  |  ${t.entry_price:.2f}/share")
        print(f"    Exit:   {format_ts(exit_ts)}   |  reason={t.exit_reason:<12s}  |  ${t.exit_price:.2f}/share")
        print(f"    P&L:    {pnl_sign}${t.pnl:,.0f}  ({pnl_sign}{t.pnl_pct*100:.2f}%)  |  size=${t.trade_size:,.0f}  |  held {t.hold_bars} bars ({t.hold_bars*5}min)")

        # Show filter decision if available
        entry_key = str(entry_ts)[:16]
        if entry_key in filter_lookup:
            fl = filter_lookup[entry_key]
            reason_str = ', '.join(fl['reasons']) if fl['reasons'] else 'PASS (no filters active)'
            print(f"    Filter: {reason_str}")

    print(f"  {'─'*110}")


def print_blocked_trades(cascade, report_dates: List[str]):
    """Print trades that were rejected by the filter during the report window."""
    if cascade is None:
        return

    date_set = set(report_dates)
    blocked = []
    for log_entry in cascade.eval_log:
        if not log_entry['rejected']:
            continue
        dt = log_entry['bar_datetime']
        date_str = str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10]
        if date_str in date_set:
            blocked.append(log_entry)

    if not blocked:
        print("  (no trades blocked by filter in window)")
        return

    print(f"\n  {'─'*110}")
    for b in sorted(blocked, key=lambda x: x['bar_datetime']):
        dt = b['bar_datetime']
        reason_str = ', '.join(b['reasons'])
        print(f"\n  [BLOCKED] {b['action']:<5} {b['signal_type'].upper()}")
        print(f"    Time:   {format_ts(dt)}")
        print(f"    Conf:   {b['conf_in']:.3f}  →  would have been {b['conf_out']:.3f}")
        print(f"    Reason: {reason_str}")
    print(f"  {'─'*110}")


def main():
    parser = argparse.ArgumentParser(description='Paper replay — show recent hypothetical trades')
    parser.add_argument('--config', type=str, default='all_50',
                        help=f'Filter config to use (default: all_50). Options: {list(CONFIGS.keys())}')
    parser.add_argument('--days-back', type=int, default=2,
                        help='Number of recent trading days to report (default: 2)')
    parser.add_argument('--show-blocked', action='store_true',
                        help='Also show trades that were blocked by the filter')
    args = parser.parse_args()

    if args.config not in CONFIGS:
        print(f"ERROR: Unknown config '{args.config}'. Options: {list(CONFIGS.keys())}")
        sys.exit(1)

    cfg = CONFIGS[args.config]

    print(f"\n{'='*70}")
    print(f"PAPER REPLAY — last {args.days_back} trading days")
    print(f"Config: {args.config}  (SQ={cfg['sq_gate']:.0%}, BP={cfg['break_pred']}, Swing={cfg['swing']})")
    print(f"{'='*70}\n")

    # ── Fetch data ─────────────────────────────────────────────────────────
    data = fetch_recent_data()

    # ── Determine report window ────────────────────────────────────────────
    report_dates = get_last_n_trading_dates(data['tsla_5m'], args.days_back)
    print(f"\nReport window: {report_dates[0]} → {report_dates[-1]}")

    # ── Build filter cascade ───────────────────────────────────────────────
    from v15.core.signal_filters import SignalFilterCascade

    # Baseline (no filters)
    print(f"\nRunning BASELINE (no filters)...")
    baseline_trades, tsla_index = run_replay(data, CONFIGS['baseline'], cascade=None)
    baseline_window = filter_trades_to_window(baseline_trades, tsla_index, report_dates)

    # Filtered run
    cascade = None
    if args.config != 'baseline':
        cascade = SignalFilterCascade(
            sq_gate_threshold=cfg['sq_gate'],
            break_predictor_enabled=cfg['break_pred'],
            swing_regime_enabled=cfg['swing'],
            swing_boost=1.2,
            break_penalty=0.5,
        )
        print(f"Running with filter: {args.config}...")
        filtered_trades, _ = run_replay(data, cfg, cascade=cascade)
        filtered_window = filter_trades_to_window(filtered_trades, tsla_index, report_dates)
    else:
        filtered_trades = baseline_trades
        filtered_window = baseline_window

    # ── Report ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TRADES EXECUTED — {args.config} — {report_dates[0]} to {report_dates[-1]}")
    print(f"{'='*70}")
    print_trade_list(filtered_window, args.config, cascade=cascade)

    if args.config != 'baseline':
        print(f"\n{'='*70}")
        print(f"BASELINE COMPARISON (what would have traded without filters)")
        print(f"{'='*70}")
        print_trade_list(baseline_window, 'baseline', cascade=None)

        if args.show_blocked:
            print(f"\n{'='*70}")
            print(f"TRADES BLOCKED BY FILTER ({args.config})")
            print(f"{'='*70}")
            print_blocked_trades(cascade, report_dates)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    def win_loss_summary(window_trades):
        if not window_trades:
            return '0 trades'
        wins   = sum(1 for t, _, _ in window_trades if t.pnl > 0)
        losses = sum(1 for t, _, _ in window_trades if t.pnl <= 0)
        total_pnl = sum(t.pnl for t, _, _ in window_trades)
        sign = '+' if total_pnl >= 0 else ''
        return f"{len(window_trades)} trades  ({wins}W / {losses}L)  total P&L: {sign}${total_pnl:,.0f}"

    print(f"  {args.config:<14}: {win_loss_summary(filtered_window)}")
    if args.config != 'baseline':
        print(f"  {'baseline':<14}: {win_loss_summary(baseline_window)}")
        if cascade is not None:
            n_blocked = sum(1 for le in cascade.eval_log
                            if le['rejected'] and
                            str(le['bar_datetime'])[:10] in set(report_dates))
            print(f"  Trades blocked by filter: {n_blocked}")
            print(f"\n  Filter stats (full 60-day window):\n{cascade.summary()}")


if __name__ == '__main__':
    main()
