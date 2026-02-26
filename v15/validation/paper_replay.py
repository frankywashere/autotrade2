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
    python3 -m v15.validation.paper_replay --tf 1h --days-back 3
    python3 -m v15.validation.paper_replay --tf 4h

Note: --days-back 3 covers a Mon+Fri+Thu window (e.g. Feb 23/24/25).
"""

import argparse
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ---------------------------------------------------------------------------
# TF-specific backtest parameters (mirrors medium_tf_backtest.TF_PARAMS)
# ---------------------------------------------------------------------------

TF_PARAMS = {
    '5min': dict(
        eval_interval=6,
        max_hold_bars=60,
        bounce_cap=12.0,
        max_trade_usd=1_000_000.0,
        tf_hours=5 / 60,           # 5 minutes in hours
    ),
    '1h': dict(
        eval_interval=1,
        max_hold_bars=10,
        bounce_cap=4.0,
        max_trade_usd=1_000_000.0,
        tf_hours=1.0,
    ),
    '4h': dict(
        eval_interval=1,
        max_hold_bars=5,
        bounce_cap=4.0,
        max_trade_usd=1_000_000.0,
        tf_hours=4.0,
    ),
}

# Reuse the config definitions from combined_backtest
CONFIGS = {
    'baseline':   {'sq_gate': 0.0,  'break_pred': False, 'swing': False, 'momentum': None},
    'sq50':       {'sq_gate': 0.50, 'break_pred': False, 'swing': False, 'momentum': None},
    'sq55':       {'sq_gate': 0.55, 'break_pred': False, 'swing': False, 'momentum': None},
    'bp_only':    {'sq_gate': 0.0,  'break_pred': True,  'swing': False, 'momentum': None},
    'swing_only': {'sq_gate': 0.0,  'break_pred': False, 'swing': True,  'momentum': None},
    'sq50_bp':    {'sq_gate': 0.50, 'break_pred': True,  'swing': False, 'momentum': None},
    'sq50_swing': {'sq_gate': 0.50, 'break_pred': False, 'swing': True,  'momentum': None},
    'all_50':     {'sq_gate': 0.50, 'break_pred': True,  'swing': True,  'momentum': None},
    'all_55':     {'sq_gate': 0.55, 'break_pred': True,  'swing': True,  'momentum': None},
    'mtf_exhaust': {
        'sq_gate': 0.0, 'break_pred': False, 'swing': False,
        'momentum': 'exhaust',  # momentum_boost=1.2, momentum_conflict_penalty=1.0
        'vix_cd': False,
    },
    # VIX Cooldown configs (x18 finding: fear receding → RSI rising = run beginning)
    'vix_cd': {
        'sq_gate': 0.0, 'break_pred': False, 'swing': False,
        'momentum': None, 'vix_cd': True,
    },
    'vix_cd_swing': {
        'sq_gate': 0.0, 'break_pred': False, 'swing': True,
        'momentum': None, 'vix_cd': True,
    },
}


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize MultiIndex columns from yfinance to lowercase flat names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV DataFrame to the given rule."""
    return df.resample(rule).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna(subset=['close'])


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
    higher_tf = {
        '1h':    _resample(tsla_5m, '1h'),
        '4h':    _resample(tsla_5m, '4h'),
        'daily': _resample(tsla_5m, '1D'),
    }
    print(f"  1h: {len(higher_tf['1h'])} bars  |  4h: {len(higher_tf['4h'])} bars  |  daily: {len(higher_tf['daily'])} bars")

    # 1h TSLA + SPY (730-day max from yfinance) — used as primary for 1h/4h replay
    # 730d × ~1.625 bars/day → ~1,186 1h bars; resampled to 4h → ~297 bars (well above 200 warmup)
    tsla_1h_long = _norm_cols(yf.download('TSLA', period='730d', interval='1h', progress=False))
    spy_1h_long  = _norm_cols(yf.download('SPY',  period='730d', interval='1h', progress=False))
    print(f"  TSLA 1h(730d): {len(tsla_1h_long)} bars  ({tsla_1h_long.index[0].date()} → {tsla_1h_long.index[-1].date()})")
    print(f"  SPY  1h(730d): {len(spy_1h_long)} bars")

    # Daily TSLA (5yr) — for swing regime weekly channel detection
    tsla_daily_long = _norm_cols(yf.download('TSLA', period='5y', interval='1d', progress=False))
    tsla_weekly     = _norm_cols(yf.download('TSLA', period='5y', interval='1wk', progress=False))
    spy_daily_long  = _norm_cols(yf.download('SPY',  period='5y', interval='1d', progress=False))
    print(f"  TSLA daily(5yr): {len(tsla_daily_long)} bars  |  weekly: {len(tsla_weekly)} bars")

    # VIX daily
    vix_daily = _norm_cols(yf.download('^VIX', period='2y', interval='1d', progress=False))
    print(f"  VIX daily: {len(vix_daily)} bars")

    return {
        'tsla_5m':         tsla_5m,
        'spy_5m':          spy_5m,
        'higher_tf':       higher_tf,
        'tsla_1h_long':    tsla_1h_long,
        'spy_1h_long':     spy_1h_long,
        'tsla_daily_long': tsla_daily_long,
        'tsla_weekly':     tsla_weekly,
        'spy_daily_long':  spy_daily_long,
        'vix_daily':       vix_daily,
    }


def get_last_n_trading_dates(tsla_df: pd.DataFrame, n: int) -> List[str]:
    """Return the last n unique trading dates from the DataFrame index."""
    dates = sorted({str(ts.date()) for ts in tsla_df.index})
    return dates[-n:]


def build_cascade(cfg: dict):
    """Build a SignalFilterCascade from a config dict. Returns None for baseline."""
    from v15.core.signal_filters import SignalFilterCascade

    momentum_mode = cfg.get('momentum')
    sq_gate   = cfg.get('sq_gate', 0.0)
    bp        = cfg.get('break_pred', False)
    swing     = cfg.get('swing', False)
    vix_cd    = cfg.get('vix_cd', False)

    # Pure baseline — no filters at all
    if sq_gate == 0.0 and not bp and not swing and momentum_mode is None and not vix_cd:
        return None

    mtm_enabled = momentum_mode is not None
    mtm_boost   = 1.2
    mtm_penalty = 1.0 if momentum_mode == 'exhaust' else (0.3 if momentum_mode in ('conflict', 'full') else 1.0)
    if momentum_mode == 'exhaust':
        mtm_boost, mtm_penalty = 1.2, 1.0
    elif momentum_mode == 'conflict':
        mtm_boost, mtm_penalty = 1.0, 0.3
    elif momentum_mode == 'full':
        mtm_boost, mtm_penalty = 1.2, 0.3

    return SignalFilterCascade(
        sq_gate_threshold=sq_gate,
        break_predictor_enabled=bp,
        swing_regime_enabled=swing,
        swing_boost=1.2,
        break_penalty=0.5,
        momentum_filter_enabled=mtm_enabled,
        momentum_boost=mtm_boost,
        momentum_conflict_penalty=mtm_penalty,
        momentum_context_tfs=['1h', '4h', 'daily'],
        momentum_min_tfs=2,
        vix_cooldown_enabled=vix_cd,
        vix_cooldown_boost=1.35,
    )


def run_replay(data: dict, cfg: dict, cascade=None, tf: str = '5min') -> tuple:
    """Run backtest on the primary TF window. Returns (trades, primary_index)."""
    from v15.core.surfer_backtest import run_backtest

    # Precompute swing regime on full historical daily data
    if cascade is not None and getattr(cascade, 'swing_regime_enabled', False):
        cascade.precompute_swing_regime(
            data['tsla_daily_long'],
            data['spy_daily_long'],
            data['vix_daily'],
            data['tsla_weekly'],
        )

    # Precompute VIX cooldown status (x18 finding)
    if cascade is not None and getattr(cascade, 'vix_cooldown_enabled', False):
        cascade.precompute_vix_cooldown(data['vix_daily'])

    p = TF_PARAMS[tf]

    # Choose the primary TSLA and SPY DataFrames based on TF
    if tf == '5min':
        tsla_primary = data['tsla_5m']
        spy_primary  = data['spy_5m']
        higher_tf_dict = data['higher_tf']
    else:
        # Use 730-day 1h bars as the source — gives ~1,186 1h bars or ~297 4h bars,
        # well above the 200-bar warmup threshold (60-day 5-min → ~97 4h bars is not enough)
        rule = '1h' if tf == '1h' else '4h'
        tsla_primary = _resample(data['tsla_1h_long'], rule) if tf == '4h' else data['tsla_1h_long']
        spy_primary  = _resample(data['spy_1h_long'],  rule) if tf == '4h' else data['spy_1h_long']
        # Higher-TF context = daily + weekly from 5yr daily data
        tsla_weekly_from_daily = _resample(data['tsla_daily_long'], '1W')
        higher_tf_dict = {
            'daily':  data['tsla_daily_long'],
            'weekly': tsla_weekly_from_daily,
        }
        print(f"    {tf} primary: {len(tsla_primary)} bars  "
              f"({tsla_primary.index[0].date()} → {tsla_primary.index[-1].date()})")

    result = run_backtest(
        days=0,
        eval_interval=p['eval_interval'],
        max_hold_bars=p['max_hold_bars'],
        position_size=10_000.0,
        min_confidence=0.45,
        use_multi_tf=True,
        tsla_df=tsla_primary,
        higher_tf_dict=higher_tf_dict,
        spy_df_input=spy_primary,
        vix_df_input=data['vix_daily'],
        realistic=True,
        slippage_bps=3.0,
        commission_per_share=0.005,
        max_leverage=4.0,
        bounce_cap=p['bounce_cap'],
        max_trade_usd=p['max_trade_usd'],
        min_trade_usd=33_000.0,
        initial_capital=100_000.0,
        signal_filters=cascade,
    )

    if result is None or len(result) < 3:
        print(f"    WARNING: backtest returned insufficient data (not enough bars for warmup?)")
        return [], tsla_primary.index

    metrics, trades, equity_curve = result[:3]
    print(f"    {tf} backtest: {len(trades)} total trades over full window")
    return trades, tsla_primary.index


def format_ts(ts) -> str:
    """Format a pandas Timestamp for display (date + time ET)."""
    try:
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


def print_trade_list(trade_tuples, label: str, cascade=None, tf: str = '5min'):
    """Print a formatted list of trades with entry/exit details."""
    tf_hours = TF_PARAMS[tf]['tf_hours']

    if not trade_tuples:
        print(f"  (no {label} trades in window)")
        return

    # Build lookup from entry datetime → filter log entry (if cascade provided)
    filter_lookup = {}
    if cascade is not None:
        for log_entry in cascade.eval_log:
            key = str(log_entry['bar_datetime'])[:16]
            filter_lookup[key] = log_entry

    print(f"\n  {'─'*110}")
    for trade, entry_ts, exit_ts in sorted(trade_tuples, key=lambda x: x[1]):
        t = trade
        direction_str = 'LONG ' if t.direction == 'BUY' else 'SHORT'
        sig_type_str  = 'bounce' if t.signal_type == 'bounce' else 'break '
        pnl_sign      = '+' if t.pnl >= 0 else ''
        win_loss      = 'WIN ' if t.pnl > 0 else 'LOSS'

        hold_hours = t.hold_bars * tf_hours
        if tf == '5min':
            hold_str = f"{t.hold_bars} bars ({t.hold_bars * 5}min)"
        else:
            hold_str = f"{t.hold_bars} bars ({hold_hours:.1f}h)"

        print(f"\n  [{win_loss}] {direction_str} {sig_type_str.upper()} — {t.primary_tf}")
        print(f"    Entry:  {format_ts(entry_ts)}  |  conf={t.confidence:.3f}  |  ${t.entry_price:.2f}/share")
        print(f"    Exit:   {format_ts(exit_ts)}   |  reason={t.exit_reason:<12s}  |  ${t.exit_price:.2f}/share")
        print(f"    P&L:    {pnl_sign}${t.pnl:,.0f}  ({pnl_sign}{t.pnl_pct*100:.2f}%)  |  "
              f"size=${t.trade_size:,.0f}  |  held {hold_str}")

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
                        help='Number of recent trading days to report (default: 2). '
                             'Use --days-back 3 to cover e.g. Mon+Fri+Thu (Feb 23/24/25).')
    parser.add_argument('--show-blocked', action='store_true',
                        help='Also show trades that were blocked by the filter')
    parser.add_argument('--tf', type=str, default='5min', choices=['5min', '1h', '4h'],
                        help='Primary timeframe for backtest: 5min, 1h, or 4h (default: 5min)')
    args = parser.parse_args()

    if args.config not in CONFIGS:
        print(f"ERROR: Unknown config '{args.config}'. Options: {list(CONFIGS.keys())}")
        sys.exit(1)

    cfg = CONFIGS[args.config]
    p   = TF_PARAMS[args.tf]

    print(f"\n{'='*70}")
    print(f"PAPER REPLAY — last {args.days_back} trading days  |  TF: {args.tf}")
    momentum_str = cfg.get('momentum') or 'none'
    print(f"Config: {args.config}  (SQ={cfg['sq_gate']:.0%}, BP={cfg['break_pred']}, "
          f"Swing={cfg['swing']}, Momentum={momentum_str})")
    print(f"TF params: eval_interval={p['eval_interval']}, max_hold_bars={p['max_hold_bars']}, "
          f"bounce_cap={p['bounce_cap']}")
    print(f"{'='*70}\n")

    # ── Fetch data ─────────────────────────────────────────────────────────
    data = fetch_recent_data()

    # ── Determine primary index for report window (depends on TF) ──────────
    if args.tf == '5min':
        primary_index_df = data['tsla_5m']
    elif args.tf == '1h':
        primary_index_df = data['tsla_1h_long']
    else:
        primary_index_df = _resample(data['tsla_1h_long'], '4h')

    report_dates = get_last_n_trading_dates(primary_index_df, args.days_back)
    print(f"Report window: {report_dates[0]} → {report_dates[-1]}")

    # VIX cooldown diagnostic — always show VIX regime context for report window
    vix_close = data['vix_daily']['close'].astype(float)
    print(f"\nVIX regime for report window:")
    for d in report_dates:
        vix_idx = data['vix_daily'].index
        date_mask = np.array([str(ts.date()) == d if hasattr(ts, 'date') else str(ts)[:10] == d
                              for ts in vix_idx])
        if date_mask.any():
            i = int(np.where(date_mask)[0][-1])
            current_vix = float(vix_close.iloc[i])
            if i >= 252:
                trailing = vix_close.iloc[i - 252:i]
                spike_thr = float(trailing.quantile(0.70))
                lookback_win = vix_close.iloc[max(0, i - 10):i + 1]
                recent_peak = float(lookback_win.max())
                was_elev = recent_peak >= spike_thr
                cooling = current_vix <= recent_peak * 0.90
                status = "*** COOLDOWN ACTIVE ***" if (was_elev and cooling) else "normal"
                print(f"  {d}: VIX={current_vix:.1f}  70pct_thr={spike_thr:.1f}  "
                      f"10d_peak={recent_peak:.1f}  [{status}]")
            else:
                print(f"  {d}: VIX={current_vix:.1f}  (insufficient history for cooldown check)")
        else:
            print(f"  {d}: no VIX data")

    # ── Run BASELINE ───────────────────────────────────────────────────────
    print(f"\nRunning BASELINE (no filters, TF={args.tf})...")
    baseline_trades, tsla_index = run_replay(data, CONFIGS['baseline'], cascade=None, tf=args.tf)
    baseline_window = filter_trades_to_window(baseline_trades, tsla_index, report_dates)
    print(f"    Window filter: {len(baseline_trades)} total trades → {len(baseline_window)} in window")

    # ── Run FILTERED config ────────────────────────────────────────────────
    cascade = build_cascade(cfg)
    if args.config != 'baseline' and cascade is not None:
        print(f"Running with filter: {args.config} (TF={args.tf})...")
        filtered_trades, _ = run_replay(data, cfg, cascade=cascade, tf=args.tf)
        filtered_window = filter_trades_to_window(filtered_trades, tsla_index, report_dates)
    else:
        filtered_trades = baseline_trades
        filtered_window = baseline_window

    # ── Report ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TRADES EXECUTED — {args.config} — {report_dates[0]} to {report_dates[-1]}")
    print(f"{'='*70}")
    print_trade_list(filtered_window, args.config, cascade=cascade, tf=args.tf)

    if args.config != 'baseline':
        print(f"\n{'='*70}")
        print(f"BASELINE COMPARISON (what would have traded without filters)")
        print(f"{'='*70}")
        print_trade_list(baseline_window, 'baseline', cascade=None, tf=args.tf)

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
        wins      = sum(1 for t, _, _ in window_trades if t.pnl > 0)
        losses    = sum(1 for t, _, _ in window_trades if t.pnl <= 0)
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
            print(f"\n  Filter stats (full window):\n{cascade.summary()}")


if __name__ == '__main__':
    main()
