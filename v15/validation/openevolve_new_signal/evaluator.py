"""
Evaluator for OpenEvolve Phase B2: New cross-asset signal discovery.

Loads 1-min TSLA/SPY/VIX bars from semicolon-delimited text files, resamples
to 5-min bars (RTH only: 09:30-16:00 ET).  Walks forward at **1-min resolution**
for honest stop/TP/timeout fills, but only calls the candidate's
generate_signals() every 5th 1-min bar (i.e. at 5-min boundaries) with 5-min
lookback windows.

Data files (semicolon-delimited: YYYYMMDD HHMMSS;open;high;low;close;volume):
  - TSLA: TSLAMin_yfinance_deprecated.txt  (2015-01-02 to 2026-02-27)
  - SPY:  SPYMin.txt                       (2015-01-02 to 2025-09-27)
  - VIX:  VIXMin_IB.txt                    (2014-12-31 to 2026-03-13)

Periods:
  - Training: 2015-01-01 to 2024-12-31 (scored)
  - Holdout:  2025-01-01 to 2025-09-27 (limited by SPY data end, reported only)

Scoring: composite of PnL, Sharpe, win rate, profit factor, drawdown.
Penalizes extreme trade counts and overfitting (min 500 trades on training).
"""

import importlib
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)


# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_START = '2015-01-01'
TRAIN_END = '2024-12-31'
HOLDOUT_START = '2025-01-01'
HOLDOUT_END = '2025-09-27'
FULL_START = TRAIN_START
FULL_END = HOLDOUT_END

LOOKBACK = 100           # Number of 5-min bars of history to pass to generate_signals()
EVAL_INTERVAL = 5        # Generate signals every N 1-min bars (= 5-min cadence)
INITIAL_EQUITY = 100_000
MAX_EQUITY_PER_TRADE = 100_000
MAX_POSITIONS = 2
COST_PER_TRADE = 2.0     # $2 round-trip commissions
SLIPPAGE_PCT = 0.0005    # 0.05% slippage per side

# Minimum trades required on training period to get a score
# (10 years of 5-min bars = ~500K bars, expect more trades than daily)
MIN_TRADES_TRAIN = 500

# RTH window (Eastern Time)
RTH_START_TIME = pd.Timestamp('09:30').time()
RTH_END_TIME = pd.Timestamp('16:00').time()

# Data file paths: try Windows server first, then Mac relative
_DATA_PATHS = {
    'TSLA': [
        r'C:\AI\x14\data\TSLAMin_yfinance_deprecated.txt',
        os.path.join(PROJECT_ROOT, 'data', 'TSLAMin_yfinance_deprecated.txt'),
    ],
    'SPY': [
        r'C:\AI\x14\data\SPYMin.txt',
        os.path.join(PROJECT_ROOT, 'data', 'SPYMin.txt'),
    ],
    'VIX': [
        r'C:\AI\x14\data\VIXMin_IB.txt',
        os.path.join(PROJECT_ROOT, 'data', 'VIXMin_IB.txt'),
    ],
}


# ── Data loading ──────────────────────────────────────────────────────────────
_CACHED_DATA = None


def _find_data_file(symbol):
    """Find the data file for a symbol, trying Windows then Mac paths."""
    for path in _DATA_PATHS[symbol]:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Data file for {symbol} not found. Tried: {_DATA_PATHS[symbol]}")


def _load_1min_bars(filepath, symbol):
    """Load 1-min bars from semicolon-delimited text file.

    Format: YYYYMMDD HHMMSS;open;high;low;close;volume
    Returns DataFrame with DatetimeIndex and columns [open, high, low, close, volume].
    """
    t0 = time.time()
    df = pd.read_csv(
        filepath,
        sep=';',
        header=None,
        names=['datetime_str', 'open', 'high', 'low', 'close', 'volume'],
        dtype={
            'datetime_str': str,
            'open': np.float64,
            'high': np.float64,
            'low': np.float64,
            'close': np.float64,
            'volume': np.float64,
        },
    )
    # Parse datetime: "YYYYMMDD HHMMSS"
    df.index = pd.to_datetime(df['datetime_str'], format='%Y%m%d %H%M%S')
    df.drop(columns=['datetime_str'], inplace=True)
    df.sort_index(inplace=True)
    elapsed = time.time() - t0
    print(f"  Loaded {symbol}: {len(df):,} 1-min bars "
          f"({df.index[0]} to {df.index[-1]}) in {elapsed:.1f}s")
    return df


def _resample_to_5min(df_1min):
    """Resample 1-min bars to 5-min bars using standard OHLCV aggregation."""
    df_5min = df_1min.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open'])
    return df_5min


def _filter_rth(df):
    """Filter to RTH hours: 09:30 <= time < 16:00 ET.

    Assumes timestamps are already in ET (the data files use ET timestamps).
    """
    times = df.index.time
    mask = (times >= RTH_START_TIME) & (times < RTH_END_TIME)
    return df.loc[mask]


def _load_data():
    """Load 1-min bars for TSLA, SPY, VIX, resample to 5-min, filter RTH, align.

    Returns dict with keys:
      '1min' -> {symbol: DataFrame} aligned 1-min RTH bars
      '5min' -> {symbol: DataFrame} aligned 5-min RTH bars
    Cached after first call.
    """
    global _CACHED_DATA
    if _CACHED_DATA is not None:
        return _CACHED_DATA

    t_total = time.time()
    print("Loading 1-min data files...")

    # Load raw 1-min bars
    raw = {}
    for symbol in ['TSLA', 'SPY', 'VIX']:
        filepath = _find_data_file(symbol)
        raw[symbol] = _load_1min_bars(filepath, symbol)

    # ── Build aligned 1-min RTH data ──
    print("Filtering 1-min bars to RTH + date range...")
    data_1min = {}
    for symbol in ['TSLA', 'SPY', 'VIX']:
        df = _filter_rth(raw[symbol])
        df = df.loc[
            (df.index >= pd.Timestamp(FULL_START)) &
            (df.index <= pd.Timestamp(FULL_END) + pd.Timedelta(days=1))
        ]
        data_1min[symbol] = df
        print(f"  {symbol}: {len(df):,} 1-min RTH bars")

    # Align 1-min to common timestamps
    common_1min_idx = data_1min['TSLA'].index
    for sym in ['SPY', 'VIX']:
        common_1min_idx = common_1min_idx.intersection(data_1min[sym].index)
    common_1min_idx = common_1min_idx.sort_values()

    for sym in data_1min:
        data_1min[sym] = data_1min[sym].loc[common_1min_idx]

    print(f"  Aligned 1-min bars: {len(common_1min_idx):,}")

    # ── Build aligned 5-min RTH data ──
    print("Resampling to 5-min bars...")
    data_5min = {}
    for symbol in ['TSLA', 'SPY', 'VIX']:
        resampled = _resample_to_5min(raw[symbol])
        resampled = _filter_rth(resampled)
        resampled = resampled.loc[
            (resampled.index >= pd.Timestamp(FULL_START)) &
            (resampled.index <= pd.Timestamp(FULL_END) + pd.Timedelta(days=1))
        ]
        data_5min[symbol] = resampled
        print(f"  {symbol}: {len(resampled):,} 5-min RTH bars")

    # Align 5-min to common timestamps
    common_5min_idx = data_5min['TSLA'].index
    for sym in ['SPY', 'VIX']:
        common_5min_idx = common_5min_idx.intersection(data_5min[sym].index)
    common_5min_idx = common_5min_idx.sort_values()

    for sym in data_5min:
        data_5min[sym] = data_5min[sym].loc[common_5min_idx]

    elapsed = time.time() - t_total
    print(f"Data loaded: {len(common_1min_idx):,} aligned 1-min RTH bars, "
          f"{len(common_5min_idx):,} aligned 5-min RTH bars "
          f"({common_1min_idx[0].strftime('%Y-%m-%d %H:%M')} to "
          f"{common_1min_idx[-1].strftime('%Y-%m-%d %H:%M')}) in {elapsed:.1f}s")

    _CACHED_DATA = {'1min': data_1min, '5min': data_5min}
    return _CACHED_DATA


# ── Simple position tracking ─────────────────────────────────────────────────

class Position:
    __slots__ = ('pos_id', 'direction', 'entry_price', 'entry_time',
                 'stop_price', 'tp_price', 'size_dollars', 'shares',
                 'best_price', 'hold_bars')

    def __init__(self, pos_id, direction, entry_price, entry_time,
                 stop_pct, tp_pct, size_dollars):
        self.pos_id = pos_id
        self.direction = direction
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.size_dollars = size_dollars
        self.shares = size_dollars / entry_price

        if direction == 'long':
            self.stop_price = entry_price * (1 - stop_pct)
            self.tp_price = entry_price * (1 + tp_pct)
            self.best_price = entry_price
        else:
            self.stop_price = entry_price * (1 + stop_pct)
            self.tp_price = entry_price * (1 - tp_pct)
            self.best_price = entry_price

        self.hold_bars = 0


class Trade:
    __slots__ = ('direction', 'entry_price', 'exit_price', 'entry_time',
                 'exit_time', 'shares', 'pnl', 'reason', 'hold_bars')

    def __init__(self, pos, exit_price, exit_time, reason):
        self.direction = pos.direction
        self.entry_price = pos.entry_price
        self.exit_price = exit_price
        self.entry_time = pos.entry_time
        self.exit_time = exit_time
        self.shares = pos.shares
        self.hold_bars = pos.hold_bars
        self.reason = reason

        if pos.direction == 'long':
            raw_pnl = (exit_price - pos.entry_price) * pos.shares
        else:
            raw_pnl = (pos.entry_price - exit_price) * pos.shares

        # Deduct costs
        slippage = pos.size_dollars * SLIPPAGE_PCT * 2  # entry + exit
        self.pnl = raw_pnl - COST_PER_TRADE - slippage


# ── Backtest engine ──────────────────────────────────────────────────────────

MAX_HOLD_BARS = 390  # Max hold in 1-min bars (~1 trading day = 390 bars)


def _run_backtest(data, generate_signals_fn, start, end):
    """Walk forward through 1-min bars for honest exit fills.

    Signal generation (generate_signals()) is called every EVAL_INTERVAL (5)
    1-min bars, receiving 5-min lookback DataFrames.  Stop/TP/timeout exits
    are checked on EVERY 1-min bar.
    """
    # 1-min data for exit checking
    tsla_1m = data['1min']['TSLA']
    spy_1m = data['1min']['SPY']
    vix_1m = data['1min']['VIX']

    # 5-min data for signal generation lookback
    tsla_5m = data['5min']['TSLA']
    spy_5m = data['5min']['SPY']
    vix_5m = data['5min']['VIX']

    # Filter 1-min bars to date range
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)  # Include last day
    mask_1m = (tsla_1m.index >= start_ts) & (tsla_1m.index < end_ts)
    bar_indices_1m = np.where(np.asarray(mask_1m))[0]

    if len(bar_indices_1m) == 0:
        return [], []

    # Pre-extract numpy arrays for fast access (1-min)
    tsla_1m_vals = tsla_1m.values  # [open, high, low, close, volume]
    tsla_1m_idx = tsla_1m.index

    # Pre-extract numpy arrays for 5-min lookback windows
    tsla_5m_vals = tsla_5m.values
    spy_5m_vals = spy_5m.values
    vix_5m_vals = vix_5m.values
    tsla_5m_idx = tsla_5m.index
    tsla_5m_cols = tsla_5m.columns

    # Build a set of 5-min boundary timestamps for fast membership check.
    # A 1-min bar is a "5-min boundary" if its timestamp is in the 5-min index.
    fivemin_ts_set = set(tsla_5m_idx)

    positions = []  # Open positions
    trades = []     # Closed trades
    pos_counter = 0
    equity_curve = []
    equity = INITIAL_EQUITY
    bars_since_signal = EVAL_INTERVAL  # Fire on the first eligible bar

    for bar_i in bar_indices_1m:
        bar_time = tsla_1m_idx[bar_i]

        bar_high = float(tsla_1m_vals[bar_i, 1])
        bar_low = float(tsla_1m_vals[bar_i, 2])
        bar_close = float(tsla_1m_vals[bar_i, 3])

        # ── Check exits on existing positions (EVERY 1-min bar) ──
        closed_ids = set()
        for pos in positions:
            pos.hold_bars += 1

            # Update best price
            if pos.direction == 'long':
                pos.best_price = max(pos.best_price, bar_high)
            else:
                pos.best_price = min(pos.best_price, bar_low)

            exit_price = None
            reason = None

            if pos.direction == 'long':
                if bar_low <= pos.stop_price:
                    exit_price = pos.stop_price
                    reason = 'stop'
                elif bar_high >= pos.tp_price:
                    exit_price = pos.tp_price
                    reason = 'tp'
                elif pos.hold_bars >= MAX_HOLD_BARS:
                    exit_price = bar_close
                    reason = 'timeout'
            else:  # short
                if bar_high >= pos.stop_price:
                    exit_price = pos.stop_price
                    reason = 'stop'
                elif bar_low <= pos.tp_price:
                    exit_price = pos.tp_price
                    reason = 'tp'
                elif pos.hold_bars >= MAX_HOLD_BARS:
                    exit_price = bar_close
                    reason = 'timeout'

            if exit_price is not None:
                trade = Trade(pos, exit_price, bar_time, reason)
                trades.append(trade)
                equity += trade.pnl
                closed_ids.add(pos.pos_id)

        # Remove closed positions
        if closed_ids:
            positions = [p for p in positions if p.pos_id not in closed_ids]

        # ── Signal generation: only at 5-min boundaries ──
        bars_since_signal += 1
        is_5min_boundary = bar_time in fivemin_ts_set

        if is_5min_boundary and bars_since_signal >= EVAL_INTERVAL:
            bars_since_signal = 0

            # Find corresponding position in 5-min arrays for lookback
            # searchsorted gives the insertion point; we want the bar AT bar_time
            pos_in_5m = tsla_5m_idx.searchsorted(bar_time, side='right')
            if pos_in_5m == 0:
                # No 5-min bar at or before this time — skip signal gen
                equity_curve.append(equity)
                continue

            start_5m = max(0, pos_in_5m - LOOKBACK)
            end_5m = pos_in_5m

            # Build 5-min lookback DataFrames
            tsla_window = pd.DataFrame(
                tsla_5m_vals[start_5m:end_5m],
                index=tsla_5m_idx[start_5m:end_5m],
                columns=tsla_5m_cols,
            )
            spy_window = pd.DataFrame(
                spy_5m_vals[start_5m:end_5m],
                index=tsla_5m_idx[start_5m:end_5m],
                columns=tsla_5m_cols,
            )
            vix_window = pd.DataFrame(
                vix_5m_vals[start_5m:end_5m],
                index=tsla_5m_idx[start_5m:end_5m],
                columns=tsla_5m_cols,
            )

            n_long = sum(1 for p in positions if p.direction == 'long')
            n_short = sum(1 for p in positions if p.direction == 'short')

            position_info = {
                'has_long': n_long > 0,
                'has_short': n_short > 0,
                'n_positions': len(positions),
                'max_positions': MAX_POSITIONS,
            }

            try:
                signals = generate_signals_fn(
                    tsla_window, spy_window, vix_window,
                    bar_time, position_info,
                )
            except Exception as _sig_err:
                if not hasattr(generate_signals_fn, '_err_logged'):
                    import traceback as _tb
                    print(f"SIGNAL_ERROR: {_tb.format_exc()[-300:]}")
                    generate_signals_fn._err_logged = True
                signals = []

            if not isinstance(signals, list):
                signals = []

            # ── Process signals: enter positions ──
            for sig in signals:
                if not isinstance(sig, dict):
                    continue
                if len(positions) >= MAX_POSITIONS:
                    break

                direction = sig.get('direction', '')
                if direction not in ('long', 'short'):
                    continue

                if direction == 'long' and n_long >= MAX_POSITIONS:
                    continue
                if direction == 'short' and n_short >= MAX_POSITIONS:
                    continue

                confidence = float(sig.get('confidence', 0.5))
                if confidence < 0.01:
                    continue

                stop_pct = float(sig.get('stop_pct', 0.005))
                tp_pct = float(sig.get('tp_pct', 0.008))

                # Clamp to reasonable ranges
                stop_pct = max(0.001, min(stop_pct, 0.05))
                tp_pct = max(0.001, min(tp_pct, 0.08))

                # Enter at current bar's close (1-min close at 5-min boundary)
                entry_price = bar_close

                pos_counter += 1
                pos = Position(
                    pos_id=f"p{pos_counter}",
                    direction=direction,
                    entry_price=entry_price,
                    entry_time=bar_time,
                    stop_pct=stop_pct,
                    tp_pct=tp_pct,
                    size_dollars=MAX_EQUITY_PER_TRADE,
                )
                positions.append(pos)

                if direction == 'long':
                    n_long += 1
                else:
                    n_short += 1

        equity_curve.append(equity)

    # Close any remaining positions at last bar's close
    if positions and len(bar_indices_1m) > 0:
        last_i = bar_indices_1m[-1]
        last_close = float(tsla_1m_vals[last_i, 3])
        last_time = tsla_1m_idx[last_i]
        for pos in positions:
            trade = Trade(pos, last_close, last_time, 'eod_close')
            trades.append(trade)
            equity += trade.pnl

    return trades, equity_curve


# ── Metrics computation ──────────────────────────────────────────────────────

def _compute_metrics(trades, initial_equity=INITIAL_EQUITY):
    """Compute performance metrics from trade list."""
    if not trades:
        return {
            'total_pnl': 0, 'n_trades': 0, 'win_rate': 0,
            'sharpe': 0, 'profit_factor': 0, 'max_drawdown_pct': 0,
            'avg_pnl': 0, 'avg_hold': 0,
        }

    pnls = np.array([t.pnl for t in trades])
    total_pnl = float(np.sum(pnls))
    n_trades = len(trades)
    winners = np.sum(pnls > 0)
    win_rate = float(winners / n_trades * 100)
    avg_pnl = float(np.mean(pnls))
    avg_hold = float(np.mean([t.hold_bars for t in trades]))

    # Sharpe (annualized from per-trade PnL)
    # Using sqrt(252) as annualization factor (trade-level, not bar-level)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0]))
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0])))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = 999.0 if gross_profit > 0 else 0.0

    # Max drawdown
    cumulative = np.cumsum(pnls)
    equity_curve = initial_equity + cumulative
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    return {
        'total_pnl': total_pnl,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_dd,
        'avg_pnl': avg_pnl,
        'avg_hold': avg_hold,
    }


# ── Scoring ──────────────────────────────────────────────────────────────────

def _compute_score(m):
    """Composite score: PnL * quality multipliers."""
    total_pnl = m['total_pnl']
    win_rate = m['win_rate'] / 100.0  # Convert to 0-1
    sharpe = m['sharpe']
    pf = m['profit_factor']
    dd = m['max_drawdown_pct']
    n_trades = m['n_trades']

    # Too few trades = likely overfit (raised thresholds for 5-min bars)
    if n_trades < 200:
        trade_mult = 0.1
    elif n_trades < MIN_TRADES_TRAIN:
        trade_mult = 0.3 + 0.7 * (n_trades - 200) / (MIN_TRADES_TRAIN - 200)
    elif n_trades <= 10000:
        trade_mult = 1.0
    elif n_trades <= 25000:
        trade_mult = 1.0 - 0.3 * (n_trades - 10000) / 15000
    else:
        trade_mult = 0.7

    # Drawdown penalty
    if dd <= 10:
        dd_mult = 1.0
    elif dd <= 20:
        dd_mult = 1.0 - 0.3 * (dd - 10) / 10
    elif dd <= 40:
        dd_mult = 0.7 - 0.4 * (dd - 20) / 20
    else:
        dd_mult = 0.3

    if total_pnl <= 0:
        return 0.0

    score = (total_pnl
             * (1.0 + max(sharpe, 0) * 0.2)
             * (0.3 + win_rate * 0.7)
             * (1.0 + max(pf - 1, 0) * 0.1)
             * trade_mult
             * dd_mult)

    return max(score, 0.0)


# ── Main evaluate() ─────────────────────────────────────────────────────────

def evaluate(program_path: str) -> dict:
    """Evaluate a signal candidate. Returns score dict."""
    # Load data
    try:
        data = _load_data()
    except Exception as e:
        return {'combined_score': 0.0, 'error': f'data load failed: {e}'}

    # Import candidate's generate_signals function
    try:
        spec = importlib.util.spec_from_file_location('candidate', program_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        generate_signals_fn = getattr(mod, 'generate_signals', None)
        if generate_signals_fn is None:
            return {'combined_score': 0.0, 'error': 'missing generate_signals()'}

    except Exception as e:
        return {'combined_score': 0.0, 'error': f'import failed: {e}'}

    # ── Run on training period ──
    try:
        train_trades, train_eq = _run_backtest(
            data, generate_signals_fn, TRAIN_START, TRAIN_END)
    except Exception as e:
        return {'combined_score': 0.0,
                'error': f'train backtest failed: {traceback.format_exc()[-500:]}'}

    train_m = _compute_metrics(train_trades)
    if train_m['n_trades'] == 0:
        return {'combined_score': 0.0, 'error': 'no trades generated (candidate code may be crashing silently)'}
    train_score = _compute_score(train_m)

    # ── Run on holdout period ──
    try:
        holdout_trades, holdout_eq = _run_backtest(
            data, generate_signals_fn, HOLDOUT_START, HOLDOUT_END)
    except Exception as e:
        holdout_trades = []

    holdout_m = _compute_metrics(holdout_trades)

    # ── Build result ──
    result = {
        'combined_score': train_score,
        # Training metrics (used for scoring)
        'total_pnl': train_m['total_pnl'],
        'n_trades': train_m['n_trades'],
        'win_rate': train_m['win_rate'],
        'sharpe': train_m['sharpe'],
        'profit_factor': train_m['profit_factor'],
        'max_drawdown_pct': train_m['max_drawdown_pct'],
        'avg_pnl': train_m['avg_pnl'],
        'avg_hold': train_m['avg_hold'],
        # Holdout metrics (reported only — NOT used for scoring)
        'holdout_pnl': holdout_m['total_pnl'],
        'holdout_trades': holdout_m['n_trades'],
        'holdout_wr': holdout_m['win_rate'],
        'holdout_sharpe': holdout_m['sharpe'],
        'holdout_dd': holdout_m['max_drawdown_pct'],
    }

    return result
