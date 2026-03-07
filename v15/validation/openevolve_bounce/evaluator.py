"""
Evaluator for OpenEvolve bounce signal evolution.

Loads TSLA + SPY data, computes TF states on signal dates,
runs the evolved bounce signal, then measures forward returns
to score the signal quality.

Scoring v2: penalizes over-firing and drawdown.
  - No artificial 7-day cooldown — every day is evaluated
  - Selectivity: signal should fire rarely (< 15% of days)
  - Overlapping trades penalized (can't enter if already in a position)
  - Drawdown penalty baked into score
"""

import os
import sys
import importlib
import traceback

import numpy as np
import pandas as pd

# Ensure project root is on path (C:\AI\x14 on Windows server)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)
# Also add explicit Windows server path
if os.path.isdir(r'C:\AI\x14'):
    sys.path.insert(0, r'C:\AI\x14')

from openevolve.evaluation_result import EvaluationResult

from v15.validation.tf_state_backtest import (
    load_all_tfs,
    compute_daily_states,
    _compute_tf_state,
    TF_WINDOWS,
    _norm_cols,
)
from v15.validation.bounce_timing import _compute_rsi

# ── Constants ────────────────────────────────────────────────────────────────

TSLA_MIN_PATH = r'C:\AI\x14\data\TSLAMin.txt'  # Windows server path
FALLBACK_PATH = 'data/TSLAMin.txt'               # Local fallback
START = '2015-01-01'
END = '2025-12-31'
FORWARD_DAYS = 10          # trading days to measure forward return
HOURS_PER_DAY = 6.5
CAPITAL = 100_000.0
HOLD_DAYS = 10             # position held for this many trading days
MAX_FIRE_RATE = 0.15       # signal should fire < 15% of evaluated days

# Cache data globally (loaded once, reused across iterations)
_CACHED = {}


def _load_data():
    """Load and cache all data."""
    if _CACHED:
        return _CACHED

    tsla_path = TSLA_MIN_PATH if os.path.isfile(TSLA_MIN_PATH) else FALLBACK_PATH
    tf_data = load_all_tfs(tsla_path, START, END)
    daily_df = tf_data['daily']
    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates, warmup_bars=260)

    # SPY RSI
    from v15.data.native_tf import fetch_native_tf
    spy = _norm_cols(fetch_native_tf('SPY', 'daily', START, END))
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_rsi = _compute_rsi(spy['close'], 14)

    # Hourly data for forward measurement
    hourly = tf_data['1h'][['high', 'low', 'close']].copy() if len(tf_data.get('1h', [])) > 100 else None

    _CACHED.update({
        'tf_data': tf_data,
        'daily_df': daily_df,
        'state_rows': state_rows,
        'spy_rsi': spy_rsi,
        'hourly': hourly,
    })
    return _CACHED


def _measure_forward(ref_price, ref_date, daily_df, hourly):
    """Measure forward return over FORWARD_DAYS from next-day open."""
    next_day = ref_date + pd.Timedelta(days=1)
    end_date = ref_date + pd.Timedelta(days=FORWARD_DAYS * 2)

    max_dd = 0.0
    min_price = ref_price
    max_price = ref_price

    if hourly is not None:
        fwd = hourly.loc[next_day: end_date]
        if len(fwd) > 0:
            cum_hours = 0.0
            for ts, row in fwd.iterrows():
                cum_hours += 1.0
                if cum_hours > FORWARD_DAYS * HOURS_PER_DAY:
                    break
                if row['low'] < min_price:
                    min_price = row['low']
                if row['high'] > max_price:
                    max_price = row['high']
            final_price = fwd.iloc[min(len(fwd)-1, int(FORWARD_DAYS * HOURS_PER_DAY))]['close']
            fwd_return = (final_price / ref_price) - 1.0
            max_dd = (min_price / ref_price) - 1.0
            return fwd_return, max_dd

    # Daily fallback
    fwd_daily = daily_df.loc[next_day: end_date].head(FORWARD_DAYS)
    if len(fwd_daily) == 0:
        return 0.0, 0.0
    for _, row in fwd_daily.iterrows():
        if row['low'] < min_price:
            min_price = row['low']
    final_price = fwd_daily.iloc[-1]['close']
    fwd_return = (final_price / ref_price) - 1.0
    max_dd = (min_price / ref_price) - 1.0
    return fwd_return, max_dd


def evaluate(program_path: str) -> EvaluationResult:
    """
    Score an evolved bounce signal program.

    Scoring v2:
      - Evaluate every day (no artificial cooldown)
      - Only take new trades when not already in a position (HOLD_DAYS gap)
      - Penalize over-firing (selectivity)
      - Penalize drawdown
    """
    try:
        # Load evolved program
        spec = importlib.util.spec_from_file_location("evolved", program_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        signal_fn = mod.evaluate_bounce_signal
    except Exception as e:
        return EvaluationResult(
            metrics={'combined_score': 0.0, 'error': 1.0},
        )

    try:
        data = _load_data()
        daily_df = data['daily_df']
        state_rows = data['state_rows']
        spy_rsi = data['spy_rsi']
        hourly = data['hourly']

        trades = []
        n_fire = 0          # how many days signal said "take_bounce"
        n_evaluated = 0     # total days evaluated
        position_exit_date = None  # date current position expires

        for row in state_rows:
            date = row['date']
            n_evaluated += 1

            # Build states dict
            states = {}
            for tf in ['5min', '1h', '4h', 'daily', 'weekly', 'monthly']:
                s = row.get(tf)
                if s:
                    states[tf] = s

            # Get SPY RSI
            idx = spy_rsi.index.searchsorted(date)
            rsi_val = float(spy_rsi.iloc[idx - 1]) if 0 < idx <= len(spy_rsi) else 50.0

            # Call evolved signal
            try:
                result = signal_fn(states, rsi_val)
            except Exception:
                continue

            if not result.get('take_bounce', False):
                continue

            n_fire += 1

            # Skip if already in a position (realistic: can't double up)
            if position_exit_date and date <= position_exit_date:
                continue

            # Get reference price
            if date not in daily_df.index:
                di = daily_df.index.searchsorted(date)
                if di >= len(daily_df):
                    continue
                date = daily_df.index[di]

            ref_price = daily_df.loc[date, 'close']
            confidence = result.get('confidence', 0.5)

            # Measure forward return
            fwd_return, max_dd = _measure_forward(ref_price, date, daily_df, hourly)

            # P&L: scale by confidence
            position_size = CAPITAL * confidence
            pnl = position_size * fwd_return

            trades.append({
                'date': date,
                'pnl': pnl,
                'fwd_return': fwd_return,
                'max_dd': max_dd,
                'confidence': confidence,
            })

            # Mark position as occupied for HOLD_DAYS trading days
            fwd_dates = daily_df.index[daily_df.index > date]
            if len(fwd_dates) >= HOLD_DAYS:
                position_exit_date = fwd_dates[HOLD_DAYS - 1]
            else:
                position_exit_date = date + pd.Timedelta(days=HOLD_DAYS * 2)

        if not trades:
            return EvaluationResult(metrics={'combined_score': 0.0, 'n_trades': 0.0})

        # ── Compute score ─────────────────────────────────────────────
        pnls = [t['pnl'] for t in trades]
        total_pnl = sum(pnls)
        n_trades = len(trades)
        avg_pnl = total_pnl / n_trades
        std_pnl = float(np.std(pnls)) if n_trades > 1 else 1.0
        sharpe = avg_pnl / max(std_pnl, 1.0)

        win_rate = sum(1 for p in pnls if p > 0) / n_trades
        avg_dd = float(np.mean([t['max_dd'] for t in trades]))

        # Fire rate: fraction of days the signal fires
        fire_rate = n_fire / max(n_evaluated, 1)

        # Selectivity multiplier: penalize signals that fire too often
        # Ideal: fire_rate < MAX_FIRE_RATE. Above that, linear penalty down to 0.2x
        if fire_rate <= MAX_FIRE_RATE:
            selectivity = 1.0
        else:
            # Linear decay: at fire_rate=1.0, selectivity=0.2
            selectivity = max(0.2, 1.0 - (fire_rate - MAX_FIRE_RATE) / (1.0 - MAX_FIRE_RATE) * 0.8)

        # Drawdown penalty: avg_dd is negative; worse drawdown = lower multiplier
        # avg_dd of 0 = 1.0x, avg_dd of -5% = 0.75x, avg_dd of -10% = 0.5x
        dd_mult = max(0.3, 1.0 + avg_dd * 5.0)

        # Final score
        score = total_pnl * (1.0 + sharpe * 0.1) * (0.5 + win_rate * 0.5) * selectivity * dd_mult

        return EvaluationResult(
            metrics={
                'combined_score': float(score),
                'n_trades': float(n_trades),
                'total_pnl': float(total_pnl),
                'win_rate': float(win_rate),
                'sharpe': float(sharpe),
                'avg_dd': float(avg_dd),
                'fire_rate': float(fire_rate),
                'selectivity': float(selectivity),
            },
        )

    except Exception as e:
        return EvaluationResult(metrics={'combined_score': 0.0, 'error': 1.0})
