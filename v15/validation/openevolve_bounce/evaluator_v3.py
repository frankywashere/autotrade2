"""
Evaluator v3 for OpenEvolve bounce signal evolution.

Adds TSLA weekly RSI, RSI-SMA relationship, and distance from 52-week SMA
as new inputs to the signal function.

Scoring v2 rules still apply:
  - No artificial cooldown — realistic hold-then-re-enter
  - Selectivity penalty for over-firing
  - Drawdown penalty
"""

import os
import sys
import importlib
import traceback

import numpy as np
import pandas as pd

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)
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

TSLA_MIN_PATH = r'C:\AI\x14\data\TSLAMin.txt'
FALLBACK_PATH = 'data/TSLAMin.txt'
START = '2015-01-01'
END = '2025-12-31'
FORWARD_DAYS = 10
HOURS_PER_DAY = 6.5
CAPITAL = 100_000.0
HOLD_DAYS = 10
MAX_FIRE_RATE = 0.15

_CACHED = {}


def _load_data():
    """Load and cache all data including TSLA weekly RSI and 52-week SMA."""
    if _CACHED:
        return _CACHED

    tsla_path = TSLA_MIN_PATH if os.path.isfile(TSLA_MIN_PATH) else FALLBACK_PATH
    tf_data = load_all_tfs(tsla_path, START, END)
    daily_df = tf_data['daily']
    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates, warmup_bars=260)

    # SPY RSI (daily)
    from v15.data.native_tf import fetch_native_tf
    spy = _norm_cols(fetch_native_tf('SPY', 'daily', START, END))
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_rsi = _compute_rsi(spy['close'], 14)

    # ── TSLA weekly RSI (14-period) + SMA(14) of RSI ──────────────────
    weekly_df = tf_data['weekly']
    tsla_rsi_w = _compute_rsi(weekly_df['close'], 14)
    tsla_rsi_sma = tsla_rsi_w.rolling(14, min_periods=1).mean()

    # ── TSLA 52-week SMA (on daily close, 252 trading days) ───────────
    tsla_52w_sma = daily_df['close'].rolling(252, min_periods=60).mean()

    # Hourly data for forward measurement
    hourly = tf_data['1h'][['high', 'low', 'close']].copy() if len(tf_data.get('1h', [])) > 100 else None

    _CACHED.update({
        'tf_data': tf_data,
        'daily_df': daily_df,
        'state_rows': state_rows,
        'spy_rsi': spy_rsi,
        'hourly': hourly,
        'tsla_rsi_w': tsla_rsi_w,
        'tsla_rsi_sma': tsla_rsi_sma,
        'tsla_52w_sma': tsla_52w_sma,
        'weekly_df': weekly_df,
    })
    return _CACHED


def _lookup_weekly(series, date):
    """Look up the most recent weekly value on or before date."""
    idx = series.index.searchsorted(date, side='right') - 1
    if 0 <= idx < len(series):
        return float(series.iloc[idx])
    return np.nan


def _measure_forward(ref_price, ref_date, daily_df, hourly):
    """Measure forward return over FORWARD_DAYS from next-day open."""
    next_day = ref_date + pd.Timedelta(days=1)
    end_date = ref_date + pd.Timedelta(days=FORWARD_DAYS * 2)

    max_dd = 0.0
    min_price = ref_price

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
            final_price = fwd.iloc[min(len(fwd)-1, int(FORWARD_DAYS * HOURS_PER_DAY))]['close']
            fwd_return = (final_price / ref_price) - 1.0
            max_dd = (min_price / ref_price) - 1.0
            return fwd_return, max_dd

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
    """Score an evolved bounce signal program with TSLA RSI features."""
    try:
        spec = importlib.util.spec_from_file_location("evolved", program_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        signal_fn = mod.evaluate_bounce_signal
    except Exception as e:
        return EvaluationResult(metrics={'combined_score': 0.0, 'error': 1.0})

    try:
        data = _load_data()
        daily_df = data['daily_df']
        state_rows = data['state_rows']
        spy_rsi = data['spy_rsi']
        hourly = data['hourly']
        tsla_rsi_w = data['tsla_rsi_w']
        tsla_rsi_sma = data['tsla_rsi_sma']
        tsla_52w_sma = data['tsla_52w_sma']

        trades = []
        n_fire = 0
        n_evaluated = 0
        position_exit_date = None

        for row in state_rows:
            date = row['date']
            n_evaluated += 1

            states = {}
            for tf in ['5min', '1h', '4h', 'daily', 'weekly', 'monthly']:
                s = row.get(tf)
                if s:
                    states[tf] = s

            # SPY RSI
            idx = spy_rsi.index.searchsorted(date)
            rsi_val = float(spy_rsi.iloc[idx - 1]) if 0 < idx <= len(spy_rsi) else 50.0

            # TSLA weekly RSI + SMA
            tw_rsi = _lookup_weekly(tsla_rsi_w, date)
            tw_sma = _lookup_weekly(tsla_rsi_sma, date)

            # Distance from 52-week SMA
            if date in daily_df.index and date in tsla_52w_sma.index:
                sma_val = tsla_52w_sma.loc[date]
                close_val = daily_df.loc[date, 'close']
                dist_52w = (close_val - sma_val) / sma_val if sma_val > 0 else 0.0
            else:
                dist_52w = 0.0

            # Call evolved signal with new features
            try:
                result = signal_fn(states, rsi_val,
                                   tsla_rsi_w=tw_rsi if not np.isnan(tw_rsi) else 50.0,
                                   tsla_rsi_sma=tw_sma if not np.isnan(tw_sma) else 50.0,
                                   dist_52w_sma=float(dist_52w))
            except Exception:
                continue

            if not result.get('take_bounce', False):
                continue

            n_fire += 1

            if position_exit_date and date <= position_exit_date:
                continue

            if date not in daily_df.index:
                di = daily_df.index.searchsorted(date)
                if di >= len(daily_df):
                    continue
                date = daily_df.index[di]

            ref_price = daily_df.loc[date, 'close']
            confidence = result.get('confidence', 0.5)
            fwd_return, max_dd = _measure_forward(ref_price, date, daily_df, hourly)
            position_size = CAPITAL * confidence
            pnl = position_size * fwd_return

            trades.append({
                'date': date,
                'pnl': pnl,
                'fwd_return': fwd_return,
                'max_dd': max_dd,
                'confidence': confidence,
            })

            fwd_dates = daily_df.index[daily_df.index > date]
            if len(fwd_dates) >= HOLD_DAYS:
                position_exit_date = fwd_dates[HOLD_DAYS - 1]
            else:
                position_exit_date = date + pd.Timedelta(days=HOLD_DAYS * 2)

        if not trades:
            return EvaluationResult(metrics={'combined_score': 0.0, 'n_trades': 0.0})

        pnls = [t['pnl'] for t in trades]
        total_pnl = sum(pnls)
        n_trades = len(trades)
        avg_pnl = total_pnl / n_trades
        std_pnl = float(np.std(pnls)) if n_trades > 1 else 1.0
        sharpe = avg_pnl / max(std_pnl, 1.0)
        win_rate = sum(1 for p in pnls if p > 0) / n_trades
        avg_dd = float(np.mean([t['max_dd'] for t in trades]))
        fire_rate = n_fire / max(n_evaluated, 1)

        if fire_rate <= MAX_FIRE_RATE:
            selectivity = 1.0
        else:
            selectivity = max(0.2, 1.0 - (fire_rate - MAX_FIRE_RATE) / (1.0 - MAX_FIRE_RATE) * 0.8)

        dd_mult = max(0.3, 1.0 + avg_dd * 5.0)
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
