"""
Evaluator v5: OOS-validated bounce signal with drawdown + extended indicators.

Key changes from v3:
  - Passes additional indicators: daily RSI, daily return, dd_from_peak,
    MACD histogram, stochastic %K, ATR %
  - OOS scoring: evaluates on BOTH train (2016-2021) and test (2022-2025)
    periods, penalizes signals that only work in-sample
  - Rewards signals that work in the harder 2022-2025 regime
"""
import os, sys, importlib, time, traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from v15.validation.tf_state_backtest import (
    load_all_tfs, compute_daily_states, _norm_cols,
)
from v15.validation.bounce_timing import _compute_rsi
from v15.data.native_tf import fetch_native_tf
from v15.features.utils import calc_macd, calc_stochastic, calc_atr

# ── Globals (loaded once) ─────────────────────────────────────────────────────
_DATA = None

def _load_data():
    global _DATA
    if _DATA is not None:
        return _DATA

    start, end = '2015-01-01', '2025-12-31'

    # Try Windows path first, then local
    tsla_path = r'C:\AI\x14\data\TSLAMin.txt'
    if not os.path.isfile(tsla_path):
        tsla_path = 'data/TSLAMin.txt'

    tf_data = load_all_tfs(tsla_path, start, end)
    daily_df = tf_data['daily'].copy()
    weekly_df = tf_data['weekly'].copy()

    # SPY
    spy = _norm_cols(fetch_native_tf('SPY', 'daily', start, end))
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_rsi = _compute_rsi(spy['close'], 14)

    # TSLA RSI
    tsla_rsi_d = _compute_rsi(daily_df['close'], 14)
    tsla_rsi_w = _compute_rsi(weekly_df['close'], 14)
    tsla_rsi_w_sma = tsla_rsi_w.rolling(14, min_periods=1).mean()
    tsla_52w_sma = daily_df['close'].rolling(252, min_periods=60).mean()

    # MACD (daily)
    d_close = daily_df['close'].values
    _, _, macd_hist_d = calc_macd(d_close)
    macd_hist_d_s = pd.Series(macd_hist_d, index=daily_df.index)

    # Stochastic (daily)
    stoch_k, _ = calc_stochastic(
        daily_df['high'].values, daily_df['low'].values, d_close
    )
    stoch_k_s = pd.Series(stoch_k, index=daily_df.index)

    # ATR normalized
    atr_d = calc_atr(daily_df['high'].values, daily_df['low'].values, d_close, 14)
    atr_pct_s = pd.Series(atr_d / d_close, index=daily_df.index)

    # Drawdown from 20-day peak
    daily_df['rolling_max_20'] = daily_df['high'].rolling(20, min_periods=1).max()
    daily_df['dd_from_peak'] = (daily_df['close'] / daily_df['rolling_max_20'] - 1) * 100
    daily_df['daily_return'] = daily_df['close'].pct_change() * 100

    # TF states
    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates, warmup_bars=260)

    _DATA = {
        'daily_df': daily_df,
        'weekly_df': weekly_df,
        'spy_rsi': spy_rsi,
        'tsla_rsi_d': tsla_rsi_d,
        'tsla_rsi_w': tsla_rsi_w,
        'tsla_rsi_w_sma': tsla_rsi_w_sma,
        'tsla_52w_sma': tsla_52w_sma,
        'macd_hist_d': macd_hist_d_s,
        'stoch_k': stoch_k_s,
        'atr_pct': atr_pct_s,
        'state_rows': state_rows,
    }
    return _DATA


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


FORWARD_DAYS = 10
HOLD_DAYS = 10
POSITION_SIZE = 100_000


def evaluate(program_path: str) -> dict:
    """Evaluate a bounce signal program. Returns {'combined_score': float, ...}."""
    try:
        data = _load_data()
    except Exception as e:
        return {'combined_score': 0.0, 'error': f'data load failed: {e}'}

    # Load the signal function
    try:
        spec = importlib.util.spec_from_file_location('candidate', program_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        signal_fn = mod.evaluate_bounce_signal
    except Exception as e:
        return {'combined_score': 0.0, 'error': f'import failed: {e}'}

    daily_df = data['daily_df']
    state_rows = data['state_rows']
    dates = daily_df.index
    closes = daily_df['close'].values.astype(float)
    opens = daily_df['open'].values.astype(float)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # Run signal on all state rows
    trades = []
    n_evaluated = 0
    n_fire = 0
    last_entry_idx = -HOLD_DAYS - 1

    for row in state_rows:
        date = row['date']
        if date.year < 2016:
            continue

        states = {tf: row.get(tf) for tf in ['5min', '1h', '4h', 'daily', 'weekly', 'monthly']}
        if not states.get('daily') or not states.get('weekly'):
            continue

        di = date_to_idx.get(date)
        if di is None or di + FORWARD_DAYS >= len(dates):
            continue

        # Can't fire if we're in a trade
        if di - last_entry_idx < HOLD_DAYS:
            continue

        n_evaluated += 1

        # Build indicator inputs
        spy_val = 50.0
        idx = data['spy_rsi'].index.searchsorted(date)
        if 0 < idx <= len(data['spy_rsi']):
            spy_val = float(data['spy_rsi'].iloc[idx - 1])

        tw_rsi = _lookup_weekly(data['tsla_rsi_w'], date)
        tw_sma = _lookup_weekly(data['tsla_rsi_w_sma'], date)
        td_rsi = _lookup_daily(data['tsla_rsi_d'], date)
        macd_h = _lookup_daily(data['macd_hist_d'], date)
        sk = _lookup_daily(data['stoch_k'], date)
        atr_p = _lookup_daily(data['atr_pct'], date)
        dd_peak = daily_df.loc[date, 'dd_from_peak'] if date in daily_df.index else 0.0
        day_ret = daily_df.loc[date, 'daily_return'] if date in daily_df.index else 0.0

        # 52w SMA distance
        dist_52w = 0.0
        if date in data['tsla_52w_sma'].index:
            sma_val = data['tsla_52w_sma'].loc[date]
            if sma_val > 0:
                dist_52w = (closes[di] - sma_val) / sma_val

        try:
            result = signal_fn(
                states, spy_val,
                tsla_rsi_w=tw_rsi if not np.isnan(tw_rsi) else 50.0,
                tsla_rsi_sma=tw_sma if not np.isnan(tw_sma) else 50.0,
                dist_52w_sma=float(dist_52w),
                tsla_rsi_d=td_rsi if not np.isnan(td_rsi) else 50.0,
                daily_return=float(day_ret) if not np.isnan(day_ret) else 0.0,
                dd_from_peak=float(dd_peak) if not np.isnan(dd_peak) else 0.0,
                macd_hist_d=macd_h if not np.isnan(macd_h) else 0.0,
                stoch_k=sk if not np.isnan(sk) else 50.0,
                atr_pct=atr_p if not np.isnan(atr_p) else 0.03,
            )
        except Exception:
            continue

        take = result.get('take_bounce', False)
        if not take:
            continue

        n_fire += 1
        conf = float(result.get('confidence', 0.5))
        conf = max(0.01, min(1.0, conf))

        # Entry at next day open
        entry_idx = di + 1
        entry_price = opens[entry_idx]
        exit_idx = min(entry_idx + FORWARD_DAYS, len(dates) - 1)
        exit_price = closes[exit_idx]

        fwd_return = (exit_price / entry_price) - 1.0
        fwd_slice = daily_df.iloc[entry_idx:exit_idx + 1]
        max_dd = (fwd_slice['low'].min() / entry_price - 1.0) if len(fwd_slice) > 0 else 0.0

        pnl = POSITION_SIZE * fwd_return * conf
        trades.append({
            'date': date,
            'year': date.year,
            'entry': entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'fwd_return': fwd_return,
            'max_dd': max_dd,
            'conf': conf,
        })
        last_entry_idx = entry_idx

    if len(trades) == 0:
        return {'combined_score': 0.0, 'n_trades': 0, 'error': 'no trades'}

    # ── Compute metrics ───────────────────────────────────────────────────────
    total_pnl = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl'] > 0]
    win_rate = len(wins) / len(trades)
    pnls = [t['pnl'] for t in trades]
    avg_pnl = np.mean(pnls)
    std_pnl = np.std(pnls) if len(pnls) > 1 else 1.0
    sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0
    avg_dd = np.mean([t['max_dd'] for t in trades])

    # Fire rate penalty
    fire_rate = n_fire / max(n_evaluated, 1)
    if fire_rate <= 0.15:
        selectivity = 1.0
    elif fire_rate <= 0.25:
        selectivity = 1.0 - (fire_rate - 0.15) * 5.0
    else:
        selectivity = max(0.2, 0.5 - (fire_rate - 0.25) * 2.0)

    # Drawdown multiplier
    drawdown_mult = max(0.3, 1.0 + avg_dd * 5.0)

    # ── OOS scoring: split by period ──────────────────────────────────────────
    train_trades = [t for t in trades if t['year'] <= 2021]
    test_trades = [t for t in trades if t['year'] >= 2022]

    train_pnl = sum(t['pnl'] for t in train_trades) if train_trades else 0
    test_pnl = sum(t['pnl'] for t in test_trades) if test_trades else 0
    train_wr = (sum(1 for t in train_trades if t['pnl'] > 0) / len(train_trades)) if train_trades else 0
    test_wr = (sum(1 for t in test_trades if t['pnl'] > 0) / len(test_trades)) if test_trades else 0

    # OOS bonus: reward signals profitable in BOTH periods
    oos_mult = 1.0
    if len(test_trades) >= 3:
        if test_pnl > 0 and test_wr >= 0.45:
            # Strong OOS: bonus proportional to test performance
            oos_mult = 1.0 + min(test_pnl / max(abs(train_pnl), 1), 1.0) * 0.5
        elif test_pnl <= 0:
            # Failed OOS: heavy penalty
            oos_mult = 0.3
    elif len(test_trades) == 0:
        oos_mult = 0.5  # no test trades = suspicious

    # ── Combined score ────────────────────────────────────────────────────────
    score = (total_pnl
             * (1.0 + sharpe * 0.1)
             * (0.5 + win_rate * 0.5)
             * selectivity
             * drawdown_mult
             * oos_mult)

    return {
        'combined_score': max(score, 0.0),
        'total_pnl': total_pnl,
        'n_trades': len(trades),
        'win_rate': win_rate,
        'sharpe': sharpe,
        'avg_dd': avg_dd,
        'fire_rate': fire_rate,
        'selectivity': selectivity,
        'oos_mult': oos_mult,
        'train_pnl': train_pnl,
        'train_n': len(train_trades),
        'train_wr': train_wr,
        'test_pnl': test_pnl,
        'test_n': len(test_trades),
        'test_wr': test_wr,
    }
