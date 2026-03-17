"""
Evaluator for OpenEvolve Phase B1: Surfer-ML entry logic evolution.

TRAIN/HOLDOUT SPLIT:
  - Training (scored):  2015-01-01 to 2024-12-31, 1-min bars, sequential stop (1-min fallback)
  - Holdout  (report):  2025-01-01 to 2026-03-16, 5-sec bars, sequential stop (5-sec honest fills)

The candidate program defines on_bar() and _compute_current_atr() functions.
These are monkey-patched onto SurferMLAlgo BEFORE the backtest runs.
The ML models are FROZEN -- only the code that uses them changes.
Exit logic is FROZEN -- only entry logic changes.

Scoring: composite of Sharpe, profit factor, total PnL, and drawdown.
Penalizes extreme trade counts (too few = overfit filter, too many = noise).
Only the TRAINING period contributes to the score.
"""

import hashlib
import importlib
import os
import pickle
import sys
import traceback
import types

import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

# -- Data paths: try Windows server first, then local Mac --

# 5-sec bar dir (holdout)
BAR_DATA_DIR = r'C:\AI\x14\data\bars_5s'
if not os.path.isdir(BAR_DATA_DIR):
    BAR_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data', 'bars_5s')

# 1-min data files (training)
_DATA_DIR_WIN = r'C:\AI\x14\data'
_DATA_DIR_MAC = os.path.join(_PROJECT_ROOT, 'data')


def _resolve_data_path(filename: str) -> str:
    """Return full path to a data file, trying Windows then Mac."""
    win = os.path.join(_DATA_DIR_WIN, filename)
    if os.path.isfile(win):
        return win
    mac = os.path.join(_DATA_DIR_MAC, filename)
    if os.path.isfile(mac):
        return mac
    raise FileNotFoundError(
        f"Data file not found: tried {win} and {mac}")


# -- Period boundaries --
TRAIN_START = '2015-01-01'
TRAIN_END = '2024-12-31'
HOLDOUT_START = '2025-01-01'
HOLDOUT_END = '2026-03-16'

# -- Cached data providers (loaded once per process) --
_TRAIN_DATA = None
_HOLDOUT_DATA = None


def _cache_key(*args) -> str:
    """Build a short hash from arguments for cache versioning."""
    h = hashlib.sha256('|'.join(str(a) for a in args).encode()).hexdigest()[:12]
    return h


def _pickle_path(name: str) -> str:
    """Return path to a pickle cache file in the output dir."""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f'{name}.pkl')


def _load_train_data():
    """Load 1-min bar data for training period (cached).

    Uses pickle cache on disk to avoid re-parsing 90MB+ text files on every
    subprocess evaluation.  First call builds the DataProvider and saves it;
    subsequent calls load from pickle (~2-5 s vs ~minutes).
    """
    global _TRAIN_DATA
    if _TRAIN_DATA is not None:
        return _TRAIN_DATA

    from v15.validation.unified_backtester.data_provider import DataProvider

    tsla_path = _resolve_data_path('TSLAMin_yfinance_deprecated.txt')
    spy_path = _resolve_data_path('SPYMin.txt')
    vix_path = _resolve_data_path('VIXMin_IB.txt')

    # Cache version key: hash of paths + date range
    cache_key = _cache_key(tsla_path, spy_path, vix_path, TRAIN_START, TRAIN_END)
    pkl_path = _pickle_path(f'train_data_{cache_key}')

    # Try loading from pickle cache
    if os.path.isfile(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                _TRAIN_DATA = pickle.load(f)
            print(f"[evaluator] Loaded train data from pickle cache ({pkl_path})")
            return _TRAIN_DATA
        except Exception as e:
            print(f"[evaluator] Pickle cache load failed ({e}), rebuilding...")

    _TRAIN_DATA = DataProvider(
        tsla_1min_path=tsla_path,
        start=TRAIN_START,
        end=TRAIN_END,
        spy_path=spy_path if os.path.isfile(spy_path) else None,
        rth_only=False,  # Extended hours
    )
    # VIX needs separate loading via _init_from_df1m override -- reload with vix
    # Actually DataProvider.__init__ calls _init_from_df1m which doesn't take
    # vix_path. We need to load VIX manually.
    if os.path.isfile(vix_path):
        from v15.validation.unified_backtester.data_provider import (
            load_1min, _resample_ohlcv, _RESAMPLE_RULES,
        )
        vix1m = load_1min(vix_path, TRAIN_START, TRAIN_END, rth_only=False)
        if len(vix1m) > 0:
            _TRAIN_DATA._vix1m = vix1m
            _TRAIN_DATA._vix_tf_data = {'1min': vix1m}
            for tf, rule in _RESAMPLE_RULES.items():
                if rule is not None:
                    _TRAIN_DATA._vix_tf_data[tf] = _resample_ohlcv(vix1m, rule)

    # Save to pickle cache for subsequent subprocess evaluations
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(_TRAIN_DATA, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[evaluator] Saved train data pickle cache ({pkl_path})")
    except Exception as e:
        print(f"[evaluator] WARNING: failed to save pickle cache: {e}")

    return _TRAIN_DATA


def _load_holdout_data():
    """Load 5-sec bar data for holdout period (cached).

    Uses pickle cache on disk to avoid re-parsing CSV files on every
    subprocess evaluation.
    """
    global _HOLDOUT_DATA
    if _HOLDOUT_DATA is not None:
        return _HOLDOUT_DATA

    from v15.validation.unified_backtester.data_provider import DataProvider

    tsla_5s = os.path.join(BAR_DATA_DIR, 'TSLA_5s.csv')
    spy_5s = os.path.join(BAR_DATA_DIR, 'SPY_5s.csv')
    if not os.path.isfile(tsla_5s):
        raise FileNotFoundError(f"TSLA 5-sec bars not found: {tsla_5s}")

    spy_path = spy_5s if os.path.isfile(spy_5s) else None

    # VIX 1-min (same file, different date range)
    vix_path = _resolve_data_path('VIXMin_IB.txt')
    vix_path = vix_path if os.path.isfile(vix_path) else None

    # Cache version key
    cache_key = _cache_key(tsla_5s, spy_path, vix_path, HOLDOUT_START, HOLDOUT_END)
    pkl_path = _pickle_path(f'holdout_data_{cache_key}')

    # Try loading from pickle cache
    if os.path.isfile(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                _HOLDOUT_DATA = pickle.load(f)
            print(f"[evaluator] Loaded holdout data from pickle cache ({pkl_path})")
            return _HOLDOUT_DATA
        except Exception as e:
            print(f"[evaluator] Pickle cache load failed ({e}), rebuilding...")

    _HOLDOUT_DATA = DataProvider.from_5sec_bars(
        tsla_5s_path=tsla_5s,
        start=HOLDOUT_START,
        end=HOLDOUT_END,
        spy_5s_path=spy_path,
        vix_path=vix_path,
        rth_only=False,  # Extended hours
    )

    # Save to pickle cache
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(_HOLDOUT_DATA, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[evaluator] Saved holdout data pickle cache ({pkl_path})")
    except Exception as e:
        print(f"[evaluator] WARNING: failed to save pickle cache: {e}")

    return _HOLDOUT_DATA


def _run_backtest(data, candidate_on_bar, candidate_compute_atr,
                  stop_check_mode: str, stop_check_secs: int,
                  stop_update_secs: int, grace_ratchet_secs: int):
    """Run a single backtest with the given data and config. Returns (metrics, error)."""
    from copy import deepcopy
    from v15.validation.unified_backtester.algos.surfer_ml import (
        SurferMLAlgo, DEFAULT_SURFER_ML_CONFIG,
    )
    from v15.validation.unified_backtester.engine import BacktestEngine
    from v15.validation.unified_backtester.portfolio import PortfolioManager
    from v15.validation.unified_backtester.results import compute_metrics

    config = deepcopy(DEFAULT_SURFER_ML_CONFIG)

    # HARDCODED config -- same as Phase A best or defaults
    config.stop_check_mode = stop_check_mode
    config.exit_grace_bars = 5
    config.stop_update_secs = stop_update_secs
    config.stop_check_secs = stop_check_secs
    config.grace_ratchet_secs = grace_ratchet_secs
    config.profit_activated_stop = True
    config.max_underwater_mins = 0
    config.max_hold_bars = 60
    config.eval_interval = 3

    algo = SurferMLAlgo(config, data)

    # Monkey-patch entry logic
    algo.on_bar = types.MethodType(candidate_on_bar, algo)
    algo._compute_current_atr = types.MethodType(candidate_compute_atr, algo)

    portfolio = PortfolioManager()
    portfolio.register_algo(
        algo_id=config.algo_id,
        initial_equity=config.initial_equity,
        max_per_trade=config.max_equity_per_trade,
        max_positions=config.max_positions,
        cost_model=config.cost_model,
    )

    engine = BacktestEngine(data, [algo], portfolio, verbose=False)
    engine.run()

    trades = portfolio.get_trades(algo_id='surfer-ml')
    if not trades:
        return None, 'no trades'

    m = compute_metrics(trades, config.initial_equity)
    return m, None


def _compute_score(m: dict) -> float:
    """Compute composite score from metrics dict."""
    total_pnl = m['total_pnl']
    win_rate = m['win_rate']
    sharpe = m['sharpe_ratio']
    pf = m['profit_factor']
    dd = m['max_drawdown_pct']
    n_trades = m['total_trades']

    # Trade count penalty: too few (<50) = overfit filter, too many (>5000) = noise
    # Scaled up for 10-year training period vs old 1-year period
    if n_trades < 200:
        trade_mult = 0.2
    elif n_trades < 800:
        trade_mult = 0.5 + 0.5 * (n_trades - 200) / 600
    elif n_trades <= 15000:
        trade_mult = 1.0
    elif n_trades <= 25000:
        trade_mult = 1.0 - 0.3 * (n_trades - 15000) / 10000
    else:
        trade_mult = 0.7

    # Drawdown penalty: higher dd = lower multiplier
    if dd <= 10:
        dd_mult = 1.0
    elif dd <= 20:
        dd_mult = 1.0 - 0.3 * (dd - 10) / 10
    elif dd <= 40:
        dd_mult = 0.7 - 0.4 * (dd - 20) / 20
    else:
        dd_mult = 0.3

    # Score: PnL * Sharpe bonus * WR bonus * PF bonus * trade count * drawdown
    if total_pnl <= 0:
        return 0.0

    score = (total_pnl
             * (1.0 + max(sharpe, 0) * 0.2)     # Sharpe bonus
             * (0.3 + win_rate * 0.7)             # Win rate bonus
             * (1.0 + max(pf - 1, 0) * 0.1)      # PF bonus
             * trade_mult
             * dd_mult)

    return max(score, 0.0)


def evaluate(program_path: str) -> dict:
    """Evaluate an entry logic candidate. Returns score dict for OpenEvolve.

    Runs TWO backtests:
      1. Training (2015-2024, 1-min, sequential stop at 1-min resolution) -- SCORED
      2. Holdout  (2025-2026, 5-sec, sequential stop mode) -- REPORTED ONLY
    """
    # Import the candidate's entry functions
    try:
        spec = importlib.util.spec_from_file_location('candidate', program_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        candidate_on_bar = getattr(mod, 'on_bar', None)
        candidate_compute_atr = getattr(mod, '_compute_current_atr', None)

        if candidate_on_bar is None:
            return {'combined_score': 0.0, 'error': 'missing on_bar()'}
        if candidate_compute_atr is None:
            return {'combined_score': 0.0, 'error': 'missing _compute_current_atr()'}

    except Exception as e:
        return {'combined_score': 0.0, 'error': f'import failed: {e}'}

    # ── 1. Training backtest (1-min data, sequential stop at 1-min resolution) ─
    try:
        train_data = _load_train_data()
    except Exception as e:
        return {'combined_score': 0.0, 'error': f'train data load failed: {e}'}

    try:
        train_m, train_err = _run_backtest(
            data=train_data,
            candidate_on_bar=candidate_on_bar,
            candidate_compute_atr=candidate_compute_atr,
            # Sequential mode with 1-min data: engine auto-falls back to
            # _check_sequential_stops (checks bar low/high vs stop each 1-min bar).
            # stop_check_secs/stop_update_secs only matter for 5-sec sub-loop.
            stop_check_mode='sequential',
            stop_check_secs=60,
            stop_update_secs=60,
            grace_ratchet_secs=60,
        )
    except Exception as e:
        return {'combined_score': 0.0,
                'error': f'train backtest failed: {traceback.format_exc()}'}

    if train_m is None:
        return {'combined_score': 0.0, 'n_trades': 0, 'error': f'train: {train_err}'}

    # Score from training period only
    score = _compute_score(train_m)

    result = {
        'combined_score': score,
        'total_pnl': train_m['total_pnl'],
        'n_trades': train_m['total_trades'],
        'win_rate': train_m['win_rate'],
        'sharpe': train_m['sharpe_ratio'],
        'profit_factor': train_m['profit_factor'],
        'max_drawdown_pct': train_m['max_drawdown_pct'],
        'avg_pnl': train_m['avg_pnl'],
    }

    # ── 2. Holdout backtest (5-sec data, sequential stop mode) ───────────────
    try:
        holdout_data = _load_holdout_data()
        holdout_m, holdout_err = _run_backtest(
            data=holdout_data,
            candidate_on_bar=candidate_on_bar,
            candidate_compute_atr=candidate_compute_atr,
            # Full honest sequential mode for holdout
            stop_check_mode='sequential',
            stop_check_secs=5,
            stop_update_secs=60,
            grace_ratchet_secs=60,
        )
        if holdout_m is not None:
            result['holdout_total_pnl'] = holdout_m['total_pnl']
            result['holdout_n_trades'] = holdout_m['total_trades']
            result['holdout_win_rate'] = holdout_m['win_rate']
            result['holdout_sharpe'] = holdout_m['sharpe_ratio']
            result['holdout_profit_factor'] = holdout_m['profit_factor']
            result['holdout_max_drawdown_pct'] = holdout_m['max_drawdown_pct']
            result['holdout_avg_pnl'] = holdout_m['avg_pnl']
        else:
            result['holdout_error'] = holdout_err
    except Exception as e:
        result['holdout_error'] = f'holdout failed: {traceback.format_exc()}'

    return result
