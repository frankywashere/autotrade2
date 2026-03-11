"""
Evaluator for OpenEvolve Phase A: Surfer-ML knob tuning.

CRITICAL: All evaluation uses 5-sec honest fill backtester with sequential
stop mode. This is hardcoded — OpenEvolve CANNOT choose fill mode.

Scoring: composite of Sharpe, profit factor, total PnL, and drawdown.
Penalizes extreme trade counts (too few = overfit filter, too many = noise).
"""

import importlib
import os
import sys
import traceback

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

# ── HARDCODED: honest fill data paths (server) ──────────────────────────────
# Try Windows server paths first, then local Mac paths
BAR_DATA_DIR = r'C:\AI\x14\data\bars_5s'
if not os.path.isdir(BAR_DATA_DIR):
    BAR_DATA_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))),
        'data', 'bars_5s')

START = '2025-01-01'
END = '2026-03-01'

# ── Cached data provider (loaded once) ───────────────────────────────────────
_DATA = None


def _load_data():
    """Load 5-sec bar data (cached across evaluations)."""
    global _DATA
    if _DATA is not None:
        return _DATA

    from v15.validation.unified_backtester.data_provider import DataProvider

    tsla_5s = os.path.join(BAR_DATA_DIR, 'TSLA_5s.csv')
    spy_5s = os.path.join(BAR_DATA_DIR, 'SPY_5s.csv')
    if not os.path.isfile(tsla_5s):
        raise FileNotFoundError(f"TSLA 5-sec bars not found: {tsla_5s}")

    spy_path = spy_5s if os.path.isfile(spy_5s) else None

    # VIX 1-min (optional)
    vix_1m = os.path.join(os.path.dirname(BAR_DATA_DIR), 'VIXMin_IB.txt')
    vix_path = vix_1m if os.path.isfile(vix_1m) else None

    _DATA = DataProvider.from_5sec_bars(
        tsla_5s_path=tsla_5s,
        start=START,
        end=END,
        spy_5s_path=spy_path,
        vix_path=vix_path,
        rth_only=False,  # Extended hours
    )
    return _DATA


def evaluate(program_path: str) -> dict:
    """Evaluate a knob configuration. Returns score dict for OpenEvolve."""
    try:
        data = _load_data()
    except Exception as e:
        return {'combined_score': 0.0, 'error': f'data load failed: {e}'}

    # Import the candidate's knobs
    try:
        spec = importlib.util.spec_from_file_location('candidate', program_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        knobs = mod.get_knobs()
    except Exception as e:
        return {'combined_score': 0.0, 'error': f'import failed: {e}'}

    # Validate knob types
    try:
        exit_grace_bars = int(knobs.get('exit_grace_bars', 5))
        stop_update_secs = int(knobs.get('stop_update_secs', 60))
        stop_check_secs = int(knobs.get('stop_check_secs', 5))
        grace_ratchet_secs = int(knobs.get('grace_ratchet_secs', 60))
        profit_activated_stop = bool(knobs.get('profit_activated_stop', False))
        max_underwater_mins = int(knobs.get('max_underwater_mins', 0))
        max_hold_bars = int(knobs.get('max_hold_bars', 60))
        breakout_stop_mult = float(knobs.get('breakout_stop_mult', 1.0))
        eval_interval = int(knobs.get('eval_interval', 3))
    except Exception as e:
        return {'combined_score': 0.0, 'error': f'knob validation failed: {e}'}

    # Clamp to valid ranges
    exit_grace_bars = max(0, min(15, exit_grace_bars))
    stop_update_secs = max(5, min(600, stop_update_secs))
    stop_check_secs = max(5, min(60, stop_check_secs))
    grace_ratchet_secs = max(0, min(300, grace_ratchet_secs))
    max_underwater_mins = max(0, min(600, max_underwater_mins))
    max_hold_bars = max(20, min(9999, max_hold_bars))
    breakout_stop_mult = max(0.05, min(2.0, breakout_stop_mult))
    eval_interval = max(1, min(6, eval_interval))

    # Run backtester programmatically
    try:
        from copy import deepcopy
        from v15.validation.unified_backtester.algos.surfer_ml import (
            SurferMLAlgo, DEFAULT_SURFER_ML_CONFIG,
        )
        from v15.validation.unified_backtester.engine import BacktestEngine
        from v15.validation.unified_backtester.portfolio import PortfolioManager
        from v15.validation.unified_backtester.results import compute_metrics

        config = deepcopy(DEFAULT_SURFER_ML_CONFIG)

        # Apply knobs to config
        config.stop_check_mode = 'sequential'  # HARDCODED — no cheating
        config.exit_grace_bars = exit_grace_bars
        config.stop_update_secs = stop_update_secs
        config.stop_check_secs = stop_check_secs
        config.grace_ratchet_secs = grace_ratchet_secs
        config.profit_activated_stop = profit_activated_stop
        config.max_underwater_mins = max_underwater_mins
        config.max_hold_bars = max_hold_bars
        config.eval_interval = eval_interval
        config.params['breakout_stop_mult'] = breakout_stop_mult

        algo = SurferMLAlgo(config, data)
        portfolio = PortfolioManager()
        portfolio.register_algo(
            algo_id=config.algo_id,
            initial_equity=config.initial_equity,
            max_per_trade=config.max_equity_per_trade,
            max_positions=config.max_positions,
            cost_model=config.cost_model,
        )

        engine = BacktestEngine(data, [algo], portfolio, verbose=False)
        results = engine.run()

        # Extract metrics
        trades = portfolio.get_trades(algo_id='surfer-ml')
        if not trades:
            return {'combined_score': 0.0, 'n_trades': 0, 'error': 'no trades'}

        m = compute_metrics(trades, config.initial_equity)

    except Exception as e:
        return {'combined_score': 0.0, 'error': f'backtest failed: {traceback.format_exc()}'}

    # ── Compute composite score ──────────────────────────────────────────────
    total_pnl = m['total_pnl']
    win_rate = m['win_rate']
    sharpe = m['sharpe_ratio']
    pf = m['profit_factor']
    dd = m['max_drawdown_pct']
    n_trades = m['total_trades']

    # Trade count penalty: too few (<200) = overfit filter, too many (>5000) = noise
    if n_trades < 50:
        trade_mult = 0.2
    elif n_trades < 200:
        trade_mult = 0.5 + 0.5 * (n_trades - 50) / 150
    elif n_trades <= 3000:
        trade_mult = 1.0
    elif n_trades <= 5000:
        trade_mult = 1.0 - 0.3 * (n_trades - 3000) / 2000
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

    # Score: PnL * Sharpe bonus * WR bonus * trade count * drawdown
    # If total_pnl <= 0, score is 0
    if total_pnl <= 0:
        score = 0.0
    else:
        score = (total_pnl
                 * (1.0 + max(sharpe, 0) * 0.2)     # Sharpe bonus
                 * (0.3 + win_rate * 0.7)             # Win rate bonus
                 * (1.0 + max(pf - 1, 0) * 0.1)      # PF bonus
                 * trade_mult
                 * dd_mult)

    return {
        'combined_score': max(score, 0.0),
        'total_pnl': total_pnl,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'profit_factor': pf,
        'max_drawdown_pct': dd,
        'avg_pnl': m['avg_pnl'],
        'knobs': knobs,
    }
