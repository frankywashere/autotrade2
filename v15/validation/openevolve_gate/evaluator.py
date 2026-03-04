"""
OpenEvolve evaluator for the trade gate function.

Runs the forward sim with the evolved gate and scores the result.
Scoring: combined_score = P&L * WR_factor - hard_stop_penalty
"""
import os
import sys
import importlib
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

# ── Globals (loaded once) ─────────────────────────────────────────────────────
_DATA = None


def _load_data():
    global _DATA
    if _DATA is not None:
        return _DATA

    from v15.validation.forward_sim_v2 import download_data_local
    print("  [evaluator] Loading data...")
    t0 = time.time()
    _DATA = download_data_local(start='2025-01-02', end='2026-02-27')
    print(f"  [evaluator] Data loaded in {time.time() - t0:.1f}s")
    return _DATA


def evaluate(program_path: str):
    """Evaluate a trade gate program.

    Returns EvaluationResult with metrics dict.
    """
    from openevolve.evaluation import EvaluationResult

    data = _load_data()

    # Import the evolved program
    try:
        spec = importlib.util.spec_from_file_location("gate_program", program_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        gate_fn = mod.should_take_trade
    except Exception as e:
        print(f"  [evaluator] Import error: {e}")
        return EvaluationResult(metrics={
            'combined_score': -1e6,
            'pnl': 0, 'wr': 0, 'trades': 0,
            'hard_stops': 999, 'error': str(e),
        })

    # Run sim on train period (2025-01-02 to 2025-09-30)
    from v15.validation.forward_sim_v2 import run_all_scanners_sim

    try:
        train_scanners = run_all_scanners_sim(
            data=data,
            sim_start='2025-01-02',
            sim_end='2025-09-30',
            capital=100_000,
            hard_stop_pct=0.02,
            equity_mode='isolated',
            dw_eod_close=False,
            trade_gate=gate_fn,
            verbose=False,
        )
    except Exception as e:
        print(f"  [evaluator] Train sim error: {e}")
        traceback.print_exc()
        return EvaluationResult(metrics={
            'combined_score': -1e6,
            'pnl': 0, 'wr': 0, 'trades': 0,
            'hard_stops': 999, 'error': str(e),
        })

    # Run sim on test period (2025-10-01 to 2026-02-27)
    try:
        test_scanners = run_all_scanners_sim(
            data=data,
            sim_start='2025-10-01',
            sim_end='2026-02-27',
            capital=100_000,
            hard_stop_pct=0.02,
            equity_mode='isolated',
            dw_eod_close=False,
            trade_gate=gate_fn,
            verbose=False,
        )
    except Exception as e:
        print(f"  [evaluator] Test sim error: {e}")
        traceback.print_exc()
        return EvaluationResult(metrics={
            'combined_score': -1e6,
            'pnl': 0, 'wr': 0, 'trades': 0,
            'hard_stops': 999, 'error': str(e),
        })

    # Collect metrics
    def _collect(scanners):
        total_trades = 0
        total_wins = 0
        total_pnl = 0.0
        total_hs = 0
        total_hs_pnl = 0.0
        for name in ('CS-5TF', 'CS-DW', 'ML'):
            sc = scanners[name]
            for t in sc.closed_trades:
                total_trades += 1
                if t.pnl > 0:
                    total_wins += 1
                total_pnl += t.pnl
                if t.exit_reason == 'hard_stop':
                    total_hs += 1
                    total_hs_pnl += t.pnl
        # Also count intraday (ungated)
        for t in scanners['Intra'].closed_trades:
            total_trades += 1
            if t.pnl > 0:
                total_wins += 1
            total_pnl += t.pnl
            if t.exit_reason == 'hard_stop':
                total_hs += 1
                total_hs_pnl += t.pnl
        wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        return total_trades, total_wins, wr, total_pnl, total_hs, total_hs_pnl

    train_trades, train_wins, train_wr, train_pnl, train_hs, train_hs_pnl = _collect(train_scanners)
    test_trades, test_wins, test_wr, test_pnl, test_hs, test_hs_pnl = _collect(test_scanners)

    total_trades = train_trades + test_trades
    total_pnl = train_pnl + test_pnl
    total_wr = (train_wins + test_wins) / total_trades * 100 if total_trades > 0 else 0
    total_hs = train_hs + test_hs

    # ── Scoring ──────────────────────────────────────────────────────────────
    # Goal: maximize P&L while maintaining good WR and reducing hard stops
    #
    # Components:
    # 1. Total P&L (primary objective)
    # 2. WR bonus: higher WR = better risk profile
    # 3. Hard stop penalty: each hard stop is a failure to filter
    # 4. OOS consistency: penalize if test << train
    # 5. Minimum trade count: don't reward just filtering everything

    # P&L component (normalized to $100K scale)
    pnl_score = total_pnl / 100.0  # $600K = 6000 points

    # WR bonus: above 60% gets bonus, below 55% gets penalty
    wr_factor = 1.0
    if total_wr >= 65:
        wr_factor = 1.0 + (total_wr - 65) * 0.02  # up to 1.7x at 100% WR
    elif total_wr < 55:
        wr_factor = 0.5  # heavy penalty for low WR

    # Hard stop penalty: each hard stop costs 500 points
    hs_penalty = total_hs * 500

    # OOS consistency: penalize if test_pnl < 30% of proportional expectation
    # (test is ~5 months out of 14, so expect ~36% of total)
    oos_penalty = 0
    if train_pnl > 0 and test_pnl < train_pnl * 0.15:
        oos_penalty = abs(train_pnl - test_pnl) / 50.0

    # Minimum trade count: at least 1000 trades or penalty
    trade_penalty = 0
    if total_trades < 1000:
        trade_penalty = (1000 - total_trades) * 10  # each missing trade = 10 points

    combined = pnl_score * wr_factor - hs_penalty - oos_penalty - trade_penalty

    metrics = {
        'combined_score': combined,
        'total_pnl': total_pnl,
        'total_wr': total_wr,
        'total_trades': total_trades,
        'total_hard_stops': total_hs,
        'train_pnl': train_pnl,
        'train_wr': train_wr,
        'train_trades': train_trades,
        'train_hs': train_hs,
        'test_pnl': test_pnl,
        'test_wr': test_wr,
        'test_trades': test_trades,
        'test_hs': test_hs,
    }

    print(f"  [evaluator] Score={combined:.0f}  P&L=${total_pnl:+,.0f}  "
          f"WR={total_wr:.1f}%  Trades={total_trades}  HS={total_hs}  "
          f"(train: ${train_pnl:+,.0f}/{train_wr:.0f}%/{train_trades}t  "
          f"test: ${test_pnl:+,.0f}/{test_wr:.0f}%/{test_trades}t)")

    return EvaluationResult(metrics=metrics)
