"""
OpenEvolve evaluator for the recovery-day trade gate.

Train: 2015-2024 (pre-2025 data only — 2025/2026 are holdout)
Test:  held out entirely — not used in scoring

Returns dict with 'combined_score' as primary metric.
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
    print("  [evaluator] Loading data (2015-2024 train only)...")
    t0 = time.time()
    # Load full range — sim windows will restrict to train period
    _DATA = download_data_local(start='2015-01-02', end='2024-12-31')
    print(f"  [evaluator] Data loaded in {time.time() - t0:.1f}s")
    return _DATA


def evaluate(program_path: str) -> dict:
    """Evaluate a recovery-day gate program. Returns dict with combined_score."""

    data = _load_data()

    # Import the evolved program
    try:
        spec = importlib.util.spec_from_file_location("gate_program", program_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        gate_fn = mod.should_take_trade
    except Exception as e:
        print(f"  [evaluator] Import error: {e}")
        return {'combined_score': -1e6, 'error': str(e)}

    from v15.validation.forward_sim_v2 import run_all_scanners_sim

    # ── Train fold 1: 2020-01-02 to 2022-06-30 ──
    try:
        train1_scanners = run_all_scanners_sim(
            data=data,
            sim_start='2020-01-02',
            sim_end='2022-06-30',
            capital=100_000,
            hard_stop_pct=0.02,
            equity_mode='isolated',
            dw_eod_close=False,
            trade_gate=gate_fn,
            verbose=False,
        )
    except Exception as e:
        print(f"  [evaluator] Train1 sim error: {e}")
        traceback.print_exc()
        return {'combined_score': -1e6, 'error': str(e)}

    # ── Train fold 2: 2022-07-01 to 2024-12-31 ──
    try:
        train2_scanners = run_all_scanners_sim(
            data=data,
            sim_start='2022-07-01',
            sim_end='2024-12-31',
            capital=100_000,
            hard_stop_pct=0.02,
            equity_mode='isolated',
            dw_eod_close=False,
            trade_gate=gate_fn,
            verbose=False,
        )
    except Exception as e:
        print(f"  [evaluator] Train2 sim error: {e}")
        traceback.print_exc()
        return {'combined_score': -1e6, 'error': str(e)}

    # Collect metrics
    def _collect(scanners):
        total_trades = 0
        total_wins = 0
        total_pnl = 0.0
        total_hs = 0
        for name in ('CS-5TF', 'CS-DW', 'ML'):
            sc = scanners[name]
            for t in sc.closed_trades:
                total_trades += 1
                if t.pnl > 0:
                    total_wins += 1
                total_pnl += t.pnl
                if t.exit_reason == 'hard_stop':
                    total_hs += 1
        for t in scanners['Intra'].closed_trades:
            total_trades += 1
            if t.pnl > 0:
                total_wins += 1
            total_pnl += t.pnl
            if t.exit_reason == 'hard_stop':
                total_hs += 1
        wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        return total_trades, total_wins, wr, total_pnl, total_hs

    t1_trades, t1_wins, t1_wr, t1_pnl, t1_hs = _collect(train1_scanners)
    t2_trades, t2_wins, t2_wr, t2_pnl, t2_hs = _collect(train2_scanners)

    total_trades = t1_trades + t2_trades
    total_pnl = t1_pnl + t2_pnl
    total_wr = (t1_wins + t2_wins) / total_trades * 100 if total_trades > 0 else 0
    total_hs = t1_hs + t2_hs

    # ── Scoring ──────────────────────────────────────────────────────────────
    # Goal: maximize P&L while keeping WR high and reducing hard stops.
    # Reward consistency across folds.

    pnl_score = total_pnl / 100.0

    # WR bonus
    wr_factor = 1.0
    if total_wr >= 65:
        wr_factor = 1.0 + (total_wr - 65) * 0.02
    elif total_wr < 55:
        wr_factor = 0.5

    # Hard stop penalty
    hs_penalty = total_hs * 500

    # Cross-fold consistency: penalize if one fold is much worse
    consistency_penalty = 0
    if t1_pnl > 0 and t2_pnl < t1_pnl * 0.2:
        consistency_penalty = abs(t1_pnl - t2_pnl) / 50.0
    elif t2_pnl > 0 and t1_pnl < t2_pnl * 0.2:
        consistency_penalty = abs(t2_pnl - t1_pnl) / 50.0

    # Minimum trade count
    trade_penalty = 0
    if total_trades < 500:
        trade_penalty = (500 - total_trades) * 10

    combined = pnl_score * wr_factor - hs_penalty - consistency_penalty - trade_penalty

    print(f"  [evaluator] Score={combined:.0f}  P&L=${total_pnl:+,.0f}  "
          f"WR={total_wr:.1f}%  Trades={total_trades}  HS={total_hs}  "
          f"(fold1: ${t1_pnl:+,.0f}/{t1_wr:.0f}%/{t1_trades}t  "
          f"fold2: ${t2_pnl:+,.0f}/{t2_wr:.0f}%/{t2_trades}t)")

    return {
        'combined_score': combined,
        'total_pnl': total_pnl,
        'total_wr': total_wr,
        'total_trades': total_trades,
        'total_hard_stops': total_hs,
        'fold1_pnl': t1_pnl,
        'fold1_wr': t1_wr,
        'fold1_trades': t1_trades,
        'fold1_hs': t1_hs,
        'fold2_pnl': t2_pnl,
        'fold2_wr': t2_wr,
        'fold2_trades': t2_trades,
        'fold2_hs': t2_hs,
    }
