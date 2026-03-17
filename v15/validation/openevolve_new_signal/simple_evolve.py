#!/usr/bin/env python3
"""
Phase B2: Evolutionary discovery of new cross-asset TSLA/SPY/VIX signal.

Clean-slate signal search: NO channel physics, NO existing ML models.
The LLM writes raw Python that reads 5-min TSLA/SPY/VIX bars (RTH only,
09:30-16:00 ET) and decides when to enter trades. The evaluator backtests
on 2015-2024 (scored) and 2025-01 to 2025-09 (holdout, reported only).

Usage:
    python -u v15/validation/openevolve_new_signal/simple_evolve.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback

# ── Config ────────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 1000
CLAUDE_CMD = r"C:\Users\frank\.local\bin\claude.exe"
MODEL = "sonnet"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_FILE = os.path.join(OUTPUT_DIR, "simple_evolve.log")
BEST_FILE = os.path.join(OUTPUT_DIR, "best_program.py")
STATE_FILE = os.path.join(OUTPUT_DIR, "evolve_state.json")

EVAL_SCRIPT = os.path.join(BASE_DIR, "eval_subprocess.py")
INITIAL_PROGRAM = os.path.join(BASE_DIR, "initial_program.py")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    """Log to file and stdout."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} - {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def evaluate_in_subprocess(program_path):
    """Run evaluator in a fresh subprocess to avoid memory leaks."""
    try:
        result = subprocess.run(
            [sys.executable, EVAL_SCRIPT, program_path],
            capture_output=True, text=True, timeout=600,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            log(f"  Eval subprocess failed: {result.stderr[-500:]}")
            return None
        # Last line of stdout is the JSON result
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        log(f"  No JSON in eval output: {result.stdout[-300:]}")
        return None
    except subprocess.TimeoutExpired:
        log("  Eval timed out (600s)")
        return None
    except Exception as e:
        log(f"  Eval error: {e}")
        return None


def call_llm(prompt):
    """Call Claude via CLI."""
    try:
        result = subprocess.run(
            [CLAUDE_CMD, "--print", "--model", MODEL,
             "--dangerously-skip-permissions"],
            input=prompt, capture_output=True, text=True,
            timeout=600, encoding="utf-8", errors="replace",
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        log("  LLM timed out (600s)")
        return None
    except Exception as e:
        log(f"  LLM error: {e}")
        return None


def read_program_code(path):
    """Read a program file and return its content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(current_code, current_score, current_metrics, history):
    """Build prompt for LLM to discover new cross-asset trading signals."""
    history_str = ""
    if history:
        history_str = "\n\nPrevious attempts (sorted by score, best first):\n"
        for h in sorted(history, key=lambda x: x["score"], reverse=True)[:10]:
            desc = h.get("description", "no description")
            holdout_info = ""
            if 'holdout_pnl' in h:
                holdout_info = f" [HOLDOUT: ${h['holdout_pnl']:.0f}/{h['holdout_trades']} trades]"
            history_str += (
                f"  Score={h['score']:.0f} PnL=${h['pnl']:.0f} "
                f"Trades={h['trades']} WR={h['wr']:.1f}% "
                f"Sharpe={h['sharpe']:.3f} DD={h['dd']:.1f}%"
                f"{holdout_info} -- {desc}\n"
            )

    holdout_str = ""
    hp = current_metrics.get('holdout_pnl', 0)
    ht = current_metrics.get('holdout_trades', 0)
    hs = current_metrics.get('holdout_sharpe', 0)
    if ht > 0:
        holdout_str = f"""
## Holdout period (2025-01 to 2025-09, NOT used for scoring — reality check only):
  PnL: ${hp:.0f}  Trades: {ht}  Sharpe: {hs:.3f}
  WARNING: If holdout PnL is much worse than training, you are likely overfitting.
  Aim for signals that generalize — structural cross-asset relationships, not curve-fitted patterns.
"""

    return f"""You are discovering NEW trading signals for TSLA using cross-asset data.
This is a CLEAN-SLATE search — no existing models, no channel physics.

You must write a Python function: generate_signals()

## Function signature:
```python
def generate_signals(tsla_bars, spy_bars, vix_bars, current_time, position_info):
    \"\"\"
    Args:
        tsla_bars: pd.DataFrame with columns [open, high, low, close, volume]
                   Last 100 FIVE-MINUTE RTH bars ending at current_time (DatetimeIndex)
                   RTH = 09:30-16:00 ET. ~78 bars per trading day.
        spy_bars: pd.DataFrame, same format, same timestamps
        vix_bars: pd.DataFrame, same format (VIX index values in OHLC)
        current_time: pd.Timestamp (current bar's timestamp, always during RTH)
        position_info: dict with 'has_long': bool, 'has_short': bool,
                       'n_positions': int, 'max_positions': int (=2)
    Returns:
        list of dicts: [{{'direction': 'long'/'short', 'confidence': 0-1,
                         'stop_pct': float (e.g. 0.005 = 0.5%),
                         'tp_pct': float (e.g. 0.008 = 0.8%)}}]
        Empty list = no signal.
    \"\"\"
```

## Execution model:
- Called once per 5-MINUTE bar during RTH 09:30-16:00 ET (2015-2024 training, 2025-01 to 2025-09 holdout)
- Entry at current bar's close, exits at stop/TP/timeout (max 78 bars = ~1 trading day)
- Max 2 simultaneous positions, flat $100K sizing per trade
- Simple fixed stop/TP exits (the stop_pct and tp_pct you return)
- Typical stop: 0.1%-2%. Typical TP: 0.2%-3%. These are INTRADAY moves.
- Slippage: 0.05% per side. Commission: $2 round-trip.

## Current best code (score={current_score:.0f}):
```python
{current_code}
```

## Current best metrics (TRAINING period 2015-2024):
  Total P&L: ${current_metrics.get('total_pnl', 0):.0f}
  Trades: {current_metrics.get('n_trades', 0):.0f}
  Win Rate: {current_metrics.get('win_rate', 0):.1f}%
  Sharpe: {current_metrics.get('sharpe', 0):.3f}
  Profit Factor: {current_metrics.get('profit_factor', 0):.3f}
  Max Drawdown: {current_metrics.get('max_drawdown_pct', 0):.1f}%
  Avg P&L per trade: ${current_metrics.get('avg_pnl', 0):.0f}
  Avg hold (5-min bars): {current_metrics.get('avg_hold', 0):.1f}
{holdout_str}{history_str}
## Scoring formula:
score = PnL * (1 + max(sharpe,0) * 0.2) * (0.3 + WR * 0.7) * (1 + max(PF-1,0) * 0.1) * trade_mult * dd_mult
- trade_mult penalizes <500 trades (overfit) or >25000 trades (noise)
- dd_mult penalizes >10% drawdown
- Minimum 500 trades on training period required for full score

## Ideas to explore (you can combine these freely):
- **Cross-asset momentum**: TSLA vs SPY relative strength, TSLA/SPY ratio mean-reversion
- **Volatility regime**: VIX level thresholds, VIX spikes, VIX mean-reversion
- **Volume patterns**: TSLA volume spikes relative to moving average, volume-price divergence
- **Multi-period momentum**: Different lookback periods (5, 12, 25, 50, 78 bars) combined
- **Mean reversion**: Bollinger band touches, distance from moving averages, z-scores
- **Trend following**: Moving average crossovers, breakout from N-bar range
- **Cross-asset divergence**: When TSLA decouples from SPY, catch the reversion
- **Intraday patterns**: Opening range breakout, VWAP deviation, time-of-day effects
- **Risk-adjusted entry**: Scale stop_pct and tp_pct based on recent realized volatility
- **Adaptive parameters**: Different behavior in high-vol vs low-vol regimes
- **Time-of-day effects**: First/last hour patterns, lunch hour lull, opening drive

## Rules:
1. You MUST define: def generate_signals(tsla_bars, spy_bars, vix_bars, current_time, position_info)
2. Return a list of dicts with 'direction', 'confidence', 'stop_pct', 'tp_pct'
3. You may import numpy and pandas (already available)
4. Do NOT use any external data — only the bars passed in
5. Keep code robust — handle edge cases (empty arrays, division by zero)
6. Include a brief # comment at the top describing your approach
7. Aim for GENERALIZABLE signals — structural relationships, not curve-fitted patterns

## Output format:
Respond with ONLY a Python code block. No explanation outside the code.
Start with ```python and end with ```.
"""


def extract_code(llm_response):
    """Extract Python code from LLM response."""
    if not llm_response:
        return None

    # Try ```python ... ```
    pattern = r'```python\s*\n(.*?)```'
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ``` ... ```
    pattern = r'```\s*\n(.*?)```'
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if 'def generate_signals' in code:
            return code

    # If response looks like code
    if 'def generate_signals' in llm_response:
        return llm_response.strip()

    return None


def validate_code(code):
    """Basic validation that the code defines the required function."""
    if 'def generate_signals' not in code:
        return False, "missing generate_signals()"
    # Check it accepts the right number of arguments
    if 'generate_signals(' not in code:
        return False, "generate_signals not callable"
    # Try to compile
    try:
        compile(code, '<candidate>', 'exec')
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    return True, "ok"


def extract_description(code):
    """Extract the first comment line as a description of changes."""
    for line in code.split('\n'):
        line = line.strip()
        if line.startswith('#') and len(line) > 3:
            return line[1:].strip()[:120]
    return "no description"


def save_state(iteration, best_score, best_metrics, history):
    """Save evolution state for resume."""
    state = {
        "iteration": iteration,
        "best_score": best_score,
        "best_metrics": {k: v for k, v in best_metrics.items()
                         if isinstance(v, (int, float, str, bool))},
        "history": history[-50:],
    }
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def load_state():
    """Load saved state if it exists."""
    if not os.path.isfile(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log(f"Warning: could not load state: {e}")
        return None


def main():
    log("=" * 60)
    log("Phase B2: New Cross-Asset Signal Discovery")
    log("=" * 60)
    log(f"Training: 2015-01-01 to 2024-12-31 (scored)")
    log(f"Holdout:  2025-01-01 to 2025-09-27 (reported only)")
    log(f"Data: 5-min TSLA/SPY/VIX bars from 1-min files (RTH only)")
    log(f"Max iterations: {MAX_ITERATIONS}")

    # Check for resume state
    saved = load_state()
    start_iteration = 1

    if saved:
        start_iteration = saved["iteration"] + 1
        best_score = saved["best_score"]
        best_metrics = saved["best_metrics"]
        history = saved["history"]
        if os.path.isfile(BEST_FILE):
            best_code = read_program_code(BEST_FILE)
        else:
            best_code = read_program_code(INITIAL_PROGRAM)
        log(f"RESUMING from iteration {start_iteration} "
            f"(best score={best_score:.0f}, "
            f"PnL=${best_metrics.get('total_pnl', 0):.0f})")
    else:
        # Evaluate initial program
        log("Evaluating initial program...")
        result = evaluate_in_subprocess(INITIAL_PROGRAM)
        if not result:
            log(f"Initial evaluation failed: {result}")
            sys.exit(1)

        best_score = result.get("combined_score", 0)
        best_metrics = result
        best_code = read_program_code(INITIAL_PROGRAM)

        log(f"Initial: score={best_score:.0f} "
            f"PnL=${result.get('total_pnl', 0):.0f} "
            f"trades={result.get('n_trades', 0):.0f} "
            f"WR={result.get('win_rate', 0):.1f}% "
            f"Sharpe={result.get('sharpe', 0):.3f} "
            f"DD={result.get('max_drawdown_pct', 0):.1f}%")

        holdout_pnl = result.get('holdout_pnl', 0)
        holdout_trades = result.get('holdout_trades', 0)
        log(f"  Holdout: PnL=${holdout_pnl:.0f} trades={holdout_trades}")

        # Save initial as best
        with open(BEST_FILE, "w", encoding="utf-8") as f:
            f.write(best_code)

        history = [{
            "score": best_score,
            "pnl": result.get("total_pnl", 0),
            "trades": result.get("n_trades", 0),
            "wr": result.get("win_rate", 0),
            "sharpe": result.get("sharpe", 0),
            "dd": result.get("max_drawdown_pct", 0),
            "holdout_pnl": holdout_pnl,
            "holdout_trades": holdout_trades,
            "description": "initial (RSI + VIX spike + SPY trend)",
        }]
        save_state(0, best_score, best_metrics, history)

    # Evolution loop
    consecutive_errors = 0
    for i in range(start_iteration, MAX_ITERATIONS + 1):
        try:
            log(f"\n--- Iteration {i}/{MAX_ITERATIONS} ---")

            # Call LLM for new signal code
            prompt = build_prompt(best_code, best_score, best_metrics, history)
            log("  Calling LLM for signal code...")
            t0 = time.time()
            response = call_llm(prompt)
            llm_time = time.time() - t0
            log(f"  LLM responded in {llm_time:.1f}s")

            code = extract_code(response)
            if not code:
                log(f"  Failed to extract code from response "
                    f"({len(response or '')} chars)")
                continue

            valid, reason = validate_code(code)
            if not valid:
                log(f"  Invalid code: {reason}")
                continue

            description = extract_description(code)
            log(f"  Approach: {description}")

            # Write candidate to temp file
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False,
                                              dir=OUTPUT_DIR, mode="w",
                                              encoding="utf-8") as tf:
                tf.write(code)
                candidate_path = tf.name

            # Evaluate
            log("  Evaluating...")
            t0 = time.time()
            result = evaluate_in_subprocess(candidate_path)
            eval_time = time.time() - t0

            # Clean up temp file
            try:
                os.unlink(candidate_path)
            except OSError:
                pass

            if not result or result.get("combined_score", 0) <= 0:
                if result and 'total_pnl' in result:
                    log(f"  Score=0 ({eval_time:.0f}s): PnL=${result['total_pnl']:.0f} trades={result.get('n_trades',0)} WR={result.get('win_rate',0):.1f}% Sharpe={result.get('sharpe',0):.3f}")
                else:
                    error = result.get("error", "unknown") if result else "no result"
                    log(f"  Evaluation failed ({eval_time:.0f}s): {error[:200]}")
                continue

            score = result["combined_score"]
            holdout_pnl = result.get('holdout_pnl', 0)
            holdout_trades = result.get('holdout_trades', 0)

            log(f"  Result ({eval_time:.0f}s): score={score:.0f} "
                f"PnL=${result['total_pnl']:.0f} "
                f"trades={result['n_trades']:.0f} "
                f"WR={result['win_rate']:.1f}% "
                f"Sharpe={result['sharpe']:.3f} "
                f"DD={result['max_drawdown_pct']:.1f}%")
            log(f"  Holdout: PnL=${holdout_pnl:.0f} "
                f"trades={holdout_trades}")

            history.append({
                "score": score,
                "pnl": result["total_pnl"],
                "trades": result["n_trades"],
                "wr": result["win_rate"],
                "sharpe": result["sharpe"],
                "dd": result["max_drawdown_pct"],
                "holdout_pnl": holdout_pnl,
                "holdout_trades": holdout_trades,
                "description": description,
            })

            if score > best_score:
                improvement = score - best_score
                log(f"  *** NEW BEST! score={score:.0f} "
                    f"(+{improvement:.0f}) ***")
                best_score = score
                best_metrics = result
                best_code = code
                with open(BEST_FILE, "w", encoding="utf-8") as f:
                    f.write(code)
                # Numbered copy
                numbered = os.path.join(OUTPUT_DIR, f"best_iter{i}.py")
                with open(numbered, "w", encoding="utf-8") as f:
                    f.write(f"# Score={score:.0f} "
                            f"PnL=${result['total_pnl']:.0f} "
                            f"Trades={result['n_trades']:.0f} "
                            f"WR={result['win_rate']:.1f}% "
                            f"Sharpe={result['sharpe']:.3f} "
                            f"Holdout=${holdout_pnl:.0f}\n")
                    f.write(code)
            else:
                log(f"  No improvement (best={best_score:.0f})")

            save_state(i, best_score, best_metrics, history)
            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            log(f"  ITERATION {i} CRASHED: {e}\n{traceback.format_exc()}")
            if consecutive_errors >= 5:
                log("  5 consecutive errors -- aborting")
                break
            continue

    log(f"\n{'=' * 60}")
    log(f"Evolution complete. Best score: {best_score:.0f}")
    log(f"Best metrics: PnL=${best_metrics.get('total_pnl', 0):.0f} "
        f"Sharpe={best_metrics.get('sharpe', 0):.3f} "
        f"DD={best_metrics.get('max_drawdown_pct', 0):.1f}%")
    log(f"Holdout: PnL=${best_metrics.get('holdout_pnl', 0):.0f}")
    log(f"Best program saved to: {BEST_FILE}")


if __name__ == "__main__":
    main()
