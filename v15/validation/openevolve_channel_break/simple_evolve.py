#!/usr/bin/env python3
"""
Phase B3: Evolutionary channel break predictor.

The LLM evolves predict_channel_break() — a function that predicts
whether a price channel will break, which direction, and generates
entry signals timed BEFORE the break.

The evaluator runs channel detection on 5-min TSLA data, extracts all
channel physics features (energy, entropy, OU, squeeze, health, etc.)
plus multi-TF channel data, and passes them to the candidate function.

Usage:
    python -u v15/validation/openevolve_channel_break/simple_evolve.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback

# ── Config ────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 1000
CLAUDE_CMD = r"C:\Users\frank\.local\bin\claude.exe"
MODEL = "sonnet"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_FILE = os.path.join(OUTPUT_DIR, "simple_evolve.log")
BEST_FILE = os.path.join(OUTPUT_DIR, "best_program.py")
STATE_FILE = os.path.join(OUTPUT_DIR, "evolve_state.json")

# Evaluator runs in subprocess to avoid memory leaks
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
            capture_output=True, text=True, timeout=3600,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            log(f"  Eval subprocess failed: {result.stderr[-500:]}")
            return None
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        log(f"  No JSON in eval output: {result.stdout[-300:]}")
        return None
    except subprocess.TimeoutExpired:
        log("  Eval timed out (3600s)")
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
            timeout=900, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            log(f"  LLM stderr: {result.stderr[-200:]}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        log("  LLM timed out (900s)")
        return None
    except Exception as e:
        log(f"  LLM error: {e}")
        return None


def read_program_code(path):
    """Read a program file and return its content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(current_code, current_score, current_metrics, history):
    """Build prompt for LLM to suggest improved channel break predictor."""
    history_str = ""
    if history:
        history_str = "\n\nPrevious attempts (sorted by score, best first):\n"
        for h in sorted(history, key=lambda x: x["score"], reverse=True)[:8]:
            desc = h.get("description", "no description")
            holdout_info = ""
            if h.get('holdout_pnl', 0) != 0 or h.get('holdout_trades', 0) > 0:
                holdout_info = f" [HOLDOUT: ${h.get('holdout_pnl',0):.0f}/{h.get('holdout_trades',0)} trades Sharpe={h.get('holdout_sharpe',0):.3f}]"
            history_str += (
                f"  Score={h['score']:.0f} PnL=${h['pnl']:.0f} "
                f"Trades={h['trades']} WR={h['wr']:.1f}% "
                f"Sharpe={h['sharpe']:.3f} DD={h['dd']:.1f}%{holdout_info} "
                f"-- {desc}\n"
            )

    return f"""You are evolving a channel break predictor for TSLA trading.

The function predict_channel_break() receives rich channel physics features from
detected price channels and must predict:
1. Whether a channel will break in the next ~20 bars (5-min bars)
2. Which direction (up or down)
3. Generate BUY/SELL entry signals timed BEFORE the break

## Available channel_features (dict, 5-min primary channel):
Physics features:
  - energy_ratio: total_energy/binding_energy (>1.2 = break imminent)
  - total_energy, potential_energy, kinetic_energy, binding_energy
  - position_pct: 0=lower bound, 0.5=center, 1=upper bound
  - center_distance: signed -1 to +1
  - momentum_direction: +1=toward upper, -1=toward lower
  - momentum_is_turning: bool
  - momentum_turn_score: 0-1

Channel structure:
  - channel_health: 0=dying, 1=strong
  - squeeze_score: 0-1 (compression -> explosive moves)
  - entropy: Shannon entropy (0=predictable, 1=random)
  - r_squared: regression fit quality (0-1)
  - width_pct: channel width as % of price
  - alternation_ratio: bounce cleanliness (0-1)
  - bounce_count, complete_cycles, quality_score
  - bars_since_last_touch, upper_touches, lower_touches
  - false_break_rate: 0-1 (resilience to false breaks)

Mean-reversion:
  - ou_theta: OU speed (<0.05 = channel failing)
  - ou_half_life: bars to half-revert
  - ou_reversion_score: 0-1
  - oscillation_period, bars_to_next_bounce

Break probability (existing hand-crafted):
  - break_prob, break_prob_up, break_prob_down (0-1)

Trend:
  - slope_pct: channel slope as % per bar
  - channel_direction: 'bull', 'bear', 'sideways'
  - volume_score: 0-1

## multi_tf_features (dict of TF -> dict):
  Higher TFs: '1h', '4h', 'daily' (when available)
  Each has the same feature keys as channel_features.
  Use for confluence: if higher TF agrees, break is more likely.

## recent_bars (pd.DataFrame, last 100 5-min OHLCV bars):
  Columns: open, high, low, close, volume
  Can use for feature engineering: volume patterns, candle analysis, etc.

## Return dict:
  'break_imminent': bool
  'break_direction': 'up' or 'down' or None
  'confidence': float 0-1
  'signal': 'BUY', 'SELL', or None
  'stop_pct': float (0.002-0.030)
  'tp_pct': float (0.005-0.050)

## Current best code (score={current_score:.0f}):
```python
{current_code}
```

## Current best metrics:
  Total P&L: ${current_metrics.get('total_pnl', 0):.0f}
  Trades: {current_metrics.get('n_trades', 0):.0f}
  Win Rate: {current_metrics.get('win_rate', 0):.1f}%
  Sharpe: {current_metrics.get('sharpe', 0):.3f}
  Profit Factor: {current_metrics.get('profit_factor', 0):.3f}
  Max Drawdown: {current_metrics.get('max_drawdown_pct', 0):.1f}%
  Avg P&L per trade: ${current_metrics.get('avg_pnl', 0):.0f}
{history_str}
## Scoring formula:
score = PnL * (1 + max(sharpe,0) * 0.2) * (0.3 + WR * 0.7) * (1 + max(PF-1,0) * 0.1) * trade_mult * dd_mult
Higher is better. Maximize PnL and Sharpe while keeping drawdown low.
Trade count penalty: <50 trades = 0.2x, 200-3000 = 1.0x, >5000 = 0.7x

## What to try:
- Novel combinations of energy_ratio, squeeze_score, health, ou_theta
- Use recent_bars for volume spike detection, candle patterns, volatility expansion
- Multi-TF confluence: require higher TF alignment for entry direction
- Rate-of-change features: is energy_ratio INCREASING? Is health DECREASING?
  (Compute from recent_bars price action as proxy)
- Asymmetric long/short thresholds (market has upward bias)
- Better stop/TP sizing: ATR-based, or dynamic from channel width + squeeze
- Time-of-day filtering using recent_bars index (if timestamps available)
- Position within oscillation cycle: bars_to_next_bounce for timing
- Channel age via complete_cycles: older channels break differently
- Combine false_break_rate with current energy: low false_break + high energy = real break
- Volume confirmation: look for volume expansion in recent_bars near boundaries
- Entropy acceleration: rapidly increasing entropy = structure breakdown

## Rules:
1. You MUST define: def predict_channel_break(channel_features, multi_tf_features, recent_bars) -> dict
2. Import numpy and pandas at the top
3. Keep it robust — use .get() with defaults for all feature access
4. Return dict with all required keys (break_imminent, break_direction, confidence, signal, stop_pct, tp_pct)
5. Do NOT try to modify the evaluator or exit logic — only the predictor function
6. Include a brief # comment at the top describing what you changed

## Output format:
Respond with ONLY a Python code block containing the COMPLETE function definition.
You MUST include the full function starting with `def predict_channel_break(channel_features, multi_tf_features, recent_bars):` — do NOT return partial code, snippets, or diffs.
Return the ENTIRE function from `def` to the final `return` statement.
Start with ```python and end with ```.
"""


def extract_code(llm_response):
    """Extract Python code from LLM response."""
    if not llm_response:
        return None

    pattern = r'```python\s*\n(.*?)```'
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern = r'```\s*\n(.*?)```'
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if 'def predict_channel_break' in code:
            return code

    if 'def predict_channel_break' in llm_response:
        return llm_response.strip()

    return None


def validate_code(code):
    """Basic validation that the code defines required function."""
    if 'def predict_channel_break' not in code:
        return False, "missing predict_channel_break()"
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
            return line[1:].strip()[:100]
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
    log("Phase B3: Channel Break Predictor Evolution")
    log("=" * 60)

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
        log("Evaluating initial program (basic energy+squeeze predictor)...")
        result = evaluate_in_subprocess(INITIAL_PROGRAM)
        if not result or result.get("combined_score", 0) <= 0:
            log(f"Initial evaluation failed: {result}")
            sys.exit(1)

        best_score = result["combined_score"]
        best_metrics = result
        best_code = read_program_code(INITIAL_PROGRAM)
        log(f"Initial: score={best_score:.0f} PnL=${result['total_pnl']:.0f} "
            f"trades={result['n_trades']:.0f} WR={result['win_rate']:.1f}% "
            f"Sharpe={result['sharpe']:.3f} DD={result['max_drawdown_pct']:.1f}%")

        with open(BEST_FILE, "w", encoding="utf-8") as f:
            f.write(best_code)

        history = [{
            "score": best_score, "pnl": result["total_pnl"],
            "trades": result["n_trades"], "wr": result["win_rate"],
            "sharpe": result["sharpe"], "dd": result["max_drawdown_pct"],
            "description": "initial (energy_ratio + squeeze + health)",
        }]
        save_state(0, best_score, best_metrics, history)

    # Evolution loop
    consecutive_errors = 0
    for i in range(start_iteration, MAX_ITERATIONS + 1):
        try:
            log(f"\n--- Iteration {i}/{MAX_ITERATIONS} ---")

            prompt = build_prompt(best_code, best_score, best_metrics, history)
            log("  Calling LLM for channel break predictor code...")
            t0 = time.time()
            response = call_llm(prompt)
            llm_time = time.time() - t0
            log(f"  LLM responded in {llm_time:.1f}s")

            code = extract_code(response)
            if not code:
                resp_len = len(response or '')
                snippet = (response or '')[:200]
                log(f"  Failed to extract code from response "
                    f"({resp_len} chars): {snippet}")
                continue

            valid, reason = validate_code(code)
            if not valid:
                # Log first 200 chars of extracted code for debugging
                log(f"  Invalid code: {reason} (first 200 chars: {code[:200]})")
                continue

            description = extract_description(code)
            log(f"  Change: {description}")

            with tempfile.NamedTemporaryFile(suffix=".py", delete=False,
                                              dir=OUTPUT_DIR, mode="w",
                                              encoding="utf-8") as tf:
                tf.write(code)
                candidate_path = tf.name

            log("  Evaluating...")
            t0 = time.time()
            result = evaluate_in_subprocess(candidate_path)
            eval_time = time.time() - t0

            try:
                os.unlink(candidate_path)
            except OSError:
                pass

            if not result or result.get("combined_score", 0) <= 0:
                error = result.get("error", "unknown") if result else "no result"
                log(f"  Evaluation failed ({eval_time:.0f}s): {error[:200]}")
                continue

            score = result["combined_score"]
            holdout_pnl = result.get('holdout_total_pnl', 0)
            holdout_trades = result.get('holdout_n_trades', 0)
            holdout_sharpe = result.get('holdout_sharpe', 0)
            log(f"  Result ({eval_time:.0f}s): score={score:.0f} "
                f"PnL=${result['total_pnl']:.0f} trades={result['n_trades']:.0f} "
                f"WR={result['win_rate']:.1f}% Sharpe={result['sharpe']:.3f} "
                f"DD={result['max_drawdown_pct']:.1f}%")
            log(f"  Holdout: PnL=${holdout_pnl:.0f} trades={holdout_trades} Sharpe={holdout_sharpe:.3f}")

            history.append({
                "score": score, "pnl": result["total_pnl"],
                "trades": result["n_trades"], "wr": result["win_rate"],
                "sharpe": result["sharpe"], "dd": result["max_drawdown_pct"],
                "holdout_pnl": holdout_pnl, "holdout_trades": holdout_trades,
                "holdout_sharpe": holdout_sharpe,
                "description": description,
            })

            if score > best_score:
                improvement = score - best_score
                log(f"  *** NEW BEST! score={score:.0f} (+{improvement:.0f}) ***")
                best_score = score
                best_metrics = result
                best_code = code
                with open(BEST_FILE, "w", encoding="utf-8") as f:
                    f.write(code)
                numbered = os.path.join(OUTPUT_DIR, f"best_iter{i}.py")
                with open(numbered, "w", encoding="utf-8") as f:
                    f.write(f"# Score={score:.0f} PnL=${result['total_pnl']:.0f} "
                            f"Trades={result['n_trades']:.0f} "
                            f"WR={result['win_rate']:.1f}% "
                            f"Sharpe={result['sharpe']:.3f}\n")
                    f.write(code)
            else:
                log(f"  No improvement (best={best_score:.0f})")

            save_state(i, best_score, best_metrics, history)
            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            log(f"  ITERATION {i} CRASHED: {e}\n{traceback.format_exc()}")
            if consecutive_errors >= 5:
                log("  5 consecutive errors -- aborting to avoid infinite loop")
                break
            continue

    log(f"\n{'=' * 60}")
    log(f"Evolution complete. Best score: {best_score:.0f}")
    log(f"Best metrics: PnL=${best_metrics.get('total_pnl', 0):.0f} "
        f"Sharpe={best_metrics.get('sharpe', 0):.3f} "
        f"DD={best_metrics.get('max_drawdown_pct', 0):.1f}%")
    log(f"Best program saved to: {BEST_FILE}")


if __name__ == "__main__":
    main()
