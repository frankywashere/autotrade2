#!/usr/bin/env python3
"""
Simple evolutionary knob tuning for intraday.

Bypasses OpenEvolve's ProcessPoolExecutor (which crashes on Windows)
and runs everything in a single process with subprocess-based evaluation
to prevent memory leaks.

Usage:
    python -u v15/validation/openevolve_intraday_knobs/simple_evolve.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time

# ── Config ────────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 200
CLAUDE_CMD = r"C:\Users\frank\.local\bin\claude.exe"
MODEL = "sonnet"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_FILE = os.path.join(OUTPUT_DIR, "simple_evolve.log")
BEST_FILE = os.path.join(OUTPUT_DIR, "best_program.py")
STATE_FILE = os.path.join(OUTPUT_DIR, "evolve_state.json")

# Evaluator runs in subprocess to avoid memory leaks
EVAL_SCRIPT = os.path.join(BASE_DIR, "eval_subprocess.py")

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
            timeout=300, encoding="utf-8", errors="replace",
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        log("  LLM timed out (300s)")
        return None
    except Exception as e:
        log(f"  LLM error: {e}")
        return None


def build_prompt(current_knobs, current_score, current_metrics, history):
    """Build prompt for LLM to suggest new knobs."""
    history_str = ""
    if history:
        history_str = "\n\nPrevious attempts (sorted by score, best first):\n"
        for h in sorted(history, key=lambda x: x["score"], reverse=True)[:15]:
            history_str += (
                f"  Score={h['score']:.0f} PnL=${h['pnl']:.0f} "
                f"Trades={h['trades']} WR={h['wr']:.1f}% "
                f"Sharpe={h['sharpe']:.3f} DD={h['dd']:.1f}% "
                f"Knobs={json.dumps(h['knobs'])}\n"
            )

    return f"""You are tuning hyperparameters for an intraday trading algorithm.
The algorithm trades TSLA using multi-timeframe channel analysis with VWAP
and divergence signals, trailing stops, and configurable exit management.

Current best configuration (score={current_score:.0f}):
{json.dumps(current_knobs, indent=2)}

Current best metrics:
  Total P&L: ${current_metrics.get('total_pnl', 0):.0f}
  Trades: {current_metrics.get('n_trades', 0):.0f}
  Win Rate: {current_metrics.get('win_rate', 0):.1f}%
  Sharpe: {current_metrics.get('sharpe', 0):.3f}
  Profit Factor: {current_metrics.get('profit_factor', 0):.3f}
  Max Drawdown: {current_metrics.get('max_drawdown_pct', 0):.1f}%
{history_str}
Knob ranges and descriptions:
  # Signal thresholds (control WHEN entries trigger)
  vwap_thresh:       -0.50 to 0.0   (max VWAP distance for VWAP signal; more negative = stricter)
  d_min:             0.05-0.60      (min daily channel position; higher = only enter when daily is higher)
  h1_min:            0.05-0.40      (min 1h channel position)
  f5_thresh:         0.10-0.50      (max 5-min channel position; lower = require more oversold)
  div_thresh:        0.05-0.50      (min divergence magnitude for div signal)
  div_f5_thresh:     0.10-0.50      (max 5-min CP for div signal)
  min_vol_ratio:     0.0-2.0        (min volume ratio vs 20-bar avg, 0=disabled)

  # Stop / TP / Trail (control exits)
  stop_pct:          0.003-0.025    (initial stop distance as fraction of price)
  tp_pct:            0.005-0.050    (take profit distance as fraction of price)
  trail_base:        0.002-0.020    (base trailing stop distance)
  trail_power:       1-12           (exponent on (1-confidence); higher = tighter at high conf)
  trail_floor:       0.0-0.008      (minimum trail distance regardless of confidence)

  # Execution timing
  exit_grace_bars:       0-15       (1-min bars of grace after entry before stops activate)
  stop_update_secs:      5-600      (how often to ratchet best_price in seconds)
  stop_check_secs:       5-60       (how often to check price vs stop in seconds)
  grace_ratchet_secs:    0-300      (ratchet during grace period, 0=disabled)
  max_hold_bars:         10-156     (5-min bars before timeout, 78=full day)
  eval_interval:         1-4        (evaluate signal every N 5-min bars)
  max_trades_per_day:    1-50       (max entries per day, 0=unlimited)

  # Profit-activated stop
  profit_activated_stop: true/false (stop only fires after trade is in profit)
  max_underwater_mins:   0-600      (force-close if never profitable, 0=disabled)

Scoring: PnL * Sharpe_bonus * WR_bonus * PF_bonus * trade_count_mult * drawdown_mult
Higher is better. Key: maximize PnL and Sharpe while keeping drawdown low.

Strategy tips:
- This is a mean-reversion intraday strategy: it buys when price is oversold (low 5-min CP)
  relative to bullish higher timeframes (daily/1h). VWAP and divergence are two entry types.
- Tighter signal filters (lower f5_thresh, more negative vwap_thresh) = fewer but higher quality trades
- Wider stops = more breathing room but larger losses on bad trades
- Trail power controls how much confidence affects trailing: high power = loose trail at low conf

Suggest a NEW configuration that might improve the score. Try something
different from previous attempts. Be creative but stay within ranges.

IMPORTANT: Respond with ONLY a JSON object containing the knobs. No explanation.
Example: {{"vwap_thresh": -0.15, "d_min": 0.25, ...}}"""


def save_state(iteration, best_score, best_knobs, best_metrics, history):
    """Save evolution state for resume."""
    state = {
        "iteration": iteration,
        "best_score": best_score,
        "best_knobs": best_knobs,
        "best_metrics": {k: v for k, v in best_metrics.items()
                         if isinstance(v, (int, float, str, bool))},
        "history": history[-50:],  # Keep last 50 entries
    }
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def load_state():
    """Load saved state if it exists. Returns None if no state."""
    if not os.path.isfile(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log(f"Warning: could not load state: {e}")
        return None


def write_program(knobs, path):
    """Write a knob configuration as a Python program file."""
    code = '''"""Auto-generated knob configuration."""

def get_knobs() -> dict:
    return %s
''' % repr(knobs)
    with open(path, "w") as f:
        f.write(code)


def parse_knobs(llm_response):
    """Extract knobs dict from LLM response."""
    if not llm_response:
        return None
    # Find JSON in response
    text = llm_response.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        raw = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None

    # Validate and clamp
    try:
        knobs = {
            # Signal thresholds
            "vwap_thresh": max(-0.50, min(0.0, float(raw.get("vwap_thresh", -0.10)))),
            "d_min": max(0.05, min(0.60, float(raw.get("d_min", 0.20)))),
            "h1_min": max(0.05, min(0.40, float(raw.get("h1_min", 0.15)))),
            "f5_thresh": max(0.10, min(0.50, float(raw.get("f5_thresh", 0.35)))),
            "div_thresh": max(0.05, min(0.50, float(raw.get("div_thresh", 0.20)))),
            "div_f5_thresh": max(0.10, min(0.50, float(raw.get("div_f5_thresh", 0.35)))),
            "min_vol_ratio": max(0.0, min(2.0, float(raw.get("min_vol_ratio", 0.8)))),
            # Stop / TP / Trail
            "stop_pct": max(0.003, min(0.025, float(raw.get("stop_pct", 0.008)))),
            "tp_pct": max(0.005, min(0.050, float(raw.get("tp_pct", 0.020)))),
            "trail_base": max(0.002, min(0.020, float(raw.get("trail_base", 0.006)))),
            "trail_power": max(1, min(12, int(raw.get("trail_power", 6)))),
            "trail_floor": max(0.0, min(0.008, float(raw.get("trail_floor", 0.0)))),
            # Execution
            "exit_grace_bars": max(0, min(15, int(raw.get("exit_grace_bars", 5)))),
            "stop_update_secs": max(5, min(600, int(raw.get("stop_update_secs", 60)))),
            "stop_check_secs": max(5, min(60, int(raw.get("stop_check_secs", 5)))),
            "grace_ratchet_secs": max(0, min(300, int(raw.get("grace_ratchet_secs", 60)))),
            "max_hold_bars": max(10, min(156, int(raw.get("max_hold_bars", 78)))),
            "eval_interval": max(1, min(4, int(raw.get("eval_interval", 1)))),
            "max_trades_per_day": max(0, min(50, int(raw.get("max_trades_per_day", 30)))),
            # Profit-activated stop
            "profit_activated_stop": bool(raw.get("profit_activated_stop", False)),
            "max_underwater_mins": max(0, min(600, int(raw.get("max_underwater_mins", 0)))),
        }
        return knobs
    except (ValueError, TypeError):
        return None


def main():
    log("=" * 60)
    log("Simple Evolve: Intraday Knob Tuning (Phase A)")
    log("=" * 60)

    # Check for resume state
    saved = load_state()
    start_iteration = 1

    if saved:
        start_iteration = saved["iteration"] + 1
        best_score = saved["best_score"]
        best_knobs = saved["best_knobs"]
        best_metrics = saved["best_metrics"]
        history = saved["history"]
        log(f"RESUMING from iteration {start_iteration} "
            f"(best score={best_score:.0f}, "
            f"PnL=${best_metrics.get('total_pnl', 0):.0f})")
    else:
        # Evaluate initial program
        initial_program = os.path.join(BASE_DIR, "initial_program.py")
        log("Evaluating initial program...")
        result = evaluate_in_subprocess(initial_program)
        if not result:
            log(f"Initial evaluation failed: {result}")
            sys.exit(1)

        best_score = result.get("combined_score", 0.0)
        best_knobs = result.get("knobs", {})
        best_metrics = result
        log(f"Initial: score={best_score:.0f} PnL=${result['total_pnl']:.0f} "
            f"trades={result['n_trades']:.0f} WR={result['win_rate']:.1f}% "
            f"Sharpe={result['sharpe']:.3f} DD={result['max_drawdown_pct']:.1f}%")

        # Save initial as best
        write_program(best_knobs, BEST_FILE)

        history = [{
            "score": best_score, "pnl": result["total_pnl"],
            "trades": result["n_trades"], "wr": result["win_rate"],
            "sharpe": result["sharpe"], "dd": result["max_drawdown_pct"],
            "knobs": best_knobs,
        }]
        save_state(0, best_score, best_knobs, best_metrics, history)

    # Evolution loop
    for i in range(start_iteration, MAX_ITERATIONS + 1):
        log(f"\n--- Iteration {i}/{MAX_ITERATIONS} ---")

        # Call LLM for new knobs
        prompt = build_prompt(best_knobs, best_score, best_metrics, history)
        log("  Calling LLM...")
        t0 = time.time()
        response = call_llm(prompt)
        llm_time = time.time() - t0
        log(f"  LLM responded in {llm_time:.1f}s")

        knobs = parse_knobs(response)
        if not knobs:
            log(f"  Failed to parse knobs from response: {(response or '')[:200]}")
            continue

        log(f"  Knobs: {json.dumps(knobs)}")

        # Write candidate program
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False,
                                          dir=OUTPUT_DIR, mode="w") as tf:
            tf.write(f'"""Auto-generated iteration {i}."""\n\n'
                     f'def get_knobs() -> dict:\n'
                     f'    return {repr(knobs)}\n')
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

        if not result:
            log(f"  Evaluation failed ({eval_time:.0f}s): {result}")
            continue

        score = result["combined_score"]
        log(f"  Result ({eval_time:.0f}s): score={score:.0f} "
            f"PnL=${result['total_pnl']:.0f} trades={result['n_trades']:.0f} "
            f"WR={result['win_rate']:.1f}% Sharpe={result['sharpe']:.3f} "
            f"DD={result['max_drawdown_pct']:.1f}%")

        history.append({
            "score": score, "pnl": result["total_pnl"],
            "trades": result["n_trades"], "wr": result["win_rate"],
            "sharpe": result["sharpe"], "dd": result["max_drawdown_pct"],
            "knobs": knobs,
        })

        if score > best_score:
            improvement = score - best_score
            log(f"  *** NEW BEST! score={score:.0f} (+{improvement:.0f}) ***")
            best_score = score
            best_knobs = knobs
            best_metrics = result
            write_program(best_knobs, BEST_FILE)
        else:
            log(f"  No improvement (best={best_score:.0f})")

        save_state(i, best_score, best_knobs, best_metrics, history)

    log(f"\n{'=' * 60}")
    log(f"Evolution complete. Best score: {best_score:.0f}")
    log(f"Best knobs: {json.dumps(best_knobs, indent=2)}")
    log(f"Best program saved to: {BEST_FILE}")


if __name__ == "__main__":
    main()
