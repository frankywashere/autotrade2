#!/usr/bin/env python3
"""
Simple evolutionary knob tuning for intraday (v3 — PnL-dominant scoring).

Improvements over v1:
- Forced exploration rounds (every 3rd iteration)
- Diverse history (top 5 + 10 diverse entries spanning different regimes)
- Random perturbation (2-3 knobs bumped 20-50% of range after LLM suggests)
- Anti-repeat detection (5 consecutive similar configs → forced big jump)
- Multiple starting basins (3 different seed configs tried at start)

Usage:
    python -u v15/validation/openevolve_intraday_knobs/simple_evolve.py
"""

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback

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

# Exploration config
EXPLORE_EVERY = 3       # Every Nth iteration is forced exploration
PERTURB_KNOBS = 2       # Number of knobs to randomly perturb
PERTURB_FRAC = 0.3      # Fraction of range to perturb by
REPEAT_WINDOW = 5       # Consecutive similar configs before forced jump
REPEAT_SIMILARITY = 0.02  # Max fractional difference to count as "similar"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Knob ranges (used for perturbation and diversity) ─────────────────────────
KNOB_RANGES = {
    "vwap_thresh":         (-0.50, 0.0,   "float"),
    "d_min":               (0.05,  0.60,  "float"),
    "h1_min":              (0.05,  0.40,  "float"),
    "f5_thresh":           (0.10,  0.50,  "float"),
    "div_thresh":          (0.05,  0.50,  "float"),
    "div_f5_thresh":       (0.10,  0.50,  "float"),
    "min_vol_ratio":       (0.0,   2.0,   "float"),
    "stop_pct":            (0.003, 0.025, "float"),
    "tp_pct":              (0.005, 0.050, "float"),
    "trail_base":          (0.002, 0.020, "float"),
    "trail_power":         (1,     12,    "int"),
    "trail_floor":         (0.0,   0.008, "float"),
    "exit_grace_bars":     (0,     15,    "int"),
    "stop_update_secs":    (5,     600,   "int"),
    "stop_check_secs":     (5,     60,    "int"),
    "grace_ratchet_secs":  (0,     300,   "int"),
    "max_hold_bars":       (10,    156,   "int"),
    "eval_interval":       (1,     4,     "int"),
    "max_trades_per_day":  (1,     50,    "int"),
    "max_underwater_mins": (0,     600,   "int"),
}

# ── Alternative starting seeds (different basins) ────────────────────────────
SEED_CONFIGS = [
    {  # Seed A: Loose filters, tight stops, high frequency
        "vwap_thresh": -0.05, "d_min": 0.10, "h1_min": 0.08,
        "f5_thresh": 0.45, "div_thresh": 0.10, "div_f5_thresh": 0.40,
        "min_vol_ratio": 0.0, "stop_pct": 0.005, "tp_pct": 0.015,
        "trail_base": 0.004, "trail_power": 3, "trail_floor": 0.002,
        "exit_grace_bars": 2, "stop_update_secs": 30, "stop_check_secs": 5,
        "grace_ratchet_secs": 0, "max_hold_bars": 40, "eval_interval": 1,
        "max_trades_per_day": 20, "profit_activated_stop": False,
        "max_underwater_mins": 0,
    },
    {  # Seed B: Strict filters, wide stops, low frequency
        "vwap_thresh": -0.30, "d_min": 0.45, "h1_min": 0.30,
        "f5_thresh": 0.15, "div_thresh": 0.35, "div_f5_thresh": 0.20,
        "min_vol_ratio": 1.5, "stop_pct": 0.018, "tp_pct": 0.040,
        "trail_base": 0.010, "trail_power": 8, "trail_floor": 0.005,
        "exit_grace_bars": 10, "stop_update_secs": 120, "stop_check_secs": 10,
        "grace_ratchet_secs": 120, "max_hold_bars": 100, "eval_interval": 3,
        "max_trades_per_day": 5, "profit_activated_stop": True,
        "max_underwater_mins": 60,
    },
    {  # Seed C: No volume filter, moderate everything, profit-activated
        "vwap_thresh": -0.20, "d_min": 0.30, "h1_min": 0.20,
        "f5_thresh": 0.25, "div_thresh": 0.15, "div_f5_thresh": 0.30,
        "min_vol_ratio": 0.0, "stop_pct": 0.012, "tp_pct": 0.030,
        "trail_base": 0.007, "trail_power": 5, "trail_floor": 0.003,
        "exit_grace_bars": 7, "stop_update_secs": 60, "stop_check_secs": 5,
        "grace_ratchet_secs": 90, "max_hold_bars": 60, "eval_interval": 2,
        "max_trades_per_day": 10, "profit_activated_stop": True,
        "max_underwater_mins": 120,
    },
]


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
        proc = subprocess.Popen(
            [sys.executable, EVAL_SCRIPT, program_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        try:
            stdout, stderr = proc.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            log("  Eval timed out (300s) — killing process tree")
            _kill_tree(proc.pid)
            proc.wait(timeout=5)
            return None
        if proc.returncode != 0:
            log(f"  Eval subprocess failed: {stderr[-500:]}")
            return None
        # Last line of stdout is the JSON result
        lines = stdout.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        log(f"  No JSON in eval output: {stdout[-300:]}")
        return None
    except Exception as e:
        log(f"  Eval error: {e}")
        return None


def call_llm(prompt):
    """Call Claude via CLI with robust timeout + process kill."""
    try:
        proc = subprocess.Popen(
            [CLAUDE_CMD, "--print", "--model", MODEL,
             "--dangerously-skip-permissions"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace",
        )
        try:
            stdout, stderr = proc.communicate(input=prompt, timeout=120)
            return stdout.strip()
        except subprocess.TimeoutExpired:
            log("  LLM timed out (120s) — killing process tree")
            _kill_tree(proc.pid)
            proc.wait(timeout=5)
            return None
    except Exception as e:
        log(f"  LLM error: {e}")
        return None


def _kill_tree(pid):
    """Kill a process and all its children (Windows-compatible)."""
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True, timeout=10,
        )
    except Exception:
        try:
            os.kill(pid, 9)
        except Exception:
            pass


def _knob_distance(k1, k2):
    """Compute normalized distance between two knob configs (0-1 scale)."""
    total = 0.0
    count = 0
    for key, (lo, hi, typ) in KNOB_RANGES.items():
        v1 = float(k1.get(key, 0))
        v2 = float(k2.get(key, 0))
        rng = hi - lo
        if rng > 0:
            total += abs(v1 - v2) / rng
            count += 1
    return total / max(count, 1)


def _select_diverse_history(history, n_top=5, n_diverse=10):
    """Select top N by score + N diverse entries spanning different regimes."""
    if len(history) <= n_top + n_diverse:
        return history

    sorted_h = sorted(history, key=lambda x: x["score"], reverse=True)
    top = sorted_h[:n_top]

    # Select diverse entries from remaining
    remaining = sorted_h[n_top:]
    diverse = []
    selected_knobs = [h["knobs"] for h in top]

    for _ in range(n_diverse):
        if not remaining:
            break
        # Pick the entry most distant from all already selected
        best_idx = 0
        best_min_dist = -1
        for idx, h in enumerate(remaining):
            min_dist = min(_knob_distance(h["knobs"], sk)
                          for sk in selected_knobs)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        chosen = remaining.pop(best_idx)
        diverse.append(chosen)
        selected_knobs.append(chosen["knobs"])

    return top + diverse


def _is_repeat(knobs, recent_knobs):
    """Check if knobs are too similar to all recent configs."""
    if len(recent_knobs) < REPEAT_WINDOW:
        return False
    for rk in recent_knobs[-REPEAT_WINDOW:]:
        if _knob_distance(knobs, rk) > REPEAT_SIMILARITY:
            return False
    return True


def _perturb_knobs(knobs, n_perturb=PERTURB_KNOBS, frac=PERTURB_FRAC):
    """Randomly perturb N knobs by a fraction of their range."""
    knobs = dict(knobs)
    numeric_keys = list(KNOB_RANGES.keys())
    to_perturb = random.sample(numeric_keys, min(n_perturb, len(numeric_keys)))

    for key in to_perturb:
        lo, hi, typ = KNOB_RANGES[key]
        rng = hi - lo
        delta = random.uniform(-frac, frac) * rng
        val = float(knobs.get(key, (lo + hi) / 2)) + delta
        if typ == "int":
            val = int(round(max(lo, min(hi, val))))
        else:
            val = round(max(lo, min(hi, val)), 6)
        knobs[key] = val

    return knobs


def _random_knobs():
    """Generate completely random knobs for a big jump."""
    knobs = {}
    for key, (lo, hi, typ) in KNOB_RANGES.items():
        if typ == "int":
            knobs[key] = random.randint(int(lo), int(hi))
        else:
            knobs[key] = round(random.uniform(lo, hi), 6)
    knobs["profit_activated_stop"] = random.choice([True, False])
    return knobs


def build_prompt(current_knobs, current_score, current_metrics, history,
                 explore_mode=False):
    """Build prompt for LLM to suggest new knobs."""
    # Use diverse history selection
    display_history = _select_diverse_history(history)

    history_str = ""
    if display_history:
        history_str = "\n\nPrevious attempts (top 5 by score + 10 diverse):\n"
        for h in display_history:
            history_str += (
                f"  Score={h['score']:.0f} PnL=${h['pnl']:.0f} "
                f"Trades={h['trades']} WR={h['wr']:.1f}% "
                f"Sharpe={h['sharpe']:.3f} DD={h['dd']:.1f}% "
                f"Knobs={json.dumps(h['knobs'])}\n"
            )

    explore_instruction = ""
    if explore_mode:
        explore_instruction = """
*** EXPLORATION ROUND ***
You MUST try something RADICALLY different from all previous attempts.
Change at least 5 knobs by large amounts. Try a completely different regime:
- If previous bests used strict filters, try loose filters with tight stops
- If previous bests used few trades/day, try many trades with tighter exits
- If previous bests used high trail_power, try low trail_power with wide trail_base
- Try disabling profit_activated_stop if it was enabled, or vice versa
- Try very different max_hold_bars (short holds vs full-day holds)
Do NOT make small tweaks — the goal is to discover new scoring basins.
"""

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
{explore_instruction}
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
        "history": history[-100:],  # Keep last 100 entries (more for diversity)
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


def eval_knobs(knobs, label=""):
    """Evaluate a knob config and return (score, result) or (None, None)."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False,
                                      dir=OUTPUT_DIR, mode="w") as tf:
        tf.write(f'"""Auto-generated {label}."""\n\n'
                 f'def get_knobs() -> dict:\n'
                 f'    return {repr(knobs)}\n')
        candidate_path = tf.name

    t0 = time.time()
    result = evaluate_in_subprocess(candidate_path)
    eval_time = time.time() - t0

    try:
        os.unlink(candidate_path)
    except OSError:
        pass

    if not result:
        log(f"  Evaluation failed ({eval_time:.0f}s)")
        return None, None

    score = result["combined_score"]
    log(f"  {label} ({eval_time:.0f}s): score={score:.0f} "
        f"PnL=${result['total_pnl']:.0f} trades={result['n_trades']:.0f} "
        f"WR={result['win_rate']:.1f}% Sharpe={result['sharpe']:.3f} "
        f"DD={result['max_drawdown_pct']:.1f}%")
    return score, result


def main():
    log("=" * 60)
    log("Simple Evolve v2: Intraday Knob Tuning (with exploration)")
    log("=" * 60)

    # Check for resume state
    saved = load_state()
    start_iteration = 1
    recent_knobs = []  # Track recent configs for anti-repeat

    if saved:
        start_iteration = saved["iteration"] + 1
        best_score = saved["best_score"]
        best_knobs = saved["best_knobs"]
        best_metrics = saved["best_metrics"]
        history = saved["history"]
        # Populate recent_knobs from history tail
        recent_knobs = [h["knobs"] for h in history[-REPEAT_WINDOW:]]
        log(f"RESUMING from iteration {start_iteration} "
            f"(best score={best_score:.0f}, "
            f"PnL=${best_metrics.get('total_pnl', 0):.0f})")
    else:
        # ── Phase 0: Evaluate multiple starting basins ────────────────────
        log("Phase 0: Evaluating seed configs from different basins...")

        # Evaluate initial program
        initial_program = os.path.join(BASE_DIR, "initial_program.py")
        log("  Evaluating initial (default) config...")
        result = evaluate_in_subprocess(initial_program)
        if not result:
            log(f"Initial evaluation failed: {result}")
            sys.exit(1)

        best_score = result.get("combined_score", 0.0)
        best_knobs = result.get("knobs", {})
        best_metrics = result
        log(f"  Default: score={best_score:.0f} PnL=${result['total_pnl']:.0f} "
            f"trades={result['n_trades']:.0f}")

        history = [{
            "score": best_score, "pnl": result["total_pnl"],
            "trades": result["n_trades"], "wr": result["win_rate"],
            "sharpe": result["sharpe"], "dd": result["max_drawdown_pct"],
            "knobs": best_knobs,
        }]

        # Evaluate alternative seeds
        for idx, seed in enumerate(SEED_CONFIGS):
            label = f"Seed {chr(65 + idx)}"
            log(f"  Evaluating {label}...")
            score, res = eval_knobs(seed, label)
            if res is None:
                continue
            history.append({
                "score": score, "pnl": res["total_pnl"],
                "trades": res["n_trades"], "wr": res["win_rate"],
                "sharpe": res["sharpe"], "dd": res["max_drawdown_pct"],
                "knobs": seed,
            })
            if score > best_score:
                log(f"  *** {label} is new best! score={score:.0f} ***")
                best_score = score
                best_knobs = seed
                best_metrics = res

        write_program(best_knobs, BEST_FILE)
        save_state(0, best_score, best_knobs, best_metrics, history)
        log(f"Phase 0 complete. Best score: {best_score:.0f}")

    # ── Evolution loop ────────────────────────────────────────────────────
    consecutive_errors = 0
    for i in range(start_iteration, MAX_ITERATIONS + 1):
      try:
        is_explore = (i % EXPLORE_EVERY == 0)
        is_anti_repeat = _is_repeat(best_knobs, recent_knobs)

        mode = "EXPLORE" if is_explore else ("ANTI-REPEAT" if is_anti_repeat else "exploit")
        log(f"\n--- Iteration {i}/{MAX_ITERATIONS} [{mode}] ---")

        if is_anti_repeat:
            # Stuck in a rut — generate random knobs
            log("  Anti-repeat triggered! Generating random config...")
            knobs = _random_knobs()
            log(f"  Random knobs: {json.dumps(knobs)}")
        else:
            # Call LLM for new knobs
            prompt = build_prompt(best_knobs, best_score, best_metrics,
                                  history, explore_mode=is_explore)
            log("  Calling LLM...")
            t0 = time.time()
            response = call_llm(prompt)
            llm_time = time.time() - t0
            log(f"  LLM responded in {llm_time:.1f}s")

            knobs = parse_knobs(response)
            if not knobs:
                log(f"  Failed to parse knobs: {(response or '')[:200]}")
                continue

            # Apply random perturbation on non-explore rounds
            if not is_explore:
                knobs = _perturb_knobs(knobs)

            log(f"  Knobs: {json.dumps(knobs)}")

        # Evaluate
        score, result = eval_knobs(knobs, f"iter {i}")
        if result is None:
            continue

        consecutive_errors = 0  # Reset on success

        recent_knobs.append(knobs)
        if len(recent_knobs) > REPEAT_WINDOW * 2:
            recent_knobs = recent_knobs[-REPEAT_WINDOW * 2:]

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

      except Exception as e:
        consecutive_errors += 1
        log(f"  ITERATION {i} CRASHED: {e}\n{traceback.format_exc()}")
        if consecutive_errors >= 5:
            log("  5 consecutive errors — aborting to avoid infinite loop")
            break
        continue

    log(f"\n{'=' * 60}")
    log(f"Evolution complete. Best score: {best_score:.0f}")
    log(f"Best knobs: {json.dumps(best_knobs, indent=2)}")
    log(f"Best program saved to: {BEST_FILE}")


if __name__ == "__main__":
    main()
