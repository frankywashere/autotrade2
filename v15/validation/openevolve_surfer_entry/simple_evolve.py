#!/usr/bin/env python3
"""
Phase B1: Evolutionary entry logic tuning for surfer-ml.

The LLM rewrites the actual Python code for on_bar() and _compute_current_atr().
The evaluator monkey-patches these onto SurferMLAlgo and runs the full backtest.
ML models are frozen -- the LLM evolves the code that uses them.
Exit logic is frozen -- only entry logic changes.

Usage:
    python -u v15/validation/openevolve_surfer_entry/simple_evolve.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback

# -- Config --
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
        # Last line of stdout is the JSON result
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        log(f"  No JSON in eval output: {result.stdout[-300:]}")
        return None
    except subprocess.TimeoutExpired:
        log("  Eval timed out (1200s)")
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
    """Build prompt for LLM to suggest improved entry logic code."""
    history_str = ""
    if history:
        history_str = "\n\nPrevious attempts (sorted by score, best first):\n"
        for h in sorted(history, key=lambda x: x["score"], reverse=True)[:8]:
            desc = h.get("description", "no description")
            history_str += (
                f"  Score={h['score']:.0f} PnL=${h['pnl']:.0f} "
                f"Trades={h['trades']} WR={h['wr']:.1f}% "
                f"Sharpe={h['sharpe']:.3f} DD={h['dd']:.1f}% "
                f"-- {desc}\n"
            )

    return f"""You are evolving ENTRY logic for a TSLA trading algorithm (surfer-ml).
The exit logic is FROZEN -- you can ONLY change how entries work.
The ML models (GBT, EL, ER, fast_rev) are FROZEN -- you can only change how their outputs are used.

You must write TWO Python functions: on_bar() and _compute_current_atr().
These are monkey-patched as methods onto the algo instance (use `self`).

## self (SurferMLAlgo instance) fields:
  self.data -- DataProvider with get_bars(tf, time, symbol='TSLA'/'SPY'/'VIX')
    TFs: '5min', '1h', '4h', 'daily'. Returns DataFrame with OHLCV columns.
  self.config.algo_id -- 'surfer-ml'
  self.config.eval_interval -- 3 (check every 3 bars = 15 min)
  self.config.params -- dict with:
    'min_confidence' (0.01), 'stop_pct' (0.015), 'tp_pct' (0.012),
    'atr_period' (14), 'breakout_stop_mult' (1.00), 'ou_half_life' (5.0)
  self._gbt_model -- dict of Boosters: {{'action': Booster, ...}} or None
  self._el_model -- Extreme Loser Booster or None
  self._er_model -- Extended Run Booster or None
  self._fast_rev_model -- Fast Reversion Booster or None
  self._el_derive, self._er_derive, self._fr_derive -- callable(full_vec) -> subset or None
  self._history_buffer -- list (for build_feature_vector temporal features)
  self._feature_names -- list or None
  self._pos_state -- dict[pos_id, dict]

## on_bar parameters:
  time -- pd.Timestamp of current bar
  bar -- dict with 'open', 'high', 'low', 'close', 'volume'
  open_positions -- list of Position objects with .direction, .signal_type, etc.
  context -- TradeContext (passed to build_feature_vector)

## on_bar must return: List[Signal]
  Signal(algo_id, direction, price, confidence, stop_pct, tp_pct, signal_type, metadata)
  Import: from v15.validation.unified_backtester.algo_base import Signal

## Channel detection pipeline (KEEP THIS STRUCTURE):
  from v15.core.channel import detect_channels_multi_window, select_best_channel
  from v15.core.channel_surfer import analyze_channels, TF_WINDOWS
  - detect_channels_multi_window(df, windows=[...]) -> multi-window results
  - select_best_channel(multi) -> (best_channel, score)
  - analyze_channels(channels_by_tf, prices_by_tf, current_prices, volumes_by_tf) -> analysis
  - analysis.signal has: .action ('BUY'/'SELL'/'HOLD'), .confidence, .signal_type ('bounce'/'break'),
    .suggested_stop_pct, .suggested_tp_pct

## ML feature vector (KEEP THIS CALL):
  from v15.core.signal_features import build_feature_vector
  feature_vec, _ = build_feature_vector(analysis, bar_data, closes, spy_df, vix_df,
                                         tsla_index, history_buffer, eval_interval, context, bars_df)

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

## What you CAN change (entry logic):
- Channel detection window sizes (e.g., [10, 15, 20, 30, 40] for 5-min)
- Higher TF window sizes (via TF_WINDOWS or custom)
- Minimum bar count thresholds (currently 20 for 5-min, 30 for higher TFs)
- Confidence thresholds and how confidence is scaled
- GBT soft gate: how ML agreement/disagreement scales confidence (currently 0.80 penalty)
- EL threshold (currently 0.18) and how it affects confidence
- ER thresholds (0.55, 0.70) and trail_width_mult values (1.5, 2.0)
- Fast reversion threshold (currently 0.55)
- ATR stop/TP sizing: floor/cap ratios for bounce vs break
- Breakout stop multiplier
- TP widening logic (currently 1.30x for bounce conf > 0.65)
- Anti-pyramid rules (currently blocks same-direction AND same-type)
- Adding time-of-day filters (market hours, avoid open/close, etc.)
- Adding volatility filters (reject signals during extreme volatility)
- Adding trend alignment filters (e.g., skip longs when daily channel is bearish)
- Using SPY/VIX data for entry gating
- Using ER/EL probabilities to adjust stop_pct/tp_pct, not just metadata
- Different lookback lengths for 5-min slice (currently 100 bars)

## What is FROZEN (do NOT change):
- Exit logic (get_effective_stop, check_exits) -- handled separately
- ML model weights (GBT, EL, ER, fast_rev) -- models are loaded externally
- build_feature_vector() call signature -- keep the same parameters
- Signal metadata keys: el_flagged, trail_width_mult, fast_reversion, ou_half_life,
  signal_bar_high, signal_bar_low -- exit logic depends on these

## Rules:
1. You MUST define both on_bar(self, time, bar, open_positions, context=None) and _compute_current_atr(self, time)
2. on_bar returns a list of Signal objects (usually 0 or 1 signals)
3. _compute_current_atr returns a float (ATR value)
4. Import Signal inside on_bar: from v15.validation.unified_backtester.algo_base import Signal
5. Do NOT modify exit logic, ML model loading, or on_position_opened
6. Keep it simple enough to not crash. If in doubt, keep existing logic and change one thing.
7. Include a brief # comment at the top describing what you changed
8. ALWAYS include the signal metadata dict with all required keys (el_flagged, trail_width_mult, etc.)
9. ALWAYS call build_feature_vector and the ML models -- they provide critical gating

## Output format:
Respond with ONLY a Python code block containing both functions. No explanation outside the code.
Start with ```python and end with ```. Include necessary imports (numpy, pandas, logging, etc.) at the top of the block.
"""


def extract_code(llm_response):
    """Extract Python code from LLM response."""
    if not llm_response:
        return None

    # Try to find ```python ... ``` block
    pattern = r'```python\s*\n(.*?)```'
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ``` ... ``` without language tag
    pattern = r'```\s*\n(.*?)```'
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if 'def on_bar' in code:
            return code

    # If response itself looks like code
    if 'def on_bar' in llm_response and 'def _compute_current_atr' in llm_response:
        return llm_response.strip()

    return None


def validate_code(code):
    """Basic validation that the code defines required functions."""
    if 'def on_bar' not in code:
        return False, "missing on_bar()"
    if 'def _compute_current_atr' not in code:
        return False, "missing _compute_current_atr()"
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
            return line[1:].strip()[:100]
    return "no description"


def save_state(iteration, best_score, best_metrics, history):
    """Save evolution state for resume."""
    state = {
        "iteration": iteration,
        "best_score": best_score,
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


def main():
    log("=" * 60)
    log("Phase B1: Surfer-ML Entry Logic Evolution")
    log("=" * 60)

    # Check for resume state
    saved = load_state()
    start_iteration = 1

    if saved:
        start_iteration = saved["iteration"] + 1
        best_score = saved["best_score"]
        best_metrics = saved["best_metrics"]
        history = saved["history"]
        # Best code is in BEST_FILE
        if os.path.isfile(BEST_FILE):
            best_code = read_program_code(BEST_FILE)
        else:
            best_code = read_program_code(INITIAL_PROGRAM)
        log(f"RESUMING from iteration {start_iteration} "
            f"(best score={best_score:.0f}, "
            f"PnL=${best_metrics.get('total_pnl', 0):.0f})")
    else:
        # Evaluate initial program
        log("Evaluating initial program (current entry logic)...")
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

        # Save initial as best
        with open(BEST_FILE, "w", encoding="utf-8") as f:
            f.write(best_code)

        history = [{
            "score": best_score, "pnl": result["total_pnl"],
            "trades": result["n_trades"], "wr": result["win_rate"],
            "sharpe": result["sharpe"], "dd": result["max_drawdown_pct"],
            "description": "initial (current production logic)",
        }]
        save_state(0, best_score, best_metrics, history)

    # Evolution loop
    consecutive_errors = 0
    for i in range(start_iteration, MAX_ITERATIONS + 1):
        try:
            log(f"\n--- Iteration {i}/{MAX_ITERATIONS} ---")

            # Call LLM for new entry logic code
            prompt = build_prompt(best_code, best_score, best_metrics, history)
            log("  Calling LLM for entry logic code...")
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
            log(f"  Change: {description}")

            # Write candidate program to temp file
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
                error = result.get("error", "unknown") if result else "no result"
                log(f"  Evaluation failed ({eval_time:.0f}s): {error[:200]}")
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
                # Also save numbered copy for reference
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
