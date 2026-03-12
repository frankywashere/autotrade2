"""
Intraday knob configuration for OpenEvolve Phase A.

OpenEvolve mutates the values inside EVOLVE-BLOCK to find optimal
hyperparameters. The evaluator imports get_knobs() and builds a
backtester config from the returned dict.

Ranges (for LLM guidance):
  # Signal thresholds
  vwap_thresh:           -0.50 to 0.0   (max VWAP dist for VWAP signal, more negative = stricter)
  d_min:                 0.05-0.60      (min daily channel position)
  h1_min:                0.05-0.40      (min 1h channel position)
  f5_thresh:             0.10-0.50      (max 5-min channel position, lower = more oversold)
  div_thresh:            0.05-0.50      (min divergence for div signal)
  div_f5_thresh:         0.10-0.50      (max 5-min CP for div signal)
  min_vol_ratio:         0.0-2.0        (min volume ratio, 0=disabled)

  # Stop / TP / Trail
  stop_pct:              0.003-0.025    (initial stop distance)
  tp_pct:                0.005-0.050    (take profit distance)
  trail_base:            0.002-0.020    (base trailing stop distance)
  trail_power:           1-12           (exponent on (1-conf), higher = tighter at high conf)
  trail_floor:           0.0-0.008      (minimum trail distance)

  # Execution
  exit_grace_bars:       0-15           (1-min bars after entry before stops activate)
  stop_update_secs:      5-600          (ratchet best_price interval)
  stop_check_secs:       5-60           (check price vs stop interval)
  grace_ratchet_secs:    0-300          (ratchet during grace, 0=none)
  max_hold_bars:         10-156         (5-min bars max hold, 78=full day)
  eval_interval:         1-4            (evaluate signal every N 5-min bars)
  max_trades_per_day:    1-50           (max entries per day, 0=unlimited)

  # Profit-activated stop
  profit_activated_stop: bool           (stop only fires once in profit)
  max_underwater_mins:   0-600          (force-close if never profitable, 0=disabled)
"""


def get_knobs() -> dict:
    """Return intraday hyperparameter configuration."""

    # EVOLVE-BLOCK-START

    knobs = {
        # Signal thresholds
        'vwap_thresh': -0.10,
        'd_min': 0.20,
        'h1_min': 0.15,
        'f5_thresh': 0.35,
        'div_thresh': 0.20,
        'div_f5_thresh': 0.35,
        'min_vol_ratio': 0.8,

        # Stop / TP / Trail
        'stop_pct': 0.008,
        'tp_pct': 0.020,
        'trail_base': 0.006,
        'trail_power': 6,
        'trail_floor': 0.0,

        # Execution
        'exit_grace_bars': 5,
        'stop_update_secs': 60,
        'stop_check_secs': 5,
        'grace_ratchet_secs': 60,
        'max_hold_bars': 78,
        'eval_interval': 1,
        'max_trades_per_day': 30,

        # Profit-activated stop
        'profit_activated_stop': False,
        'max_underwater_mins': 0,
    }

    # EVOLVE-BLOCK-END

    return knobs
