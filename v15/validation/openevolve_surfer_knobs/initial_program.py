"""
Surfer-ML knob configuration for OpenEvolve Phase A.

OpenEvolve mutates the values inside EVOLVE-BLOCK to find optimal
hyperparameters. The evaluator imports get_knobs() and builds a
backtester command from the returned dict.

Ranges (for LLM guidance):
  exit_grace_bars:       0-15    (1-min bars after entry before stops activate)
  stop_update_secs:      5-600   (how often to ratchet best_price, seconds)
  stop_check_secs:       5-60    (how often to check price vs stop, seconds)
  grace_ratchet_secs:    0-300   (ratchet during grace period, 0=none)
  profit_activated_stop: bool    (stop only fires after trade is in profit)
  max_underwater_mins:   0-600   (force-close if never profitable, 0=disabled)
  max_hold_bars:         20-9999 (5-min bars before timeout, 60=5hrs)
  breakout_stop_mult:    0.05-2.0 (multiplier for breakout stop distance)
  eval_interval:         1-6     (evaluate signal every N 5-min bars)
"""


def get_knobs() -> dict:
    """Return surfer-ml hyperparameter configuration."""

    # EVOLVE-BLOCK-START

    knobs = {
        # Stop activation timing
        'exit_grace_bars': 5,         # 1-min bars of grace after entry
        'stop_update_secs': 60,       # Ratchet best_price every N seconds
        'stop_check_secs': 5,         # Check price vs stop every N seconds
        'grace_ratchet_secs': 300,    # Ratchet during grace (0=none)

        # Profit-activated stop
        'profit_activated_stop': True, # Only fire stop once in profit
        'max_underwater_mins': 0,      # Force-close if never profitable (0=disabled)

        # Hold duration
        'max_hold_bars': 60,          # 5-min bars max hold (60=5hrs)

        # Signal tuning
        'breakout_stop_mult': 1.00,   # Breakout stop distance multiplier
        'eval_interval': 3,           # Signal evaluation interval (5-min bars)
    }

    # EVOLVE-BLOCK-END

    return knobs
