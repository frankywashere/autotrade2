"""Trade gate function for forward sim V2.

OpenEvolve will evolve the logic inside EVOLVE-BLOCK to learn which trades
to KEEP and which to SKIP, based on per-trade features.

This function is called for each potential trade entry. It returns True to
TAKE the trade, False to SKIP it.

IMPORTANT: Only CS-5TF, CS-DW, and ML signals are gated. Intraday signals
bypass this gate entirely (handled in the evaluator).

Available features (all floats):
  scanner:       0=CS-5TF, 1=CS-DW, 2=ML
  signal_type:   0=bounce, 1=break
  direction:     0=long, 1=short
  primary_tf:    0=5min, 1=1h, 2=4h, 3=daily, 4=weekly
  confidence:    0.0-1.0 (signal confidence)
  entry_hour:    9.5-16.0 (ET, decimal hours)
  prev_vix:      previous day's VIX close
  day_of_week:   0=Mon, 1=Tue, ..., 4=Fri
  is_event_day:  1.0 if calendar event day, else 0.0
  is_near_event: 1.0 if within 1 trading day of event, else 0.0
"""

import numpy as np


def should_take_trade(
    scanner: float,
    signal_type: float,
    direction: float,
    primary_tf: float,
    confidence: float,
    entry_hour: float,
    prev_vix: float,
    day_of_week: float,
    is_event_day: float,
    is_near_event: float,
) -> bool:
    """Return True to TAKE the trade, False to SKIP it.

    Goal: maximize P&L while keeping win rate above 60%.
    The baseline takes ALL trades (always returns True).
    Hard stops cost -$893K on 211 trades. If we can skip even half
    of those while keeping most winners, we add $400K+ to P&L.
    """

    # EVOLVE-BLOCK-START

    # ── Gate 1: Skip low-confidence signals ──────────────────────────────
    if confidence < 0.45:
        return False

    # ── Gate 2: Skip ML break signals on 5min TF ────────────────────────
    # These are 58% of all hard stops (122/211 trades, -$527K)
    if scanner == 2 and signal_type == 1 and primary_tf == 0:
        # ML + break + 5min: only take if high confidence
        if confidence < 0.70:
            return False

    # ── Gate 3: Skip overnight entries (after 15:30 ET) ──────────────────
    # Many hard stops come from 15:55 entries that gap against next day
    if entry_hour >= 15.5:
        return False

    # ── Gate 4: High VIX caution ─────────────────────────────────────────
    # When VIX > 30, restrict to first hour only
    if prev_vix > 30 and entry_hour > 10.5:
        return False

    # ── Gate 5: Low-confidence bounce signals ────────────────────────────
    if signal_type == 0 and confidence < 0.50:
        return False

    # EVOLVE-BLOCK-END

    return True
