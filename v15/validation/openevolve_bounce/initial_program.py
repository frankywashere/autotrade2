"""
Bounce timing signal: decides whether an oversold TSLA bounce will
materialize within N days.  OpenEvolve mutates the code between
EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers.

Input:  dict with keys 'daily', 'weekly', 'monthly' — each a dict with:
          pos_pct (0-1), is_turning (bool), at_bottom (bool),
          near_bottom (bool), energy_ratio (float)
        Plus 'spy_rsi' (float, 14-period RSI of SPY daily).

Output: dict with:
          'take_bounce' (bool)  — True if we expect a tradeable bounce
          'delay_hours' (int)   — suggested hours to wait before entry (0 = immediate)
          'confidence'  (float) — 0-1 signal strength
"""

import numpy as np


def evaluate_bounce_signal(states: dict, spy_rsi: float) -> dict:
    """Evaluate whether an oversold condition will produce a tradeable bounce."""

    daily = states.get('daily')
    weekly = states.get('weekly')
    monthly = states.get('monthly')

    if not daily or not weekly:
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    # EVOLVE-BLOCK-START

    # --- Position checks ---
    daily_oversold = daily['pos_pct'] < 0.35
    weekly_oversold = weekly['pos_pct'] < 0.35
    monthly_oversold = monthly['pos_pct'] < 0.35 if monthly else False

    # --- Momentum checks ---
    daily_turning = daily['is_turning']
    weekly_turning = weekly['is_turning']

    # --- SPY regime ---
    spy_weak = spy_rsi < 35       # broad market washout
    spy_neutral = 35 <= spy_rsi <= 65
    spy_strong = spy_rsi > 65     # strong broad market

    # --- Core logic ---
    # Require at least daily + weekly oversold
    if not (daily_oversold and weekly_oversold):
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    # Base confidence from position depth
    confidence = 0.5
    confidence += (0.35 - daily['pos_pct']) * 1.0   # deeper = more confident
    confidence += (0.35 - weekly['pos_pct']) * 0.5

    # Monthly oversold boosts confidence
    if monthly_oversold:
        confidence += 0.15

    # Momentum turning is a key confirmation
    if daily_turning:
        confidence += 0.10
    if weekly_turning:
        confidence += 0.15

    # SPY regime adjustment
    if spy_strong:
        confidence += 0.10   # rising tide lifts all boats
    elif spy_weak:
        confidence -= 0.15   # broad washout = slower bounce

    confidence = float(np.clip(confidence, 0.0, 1.0))

    # Delay logic: if no momentum turn yet, delay entry
    delay_hours = 0
    if not daily_turning and not weekly_turning:
        delay_hours = 24     # wait 1 trading day for stabilization
    elif spy_weak and not weekly_turning:
        delay_hours = 12     # half-day delay in weak markets

    # Decision threshold
    take_bounce = confidence >= 0.50

    # EVOLVE-BLOCK-END

    return {
        'take_bounce': take_bounce,
        'delay_hours': delay_hours,
        'confidence': confidence,
    }
