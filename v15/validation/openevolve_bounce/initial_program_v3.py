"""
Bounce timing signal v3: adds TSLA's own weekly RSI + distance from
52-week SMA as sell-off regime indicators.

Input:  states dict with keys 'daily', 'weekly', 'monthly' — each a dict with:
          pos_pct (0-1), is_turning (bool), at_bottom (bool),
          near_bottom (bool), energy_ratio (float)
        spy_rsi      (float) — 14-period RSI of SPY daily
        tsla_rsi_w   (float) — 14-period RSI of TSLA weekly
        tsla_rsi_sma (float) — 14-period SMA of the weekly RSI
        dist_52w_sma (float) — (close - 52wk_sma) / 52wk_sma, e.g. -0.15 = 15% below

Output: dict with:
          'take_bounce' (bool)  — True if we expect a tradeable bounce
          'delay_hours' (int)   — suggested hours to wait before entry (0 = immediate)
          'confidence'  (float) — 0-1 signal strength
"""

import numpy as np


def evaluate_bounce_signal(states: dict, spy_rsi: float,
                           tsla_rsi_w: float = 50.0,
                           tsla_rsi_sma: float = 50.0,
                           dist_52w_sma: float = 0.0) -> dict:
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
    spy_weak = spy_rsi < 35
    spy_strong = spy_rsi > 65

    # --- TSLA weekly RSI regime (NEW) ---
    # RSI < 50 = TSLA in sell-off mode; < 30 = deeply oversold
    tsla_selloff = tsla_rsi_w < 50
    tsla_deep_oversold = tsla_rsi_w < 30
    # RSI crossing above its SMA = momentum turning positive
    rsi_above_sma = tsla_rsi_w > tsla_rsi_sma
    # RSI rising (above SMA) after being below = potential bottom
    rsi_recovering = tsla_rsi_w > tsla_rsi_sma and tsla_rsi_w < 50

    # --- Distance from 52-week SMA (NEW) ---
    # Negative = below 52wk avg; more negative = deeper sell-off
    far_below_avg = dist_52w_sma < -0.15    # >15% below 52wk SMA
    moderately_below = dist_52w_sma < -0.05  # >5% below

    # --- Core logic ---
    # Require at least daily + weekly oversold
    if not (daily_oversold and weekly_oversold):
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    # If TSLA weekly RSI is still falling (below SMA and > 40), sell-off
    # isn't done yet — don't enter
    if tsla_selloff and not rsi_above_sma and tsla_rsi_w > 35:
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    # Base confidence from position depth
    confidence = 0.45
    confidence += (0.35 - daily['pos_pct']) * 1.0
    confidence += (0.35 - weekly['pos_pct']) * 0.5

    # Monthly oversold boosts confidence
    if monthly_oversold:
        confidence += 0.15

    # Momentum turning is key confirmation
    if daily_turning:
        confidence += 0.10
    if weekly_turning:
        confidence += 0.15

    # SPY regime
    if spy_strong:
        confidence += 0.10
    elif spy_weak:
        confidence -= 0.15

    # TSLA RSI regime adjustments (NEW)
    if tsla_deep_oversold:
        confidence += 0.15       # deeply oversold = high bounce potential
    if rsi_recovering:
        confidence += 0.20       # RSI turning up from below 50 = strong signal
    if rsi_above_sma and not tsla_selloff:
        confidence += 0.10       # RSI back above SMA and >50 = sell-off over

    # Distance from 52-week SMA adjustments (NEW)
    if far_below_avg:
        confidence += 0.10       # mean reversion pressure
    elif not moderately_below:
        confidence -= 0.10       # near or above avg = less bounce potential

    confidence = float(np.clip(confidence, 0.0, 1.0))

    # Delay logic
    delay_hours = 0
    if not daily_turning and not weekly_turning:
        delay_hours = 24
    elif spy_weak and not weekly_turning:
        delay_hours = 12
    # Extra delay if TSLA RSI still falling (NEW)
    if tsla_selloff and not rsi_above_sma:
        delay_hours = max(delay_hours, 24)

    # Decision threshold
    take_bounce = confidence >= 0.50

    # EVOLVE-BLOCK-END

    return {
        'take_bounce': take_bounce,
        'delay_hours': delay_hours,
        'confidence': confidence,
    }
