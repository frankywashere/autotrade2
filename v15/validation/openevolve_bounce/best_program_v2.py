"""
Bounce timing signal: decides whether an oversold TSLA bounce will
materialize within N days.  OpenEvolve mutates the code between
EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers.

Input:  dict with keys 'daily', 'weekly', 'monthly' – each a dict with:
          pos_pct (0-1), is_turning (bool), at_bottom (bool),
          near_bottom (bool), energy_ratio (float)
        Plus 'spy_rsi' (float, 14-period RSI of SPY daily).

Output: dict with:
          'take_bounce' (bool)  – True if we expect a tradeable bounce
          'delay_hours' (int)   – suggested hours to wait before entry (0 = immediate)
          'confidence'  (float) – 0-1 signal strength
"""

import numpy as np


def evaluate_bounce_signal(states: dict, spy_rsi: float) -> dict:
    """Evaluate whether an oversold condition will produce a tradeable bounce."""

    daily = states.get('daily')
    weekly = states.get('weekly')
    monthly = states.get('monthly')

    if not (daily and weekly):
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    # EVOLVE-BLOCK-START

    t = 0.40
    if daily['pos_pct'] >= t or weekly['pos_pct'] >= t:
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    # Depth signals with refined weighting
    dd = t - daily['pos_pct']
    dw = t - weekly['pos_pct']
    conf = 0.51 + dd * 1.38 + dw * 0.61

    # Multi-timeframe confirmations
    if monthly and monthly['pos_pct'] < t:
        conf += 0.18
    conf += 0.13 * daily['is_turning'] + 0.23 * weekly['is_turning']

    # Bottom detection
    if daily.get('at_bottom'):
        conf += 0.10
    elif daily.get('near_bottom'):
        conf += 0.07

    # Energy ratio calibration
    e = daily.get('energy_ratio', 1.0)
    if e > 1.28:
        conf += 0.16
    elif e < 0.52:
        conf -= 0.06

    # Market regime from SPY
    if spy_rsi > 70:
        conf += 0.18
    elif spy_rsi < 27:
        conf -= 0.21

    conf = float(np.clip(conf, 0.0, 1.0))

    # Entry timing with turning point priority
    if daily['is_turning'] or weekly['is_turning']:
        delay_hours = 2 if spy_rsi < 27 and not weekly['is_turning'] else 3
    else:
        delay_hours = 17

    # EVOLVE-BLOCK-END

    return {
        'take_bounce': conf >= 0.47,
        'delay_hours': delay_hours,
        'confidence': conf,
    }
