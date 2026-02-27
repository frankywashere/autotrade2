"""Bounce timing signal v3: Simplified confidence with refined signal weighting."""

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

    if daily['pos_pct'] >= 0.35 or weekly['pos_pct'] >= 0.35:
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    if tsla_rsi_w < 42 and tsla_rsi_w <= tsla_rsi_sma:
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    conf = 0.48
    conf += (0.35 - daily['pos_pct']) * 1.14
    conf += (0.35 - weekly['pos_pct']) * 0.53
    conf += (monthly is not None and monthly['pos_pct'] < 0.35) * 0.17
    conf += daily['is_turning'] * 0.14 + weekly['is_turning'] * 0.23
    conf += (spy_rsi > 65) * 0.10 - (spy_rsi < 35) * 0.10
    conf += (tsla_rsi_w < 32) * 0.22
    conf += (tsla_rsi_w > tsla_rsi_sma and tsla_rsi_w < 50) * 0.19
    conf += (dist_52w_sma < -0.15) * 0.15 - (dist_52w_sma >= -0.05) * 0.05

    conf = float(np.clip(conf, 0.0, 1.0))

    delay_hours = 0
    if not daily['is_turning'] and not weekly['is_turning']:
        delay_hours = 24
    elif spy_rsi < 35 and not weekly['is_turning']:
        delay_hours = 12
    if tsla_rsi_w <= tsla_rsi_sma:
        delay_hours = max(delay_hours, 12)

    return {
        'take_bounce': conf >= 0.49,
        'delay_hours': delay_hours,
        'confidence': conf,
    }
