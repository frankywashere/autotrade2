"""Bounce timing signal v3: Enhanced aggressive signal weighting for maximum profitability."""

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

    if daily['pos_pct'] >= 0.31 or weekly['pos_pct'] >= 0.31:
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    if tsla_rsi_w < 36 and tsla_rsi_w <= tsla_rsi_sma:
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    conf = 0.96
    conf += (0.31 - daily['pos_pct']) * 4.08
    conf += (0.31 - weekly['pos_pct']) * 2.32
    conf += (monthly is not None and monthly['pos_pct'] < 0.31) * 0.97
    conf += daily['is_turning'] * 0.97 + weekly['is_turning'] * 1.36
    conf += (spy_rsi > 68) * 0.79 - (spy_rsi < 32) * 0.47
    conf += (tsla_rsi_w < 26) * 1.62 + (tsla_rsi_w < 29) * 1.19
    conf += (tsla_rsi_w > tsla_rsi_sma and tsla_rsi_w < 50) * 1.18
    conf += (dist_52w_sma < -0.16) * 1.05 - (dist_52w_sma >= -0.04) * 0.43
    conf = float(np.clip(conf, 0.0, 1.0))

    delay_hours = 0
    if not daily['is_turning'] and not weekly['is_turning']:
        delay_hours = 17
    elif spy_rsi < 32 and not weekly['is_turning']:
        delay_hours = 11
    if tsla_rsi_w <= tsla_rsi_sma:
        delay_hours = max(delay_hours, 7)

    return {'take_bounce': conf >= 0.15, 'delay_hours': delay_hours, 'confidence': conf}
