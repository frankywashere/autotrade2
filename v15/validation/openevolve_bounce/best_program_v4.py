"""
Directional bounce/short signal v4: detects sell-off regime and trades
both sides – shorts during active sell-off, longs when bounce arrives.

Input:  states dict with keys 'daily', 'weekly', 'monthly' – each a dict with:
          pos_pct (0-1), is_turning (bool), at_bottom (bool),
          near_bottom (bool), energy_ratio (float)
        spy_rsi      (float) – 14-period RSI of SPY daily
        tsla_rsi_w   (float) – 14-period RSI of TSLA weekly
        tsla_rsi_sma (float) – 14-period SMA of the weekly RSI
        dist_52w_sma (float) – (close - 52wk_sma) / 52wk_sma

Output: dict with:
          'direction'   (str)   – 'long', 'short', or 'none'
          'confidence'  (float) – 0-1 signal strength
          'delay_hours' (int)   – suggested hours to wait before entry
"""

import numpy as np


def evaluate_bounce_signal(states: dict, spy_rsi: float,
                           tsla_rsi_w: float = 50.0,
                           tsla_rsi_sma: float = 50.0,
                           dist_52w_sma: float = 0.0) -> dict:
    """Evaluate whether to go long (bounce) or short (sell-off continues)."""

    daily = states.get('daily')
    weekly = states.get('weekly')
    monthly = states.get('monthly')

    if not daily or not weekly:
        return {'direction': 'none', 'confidence': 0.0, 'delay_hours': 0}

    # EVOLVE-BLOCK-START

    d_pos, w_pos = daily['pos_pct'], weekly['pos_pct']
    m_pos = monthly['pos_pct'] if monthly else 0.5
    d_turn, w_turn = daily['is_turning'], weekly['is_turning']

    d_near = daily.get('near_bottom', False)
    w_near = weekly.get('near_bottom', False)
    d_bottom = daily.get('at_bottom', False)
    w_bottom = weekly.get('at_bottom', False)

    ts_sell = tsla_rsi_w < 50
    ts_deep = tsla_rsi_w < 30
    ts_extreme = tsla_rsi_w < 25
    rsi_fall = tsla_rsi_w < tsla_rsi_sma - 5
    rsi_rise = tsla_rsi_w > tsla_rsi_sma
    rsi_recovery = tsla_rsi_w > tsla_rsi_sma + 3

    spy_weak = spy_rsi < 35
    spy_strong = spy_rsi > 65

    # ============================================================
    # LONG SIGNAL: Oversold bounce with enhanced confirmation
    # ============================================================
    if d_pos < 0.38 and w_pos < 0.38:
        conf = 0.50
        conf += (0.38 - d_pos) * 1.2
        conf += (0.38 - w_pos) * 0.65

        if rsi_recovery and ts_sell:
            conf += 0.24
        elif rsi_rise and ts_sell:
            conf += 0.18
        elif ts_extreme:
            conf += 0.16

        if m_pos < 0.35:
            conf += 0.14

        if d_turn:
            conf += 0.12
        if w_turn:
            conf += 0.15

        if d_bottom or d_near:
            conf += 0.08
        if w_bottom or w_near:
            conf += 0.10

        if spy_strong:
            conf += 0.10
        elif spy_weak:
            conf -= 0.09

        if dist_52w_sma < -0.18:
            conf += 0.11
        elif dist_52w_sma < -0.10:
            conf += 0.05

        if rsi_fall and tsla_rsi_w > 35 and not d_turn and not w_turn:
            conf -= 0.27

        conf = float(np.clip(conf, 0.0, 1.0))

        delay_hours = 16 if (not d_turn and not w_turn and ts_sell and not rsi_recovery) else 0

        if conf >= 0.53:
            return {'direction': 'long', 'confidence': conf,
                    'delay_hours': delay_hours}

    # ============================================================
    # SHORT SIGNAL: High-selectivity with strict momentum filter
    # ============================================================
    if (ts_sell and rsi_fall and dist_52w_sma < -0.06):
        if d_pos > 0.60 and w_pos < 0.30:
            if d_turn or w_turn:
                return {'direction': 'none', 'confidence': 0.0, 'delay_hours': 0}

            if rsi_recovery:
                return {'direction': 'none', 'confidence': 0.0, 'delay_hours': 0}

            conf = 0.70
            conf += (d_pos - 0.60) * 1.0
            conf += (0.30 - w_pos) * 0.5

            if ts_extreme:
                conf += 0.12

            if dist_52w_sma < -0.22:
                conf += 0.14

            if spy_weak:
                conf += 0.09

            if tsla_rsi_w < 30 and rsi_fall:
                conf += 0.08

            conf = float(np.clip(conf, 0.0, 1.0))

            if conf >= 0.75:
                return {'direction': 'short', 'confidence': conf,
                        'delay_hours': 0}

    # EVOLVE-BLOCK-END

    return {'direction': 'none', 'confidence': 0.0, 'delay_hours': 0}