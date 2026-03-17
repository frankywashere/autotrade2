"""
Phase B3 initial program: Channel break predictor.

Predicts whether a channel will break in the next ~20 bars, which direction,
and generates entry signals timed BEFORE the break happens.

This is the starting point — the LLM evolves from here.

Args:
    channel_features: dict of 5-min channel features from TFChannelState:
        - break_prob, break_prob_up, break_prob_down (0-1)
        - energy_ratio (total/binding, >1.2 = imminent break)
        - squeeze_score (0-1, compression -> explosive moves)
        - channel_health (0=dying, 1=strong)
        - false_break_rate (0-1, resilience)
        - ou_theta (mean-reversion speed, <0.05 = channel failing)
        - ou_half_life (bars to half-revert)
        - ou_reversion_score (0-1)
        - entropy (Shannon entropy, 0=predictable, 1=random)
        - alternation_ratio (bounce cleanliness, 0-1)
        - r_squared (regression fit quality, 0-1)
        - width_pct (channel width as % of price)
        - bounce_count (alternating boundary touches)
        - bars_since_last_touch (bars since last boundary touch)
        - momentum_direction (+1=toward upper, -1=toward lower)
        - momentum_is_turning (bool)
        - momentum_turn_score (0-1)
        - kinetic_energy (absolute momentum toward boundary)
        - potential_energy (0 at center, 1 at boundaries)
        - total_energy (PE + KE)
        - binding_energy (channel strength)
        - position_pct (0=lower, 0.5=center, 1=upper)
        - slope_pct (channel slope as % per bar)
        - oscillation_period (bars per full cycle)
        - bars_to_next_bounce (predicted bars until boundary touch)
        - volume_score (0-1, volume confirmation)
        - center_distance (signed, -1 to +1)
        - channel_direction ('bull', 'bear', 'sideways')
        - complete_cycles (full round-trips)
        - quality_score (composite channel quality)

    multi_tf_features: dict of TF -> dict of channel features for higher TFs
        e.g. {'1h': {'direction': 'bull', 'position_pct': 0.7, 'break_prob': 0.3, ...},
               '4h': {...}, 'daily': {...}}
        Each dict has the same keys as channel_features (where available).

    recent_bars: pd.DataFrame with last 100 5-min OHLCV bars
        columns: open, high, low, close, volume

Returns:
    dict with:
        - 'break_imminent': bool (predict break in next 20 bars)
        - 'break_direction': 'up' or 'down' or None
        - 'confidence': float 0-1
        - 'signal': 'BUY', 'SELL', or None
        - 'stop_pct': float (e.g., 0.008)
        - 'tp_pct': float (e.g., 0.015)
"""

import numpy as np
import pandas as pd


def predict_channel_break(channel_features: dict, multi_tf_features: dict,
                          recent_bars: pd.DataFrame) -> dict:
    """Predict imminent channel break and generate entry signal.

    Initial simple version:
    - energy_ratio > 1.3 (particle energy exceeds binding → about to escape)
    - squeeze_score > 0.7 (channel compressing → explosive move coming)
    - momentum direction for break direction
    - Multi-TF alignment as confirmation
    """
    result = {
        'break_imminent': False,
        'break_direction': None,
        'confidence': 0.0,
        'signal': None,
        'stop_pct': 0.010,
        'tp_pct': 0.015,
    }

    # -- Extract primary channel features --
    energy_ratio = channel_features.get('energy_ratio', 0.0)
    squeeze = channel_features.get('squeeze_score', 0.0)
    break_prob = channel_features.get('break_prob', 0.0)
    break_prob_up = channel_features.get('break_prob_up', 0.0)
    break_prob_down = channel_features.get('break_prob_down', 0.0)
    health = channel_features.get('channel_health', 1.0)
    ou_theta = channel_features.get('ou_theta', 0.1)
    entropy = channel_features.get('entropy', 0.5)
    position_pct = channel_features.get('position_pct', 0.5)
    mom_dir = channel_features.get('momentum_direction', 0.0)
    mom_turning = channel_features.get('momentum_is_turning', False)
    kinetic = channel_features.get('kinetic_energy', 0.0)
    width_pct = channel_features.get('width_pct', 1.0)
    bounce_count = channel_features.get('bounce_count', 0)
    r_squared = channel_features.get('r_squared', 0.5)

    # -- Break detection: energy + squeeze + low health --
    # A channel is about to break when:
    # 1. Energy exceeds binding (energy_ratio > 1.3)
    # 2. Channel is compressing (squeeze_score > 0.7)
    # 3. Channel health is declining (< 0.4)
    # 4. Mean-reversion is weakening (ou_theta < 0.05)

    break_score = 0.0

    # Energy signal (strongest indicator)
    if energy_ratio > 1.3:
        break_score += 0.35 * min(1.0, (energy_ratio - 1.0) / 1.0)

    # Squeeze signal (compression precedes explosion)
    if squeeze > 0.7:
        break_score += 0.25 * min(1.0, squeeze)

    # Health declining
    if health < 0.4:
        break_score += 0.20 * (1.0 - health)

    # OU mean-reversion weakening (channel losing grip)
    if ou_theta < 0.05:
        break_score += 0.10 * (1.0 - min(1.0, ou_theta / 0.05))

    # Entropy (high = chaotic, channel structure breaking down)
    if entropy > 0.7:
        break_score += 0.10 * min(1.0, entropy)

    # Need meaningful score to declare break imminent
    if break_score < 0.35:
        return result

    result['break_imminent'] = True

    # -- Direction: which way will it break? --
    # Use momentum direction + position in channel + higher TF alignment
    up_score = 0.0
    down_score = 0.0

    # Momentum direction (strongest short-term indicator)
    if mom_dir > 0:
        up_score += 0.4 * abs(mom_dir)
    else:
        down_score += 0.4 * abs(mom_dir)

    # Position in channel (near top = likely up, near bottom = likely down)
    if position_pct > 0.7:
        up_score += 0.2 * (position_pct - 0.5) * 2
    elif position_pct < 0.3:
        down_score += 0.2 * (0.5 - position_pct) * 2

    # Break probability from physics model
    if break_prob_up > break_prob_down:
        up_score += 0.2 * break_prob_up
    else:
        down_score += 0.2 * break_prob_down

    # Multi-TF alignment
    for tf_label in ('1h', '4h', 'daily'):
        tf_data = multi_tf_features.get(tf_label, {})
        if not tf_data:
            continue
        tf_dir = tf_data.get('channel_direction', 'sideways')
        tf_pos = tf_data.get('position_pct', 0.5)
        tf_mom = tf_data.get('momentum_direction', 0.0)

        weight = 0.067  # ~0.2 / 3 TFs
        if tf_dir == 'bull' or tf_pos > 0.6:
            up_score += weight
        elif tf_dir == 'bear' or tf_pos < 0.4:
            down_score += weight
        if tf_mom > 0:
            up_score += weight * 0.5
        elif tf_mom < 0:
            down_score += weight * 0.5

    # Determine direction
    if up_score > down_score and up_score > 0.2:
        result['break_direction'] = 'up'
        result['signal'] = 'BUY'
        direction_confidence = up_score / max(up_score + down_score, 0.01)
    elif down_score > up_score and down_score > 0.2:
        result['break_direction'] = 'down'
        result['signal'] = 'SELL'
        direction_confidence = down_score / max(up_score + down_score, 0.01)
    else:
        # Direction unclear — no signal
        result['break_imminent'] = True  # Break detected but direction unclear
        return result

    # -- Confidence --
    confidence = break_score * direction_confidence

    # Bonus for squeeze + energy combo (very reliable pre-break signal)
    if squeeze > 0.7 and energy_ratio > 1.3:
        confidence *= 1.2

    # Penalty for turning momentum (might reverse before breaking)
    if mom_turning:
        confidence *= 0.8

    # Penalty for very wide channels (less likely to break cleanly)
    if width_pct > 3.0:
        confidence *= 0.85

    # Bonus for well-structured channels (more meaningful when they break)
    if bounce_count >= 4 and r_squared > 0.6:
        confidence *= 1.1

    result['confidence'] = min(0.95, max(0.05, confidence))

    # -- Stop and TP sizing --
    # Stop: based on channel width — tighter for squeezes
    base_stop = width_pct / 100.0 * 0.5  # Half channel width
    if squeeze > 0.7:
        base_stop *= 0.7  # Tighter stop for squeezes (less room needed)
    result['stop_pct'] = max(0.004, min(0.020, base_stop))

    # TP: wider for high-energy breaks (they tend to run)
    base_tp = width_pct / 100.0  # Full channel width
    if energy_ratio > 1.5:
        base_tp *= 1.5  # Higher energy → bigger moves
    result['tp_pct'] = max(0.008, min(0.035, base_tp))

    return result
