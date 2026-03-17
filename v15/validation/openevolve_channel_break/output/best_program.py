"""
Changes vs score=1030131:
- Multi-TF hard veto: if 2+ higher TFs strongly oppose direction, abort signal
- Tiered TP/stop R:R: high-quality = 5:1, medium = 3.5:1, base = 2.8:1 (fix low PF)
- Volume building: progressive 3-window accumulation before break
- Candle body direction: recent_body_dir from last 5 candle bodies for directional bias
- Momentum turn POSITIVE at boundary (turning up near upper = break starting, not penalty)
- Channel slope/direction bonus: trend-continuation breaks get wider TP
- BUY TP asymmetric: 1.20x (TSLA upward bias, let longs run)
- Break threshold raised 0.47 → 0.50 for quality
- quality_score < 0.4 penalty added
"""

import numpy as np
import pandas as pd


def predict_channel_break(channel_features: dict, multi_tf_features: dict,
                          recent_bars: pd.DataFrame) -> dict:
    result = {
        'break_imminent': False,
        'break_direction': None,
        'confidence': 0.0,
        'signal': None,
        'stop_pct': 0.010,
        'tp_pct': 0.020,
    }

    # -- Primary channel features --
    energy_ratio      = channel_features.get('energy_ratio', 0.0)
    squeeze           = channel_features.get('squeeze_score', 0.0)
    break_prob        = channel_features.get('break_prob', 0.0)
    break_prob_up     = channel_features.get('break_prob_up', 0.0)
    break_prob_down   = channel_features.get('break_prob_down', 0.0)
    health            = channel_features.get('channel_health', 1.0)
    ou_theta          = channel_features.get('ou_theta', 0.1)
    ou_half_life      = channel_features.get('ou_half_life', 20.0)
    entropy           = channel_features.get('entropy', 0.5)
    position_pct      = channel_features.get('position_pct', 0.5)
    mom_dir           = channel_features.get('momentum_direction', 0.0)
    mom_turning       = channel_features.get('momentum_is_turning', False)
    mom_turn_score    = channel_features.get('momentum_turn_score', 0.0)
    width_pct         = channel_features.get('width_pct', 1.0)
    bounce_count      = channel_features.get('bounce_count', 0)
    r_squared         = channel_features.get('r_squared', 0.5)
    false_break_rate  = channel_features.get('false_break_rate', 0.3)
    alternation_ratio = channel_features.get('alternation_ratio', 0.5)
    complete_cycles   = channel_features.get('complete_cycles', 0)
    quality_score     = channel_features.get('quality_score', 0.5)
    volume_score      = channel_features.get('volume_score', 0.5)
    bars_to_bounce    = channel_features.get('bars_to_next_bounce', 10)
    bars_since_touch  = channel_features.get('bars_since_last_touch', 5)
    upper_touches     = channel_features.get('upper_touches', 0)
    lower_touches     = channel_features.get('lower_touches', 0)
    channel_dir       = channel_features.get('channel_direction', 'sideways')
    slope_pct         = channel_features.get('slope_pct', 0.0)

    # -- Recent bars feature engineering --
    atr_pct        = 0.010
    vol_spike      = False
    vol_expansion  = False
    vol_trend      = 0.0
    vol_building   = False   # progressive 3-window volume accumulation
    price_accel    = 0.0
    energy_rising  = False
    closes_above   = 0.5
    tod_ok         = True
    candle_quality = 0.5     # avg body/range ratio (0=doji, 1=strong)
    recent_body_dir = 0.0    # mean sign of (close-open) last 5 bars

    if recent_bars is not None and len(recent_bars) >= 15:
        try:
            hi  = recent_bars['high'].values.astype(float)
            lo  = recent_bars['low'].values.astype(float)
            cl  = recent_bars['close'].values.astype(float)
            vol = recent_bars['volume'].values.astype(float)
            op  = recent_bars['open'].values.astype(float)

            last_close = cl[-1] if cl[-1] > 0 else 1.0

            # ATR (20-bar)
            n = min(20, len(cl))
            prev_cl = np.concatenate(([cl[-n-1] if len(cl) > n else cl[-n]], cl[-n:-1]))
            tr = np.maximum(hi[-n:] - lo[-n:],
                            np.maximum(np.abs(hi[-n:] - prev_cl),
                                       np.abs(lo[-n:] - prev_cl)))
            atr_pct = np.mean(tr) / last_close

            # Volume spike, expansion, trend
            if len(vol) >= 20:
                recent_vol = np.mean(vol[-3:])
                base_vol   = np.mean(vol[-20:-3]) + 1e-9
                vol_ratio  = recent_vol / base_vol
                vol_spike     = vol_ratio > 1.7
                vol_expansion = vol_ratio > 1.35

                vol10 = vol[-10:]
                if len(vol10) >= 5:
                    slope = np.polyfit(np.arange(len(vol10)), vol10, 1)[0]
                    vol_trend = slope / (np.mean(vol10) + 1e-9)

                # Progressive volume build: each 3-bar window larger than previous
                if len(vol) >= 9:
                    v1 = np.mean(vol[-9:-6])
                    v2 = np.mean(vol[-6:-3])
                    v3 = np.mean(vol[-3:])
                    vol_building = (v2 > v1 * 1.05) and (v3 > v2 * 1.05)

            # Volatility acceleration (squeeze releasing)
            if len(hi) >= 20:
                early_range  = np.mean(hi[-20:-10] - lo[-20:-10]) + 1e-9
                mid_range    = np.mean(hi[-10:-5]  - lo[-10:-5])
                recent_range = np.mean(hi[-5:]     - lo[-5:])
                price_accel  = recent_range / early_range - 1.0
                energy_rising = (recent_range / (mid_range + 1e-9)) > 1.25

            # Closes above midpoint (directional bias)
            if len(cl) >= 10:
                mid_approx = (hi[-10:] + lo[-10:]) / 2.0
                closes_above = np.mean(cl[-10:] > mid_approx)

            # Candle quality: body/range + direction (last 5 bars)
            if len(cl) >= 5:
                bodies = np.abs(cl[-5:] - op[-5:])
                ranges = hi[-5:] - lo[-5:] + 1e-9
                candle_quality  = np.mean(bodies / ranges)
                recent_body_dir = np.mean(np.sign(cl[-5:] - op[-5:]))

            # Time-of-day filter
            if hasattr(recent_bars.index, 'hour'):
                last_dt = recent_bars.index[-1]
                minute_of_day = last_dt.hour * 60 + last_dt.minute
                if minute_of_day < 570 or minute_of_day >= 945:
                    tod_ok = False

        except Exception:
            pass

    # -- Break score --
    break_score = 0.0

    # Core energy signal
    if energy_ratio > 1.2:
        energy_contrib = 0.35 * min(1.0, (energy_ratio - 1.0) / 0.8)
        if energy_rising:
            energy_contrib *= 1.30
        break_score += energy_contrib

    # Squeeze (compression → explosion)
    if squeeze > 0.55:
        break_score += 0.25 * squeeze

    # Combined channel decay signal
    health_decay   = (1.0 - health) if health < 0.5 else 0.0
    theta_decay    = (1.0 - min(1.0, ou_theta / 0.08)) if ou_theta < 0.08 else 0.0
    combined_decay = max(health_decay, theta_decay) + 0.5 * min(health_decay, theta_decay)
    if combined_decay > 0:
        break_score += 0.18 * combined_decay

    # Entropy (structure breakdown)
    if entropy > 0.60:
        break_score += 0.08 * entropy

    # Physics break probability
    if break_prob > 0.45:
        break_score += 0.10 * break_prob

    # Volume signals
    if vol_spike:
        break_score += 0.14
    elif vol_expansion:
        break_score += 0.07
    if vol_trend > 0.05:
        break_score += 0.06 * min(1.0, vol_trend / 0.15)
    if vol_building:
        break_score += 0.06  # progressive accumulation = institutional buildup

    # Volatility acceleration
    if price_accel > 0.25:
        break_score += 0.08 * min(1.0, price_accel)

    # Energy + squeeze synergy
    if energy_ratio > 1.3 and squeeze > 0.60:
        break_score += 0.10

    # Strong directional candles add conviction
    if candle_quality > 0.60:
        break_score += 0.04 * candle_quality

    # False break dampening (energy override for very high energy)
    if false_break_rate > 0.5:
        energy_override = max(0.0, energy_ratio - 1.4) * 2.0
        damp_factor = 1.0 - (false_break_rate - 0.5) * 0.6 * (1.0 - energy_override)
        break_score *= max(0.5, damp_factor)

    # Channel still has grip → suppress
    if ou_half_life < 5 and ou_theta > 0.15:
        break_score *= 0.70

    # Young channel → unreliable
    if complete_cycles < 2:
        break_score *= 0.85

    # Time-of-day trap avoidance
    if not tod_ok:
        break_score *= 0.75

    # Threshold raised to 0.50 for quality (was 0.47)
    if break_score < 0.50:
        return result

    result['break_imminent'] = True

    # -- Direction scoring --
    up_score   = 0.0
    down_score = 0.0

    # Momentum direction (primary signal)
    if mom_dir > 0:
        up_score   += 0.35 * abs(mom_dir)
    else:
        down_score += 0.35 * abs(mom_dir)

    # Position in channel (proximity to boundary)
    if position_pct > 0.65:
        up_score   += 0.22 * (position_pct - 0.5) * 2.0
    elif position_pct < 0.35:
        down_score += 0.22 * (0.5 - position_pct) * 2.0

    # Physics model break direction
    up_score   += 0.22 * break_prob_up
    down_score += 0.22 * break_prob_down

    # Candle closes above midpoint (directional pressure)
    if closes_above > 0.65:
        up_score   += 0.10 * (closes_above - 0.5) * 2.0
    elif closes_above < 0.35:
        down_score += 0.10 * (0.5 - closes_above) * 2.0

    # Recent candle body direction (last 5 bars)
    if recent_body_dir > 0.3:
        up_score   += 0.08 * recent_body_dir
    elif recent_body_dir < -0.3:
        down_score += 0.08 * abs(recent_body_dir)

    # Touch imbalance (more upper = resistance being tested repeatedly)
    total_touches = upper_touches + lower_touches + 1
    if upper_touches > lower_touches * 1.5:
        up_score   += 0.08 * min(1.0, (upper_touches - lower_touches) / total_touches)
    elif lower_touches > upper_touches * 1.5:
        down_score += 0.08 * min(1.0, (lower_touches - upper_touches) / total_touches)

    # Channel direction alignment (trend continuation breaks are stronger)
    if channel_dir == 'bull':
        up_score   *= 1.12
    elif channel_dir == 'bear':
        down_score *= 1.12

    # Slope alignment
    if slope_pct > 0.001:
        up_score   += 0.05
    elif slope_pct < -0.001:
        down_score += 0.05

    # Momentum turning at boundary: POSITIVE (break is starting)
    # vs turning mid-channel: uncertain (mild penalty later)
    if mom_turning and mom_turn_score > 0.5:
        at_upper = position_pct > 0.72
        at_lower = position_pct < 0.28
        if at_upper and mom_dir >= 0:
            up_score   += 0.12 * mom_turn_score  # turning up at upper = upward break
        elif at_lower and mom_dir <= 0:
            down_score += 0.12 * mom_turn_score  # turning down at lower = downward break

    # TSLA asymmetric upward bias
    up_score *= 1.08

    # -- Multi-TF confluence --
    tf_up = 0.0; tf_down = 0.0; tf_count = 0
    tf_weights = {'1h': 1.0, '4h': 1.5, 'daily': 2.0}
    tf_total_w = 0.0

    for tf_label in ('1h', '4h', 'daily'):
        tf = multi_tf_features.get(tf_label, {})
        if not tf:
            continue
        w = tf_weights.get(tf_label, 1.0)
        tf_total_w += w
        tf_count   += 1

        tf_dir     = tf.get('channel_direction', 'sideways')
        tf_pos     = tf.get('position_pct', 0.5)
        tf_mom     = tf.get('momentum_direction', 0.0)
        tf_bpu     = tf.get('break_prob_up', 0.0)
        tf_bpd     = tf.get('break_prob_down', 0.0)
        tf_energy  = tf.get('energy_ratio', 0.0)
        tf_squeeze = tf.get('squeeze_score', 0.0)

        local_up = 0.0; local_down = 0.0
        if tf_dir == 'bull':
            local_up   += 1.5
        elif tf_dir == 'bear':
            local_down += 1.5
        if tf_pos > 0.6:
            local_up   += 0.6
        elif tf_pos < 0.4:
            local_down += 0.6
        if tf_mom > 0:
            local_up   += 0.5 * tf_mom
        elif tf_mom < 0:
            local_down += 0.5 * abs(tf_mom)
        local_up   += tf_bpu * 0.8
        local_down += tf_bpd * 0.8
        if tf_energy > 1.2:
            if tf_bpu > tf_bpd:
                local_up   += 0.5
            else:
                local_down += 0.5
        if tf_squeeze > 0.6:
            if local_up > local_down:
                local_up   += 0.3
            else:
                local_down += 0.3

        tf_up   += w * local_up
        tf_down += w * local_down

    if tf_total_w > 0:
        norm = 0.18 / tf_total_w
        up_score   += norm * tf_up
        down_score += norm * tf_down

    # Direction decision
    if up_score > down_score and up_score > 0.32:
        direction = 'up'
        dir_conf  = up_score / max(up_score + down_score, 0.01)
    elif down_score > up_score and down_score > 0.32:
        direction = 'down'
        dir_conf  = down_score / max(up_score + down_score, 0.01)
    else:
        return result

    # -- Multi-TF HARD VETO: block if higher TFs strongly oppose direction --
    # Requires 2+ TFs available and >2.5x weight opposing vs supporting
    if tf_count >= 2 and tf_total_w > 0:
        tf_bull_norm = tf_up   / tf_total_w
        tf_bear_norm = tf_down / tf_total_w
        if direction == 'up'   and tf_bear_norm > tf_bull_norm * 2.5:
            return result  # Higher TFs strongly bearish, abort BUY
        if direction == 'down' and tf_bull_norm > tf_bear_norm * 2.5:
            return result  # Higher TFs strongly bullish, abort SELL

    result['break_direction'] = direction
    result['signal'] = 'BUY' if direction == 'up' else 'SELL'

    # -- Confidence --
    confidence = break_score * dir_conf

    # Squeeze + energy combo
    if squeeze > 0.55 and energy_ratio > 1.3:
        confidence *= 1.25
        if energy_rising:
            confidence *= 1.10

    # Volume confirmation
    if vol_spike:
        confidence *= 1.20
    elif vol_expansion:
        confidence *= 1.10
    if vol_trend > 0.08:
        confidence *= 1.06
    if vol_building:
        confidence *= 1.06

    # Volatility expansion
    if price_accel > 0.3:
        confidence *= 1.10

    # Strong directional candles = higher conviction
    if candle_quality > 0.65:
        confidence *= 1.05 + 0.04 * candle_quality

    # Momentum turning context
    if mom_turning:
        at_boundary_aligned = (
            (direction == 'up'   and position_pct > 0.72 and mom_dir >= 0) or
            (direction == 'down' and position_pct < 0.28 and mom_dir <= 0)
        )
        if at_boundary_aligned:
            confidence *= 1.12  # Turning at boundary in break direction = confirmation
        else:
            turn_pen = 0.65 + 0.25 * (1.0 - mom_turn_score)
            confidence *= turn_pen

    # Channel quality
    if alternation_ratio < 0.4:
        confidence *= 0.80
    if width_pct > 3.0:
        confidence *= 0.85
    if bounce_count >= 4 and r_squared > 0.65:
        confidence *= 1.12
    if quality_score > 0.7:
        confidence *= 1.08
    elif quality_score < 0.4:
        confidence *= 0.82

    # Channel age
    if complete_cycles < 2:
        confidence *= 0.85

    # Timing within oscillation
    if bars_to_bounce < 3:
        confidence *= 0.80

    # Time-of-day
    if not tod_ok:
        confidence *= 0.80

    result['confidence'] = min(0.95, max(0.05, confidence))

    # -- Tiered TP/stop sizing for improved profit factor --
    atr_safe = max(0.003, min(0.025, atr_pct))
    conf     = result['confidence']

    # Quality tier classification
    high_quality = (energy_ratio > 1.4 and squeeze > 0.65 and (vol_spike or vol_expansion))
    med_quality  = (energy_ratio > 1.3 and squeeze > 0.55)

    # Stop: tighter on higher-confidence setups (we're more certain of direction)
    if high_quality:
        stop_mult = 1.0 + 0.3 * (1.0 - conf)   # 1.0–1.3x ATR
    elif med_quality:
        stop_mult = 1.2 + 0.4 * (1.0 - conf)   # 1.2–1.6x ATR
    else:
        stop_mult = 1.5 + 0.5 * (1.0 - conf)   # 1.5–2.0x ATR

    base_stop = atr_safe * stop_mult
    cw_stop   = (width_pct / 100.0) * 0.40
    result['stop_pct'] = max(0.004, min(0.020, 0.55 * base_stop + 0.45 * cw_stop))

    # TP: wide multipliers to fix low profit factor
    if high_quality:
        tp_mult = 5.5 if energy_rising else 4.5   # highest quality: 4.5–5.5x ATR
    elif med_quality:
        tp_mult = 4.2 if energy_rising else 3.5   # medium quality: 3.5–4.2x ATR
    else:
        tp_mult = 2.8                              # base: 2.8x ATR

    # Additional TP boosters
    if squeeze > 0.70:
        tp_mult += 0.7   # squeeze breaks overshoot significantly
    if vol_spike:
        tp_mult += 0.4
    if vol_building:
        tp_mult += 0.2

    # Trend continuation break: let it run further
    if (direction == 'up'   and channel_dir == 'bull') or \
       (direction == 'down' and channel_dir == 'bear'):
        tp_mult *= 1.15

    # TSLA upward bias: BUY signals get 20% wider TP
    if direction == 'up':
        tp_mult *= 1.20

    base_tp = atr_safe * tp_mult
    cw_tp   = (width_pct / 100.0) * 0.95
    result['tp_pct'] = max(0.010, min(0.050, 0.55 * base_tp + 0.45 * cw_tp))

    return result