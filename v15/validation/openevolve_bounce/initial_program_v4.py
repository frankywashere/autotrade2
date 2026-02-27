"""
Directional bounce/short signal v4: detects sell-off regime and trades
both sides — shorts during active sell-off, longs when bounce arrives.

Input:  states dict with keys 'daily', 'weekly', 'monthly' — each a dict with:
          pos_pct (0-1), is_turning (bool), at_bottom (bool),
          near_bottom (bool), energy_ratio (float)
        spy_rsi      (float) — 14-period RSI of SPY daily
        tsla_rsi_w   (float) — 14-period RSI of TSLA weekly
        tsla_rsi_sma (float) — 14-period SMA of the weekly RSI
        dist_52w_sma (float) — (close - 52wk_sma) / 52wk_sma

Output: dict with:
          'direction'   (str)   — 'long', 'short', or 'none'
          'confidence'  (float) — 0-1 signal strength
          'delay_hours' (int)   — suggested hours to wait before entry
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

    # --- Channel positions ---
    d_pos = daily['pos_pct']
    w_pos = weekly['pos_pct']
    m_pos = monthly['pos_pct'] if monthly else 0.5

    daily_turning = daily['is_turning']
    weekly_turning = weekly['is_turning']

    # --- TSLA weekly RSI regime ---
    tsla_selloff = tsla_rsi_w < 50           # below 50 = bearish
    tsla_deep_oversold = tsla_rsi_w < 30     # deeply oversold
    rsi_falling = tsla_rsi_w < tsla_rsi_sma  # RSI below its SMA = falling
    rsi_rising = tsla_rsi_w > tsla_rsi_sma   # RSI above SMA = recovering

    # --- SPY context ---
    spy_weak = spy_rsi < 35
    spy_strong = spy_rsi > 65

    # ============================================================
    # SHORT SIGNAL: sell-off in progress, short relief rallies
    # ============================================================
    # Strict conditions: weekly RSI actively falling + well below SMA +
    # price below 52wk avg + daily has rallied into mid-channel (short the rally)
    if (tsla_selloff and rsi_falling
            and tsla_rsi_w < (tsla_rsi_sma - 5)   # RSI well below its SMA
            and dist_52w_sma < -0.05):              # below 52wk average
        # Short when daily has rallied but weekly still bearish
        if d_pos > 0.40 and w_pos < 0.30:
            confidence = 0.45
            # Stronger if daily is high in channel (rally overextended)
            confidence += (d_pos - 0.40) * 1.5
            # Weekly deeply oversold = trend intact
            confidence += (0.30 - w_pos) * 0.5
            # SPY weak confirms broader selling
            if spy_weak:
                confidence += 0.10
            # Far below 52wk SMA = strong downtrend
            if dist_52w_sma < -0.15:
                confidence += 0.10
            # NEVER short if both TFs turning (bounce in progress)
            if daily_turning and weekly_turning:
                return {'direction': 'none', 'confidence': 0.0, 'delay_hours': 0}
            # Reduce if daily momentum turning
            if daily_turning:
                confidence -= 0.15

            confidence = float(np.clip(confidence, 0.0, 1.0))
            if confidence >= 0.50:
                return {'direction': 'short', 'confidence': confidence,
                        'delay_hours': 0}

    # ============================================================
    # LONG SIGNAL: sell-off ending, bounce expected
    # ============================================================
    # Conditions: deeply oversold + RSI showing signs of recovery
    if d_pos < 0.35 and w_pos < 0.35:
        confidence = 0.45
        confidence += (0.35 - d_pos) * 1.0
        confidence += (0.35 - w_pos) * 0.5

        # RSI recovering = strongest long signal
        if rsi_rising and tsla_selloff:
            confidence += 0.25   # RSI turning up from below 50
        elif tsla_deep_oversold:
            confidence += 0.15   # extreme oversold even if still falling

        # Monthly oversold
        if monthly and m_pos < 0.35:
            confidence += 0.15

        # Momentum confirmation
        if daily_turning:
            confidence += 0.10
        if weekly_turning:
            confidence += 0.15

        # SPY regime
        if spy_strong:
            confidence += 0.10
        elif spy_weak:
            confidence -= 0.10

        # Mean reversion: far below 52wk SMA = bounce due
        if dist_52w_sma < -0.15:
            confidence += 0.10

        # DON'T go long if RSI still falling hard
        if rsi_falling and tsla_rsi_w > 35 and not daily_turning:
            confidence -= 0.30   # sell-off not done

        confidence = float(np.clip(confidence, 0.0, 1.0))

        delay_hours = 0
        if not daily_turning and not weekly_turning:
            delay_hours = 24
        if tsla_selloff and rsi_falling:
            delay_hours = max(delay_hours, 24)

        if confidence >= 0.50:
            return {'direction': 'long', 'confidence': confidence,
                    'delay_hours': delay_hours}

    # EVOLVE-BLOCK-END

    return {'direction': 'none', 'confidence': 0.0, 'delay_hours': 0}
