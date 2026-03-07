"""Recovery-day trade gate for forward sim V2.

OpenEvolve will evolve the logic inside EVOLVE-BLOCK to learn which
afternoon trades to KEEP vs SKIP based on what happened in the morning.

The key insight: on recovery days (morning bearish, afternoon reversal),
shorting in the afternoon causes large losses. The gate should detect
these patterns and suppress shorts while allowing longs.

Available features (all floats):
  scanner:        0=CS-5TF, 1=CS-DW, 2=ML
  signal_type:    0=bounce, 1=break
  direction:      0=long, 1=short
  primary_tf:     0=5min, 1=1h, 2=4h, 3=daily, 4=weekly
  confidence:     0.0-1.0 (signal confidence)
  entry_hour:     9.5-16.0 (ET, decimal hours)
  prev_vix:       previous day's VIX close
  day_of_week:    0=Mon, 1=Tue, ..., 4=Fri
  is_event_day:   1.0 if calendar event day, else 0.0
  is_near_event:  1.0 if within 1 trading day of event, else 0.0

  ** Intraday context (new for recovery detection) **
  morning_return: % change from day open to current price (negative = down day so far)
  morning_range:  % range of today's high-low relative to open (volatility)
  am_return:      % return at AM cutoff (first hour close vs open)
  price_vs_am:    % change from AM close to current price (positive = recovering)
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
    morning_return: float = 0.0,
    morning_range: float = 0.0,
    am_return: float = 0.0,
    price_vs_am: float = 0.0,
) -> bool:
    """Return True to TAKE the trade, False to SKIP it.

    Goal: maximize P&L by detecting recovery days and suppressing shorts
    that would fight the reversal. Keep longs and morning trades intact.

    Recovery pattern: am_return < 0 (bearish morning) but price_vs_am > 0
    (price recovering above AM close). On these days, afternoon shorts
    are likely to lose.
    """

    # EVOLVE-BLOCK-START

    # ── First hour trades: always allow ─────────────────────────────────
    if entry_hour <= 10.5:
        return True

    # ── Afternoon longs: always allow ──────────────────────────────────
    if direction == 0:  # long
        return True

    # ── Recovery detection: suppress afternoon shorts on recovery days ──
    # Morning was bearish (am_return < -0.5%) but price is recovering
    if am_return < -0.5 and price_vs_am > 0.3:
        # Recovering from bearish morning — skip shorts
        return False

    # ── Wide morning range + recovering: likely reversal day ───────────
    if morning_range > 2.0 and price_vs_am > 0:
        # Big morning range and price above AM close — skip shorts
        return False

    # ── High VIX + afternoon shorts: extra caution ─────────────────────
    if prev_vix > 25 and entry_hour > 12.0:
        if morning_return > -0.5:
            # VIX high but market not dropping — skip shorts
            return False

    # ── Low confidence afternoon shorts: skip ──────────────────────────
    if confidence < 0.60 and entry_hour > 11.0:
        return False

    # EVOLVE-BLOCK-END

    return True
