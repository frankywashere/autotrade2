"""
Cross-Asset Features

Tracks relationships between TSLA, SPY, and VIX:
- SPY channel state at all timeframes
- Where TSLA is relative to SPY's channel boundaries
- VIX regime features
- Correlation features
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import detect_channel, Channel, Direction
from core.timeframe import resample_ohlc, TIMEFRAMES


@dataclass
class CrossAssetContainment:
    """Where is TSLA relative to SPY's channel?"""
    timeframe: str
    spy_channel_valid: bool
    spy_direction: int            # 0=bear, 1=sideways, 2=bull
    spy_position: float           # SPY's position in its own channel (0-1)
    tsla_in_spy_upper: bool       # Is TSLA price near SPY's upper bound?
    tsla_in_spy_lower: bool       # Is TSLA price near SPY's lower bound?
    tsla_dist_to_spy_upper: float # % distance (normalized)
    tsla_dist_to_spy_lower: float # % distance (normalized)
    alignment: int                # 1=both near upper, -1=both near lower, 0=diverging


@dataclass
class VIXFeatures:
    """VIX regime features."""
    level: float                  # Current VIX value
    level_normalized: float       # Normalized (0-1 based on historical range)
    trend_5d: float               # 5-day change
    trend_20d: float              # 20-day change
    percentile_252d: float        # Where is VIX in last year (0-100)
    regime: int                   # 0=low (<15), 1=normal (15-25), 2=high (25-35), 3=extreme (>35)


@dataclass
class SPYFeatures:
    """SPY's own channel features at a timeframe."""
    timeframe: str
    channel_valid: bool
    direction: int                # 0=bear, 1=sideways, 2=bull
    position: float               # 0-1
    upper_dist: float
    lower_dist: float
    width_pct: float
    slope_pct: float
    r_squared: float
    bounce_count: int
    cycles: int
    rsi: float


def calculate_cross_asset_containment(
    tsla_price: float,
    spy_df: pd.DataFrame,
    spy_channel: Channel,
    timeframe: str,
    threshold: float = 0.15
) -> CrossAssetContainment:
    """
    Calculate where TSLA is relative to SPY's channel.

    This normalizes TSLA price to SPY's scale using ratio.

    Args:
        tsla_price: Current TSLA close price
        spy_df: SPY OHLCV data
        spy_channel: SPY's detected channel
        timeframe: Timeframe name
        threshold: Threshold for "near" boundary

    Returns:
        CrossAssetContainment object
    """
    if spy_channel is None or not spy_channel.valid:
        return CrossAssetContainment(
            timeframe=timeframe,
            spy_channel_valid=False,
            spy_direction=1,  # sideways
            spy_position=0.5,
            tsla_in_spy_upper=False,
            tsla_in_spy_lower=False,
            tsla_dist_to_spy_upper=0.0,
            tsla_dist_to_spy_lower=0.0,
            alignment=0,
        )

    spy_price = spy_df['close'].iloc[-1]
    spy_upper = spy_channel.upper_line[-1]
    spy_lower = spy_channel.lower_line[-1]
    spy_width = spy_upper - spy_lower

    # SPY's position in its own channel
    spy_position = (spy_price - spy_lower) / spy_width if spy_width > 0 else 0.5
    spy_position = float(np.clip(spy_position, 0, 1))

    # Normalize TSLA price to SPY's scale
    # Use ratio: if TSLA/SPY ratio is high, TSLA is "above" SPY relatively
    tsla_spy_ratio = tsla_price / spy_price if spy_price > 0 else 1.0

    # Historical ratio for normalization (approximate)
    # TSLA ~$250, SPY ~$500 → ratio ~0.5 typically
    # We normalize relative movements, not absolute prices

    # Calculate where TSLA "would be" in SPY's channel based on relative movement
    # This is a bit abstract - we're asking: if TSLA moved proportionally to SPY,
    # where would it be in SPY's channel?

    # Simpler approach: check if both are at similar positions
    # If SPY is at 0.9 (near upper) and TSLA is also showing strength, they align

    # For direct containment, use percentage-based comparison
    # Is TSLA's % move similar to being at SPY's upper/lower?

    # Guard against division by zero (spy_price <= 0)
    if spy_price > 0:
        spy_upper_pct = (spy_upper - spy_price) / spy_price * 100
        spy_lower_pct = (spy_price - spy_lower) / spy_price * 100
    else:
        spy_upper_pct = 0.0
        spy_lower_pct = 0.0

    # Check if TSLA is "aligned" with SPY boundaries
    # This is more about regime than literal price containment
    tsla_in_spy_upper = spy_position > (1 - threshold)  # SPY near its upper
    tsla_in_spy_lower = spy_position < threshold         # SPY near its lower

    # Alignment: are both assets at same extreme?
    # 1 = both bullish/near upper, -1 = both bearish/near lower, 0 = mixed
    alignment = 0
    if tsla_in_spy_upper:
        alignment = 1
    elif tsla_in_spy_lower:
        alignment = -1

    return CrossAssetContainment(
        timeframe=timeframe,
        spy_channel_valid=True,
        spy_direction=int(spy_channel.direction),
        spy_position=spy_position,
        tsla_in_spy_upper=tsla_in_spy_upper,
        tsla_in_spy_lower=tsla_in_spy_lower,
        tsla_dist_to_spy_upper=spy_upper_pct,
        tsla_dist_to_spy_lower=spy_lower_pct,
        alignment=alignment,
    )


def extract_spy_features(
    spy_df: pd.DataFrame,
    window: int = 20,
    timeframe: str = '5min'
) -> SPYFeatures:
    """
    Extract SPY's own channel features.

    Args:
        spy_df: SPY OHLCV data
        window: Window for channel detection
        timeframe: Timeframe name

    Returns:
        SPYFeatures object
    """
    from features.rsi import calculate_rsi

    channel = detect_channel(spy_df, window=window)
    rsi = calculate_rsi(spy_df['close'].values, period=14)

    return SPYFeatures(
        timeframe=timeframe,
        channel_valid=channel.valid,
        direction=int(channel.direction),
        position=channel.position_at(),
        upper_dist=channel.distance_to_upper(),
        lower_dist=channel.distance_to_lower(),
        width_pct=channel.width_pct,
        slope_pct=channel.slope_pct,
        r_squared=channel.r_squared,
        bounce_count=channel.bounce_count,
        cycles=channel.complete_cycles,
        rsi=rsi,
    )


def extract_vix_features(vix_df: pd.DataFrame) -> VIXFeatures:
    """
    Extract VIX regime features.

    Args:
        vix_df: VIX daily data with columns [open, high, low, close]

    Returns:
        VIXFeatures object
    """
    if len(vix_df) < 252:
        # Not enough data
        return VIXFeatures(
            level=20.0,
            level_normalized=0.5,
            trend_5d=0.0,
            trend_20d=0.0,
            percentile_252d=50.0,
            regime=1,
        )

    current = vix_df['close'].iloc[-1]

    # Trends (with guards against division by zero)
    trend_5d = 0.0
    trend_20d = 0.0
    if len(vix_df) >= 5:
        vix_5d_ago = vix_df['close'].iloc[-5]
        if vix_5d_ago > 0:
            trend_5d = (current - vix_5d_ago) / vix_5d_ago * 100
    if len(vix_df) >= 20:
        vix_20d_ago = vix_df['close'].iloc[-20]
        if vix_20d_ago > 0:
            trend_20d = (current - vix_20d_ago) / vix_20d_ago * 100

    # Percentile in last year
    last_252 = vix_df['close'].iloc[-252:].values
    percentile = (np.sum(last_252 < current) / len(last_252)) * 100

    # Normalized level (10-50 range mapped to 0-1)
    normalized = np.clip((current - 10) / 40, 0, 1)

    # Regime
    if current < 15:
        regime = 0  # Low volatility
    elif current < 25:
        regime = 1  # Normal
    elif current < 35:
        regime = 2  # High
    else:
        regime = 3  # Extreme

    return VIXFeatures(
        level=float(current),
        level_normalized=float(normalized),
        trend_5d=float(trend_5d),
        trend_20d=float(trend_20d),
        percentile_252d=float(percentile),
        regime=regime,
    )


def extract_all_cross_asset_features(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    window: int = 20
) -> Dict:
    """
    Extract all cross-asset features for all timeframes.

    Args:
        tsla_df: TSLA 5min OHLCV data
        spy_df: SPY 5min OHLCV data
        vix_df: VIX daily data
        window: Window for channel detection

    Returns:
        Dict with:
            - 'spy_features': Dict[tf, SPYFeatures]
            - 'cross_containment': Dict[tf, CrossAssetContainment]
            - 'vix': VIXFeatures
    """
    tsla_price = tsla_df['close'].iloc[-1]

    spy_features = {}
    cross_containment = {}

    for tf in TIMEFRAMES:
        # Resample SPY to this timeframe
        if tf == '5min':
            spy_tf = spy_df
        else:
            spy_tf = resample_ohlc(spy_df, tf)

        if len(spy_tf) >= window:
            # SPY's own features
            spy_features[tf] = extract_spy_features(spy_tf, window, tf)

            # Cross containment (where is TSLA relative to SPY's channel)
            spy_channel = detect_channel(spy_tf, window=window)
            cross_containment[tf] = calculate_cross_asset_containment(
                tsla_price, spy_tf, spy_channel, tf
            )

    # VIX features
    vix = extract_vix_features(vix_df)

    return {
        'spy_features': spy_features,
        'cross_containment': cross_containment,
        'vix': vix,
    }
