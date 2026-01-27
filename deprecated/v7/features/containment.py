"""
Multi-Timeframe Containment Checker

For each timeframe, checks if current price is near the boundaries
of all LONGER timeframes. This helps predict breaks - if price is
hitting a daily upper bound while in a 5min channel, it's more
likely to reverse.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import detect_channel, Channel
from core.timeframe import resample_ohlc, TIMEFRAMES, get_longer_timeframes


@dataclass
class ContainmentInfo:
    """Information about containment against a longer timeframe."""
    longer_tf: str
    near_upper: bool          # Is price within threshold of longer TF upper?
    near_lower: bool          # Is price within threshold of longer TF lower?
    distance_to_upper: float  # % distance to longer TF upper
    distance_to_lower: float  # % distance to longer TF lower
    longer_direction: str     # Direction of longer TF channel (BULL/BEAR/SIDEWAYS)
    longer_valid: bool        # Is longer TF channel valid?


def check_containment(
    current_price: float,
    current_tf: str,
    longer_tf_channels: Dict[str, Channel],
    threshold: float = 0.10
) -> Dict[str, ContainmentInfo]:
    """
    Check if current price is near boundaries of longer timeframe channels.

    Args:
        current_price: Current close price
        current_tf: Current timeframe (e.g., '5min')
        longer_tf_channels: Dict mapping longer TF names to their Channel objects
        threshold: Threshold for "near" as fraction of channel width

    Returns:
        Dict mapping longer TF names to ContainmentInfo
    """
    result = {}

    for tf, channel in longer_tf_channels.items():
        if channel is None or not channel.valid:
            result[tf] = ContainmentInfo(
                longer_tf=tf,
                near_upper=False,
                near_lower=False,
                distance_to_upper=0.0,
                distance_to_lower=0.0,
                longer_direction='UNKNOWN',
                longer_valid=False,
            )
            continue

        # Get current boundary values (last bar of longer TF channel)
        upper = channel.upper_line[-1]
        lower = channel.lower_line[-1]
        width = upper - lower

        if width <= 0:
            result[tf] = ContainmentInfo(
                longer_tf=tf,
                near_upper=False,
                near_lower=False,
                distance_to_upper=0.0,
                distance_to_lower=0.0,
                longer_direction=channel.direction.name,
                longer_valid=False,
            )
            continue

        # Skip if current_price is invalid (prevents division by zero)
        if current_price <= 0:
            result[tf] = ContainmentInfo(
                longer_tf=tf,
                near_upper=False,
                near_lower=False,
                distance_to_upper=0.0,
                distance_to_lower=0.0,
                longer_direction=channel.direction.name,
                longer_valid=False,
            )
            continue

        # Calculate distances
        dist_to_upper = (upper - current_price) / current_price * 100
        dist_to_lower = (current_price - lower) / current_price * 100

        # Check if "near" (within threshold of channel width)
        near_upper = (upper - current_price) / width <= threshold
        near_lower = (current_price - lower) / width <= threshold

        result[tf] = ContainmentInfo(
            longer_tf=tf,
            near_upper=near_upper,
            near_lower=near_lower,
            distance_to_upper=dist_to_upper,
            distance_to_lower=dist_to_lower,
            longer_direction=channel.direction.name,
            longer_valid=channel.valid,
        )

    return result


def check_all_containments(
    df_base: pd.DataFrame,
    current_tf: str,
    window: int = 20,
    resample_cache: Optional[Dict[str, pd.DataFrame]] = None,
    channel_cache: Optional[Dict[str, Channel]] = None,
) -> Dict[str, ContainmentInfo]:
    """
    Check containment against all longer timeframes.

    Args:
        df_base: Base OHLCV data (e.g., 5min bars)
        current_tf: Current timeframe
        window: Window size for channel detection
        resample_cache: Optional cache dict for resampled DataFrames (keyed by timeframe).
                        If provided, will use cached data and store new resamples.
        channel_cache: Optional cache dict for detected Channels (keyed by timeframe).
                       If provided, will use cached channels and skip re-detection.

    Returns:
        Dict mapping longer TF names to ContainmentInfo
    """
    current_price = df_base['close'].iloc[-1]
    longer_tfs = get_longer_timeframes(current_tf)

    # Initialize local caches if not provided
    if resample_cache is None:
        resample_cache = {}
    if channel_cache is None:
        channel_cache = {}

    # Detect channels at each longer timeframe
    longer_channels = {}
    for tf in longer_tfs:
        # Check channel cache first
        if tf in channel_cache:
            longer_channels[tf] = channel_cache[tf]
            continue

        # Check resample cache or compute
        if tf in resample_cache:
            df_tf = resample_cache[tf]
        else:
            df_tf = resample_ohlc(df_base, tf)
            resample_cache[tf] = df_tf

        try:
            channel = detect_channel(df_tf, window=window)
            longer_channels[tf] = channel
            channel_cache[tf] = channel
        except (ValueError, IndexError):
            # Insufficient data - set to None
            longer_channels[tf] = None
            channel_cache[tf] = None

    return check_containment(current_price, current_tf, longer_channels)


def get_closest_boundary(containments: Dict[str, ContainmentInfo]) -> Optional[str]:
    """
    Find which longer timeframe boundary is closest to current price.

    Returns:
        String like "1h_upper" or "daily_lower", or None
    """
    closest = None
    closest_dist = float('inf')

    for tf, info in containments.items():
        if not info.longer_valid:
            continue

        if info.distance_to_upper < closest_dist:
            closest_dist = info.distance_to_upper
            closest = f"{tf}_upper"

        if info.distance_to_lower < closest_dist:
            closest_dist = info.distance_to_lower
            closest = f"{tf}_lower"

    return closest
