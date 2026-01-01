"""
Channel Feature Extractor

Extracts all per-bar features from channels, RSI, and containment.
This is the main feature extraction interface.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import detect_channel, Channel, Direction, TouchType
from core.timeframe import resample_ohlc, TIMEFRAMES, get_longer_timeframes
from features.rsi import calculate_rsi, calculate_rsi_series, detect_rsi_divergence
from features.containment import check_all_containments, ContainmentInfo


@dataclass
class BarFeatures:
    """All features for a single bar."""
    timestamp: pd.Timestamp

    # Channel state
    channel_valid: bool
    channel_direction: int        # 0=bear, 1=sideways, 2=bull
    channel_position: float       # 0-1 (lower to upper)
    channel_upper_dist: float     # % distance to upper
    channel_lower_dist: float     # % distance to lower
    channel_width_pct: float      # channel width as % of price
    channel_slope_pct: float      # slope as % per bar
    channel_r_squared: float      # linear fit quality
    channel_bounce_count: int     # bounces so far
    channel_cycles: int           # complete cycles
    channel_bars_since_bounce: int
    channel_last_touch: int       # 0=lower, 1=upper, -1=none

    # RSI
    rsi_14: float
    rsi_divergence: int           # 1=bullish, -1=bearish, 0=none

    # Containment (against longer TFs)
    containments: Dict[str, ContainmentInfo] = field(default_factory=dict)
    nearest_boundary: Optional[str] = None


def extract_channel_features(
    df: pd.DataFrame,
    window: int = 20,
    timeframe: str = '5min',
    include_containment: bool = True
) -> BarFeatures:
    """
    Extract all features for the current (last) bar.

    Args:
        df: OHLCV DataFrame
        window: Window size for channel detection
        timeframe: Current timeframe
        include_containment: Whether to check containment against longer TFs

    Returns:
        BarFeatures object with all extracted features
    """
    # Detect channel
    channel = detect_channel(df, window=window)

    # Calculate RSI
    rsi = calculate_rsi(df['close'].values, period=14)
    rsi_series = calculate_rsi_series(df['close'].values, period=14)
    divergence = detect_rsi_divergence(df['close'].values, rsi_series, lookback=10)

    # Last touch
    if channel.last_touch is None:
        last_touch = -1
    else:
        last_touch = int(channel.last_touch)

    # Check containment against longer TFs
    containments = {}
    nearest_boundary = None
    if include_containment:
        containments = check_all_containments(df, timeframe, window)
        # Find nearest boundary
        nearest_dist = float('inf')
        for tf, info in containments.items():
            if not info.longer_valid:
                continue
            if abs(info.distance_to_upper) < nearest_dist:
                nearest_dist = abs(info.distance_to_upper)
                nearest_boundary = f"{tf}_upper"
            if abs(info.distance_to_lower) < nearest_dist:
                nearest_dist = abs(info.distance_to_lower)
                nearest_boundary = f"{tf}_lower"

    return BarFeatures(
        timestamp=df.index[-1],
        channel_valid=channel.valid,
        channel_direction=int(channel.direction),
        channel_position=channel.position_at(),
        channel_upper_dist=channel.distance_to_upper(),
        channel_lower_dist=channel.distance_to_lower(),
        channel_width_pct=channel.width_pct,
        channel_slope_pct=channel.slope_pct,
        channel_r_squared=channel.r_squared,
        channel_bounce_count=channel.bounce_count,
        channel_cycles=channel.complete_cycles,
        channel_bars_since_bounce=channel.bars_since_last_touch,
        channel_last_touch=last_touch,
        rsi_14=rsi,
        rsi_divergence=divergence,
        containments=containments,
        nearest_boundary=nearest_boundary,
    )


def features_to_dict(features: BarFeatures) -> Dict[str, Any]:
    """Convert BarFeatures to flat dictionary for model input."""
    d = {
        'timestamp': features.timestamp,
        'channel_valid': int(features.channel_valid),
        'channel_direction': features.channel_direction,
        'channel_position': features.channel_position,
        'channel_upper_dist': features.channel_upper_dist,
        'channel_lower_dist': features.channel_lower_dist,
        'channel_width_pct': features.channel_width_pct,
        'channel_slope_pct': features.channel_slope_pct,
        'channel_r_squared': features.channel_r_squared,
        'channel_bounce_count': features.channel_bounce_count,
        'channel_cycles': features.channel_cycles,
        'channel_bars_since_bounce': features.channel_bars_since_bounce,
        'channel_last_touch': features.channel_last_touch,
        'rsi_14': features.rsi_14,
        'rsi_divergence': features.rsi_divergence,
    }

    # Flatten containment info
    for tf, info in features.containments.items():
        d[f'near_{tf}_upper'] = int(info.near_upper)
        d[f'near_{tf}_lower'] = int(info.near_lower)
        d[f'dist_{tf}_upper'] = info.distance_to_upper
        d[f'dist_{tf}_lower'] = info.distance_to_lower

    return d


def extract_features_for_all_timeframes(
    df_base: pd.DataFrame,
    window: int = 20
) -> Dict[str, BarFeatures]:
    """
    Extract features for all timeframes from base data.

    Args:
        df_base: Base OHLCV data (e.g., 5min bars)
        window: Window size for channel detection

    Returns:
        Dict mapping timeframe to BarFeatures
    """
    result = {}

    for tf in TIMEFRAMES:
        # Resample to this timeframe
        if tf == '5min':
            df_tf = df_base
        else:
            df_tf = resample_ohlc(df_base, tf)

        try:
            result[tf] = extract_channel_features(
                df_tf,
                window=window,
                timeframe=tf,
                include_containment=True
            )
        except (ValueError, IndexError):
            # Skip if insufficient data
            pass

    return result
