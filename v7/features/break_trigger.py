"""
Break Trigger Feature Calculator

Calculates distance to longer timeframe boundaries to predict when a shorter
timeframe channel will break. Channels often break when hitting longer TF
boundaries - this module quantifies those distances and alignments.

Key insight: A 5min channel approaching a daily upper boundary with RSI > 70
is a strong candidate for a break/reversal.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import Channel
from core.timeframe import TIMEFRAMES, get_longer_timeframes


@dataclass
class BreakTriggerFeatures:
    """
    Features for predicting when a channel will break based on longer TF boundaries.

    Attributes:
        nearest_boundary: String like "1h_lower" or "daily_upper" identifying
                         the closest longer timeframe boundary
        nearest_boundary_dist: Percentage distance to the nearest boundary
        distances: Dict mapping "tf_upper"/"tf_lower" to distance as percentage
        rsi_alignment_with_boundary: Score (-1 to 1) indicating if RSI confirms
                                     the boundary signal:
                                     - Near lower boundary + low RSI = positive (bullish alignment)
                                     - Near upper boundary + high RSI = positive (bearish alignment)
                                     - Misaligned = negative
    """
    nearest_boundary: Optional[str]
    nearest_boundary_dist: float
    distances: Dict[str, float] = field(default_factory=dict)
    rsi_alignment_with_boundary: float = 0.0


def calculate_distance_to_boundary(
    current_price: float,
    channel: Channel,
    boundary_type: str
) -> float:
    """
    Calculate percentage distance from current price to a channel boundary.

    Args:
        current_price: Current close price
        channel: Channel object with upper/lower lines
        boundary_type: "upper" or "lower"

    Returns:
        Percentage distance (positive = price below boundary for upper,
                            positive = price above boundary for lower)
    """
    if channel is None or not channel.valid:
        return float('inf')

    if current_price <= 0:
        return float('inf')

    if boundary_type == "upper":
        boundary_value = channel.upper_line[-1]
        # Distance to upper: how far price needs to move up (positive if below)
        distance_pct = ((boundary_value - current_price) / current_price) * 100
    elif boundary_type == "lower":
        boundary_value = channel.lower_line[-1]
        # Distance to lower: how far price is above lower (positive if above)
        distance_pct = ((current_price - boundary_value) / current_price) * 100
    else:
        raise ValueError(f"boundary_type must be 'upper' or 'lower', got {boundary_type}")

    return distance_pct


def calculate_rsi_alignment(
    rsi: float,
    nearest_boundary: Optional[str],
    nearest_distance: float,
    alignment_threshold: float = 1.0
) -> float:
    """
    Calculate how well RSI aligns with the nearest boundary signal.

    Strong alignment means:
    - Near lower boundary (potential bounce up) + RSI oversold (< 30) = strong bullish
    - Near upper boundary (potential bounce down) + RSI overbought (> 70) = strong bearish

    Misalignment means:
    - Near lower boundary + RSI overbought = weak/contradictory signal
    - Near upper boundary + RSI oversold = weak/contradictory signal

    Args:
        rsi: Current RSI value (0-100)
        nearest_boundary: String like "1h_lower" or "daily_upper"
        nearest_distance: Percentage distance to nearest boundary
        alignment_threshold: Maximum distance (%) to consider "near" boundary

    Returns:
        Alignment score from -1 to 1:
        - Positive = aligned (RSI confirms boundary signal)
        - Negative = misaligned (RSI contradicts boundary signal)
        - 0 = neutral or not near any boundary
    """
    if nearest_boundary is None:
        return 0.0

    # Only calculate alignment if actually near a boundary
    if abs(nearest_distance) > alignment_threshold:
        return 0.0

    # Normalize RSI to -1 to 1 scale (50 = 0, 0 = -1, 100 = 1)
    rsi_normalized = (rsi - 50) / 50

    # Proximity factor: closer = stronger signal (linear decay)
    proximity = 1.0 - (abs(nearest_distance) / alignment_threshold)

    if "_lower" in nearest_boundary:
        # Near lower boundary: low RSI is aligned (bullish setup)
        # RSI < 50 gives positive alignment, RSI > 50 gives negative
        alignment = -rsi_normalized * proximity
    elif "_upper" in nearest_boundary:
        # Near upper boundary: high RSI is aligned (bearish setup)
        # RSI > 50 gives positive alignment, RSI < 50 gives negative
        alignment = rsi_normalized * proximity
    else:
        alignment = 0.0

    return float(np.clip(alignment, -1.0, 1.0))


def calculate_break_trigger_features(
    current_price: float,
    current_tf: str,
    longer_tf_channels: Dict[str, Channel],
    rsi: Optional[float] = None
) -> BreakTriggerFeatures:
    """
    Calculate break trigger features: distances to all longer TF boundaries.

    This is the main function for predicting WHEN a channel might break.
    Channels often break when price hits a longer timeframe boundary.

    Args:
        current_price: Current close price
        current_tf: Current timeframe (e.g., '5min', '15min')
        longer_tf_channels: Dict mapping timeframe names to Channel objects
                           for all longer timeframes
        rsi: Optional RSI value for alignment calculation

    Returns:
        BreakTriggerFeatures with distances to all boundaries and nearest info

    Example:
        >>> channels = {'1h': channel_1h, 'daily': channel_daily}
        >>> features = calculate_break_trigger_features(150.0, '5min', channels, rsi=25)
        >>> print(features.nearest_boundary)  # e.g., "1h_lower"
        >>> print(features.distances)  # e.g., {"1h_upper": 2.5, "1h_lower": 0.3, ...}
    """
    distances: Dict[str, float] = {}
    nearest_boundary: Optional[str] = None
    nearest_dist: float = float('inf')

    # Get all longer timeframes for this current TF
    longer_tfs = get_longer_timeframes(current_tf)

    # Calculate distances to all longer TF boundaries
    for tf in longer_tfs:
        channel = longer_tf_channels.get(tf)

        if channel is None or not channel.valid:
            # Mark as infinity if no valid channel
            distances[f"{tf}_upper"] = float('inf')
            distances[f"{tf}_lower"] = float('inf')
            continue

        # Distance to upper boundary
        dist_upper = calculate_distance_to_boundary(current_price, channel, "upper")
        distances[f"{tf}_upper"] = dist_upper

        # Distance to lower boundary
        dist_lower = calculate_distance_to_boundary(current_price, channel, "lower")
        distances[f"{tf}_lower"] = dist_lower

        # Track nearest boundary (use absolute distance for comparison)
        # For upper: positive dist means price is below (need to go up)
        # For lower: positive dist means price is above (need to go down)
        abs_dist_upper = abs(dist_upper) if dist_upper != float('inf') else float('inf')
        abs_dist_lower = abs(dist_lower) if dist_lower != float('inf') else float('inf')

        if abs_dist_upper < nearest_dist:
            nearest_dist = abs_dist_upper
            nearest_boundary = f"{tf}_upper"

        if abs_dist_lower < nearest_dist:
            nearest_dist = abs_dist_lower
            nearest_boundary = f"{tf}_lower"

    # Handle case where no valid boundaries found
    if nearest_dist == float('inf'):
        nearest_dist = 0.0
        nearest_boundary = None

    # Calculate RSI alignment if RSI provided
    rsi_alignment = 0.0
    if rsi is not None and nearest_boundary is not None:
        rsi_alignment = calculate_rsi_alignment(
            rsi, nearest_boundary, nearest_dist
        )

    return BreakTriggerFeatures(
        nearest_boundary=nearest_boundary,
        nearest_boundary_dist=nearest_dist,
        distances=distances,
        rsi_alignment_with_boundary=rsi_alignment
    )


def get_critical_boundaries(
    features: BreakTriggerFeatures,
    threshold_pct: float = 0.5
) -> Dict[str, float]:
    """
    Get boundaries that are critically close (within threshold).

    These are the boundaries most likely to cause a break/reversal.

    Args:
        features: BreakTriggerFeatures object
        threshold_pct: Maximum distance (%) to consider "critical"

    Returns:
        Dict mapping boundary names to distances for those within threshold
    """
    critical = {}
    for boundary, dist in features.distances.items():
        if dist != float('inf') and abs(dist) <= threshold_pct:
            critical[boundary] = dist
    return critical


def features_to_dict(features: BreakTriggerFeatures) -> Dict[str, float]:
    """
    Convert BreakTriggerFeatures to flat dictionary for model input.

    Args:
        features: BreakTriggerFeatures object

    Returns:
        Flat dictionary with all numeric features
    """
    result = {
        'nearest_boundary_dist': features.nearest_boundary_dist,
        'rsi_alignment_with_boundary': features.rsi_alignment_with_boundary,
    }

    # Encode nearest boundary as categorical integers
    # None = 0, tf_lower = 1, tf_upper = 2 for each tf
    if features.nearest_boundary is None:
        result['nearest_boundary_encoded'] = 0
    elif "_lower" in features.nearest_boundary:
        result['nearest_boundary_encoded'] = 1
    else:  # _upper
        result['nearest_boundary_encoded'] = 2

    # Add all individual distances (replace inf with large value for model)
    for boundary, dist in features.distances.items():
        key = f"dist_{boundary}"
        result[key] = dist if dist != float('inf') else 100.0

    return result


def calculate_break_probability_modifier(
    features: BreakTriggerFeatures,
    base_probability: float = 0.5
) -> float:
    """
    Calculate a probability modifier based on break trigger features.

    This can be used to adjust a base break probability:
    - Very close to boundary + aligned RSI = higher break probability
    - Far from all boundaries = lower break probability

    Args:
        features: BreakTriggerFeatures object
        base_probability: Starting probability (default 0.5)

    Returns:
        Modified probability (0 to 1)
    """
    if features.nearest_boundary is None:
        return base_probability

    # Distance effect: closer = higher probability of break/reversal
    # Use sigmoid-like scaling
    dist = features.nearest_boundary_dist
    if dist <= 0.1:
        distance_factor = 0.3  # Very close
    elif dist <= 0.5:
        distance_factor = 0.2  # Close
    elif dist <= 1.0:
        distance_factor = 0.1  # Moderately close
    else:
        distance_factor = 0.0  # Far

    # RSI alignment effect: aligned RSI increases probability
    alignment_factor = features.rsi_alignment_with_boundary * 0.15

    # Combine effects
    modifier = distance_factor + alignment_factor

    # Apply to base probability
    result = base_probability + modifier
    return float(np.clip(result, 0.0, 1.0))
