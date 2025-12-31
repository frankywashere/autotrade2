"""
Learned Channel Projection Calculator (v5.6)

Calculates price projections AFTER model inference, using the model's
predicted channel continuation duration.

Flow:
1. Model predicts duration_bars for each TF (how long channel will continue)
2. This module projects the channel forward by that predicted duration
3. Returns projected price targets at the learned horizon

This replaces the old fixed 24-hour projection features that were
calculated during feature extraction.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class ChannelProjection:
    """Projected channel based on learned duration prediction."""
    timeframe: str
    window: int
    symbol: str

    # Model prediction
    predicted_duration_bars: float
    prediction_confidence: float

    # Current channel state
    current_price: float
    channel_upper: float
    channel_lower: float
    channel_center: float
    high_slope: float  # $/bar
    low_slope: float   # $/bar
    close_slope: float  # $/bar

    # Calculated projections
    projected_upper: float
    projected_lower: float
    projected_center: float

    # As percentages from current price
    projected_high_pct: float
    projected_low_pct: float
    projected_center_pct: float

    # Trading signals
    upside_potential: float  # % to projected upper
    downside_risk: float     # % to projected lower
    risk_reward_ratio: float


def calculate_single_projection(
    predicted_duration: float,
    current_price: float,
    channel_upper: float,
    channel_lower: float,
    high_slope: float,
    low_slope: float,
    close_slope: float,
    confidence: float = 1.0,
    timeframe: str = "",
    window: int = 0,
    symbol: str = "tsla",
) -> ChannelProjection:
    """
    Calculate projected price targets using learned duration.

    Args:
        predicted_duration: Model's predicted bars until channel break
        current_price: Current close price
        channel_upper: Current upper bound of channel
        channel_lower: Current lower bound of channel
        high_slope: Slope of upper bound ($/bar)
        low_slope: Slope of lower bound ($/bar)
        close_slope: Slope of center line ($/bar)
        confidence: Model's confidence in prediction
        timeframe: Timeframe name
        window: Window size
        symbol: Symbol name

    Returns:
        ChannelProjection with calculated targets
    """
    # Clamp duration to reasonable range
    duration = max(1.0, min(predicted_duration, 200.0))

    # Current channel center
    channel_center = (channel_upper + channel_lower) / 2

    # Project channel forward by predicted duration
    projected_upper = channel_upper + (high_slope * duration)
    projected_lower = channel_lower + (low_slope * duration)
    projected_center = channel_center + (close_slope * duration)

    # Convert to percentages from current price
    current_price_safe = max(current_price, 0.01)
    projected_high_pct = (projected_upper - current_price) / current_price_safe * 100
    projected_low_pct = (projected_lower - current_price) / current_price_safe * 100
    projected_center_pct = (projected_center - current_price) / current_price_safe * 100

    # Calculate trading signals
    upside = max(0, (projected_upper - current_price) / current_price_safe * 100)
    downside = max(0, (current_price - projected_lower) / current_price_safe * 100)
    rr_ratio = upside / downside if downside > 0.01 else float('inf')

    return ChannelProjection(
        timeframe=timeframe,
        window=window,
        symbol=symbol,
        predicted_duration_bars=duration,
        prediction_confidence=confidence,
        current_price=current_price,
        channel_upper=channel_upper,
        channel_lower=channel_lower,
        channel_center=channel_center,
        high_slope=high_slope,
        low_slope=low_slope,
        close_slope=close_slope,
        projected_upper=projected_upper,
        projected_lower=projected_lower,
        projected_center=projected_center,
        projected_high_pct=projected_high_pct,
        projected_low_pct=projected_low_pct,
        projected_center_pct=projected_center_pct,
        upside_potential=upside,
        downside_risk=downside,
        risk_reward_ratio=rr_ratio,
    )


def extract_channel_state_from_features(
    features: Dict[str, float],
    symbol: str,
    tf: str,
    window: int,
) -> Optional[Dict[str, float]]:
    """
    Extract current channel state from feature dict.

    Args:
        features: Dict of feature name -> value
        symbol: 'tsla' or 'spy'
        tf: Timeframe name
        window: Window size

    Returns:
        Dict with channel state values, or None if not available
    """
    prefix = f'{symbol}_channel_{tf}_w{window}'

    required_keys = [
        f'{prefix}_high_slope',
        f'{prefix}_low_slope',
        f'{prefix}_close_slope',
        f'{prefix}_position',
        f'{prefix}_upper_dist',
        f'{prefix}_lower_dist',
    ]

    # Check if all required features exist
    for key in required_keys:
        if key not in features:
            return None

    return {
        'high_slope': features[f'{prefix}_high_slope'],
        'low_slope': features[f'{prefix}_low_slope'],
        'close_slope': features[f'{prefix}_close_slope'],
        'position': features[f'{prefix}_position'],
        'upper_dist': features[f'{prefix}_upper_dist'],
        'lower_dist': features[f'{prefix}_lower_dist'],
        'close_slope_pct': features.get(f'{prefix}_close_slope_pct', 0),
        'quality_score': features.get(f'{prefix}_quality_score', 0),
        'is_valid': features.get(f'{prefix}_is_valid', 0),
    }


def reconstruct_channel_bounds(
    current_price: float,
    position: float,
    upper_dist: float,
    lower_dist: float,
) -> Tuple[float, float]:
    """
    Reconstruct channel bounds from position and distances.

    Args:
        current_price: Current close price
        position: Position in channel (0-1)
        upper_dist: Distance to upper bound (%)
        lower_dist: Distance to lower bound (%)

    Returns:
        (channel_upper, channel_lower)
    """
    # upper_dist = (upper - price) / price * 100
    # lower_dist = (price - lower) / price * 100
    channel_upper = current_price * (1 + upper_dist / 100)
    channel_lower = current_price * (1 - lower_dist / 100)

    return channel_upper, channel_lower


def calculate_projections_from_model_output(
    model_output: Dict,
    features: Dict[str, float],
    current_price: float,
    timeframes: List[str] = None,
    windows: List[int] = None,
    symbol: str = 'tsla',
) -> Dict[str, ChannelProjection]:
    """
    Calculate projections for all TF/window combinations from model output.

    Args:
        model_output: Dict from model forward pass, containing:
            - 'duration': Dict[tf, {'mean': tensor, 'std': tensor}]
            - 'per_tf_continuation': Dict[f'cont_{tf}_duration': tensor]
        features: Dict of feature values
        current_price: Current close price
        timeframes: List of timeframes to process
        windows: List of window sizes to process
        symbol: Symbol to process

    Returns:
        Dict of ChannelProjection objects keyed by '{tf}_w{window}'
    """
    if timeframes is None:
        timeframes = ['5min', '15min', '30min', '1h', '2h', '3h', '4h',
                      'daily', 'weekly', 'monthly', '3month']
    if windows is None:
        windows = [100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10]

    projections = {}

    for tf in timeframes:
        # Get predicted duration for this TF
        predicted_duration = None
        confidence = 1.0

        # Try 'duration' output first (probabilistic heads)
        if 'duration' in model_output and tf in model_output['duration']:
            dur_data = model_output['duration'][tf]
            if isinstance(dur_data, dict):
                if 'mean' in dur_data:
                    mean_tensor = dur_data['mean']
                    if isinstance(mean_tensor, torch.Tensor):
                        predicted_duration = mean_tensor[0, 0].item()
                    else:
                        predicted_duration = float(mean_tensor)
                if 'confidence' in dur_data:
                    conf_tensor = dur_data['confidence']
                    if isinstance(conf_tensor, torch.Tensor):
                        confidence = conf_tensor[0, 0].item()
                    else:
                        confidence = float(conf_tensor)

        # Fallback to per_tf_continuation output
        if predicted_duration is None and 'per_tf_continuation' in model_output:
            key = f'cont_{tf}_duration'
            if key in model_output['per_tf_continuation']:
                dur_tensor = model_output['per_tf_continuation'][key]
                if isinstance(dur_tensor, torch.Tensor):
                    predicted_duration = dur_tensor[0, 0].item()
                else:
                    predicted_duration = float(dur_tensor)

        if predicted_duration is None:
            continue

        # Calculate projection for each window
        for window in windows:
            # Get channel state from features
            channel_state = extract_channel_state_from_features(
                features, symbol, tf, window
            )

            if channel_state is None:
                continue

            # Reconstruct channel bounds
            channel_upper, channel_lower = reconstruct_channel_bounds(
                current_price,
                channel_state['position'],
                channel_state['upper_dist'],
                channel_state['lower_dist'],
            )

            # Calculate projection
            projection = calculate_single_projection(
                predicted_duration=predicted_duration,
                current_price=current_price,
                channel_upper=channel_upper,
                channel_lower=channel_lower,
                high_slope=channel_state['high_slope'],
                low_slope=channel_state['low_slope'],
                close_slope=channel_state['close_slope'],
                confidence=confidence,
                timeframe=tf,
                window=window,
                symbol=symbol,
            )

            projections[f'{tf}_w{window}'] = projection

    return projections


def calculate_best_projection(
    projections: Dict[str, ChannelProjection],
    min_confidence: float = 0.5,
    min_rr_ratio: float = 1.5,
) -> Optional[ChannelProjection]:
    """
    Select the best projection based on risk/reward and confidence.

    Args:
        projections: Dict of projections from calculate_projections_from_model_output
        min_confidence: Minimum confidence threshold
        min_rr_ratio: Minimum risk/reward ratio

    Returns:
        Best ChannelProjection, or None if none qualify
    """
    candidates = []

    for key, proj in projections.items():
        if proj.prediction_confidence >= min_confidence:
            if proj.risk_reward_ratio >= min_rr_ratio:
                candidates.append(proj)

    if not candidates:
        return None

    # Sort by risk/reward ratio, weighted by confidence
    candidates.sort(
        key=lambda p: p.risk_reward_ratio * p.prediction_confidence,
        reverse=True
    )

    return candidates[0]


def format_projection_summary(
    projection: ChannelProjection,
) -> str:
    """
    Format a projection for display.

    Args:
        projection: ChannelProjection to format

    Returns:
        Formatted string
    """
    return (
        f"[{projection.symbol.upper()}] {projection.timeframe} w{projection.window}: "
        f"Duration={projection.predicted_duration_bars:.1f} bars, "
        f"Target High={projection.projected_high_pct:+.2f}%, "
        f"Target Low={projection.projected_low_pct:+.2f}%, "
        f"R:R={projection.risk_reward_ratio:.2f}, "
        f"Conf={projection.prediction_confidence:.2f}"
    )
