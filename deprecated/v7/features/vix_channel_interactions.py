"""
VIX-Channel Interaction Features

Captures meaningful relationships between VIX regime and channel behavior:
1. VIX at channel events (when bounces/breaks occur)
2. VIX-bounce correlations (do bounces happen at specific VIX levels?)
3. VIX regime effects (how channel behavior changes with volatility)
4. Predictive features (signals of imminent channel breaks)

This module extracts features that help predict:
- Channel break reliability (breaks more likely in high VIX?)
- Bounce reliability (bounces hold better in low VIX?)
- Channel duration by regime (do channels live longer in low volatility?)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy import stats


@dataclass
class VIXChannelInteractionFeatures:
    """
    VIX-channel interaction metrics.

    All features are float type for model compatibility.
    """
    # 1. VIX at channel events (3 features)
    vix_at_last_bounce: float = 0.0
    """VIX level when the last bounce (channel touch) occurred"""

    vix_at_channel_start: float = 0.0
    """VIX level when current channel formed (first bar in window)"""

    vix_change_during_channel: float = 0.0
    """% change in VIX since channel started (positive = VIX rising)"""

    # 2. VIX-bounce relationships (3 features)
    avg_vix_at_upper_bounces: float = 0.0
    """Average VIX level at all upper boundary touches (high volatility at tops?)"""

    avg_vix_at_lower_bounces: float = 0.0
    """Average VIX level at all lower boundary touches (lower volatility at bottoms?)"""

    vix_bounce_level_ratio: float = 0.0
    """Ratio: avg_vix_upper / avg_vix_lower (shows if VIX differs at boundaries)"""

    # 3. VIX regime effects (4 features)
    bounces_in_high_vix_count: float = 0.0
    """Count of bounces when VIX > 25 (bounces in stress regimes?)"""

    bounces_in_low_vix_count: float = 0.0
    """Count of bounces when VIX < 15 (bounces in calm regimes?)"""

    high_vix_bounce_ratio: float = 0.0
    """Proportion of total bounces in high VIX (0-1, higher=more bounces in stress)"""

    channel_age_vs_vix_correlation: float = 0.0
    """Correlation between bars_in_channel and VIX level (do channels age with VIX?)"""

    # 4. Predictive features for break likelihood (3 features)
    vix_momentum_at_boundary: float = 0.0
    """VIX 3-bar momentum when price near channel edge (positive=rising VIX at edge)"""

    vix_distance_from_mean: float = 0.0
    """Current VIX as distance from 20-bar mean (std devs) (extreme=setup for break?)"""

    vix_regime_alignment: float = 0.0
    """Is channel direction aligned with VIX regime? 1=aligned, 0=neutral, -1=diverged"""

    # 5. Bounce resilience predictors (2 features)
    avg_bars_between_bounces_by_vix: float = 0.0
    """Average bars between bounces, scaled by VIX regime (higher=more stable)"""

    high_vix_bounce_frequency: float = 0.0
    """Bounces per bar when VIX > 25 (frequency of bounces in stress)"""

    # Metadata for debugging
    num_bounces_in_window: int = 0
    num_upper_touches: int = 0
    num_lower_touches: int = 0
    bars_in_channel: int = 0


def calculate_vix_channel_interactions(
    df_price: pd.DataFrame,
    df_vix: pd.DataFrame,
    channel,  # Channel object from detect_channel()
    window: int = 50
) -> VIXChannelInteractionFeatures:
    """
    Calculate VIX-channel interaction features.

    Args:
        df_price: Price DataFrame with OHLCV data (index is DatetimeIndex)
        df_vix: VIX DataFrame with index DatetimeIndex, needs 'close' column
        channel: Channel object from detect_channel()
        window: Window size used for channel detection

    Returns:
        VIXChannelInteractionFeatures object with all 15 features

    Raises:
        ValueError: If data alignment issues or insufficient data
    """
    # Initialize with defaults
    features = VIXChannelInteractionFeatures()

    # Store metadata
    features.num_bounces_in_window = len(channel.touches)
    features.num_upper_touches = channel.upper_touches
    features.num_lower_touches = channel.lower_touches
    features.bars_in_channel = window

    # Get the data slice used for channel detection
    if len(df_price) < window:
        return features  # Insufficient data

    df_slice = df_price.iloc[-window:]

    # Align VIX data with price data
    vix_aligned = _align_vix_to_price(df_slice, df_vix)
    if vix_aligned is None or len(vix_aligned) < 2:
        return features  # Can't align VIX data

    # ========== 1. VIX at channel events ==========
    features.vix_at_channel_start = vix_aligned.iloc[0]

    if len(channel.touches) > 0:
        last_touch_idx = channel.touches[-1].bar_index
        if 0 <= last_touch_idx < len(vix_aligned):
            features.vix_at_last_bounce = vix_aligned.iloc[last_touch_idx]

    if len(vix_aligned) > 0:
        features.vix_change_during_channel = (
            (vix_aligned.iloc[-1] - vix_aligned.iloc[0]) / vix_aligned.iloc[0] * 100
            if vix_aligned.iloc[0] > 0 else 0.0
        )

    # ========== 2. VIX-bounce relationships ==========
    if len(channel.touches) > 0:
        upper_vix_levels = []
        lower_vix_levels = []

        for touch in channel.touches:
            if 0 <= touch.bar_index < len(vix_aligned):
                vix_at_touch = vix_aligned.iloc[touch.bar_index]
                if touch.touch_type == 1:  # TouchType.UPPER
                    upper_vix_levels.append(vix_at_touch)
                else:  # TouchType.LOWER
                    lower_vix_levels.append(vix_at_touch)

        if upper_vix_levels:
            features.avg_vix_at_upper_bounces = float(np.mean(upper_vix_levels))

        if lower_vix_levels:
            features.avg_vix_at_lower_bounces = float(np.mean(lower_vix_levels))

        # Ratio of VIX at upper vs lower
        if features.avg_vix_at_lower_bounces > 0:
            features.vix_bounce_level_ratio = (
                features.avg_vix_at_upper_bounces / features.avg_vix_at_lower_bounces
            )

    # ========== 3. VIX regime effects ==========
    high_vix_bounces = 0
    low_vix_bounces = 0
    high_vix_bars = 0

    for i, vix_val in enumerate(vix_aligned):
        if vix_val > 25:
            high_vix_bars += 1

        # Check if this bar had a bounce
        for touch in channel.touches:
            if touch.bar_index == i:
                if vix_val > 25:
                    high_vix_bounces += 1
                elif vix_val < 15:
                    low_vix_bounces += 1

    features.bounces_in_high_vix_count = float(high_vix_bounces)
    features.bounces_in_low_vix_count = float(low_vix_bounces)

    if len(channel.touches) > 0:
        features.high_vix_bounce_ratio = (
            high_vix_bounces / len(channel.touches)
        )

    # Correlation: channel age vs VIX level
    if len(vix_aligned) >= 3:
        bar_indices = np.arange(len(vix_aligned))
        vix_values = vix_aligned.values.astype(float)

        # Remove any NaN values
        valid_mask = ~(np.isnan(bar_indices) | np.isnan(vix_values))
        if valid_mask.sum() >= 3:
            bar_indices_clean = bar_indices[valid_mask]
            vix_values_clean = vix_values[valid_mask]

            corr = np.corrcoef(bar_indices_clean, vix_values_clean)[0, 1]
            features.channel_age_vs_vix_correlation = float(np.nan_to_num(corr, 0.0))

    # ========== 4. Predictive features for breaks ==========
    # VIX momentum when near boundary
    last_position = channel.position_at()
    if last_position > 0.8 or last_position < 0.2:  # Near boundary
        # Calculate 3-bar VIX momentum
        if len(vix_aligned) >= 3:
            vix_momentum = (vix_aligned.iloc[-1] - vix_aligned.iloc[-3]) / vix_aligned.iloc[-3] * 100
            features.vix_momentum_at_boundary = vix_momentum

    # VIX distance from 20-bar mean
    if len(vix_aligned) >= 20:
        vix_20ma = vix_aligned.iloc[-20:].mean()
        vix_std = vix_aligned.iloc[-20:].std()
        current_vix = vix_aligned.iloc[-1]

        if vix_std > 0:
            features.vix_distance_from_mean = (current_vix - vix_20ma) / vix_std

    # VIX regime alignment with channel
    # Channel in uptrend + VIX falling = alignment (confidence in move)
    # Channel in uptrend + VIX rising = divergence (warning signal)
    vix_trend = 1 if features.vix_change_during_channel > 0 else -1
    channel_direction = channel.direction  # 0=bear, 1=sideways, 2=bull

    if channel_direction == 2:  # Bull
        features.vix_regime_alignment = 1.0 if vix_trend < 0 else (-1.0 if vix_trend > 0 else 0.0)
    elif channel_direction == 0:  # Bear
        features.vix_regime_alignment = 1.0 if vix_trend > 0 else (-1.0 if vix_trend < 0 else 0.0)
    else:  # Sideways
        features.vix_regime_alignment = 0.0

    # ========== 5. Bounce resilience predictors ==========
    # Average bars between bounces, scaled by VIX
    if len(channel.touches) >= 2:
        inter_bounce_bars = []
        for i in range(len(channel.touches) - 1):
            bars = channel.touches[i+1].bar_index - channel.touches[i].bar_index
            inter_bounce_bars.append(bars)

        if inter_bounce_bars:
            avg_bars = np.mean(inter_bounce_bars)
            # Scale by VIX regime: divide by (current_vix / 20) to normalize
            current_vix = vix_aligned.iloc[-1]
            vix_scalar = (current_vix / 20.0) if current_vix > 0 else 1.0
            features.avg_bars_between_bounces_by_vix = avg_bars / vix_scalar

    # Bounce frequency in high VIX
    if high_vix_bars > 0:
        features.high_vix_bounce_frequency = high_vix_bounces / high_vix_bars

    return features


def _align_vix_to_price(df_price: pd.DataFrame, df_vix: pd.DataFrame) -> Optional[pd.Series]:
    """
    Align VIX data to price data by matching dates.

    VIX is typically daily data, so we forward-fill to intraday prices.

    Args:
        df_price: Price DataFrame with DatetimeIndex
        df_vix: VIX DataFrame with DatetimeIndex

    Returns:
        Series of VIX values aligned to price dates, or None if can't align
    """
    try:
        # Get the date range from price data
        start_date = df_price.index.min()
        end_date = df_price.index.max()

        # Filter VIX to overlapping range
        vix_filtered = df_vix.loc[
            (df_vix.index >= start_date) & (df_vix.index <= end_date),
            'close'
        ]

        if len(vix_filtered) == 0:
            return None

        # Reindex to price dates and forward-fill
        vix_aligned = vix_filtered.reindex(df_price.index, method='ffill')

        # Fill any remaining NaNs with backward fill
        vix_aligned = vix_aligned.bfill()

        if vix_aligned.isna().all():
            return None

        return vix_aligned

    except Exception as e:
        return None


def features_to_dict(features: VIXChannelInteractionFeatures) -> Dict[str, float]:
    """
    Convert VIXChannelInteractionFeatures to flat dictionary for model input.

    Returns 15 features as floats:
    1. vix_at_last_bounce
    2. vix_at_channel_start
    3. vix_change_during_channel
    4. avg_vix_at_upper_bounces
    5. avg_vix_at_lower_bounces
    6. vix_bounce_level_ratio
    7. bounces_in_high_vix_count
    8. bounces_in_low_vix_count
    9. high_vix_bounce_ratio
    10. channel_age_vs_vix_correlation
    11. vix_momentum_at_boundary
    12. vix_distance_from_mean
    13. vix_regime_alignment
    14. avg_bars_between_bounces_by_vix
    15. high_vix_bounce_frequency

    Args:
        features: VIXChannelInteractionFeatures object

    Returns:
        Dict mapping feature names to float values
    """
    return {
        'vix_at_last_bounce': features.vix_at_last_bounce,
        'vix_at_channel_start': features.vix_at_channel_start,
        'vix_change_during_channel': features.vix_change_during_channel,
        'avg_vix_at_upper_bounces': features.avg_vix_at_upper_bounces,
        'avg_vix_at_lower_bounces': features.avg_vix_at_lower_bounces,
        'vix_bounce_level_ratio': features.vix_bounce_level_ratio,
        'bounces_in_high_vix_count': features.bounces_in_high_vix_count,
        'bounces_in_low_vix_count': features.bounces_in_low_vix_count,
        'high_vix_bounce_ratio': features.high_vix_bounce_ratio,
        'channel_age_vs_vix_correlation': features.channel_age_vs_vix_correlation,
        'vix_momentum_at_boundary': features.vix_momentum_at_boundary,
        'vix_distance_from_mean': features.vix_distance_from_mean,
        'vix_regime_alignment': features.vix_regime_alignment,
        'avg_bars_between_bounces_by_vix': features.avg_bars_between_bounces_by_vix,
        'high_vix_bounce_frequency': features.high_vix_bounce_frequency,
    }


def get_feature_names() -> List[str]:
    """
    Get canonical list of VIX-channel interaction feature names.

    Returns:
        List of 15 feature names in canonical order
    """
    return [
        'vix_at_last_bounce',
        'vix_at_channel_start',
        'vix_change_during_channel',
        'avg_vix_at_upper_bounces',
        'avg_vix_at_lower_bounces',
        'vix_bounce_level_ratio',
        'bounces_in_high_vix_count',
        'bounces_in_low_vix_count',
        'high_vix_bounce_ratio',
        'channel_age_vs_vix_correlation',
        'vix_momentum_at_boundary',
        'vix_distance_from_mean',
        'vix_regime_alignment',
        'avg_bars_between_bounces_by_vix',
        'high_vix_bounce_frequency',
    ]
