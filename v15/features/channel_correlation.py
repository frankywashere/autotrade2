"""
v15/features/channel_correlation.py - TSLA-SPY Channel Cross-Correlation Features

Extracts cross-correlation features between TSLA and SPY channel features.
For every matching channel feature, creates spread/ratio/aligned metrics.

Feature Categories (~50 features per timeframe):
1. INDIVIDUAL CORRELATIONS (39): spread/ratio/aligned for 13 key features
2. AGGREGATE FEATURES (4): composite scores for overall channel similarity
3. ADDITIONAL DERIVED (7): specialized cross-asset channel metrics

All features return valid floats with safe defaults (no NaN, no Inf).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from .utils import safe_float, safe_divide


# =============================================================================
# Key Channel Features to Correlate
# =============================================================================

# These are the channel features that exist in both TSLA and SPY channels
# For each, we create: spread, ratio, and aligned variants (3 features each)
KEY_CHANNEL_FEATURES: List[str] = [
    'position_in_channel',
    'distance_to_upper_pct',
    'distance_to_lower_pct',
    'breakout_pressure_up',
    'breakout_pressure_down',
    'touch_velocity',
    'channel_slope_normalized',
    'channel_r_squared',
    'excursions_above_upper',
    'excursions_below_lower',
    'max_excursion_above_pct',
    'max_excursion_below_pct',
    'excursion_rate',
]

# Total individual correlation features: 13 features * 3 variants = 39
# Plus 4 aggregate + ~7 derived = ~50 features per TF


# =============================================================================
# Safe Calculation Helpers
# =============================================================================

def _safe_spread(tsla_val: float, spy_val: float) -> float:
    """
    Calculate spread (TSLA - SPY) with safety checks.

    Args:
        tsla_val: TSLA channel feature value
        spy_val: SPY channel feature value

    Returns:
        Spread value, default 0.0 if invalid
    """
    if not np.isfinite(tsla_val) or not np.isfinite(spy_val):
        return 0.0
    spread = tsla_val - spy_val
    return safe_float(spread, 0.0)


def _safe_ratio(tsla_val: float, spy_val: float, default: float = 1.0) -> float:
    """
    Calculate ratio (TSLA / SPY) with safe division.

    Args:
        tsla_val: TSLA channel feature value (numerator)
        spy_val: SPY channel feature value (denominator)
        default: Default value when division is invalid

    Returns:
        Ratio value, clamped to reasonable bounds [-10, 10]
    """
    if not np.isfinite(tsla_val) or not np.isfinite(spy_val):
        return default

    ratio = safe_divide(tsla_val, spy_val, default)
    # Clamp to reasonable bounds to avoid extreme values
    return safe_float(np.clip(ratio, -10.0, 10.0), default)


def _safe_aligned(tsla_val: float, spy_val: float, threshold: float = 0.5) -> float:
    """
    Check if both values are on the same side of a threshold or same direction.

    For position-type features (0-1 range): both above or both below 0.5
    For direction-type features: both positive or both negative

    Args:
        tsla_val: TSLA channel feature value
        spy_val: SPY channel feature value
        threshold: Threshold for determining alignment (default 0.5)

    Returns:
        1.0 if aligned, 0.0 if not aligned
    """
    if not np.isfinite(tsla_val) or not np.isfinite(spy_val):
        return 0.0

    # For features normalized to 0-1, check if both are on same side of 0.5
    tsla_above = tsla_val >= threshold
    spy_above = spy_val >= threshold

    return 1.0 if tsla_above == spy_above else 0.0


def _direction_aligned(tsla_val: float, spy_val: float) -> float:
    """
    Check if both values have the same sign/direction.

    Args:
        tsla_val: TSLA channel feature value
        spy_val: SPY channel feature value

    Returns:
        1.0 if same direction, 0.0 if opposite, 0.5 if one is near zero
    """
    if not np.isfinite(tsla_val) or not np.isfinite(spy_val):
        return 0.5

    # Near-zero threshold
    near_zero = 0.001

    tsla_pos = tsla_val > near_zero
    tsla_neg = tsla_val < -near_zero
    spy_pos = spy_val > near_zero
    spy_neg = spy_val < -near_zero

    # If either is near zero, partial alignment
    if (not tsla_pos and not tsla_neg) or (not spy_pos and not spy_neg):
        return 0.5

    # Both positive or both negative
    if (tsla_pos and spy_pos) or (tsla_neg and spy_neg):
        return 1.0

    return 0.0


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_channel_correlation_features(
    tsla_channel_features: Dict[str, float],
    spy_channel_features: Dict[str, float],
    tf: str
) -> Dict[str, float]:
    """
    Extract cross-correlation features between TSLA and SPY channel features.

    For each key channel feature that exists in both TSLA and SPY, creates:
    - {feature}_spread: TSLA value minus SPY value
    - {feature}_ratio: TSLA value / SPY value (safe division, default 1.0)
    - {feature}_aligned: 1 if both on same side of threshold, 0 otherwise

    Also creates aggregate features:
    - channel_correlation_score: Overall similarity measure
    - breakout_pressure_alignment: Both showing same breakout direction
    - excursion_divergence: One having excursions while other isn't
    - channel_regime_match: Both in same direction (bull/bear/sideways)

    Args:
        tsla_channel_features: Dict of TSLA channel features (base names, no prefix)
        spy_channel_features: Dict of SPY channel features (base names, no prefix)
        tf: Timeframe name for prefixing output features

    Returns:
        Dict[str, float] with ~50 features per timeframe, all guaranteed finite.
        Keys are prefixed with tf (e.g., 'daily_position_in_channel_spread')
    """
    features: Dict[str, float] = {}

    # Handle None or empty inputs
    if tsla_channel_features is None:
        tsla_channel_features = {}
    if spy_channel_features is None:
        spy_channel_features = {}

    # =========================================================================
    # 1. INDIVIDUAL CORRELATIONS (39 features: 13 key features x 3 variants)
    # =========================================================================

    # Position in channel - where price is within the channel (0=floor, 1=ceiling)
    tsla_pos = tsla_channel_features.get('position_in_channel', 0.5)
    spy_pos = spy_channel_features.get('position_in_channel', 0.5)
    features['position_in_channel_spread'] = _safe_spread(tsla_pos, spy_pos)
    features['position_in_channel_ratio'] = _safe_ratio(tsla_pos, spy_pos, 1.0)
    features['position_in_channel_aligned'] = _safe_aligned(tsla_pos, spy_pos, 0.5)

    # Distance to upper boundary (%)
    tsla_dist_upper = tsla_channel_features.get('distance_to_upper_pct', 0.0)
    spy_dist_upper = spy_channel_features.get('distance_to_upper_pct', 0.0)
    features['distance_to_upper_pct_spread'] = _safe_spread(tsla_dist_upper, spy_dist_upper)
    features['distance_to_upper_pct_ratio'] = _safe_ratio(tsla_dist_upper, spy_dist_upper, 1.0)
    features['distance_to_upper_pct_aligned'] = _direction_aligned(tsla_dist_upper, spy_dist_upper)

    # Distance to lower boundary (%)
    tsla_dist_lower = tsla_channel_features.get('distance_to_lower_pct', 0.0)
    spy_dist_lower = spy_channel_features.get('distance_to_lower_pct', 0.0)
    features['distance_to_lower_pct_spread'] = _safe_spread(tsla_dist_lower, spy_dist_lower)
    features['distance_to_lower_pct_ratio'] = _safe_ratio(tsla_dist_lower, spy_dist_lower, 1.0)
    features['distance_to_lower_pct_aligned'] = _direction_aligned(tsla_dist_lower, spy_dist_lower)

    # Breakout pressure up (proximity to upper boundary)
    tsla_press_up = tsla_channel_features.get('breakout_pressure_up', 0.0)
    spy_press_up = spy_channel_features.get('breakout_pressure_up', 0.0)
    features['breakout_pressure_up_spread'] = _safe_spread(tsla_press_up, spy_press_up)
    features['breakout_pressure_up_ratio'] = _safe_ratio(tsla_press_up, spy_press_up, 1.0)
    features['breakout_pressure_up_aligned'] = _safe_aligned(tsla_press_up, spy_press_up, 0.5)

    # Breakout pressure down (proximity to lower boundary)
    tsla_press_down = tsla_channel_features.get('breakout_pressure_down', 0.0)
    spy_press_down = spy_channel_features.get('breakout_pressure_down', 0.0)
    features['breakout_pressure_down_spread'] = _safe_spread(tsla_press_down, spy_press_down)
    features['breakout_pressure_down_ratio'] = _safe_ratio(tsla_press_down, spy_press_down, 1.0)
    features['breakout_pressure_down_aligned'] = _safe_aligned(tsla_press_down, spy_press_down, 0.5)

    # Touch velocity (bounces per bar)
    tsla_touch_vel = tsla_channel_features.get('touch_velocity', 0.0)
    spy_touch_vel = spy_channel_features.get('touch_velocity', 0.0)
    features['touch_velocity_spread'] = _safe_spread(tsla_touch_vel, spy_touch_vel)
    features['touch_velocity_ratio'] = _safe_ratio(tsla_touch_vel, spy_touch_vel, 1.0)
    features['touch_velocity_aligned'] = _safe_aligned(tsla_touch_vel, spy_touch_vel, 0.05)

    # Channel slope normalized (slope / price level)
    tsla_slope = tsla_channel_features.get('channel_slope_normalized', 0.0)
    spy_slope = spy_channel_features.get('channel_slope_normalized', 0.0)
    features['channel_slope_normalized_spread'] = _safe_spread(tsla_slope, spy_slope)
    features['channel_slope_normalized_ratio'] = _safe_ratio(tsla_slope, spy_slope, 1.0)
    features['channel_slope_normalized_aligned'] = _direction_aligned(tsla_slope, spy_slope)

    # Channel R-squared (quality of linear fit)
    tsla_rsq = tsla_channel_features.get('channel_r_squared', 0.0)
    spy_rsq = spy_channel_features.get('channel_r_squared', 0.0)
    features['channel_r_squared_spread'] = _safe_spread(tsla_rsq, spy_rsq)
    features['channel_r_squared_ratio'] = _safe_ratio(tsla_rsq, spy_rsq, 1.0)
    features['channel_r_squared_aligned'] = _safe_aligned(tsla_rsq, spy_rsq, 0.5)

    # Excursions above upper boundary (count)
    tsla_exc_above = tsla_channel_features.get('excursions_above_upper', 0.0)
    spy_exc_above = spy_channel_features.get('excursions_above_upper', 0.0)
    features['excursions_above_upper_spread'] = _safe_spread(tsla_exc_above, spy_exc_above)
    features['excursions_above_upper_ratio'] = _safe_ratio(tsla_exc_above, spy_exc_above, 1.0)
    features['excursions_above_upper_aligned'] = _safe_aligned(tsla_exc_above, spy_exc_above, 0.5)

    # Excursions below lower boundary (count)
    tsla_exc_below = tsla_channel_features.get('excursions_below_lower', 0.0)
    spy_exc_below = spy_channel_features.get('excursions_below_lower', 0.0)
    features['excursions_below_lower_spread'] = _safe_spread(tsla_exc_below, spy_exc_below)
    features['excursions_below_lower_ratio'] = _safe_ratio(tsla_exc_below, spy_exc_below, 1.0)
    features['excursions_below_lower_aligned'] = _safe_aligned(tsla_exc_below, spy_exc_below, 0.5)

    # Max excursion above upper (%)
    tsla_max_exc_above = tsla_channel_features.get('max_excursion_above_pct', 0.0)
    spy_max_exc_above = spy_channel_features.get('max_excursion_above_pct', 0.0)
    features['max_excursion_above_pct_spread'] = _safe_spread(tsla_max_exc_above, spy_max_exc_above)
    features['max_excursion_above_pct_ratio'] = _safe_ratio(tsla_max_exc_above, spy_max_exc_above, 1.0)
    features['max_excursion_above_pct_aligned'] = _safe_aligned(tsla_max_exc_above, spy_max_exc_above, 0.01)

    # Max excursion below lower (%)
    tsla_max_exc_below = tsla_channel_features.get('max_excursion_below_pct', 0.0)
    spy_max_exc_below = spy_channel_features.get('max_excursion_below_pct', 0.0)
    features['max_excursion_below_pct_spread'] = _safe_spread(tsla_max_exc_below, spy_max_exc_below)
    features['max_excursion_below_pct_ratio'] = _safe_ratio(tsla_max_exc_below, spy_max_exc_below, 1.0)
    features['max_excursion_below_pct_aligned'] = _safe_aligned(tsla_max_exc_below, spy_max_exc_below, 0.01)

    # Excursion rate (excursions per bar)
    tsla_exc_rate = tsla_channel_features.get('excursion_rate', 0.0)
    spy_exc_rate = spy_channel_features.get('excursion_rate', 0.0)
    features['excursion_rate_spread'] = _safe_spread(tsla_exc_rate, spy_exc_rate)
    features['excursion_rate_ratio'] = _safe_ratio(tsla_exc_rate, spy_exc_rate, 1.0)
    features['excursion_rate_aligned'] = _safe_aligned(tsla_exc_rate, spy_exc_rate, 0.05)

    # =========================================================================
    # 2. AGGREGATE FEATURES (4 features)
    # =========================================================================

    # Channel correlation score - overall similarity of TSLA and SPY channels
    # Computed as average of aligned features (1 = perfectly similar channels)
    aligned_features = [
        features['position_in_channel_aligned'],
        features['breakout_pressure_up_aligned'],
        features['breakout_pressure_down_aligned'],
        features['channel_slope_normalized_aligned'],
        features['channel_r_squared_aligned'],
    ]
    features['channel_correlation_score'] = safe_float(np.mean(aligned_features), 0.5)

    # Breakout pressure alignment - are both showing same breakout direction pressure
    # 1 = both pressing up, -1 = both pressing down, 0 = diverging
    tsla_pressure_bias = tsla_press_up - tsla_press_down
    spy_pressure_bias = spy_press_up - spy_press_down

    # Check if both biased in same direction
    if tsla_pressure_bias > 0.1 and spy_pressure_bias > 0.1:
        pressure_alignment = 1.0  # Both pressing up
    elif tsla_pressure_bias < -0.1 and spy_pressure_bias < -0.1:
        pressure_alignment = -1.0  # Both pressing down
    elif abs(tsla_pressure_bias) < 0.1 and abs(spy_pressure_bias) < 0.1:
        pressure_alignment = 0.0  # Both neutral
    else:
        # Diverging - one pressing up, other down or neutral
        pressure_alignment = -0.5
    features['breakout_pressure_alignment'] = safe_float(pressure_alignment, 0.0)

    # Excursion divergence - is one having excursions while other isn't
    # 1 = TSLA has more excursions, -1 = SPY has more, 0 = similar
    tsla_total_exc = tsla_exc_above + tsla_exc_below
    spy_total_exc = spy_exc_above + spy_exc_below

    exc_diff = tsla_total_exc - spy_total_exc
    if abs(exc_diff) < 0.5:
        excursion_div = 0.0  # Similar excursion patterns
    elif exc_diff > 0:
        excursion_div = min(exc_diff / 5.0, 1.0)  # TSLA more excursions
    else:
        excursion_div = max(exc_diff / 5.0, -1.0)  # SPY more excursions
    features['excursion_divergence'] = safe_float(excursion_div, 0.0)

    # Channel regime match - do both have same direction (bull/bear/sideways)
    # Based on slope direction: 1 = both same direction, 0 = different
    tsla_direction = tsla_channel_features.get('channel_direction', 1.0)  # 0=bear, 1=sideways, 2=bull
    spy_direction = spy_channel_features.get('channel_direction', 1.0)

    if tsla_direction == spy_direction:
        regime_match = 1.0  # Same regime
    elif abs(tsla_direction - spy_direction) == 1:
        regime_match = 0.5  # Adjacent regimes (e.g., bull and sideways)
    else:
        regime_match = 0.0  # Opposite regimes (bull vs bear)
    features['channel_regime_match'] = safe_float(regime_match, 0.5)

    # =========================================================================
    # 3. ADDITIONAL DERIVED FEATURES (7 features)
    # =========================================================================

    # Position spread extreme - how different are their channel positions
    # Normalized to 0-1 where 1 = completely opposite positions
    features['position_spread_extreme'] = safe_float(
        abs(features['position_in_channel_spread']), 0.0
    )

    # Breakout pressure divergence - difference in overall breakout pressure
    tsla_max_pressure = max(tsla_press_up, tsla_press_down)
    spy_max_pressure = max(spy_press_up, spy_press_down)
    features['breakout_pressure_divergence'] = safe_float(
        abs(tsla_max_pressure - spy_max_pressure), 0.0
    )

    # Channel quality agreement - are both channels well-defined or poorly-defined
    # 1 = both have similar quality, 0 = quality mismatch
    quality_diff = abs(tsla_rsq - spy_rsq)
    features['channel_quality_agreement'] = safe_float(1.0 - quality_diff, 0.5)

    # Slope magnitude comparison - which has steeper channel
    # Positive = TSLA steeper, negative = SPY steeper
    tsla_slope_mag = abs(tsla_slope)
    spy_slope_mag = abs(spy_slope)
    features['slope_magnitude_spread'] = _safe_spread(tsla_slope_mag, spy_slope_mag)

    # Touch activity comparison - which has more active bouncing
    features['touch_activity_spread'] = _safe_spread(tsla_touch_vel, spy_touch_vel)

    # Combined excursion intensity
    # Measures total excursion activity across both assets
    total_excursions = tsla_total_exc + spy_total_exc
    features['combined_excursion_intensity'] = safe_float(
        total_excursions / 10.0, 0.0  # Normalize (10 excursions = 1.0)
    )

    # Relative channel stability
    # Based on R-squared and touch regularity
    tsla_stability = tsla_channel_features.get('channel_stability', 0.0)
    spy_stability = spy_channel_features.get('channel_stability', 0.0)
    features['relative_stability_spread'] = _safe_spread(tsla_stability, spy_stability)

    # =========================================================================
    # Final Validation and Prefixing
    # =========================================================================

    # Ensure all values are finite
    for key in features:
        if not np.isfinite(features[key]):
            features[key] = 0.0

    # Add timeframe prefix
    prefixed_features = {f"{tf}_{k}": v for k, v in features.items()}

    return prefixed_features


# =============================================================================
# Default Features and Utility Functions
# =============================================================================

def _get_default_features() -> Dict[str, float]:
    """Return default feature values when inputs are invalid."""
    defaults = {}

    # Individual correlations for each key feature (3 variants each)
    for feature in KEY_CHANNEL_FEATURES:
        defaults[f'{feature}_spread'] = 0.0
        defaults[f'{feature}_ratio'] = 1.0
        defaults[f'{feature}_aligned'] = 0.0

    # Aggregate features
    defaults['channel_correlation_score'] = 0.5
    defaults['breakout_pressure_alignment'] = 0.0
    defaults['excursion_divergence'] = 0.0
    defaults['channel_regime_match'] = 0.5

    # Additional derived features
    defaults['position_spread_extreme'] = 0.0
    defaults['breakout_pressure_divergence'] = 0.0
    defaults['channel_quality_agreement'] = 0.5
    defaults['slope_magnitude_spread'] = 0.0
    defaults['touch_activity_spread'] = 0.0
    defaults['combined_excursion_intensity'] = 0.0
    defaults['relative_stability_spread'] = 0.0

    return defaults


def get_channel_correlation_feature_names() -> List[str]:
    """Get base channel correlation feature names (without TF prefix)."""
    return list(_get_default_features().keys())


def get_channel_correlation_feature_names_tf(tf: str) -> List[str]:
    """Get feature names with TF prefix."""
    base_names = get_channel_correlation_feature_names()
    return [f"{tf}_{name}" for name in base_names]


def get_channel_correlation_feature_count() -> int:
    """Get total number of channel correlation features per TF."""
    return len(_get_default_features())


def get_all_channel_correlation_feature_names() -> List[str]:
    """Get ALL channel correlation feature names across all TFs."""
    from v15.config import TIMEFRAMES

    all_names = []
    for tf in TIMEFRAMES:
        all_names.extend(get_channel_correlation_feature_names_tf(tf))
    return all_names


def get_total_channel_correlation_features() -> int:
    """
    Total channel correlation features across all TFs.

    Returns:
        count * 10 TFs (approximately 500 features total)
    """
    from v15.config import N_TIMEFRAMES
    return get_channel_correlation_feature_count() * N_TIMEFRAMES


# =============================================================================
# Batch Extraction for Multiple Windows
# =============================================================================

def extract_channel_correlation_features_multi_window(
    tsla_channel_features_by_window: Dict[int, Dict[str, float]],
    spy_channel_features_by_window: Dict[int, Dict[str, float]],
    tf: str,
    windows: List[int] = None
) -> Dict[str, float]:
    """
    Extract channel correlation features across multiple window sizes.

    This aggregates correlation features across different channel window sizes,
    useful when comparing channel behavior at different time scales.

    Args:
        tsla_channel_features_by_window: Dict mapping window size to TSLA features
        spy_channel_features_by_window: Dict mapping window size to SPY features
        tf: Timeframe name
        windows: List of window sizes to use (default: use all available)

    Returns:
        Dict of aggregated correlation features across all windows
    """
    if windows is None:
        windows = sorted(set(tsla_channel_features_by_window.keys()) &
                        set(spy_channel_features_by_window.keys()))

    if not windows:
        # Return defaults with prefix
        defaults = _get_default_features()
        return {f"{tf}_{k}": v for k, v in defaults.items()}

    # Collect features from all windows
    all_window_features = []
    for window in windows:
        tsla_feats = tsla_channel_features_by_window.get(window, {})
        spy_feats = spy_channel_features_by_window.get(window, {})

        # Extract correlation features (without tf prefix for aggregation)
        window_feats = extract_channel_correlation_features(tsla_feats, spy_feats, tf)
        # Remove tf prefix for aggregation
        window_feats = {k.replace(f"{tf}_", ""): v for k, v in window_feats.items()}
        all_window_features.append(window_feats)

    # Aggregate across windows (use mean for most features)
    aggregated = {}
    feature_names = get_channel_correlation_feature_names()

    for feat_name in feature_names:
        values = [wf.get(feat_name, 0.0) for wf in all_window_features]
        valid_values = [v for v in values if np.isfinite(v)]

        if valid_values:
            aggregated[feat_name] = safe_float(np.mean(valid_values), 0.0)
        else:
            aggregated[feat_name] = _get_default_features().get(feat_name, 0.0)

    # Add tf prefix
    return {f"{tf}_{k}": v for k, v in aggregated.items()}
