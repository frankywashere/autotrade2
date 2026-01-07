"""
Explicit Feature Ordering for Model Input

This module defines the canonical order of features for model input.
Both feature generation (features_to_tensor_dict) and model consumption (trainer.py)
must use this ordering to ensure consistency.

CRITICAL: The ordering is TIMEFRAME-GROUPED, not alphabetical!
- Each timeframe's features are contiguous: [tsla_tf, spy_tf, cross_tf]
- This allows the model's TF branches to process coherent feature blocks
- Shared features come last: [vix, tsla_history, spy_history, alignment, events, window_scores]

Feature Dimensions:
- TSLA per TF: 35 features (18 base + 10 exit_tracking + 2 break_trigger + 5 return_tracking)
    Return tracking features (5):
      - return_rate (float)
      - channel_resilience_score (float)
      - avg_duration_after_return (float)
      - max_duration_after_return (int)
      - returns_leading_to_new_channel (int)
- SPY per TF: 11 features (channel metrics + RSI)
- Cross per TF: 10 features (TSLA-in-SPY containment + RSI correlation)
    RSI correlation features (2):
      - rsi_correlation (float)
      - rsi_correlation_trend (int)
- VIX: 6 features
- TSLA History: 25 features (5+5+5 lists + 10 scalars)
- SPY History: 25 features (same structure)
- Alignment: 3 features
- Events: 46 features (optional, zeros if not provided)
- Window Scores: 40 features (8 windows x 5 metrics per window)

Total: (35+11+10) × 11 + (6+25+25+3+46+40) = 56×11 + 145 = 616 + 145 = 761
"""

from typing import List, Dict
import numpy as np

# Import TIMEFRAMES from core
from v7.core.timeframe import TIMEFRAMES


# =============================================================================
# Feature Dimension Constants
# =============================================================================

# Per-timeframe feature dimensions
TSLA_PER_TF = 35    # 18 base + 10 exit_tracking + 2 break_trigger + 5 return_tracking
SPY_PER_TF = 11     # channel_valid, direction, position, upper/lower_dist, width, slope, r2, bounces, cycles, rsi
CROSS_PER_TF = 10   # spy_valid, spy_dir, spy_pos, tsla_in_upper/lower, dist_to_upper/lower, alignment + rsi_correlation (2)

# TSLA per-TF feature breakdown (35 total):
#   Base channel features (18):
#     channel_valid, direction, position, upper_dist, lower_dist, width_pct, slope_pct, r2,
#     bounces, cycles, touch_density, alternation, last_touch_dist, bar_depth, rsi, ...
#   Exit tracking features (10):
#     exit_count, avg_bars_outside, max_bars_outside, exit_frequency, exits_accelerating,
#     exits_up_count, exits_down_count, avg_return_speed, return_speed_slowing, bounces_after_last_return
#   Break trigger features (2):
#     break_trigger_up, break_trigger_down
#   Return tracking features (5):
#     return_rate, channel_resilience_score, avg_duration_after_return,
#     max_duration_after_return, returns_leading_to_new_channel

# Cross-asset per-TF feature breakdown (10 total):
#   TSLA-in-SPY containment (8):
#     spy_valid, spy_dir, spy_pos, tsla_in_upper, tsla_in_lower, dist_to_upper, dist_to_lower, alignment
#   RSI correlation features (2):
#     rsi_correlation, rsi_correlation_trend

# Shared feature dimensions
VIX_FEATURES = 6
TSLA_HISTORY_FEATURES = 25  # 5+5+5 lists + 10 scalars
SPY_HISTORY_FEATURES = 25   # Same structure as TSLA
ALIGNMENT_FEATURES = 3
EVENT_FEATURES = 46
WINDOW_SCORE_FEATURES = 40  # 8 windows x 5 metrics (bounce_count, r_squared, quality, alternation_ratio, width)

# Derived constants
PER_TF_FEATURES = TSLA_PER_TF + SPY_PER_TF + CROSS_PER_TF  # 35 + 11 + 10 = 56
SHARED_FEATURES = VIX_FEATURES + TSLA_HISTORY_FEATURES + SPY_HISTORY_FEATURES + ALIGNMENT_FEATURES + EVENT_FEATURES + WINDOW_SCORE_FEATURES  # 145
N_TIMEFRAMES = len(TIMEFRAMES)  # 11
TOTAL_FEATURES = PER_TF_FEATURES * N_TIMEFRAMES + SHARED_FEATURES  # 56*11 + 145 = 761


# =============================================================================
# Explicit Feature Key Ordering
# =============================================================================

def build_feature_order() -> List[str]:
    """
    Build the canonical feature key ordering.

    The ordering is TIMEFRAME-GROUPED:
    - For each timeframe: [tsla_{tf}, spy_{tf}, cross_{tf}]
    - Then shared: [vix, tsla_history, spy_history, alignment, events]

    This produces contiguous 56-feature blocks per timeframe that the model's
    TF branches can slice directly.

    Returns:
        List of feature keys in canonical order
    """
    order = []

    # Per-timeframe features: grouped by timeframe (TF0, TF1, ..., TF10)
    # Each TF block has: tsla (35) + spy (11) + cross (10) = 56 features
    # TSLA includes: 18 base + 10 exit_tracking + 2 break_trigger + 5 return_tracking
    # Cross includes: 8 containment + 2 RSI correlation
    for tf in TIMEFRAMES:
        order.append(f'tsla_{tf}')
        order.append(f'spy_{tf}')
        order.append(f'cross_{tf}')

    # Shared features: same for all timeframes
    order.extend([
        'vix',            # 6 features
        'tsla_history',   # 25 features
        'spy_history',    # 25 features
        'alignment',      # 3 features
        'events',         # 46 features (zeros if not provided)
        'window_scores',  # 40 features (8 windows x 5 metrics)
    ])

    return order


# The canonical feature ordering - use this everywhere
FEATURE_ORDER: List[str] = build_feature_order()

# Quick lookup for validation
FEATURE_ORDER_SET: set = set(FEATURE_ORDER)

# Required features (events is optional but we always include zeros)
REQUIRED_FEATURES: set = FEATURE_ORDER_SET


# =============================================================================
# Dimension Lookup
# =============================================================================

def get_feature_dim(key: str) -> int:
    """
    Get the expected dimension for a feature key.

    Args:
        key: Feature key (e.g., 'tsla_5min', 'vix', 'alignment')

    Returns:
        Expected number of features for this key

    Raises:
        ValueError: If key is unknown
    """
    if key.startswith('tsla_') and key != 'tsla_history':
        return TSLA_PER_TF
    elif key.startswith('spy_') and key != 'spy_history':
        return SPY_PER_TF
    elif key.startswith('cross_'):
        return CROSS_PER_TF
    elif key == 'vix':
        return VIX_FEATURES
    elif key == 'tsla_history':
        return TSLA_HISTORY_FEATURES
    elif key == 'spy_history':
        return SPY_HISTORY_FEATURES
    elif key == 'alignment':
        return ALIGNMENT_FEATURES
    elif key == 'events':
        return EVENT_FEATURES
    elif key == 'window_scores':
        return WINDOW_SCORE_FEATURES
    else:
        raise ValueError(f"Unknown feature key: {key}")


def get_expected_dimensions() -> Dict[str, int]:
    """
    Get expected dimensions for all features.

    Returns:
        Dict mapping feature keys to expected dimensions
    """
    return {key: get_feature_dim(key) for key in FEATURE_ORDER}


# =============================================================================
# Validation Functions
# =============================================================================

def validate_feature_dict(features: Dict[str, np.ndarray], raise_on_error: bool = True) -> List[str]:
    """
    Validate a feature dictionary against expected ordering and dimensions.

    Args:
        features: Dict of feature arrays from features_to_tensor_dict()
        raise_on_error: If True, raise ValueError on mismatch; else return error list

    Returns:
        List of error messages (empty if valid)

    Raises:
        ValueError: If raise_on_error=True and validation fails
    """
    errors = []

    # Check for missing keys
    missing = REQUIRED_FEATURES - set(features.keys())
    if missing:
        errors.append(f"Missing required features: {sorted(missing)}")

    # Check for unexpected keys
    unexpected = set(features.keys()) - FEATURE_ORDER_SET
    if unexpected:
        errors.append(f"Unexpected features: {sorted(unexpected)}")

    # Check dimensions
    expected_dims = get_expected_dimensions()
    for key, arr in features.items():
        if key in expected_dims:
            expected = expected_dims[key]
            actual = arr.shape[-1] if arr.ndim > 0 else 1
            if actual != expected:
                errors.append(f"Dimension mismatch for '{key}': expected {expected}, got {actual}")

    if errors and raise_on_error:
        raise ValueError("Feature validation failed:\n  " + "\n  ".join(errors))

    return errors


def get_tf_index_range(tf_idx: int) -> tuple:
    """
    Get the start and end indices for a timeframe's features in the concatenated tensor.

    This replaces FeatureConfig.get_tf_slice() with correct indexing.

    Args:
        tf_idx: Timeframe index (0-10, corresponding to TIMEFRAMES order)

    Returns:
        Tuple of (start_idx, end_idx) for slicing
    """
    start = tf_idx * PER_TF_FEATURES
    end = start + PER_TF_FEATURES
    return start, end


def get_shared_index_range() -> tuple:
    """
    Get the start and end indices for shared features in the concatenated tensor.

    This replaces FeatureConfig.get_shared_slice() with correct indexing.

    Returns:
        Tuple of (start_idx, end_idx) for slicing
    """
    start = N_TIMEFRAMES * PER_TF_FEATURES  # 11 * 56 = 616
    end = start + SHARED_FEATURES           # 616 + 145 = 761
    return start, end


# =============================================================================
# Feature Concatenation
# =============================================================================

def concatenate_features_in_order(features_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate feature dict into single array using canonical ordering.

    Uses FEATURE_ORDER to ensure consistent ordering across all code paths.
    This is the canonical way to convert a features dict to a flat tensor.

    Args:
        features_dict: Output from features_to_tensor_dict(), mapping feature
                       keys (e.g., 'tsla_5min', 'vix') to numpy arrays

    Returns:
        Concatenated numpy array of shape (TOTAL_FEATURES,) = (761,)

    Raises:
        KeyError: If a required feature key is missing from features_dict

    Example:
        features = FullFeatures(...)
        features_dict = features_to_tensor_dict(features)
        # features_dict = {'tsla_5min': array([...]), 'spy_5min': array([...]), ...}

        features_array = concatenate_features_in_order(features_dict)
        # features_array.shape = (761,)
    """
    arrays = []
    for key in FEATURE_ORDER:
        if key not in features_dict:
            raise KeyError(f"Missing required feature key: '{key}'")
        arrays.append(features_dict[key])

    result = np.concatenate(arrays)

    # Validate output shape
    if result.shape[0] != TOTAL_FEATURES:
        raise ValueError(
            f"Concatenated features have wrong shape: expected ({TOTAL_FEATURES},), "
            f"got ({result.shape[0]},)"
        )

    return result


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == '__main__':
    print("Feature Ordering Configuration")
    print("=" * 60)

    print(f"\nTimeframes ({N_TIMEFRAMES}): {TIMEFRAMES}")

    print(f"\nPer-TF dimensions:")
    print(f"  TSLA: {TSLA_PER_TF}")
    print(f"  SPY:  {SPY_PER_TF}")
    print(f"  Cross: {CROSS_PER_TF}")
    print(f"  Total per TF: {PER_TF_FEATURES}")

    print(f"\nShared dimensions:")
    print(f"  VIX: {VIX_FEATURES}")
    print(f"  TSLA History: {TSLA_HISTORY_FEATURES}")
    print(f"  SPY History: {SPY_HISTORY_FEATURES}")
    print(f"  Alignment: {ALIGNMENT_FEATURES}")
    print(f"  Events: {EVENT_FEATURES}")
    print(f"  Window Scores: {WINDOW_SCORE_FEATURES}")
    print(f"  Total shared: {SHARED_FEATURES}")

    print(f"\nTotal features: {PER_TF_FEATURES} × {N_TIMEFRAMES} + {SHARED_FEATURES} = {TOTAL_FEATURES}")

    print(f"\nFeature order ({len(FEATURE_ORDER)} keys):")
    for i, key in enumerate(FEATURE_ORDER):
        dim = get_feature_dim(key)
        print(f"  [{i:2d}] {key}: {dim} dims")

    print(f"\nIndex ranges:")
    for i, tf in enumerate(TIMEFRAMES):
        start, end = get_tf_index_range(i)
        print(f"  TF{i:2d} ({tf:>7s}): indices {start:3d}-{end-1:3d} ({end-start} features)")

    shared_start, shared_end = get_shared_index_range()
    print(f"  Shared:         indices {shared_start:3d}-{shared_end-1:3d} ({shared_end-shared_start} features)")

    print(f"\n  Note: TSLA per-TF includes:")
    print(f"    - 18 base channel features")
    print(f"    - 10 exit tracking features")
    print(f"    - 2 break trigger features")
    print(f"    - 5 return tracking features (return_rate, channel_resilience_score,")
    print(f"        avg_duration_after_return, max_duration_after_return, returns_leading_to_new_channel)")
    print(f"\n  Note: Cross per-TF includes:")
    print(f"    - 8 TSLA-in-SPY containment features")
    print(f"    - 2 RSI correlation features (rsi_correlation, rsi_correlation_trend)")

    print("\n" + "=" * 60)
    print("Feature ordering module loaded successfully.")
