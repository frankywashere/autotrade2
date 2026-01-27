"""
Test that the actual feature extraction produces the expected number of features.

This test validates the end-to-end feature extraction pipeline:
1. Creates a minimal mock FullFeatures object
2. Calls features_to_tensor_dict() to convert to arrays
3. Verifies the vix array has 21 features
4. Calls concatenate_features_in_order() to get the final tensor
5. Verifies the final tensor has the expected number of features
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v7.features.full_features import (
    FullFeatures, TSLAChannelFeatures, SPYFeatures,
    features_to_tensor_dict
)
from v7.features.cross_asset import VIXFeatures, CrossAssetContainment, VIXChannelFeatures
from v7.features.history import ChannelHistoryFeatures
from v7.features.feature_ordering import (
    FEATURE_ORDER, get_expected_dimensions, concatenate_features_in_order,
    TOTAL_FEATURES, VIX_FEATURES, N_TIMEFRAMES, PER_TF_FEATURES
)
from v7.core.timeframe import TIMEFRAMES


def create_minimal_mock_features():
    """Create a minimal but complete FullFeatures object with all required fields."""

    # Create minimal TSLA features for all timeframes
    tsla_features = {}
    for tf in TIMEFRAMES:
        tsla_features[tf] = TSLAChannelFeatures(
            timeframe=tf,
            channel_valid=True,
            direction=1,  # sideways
            position=0.5,
            upper_dist=0.1,
            lower_dist=0.1,
            width_pct=1.0,
            slope_pct=0.0,
            r_squared=0.5,
            bounce_count=3,
            cycles=2,
            bars_since_bounce=10,
            last_touch=0,
            rsi=50.0,
            rsi_divergence=0,
            rsi_at_last_upper=60.0,
            rsi_at_last_lower=40.0,
            channel_quality=0.7,
            rsi_confidence=0.8,
            containments={},
            exit_tracking=None,
            break_trigger=None,
        )

    # Create minimal SPY features for all timeframes
    spy_features = {}
    for tf in TIMEFRAMES:
        spy_features[tf] = SPYFeatures(
            timeframe=tf,
            channel_valid=True,
            direction=1,  # sideways
            position=0.5,
            upper_dist=0.1,
            lower_dist=0.1,
            width_pct=1.0,
            slope_pct=0.0,
            r_squared=0.5,
            bounce_count=2,
            cycles=1,
            rsi=50.0,
        )

    # Create minimal cross-asset containment for all timeframes
    cross_containment = {}
    for tf in TIMEFRAMES:
        cross_containment[tf] = CrossAssetContainment(
            timeframe=tf,
            spy_channel_valid=True,
            spy_direction=1,
            spy_position=0.5,
            tsla_in_spy_upper=False,
            tsla_in_spy_lower=False,
            tsla_dist_to_spy_upper=0.2,
            tsla_dist_to_spy_lower=0.2,
            alignment=0,
            rsi_correlation=0.5,
            rsi_correlation_trend=0,
        )

    # Create VIX features with basic and channel interaction components
    vix = VIXFeatures(
        level=20.0,
        level_normalized=0.5,
        trend_5d=0.0,
        trend_20d=0.0,
        percentile_252d=0.5,
        regime=1,  # normal
    )

    # Create VIX-channel interaction features (optional but included)
    vix_channel = VIXChannelFeatures(
        vix_at_channel_start=20.0,
        vix_at_last_bounce=20.0,
        vix_change_during_channel=0.0,
        vix_regime_at_start=1,
        vix_regime_at_current=1,
        avg_vix_at_upper_bounces=20.0,
        avg_vix_at_lower_bounces=20.0,
        vix_upper_minus_lower=0.0,
        pct_bounces_high_vix=0.0,
        vix_trend_during_channel=0,
        vix_volatility_during_channel=0.0,
        vix_regime_changes_count=0,
        bounce_hold_rate_low_vix=0.5,
        bounce_hold_rate_high_vix=0.5,
        vix_bounce_quality_diff=0.0,
    )

    # Create history features for both TSLA and SPY
    tsla_history = ChannelHistoryFeatures(
        last_n_directions=[1] * 5,
        last_n_durations=[50.0] * 5,
        last_n_break_dirs=[1] * 5,
        avg_duration=50.0,
        direction_streak=0,
        bear_count_last_5=0,
        bull_count_last_5=5,
        sideways_count_last_5=0,
        avg_rsi_at_upper_bounce=60.0,
        avg_rsi_at_lower_bounce=40.0,
        rsi_at_last_break=50.0,
        break_up_after_bear_pct=0.5,
        break_down_after_bull_pct=0.5,
    )

    spy_history = ChannelHistoryFeatures(
        last_n_directions=[1] * 5,
        last_n_durations=[50.0] * 5,
        last_n_break_dirs=[1] * 5,
        avg_duration=50.0,
        direction_streak=0,
        bear_count_last_5=0,
        bull_count_last_5=5,
        sideways_count_last_5=0,
        avg_rsi_at_upper_bounce=60.0,
        avg_rsi_at_lower_bounce=40.0,
        rsi_at_last_break=50.0,
        break_up_after_bear_pct=0.5,
        break_down_after_bull_pct=0.5,
    )

    # Create multi-window channel scores (8 windows x 5 metrics)
    tsla_window_scores = np.array([
        [3, 0.5, 0.7, 0.8, 1.0],  # window 10
        [3, 0.5, 0.7, 0.8, 1.0],  # window 20
        [3, 0.5, 0.7, 0.8, 1.0],  # window 30
        [3, 0.5, 0.7, 0.8, 1.0],  # window 40
        [3, 0.5, 0.7, 0.8, 1.0],  # window 50
        [3, 0.5, 0.7, 0.8, 1.0],  # window 60
        [3, 0.5, 0.7, 0.8, 1.0],  # window 70
        [3, 0.5, 0.7, 0.8, 1.0],  # window 80
    ], dtype=np.float32)

    # Create the complete FullFeatures object
    features = FullFeatures(
        timestamp=pd.Timestamp.now(),
        tsla=tsla_features,
        spy=spy_features,
        cross_containment=cross_containment,
        vix=vix,
        vix_channel=vix_channel,
        tsla_history=tsla_history,
        spy_history=spy_history,
        tsla_spy_direction_match=True,
        both_near_upper=False,
        both_near_lower=False,
        events=None,  # Events are optional
        tsla_window_scores=tsla_window_scores,
    )

    return features


def test_feature_extraction():
    """Test the complete feature extraction pipeline."""

    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION TEST")
    print("=" * 80)

    # Step 1: Create minimal mock features
    print("\n[1] Creating minimal mock FullFeatures object...")
    features = create_minimal_mock_features()
    print("    ✓ FullFeatures created with all required fields")

    # Step 2: Convert to tensor dict
    print("\n[2] Calling features_to_tensor_dict()...")
    arrays = features_to_tensor_dict(features)
    print(f"    ✓ Converted to dict with {len(arrays)} keys")

    # Step 3: Verify VIX array has 21 features
    print("\n[3] Verifying VIX array has 21 features...")
    assert 'vix' in arrays, "'vix' key not found in features dict"

    vix_array = arrays['vix']
    vix_shape = vix_array.shape[0]

    print(f"    arrays['vix'].shape = {vix_array.shape}")
    print(f"    VIX features count: {vix_shape}")

    assert vix_shape == 21, f"Expected {VIX_FEATURES} features, got {vix_shape}"
    print(f"    ✓ VIX array has exactly {VIX_FEATURES} features (6 basic + 15 channel)")

    # Step 4: Print all feature dimensions
    print("\n[4] Feature dimensions from arrays dict:")
    expected_dims = get_expected_dimensions()
    total_from_dict = 0

    for key in FEATURE_ORDER:
        if key in arrays:
            arr = arrays[key]
            actual_dim = arr.shape[0]
            expected_dim = expected_dims[key]
            status = "✓" if actual_dim == expected_dim else "✗"
            print(f"    {status} {key:20s}: {actual_dim:3d} features (expected {expected_dim:3d})")
            assert actual_dim == expected_dim, f"Dimension mismatch for {key}: expected {expected_dim}, got {actual_dim}"
            total_from_dict += actual_dim
        else:
            raise KeyError(f"Missing required key: {key}")

    print(f"\n    Total from dict: {total_from_dict}")

    # Step 5: Call concatenate_features_in_order
    print("\n[5] Calling concatenate_features_in_order()...")
    final_tensor = concatenate_features_in_order(arrays)
    print(f"    ✓ Concatenated successfully")

    # Step 6: Verify final tensor shape
    print("\n[6] Verifying final tensor shape...")
    final_shape = final_tensor.shape[0]
    print(f"    Final tensor shape: {final_tensor.shape}")
    print(f"    Number of features: {final_shape}")
    print(f"    Expected: {TOTAL_FEATURES}")

    # Calculate what we expect based on formula
    expected_total = PER_TF_FEATURES * N_TIMEFRAMES + 160  # 160 = 21+25+25+3+46+40
    print(f"    Calculated: {PER_TF_FEATURES} * {N_TIMEFRAMES} + 160 = {expected_total}")

    # Check if we got 776
    assert final_shape == 776, f"Expected 776 features, got {final_shape}"
    print(f"    ✓ Final tensor has 776 features (as required)")


if __name__ == '__main__':
    try:
        test_feature_extraction()
        print("\n" + "=" * 80)
        print("SUCCESS: Feature extraction produces the expected number of features!")
        print("=" * 80 + "\n")
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"ERROR: {e}")
        print("=" * 80 + "\n")
        import traceback
        traceback.print_exc()
        exit(1)
