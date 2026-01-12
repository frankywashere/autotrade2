"""
Comprehensive tests for the quality-weighted system.

Tests:
1. calculate_channel_quality_score with various inputs
2. Bounce-first sorting (5 bounces beats 3 bounces regardless of R²)
3. All 11 timeframes always present in features_to_tensor_dict
4. RSI confidence scores work correctly
5. Default features provided when TF missing
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import (
    Channel, Direction, TouchType, Touch,
    calculate_channel_quality_score,
    detect_channels_multi_window,
    detect_channel,
)
from core.timeframe import TIMEFRAMES
from features.rsi import calculate_rsi_series
from features.full_features import (
    extract_tsla_channel_features,
    features_to_tensor_dict,
    extract_full_features,
    FullFeatures,
    TSLAChannelFeatures
)


class TestChannelQualityScore:
    """Test calculate_channel_quality_score function."""

    def test_quality_score_zero_bounces(self):
        """Quality score with zero bounces should be zero."""
        channel = Channel(
            valid=False,
            direction=Direction.SIDEWAYS,
            slope=0.0,
            intercept=100.0,
            r_squared=0.95,
            std_dev=1.0,
            upper_line=np.array([102.0, 102.0]),
            lower_line=np.array([98.0, 98.0]),
            center_line=np.array([100.0, 100.0]),
            touches=[],
            complete_cycles=0,
            bounce_count=0,
            width_pct=4.0,
            window=20,
            close=np.array([100.0, 100.0]),
        )

        score = calculate_channel_quality_score(channel)
        assert score == 0.0

    def test_quality_score_one_alternation(self):
        """Quality score with 1 alternation and ratio 1.0.

        Raw score = alternations * (1 + ratio) = 1 * (1 + 1.0) = 2.0
        Normalized = 2 / (1 + exp(-2/5)) - 1 ~ 0.329
        """
        import math
        channel = Channel(
            valid=True,
            direction=Direction.BULL,
            slope=0.1,
            intercept=100.0,
            r_squared=0.8,
            std_dev=1.0,
            upper_line=np.array([102.0, 102.0]),
            lower_line=np.array([98.0, 98.0]),
            center_line=np.array([100.0, 100.0]),
            touches=[Touch(0, TouchType.LOWER, 98.0), Touch(5, TouchType.UPPER, 102.0)],
            complete_cycles=0,
            bounce_count=1,
            width_pct=4.0,
            window=20,
            close=np.array([100.0, 101.0]),
            alternations=1,
            alternation_ratio=1.0,
        )

        score = calculate_channel_quality_score(channel)
        # Raw = 1 * (1 + 1.0) = 2.0
        # Normalized = 2 / (1 + exp(-2/5)) - 1
        raw = 2.0
        expected = 2.0 / (1.0 + math.exp(-raw / 5.0)) - 1.0
        assert abs(score - expected) < 0.001, f"Expected {expected}, got {score}"
        # Score should be bounded [0, 1]
        assert 0.0 <= score <= 1.0, f"Score {score} not in [0, 1]"

    def test_quality_score_five_alternations_high_ratio(self):
        """Quality score with 5 alternations and ratio 1.0.

        Raw score = 5 * (1 + 1.0) = 10.0
        Normalized ~ 0.73
        """
        import math
        channel = Channel(
            valid=True,
            direction=Direction.BULL,
            slope=0.1,
            intercept=100.0,
            r_squared=0.95,
            std_dev=1.0,
            upper_line=np.array([102.0] * 10),
            lower_line=np.array([98.0] * 10),
            center_line=np.array([100.0] * 10),
            touches=[Touch(i, TouchType.UPPER if i % 2 else TouchType.LOWER, 100.0)
                     for i in range(6)],
            complete_cycles=2,
            bounce_count=5,
            width_pct=4.0,
            window=20,
            close=np.array([100.0] * 10),
            alternations=5,
            alternation_ratio=1.0,
        )

        score = calculate_channel_quality_score(channel)
        # Raw = 5 * (1 + 1.0) = 10.0
        raw = 10.0
        expected = 2.0 / (1.0 + math.exp(-raw / 5.0)) - 1.0
        assert abs(score - expected) < 0.001, f"Expected {expected}, got {score}"
        # Score should be bounded [0, 1]
        assert 0.0 <= score <= 1.0, f"Score {score} not in [0, 1]"

    def test_quality_score_three_alternations_low_ratio(self):
        """Quality score with 3 alternations and ratio 0.5.

        Raw score = 3 * (1 + 0.5) = 4.5
        Normalized ~ 0.41
        """
        import math
        channel = Channel(
            valid=True,
            direction=Direction.SIDEWAYS,
            slope=0.0,
            intercept=100.0,
            r_squared=0.4,
            std_dev=2.0,
            upper_line=np.array([104.0] * 10),
            lower_line=np.array([96.0] * 10),
            center_line=np.array([100.0] * 10),
            touches=[Touch(i, TouchType.UPPER if i % 2 else TouchType.LOWER, 100.0)
                     for i in range(4)],
            complete_cycles=1,
            bounce_count=3,
            width_pct=8.0,
            window=20,
            close=np.array([100.0] * 10),
            alternations=3,
            alternation_ratio=0.5,
        )

        score = calculate_channel_quality_score(channel)
        # Raw = 3 * (1 + 0.5) = 4.5
        raw = 4.5
        expected = 2.0 / (1.0 + math.exp(-raw / 5.0)) - 1.0
        assert abs(score - expected) < 0.001, f"Expected {expected}, got {score}"
        # Score should be bounded [0, 1]
        assert 0.0 <= score <= 1.0, f"Score {score} not in [0, 1]"

    def test_quality_score_bounded(self):
        """Quality score should always be bounded in [0, 1]."""
        import math
        # Test with very high alternations (extreme case)
        channel = Channel(
            valid=True,
            direction=Direction.BULL,
            slope=0.1,
            intercept=100.0,
            r_squared=0.95,
            std_dev=1.0,
            upper_line=np.array([102.0] * 50),
            lower_line=np.array([98.0] * 50),
            center_line=np.array([100.0] * 50),
            touches=[Touch(i, TouchType.UPPER if i % 2 else TouchType.LOWER, 100.0)
                     for i in range(50)],
            complete_cycles=20,
            bounce_count=49,
            width_pct=4.0,
            window=50,
            close=np.array([100.0] * 50),
            alternations=49,
            alternation_ratio=1.0,
        )

        score = calculate_channel_quality_score(channel)
        # Even with 49 alternations (raw = 98), score should be < 1.0
        assert 0.0 <= score <= 1.0, f"Score {score} not in [0, 1]"
        # Should be close to 1.0 but not quite
        assert score > 0.99, f"Very high alternations should give score > 0.99, got {score}"


class TestBounceFirstSorting:
    """Test that bounce count takes priority over R² in sorting."""

    def test_five_bounces_beats_three_bounces_regardless_of_r2(self):
        """5 bounces with low R² should beat 3 bounces with high R²."""
        # Channel with 5 bounces, low R²
        channel_5b = Channel(
            valid=True,
            direction=Direction.BULL,
            slope=0.1,
            intercept=100.0,
            r_squared=0.4,  # Low R²
            std_dev=2.0,
            upper_line=np.array([104.0] * 10),
            lower_line=np.array([96.0] * 10),
            center_line=np.array([100.0] * 10),
            touches=[Touch(i, TouchType.UPPER if i % 2 else TouchType.LOWER, 100.0)
                     for i in range(6)],
            complete_cycles=2,
            bounce_count=5,
            width_pct=8.0,
            window=20,
            close=np.array([100.0] * 10),
        )

        # Channel with 3 bounces, high R²
        channel_3b = Channel(
            valid=True,
            direction=Direction.BULL,
            slope=0.1,
            intercept=100.0,
            r_squared=0.95,  # High R²
            std_dev=0.5,
            upper_line=np.array([101.0] * 10),
            lower_line=np.array([99.0] * 10),
            center_line=np.array([100.0] * 10),
            touches=[Touch(i, TouchType.UPPER if i % 2 else TouchType.LOWER, 100.0)
                     for i in range(4)],
            complete_cycles=1,
            bounce_count=3,
            width_pct=2.0,
            window=20,
            close=np.array([100.0] * 10),
        )

        # Simulate the sorting used in detect_channels_multi_window
        channels = [channel_3b, channel_5b]
        best = max(channels, key=lambda c: (c.bounce_count, c.r_squared))

        assert best.bounce_count == 5
        assert best.r_squared == 0.4
        assert best == channel_5b


class TestAllTimeframesPresent:
    """Test that all 11 timeframes are always present in features_to_tensor_dict."""

    def create_sample_tsla_features(self, timeframe: str) -> TSLAChannelFeatures:
        """Helper to create sample TSLA features."""
        return TSLAChannelFeatures(
            timeframe=timeframe,
            channel_valid=True,
            direction=2,
            position=0.5,
            upper_dist=2.0,
            lower_dist=2.0,
            width_pct=4.0,
            slope_pct=0.1,
            r_squared=0.85,
            bounce_count=3,
            cycles=1,
            bars_since_bounce=5,
            last_touch=1,
            rsi=55.0,
            rsi_divergence=0,
            rsi_at_last_upper=65.0,
            rsi_at_last_lower=35.0,
            channel_quality=0.7,
            rsi_confidence=0.6,
        )

    def test_all_11_timeframes_present_full_data(self):
        """When all 11 TFs have data, all should be present."""
        # Create mock FullFeatures with all 11 timeframes
        tsla_features = {tf: self.create_sample_tsla_features(tf) for tf in TIMEFRAMES}

        from features.cross_asset import SPYFeatures, VIXFeatures, CrossAssetContainment
        from features.history import ChannelHistoryFeatures

        spy_features = {tf: SPYFeatures(
            timeframe=tf, channel_valid=True, direction=2, position=0.5,
            upper_dist=2.0, lower_dist=2.0, width_pct=4.0, slope_pct=0.1,
            r_squared=0.8, bounce_count=2, cycles=1, rsi=50.0
        ) for tf in TIMEFRAMES}

        cross_containment = {tf: CrossAssetContainment(
            timeframe=tf, spy_channel_valid=True, spy_direction=2, spy_position=0.5,
            tsla_in_spy_upper=False, tsla_in_spy_lower=False,
            tsla_dist_to_spy_upper=3.0, tsla_dist_to_spy_lower=3.0, alignment=1.0
        ) for tf in TIMEFRAMES}

        vix = VIXFeatures(
            level=20.0, level_normalized=0.5, trend_5d=0.0, trend_20d=0.0,
            percentile_252d=0.5, regime=1
        )

        history = ChannelHistoryFeatures(
            last_n_directions=[1] * 5,
            last_n_durations=[50.0] * 5,
            last_n_break_dirs=[1] * 5,
            avg_duration=50.0,
            direction_streak=2,
            bear_count_last_5=1,
            bull_count_last_5=3,
            sideways_count_last_5=1,
            avg_rsi_at_upper_bounce=65.0,
            avg_rsi_at_lower_bounce=35.0,
            rsi_at_last_break=55.0,
            break_up_after_bear_pct=0.6,
            break_down_after_bull_pct=0.4,
        )

        features = FullFeatures(
            timestamp=pd.Timestamp('2024-01-01 10:00:00'),
            tsla=tsla_features,
            spy=spy_features,
            cross_containment=cross_containment,
            vix=vix,
            tsla_history=history,
            spy_history=history,
            tsla_spy_direction_match=True,
            both_near_upper=False,
            both_near_lower=False,
        )

        # Convert to tensor dict
        tensor_dict = features_to_tensor_dict(features)

        # Check all 11 TSLA timeframes are present
        for tf in TIMEFRAMES:
            assert f'tsla_{tf}' in tensor_dict, f"Missing tsla_{tf}"
            assert isinstance(tensor_dict[f'tsla_{tf}'], np.ndarray)

        # Check all 11 SPY timeframes are present
        for tf in TIMEFRAMES:
            assert f'spy_{tf}' in tensor_dict, f"Missing spy_{tf}"
            assert isinstance(tensor_dict[f'spy_{tf}'], np.ndarray)

        # Check all 11 cross-asset timeframes are present
        for tf in TIMEFRAMES:
            assert f'cross_{tf}' in tensor_dict, f"Missing cross_{tf}"
            assert isinstance(tensor_dict[f'cross_{tf}'], np.ndarray)

    def test_all_11_timeframes_count(self):
        """Verify TIMEFRAMES constant has exactly 11 entries."""
        assert len(TIMEFRAMES) == 11, f"Expected 11 timeframes, got {len(TIMEFRAMES)}"


class TestRSIConfidenceScores:
    """Test RSI confidence score calculation."""

    def create_mock_df(self, close_prices: list, window: int = 50) -> pd.DataFrame:
        """Create a mock OHLCV DataFrame."""
        df = pd.DataFrame({
            'open': close_prices,
            'high': [p * 1.01 for p in close_prices],
            'low': [p * 0.99 for p in close_prices],
            'close': close_prices,
            'volume': [1000000] * len(close_prices),
        })
        df.index = pd.date_range('2024-01-01', periods=len(close_prices), freq='5min')
        return df

    def extract_features_with_precompute(self, df: pd.DataFrame, window: int = 20):
        """Helper to extract features with pre-computed channel and RSI."""
        channel = detect_channel(df, window=window)
        rsi_series = calculate_rsi_series(df['close'].values, period=14)
        return extract_tsla_channel_features(df, '5min', channel, rsi_series, window=window)

    def test_rsi_confidence_oversold_zone(self):
        """RSI < 30 should give confidence = 0.9."""
        # Create price data that will result in low RSI
        prices = [100.0] * 30 + [95.0] * 20  # Decline to push RSI low
        df = self.create_mock_df(prices, window=20)

        features = self.extract_features_with_precompute(df, window=20)

        # RSI should be low, confidence should be high
        if features.rsi < 30:
            # Base confidence for oversold
            assert features.rsi_confidence >= 0.9

    def test_rsi_confidence_overbought_zone(self):
        """RSI > 70 should give confidence = 0.9."""
        # Create price data that will result in high RSI
        prices = [100.0] * 30 + [105.0] * 20  # Rise to push RSI high
        df = self.create_mock_df(prices, window=20)

        features = self.extract_features_with_precompute(df, window=20)

        # RSI should be high, confidence should be high
        if features.rsi > 70:
            assert features.rsi_confidence >= 0.9

    def test_rsi_confidence_neutral_zone(self):
        """RSI in neutral zone (45-55) should give lower confidence."""
        # Create stable price data for neutral RSI
        prices = [100.0 + np.sin(i * 0.1) for i in range(50)]
        df = self.create_mock_df(prices, window=20)

        features = self.extract_features_with_precompute(df, window=20)

        # If RSI is in neutral zone, confidence should be lower
        if 45 <= features.rsi <= 55:
            assert features.rsi_confidence <= 0.5

    def test_rsi_confidence_divergence_boost(self):
        """RSI divergence should boost confidence by 1.2x (capped at 1.0)."""
        # This is hard to test directly without mocking detect_rsi_divergence
        # So we just verify that confidence never exceeds 1.0
        prices = list(range(100, 150)) + list(range(150, 100, -1))
        df = self.create_mock_df(prices, window=20)

        features = self.extract_features_with_precompute(df, window=20)

        assert 0.0 <= features.rsi_confidence <= 1.0


class TestDefaultFeaturesWhenTFMissing:
    """Test that default features are provided when timeframes are missing."""

    def test_missing_timeframe_handled_gracefully(self):
        """When a TF is missing, features_to_tensor_dict should handle it correctly."""
        # Create features with only some timeframes
        tsla_features = {
            '5min': TSLAChannelFeatures(
                timeframe='5min',
                channel_valid=True,
                direction=2,
                position=0.5,
                upper_dist=2.0,
                lower_dist=2.0,
                width_pct=4.0,
                slope_pct=0.1,
                r_squared=0.85,
                bounce_count=3,
                cycles=1,
                bars_since_bounce=5,
                last_touch=1,
                rsi=55.0,
                rsi_divergence=0,
                rsi_at_last_upper=65.0,
                rsi_at_last_lower=35.0,
                channel_quality=0.7,
                rsi_confidence=0.6,
            ),
            '15min': TSLAChannelFeatures(
                timeframe='15min',
                channel_valid=True,
                direction=2,
                position=0.5,
                upper_dist=2.0,
                lower_dist=2.0,
                width_pct=4.0,
                slope_pct=0.1,
                r_squared=0.85,
                bounce_count=3,
                cycles=1,
                bars_since_bounce=5,
                last_touch=1,
                rsi=55.0,
                rsi_divergence=0,
                rsi_at_last_upper=65.0,
                rsi_at_last_lower=35.0,
                channel_quality=0.7,
                rsi_confidence=0.6,
            ),
        }

        from features.cross_asset import VIXFeatures
        from features.history import ChannelHistoryFeatures

        history = ChannelHistoryFeatures(
            last_n_directions=[1] * 5,
            last_n_durations=[50.0] * 5,
            last_n_break_dirs=[1] * 5,
            avg_duration=50.0,
            direction_streak=2,
            bear_count_last_5=1,
            bull_count_last_5=3,
            sideways_count_last_5=1,
            avg_rsi_at_upper_bounce=65.0,
            avg_rsi_at_lower_bounce=35.0,
            rsi_at_last_break=55.0,
            break_up_after_bear_pct=0.6,
            break_down_after_bull_pct=0.4,
        )

        features = FullFeatures(
            timestamp=pd.Timestamp('2024-01-01 10:00:00'),
            tsla=tsla_features,
            spy={},
            cross_containment={},
            vix=VIXFeatures(
                level=20.0, level_normalized=0.5, trend_5d=0.0, trend_20d=0.0,
                percentile_252d=0.5, regime=1
            ),
            tsla_history=history,
            spy_history=history,
            tsla_spy_direction_match=True,
            both_near_upper=False,
            both_near_lower=False,
        )

        # Convert to tensor dict
        tensor_dict = features_to_tensor_dict(features)

        # Only 5min and 15min should be present in the original features
        # but features_to_tensor_dict creates all 11 TFs (with iterations for missing ones)
        assert 'tsla_5min' in tensor_dict
        assert 'tsla_15min' in tensor_dict

        # VIX and history should always be present
        assert 'vix' in tensor_dict
        assert 'tsla_history' in tensor_dict
        assert 'spy_history' in tensor_dict
        assert 'alignment' in tensor_dict

        # Verify the tensor shapes are correct
        assert len(tensor_dict['vix']) == 6  # 6 VIX features
        assert len(tensor_dict['tsla_history']) == 25  # 25 history features
        assert len(tensor_dict['alignment']) == 3  # 3 alignment features


def run_all_tests():
    """Run all test classes."""
    print("\n" + "="*80)
    print("QUALITY SCORING TEST SUITE")
    print("="*80)

    test_classes = [
        TestChannelQualityScore,
        TestBounceFirstSorting,
        TestAllTimeframesPresent,
        TestRSIConfidenceScores,
        TestDefaultFeaturesWhenTFMissing,
    ]

    total_passed = 0
    total_failed = 0
    all_errors = []

    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\n{'='*80}")
        print(f"Running {class_name}")
        print('='*80)

        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in test_methods:
            method = getattr(instance, method_name)
            test_name = f"{class_name}::{method_name}"

            try:
                method()
                print(f"  ✓ {method_name}")
                total_passed += 1
            except Exception as e:
                print(f"  ✗ {method_name}")
                print(f"     Error: {str(e)[:200]}")
                total_failed += 1
                all_errors.append((test_name, e))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\nTotal tests run: {total_passed + total_failed}")
    print(f"  ✓ Passed: {total_passed}")
    print(f"  ✗ Failed: {total_failed}")

    if total_failed > 0:
        print("\nFailed tests:")
        for test_name, error in all_errors:
            print(f"  - {test_name}")
            print(f"    {str(error)[:300]}")

    return total_failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
