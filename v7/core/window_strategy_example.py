"""
Window Selection Strategy Framework - Usage Examples

This script demonstrates practical usage of the window selection strategy framework
with realistic scenarios and edge cases.

Run with:
    python v7/core/window_strategy_example.py
"""

import numpy as np
import pandas as pd
from typing import Dict

from window_strategy import (
    get_strategy,
    SelectionStrategy,
    select_best_window_bounce_first,
    select_best_window_by_labels,
    select_best_window_balanced,
    BounceFirstStrategy,
    LabelValidityStrategy,
    BalancedScoreStrategy,
    # NOTE: register_strategy() was removed - custom strategies should be added
    # directly to the _STRATEGY_REGISTRY in window_strategy.py
)
from channel import Channel, Direction, Touch, TouchType, detect_channels_multi_window, STANDARD_WINDOWS
from ..training.labels import ChannelLabels, generate_labels_multi_window


# =============================================================================
# Example 1: Basic Usage
# =============================================================================

def example_basic_usage():
    """Demonstrate basic strategy usage."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)

    # Create mock data
    df = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 102,
        'low': np.random.randn(500).cumsum() + 98,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 500),
    })

    # Detect channels at multiple windows
    print("\nDetecting channels at multiple window sizes...")
    channels = detect_channels_multi_window(df, windows=STANDARD_WINDOWS)

    # Show what we detected
    print(f"\nDetected {len(channels)} channels:")
    for window, channel in channels.items():
        print(f"  Window {window:3d}: valid={channel.valid}, "
              f"bounces={channel.bounce_count}, r²={channel.r_squared:.3f}")

    # Strategy 1: Bounce-first (v7-v9 default)
    print("\n" + "-" * 80)
    print("Strategy 1: BOUNCE_FIRST (v7-v9 default)")
    print("-" * 80)

    strategy = get_strategy(SelectionStrategy.BOUNCE_FIRST)
    best_window, confidence = strategy.select_window(channels)

    print(f"Selected window: {best_window}")
    print(f"Confidence: {confidence:.3f}")

    if best_window:
        best_channel = channels[best_window]
        print(f"Channel details:")
        print(f"  Bounces: {best_channel.bounce_count}")
        print(f"  R²: {best_channel.r_squared:.3f}")
        print(f"  Direction: {best_channel.direction.name}")

    # Convenience function (same result)
    print("\nUsing convenience function:")
    best_window_conv, conf_conv = select_best_window_bounce_first(channels)
    print(f"Selected window: {best_window_conv} (same as above: {best_window == best_window_conv})")


# =============================================================================
# Example 2: Strategy Comparison
# =============================================================================

def example_strategy_comparison():
    """Compare different strategies on the same data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Strategy Comparison")
    print("=" * 80)

    # Create mock channels with interesting properties
    channels = {}

    # Window 50: Many bounces, low r²
    channels[50] = Channel(
        valid=True, direction=Direction.SIDEWAYS, slope=0.0, intercept=100.0,
        r_squared=0.5, std_dev=2.0,
        upper_line=np.array([104.0] * 50), lower_line=np.array([96.0] * 50),
        center_line=np.array([100.0] * 50),
        touches=[Touch(i*5, TouchType.UPPER if i%2 else TouchType.LOWER, 100) for i in range(8)],
        complete_cycles=3, bounce_count=8, width_pct=8.0, window=50, quality_score=0.922  # Normalized
    )

    # Window 100: Fewer bounces, high r²
    channels[100] = Channel(
        valid=True, direction=Direction.BULL, slope=0.02, intercept=100.0,
        r_squared=0.9, std_dev=1.0,
        upper_line=np.array([102.0] * 100), lower_line=np.array([98.0] * 100),
        center_line=np.array([100.0] * 100),
        touches=[Touch(i*20, TouchType.UPPER if i%2 else TouchType.LOWER, 100) for i in range(4)],
        complete_cycles=1, bounce_count=4, width_pct=4.0, window=100, quality_score=0.664  # Normalized
    )

    # Window 150: Middle ground
    channels[150] = Channel(
        valid=True, direction=Direction.BULL, slope=0.01, intercept=100.0,
        r_squared=0.7, std_dev=1.5,
        upper_line=np.array([103.0] * 150), lower_line=np.array([97.0] * 150),
        center_line=np.array([100.0] * 150),
        touches=[Touch(i*25, TouchType.UPPER if i%2 else TouchType.LOWER, 100) for i in range(6)],
        complete_cycles=2, bounce_count=6, width_pct=6.0, window=150, quality_score=0.834  # Normalized
    )

    # Create mock labels with different validity patterns
    labels_per_window = {
        50: {  # Many valid labels
            '5min': ChannelLabels(10, 1, 5, 2, True, True, True, True, True),
            '15min': ChannelLabels(8, 0, 3, 1, True, True, True, True, True),
            '30min': ChannelLabels(6, 1, 0, 2, True, True, True, False, True),
            '1h': ChannelLabels(5, 0, 0, 1, True, True, True, False, False),
            'daily': ChannelLabels(3, 1, 0, 2, True, True, True, False, True),
        },
        100: {  # Fewer valid labels
            '5min': ChannelLabels(15, 1, 7, 2, True, True, True, True, True),
            '15min': ChannelLabels(12, 0, 5, 1, True, True, True, True, True),
            '30min': None,
            '1h': None,
            'daily': None,
        },
        150: {  # Moderate valid labels
            '5min': ChannelLabels(20, 1, 9, 2, True, True, True, True, True),
            '15min': ChannelLabels(15, 0, 7, 1, True, True, True, True, True),
            '30min': ChannelLabels(12, 1, 5, 2, True, True, True, True, True),
            '1h': None,
            'daily': None,
        },
    }

    # Compare all strategies
    strategies = [
        ("BOUNCE_FIRST", SelectionStrategy.BOUNCE_FIRST, {}),
        ("LABEL_VALIDITY", SelectionStrategy.LABEL_VALIDITY, {}),
        ("BALANCED (40/60)", SelectionStrategy.BALANCED_SCORE, {}),
        ("BALANCED (70/30)", SelectionStrategy.BALANCED_SCORE,
         {'bounce_weight': 0.7, 'label_weight': 0.3}),
        ("QUALITY_SCORE", SelectionStrategy.QUALITY_SCORE, {}),
    ]

    print("\nComparing strategies:")
    print(f"\n{'Strategy':<20} {'Window':<10} {'Confidence':<12} {'Reason'}")
    print("-" * 70)

    for name, strategy_type, params in strategies:
        strategy = get_strategy(strategy_type, **params)
        best_window, confidence = strategy.select_window(channels, labels_per_window)

        if best_window:
            ch = channels[best_window]
            reason = f"bounces={ch.bounce_count}, r²={ch.r_squared:.2f}"
            if labels_per_window and best_window in labels_per_window:
                valid_labels = sum(1 for l in labels_per_window[best_window].values() if l)
                reason += f", valid_labels={valid_labels}"
        else:
            reason = "No valid window"

        print(f"{name:<20} {str(best_window):<10} {confidence:<12.3f} {reason}")


# =============================================================================
# Example 3: Custom Strategy (Informational - Registration API Removed)
# =============================================================================

def example_custom_strategy():
    """
    Demonstrate creating a custom strategy.

    NOTE: The register_strategy() function was removed in v10. Custom strategies
    can still be created by implementing the WindowSelectionStrategy protocol,
    but they must be added directly to the _STRATEGY_REGISTRY in window_strategy.py
    or used directly without going through get_strategy().
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Strategy (Using Direct Instantiation)")
    print("=" * 80)

    # Define a custom strategy that prefers smaller windows with good quality
    class SmallWindowPreferenceStrategy:
        """Prefer smallest window that meets quality threshold."""

        def __init__(self, min_bounces: int = 3, min_r_squared: float = 0.6):
            self.min_bounces = min_bounces
            self.min_r_squared = min_r_squared

        def select_window(self, channels, labels_per_window=None):
            if not channels:
                return None, 0.0

            # Filter to valid channels meeting quality threshold
            qualified = {
                w: ch for w, ch in channels.items()
                if ch.valid and ch.bounce_count >= self.min_bounces
                and ch.r_squared >= self.min_r_squared
            }

            if not qualified:
                # No channels meet threshold - fall back to any valid
                valid_channels = {w: ch for w, ch in channels.items() if ch.valid}
                if not valid_channels:
                    return None, 0.0
                best_window = min(valid_channels.keys())
                confidence = 0.3  # Low confidence
            else:
                # Pick smallest qualified window
                best_window = min(qualified.keys())
                # High confidence if many channels qualified
                confidence = min(1.0, 0.5 + 0.1 * len(qualified))

            return best_window, confidence

    # NOTE: register_strategy() was removed in v10. Instead, instantiate directly:
    print("\nCustom strategies can be used directly without registration:")

    # Create test channels
    channels = detect_channels_multi_window(
        pd.DataFrame({
            'open': np.random.randn(500).cumsum() + 100,
            'high': np.random.randn(500).cumsum() + 102,
            'low': np.random.randn(500).cumsum() + 98,
            'close': np.random.randn(500).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 500),
        }),
        windows=STANDARD_WINDOWS
    )

    # Use the custom strategy directly (no registration needed)
    print("\nUsing custom strategy with min_bounces=3, min_r_squared=0.6:")
    strategy = SmallWindowPreferenceStrategy(min_bounces=3, min_r_squared=0.6)
    best_window, confidence = strategy.select_window(channels)

    print(f"Selected window: {best_window}")
    print(f"Confidence: {confidence:.3f}")

    # Compare with standard BOUNCE_FIRST
    bounce_first = BounceFirstStrategy()
    bf_window, bf_conf = bounce_first.select_window(channels)

    print(f"\nComparison with BOUNCE_FIRST:")
    print(f"  Custom strategy: window={best_window}")
    print(f"  BOUNCE_FIRST:    window={bf_window}")
    print(f"  Different choice: {best_window != bf_window}")


# =============================================================================
# Example 4: Confidence-Based Filtering
# =============================================================================

def example_confidence_filtering():
    """Demonstrate using confidence scores for quality control."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Confidence-Based Filtering")
    print("=" * 80)

    MIN_CONFIDENCE = 0.7

    # Simulate processing multiple samples
    print(f"\nProcessing samples with MIN_CONFIDENCE={MIN_CONFIDENCE}")
    print("\nSimulating 5 samples:")

    # Sample scenarios with varying quality
    scenarios = [
        ("Clear winner", {
            50: (True, 5, 0.8),  # (valid, bounces, r²)
            100: (True, 2, 0.5),
        }),
        ("Close tie", {
            50: (True, 5, 0.7),
            100: (True, 5, 0.72),  # Very close
        }),
        ("No valid channels", {
            50: (False, 0, 0.1),
            100: (False, 0, 0.1),
        }),
        ("Single valid", {
            50: (True, 3, 0.6),
            100: (False, 0, 0.1),
        }),
        ("Moderate gap", {
            50: (True, 4, 0.7),
            100: (True, 6, 0.75),
        }),
    ]

    strategy = BounceFirstStrategy()

    accepted = 0
    rejected = 0

    for i, (name, scenario_data) in enumerate(scenarios, 1):
        # Create channels from scenario
        channels = {}
        for window, (valid, bounces, r2) in scenario_data.items():
            # Compute normalized quality_score using sigmoid: 2/(1+exp(-raw/5))-1
            # raw = bounces * 2.0 (assuming alternation_ratio=1.0)
            import math
            raw_score = bounces * 2.0
            normalized_score = 2.0 / (1.0 + math.exp(-raw_score / 5.0)) - 1.0 if raw_score > 0 else 0.0
            channels[window] = Channel(
                valid=valid, direction=Direction.BULL, slope=0.01, intercept=100.0,
                r_squared=r2, std_dev=1.0,
                upper_line=np.array([102.0] * window),
                lower_line=np.array([98.0] * window),
                center_line=np.array([100.0] * window),
                touches=[], complete_cycles=0, bounce_count=bounces,
                width_pct=4.0, window=window, quality_score=normalized_score
            )

        best_window, confidence = strategy.select_window(channels)

        status = "✓ ACCEPTED" if confidence >= MIN_CONFIDENCE else "✗ REJECTED"
        if confidence >= MIN_CONFIDENCE:
            accepted += 1
        else:
            rejected += 1

        print(f"\nSample {i}: {name}")
        print(f"  Window: {best_window}, Confidence: {confidence:.3f}")
        print(f"  Status: {status}")

    print(f"\n{'='*80}")
    print(f"Summary: {accepted} accepted, {rejected} rejected")
    print(f"Acceptance rate: {accepted/(accepted+rejected)*100:.1f}%")


# =============================================================================
# Example 5: Production Integration
# =============================================================================

def example_production_integration():
    """Demonstrate production-ready integration pattern."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Production Integration Pattern")
    print("=" * 80)

    class WindowSelectionConfig:
        """Configuration for window selection in production."""
        STRATEGY = SelectionStrategy.BALANCED_SCORE
        STRATEGY_PARAMS = {'bounce_weight': 0.4, 'label_weight': 0.6}
        MIN_CONFIDENCE = 0.7
        FALLBACK_STRATEGY = SelectionStrategy.BOUNCE_FIRST
        ENABLE_LOGGING = True

    def select_best_window_production(channels, labels_per_window, config=WindowSelectionConfig):
        """Production-ready window selection with fallback and logging."""
        # Try primary strategy
        strategy = get_strategy(config.STRATEGY, **config.STRATEGY_PARAMS)
        best_window, confidence = strategy.select_window(channels, labels_per_window)

        if config.ENABLE_LOGGING:
            print(f"  Primary strategy ({config.STRATEGY.value}): "
                  f"window={best_window}, confidence={confidence:.3f}")

        # Check confidence threshold
        if confidence < config.MIN_CONFIDENCE and best_window is not None:
            if config.ENABLE_LOGGING:
                print(f"  Low confidence ({confidence:.3f}), trying fallback...")

            # Try fallback strategy
            fallback = get_strategy(config.FALLBACK_STRATEGY)
            fb_window, fb_confidence = fallback.select_window(channels, labels_per_window)

            if config.ENABLE_LOGGING:
                print(f"  Fallback strategy ({config.FALLBACK_STRATEGY.value}): "
                      f"window={fb_window}, confidence={fb_confidence:.3f}")

            # Use fallback if it has better confidence
            if fb_confidence > confidence:
                best_window = fb_window
                confidence = fb_confidence
                if config.ENABLE_LOGGING:
                    print(f"  Using fallback result")

        return best_window, confidence

    # Simulate production usage
    print("\nSimulating production window selection:")

    # Create realistic channels
    df = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 102,
        'low': np.random.randn(500).cumsum() + 98,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 500),
    })

    channels = detect_channels_multi_window(df, windows=STANDARD_WINDOWS)

    print("\nChannels detected:")
    for w, ch in channels.items():
        print(f"  Window {w}: valid={ch.valid}, bounces={ch.bounce_count}")

    # Mock labels (in production, this would come from generate_labels_multi_window)
    labels_per_window = {
        w: {'5min': ChannelLabels(10, 1, 5, 2, True, True, True, True, True)}
        for w in channels.keys()
    }

    print("\nRunning production selection:")
    best_window, confidence = select_best_window_production(channels, labels_per_window)

    print(f"\nFinal result: window={best_window}, confidence={confidence:.3f}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║           Window Selection Strategy Framework - Examples                ║
    ║                                                                          ║
    ║                          v10.0.0 - 2026-01-06                           ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        example_basic_usage()
        example_strategy_comparison()
        example_custom_strategy()
        example_confidence_filtering()
        example_production_integration()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
