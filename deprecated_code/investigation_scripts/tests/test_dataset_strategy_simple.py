"""Simple test for window selection strategy in dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from v7.training.dataset import ChannelDataset, ChannelSample
from v7.core.channel import Channel, Direction
from v7.training.labels import ChannelLabels
import pandas as pd
import numpy as np


def create_simple_channel(window, bounces, r_squared):
    """Create a simple channel for testing."""
    return Channel(
        valid=True,
        direction=Direction.BULL,
        slope=0.1,
        intercept=100.0,
        r_squared=r_squared,
        std_dev=2.0,
        upper_line=np.array([101.0, 102.0, 103.0]),
        lower_line=np.array([99.0, 100.0, 101.0]),
        center_line=np.array([100.0, 101.0, 102.0]),
        touches=[],
        complete_cycles=3,
        bounce_count=bounces,
        width_pct=2.0,
        window=window,
        quality_score=0.8
    )


def create_simple_labels(duration=50):
    """Create simple labels for testing."""
    return {
        '5min': ChannelLabels(
            duration_bars=duration,
            break_direction=1,
            new_channel_direction=2,
            permanent_break=False,
            break_trigger_tf=5,
            duration_valid=True,
            direction_valid=True,
            new_channel_valid=True,
            trigger_tf_valid=True
        ),
        '15min': ChannelLabels(
            duration_bars=duration // 2,
            break_direction=1,
            new_channel_direction=2,
            permanent_break=False,
            break_trigger_tf=6,
            duration_valid=True,
            direction_valid=True,
            new_channel_valid=True,
            trigger_tf_valid=True
        )
    }


def test_select_window_old_format():
    """Test _select_window with old cache format (single channel)."""
    print("\n=== Test 1: Old Cache Format (Single Channel) ===")

    # Create old-style sample
    sample = ChannelSample(
        timestamp=pd.Timestamp('2023-01-01'),
        channel_end_idx=1000,
        channel=create_simple_channel(50, 5, 0.85),
        features=None,  # Not needed for this test
        labels=create_simple_labels(50),
        channels=None,  # Old format - no multi-window
        best_window=50,
        labels_per_window=None  # Old format
    )

    # Create dataset
    dataset = ChannelDataset([sample], strategy="bounce_first")

    # Test _select_window
    window_size, channel, labels = dataset._select_window(sample)

    print(f"Selected window: {window_size}")
    print(f"Selected channel valid: {channel.valid if channel else False}")
    print(f"Selected labels keys: {list(labels.keys()) if labels else []}")

    # Verify
    assert window_size == 50, f"Expected window 50, got {window_size}"
    assert channel is not None, "Expected channel, got None"
    assert channel.valid, "Expected valid channel"
    assert labels is not None, "Expected labels, got None"
    print("✓ Old format test PASSED")


def test_select_window_multi_window_bounce_first():
    """Test _select_window with multi-window and bounce-first strategy."""
    print("\n=== Test 2: Multi-Window Bounce-First Strategy ===")

    # Create multi-window sample
    # Window 50: 5 bounces, r²=0.85
    # Window 100: 8 bounces, r²=0.88 (should win - more bounces)
    # Window 150: 6 bounces, r²=0.90 (loses despite higher r²)
    channels = {
        50: create_simple_channel(50, 5, 0.85),
        100: create_simple_channel(100, 8, 0.88),  # Winner
        150: create_simple_channel(150, 6, 0.90)
    }

    labels_per_window = {
        50: create_simple_labels(50),
        100: create_simple_labels(55),
        150: create_simple_labels(60)
    }

    sample = ChannelSample(
        timestamp=pd.Timestamp('2023-01-01'),
        channel_end_idx=1000,
        channel=channels[100],
        features=None,
        labels=labels_per_window[100],
        channels=channels,
        best_window=100,
        labels_per_window=labels_per_window
    )

    # Test with bounce-first strategy
    dataset = ChannelDataset([sample], strategy="bounce_first")
    window_size, channel, labels = dataset._select_window(sample)

    print(f"Selected window: {window_size}")
    print(f"Expected: 100 (most bounces)")

    assert window_size == 100, f"Expected window 100 (most bounces), got {window_size}"
    assert channel.bounce_count == 8, f"Expected 8 bounces, got {channel.bounce_count}"
    print("✓ Bounce-first strategy test PASSED")


def test_select_window_label_validity():
    """Test _select_window with label validity strategy."""
    print("\n=== Test 3: Multi-Window Label Validity Strategy ===")

    # Window 50: 2 valid TF labels
    # Window 100: 2 valid TF labels
    # Window 150: 2 valid TF labels (tie - smallest wins)
    channels = {
        50: create_simple_channel(50, 5, 0.85),
        100: create_simple_channel(100, 8, 0.88),
        150: create_simple_channel(150, 6, 0.90)
    }

    labels_per_window = {
        50: create_simple_labels(50),
        100: create_simple_labels(55),
        150: create_simple_labels(60)
    }

    sample = ChannelSample(
        timestamp=pd.Timestamp('2023-01-01'),
        channel_end_idx=1000,
        channel=channels[100],
        features=None,
        labels=labels_per_window[100],
        channels=channels,
        best_window=100,
        labels_per_window=labels_per_window
    )

    # Test with label validity strategy
    dataset = ChannelDataset([sample], strategy="label_validity")
    window_size, channel, labels = dataset._select_window(sample)

    print(f"Selected window: {window_size}")
    print(f"Expected: 50 (tie on labels, smallest window wins)")

    assert window_size == 50, f"Expected window 50 (smallest on tie), got {window_size}"
    print("✓ Label validity strategy test PASSED")


def test_select_window_balanced():
    """Test _select_window with balanced score strategy."""
    print("\n=== Test 4: Multi-Window Balanced Score Strategy ===")

    channels = {
        50: create_simple_channel(50, 5, 0.85),
        100: create_simple_channel(100, 8, 0.88),
        150: create_simple_channel(150, 6, 0.90)
    }

    labels_per_window = {
        50: create_simple_labels(50),
        100: create_simple_labels(55),
        150: create_simple_labels(60)
    }

    sample = ChannelSample(
        timestamp=pd.Timestamp('2023-01-01'),
        channel_end_idx=1000,
        channel=channels[100],
        features=None,
        labels=labels_per_window[100],
        channels=channels,
        best_window=100,
        labels_per_window=labels_per_window
    )

    # Test with balanced strategy
    dataset = ChannelDataset([sample], strategy="balanced_score")
    window_size, channel, labels = dataset._select_window(sample)

    print(f"Selected window: {window_size}")
    print(f"Selected based on balanced score (40% bounce, 60% labels)")

    assert window_size is not None, "Expected valid window selection"
    assert window_size in [50, 100, 150], f"Expected valid window, got {window_size}"
    print("✓ Balanced score strategy test PASSED")


def test_custom_weights():
    """Test balanced strategy with custom weights."""
    print("\n=== Test 5: Custom Strategy Weights ===")

    channels = {
        50: create_simple_channel(50, 5, 0.85),
        100: create_simple_channel(100, 8, 0.88),
    }

    labels_per_window = {
        50: create_simple_labels(50),
        100: create_simple_labels(55),
    }

    sample = ChannelSample(
        timestamp=pd.Timestamp('2023-01-01'),
        channel_end_idx=1000,
        channel=channels[100],
        features=None,
        labels=labels_per_window[100],
        channels=channels,
        best_window=100,
        labels_per_window=labels_per_window
    )

    # Test with custom weights (prioritize bounces)
    dataset = ChannelDataset(
        [sample],
        strategy="balanced_score",
        bounce_weight=0.9,
        label_weight=0.1
    )
    window_size, _, _ = dataset._select_window(sample)

    print(f"Selected window with (0.9, 0.1) weights: {window_size}")
    print(f"Expected: 100 (high bounce weight favors more bounces)")

    assert window_size == 100, f"Expected window 100, got {window_size}"
    print("✓ Custom weights test PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Window Selection Strategy Tests")
    print("=" * 60)

    try:
        test_select_window_old_format()
        test_select_window_multi_window_bounce_first()
        test_select_window_label_validity()
        test_select_window_balanced()
        test_custom_weights()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
