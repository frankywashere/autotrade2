"""
Tests for Walk-Forward Validation Logic

Tests cover:
1. Window generation (expanding vs rolling)
2. Sample filtering by window dates
3. Chronological ordering guarantees
4. Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, NamedTuple


# =============================================================================
# Mock Data Structures (to be replaced with actual implementation)
# =============================================================================

class WalkForwardWindow(NamedTuple):
    """Represents a single walk-forward validation window."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    window_id: int


class ChannelSample(NamedTuple):
    """Mock channel sample with timestamp."""
    timestamp: pd.Timestamp
    features: dict
    labels: dict


# =============================================================================
# Helper Functions (simulate expected implementation)
# =============================================================================

def generate_walk_forward_windows(
    start_date: str,
    end_date: str,
    num_windows: int,
    mode: str = "expanding",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> List[WalkForwardWindow]:
    """
    Generate walk-forward validation windows.

    Args:
        start_date: Overall start date (YYYY-MM-DD)
        end_date: Overall end date (YYYY-MM-DD)
        num_windows: Number of windows to generate
        mode: "expanding" or "rolling"
        val_ratio: Ratio of each window for validation
        test_ratio: Ratio of each window for testing

    Returns:
        List of WalkForwardWindow objects in chronological order
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = (end - start).days

    if num_windows < 1:
        raise ValueError("num_windows must be at least 1")

    if mode not in ["expanding", "rolling"]:
        raise ValueError(f"mode must be 'expanding' or 'rolling', got '{mode}'")

    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    windows = []
    window_size_days = total_days // num_windows

    for i in range(num_windows):
        window_end = start + timedelta(days=window_size_days * (i + 1))
        if i == num_windows - 1:
            window_end = end  # Last window extends to end_date

        # Calculate split points
        window_duration = (window_end - start).days if mode == "expanding" else window_size_days
        test_days = int(window_duration * test_ratio)
        val_days = int(window_duration * val_ratio)

        test_start = window_end - timedelta(days=test_days)
        val_start = test_start - timedelta(days=val_days)

        if mode == "expanding":
            train_start = start
        else:  # rolling
            train_start = start + timedelta(days=window_size_days * i)

        train_end = val_start - timedelta(days=1)
        val_end = test_start - timedelta(days=1)
        test_end = window_end

        windows.append(WalkForwardWindow(
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
            window_id=i
        ))

    return windows


def split_samples_by_window(
    samples: List[ChannelSample],
    window: WalkForwardWindow
) -> Tuple[List[ChannelSample], List[ChannelSample], List[ChannelSample]]:
    """
    Split samples into train/val/test based on window dates.

    Args:
        samples: List of all samples
        window: WalkForwardWindow defining date ranges

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    train = [s for s in samples if window.train_start <= s.timestamp <= window.train_end]
    val = [s for s in samples if window.val_start <= s.timestamp <= window.val_end]
    test = [s for s in samples if window.test_start <= s.timestamp <= window.test_end]

    return train, val, test


# =============================================================================
# Test: Window Generation
# =============================================================================

class TestGenerateWalkForwardWindows:
    """Test suite for window generation logic."""

    def test_basic_window_generation(self):
        """Test basic window generation with default parameters."""
        windows = generate_walk_forward_windows(
            start_date="2020-01-01",
            end_date="2023-12-31",
            num_windows=4,
            mode="expanding"
        )

        assert len(windows) == 4
        assert all(isinstance(w, WalkForwardWindow) for w in windows)
        assert windows[0].window_id == 0
        assert windows[-1].window_id == 3

    def test_expanding_windows(self):
        """Test expanding window mode - train set grows over time."""
        windows = generate_walk_forward_windows(
            start_date="2020-01-01",
            end_date="2024-01-01",
            num_windows=3,
            mode="expanding"
        )

        # All windows should start from the same date
        for w in windows:
            assert w.train_start == pd.Timestamp("2020-01-01")

        # Train end should increase
        assert windows[0].train_end < windows[1].train_end < windows[2].train_end

        # Each window's train set should be larger than previous
        train_size_0 = (windows[0].train_end - windows[0].train_start).days
        train_size_1 = (windows[1].train_end - windows[1].train_start).days
        assert train_size_1 > train_size_0

    def test_rolling_windows(self):
        """Test rolling window mode - train set size stays constant."""
        windows = generate_walk_forward_windows(
            start_date="2020-01-01",
            end_date="2024-01-01",
            num_windows=3,
            mode="rolling"
        )

        # Train start should advance over time
        assert windows[0].train_start < windows[1].train_start < windows[2].train_start

        # Train window size should be approximately constant
        train_size_0 = (windows[0].train_end - windows[0].train_start).days
        train_size_1 = (windows[1].train_end - windows[1].train_start).days

        # Allow some tolerance for rounding
        assert abs(train_size_0 - train_size_1) <= 2

    def test_single_window(self):
        """Test generation with a single window."""
        windows = generate_walk_forward_windows(
            start_date="2020-01-01",
            end_date="2021-01-01",
            num_windows=1,
            mode="expanding"
        )

        assert len(windows) == 1
        assert windows[0].train_start == pd.Timestamp("2020-01-01")

    def test_custom_split_ratios(self):
        """Test custom validation and test ratios."""
        windows = generate_walk_forward_windows(
            start_date="2020-01-01",
            end_date="2021-01-01",
            num_windows=1,
            mode="expanding",
            val_ratio=0.2,
            test_ratio=0.2
        )

        window = windows[0]
        total_days = (window.test_end - window.train_start).days
        val_days = (window.val_end - window.val_start).days + 1
        test_days = (window.test_end - window.test_start).days + 1

        # Check ratios are approximately correct (within rounding)
        assert abs(val_days / total_days - 0.2) < 0.05
        assert abs(test_days / total_days - 0.2) < 0.05

    def test_invalid_num_windows(self):
        """Test error handling for invalid num_windows."""
        with pytest.raises(ValueError, match="num_windows must be at least 1"):
            generate_walk_forward_windows(
                start_date="2020-01-01",
                end_date="2021-01-01",
                num_windows=0,
                mode="expanding"
            )

    def test_invalid_mode(self):
        """Test error handling for invalid mode."""
        with pytest.raises(ValueError, match="mode must be"):
            generate_walk_forward_windows(
                start_date="2020-01-01",
                end_date="2021-01-01",
                num_windows=2,
                mode="invalid_mode"
            )

    def test_invalid_ratios(self):
        """Test error handling for invalid split ratios."""
        with pytest.raises(ValueError, match="val_ratio \\+ test_ratio must be"):
            generate_walk_forward_windows(
                start_date="2020-01-01",
                end_date="2021-01-01",
                num_windows=2,
                mode="expanding",
                val_ratio=0.6,
                test_ratio=0.6
            )


# =============================================================================
# Test: Sample Filtering
# =============================================================================

class TestSplitSamplesByWindow:
    """Test suite for sample filtering by window dates."""

    def create_mock_samples(self, start: str, end: str, freq: str = "D") -> List[ChannelSample]:
        """Helper to create mock samples."""
        dates = pd.date_range(start=start, end=end, freq=freq)
        return [
            ChannelSample(timestamp=date, features={}, labels={})
            for date in dates
        ]

    def test_basic_sample_splitting(self):
        """Test basic sample splitting across train/val/test."""
        samples = self.create_mock_samples("2020-01-01", "2020-12-31", freq="D")

        window = WalkForwardWindow(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-09-30"),
            val_start=pd.Timestamp("2020-10-01"),
            val_end=pd.Timestamp("2020-11-15"),
            test_start=pd.Timestamp("2020-11-16"),
            test_end=pd.Timestamp("2020-12-31"),
            window_id=0
        )

        train, val, test = split_samples_by_window(samples, window)

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(samples)

    def test_no_overlap_between_sets(self):
        """Verify no samples appear in multiple sets."""
        samples = self.create_mock_samples("2020-01-01", "2020-12-31", freq="D")

        window = WalkForwardWindow(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-09-30"),
            val_start=pd.Timestamp("2020-10-01"),
            val_end=pd.Timestamp("2020-11-15"),
            test_start=pd.Timestamp("2020-11-16"),
            test_end=pd.Timestamp("2020-12-31"),
            window_id=0
        )

        train, val, test = split_samples_by_window(samples, window)

        train_timestamps = {s.timestamp for s in train}
        val_timestamps = {s.timestamp for s in val}
        test_timestamps = {s.timestamp for s in test}

        # Check no overlap
        assert len(train_timestamps & val_timestamps) == 0
        assert len(val_timestamps & test_timestamps) == 0
        assert len(train_timestamps & test_timestamps) == 0

    def test_all_samples_accounted_for(self):
        """Verify all samples within window range are included."""
        samples = self.create_mock_samples("2020-01-01", "2020-12-31", freq="D")

        window = WalkForwardWindow(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-09-30"),
            val_start=pd.Timestamp("2020-10-01"),
            val_end=pd.Timestamp("2020-11-15"),
            test_start=pd.Timestamp("2020-11-16"),
            test_end=pd.Timestamp("2020-12-31"),
            window_id=0
        )

        train, val, test = split_samples_by_window(samples, window)

        all_split_timestamps = {s.timestamp for s in train + val + test}
        all_sample_timestamps = {s.timestamp for s in samples}

        # All samples in range should be accounted for
        assert all_split_timestamps == all_sample_timestamps

    def test_samples_outside_window(self):
        """Test that samples outside window are excluded."""
        # Create samples from 2019-2021
        samples = self.create_mock_samples("2019-01-01", "2021-12-31", freq="D")

        # Window only covers 2020
        window = WalkForwardWindow(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-09-30"),
            val_start=pd.Timestamp("2020-10-01"),
            val_end=pd.Timestamp("2020-11-15"),
            test_start=pd.Timestamp("2020-11-16"),
            test_end=pd.Timestamp("2020-12-31"),
            window_id=0
        )

        train, val, test = split_samples_by_window(samples, window)

        # Should only include 2020 samples
        total_included = len(train) + len(val) + len(test)
        samples_2020 = [s for s in samples if s.timestamp.year == 2020]
        assert total_included == len(samples_2020)

    def test_empty_sets_possible(self):
        """Test that empty train/val/test sets are possible if no samples in range."""
        samples = self.create_mock_samples("2019-01-01", "2019-12-31", freq="D")

        # Window in 2020, no samples there
        window = WalkForwardWindow(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-09-30"),
            val_start=pd.Timestamp("2020-10-01"),
            val_end=pd.Timestamp("2020-11-15"),
            test_start=pd.Timestamp("2020-11-16"),
            test_end=pd.Timestamp("2020-12-31"),
            window_id=0
        )

        train, val, test = split_samples_by_window(samples, window)

        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0


# =============================================================================
# Test: Chronological Ordering
# =============================================================================

class TestWindowDateOrdering:
    """Test suite for chronological ordering guarantees."""

    def test_windows_chronological_order(self):
        """Ensure windows are in chronological order."""
        windows = generate_walk_forward_windows(
            start_date="2020-01-01",
            end_date="2024-01-01",
            num_windows=5,
            mode="expanding"
        )

        for i in range(len(windows) - 1):
            # Each window should start after or at the same time as previous
            assert windows[i].train_start <= windows[i + 1].train_start
            assert windows[i].test_end <= windows[i + 1].test_end

    def test_within_window_ordering(self):
        """Ensure train < val < test within each window."""
        windows = generate_walk_forward_windows(
            start_date="2020-01-01",
            end_date="2024-01-01",
            num_windows=3,
            mode="expanding"
        )

        for w in windows:
            # Train comes before val
            assert w.train_start <= w.train_end < w.val_start
            # Val comes before test
            assert w.val_start <= w.val_end < w.test_start
            # Test is last
            assert w.test_start <= w.test_end

    def test_no_future_data_leakage(self):
        """Ensure no future data leakage - train cannot use val/test data."""
        windows = generate_walk_forward_windows(
            start_date="2020-01-01",
            end_date="2024-01-01",
            num_windows=4,
            mode="expanding"
        )

        for w in windows:
            # Train must end before validation starts
            assert w.train_end < w.val_start
            # Validation must end before test starts
            assert w.val_end < w.test_start

    def test_window_ids_sequential(self):
        """Ensure window IDs are sequential."""
        windows = generate_walk_forward_windows(
            start_date="2020-01-01",
            end_date="2024-01-01",
            num_windows=6,
            mode="rolling"
        )

        for i, w in enumerate(windows):
            assert w.window_id == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
