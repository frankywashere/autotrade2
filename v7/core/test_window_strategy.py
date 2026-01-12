"""
Tests for Window Selection Strategy Framework

This module provides comprehensive tests for the window selection strategies,
covering all concrete implementations and edge cases.

Run with:
    pytest v7/core/test_window_strategy.py -v
"""

import pytest
import numpy as np
from typing import Dict

from .window_strategy import (
    WindowSelectionStrategy,
    BounceFirstStrategy,
    LabelValidityStrategy,
    BalancedScoreStrategy,
    QualityScoreStrategy,
    SelectionStrategy,
    get_strategy,
    # NOTE: register_strategy() was removed - custom strategies should be added
    # directly to the _STRATEGY_REGISTRY in window_strategy.py
    select_best_window_bounce_first,
    select_best_window_by_labels,
    select_best_window_balanced,
    # NOTE: select_best_channel_v7() was removed - use BounceFirstStrategy.select_window()
    # and then retrieve the channel from the channels dict with the returned window
)
from .channel import Channel, Direction, Touch, TouchType
from ..training.labels import ChannelLabels


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_channels() -> Dict[int, Channel]:
    """Create mock channels with varying bounce counts and r_squared values."""
    channels = {}

    # Window 50: 3 bounces, r_squared=0.7
    channels[50] = Channel(
        valid=True,
        direction=Direction.BULL,
        slope=0.01,
        intercept=100.0,
        r_squared=0.7,
        std_dev=1.0,
        upper_line=np.array([102.0] * 50),
        lower_line=np.array([98.0] * 50),
        center_line=np.array([100.0] * 50),
        touches=[
            Touch(10, TouchType.LOWER, 98.0),
            Touch(20, TouchType.UPPER, 102.0),
            Touch(30, TouchType.LOWER, 98.0),
        ],
        complete_cycles=1,
        bounce_count=3,
        width_pct=4.0,
        window=50,
        quality_score=0.537  # Normalized from raw 6.0 = 3 * (1 + 1.0)
    )

    # Window 100: 5 bounces, r_squared=0.6
    channels[100] = Channel(
        valid=True,
        direction=Direction.BULL,
        slope=0.01,
        intercept=100.0,
        r_squared=0.6,
        std_dev=1.0,
        upper_line=np.array([102.0] * 100),
        lower_line=np.array([98.0] * 100),
        center_line=np.array([100.0] * 100),
        touches=[
            Touch(20, TouchType.LOWER, 98.0),
            Touch(40, TouchType.UPPER, 102.0),
            Touch(60, TouchType.LOWER, 98.0),
            Touch(80, TouchType.UPPER, 102.0),
            Touch(90, TouchType.LOWER, 98.0),
        ],
        complete_cycles=2,
        bounce_count=5,
        width_pct=4.0,
        window=100,
        quality_score=0.762  # Normalized from raw 10.0 = 5 * (1 + 1.0)
    )

    # Window 150: 5 bounces (tie), r_squared=0.8 (higher)
    channels[150] = Channel(
        valid=True,
        direction=Direction.BULL,
        slope=0.01,
        intercept=100.0,
        r_squared=0.8,
        std_dev=1.0,
        upper_line=np.array([102.0] * 150),
        lower_line=np.array([98.0] * 150),
        center_line=np.array([100.0] * 150),
        touches=[
            Touch(30, TouchType.LOWER, 98.0),
            Touch(60, TouchType.UPPER, 102.0),
            Touch(90, TouchType.LOWER, 98.0),
            Touch(120, TouchType.UPPER, 102.0),
            Touch(140, TouchType.LOWER, 98.0),
        ],
        complete_cycles=2,
        bounce_count=5,
        width_pct=4.0,
        window=150,
        quality_score=0.762  # Normalized from raw 10.0 = 5 * (1 + 1.0)
    )

    return channels


@pytest.fixture
def mock_labels() -> Dict[int, Dict[str, ChannelLabels]]:
    """Create mock labels with varying validity across timeframes."""
    labels_per_window = {}

    # Window 50: 3 valid TF labels (5min, 15min, 30min)
    labels_per_window[50] = {
        '5min': ChannelLabels(10, 1, 5, 2, True, True, True, True, True),
        '15min': ChannelLabels(8, 0, 3, 1, True, True, True, True, True),
        '30min': ChannelLabels(6, 1, 0, 2, True, True, True, False, True),
        '1h': None,
        'daily': None,
    }

    # Window 100: 5 valid TF labels (all)
    labels_per_window[100] = {
        '5min': ChannelLabels(15, 1, 7, 2, True, True, True, True, True),
        '15min': ChannelLabels(12, 0, 5, 1, True, True, True, True, True),
        '30min': ChannelLabels(10, 1, 3, 2, True, True, True, True, True),
        '1h': ChannelLabels(8, 0, 1, 1, True, True, True, True, True),
        'daily': ChannelLabels(5, 1, 0, 2, True, True, True, False, True),
    }

    # Window 150: 4 valid TF labels
    labels_per_window[150] = {
        '5min': ChannelLabels(20, 1, 9, 2, True, True, True, True, True),
        '15min': ChannelLabels(15, 0, 7, 1, True, True, True, True, True),
        '30min': ChannelLabels(12, 1, 5, 2, True, True, True, True, True),
        '1h': ChannelLabels(10, 0, 3, 1, True, True, True, True, True),
        'daily': None,
    }

    return labels_per_window


@pytest.fixture
def invalid_channels() -> Dict[int, Channel]:
    """Create mock channels where all are invalid."""
    channels = {}

    channels[50] = Channel(
        valid=False,
        direction=Direction.SIDEWAYS,
        slope=0.0,
        intercept=100.0,
        r_squared=0.1,
        std_dev=1.0,
        upper_line=np.array([101.0] * 50),
        lower_line=np.array([99.0] * 50),
        center_line=np.array([100.0] * 50),
        touches=[],
        complete_cycles=0,
        bounce_count=0,
        width_pct=2.0,
        window=50,
        quality_score=0.0
    )

    return channels


# =============================================================================
# BounceFirstStrategy Tests
# =============================================================================

def test_bounce_first_selects_most_bounces(mock_channels):
    """Test that BounceFirstStrategy selects window with most bounces."""
    strategy = BounceFirstStrategy()
    best_window, confidence = strategy.select_window(mock_channels)

    # Window 100 and 150 both have 5 bounces, but 150 has higher r_squared
    assert best_window == 150
    assert 0.0 < confidence <= 1.0


def test_bounce_first_uses_r_squared_as_tiebreaker(mock_channels):
    """Test that r_squared breaks ties when bounce_count is equal."""
    strategy = BounceFirstStrategy()

    # Remove window 50 so we only have 100 and 150 (both with 5 bounces)
    channels = {k: v for k, v in mock_channels.items() if k != 50}

    best_window, confidence = strategy.select_window(channels)

    # Window 150 has higher r_squared (0.8 vs 0.6)
    assert best_window == 150
    assert confidence > 0.5  # Should have reasonable confidence


def test_bounce_first_confidence_clear_winner(mock_channels):
    """Test confidence scoring when there's a clear winner."""
    strategy = BounceFirstStrategy()

    # Keep only windows 50 (3 bounces) and 100 (5 bounces)
    channels = {50: mock_channels[50], 100: mock_channels[100]}

    best_window, confidence = strategy.select_window(channels)

    # Window 100 has 2 more bounces - clear winner
    assert best_window == 100
    assert confidence >= 0.8  # High confidence


def test_bounce_first_empty_channels():
    """Test handling of empty channels dict."""
    strategy = BounceFirstStrategy()
    best_window, confidence = strategy.select_window({})

    assert best_window is None
    assert confidence == 0.0


def test_bounce_first_all_invalid(invalid_channels):
    """Test handling when all channels are invalid."""
    strategy = BounceFirstStrategy()
    best_window, confidence = strategy.select_window(invalid_channels)

    assert best_window is None
    assert confidence == 0.0


def test_bounce_first_single_channel(mock_channels):
    """Test confidence is maximum when only one valid channel."""
    strategy = BounceFirstStrategy()
    channels = {50: mock_channels[50]}

    best_window, confidence = strategy.select_window(channels)

    assert best_window == 50
    assert confidence == 1.0


# =============================================================================
# LabelValidityStrategy Tests
# =============================================================================

def test_label_validity_selects_most_valid(mock_channels, mock_labels):
    """Test that LabelValidityStrategy selects window with most valid labels."""
    strategy = LabelValidityStrategy()
    best_window, confidence = strategy.select_window(mock_channels, mock_labels)

    # Window 100 has 5 valid labels (most)
    assert best_window == 100
    assert confidence >= 0.9  # High confidence (100% valid)


def test_label_validity_confidence_scoring(mock_channels, mock_labels):
    """Test confidence scoring based on validity rate."""
    strategy = LabelValidityStrategy()

    # Test with window 50 only (3/5 valid = 60%)
    channels = {50: mock_channels[50]}
    labels = {50: mock_labels[50]}

    best_window, confidence = strategy.select_window(channels, labels)

    assert best_window == 50
    assert 0.7 <= confidence <= 0.9  # Should be in 60-80% range


def test_label_validity_tie_prefers_smaller_window(mock_channels, mock_labels):
    """Test that ties are broken by preferring smaller window."""
    strategy = LabelValidityStrategy()

    # Make windows 100 and 150 have same number of valid labels
    modified_labels = mock_labels.copy()
    modified_labels[150] = mock_labels[100].copy()

    channels = {100: mock_channels[100], 150: mock_channels[150]}

    best_window, confidence = strategy.select_window(channels, modified_labels)

    # Should prefer smaller window (100)
    assert best_window == 100


def test_label_validity_no_labels_provided(mock_channels):
    """Test fallback to bounce-first when no labels provided."""
    strategy = LabelValidityStrategy()
    best_window, confidence = strategy.select_window(mock_channels, None)

    # Should fall back to bounce-first behavior (window 150 has most bounces + highest r2)
    assert best_window == 150


def test_label_validity_empty_labels():
    """Test handling when labels dict is empty."""
    strategy = LabelValidityStrategy()
    best_window, confidence = strategy.select_window({}, {})

    assert best_window is None
    assert confidence == 0.0


def test_label_validity_all_labels_none(mock_channels):
    """Test handling when all labels are None."""
    strategy = LabelValidityStrategy()

    # Create labels where everything is None
    labels = {
        50: {'5min': None, '15min': None, '1h': None},
        100: {'5min': None, '15min': None, '1h': None},
    }

    best_window, confidence = strategy.select_window(mock_channels, labels)

    assert best_window is None
    assert confidence == 0.0


# =============================================================================
# BalancedScoreStrategy Tests
# =============================================================================

def test_balanced_score_default_weights(mock_channels, mock_labels):
    """Test BalancedScoreStrategy with default weights (0.4 bounce, 0.6 label)."""
    strategy = BalancedScoreStrategy()
    best_window, confidence = strategy.select_window(mock_channels, mock_labels)

    # With default weights, label validity (0.6) should dominate
    # Window 100 has best label validity (5/5 valid)
    assert best_window == 100
    assert confidence > 0.0


def test_balanced_score_custom_weights(mock_channels, mock_labels):
    """Test BalancedScoreStrategy with custom weights."""
    # High bounce weight (0.8 bounce, 0.2 label)
    strategy = BalancedScoreStrategy(bounce_weight=0.8, label_weight=0.2)
    best_window, confidence = strategy.select_window(mock_channels, mock_labels)

    # With high bounce weight, should prefer windows 100/150 (most bounces)
    # Tie will be broken by r_squared (150 > 100)
    assert best_window in [100, 150]


def test_balanced_score_no_labels_falls_back_to_bounces(mock_channels):
    """Test that BalancedScoreStrategy handles missing labels gracefully."""
    strategy = BalancedScoreStrategy()
    best_window, confidence = strategy.select_window(mock_channels, None)

    # Without labels, should use bounce component only
    # Windows 100 and 150 both have 5 bounces (tied), so best_window could be either
    # (the selection is deterministic based on max() on dict keys)
    assert best_window in [100, 150]


def test_balanced_score_weight_validation():
    """Test that invalid weights raise errors."""
    with pytest.raises(ValueError):
        BalancedScoreStrategy(bounce_weight=-0.1, label_weight=1.1)

    with pytest.raises(ValueError):
        BalancedScoreStrategy(bounce_weight=0.5, label_weight=-0.5)


def test_balanced_score_weights_sum_warning():
    """Test that warning is issued when weights don't sum to 1.0."""
    with pytest.warns(UserWarning):
        # Weights sum to 1.5 - should warn
        BalancedScoreStrategy(bounce_weight=0.8, label_weight=0.7)


def test_balanced_score_confidence_large_gap(mock_channels, mock_labels):
    """Test high confidence when there's a large score gap."""
    strategy = BalancedScoreStrategy()

    # Use only windows 50 and 100 - 100 should dominate
    channels = {50: mock_channels[50], 100: mock_channels[100]}
    labels = {50: mock_labels[50], 100: mock_labels[100]}

    best_window, confidence = strategy.select_window(channels, labels)

    # Window 100 should win with high confidence
    assert best_window == 100
    assert confidence >= 0.7


def test_balanced_score_all_zero_scores(invalid_channels):
    """Test handling when all composite scores are zero."""
    strategy = BalancedScoreStrategy()

    # Invalid channels with 0 bounces
    best_window, confidence = strategy.select_window(invalid_channels, None)

    assert best_window is None
    assert confidence == 0.0


# =============================================================================
# QualityScoreStrategy Tests
# =============================================================================

def test_quality_score_selects_highest_score(mock_channels):
    """Test that QualityScoreStrategy selects window with highest quality_score."""
    strategy = QualityScoreStrategy()
    best_window, confidence = strategy.select_window(mock_channels)

    # Windows 100 and 150 both have quality_score=0.762 (normalized)
    # Should prefer 100 (smaller window)
    assert best_window in [100, 150]


def test_quality_score_tie_prefers_smaller_window(mock_channels):
    """Test that ties are broken by preferring smaller window."""
    strategy = QualityScoreStrategy()

    # Windows 100 and 150 have same quality_score (0.762 normalized)
    channels = {100: mock_channels[100], 150: mock_channels[150]}

    best_window, confidence = strategy.select_window(channels)

    # Should prefer smaller window
    assert best_window == 100


def test_quality_score_confidence_clear_winner(mock_channels):
    """Test confidence when there's a clear winner."""
    strategy = QualityScoreStrategy()

    # Windows 50 (score 0.537) vs 100 (score 0.762)
    channels = {50: mock_channels[50], 100: mock_channels[100]}

    best_window, confidence = strategy.select_window(channels)

    # Window 100 has higher score
    assert best_window == 100
    # Relative gap = (10-6)/10 = 0.4 -> confidence ~0.8-0.9
    assert confidence >= 0.7


def test_quality_score_all_zero(invalid_channels):
    """Test handling when all quality scores are zero."""
    strategy = QualityScoreStrategy()
    best_window, confidence = strategy.select_window(invalid_channels)

    assert best_window is None
    assert confidence == 0.0


# =============================================================================
# Strategy Factory Tests
# =============================================================================

def test_get_strategy_bounce_first():
    """Test factory creates BounceFirstStrategy."""
    strategy = get_strategy(SelectionStrategy.BOUNCE_FIRST)
    assert isinstance(strategy, BounceFirstStrategy)


def test_get_strategy_label_validity():
    """Test factory creates LabelValidityStrategy."""
    strategy = get_strategy(SelectionStrategy.LABEL_VALIDITY)
    assert isinstance(strategy, LabelValidityStrategy)


def test_get_strategy_balanced_score():
    """Test factory creates BalancedScoreStrategy."""
    strategy = get_strategy(SelectionStrategy.BALANCED_SCORE)
    assert isinstance(strategy, BalancedScoreStrategy)
    assert strategy.bounce_weight == 0.4
    assert strategy.label_weight == 0.6


def test_get_strategy_balanced_score_custom_weights():
    """Test factory creates BalancedScoreStrategy with custom weights."""
    strategy = get_strategy(
        SelectionStrategy.BALANCED_SCORE,
        bounce_weight=0.3,
        label_weight=0.7
    )
    assert isinstance(strategy, BalancedScoreStrategy)
    assert strategy.bounce_weight == 0.3
    assert strategy.label_weight == 0.7


def test_get_strategy_quality_score():
    """Test factory creates QualityScoreStrategy."""
    strategy = get_strategy(SelectionStrategy.QUALITY_SCORE)
    assert isinstance(strategy, QualityScoreStrategy)


def test_get_strategy_invalid():
    """Test factory raises error for invalid strategy."""
    with pytest.raises(ValueError):
        get_strategy("invalid_strategy")


# =============================================================================
# Custom Strategy Registration Tests
# =============================================================================
# NOTE: register_strategy() was removed in v10. Custom strategies should now be
# added directly to _STRATEGY_REGISTRY in window_strategy.py or used directly.
# The tests below demonstrate the new recommended pattern.

def test_custom_strategy_direct_usage(mock_channels):
    """Test using a custom strategy directly without registration."""
    class CustomStrategy:
        def select_window(self, channels, labels_per_window=None):
            # Always return first window
            if not channels:
                return None, 0.0
            first_window = min(channels.keys())
            return first_window, 0.5

    # Use it directly (no registration needed)
    strategy = CustomStrategy()
    best_window, confidence = strategy.select_window(mock_channels)

    # Should return smallest window (50)
    assert best_window == 50
    assert confidence == 0.5


def test_custom_strategy_protocol_compliance(mock_channels):
    """Test that custom strategies implementing the protocol work correctly."""
    class ProtocolCompliantStrategy:
        """A custom strategy that implements WindowSelectionStrategy protocol."""
        def select_window(self, channels, labels_per_window=None):
            # Return window with highest r_squared
            if not channels:
                return None, 0.0
            valid_channels = {w: ch for w, ch in channels.items() if ch.valid}
            if not valid_channels:
                return None, 0.0
            best_window = max(valid_channels.keys(), key=lambda w: valid_channels[w].r_squared)
            return best_window, 0.8

    strategy = ProtocolCompliantStrategy()
    best_window, confidence = strategy.select_window(mock_channels)

    # Window 150 has highest r_squared (0.8)
    assert best_window == 150
    assert confidence == 0.8


# =============================================================================
# Convenience Function Tests
# =============================================================================

def test_select_best_window_bounce_first_convenience(mock_channels):
    """Test convenience function for bounce-first selection."""
    best_window, confidence = select_best_window_bounce_first(mock_channels)

    # Should match BounceFirstStrategy behavior
    strategy = BounceFirstStrategy()
    expected_window, expected_conf = strategy.select_window(mock_channels)

    assert best_window == expected_window
    assert confidence == expected_conf


def test_select_best_window_by_labels_convenience(mock_labels):
    """Test convenience function for label validity selection."""
    best_window, confidence = select_best_window_by_labels(mock_labels)

    # Should create dummy channels and use LabelValidityStrategy
    assert best_window == 100  # Window with most valid labels
    assert confidence >= 0.9


def test_select_best_window_by_labels_with_channels(mock_channels, mock_labels):
    """Test convenience function with channels provided."""
    best_window, confidence = select_best_window_by_labels(mock_labels, mock_channels)

    # Should use actual channels
    assert best_window == 100
    assert confidence >= 0.9


def test_select_best_window_balanced_convenience(mock_channels, mock_labels):
    """Test convenience function for balanced selection."""
    best_window, confidence = select_best_window_balanced(
        mock_channels, mock_labels,
        bounce_weight=0.3, label_weight=0.7
    )

    # Should use BalancedScoreStrategy with custom weights
    assert best_window is not None
    assert confidence > 0.0


# NOTE: select_best_channel_v7() was removed in v10. The tests below demonstrate
# the new recommended pattern using BounceFirstStrategy directly.

def test_select_best_channel_using_bounce_first(mock_channels):
    """Test getting best channel using BounceFirstStrategy (replacement for select_best_channel_v7)."""
    # New pattern: use BounceFirstStrategy and retrieve channel from dict
    strategy = BounceFirstStrategy()
    best_window, confidence = strategy.select_window(mock_channels)

    # Retrieve the channel
    best_channel = mock_channels.get(best_window) if best_window else None

    # Should return window and Channel object
    assert best_window in [100, 150]  # One of the winners
    assert best_channel == mock_channels[best_window]
    assert isinstance(best_channel, Channel)


def test_select_best_channel_empty_using_bounce_first():
    """Test handling empty channels with BounceFirstStrategy."""
    strategy = BounceFirstStrategy()
    best_window, confidence = strategy.select_window({})

    # Retrieve the channel (will be None since best_window is None)
    best_channel = None if best_window is None else {}[best_window]

    assert best_channel is None
    assert best_window is None


# =============================================================================
# Edge Case Tests
# =============================================================================

def test_mixed_valid_invalid_channels(mock_channels, invalid_channels):
    """Test that strategies handle mix of valid and invalid channels."""
    # Combine valid and invalid channels
    mixed = {**mock_channels, **invalid_channels}

    strategy = BounceFirstStrategy()
    best_window, confidence = strategy.select_window(mixed)

    # Should select from valid channels only
    assert best_window in [100, 150]
    assert confidence > 0.0


def test_labels_missing_for_some_windows(mock_channels, mock_labels):
    """Test BalancedScoreStrategy when labels missing for some windows."""
    strategy = BalancedScoreStrategy()

    # Remove labels for window 150
    labels = {50: mock_labels[50], 100: mock_labels[100]}

    best_window, confidence = strategy.select_window(mock_channels, labels)

    # Should still work, treating missing labels as 0 score
    assert best_window in [50, 100]
    assert confidence > 0.0


# =============================================================================
# Integration Tests
# =============================================================================

def test_all_strategies_agree_on_single_channel(mock_channels):
    """Test that all strategies agree when there's only one valid channel."""
    channels = {50: mock_channels[50]}
    labels = {50: {'5min': ChannelLabels(10, 1, 5, 2, True, True, True, True, True)}}

    strategies = [
        BounceFirstStrategy(),
        LabelValidityStrategy(),
        BalancedScoreStrategy(),
        QualityScoreStrategy(),
    ]

    results = [s.select_window(channels, labels) for s in strategies]

    # All should select window 50
    for window, confidence in results:
        assert window == 50
        assert confidence > 0.0


def test_strategies_handle_complex_scenario(mock_channels, mock_labels):
    """Test all strategies on a complex real-world scenario."""
    # Add one more channel with interesting properties
    mock_channels[200] = Channel(
        valid=True,
        direction=Direction.BULL,
        slope=0.015,
        intercept=100.0,
        r_squared=0.95,  # Very high r_squared
        std_dev=1.0,
        upper_line=np.array([102.0] * 200),
        lower_line=np.array([98.0] * 200),
        center_line=np.array([100.0] * 200),
        touches=[
            Touch(40, TouchType.LOWER, 98.0),
            Touch(80, TouchType.UPPER, 102.0),
            Touch(120, TouchType.LOWER, 98.0),
            Touch(160, TouchType.UPPER, 102.0),
        ],
        complete_cycles=1,
        bounce_count=4,
        width_pct=4.0,
        window=200,
        quality_score=0.664  # Normalized from raw 8.0 = 4 * (1 + 1.0)
    )

    # Add labels for window 200 (2 valid TFs)
    mock_labels[200] = {
        '5min': ChannelLabels(25, 1, 11, 2, True, True, True, True, True),
        '15min': ChannelLabels(20, 0, 9, 1, True, True, True, True, True),
        '30min': None,
        '1h': None,
        'daily': None,
    }

    # Test all strategies
    bounce_first = BounceFirstStrategy()
    label_validity = LabelValidityStrategy()
    balanced = BalancedScoreStrategy()
    quality_score = QualityScoreStrategy()

    b_win, b_conf = bounce_first.select_window(mock_channels, mock_labels)
    l_win, l_conf = label_validity.select_window(mock_channels, mock_labels)
    bal_win, bal_conf = balanced.select_window(mock_channels, mock_labels)
    q_win, q_conf = quality_score.select_window(mock_channels, mock_labels)

    # BounceFirst: Should prefer 100/150 (5 bounces)
    assert b_win in [100, 150]

    # LabelValidity: Should prefer 100 (5 valid labels)
    assert l_win == 100

    # Balanced: Should prefer 100 (best overall balance)
    assert bal_win == 100

    # QualityScore: Should prefer 100/150 (score 10.0)
    assert q_win in [100, 150]

    # All should have reasonable confidence
    assert all(conf > 0.0 for conf in [b_conf, l_conf, bal_conf, q_conf])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
