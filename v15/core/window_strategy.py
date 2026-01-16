"""
Window Selection Strategy Framework for V15.

This module provides a flexible, extensible framework for selecting the best window
size in a multi-window channel detection system. The framework allows different
strategies to be used based on different criteria (bounce quality, label validity,
balanced scoring, etc.).

The system is designed for v15 multi-window architecture where channels are detected
at multiple window sizes and the best one must be selected for training.

Architecture:
    - WindowSelectionStrategy: Protocol defining the strategy interface
    - Concrete strategy implementations (BounceFirstStrategy, LabelValidityStrategy, etc.)
    - SelectionStrategy: Enum for configuration
    - get_strategy(): Factory function for creating strategy instances

Usage Example:
    ```python
    from v15.core.window_strategy import get_strategy, SelectionStrategy

    # Get a strategy
    strategy = get_strategy(SelectionStrategy.BOUNCE_FIRST)

    # Select best window from a ChannelSample
    window = strategy.select_window(sample)
    ```
"""

from typing import Dict, Optional, Tuple, Protocol, Any, TYPE_CHECKING, Union
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from ..types import ChannelSample, ChannelLabels


# =============================================================================
# Core Protocol
# =============================================================================

class WindowSelectionStrategy(Protocol):
    """
    Protocol defining the interface for window selection strategies.

    All concrete strategies must implement this protocol by providing
    a select_window() method that takes a ChannelSample and returns
    the best window size.
    """

    def select_window(self, sample: 'ChannelSample') -> int:
        """
        Select the best window size from a ChannelSample.

        Args:
            sample: ChannelSample with labels_per_window containing
                   labels for each window across timeframes.

        Returns:
            The selected window size (int).

        Edge Cases:
            - If labels_per_window is empty: returns sample.best_window
            - If all windows have no valid labels: returns smallest window
        """
        ...


# =============================================================================
# Concrete Strategy Implementations
# =============================================================================

@dataclass
class BounceFirstStrategy:
    """
    Select window based on bounce quality (current v7-v9 default).

    This strategy simply returns the sample's best_window which was
    computed during channel detection based on bounce quality.

    Selection criteria (in priority order):
        1. Most bounces (bounce_count) - primary criterion
        2. Best linear fit (r_squared) - tiebreaker

    This strategy prioritizes channels that demonstrate clear oscillation
    behavior (high bounce_count) with good linear trend quality (high r_squared).

    Note:
        This is the default strategy that maintains backward compatibility
        with the existing behavior where best_window is pre-computed.
    """

    def select_window(self, sample: 'ChannelSample') -> int:
        """
        Select window using pre-computed best_window from sample.

        Returns:
            sample.best_window (computed during channel detection)
        """
        return sample.best_window


@dataclass
class LabelValidityStrategy:
    """
    Select window based on label validity across timeframes.

    Selection criteria:
        1. Count valid (non-None) labels across all timeframes
        2. Window with most valid TF labels wins
        3. On tie: prefer smaller window size

    This strategy is useful when label quality matters more than channel
    bounce quality. It ensures the selected window has the most complete
    label information across multiple timeframes.

    Attributes:
        target_tf: If specified, prioritize windows with valid labels for this TF.
                  If None, considers validity across all TFs equally.
    """
    target_tf: Optional[str] = None

    def select_window(self, sample: 'ChannelSample') -> int:
        """
        Select window with most valid labels across timeframes.

        Returns:
            Window size with maximum valid label count
        """
        if not sample.labels_per_window:
            return sample.best_window

        # Count valid labels per window
        validity_counts = {}

        for window_size, tf_labels in sample.labels_per_window.items():
            if self.target_tf:
                # Only count if target TF is valid
                label = tf_labels.get(self.target_tf)
                if label is not None:
                    validity_counts[window_size] = 1
                else:
                    validity_counts[window_size] = 0
            else:
                # Count all valid TF labels
                valid_count = sum(
                    1 for label in tf_labels.values()
                    if label is not None
                )
                validity_counts[window_size] = valid_count

        if not validity_counts:
            return sample.best_window

        # Find window(s) with most valid labels
        max_valid = max(validity_counts.values())

        if max_valid == 0:
            return sample.best_window

        # Among windows with max valid labels, prefer smallest
        best_windows = [w for w, count in validity_counts.items() if count == max_valid]
        return min(best_windows)


@dataclass
class BalancedScoreStrategy:
    """
    Select window using weighted combination of criteria.

    This strategy considers multiple factors:
        - Label validity count across timeframes
        - Preference for smaller windows (stability)

    Default behavior prioritizes label validity while preferring
    smaller windows on ties.

    Attributes:
        prefer_smaller: If True, prefer smaller windows on ties (default True)
        target_tf: If specified, prioritize windows with valid labels for this TF
    """
    prefer_smaller: bool = True
    target_tf: Optional[str] = None

    def select_window(self, sample: 'ChannelSample') -> int:
        """
        Select window using balanced scoring.

        Returns:
            Best window based on weighted criteria
        """
        if not sample.labels_per_window:
            return sample.best_window

        # Calculate scores per window
        scores = {}

        for window_size, tf_labels in sample.labels_per_window.items():
            # Label validity score
            if self.target_tf:
                label = tf_labels.get(self.target_tf)
                label_score = 1.0 if label is not None else 0.0
            else:
                total_tfs = len(tf_labels) if tf_labels else 1
                valid_count = sum(
                    1 for label in tf_labels.values()
                    if label is not None
                )
                label_score = valid_count / total_tfs if total_tfs > 0 else 0.0

            # Window size score (smaller is better)
            # Normalize assuming windows are in range [10, 80]
            max_window = 80
            min_window = 10
            window_score = 1.0 - (window_size - min_window) / (max_window - min_window)

            # Combine scores (label validity weighted higher)
            scores[window_size] = 0.7 * label_score + 0.3 * window_score

        if not scores:
            return sample.best_window

        # Find best window
        if self.prefer_smaller:
            # On tie, prefer smaller window
            best_window = max(
                scores.keys(),
                key=lambda w: (scores[w], -w)
            )
        else:
            best_window = max(scores.keys(), key=lambda w: scores[w])

        return best_window


@dataclass
class QualityScoreStrategy:
    """
    Select window based on a pre-computed quality score.

    This strategy looks for a quality_score field in the labels
    and selects the window with the highest score.

    Falls back to best_window if quality scores are not available.
    """

    def select_window(self, sample: 'ChannelSample') -> int:
        """
        Select window with highest quality score.

        Returns:
            Window with maximum quality_score, or best_window as fallback
        """
        # For v15, we don't have quality_score in labels_per_window
        # This is a placeholder for future implementation
        # Fall back to best_window
        return sample.best_window


@dataclass
class LearnedStrategy:
    """
    Strategy for learned window selection.

    This strategy is special - it doesn't select a single window but instead
    signals that the model should learn to select the window. During data
    loading, this causes the dataset to return features for ALL windows
    so the model can learn which window is best.

    The actual window selection happens at inference time using a trained
    window selection head in the model.
    """

    def select_window(self, sample: 'ChannelSample') -> int:
        """
        Return best_window as fallback.

        For learned mode, the dataset handles this specially by
        returning features for all windows. This method is only
        called as a fallback.

        Returns:
            sample.best_window (fallback for data loading)
        """
        return sample.best_window


# =============================================================================
# Strategy Enum and Registry
# =============================================================================

class SelectionStrategy(Enum):
    """
    Enumeration of available window selection strategies.

    Use this enum to specify which strategy to use when selecting windows.
    Each enum value maps to a concrete strategy implementation.

    Values:
        BOUNCE_FIRST: Use BounceFirstStrategy (current default)
        LABEL_VALIDITY: Use LabelValidityStrategy (maximize valid labels)
        BALANCED_SCORE: Use BalancedScoreStrategy (weighted combination)
        QUALITY_SCORE: Use QualityScoreStrategy (use pre-computed quality_score)
        LEARNED: Use LearnedStrategy (model learns to select)
    """
    BOUNCE_FIRST = "bounce_first"
    LABEL_VALIDITY = "label_validity"
    BALANCED_SCORE = "balanced_score"
    QUALITY_SCORE = "quality_score"
    LEARNED = "learned"


# Strategy registry mapping enum values to strategy classes
_STRATEGY_REGISTRY = {
    SelectionStrategy.BOUNCE_FIRST: BounceFirstStrategy,
    SelectionStrategy.LABEL_VALIDITY: LabelValidityStrategy,
    SelectionStrategy.BALANCED_SCORE: BalancedScoreStrategy,
    SelectionStrategy.QUALITY_SCORE: QualityScoreStrategy,
    SelectionStrategy.LEARNED: LearnedStrategy,
}


def get_strategy(
    strategy: Union[SelectionStrategy, str],
    **kwargs
) -> WindowSelectionStrategy:
    """
    Factory function to create a strategy instance.

    Args:
        strategy: Which strategy to create (SelectionStrategy enum or string name)
        **kwargs: Additional arguments to pass to strategy constructor
                 (e.g., target_tf='daily' for LabelValidityStrategy)

    Returns:
        Instance of the requested strategy

    Raises:
        ValueError: If strategy is not recognized

    Example:
        ```python
        # Get default bounce-first strategy
        strategy = get_strategy(SelectionStrategy.BOUNCE_FIRST)

        # Get label validity strategy for specific TF
        strategy = get_strategy(
            SelectionStrategy.LABEL_VALIDITY,
            target_tf='daily'
        )

        # Get learned strategy (for model-based selection)
        strategy = get_strategy(SelectionStrategy.LEARNED)

        # Can also use string names
        strategy = get_strategy('bounce_first')
        ```
    """
    # Convert string to enum if needed
    if isinstance(strategy, str):
        try:
            strategy = SelectionStrategy(strategy)
        except ValueError:
            raise ValueError(
                f"Unknown strategy: '{strategy}'. "
                f"Available strategies: {[s.value for s in SelectionStrategy]}"
            )

    if strategy not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Available strategies: {list(_STRATEGY_REGISTRY.keys())}"
        )

    strategy_class = _STRATEGY_REGISTRY[strategy]
    return strategy_class(**kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================

def select_best_window_bounce_first(sample: 'ChannelSample') -> int:
    """
    Convenience function for bounce-first selection (default).

    Args:
        sample: ChannelSample object

    Returns:
        Best window size based on bounce quality
    """
    strategy = BounceFirstStrategy()
    return strategy.select_window(sample)


def select_best_window_by_labels(
    sample: 'ChannelSample',
    target_tf: Optional[str] = None
) -> int:
    """
    Convenience function for label validity selection.

    Args:
        sample: ChannelSample object
        target_tf: Optional target timeframe to prioritize

    Returns:
        Best window size based on label validity
    """
    strategy = LabelValidityStrategy(target_tf=target_tf)
    return strategy.select_window(sample)


def select_best_window_balanced(
    sample: 'ChannelSample',
    target_tf: Optional[str] = None
) -> int:
    """
    Convenience function for balanced score selection.

    Args:
        sample: ChannelSample object
        target_tf: Optional target timeframe to prioritize

    Returns:
        Best window size based on balanced criteria
    """
    strategy = BalancedScoreStrategy(target_tf=target_tf)
    return strategy.select_window(sample)
