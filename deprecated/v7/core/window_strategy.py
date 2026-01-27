"""
Window Selection Strategy Framework

This module provides a flexible, extensible framework for selecting the best window
size in a multi-window channel detection system. The framework allows different
strategies to be used based on different criteria (bounce quality, label validity,
balanced scoring, etc.).

The system is designed for v10+ multi-window architecture where channels are detected
at multiple window sizes and the best one must be selected for training.

Architecture:
    - WindowSelectionStrategy: Protocol defining the strategy interface
    - Concrete strategy implementations (BounceFirstStrategy, LabelValidityStrategy, etc.)
    - SelectionStrategy: Enum for configuration
    - get_strategy(): Factory function for creating strategy instances

Usage Example:
    ```python
    from v7.core.window_strategy import get_strategy, SelectionStrategy

    # Get a strategy
    strategy = get_strategy(SelectionStrategy.BOUNCE_FIRST)

    # Select best window
    best_window, confidence = strategy.select_window(
        channels={50: channel_50, 100: channel_100},
        labels_per_window={50: labels_50, 100: labels_100}
    )
    ```

Author: Claude Sonnet 4.5
Date: 2026-01-06
"""

from typing import Dict, Optional, Tuple, Protocol
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Import types from existing modules
# Use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core.channel import Channel
    from ..training.labels import ChannelLabels


# =============================================================================
# Core Protocol
# =============================================================================

class WindowSelectionStrategy(Protocol):
    """
    Protocol defining the interface for window selection strategies.

    All concrete strategies must implement this protocol by providing
    a select_window() method that takes channels and labels and returns
    the best window size with a confidence score.
    """

    def select_window(
        self,
        channels: Dict[int, 'Channel'],
        labels_per_window: Optional[Dict[int, Dict[str, Optional['ChannelLabels']]]] = None
    ) -> Tuple[Optional[int], float]:
        """
        Select the best window size from available channels.

        Args:
            channels: Dict mapping window_size -> Channel object.
                     May contain invalid channels (valid=False).
            labels_per_window: Optional dict mapping window_size -> {tf_name -> ChannelLabels}.
                              Used by strategies that consider label validity.

        Returns:
            Tuple of (best_window_size, confidence_score):
                - best_window_size: The selected window size (int), or None if no valid windows
                - confidence_score: Confidence in the selection (0.0-1.0), where:
                    * 1.0 = very confident (clear winner)
                    * 0.5 = moderate confidence (close tie)
                    * 0.0 = no confidence (no valid windows or all equally bad)

        Edge Cases:
            - If channels is empty: returns (None, 0.0)
            - If all channels are invalid: returns (None, 0.0)
            - If tie between multiple windows: returns smallest window, confidence < 1.0
        """
        ...


# =============================================================================
# Concrete Strategy Implementations
# =============================================================================

@dataclass
class BounceFirstStrategy:
    """
    Select window based on bounce quality (current v7-v9 default).

    Selection criteria (in priority order):
        1. Most bounces (bounce_count) - primary criterion
        2. Best linear fit (r_squared) - tiebreaker

    This strategy prioritizes channels that demonstrate clear oscillation
    behavior (high bounce_count) with good linear trend quality (high r_squared).

    Confidence scoring:
        - Based on relative dominance of winner vs runner-up
        - High confidence when winner has significantly more bounces
        - Lower confidence when bounces are tied and r_squared is close

    Note:
        This is the current production strategy used in v7-v9.
        It matches the behavior of select_best_channel() in channel.py.
    """

    def select_window(
        self,
        channels: Dict[int, 'Channel'],
        labels_per_window: Optional[Dict[int, Dict[str, Optional['ChannelLabels']]]] = None
    ) -> Tuple[Optional[int], float]:
        """
        Select window with most bounces, using r_squared as tiebreaker.

        Returns:
            (window_size, confidence) where confidence is based on gap to runner-up
        """
        if not channels:
            return None, 0.0

        # Filter to valid channels only
        valid_channels = {w: ch for w, ch in channels.items() if ch.valid}

        if not valid_channels:
            return None, 0.0

        # Sort by (bounce_count, r_squared, -window) descending
        # The -w ensures smaller windows win on full ties (because -10 > -50 when sorting descending)
        sorted_windows = sorted(
            valid_channels.keys(),
            key=lambda w: (valid_channels[w].bounce_count, valid_channels[w].r_squared, -w),
            reverse=True
        )

        best_window = sorted_windows[0]
        best_channel = valid_channels[best_window]

        # Calculate confidence based on gap to runner-up
        if len(sorted_windows) == 1:
            # Only one valid channel - maximum confidence
            confidence = 1.0
        else:
            runner_up_window = sorted_windows[1]
            runner_up = valid_channels[runner_up_window]

            # Confidence based on bounce_count gap
            bounce_gap = best_channel.bounce_count - runner_up.bounce_count

            if bounce_gap > 2:
                # Clear winner (3+ more bounces)
                confidence = 1.0
            elif bounce_gap > 0:
                # Moderate winner (1-2 more bounces)
                confidence = 0.8
            else:
                # Tied on bounces, won on r_squared
                r2_gap = best_channel.r_squared - runner_up.r_squared
                if r2_gap > 0.1:
                    confidence = 0.7
                elif r2_gap > 0.05:
                    confidence = 0.6
                else:
                    confidence = 0.5  # Very close tie

        return best_window, confidence


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

    Confidence scoring:
        - Based on percentage of valid labels (# valid / # total TFs)
        - High confidence when winner has many valid labels
        - Lower confidence when all windows have few valid labels

    Note:
        This matches the behavior of select_best_window_by_labels() in labels.py.
        Requires labels_per_window to be provided.
    """

    def select_window(
        self,
        channels: Dict[int, 'Channel'],
        labels_per_window: Optional[Dict[int, Dict[str, Optional['ChannelLabels']]]] = None
    ) -> Tuple[Optional[int], float]:
        """
        Select window with most valid labels across timeframes.

        Returns:
            (window_size, confidence) where confidence is based on label validity rate
        """
        if not channels:
            return None, 0.0

        if labels_per_window is None:
            # Fall back to bounce-first if no labels provided
            fallback = BounceFirstStrategy()
            return fallback.select_window(channels, labels_per_window)

        # Filter to windows that exist in both dicts
        valid_windows = set(channels.keys()) & set(labels_per_window.keys())

        if not valid_windows:
            return None, 0.0

        # Count valid labels per window
        validity_counts = {}
        total_tfs = 0

        for window_size in valid_windows:
            tf_labels = labels_per_window[window_size]
            valid_count = sum(1 for labels in tf_labels.values() if labels is not None)
            validity_counts[window_size] = valid_count
            total_tfs = max(total_tfs, len(tf_labels))

        if total_tfs == 0:
            return None, 0.0

        # Find window(s) with most valid labels
        max_valid = max(validity_counts.values())

        if max_valid == 0:
            return None, 0.0

        # Among windows with max valid labels, prefer smallest
        best_windows = [w for w, count in validity_counts.items() if count == max_valid]
        best_window = min(best_windows)

        # Calculate confidence based on validity rate
        validity_rate = max_valid / total_tfs

        # Confidence scoring:
        # - 100% valid -> confidence 1.0
        # - 80%+ valid -> confidence 0.9
        # - 60%+ valid -> confidence 0.8
        # - etc.
        if validity_rate >= 0.95:
            confidence = 1.0
        elif validity_rate >= 0.8:
            confidence = 0.9
        elif validity_rate >= 0.6:
            confidence = 0.8
        elif validity_rate >= 0.4:
            confidence = 0.7
        elif validity_rate >= 0.2:
            confidence = 0.6
        else:
            confidence = 0.5

        return best_window, confidence


@dataclass
class BalancedScoreStrategy:
    """
    Select window using weighted combination of bounce quality and label validity.

    Composite score:
        score = (bounce_weight × normalized_bounce_score) +
                (label_weight × normalized_label_score)

    Default weights:
        - bounce_weight: 0.4 (40% weight to channel quality)
        - label_weight: 0.6 (60% weight to label validity)

    Normalized scores:
        - bounce_score: bounce_count / max_bounce_count across all windows
        - label_score: valid_label_count / total_tf_count

    This strategy balances channel quality (bounces) with downstream
    usefulness (valid labels). The weights can be tuned based on your
    priorities.

    Confidence scoring:
        - Based on gap between winner's score and runner-up's score
        - High confidence when winner clearly dominates
        - Lower confidence when scores are close

    Attributes:
        bounce_weight: Weight for bounce component (0.0-1.0)
        label_weight: Weight for label validity component (0.0-1.0)

    Note:
        Weights should sum to 1.0 for interpretability, but this is not enforced.
    """

    bounce_weight: float = 0.4
    label_weight: float = 0.6

    def __post_init__(self):
        """Validate weights."""
        if self.bounce_weight < 0 or self.label_weight < 0:
            raise ValueError("Weights must be non-negative")

        if abs(self.bounce_weight + self.label_weight - 1.0) > 1e-6:
            # Allow slight numerical error but warn about large deviations
            if abs(self.bounce_weight + self.label_weight - 1.0) > 0.01:
                import warnings
                warnings.warn(
                    f"Weights sum to {self.bounce_weight + self.label_weight:.3f}, "
                    f"not 1.0. This may affect score interpretability."
                )

    def select_window(
        self,
        channels: Dict[int, 'Channel'],
        labels_per_window: Optional[Dict[int, Dict[str, Optional['ChannelLabels']]]] = None
    ) -> Tuple[Optional[int], float]:
        """
        Select window using weighted combination of bounce and label scores.

        Returns:
            (window_size, confidence) where confidence is based on score gap
        """
        if not channels:
            return None, 0.0

        # Filter to valid channels
        valid_channels = {w: ch for w, ch in channels.items() if ch.valid}

        if not valid_channels:
            return None, 0.0

        # Calculate bounce scores (normalized)
        max_bounces = max(ch.bounce_count for ch in valid_channels.values())

        if max_bounces == 0:
            # All channels have 0 bounces - fall back to r_squared only
            bounce_scores = {w: ch.r_squared for w, ch in valid_channels.items()}
            max_bounce_score = max(bounce_scores.values()) if bounce_scores else 1.0
            if max_bounce_score > 0:
                bounce_scores = {w: s / max_bounce_score for w, s in bounce_scores.items()}
            else:
                bounce_scores = {w: 0.0 for w in valid_channels}
        else:
            # Normalize by max bounces
            bounce_scores = {
                w: ch.bounce_count / max_bounces
                for w, ch in valid_channels.items()
            }

        # Calculate label validity scores (normalized)
        label_scores = {}
        total_tfs = 0

        if labels_per_window is not None:
            for window_size in valid_channels.keys():
                if window_size in labels_per_window:
                    tf_labels = labels_per_window[window_size]
                    valid_count = sum(1 for labels in tf_labels.values() if labels is not None)
                    total_tfs = max(total_tfs, len(tf_labels))
                    label_scores[window_size] = valid_count
                else:
                    label_scores[window_size] = 0

            # Normalize label scores
            if total_tfs > 0:
                label_scores = {w: count / total_tfs for w, count in label_scores.items()}
            else:
                label_scores = {w: 0.0 for w in valid_channels}
        else:
            # No labels available - set all to 0
            label_scores = {w: 0.0 for w in valid_channels}

        # Calculate composite scores
        composite_scores = {}
        for window_size in valid_channels.keys():
            bounce_component = self.bounce_weight * bounce_scores.get(window_size, 0.0)
            label_component = self.label_weight * label_scores.get(window_size, 0.0)
            composite_scores[window_size] = bounce_component + label_component

        # Find best window
        best_window = max(composite_scores.keys(), key=lambda w: composite_scores[w])
        best_score = composite_scores[best_window]

        # Calculate confidence based on score gap
        sorted_scores = sorted(composite_scores.values(), reverse=True)

        if len(sorted_scores) == 1:
            confidence = 1.0
        else:
            runner_up_score = sorted_scores[1]

            if best_score == 0.0:
                # All scores are 0 - no confidence
                confidence = 0.0
            else:
                # Confidence based on relative gap
                score_gap = best_score - runner_up_score
                relative_gap = score_gap / best_score

                # Map relative gap to confidence
                # 50%+ gap -> confidence 1.0
                # 30%+ gap -> confidence 0.9
                # 20%+ gap -> confidence 0.8
                # 10%+ gap -> confidence 0.7
                # 5%+ gap -> confidence 0.6
                # <5% gap -> confidence 0.5
                if relative_gap >= 0.5:
                    confidence = 1.0
                elif relative_gap >= 0.3:
                    confidence = 0.9
                elif relative_gap >= 0.2:
                    confidence = 0.8
                elif relative_gap >= 0.1:
                    confidence = 0.7
                elif relative_gap >= 0.05:
                    confidence = 0.6
                else:
                    confidence = 0.5

        return best_window, confidence


@dataclass
class QualityScoreStrategy:
    """
    Select window based on channel quality_score field.

    Selection criteria:
        1. Highest quality_score (from calculate_channel_quality_score)
        2. On tie: prefer smaller window size

    The quality_score incorporates:
        - Alternations (bounces)
        - Alternation ratio (bounce cleanliness)
        - Optional false break resilience

    This strategy uses the pre-computed quality score from the Channel
    object, which already combines multiple quality metrics.

    Confidence scoring:
        - Based on relative gap between winner and runner-up scores
        - High confidence when winner has significantly higher score
        - Lower confidence when scores are close

    Note:
        This strategy is useful when quality_score has been carefully
        tuned to capture all relevant quality metrics. It's simpler
        than BalancedScoreStrategy but less flexible.
    """

    def select_window(
        self,
        channels: Dict[int, 'Channel'],
        labels_per_window: Optional[Dict[int, Dict[str, Optional['ChannelLabels']]]] = None
    ) -> Tuple[Optional[int], float]:
        """
        Select window with highest quality_score.

        Returns:
            (window_size, confidence) where confidence is based on score gap
        """
        if not channels:
            return None, 0.0

        # Filter to valid channels
        valid_channels = {w: ch for w, ch in channels.items() if ch.valid}

        if not valid_channels:
            return None, 0.0

        # Sort by quality_score (descending), then by window size (ascending for ties)
        sorted_windows = sorted(
            valid_channels.keys(),
            key=lambda w: (valid_channels[w].quality_score, -w),
            reverse=True
        )

        best_window = sorted_windows[0]
        best_score = valid_channels[best_window].quality_score

        # Calculate confidence based on gap to runner-up
        if len(sorted_windows) == 1:
            confidence = 1.0
        else:
            runner_up_window = sorted_windows[1]
            runner_up_score = valid_channels[runner_up_window].quality_score

            if best_score == 0.0:
                # All scores are 0 - no confidence
                confidence = 0.0
            else:
                # Confidence based on relative gap
                score_gap = best_score - runner_up_score
                relative_gap = score_gap / best_score

                # Map relative gap to confidence (same as BalancedScoreStrategy)
                if relative_gap >= 0.5:
                    confidence = 1.0
                elif relative_gap >= 0.3:
                    confidence = 0.9
                elif relative_gap >= 0.2:
                    confidence = 0.8
                elif relative_gap >= 0.1:
                    confidence = 0.7
                elif relative_gap >= 0.05:
                    confidence = 0.6
                else:
                    confidence = 0.5

        return best_window, confidence


# =============================================================================
# Strategy Enum and Registry
# =============================================================================

class SelectionStrategy(Enum):
    """
    Enumeration of available window selection strategies.

    Use this enum to specify which strategy to use when selecting windows.
    Each enum value maps to a concrete strategy implementation.

    Values:
        BOUNCE_FIRST: Use BounceFirstStrategy (current v7-v9 default)
        LABEL_VALIDITY: Use LabelValidityStrategy (maximize valid labels)
        BALANCED_SCORE: Use BalancedScoreStrategy (40% bounce, 60% labels)
        QUALITY_SCORE: Use QualityScoreStrategy (use pre-computed quality_score)
    """
    BOUNCE_FIRST = "bounce_first"
    LABEL_VALIDITY = "label_validity"
    BALANCED_SCORE = "balanced_score"
    QUALITY_SCORE = "quality_score"


# Strategy registry mapping enum values to strategy classes
_STRATEGY_REGISTRY = {
    SelectionStrategy.BOUNCE_FIRST: BounceFirstStrategy,
    SelectionStrategy.LABEL_VALIDITY: LabelValidityStrategy,
    SelectionStrategy.BALANCED_SCORE: BalancedScoreStrategy,
    SelectionStrategy.QUALITY_SCORE: QualityScoreStrategy,
}


def get_strategy(
    strategy: SelectionStrategy,
    **kwargs
) -> WindowSelectionStrategy:
    """
    Factory function to create a strategy instance.

    Args:
        strategy: Which strategy to create (from SelectionStrategy enum)
        **kwargs: Additional arguments to pass to strategy constructor
                 (e.g., bounce_weight=0.3, label_weight=0.7 for BalancedScoreStrategy)

    Returns:
        Instance of the requested strategy

    Raises:
        ValueError: If strategy is not recognized

    Example:
        ```python
        # Get default bounce-first strategy
        strategy = get_strategy(SelectionStrategy.BOUNCE_FIRST)

        # Get balanced strategy with custom weights
        strategy = get_strategy(
            SelectionStrategy.BALANCED_SCORE,
            bounce_weight=0.3,
            label_weight=0.7
        )

        # Get label validity strategy
        strategy = get_strategy(SelectionStrategy.LABEL_VALIDITY)
        ```
    """
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

def select_best_window_bounce_first(
    channels: Dict[int, 'Channel']
) -> Tuple[Optional[int], float]:
    """
    Convenience function for bounce-first selection (v7-v9 default).

    This is equivalent to calling:
        strategy = get_strategy(SelectionStrategy.BOUNCE_FIRST)
        return strategy.select_window(channels)

    Args:
        channels: Dict mapping window_size -> Channel object

    Returns:
        Tuple of (best_window_size, confidence_score)
    """
    strategy = BounceFirstStrategy()
    return strategy.select_window(channels)


def select_best_window_by_labels(
    labels_per_window: Dict[int, Dict[str, Optional['ChannelLabels']]],
    channels: Optional[Dict[int, 'Channel']] = None
) -> Tuple[Optional[int], float]:
    """
    Convenience function for label validity selection.

    This is equivalent to calling:
        strategy = get_strategy(SelectionStrategy.LABEL_VALIDITY)
        return strategy.select_window(channels, labels_per_window)

    Args:
        labels_per_window: Dict mapping window_size -> {tf_name -> ChannelLabels}
        channels: Optional dict mapping window_size -> Channel (required by protocol)

    Returns:
        Tuple of (best_window_size, confidence_score)

    Note:
        If channels is not provided, creates a dummy dict with all valid channels.
        This maintains compatibility with the existing select_best_window_by_labels()
        function in labels.py.
    """
    # Create dummy channels dict if not provided (for backward compatibility)
    if channels is None:
        from ..core.channel import Channel, Direction
        import numpy as np

        # Create minimal valid channels
        channels = {}
        for window_size in labels_per_window.keys():
            # Minimal valid channel (just needs valid=True)
            channels[window_size] = Channel(
                valid=True,
                direction=Direction.SIDEWAYS,
                slope=0.0,
                intercept=100.0,
                r_squared=0.5,
                std_dev=1.0,
                upper_line=np.array([100.0]),
                lower_line=np.array([100.0]),
                center_line=np.array([100.0]),
                touches=[],
                complete_cycles=0,
                bounce_count=0,
                width_pct=1.0,
                window=window_size
            )

    strategy = LabelValidityStrategy()
    return strategy.select_window(channels, labels_per_window)


def select_best_window_balanced(
    channels: Dict[int, 'Channel'],
    labels_per_window: Optional[Dict[int, Dict[str, Optional['ChannelLabels']]]] = None,
    bounce_weight: float = 0.4,
    label_weight: float = 0.6
) -> Tuple[Optional[int], float]:
    """
    Convenience function for balanced score selection.

    Args:
        channels: Dict mapping window_size -> Channel object
        labels_per_window: Optional dict mapping window_size -> {tf_name -> ChannelLabels}
        bounce_weight: Weight for bounce component (default 0.4)
        label_weight: Weight for label validity component (default 0.6)

    Returns:
        Tuple of (best_window_size, confidence_score)
    """
    strategy = BalancedScoreStrategy(
        bounce_weight=bounce_weight,
        label_weight=label_weight
    )
    return strategy.select_window(channels, labels_per_window)
