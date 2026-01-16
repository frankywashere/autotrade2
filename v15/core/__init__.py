"""
V15 Core Module - Channel detection and window selection utilities.
"""

from .window_strategy import (
    SelectionStrategy,
    WindowSelectionStrategy,
    BounceFirstStrategy,
    LabelValidityStrategy,
    BalancedScoreStrategy,
    QualityScoreStrategy,
    LearnedStrategy,
    get_strategy,
    select_best_window_bounce_first,
    select_best_window_by_labels,
    select_best_window_balanced,
)

__all__ = [
    'SelectionStrategy',
    'WindowSelectionStrategy',
    'BounceFirstStrategy',
    'LabelValidityStrategy',
    'BalancedScoreStrategy',
    'QualityScoreStrategy',
    'LearnedStrategy',
    'get_strategy',
    'select_best_window_bounce_first',
    'select_best_window_by_labels',
    'select_best_window_balanced',
]
