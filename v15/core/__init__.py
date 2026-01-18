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

from .break_scanner import (
    BreakScannerError,
    InsufficientDataError,
    BreakDirection,
    ExitEvent,
    BreakResult,
    project_channel_bounds,
    scan_for_break,
    calculate_durability_score,
    compute_durability_from_result,
)

__all__ = [
    # Window selection
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
    # Break scanning
    'BreakScannerError',
    'InsufficientDataError',
    'BreakDirection',
    'ExitEvent',
    'BreakResult',
    'project_channel_bounds',
    'scan_for_break',
    'calculate_durability_score',
    'compute_durability_from_result',
]
