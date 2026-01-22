"""
V15 Core Module - Channel detection, window selection, and data utilities.
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

from .resample import (
    resample_ohlc,
    resample_multi_tf,
    normalize_timeframe,
    get_resample_rule,
    get_bars_per_tf,
    get_longer_timeframes,
    get_shorter_timeframes,
    validate_ohlc,
    TIMEFRAMES,
    RESAMPLE_RULES,
    BARS_PER_TF,
    TF_ALIASES,
    BarMetadata,
    ResamplingError,
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
    # OHLC Resampling (canonical implementation)
    'resample_ohlc',
    'resample_multi_tf',
    'normalize_timeframe',
    'get_resample_rule',
    'get_bars_per_tf',
    'get_longer_timeframes',
    'get_shorter_timeframes',
    'validate_ohlc',
    'TIMEFRAMES',
    'RESAMPLE_RULES',
    'BARS_PER_TF',
    'TF_ALIASES',
    'BarMetadata',
    'ResamplingError',
]
