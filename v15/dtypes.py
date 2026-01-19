"""
v15/types.py - Clean data structures for channel labeling system.

Simple dataclasses and constants. No complex logic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

# 10 TFs - no 3month due to data limitations
TIMEFRAMES = [
    '5min', '15min', '30min', '1h', '2h', '3h', '4h',
    'daily', 'weekly', 'monthly'
]

STANDARD_WINDOWS = [10, 20, 30, 40, 50, 60, 70, 80]

# How many 5min bars per timeframe bar
BARS_PER_TF: Dict[str, int] = {
    '5min': 1,
    '15min': 3,
    '30min': 6,
    '1h': 12,
    '2h': 24,
    '3h': 36,
    '4h': 48,
    'daily': 78,      # 6.5 hours * 12
    'weekly': 390,    # 5 days * 78
    'monthly': 1638,  # ~21 trading days * 78
}

# Maximum bars to scan forward for break detection per timeframe
TF_MAX_SCAN: Dict[str, int] = {
    '5min': 500,
    '15min': 400,
    '30min': 350,
    '1h': 300,
    '2h': 250,
    '3h': 200,
    '4h': 150,
    'daily': 100,
    'weekly': 52,
    'monthly': 24,
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ChannelLabels:
    """
    Labels for a single channel at a specific window size.

    PREDICTION TARGETS (what we want the model to predict):
        duration_bars: PRIMARY - Number of bars until channel breaks
        next_channel_direction: SECONDARY - Direction after break (0=BEAR, 1=SIDEWAYS, 2=BULL)
        permanent_break: SECONDARY - Whether the break sticks (True/False)

    BREAK SCAN FEATURES (tracked for model input, NOT prediction targets):
        FIRST BREAK (initial break, may be false break that returns):
            break_direction: Which bound was breached FIRST (UP=1, DOWN=0)
            break_magnitude: How far outside bounds on FIRST break (in std devs)
            bars_to_first_break: When FIRST break occurred (bars from channel end)
            returned_to_channel: Did price come back inside bounds after first break
            bounces_after_return: Count of false breaks before final exit
            channel_continued: Did the original channel pattern resume after return

        PERMANENT BREAK (final/lasting break, may differ from first):
            permanent_break_direction: Direction of FINAL break (-1=none, 0=DOWN, 1=UP)
            permanent_break_magnitude: Magnitude of permanent break (in std devs)
            bars_to_permanent_break: When permanent break occurred (-1 if none)

    VALIDITY FLAGS (indicate whether each component is valid/usable):
        duration_valid: True if duration_bars is meaningful
        direction_valid: True if break_direction is meaningful
        next_channel_valid: True if next_channel_direction is meaningful
        break_scan_valid: True if forward scan was performed
    """
    # -------------------------------------------------------------------------
    # PREDICTION TARGETS - What we want the model to predict
    # -------------------------------------------------------------------------
    # PRIMARY: How long until channel breaks
    duration_bars: int = 0

    # SECONDARY: Direction of next channel after break (0=BEAR, 1=SIDEWAYS, 2=BULL)
    next_channel_direction: int = 1

    # SECONDARY: Whether the break sticks (no return to channel)
    permanent_break: bool = False

    # Metadata (not a prediction target)
    timeframe: str = ""  # Timeframe this label is for

    # -------------------------------------------------------------------------
    # BREAK SCAN FEATURES - For model input, NOT prediction targets
    # These describe what happened during/after the break for feature engineering
    # -------------------------------------------------------------------------
    # TSLA FIRST break dynamics (initial break, may be false break)
    break_direction: int = 0           # Which bound was breached FIRST (0=DOWN, 1=UP)
    break_magnitude: float = 0.0       # How far outside bounds on FIRST break (in std devs)
    bars_to_first_break: int = 0       # When FIRST break occurred (bars from channel end)
    returned_to_channel: bool = False  # Did price come back inside bounds after first break
    bounces_after_return: int = 0      # Count of false breaks before final exit
    channel_continued: bool = False    # Did original channel pattern resume after return

    # TSLA PERMANENT break dynamics (final/lasting break, may differ from first)
    permanent_break_direction: int = -1     # Direction of FINAL break (-1=none, 0=DOWN, 1=UP)
    permanent_break_magnitude: float = 0.0  # Magnitude of permanent break (in std devs)
    bars_to_permanent_break: int = -1       # When permanent break occurred (-1 if none)

    # SPY FIRST break dynamics (mirrored features for cross-asset analysis)
    spy_break_direction: int = 0           # Which bound was breached FIRST (0=DOWN, 1=UP)
    spy_break_magnitude: float = 0.0       # How far outside bounds on FIRST break (in std devs)
    spy_bars_to_first_break: int = 0       # When FIRST break occurred (bars from channel end)
    spy_returned_to_channel: bool = False  # Did price come back inside bounds after first break
    spy_bounces_after_return: int = 0      # Count of false breaks before final exit
    spy_channel_continued: bool = False    # Did original channel pattern resume after return

    # SPY PERMANENT break dynamics (final/lasting break, may differ from first)
    spy_permanent_break_direction: int = -1     # Direction of FINAL break (-1=none, 0=DOWN, 1=UP)
    spy_permanent_break_magnitude: float = 0.0  # Magnitude of permanent break (in std devs)
    spy_bars_to_permanent_break: int = -1       # When permanent break occurred (-1 if none)

    # -------------------------------------------------------------------------
    # VALIDITY FLAGS - Indicate whether each label component is valid/usable
    # -------------------------------------------------------------------------
    duration_valid: bool = False       # True if duration_bars is meaningful
    direction_valid: bool = False      # True if break_direction is meaningful
    next_channel_valid: bool = False   # True if next_channel_direction is meaningful
    break_scan_valid: bool = False     # True if forward scan was performed


@dataclass
class ChannelSample:
    """
    A complete sample for V15 channel prediction.

    Contains:
        - timestamp: When this sample was created
        - channel_end_idx: Index in 5min data where channels end
        - tf_features: Dict of all 8,665 features (flat, TF-prefixed)
        - labels_per_window: Labels for each window/asset/TF combination
          Structure: {window: {'tsla': {tf: ChannelLabels}, 'spy': {tf: ChannelLabels}}}
        - bar_metadata: Partial bar completion info per TF
        - best_window: Optimal window size
    """
    timestamp: pd.Timestamp = None
    channel_end_idx: int = 0
    tf_features: Dict[str, float] = field(default_factory=dict)
    labels_per_window: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    bar_metadata: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_window: int = 50


# =============================================================================
# HELPER CONSTANTS
# =============================================================================

# Break direction encoding
BREAK_DOWN = 0
BREAK_UP = 1

# Next channel direction encoding (for next_channel_direction field)
DIRECTION_BEAR = 0
DIRECTION_SIDEWAYS = 1
DIRECTION_BULL = 2

# Timeframe to index mapping for encoding
TF_TO_INDEX: Dict[str, int] = {tf: i for i, tf in enumerate(TIMEFRAMES)}
INDEX_TO_TF: Dict[int, str] = {i: tf for i, tf in enumerate(TIMEFRAMES)}


# =============================================================================
# CROSS-CORRELATION LABELS
# =============================================================================

@dataclass
class CrossCorrelationLabels:
    """
    Labels comparing TSLA and SPY channel break behavior for alignment patterns.

    These labels capture how the two assets' breaks relate to each other,
    useful for detecting lead/lag relationships and correlated movements.

    FIRST BREAK CROSS-CORRELATION (initial break, may be false break):
        break_direction_aligned: Did TSLA and SPY FIRST break in the same direction?
        tsla_broke_first: Did TSLA's FIRST break occur before SPY's?
        spy_broke_first: Did SPY's FIRST break occur before TSLA's?
        break_lag_bars: Bars between TSLA and SPY FIRST breaks (absolute)
        magnitude_spread: Difference in FIRST break magnitudes (TSLA - SPY)

    PERMANENT BREAK CROSS-CORRELATION (final/lasting break):
        permanent_direction_aligned: Did TSLA and SPY PERMANENT break same direction?
        tsla_permanent_first: Did TSLA achieve permanent break before SPY?
        spy_permanent_first: Did SPY achieve permanent break before TSLA?
        permanent_break_lag_bars: Bars between permanent breaks (absolute)
        permanent_magnitude_spread: Difference in permanent break magnitudes

    DIRECTION TRANSITION PATTERNS (first vs permanent):
        tsla_direction_diverged: Did TSLA's permanent direction differ from first?
        spy_direction_diverged: Did SPY's permanent direction differ from first?
        both_direction_diverged: Did both change direction from first to permanent?
        direction_divergence_aligned: Did divergence pattern match?

    RETURN/PERMANENCE PATTERNS:
        both_returned: Did both assets return to their channels after first break?
        both_permanent: Were both FIRST breaks permanent (no return)?
        return_pattern_aligned: Did return behavior match?
        continuation_aligned: Did continuation pattern match (same next channel direction)?

    VALIDITY:
        cross_valid: True if both TSLA and SPY had valid breaks for comparison
        permanent_cross_valid: True if both had valid permanent breaks
    """
    # FIRST break cross-correlation
    break_direction_aligned: bool = False
    tsla_broke_first: bool = False
    spy_broke_first: bool = False
    break_lag_bars: int = 0
    magnitude_spread: float = 0.0

    # PERMANENT break cross-correlation
    permanent_direction_aligned: bool = False
    tsla_permanent_first: bool = False
    spy_permanent_first: bool = False
    permanent_break_lag_bars: int = 0
    permanent_magnitude_spread: float = 0.0

    # Direction transition patterns (first vs permanent)
    tsla_direction_diverged: bool = False  # TSLA permanent != first break direction
    spy_direction_diverged: bool = False   # SPY permanent != first break direction
    both_direction_diverged: bool = False  # Both changed direction
    direction_divergence_aligned: bool = False  # Divergence pattern matched

    # Return/permanence patterns
    both_returned: bool = False
    both_permanent: bool = False
    return_pattern_aligned: bool = False
    continuation_aligned: bool = False

    # Validity flags
    cross_valid: bool = False
    permanent_cross_valid: bool = False  # Both had valid permanent breaks
