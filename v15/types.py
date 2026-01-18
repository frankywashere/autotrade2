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

    Attributes:
        duration_bars: Number of bars until permanent channel break
        break_direction: Direction of break (0=DOWN, 1=UP)
        break_trigger_tf: Encoded timeframe that triggered break (0-9)
        new_channel_direction: Direction after break (0=BEAR, 1=SIDEWAYS, 2=BULL)
        permanent_break: Whether a permanent break occurred
        break_return: Return percentage at time of break
        timeframe: The timeframe this label was computed for

    Break scan fields (TSLA):
        bars_to_first_break: Bars from channel end until first bar outside bounds
        first_break_direction: Direction of first exit (0=DOWN, 1=UP)
        break_magnitude: How far outside bounds (in std devs)
        bars_outside: Total bars spent outside bounds
        returned_to_channel: Whether price came back inside
        bounces_after_return: If returned, how many bounces before final exit
        channel_continued: Whether original channel pattern continued after return

    Break scan fields (SPY - mirrored):
        spy_bars_to_first_break: Bars from channel end until first bar outside bounds
        spy_first_break_direction: Direction of first exit (0=DOWN, 1=UP)
        spy_break_magnitude: How far outside bounds (in std devs)
        spy_bars_outside: Total bars spent outside bounds
        spy_returned_to_channel: Whether price came back inside
        spy_bounces_after_return: If returned, how many bounces before final exit
        spy_channel_continued: Whether original channel pattern continued after return

    Validity flags indicate whether each label component is valid/usable:
        duration_valid: True if duration_bars is meaningful
        direction_valid: True if break_direction is meaningful
        trigger_tf_valid: True if break_trigger_tf is meaningful
        new_channel_valid: True if new_channel_direction is meaningful
        break_scan_valid: True if forward scan was performed
    """
    # Core label values
    duration_bars: int = 0
    break_direction: int = 0      # 0=DOWN, 1=UP
    break_trigger_tf: int = 0     # encoded 0-9 (index into TIMEFRAMES)
    new_channel_direction: int = 1  # 0=BEAR, 1=SIDEWAYS, 2=BULL
    permanent_break: bool = False
    break_return: float = 0.0     # Return at break point
    timeframe: str = ""           # Timeframe this label is for

    # Break scan fields - TSLA dynamics
    bars_to_first_break: int = 0       # Bars from channel end until first bar outside bounds
    first_break_direction: int = 0     # 0=DOWN, 1=UP (direction of first exit)
    break_magnitude: float = 0.0       # How far outside bounds (in std devs)
    bars_outside: int = 0              # Total bars spent outside bounds
    returned_to_channel: bool = False  # Whether price came back inside
    bounces_after_return: int = 0      # If returned, how many bounces before final exit
    channel_continued: bool = False    # Whether original channel pattern continued after return

    # Break scan fields - SPY dynamics (mirrored)
    spy_bars_to_first_break: int = 0       # Bars from channel end until first bar outside bounds
    spy_first_break_direction: int = 0     # 0=DOWN, 1=UP (direction of first exit)
    spy_break_magnitude: float = 0.0       # How far outside bounds (in std devs)
    spy_bars_outside: int = 0              # Total bars spent outside bounds
    spy_returned_to_channel: bool = False  # Whether price came back inside
    spy_bounces_after_return: int = 0      # If returned, how many bounces before final exit
    spy_channel_continued: bool = False    # Whether original channel pattern continued after return

    # Validity flags
    duration_valid: bool = False
    direction_valid: bool = False
    trigger_tf_valid: bool = False
    new_channel_valid: bool = False
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

# New channel direction encoding
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

    Attributes:
        break_direction_aligned: Did TSLA and SPY break in the same direction?
        tsla_broke_first: Did TSLA break before SPY?
        spy_broke_first: Did SPY break before TSLA?
        break_lag_bars: Number of bars between TSLA and SPY breaks (absolute)
        magnitude_spread: Difference in break magnitudes (TSLA - SPY return)
        both_returned: Did both assets return to their channels?
        both_permanent: Were both breaks permanent (no return)?
        return_pattern_aligned: Did return behavior match (both returned or both permanent)?
        continuation_aligned: Did continuation pattern match (same new channel direction)?
        cross_valid: True if both TSLA and SPY had valid breaks for comparison
    """
    break_direction_aligned: bool = False
    tsla_broke_first: bool = False
    spy_broke_first: bool = False
    break_lag_bars: int = 0
    magnitude_spread: float = 0.0
    both_returned: bool = False
    both_permanent: bool = False
    return_pattern_aligned: bool = False
    continuation_aligned: bool = False
    cross_valid: bool = False
