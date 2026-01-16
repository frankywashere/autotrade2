"""
v15/types.py - Clean data structures for channel labeling system.

Simple dataclasses and constants. No complex logic.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

TIMEFRAMES = [
    '5min', '15min', '30min', '1h', '2h', '3h', '4h',
    'daily', 'weekly', 'monthly', '3month'
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
    '3month': 4914,   # ~63 trading days * 78
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
    '3month': 12,
}

# Return threshold percentage for significant moves per timeframe
TF_RETURN_THRESHOLD: Dict[str, float] = {
    '5min': 0.002,    # 0.2%
    '15min': 0.003,   # 0.3%
    '30min': 0.004,   # 0.4%
    '1h': 0.005,      # 0.5%
    '2h': 0.007,      # 0.7%
    '3h': 0.008,      # 0.8%
    '4h': 0.010,      # 1.0%
    'daily': 0.015,   # 1.5%
    'weekly': 0.030,  # 3.0%
    'monthly': 0.050, # 5.0%
    '3month': 0.080,  # 8.0%
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
        break_trigger_tf: Encoded timeframe that triggered break (0-10)
        new_channel_direction: Direction after break (0=BEAR, 1=SIDEWAYS, 2=BULL)
        permanent_break: Whether a permanent break occurred
        break_return: Return percentage at time of break
        timeframe: The timeframe this label was computed for

    Validity flags indicate whether each label component is valid/usable:
        duration_valid: True if duration_bars is meaningful
        direction_valid: True if break_direction is meaningful
        trigger_tf_valid: True if break_trigger_tf is meaningful
        new_channel_valid: True if new_channel_direction is meaningful
    """
    # Core label values
    duration_bars: int = 0
    break_direction: int = 0      # 0=DOWN, 1=UP
    break_trigger_tf: int = 0     # encoded 0-10 (index into TIMEFRAMES)
    new_channel_direction: int = 1  # 0=BEAR, 1=SIDEWAYS, 2=BULL
    permanent_break: bool = False
    break_return: float = 0.0     # Return at break point
    timeframe: str = ""           # Timeframe this label is for

    # Validity flags
    duration_valid: bool = False
    direction_valid: bool = False
    trigger_tf_valid: bool = False
    new_channel_valid: bool = False


@dataclass
class ChannelSample:
    """
    A complete sample for V15 channel prediction.

    Contains:
        - timestamp: When this sample was created
        - channel_end_idx: Index in 5min data where channels end
        - tf_features: Dict of all 8,665 features (flat, TF-prefixed)
        - labels_per_window: Labels for each window across TFs
        - bar_metadata: Partial bar completion info per TF
        - best_window: Optimal window size
    """
    timestamp: pd.Timestamp = None
    channel_end_idx: int = 0
    tf_features: Dict[str, float] = field(default_factory=dict)
    labels_per_window: Dict[int, Dict[str, Optional[ChannelLabels]]] = field(default_factory=dict)
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
