#!/usr/bin/env python3
"""
Python loader for V15 binary sample files.

This script loads C++ serialized samples and transforms them to match
the structure expected by the Python inspector (v15/inspector.py).

Compatible with:
- Binary format produced by v15_cpp serialization (V15SAMP format)
- Python inspector's expected sample structure

Key transformations:
- Timestamp: int64 milliseconds -> pd.Timestamp
- Labels: labels_per_window[window][tf] -> labels_per_window[window][asset][tf]
- Split combined TSLA/SPY labels into separate ChannelLabels objects
"""

import struct
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Try to import pandas for timestamp conversion
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, timestamps will be int64 milliseconds")


class ChannelLabels:
    """
    Python representation of ChannelLabels struct.

    This class represents labels for a SINGLE asset (either TSLA or SPY).
    When loaded from binary, the combined C++ labels are split into separate
    TSLA and SPY ChannelLabels objects.
    """

    def __init__(self):
        # Core prediction targets
        self.duration_bars = 0
        self.next_channel_direction = 0
        self.permanent_break = False
        self.timeframe = 0

        # Break scan features (asset-specific, no prefix in this view)
        self.break_direction = 0
        self.break_magnitude = 0.0
        self.bars_to_first_break = 0
        self.returned_to_channel = False
        self.bounces_after_return = 0
        self.round_trip_bounces = 0
        self.channel_continued = False
        self.permanent_break_direction = 0
        self.permanent_break_magnitude = 0.0
        self.bars_to_permanent_break = 0
        self.duration_to_permanent = 0
        self.avg_bars_outside = 0.0
        self.total_bars_outside = 0
        self.durability_score = 0.0
        self.first_break_returned = False
        self.exit_return_rate = 0.0
        self.exits_returned_count = 0
        self.exits_stayed_out_count = 0
        self.scan_timed_out = False
        self.bars_verified_permanent = 0

        # Exit events (lists)
        self.exit_bars = []
        self.exit_magnitudes = []
        self.exit_durations = []
        self.exit_types = []
        self.exit_returned = []

        # Source channel parameters
        self.source_channel_slope = 0.0
        self.source_channel_intercept = 0.0
        self.source_channel_std_dev = 0.0
        self.source_channel_r_squared = 0.0
        self.source_channel_direction = 0
        self.source_channel_bounce_count = 0
        self.source_channel_start_ts = 0
        self.source_channel_end_ts = 0

        # Next channel labels
        self.best_next_channel_direction = 0
        self.best_next_channel_bars_away = 0
        self.best_next_channel_duration = 0
        self.best_next_channel_r_squared = 0.0
        self.best_next_channel_bounce_count = 0
        self.shortest_next_channel_direction = 0
        self.shortest_next_channel_bars_away = 0
        self.shortest_next_channel_duration = 0
        self.small_channels_before_best = 0

        # RSI labels
        self.rsi_at_first_break = 0.0
        self.rsi_at_permanent_break = 0.0
        self.rsi_at_channel_end = 0.0
        self.rsi_overbought_at_break = False
        self.rsi_oversold_at_break = False
        self.rsi_divergence_at_break = 0
        self.rsi_trend_in_channel = 0
        self.rsi_range_in_channel = 0.0

        # Validity flags
        self.duration_valid = False
        self.direction_valid = False
        self.next_channel_valid = False
        self.break_scan_valid = False


class CombinedChannelLabels:
    """
    Internal class for reading the combined TSLA+SPY labels from binary.
    This is an intermediate representation that gets split into separate
    TSLA and SPY ChannelLabels objects.
    """

    def __init__(self):
        # Core prediction targets (shared)
        self.duration_bars = 0
        self.next_channel_direction = 0
        self.permanent_break = False
        self.timeframe = 0

        # TSLA break scan features
        self.break_direction = 0
        self.break_magnitude = 0.0
        self.bars_to_first_break = 0
        self.returned_to_channel = False
        self.bounces_after_return = 0
        self.round_trip_bounces = 0
        self.channel_continued = False
        self.permanent_break_direction = 0
        self.permanent_break_magnitude = 0.0
        self.bars_to_permanent_break = 0
        self.duration_to_permanent = 0
        self.avg_bars_outside = 0.0
        self.total_bars_outside = 0
        self.durability_score = 0.0
        self.first_break_returned = False
        self.exit_return_rate = 0.0
        self.exits_returned_count = 0
        self.exits_stayed_out_count = 0
        self.scan_timed_out = False
        self.bars_verified_permanent = 0

        # TSLA exit events
        self.exit_bars = []
        self.exit_magnitudes = []
        self.exit_durations = []
        self.exit_types = []
        self.exit_returned = []

        # SPY break scan features
        self.spy_break_direction = 0
        self.spy_break_magnitude = 0.0
        self.spy_bars_to_first_break = 0
        self.spy_returned_to_channel = False
        self.spy_bounces_after_return = 0
        self.spy_round_trip_bounces = 0
        self.spy_channel_continued = False
        self.spy_permanent_break_direction = 0
        self.spy_permanent_break_magnitude = 0.0
        self.spy_bars_to_permanent_break = 0
        self.spy_duration_to_permanent = 0
        self.spy_avg_bars_outside = 0.0
        self.spy_total_bars_outside = 0
        self.spy_durability_score = 0.0
        self.spy_first_break_returned = False
        self.spy_exit_return_rate = 0.0
        self.spy_exits_returned_count = 0
        self.spy_exits_stayed_out_count = 0
        self.spy_scan_timed_out = False
        self.spy_bars_verified_permanent = 0

        # SPY exit events
        self.spy_exit_bars = []
        self.spy_exit_magnitudes = []
        self.spy_exit_durations = []
        self.spy_exit_types = []
        self.spy_exit_returned = []

        # TSLA source channel parameters
        self.source_channel_slope = 0.0
        self.source_channel_intercept = 0.0
        self.source_channel_std_dev = 0.0
        self.source_channel_r_squared = 0.0
        self.source_channel_direction = 0
        self.source_channel_bounce_count = 0
        self.source_channel_start_ts = 0
        self.source_channel_end_ts = 0

        # SPY source channel parameters
        self.spy_source_channel_slope = 0.0
        self.spy_source_channel_intercept = 0.0
        self.spy_source_channel_std_dev = 0.0
        self.spy_source_channel_r_squared = 0.0
        self.spy_source_channel_direction = 0
        self.spy_source_channel_bounce_count = 0
        self.spy_source_channel_start_ts = 0
        self.spy_source_channel_end_ts = 0

        # TSLA next channel labels
        self.best_next_channel_direction = 0
        self.best_next_channel_bars_away = 0
        self.best_next_channel_duration = 0
        self.best_next_channel_r_squared = 0.0
        self.best_next_channel_bounce_count = 0
        self.shortest_next_channel_direction = 0
        self.shortest_next_channel_bars_away = 0
        self.shortest_next_channel_duration = 0
        self.small_channels_before_best = 0

        # SPY next channel labels
        self.spy_best_next_channel_direction = 0
        self.spy_best_next_channel_bars_away = 0
        self.spy_best_next_channel_duration = 0
        self.spy_best_next_channel_r_squared = 0.0
        self.spy_best_next_channel_bounce_count = 0
        self.spy_shortest_next_channel_direction = 0
        self.spy_shortest_next_channel_bars_away = 0
        self.spy_shortest_next_channel_duration = 0
        self.spy_small_channels_before_best = 0

        # TSLA RSI labels
        self.rsi_at_first_break = 0.0
        self.rsi_at_permanent_break = 0.0
        self.rsi_at_channel_end = 0.0
        self.rsi_overbought_at_break = False
        self.rsi_oversold_at_break = False
        self.rsi_divergence_at_break = 0
        self.rsi_trend_in_channel = 0
        self.rsi_range_in_channel = 0.0

        # SPY RSI labels
        self.spy_rsi_at_first_break = 0.0
        self.spy_rsi_at_permanent_break = 0.0
        self.spy_rsi_at_channel_end = 0.0
        self.spy_rsi_overbought_at_break = False
        self.spy_rsi_oversold_at_break = False
        self.spy_rsi_divergence_at_break = 0
        self.spy_rsi_trend_in_channel = 0
        self.spy_rsi_range_in_channel = 0.0

        # Validity flags (shared)
        self.duration_valid = False
        self.direction_valid = False
        self.next_channel_valid = False
        self.break_scan_valid = False

    def to_tsla_labels(self) -> ChannelLabels:
        """Extract TSLA-specific labels into a standalone ChannelLabels object."""
        labels = ChannelLabels()

        # Core (shared)
        labels.duration_bars = self.duration_bars
        labels.next_channel_direction = self.next_channel_direction
        labels.permanent_break = self.permanent_break
        labels.timeframe = self.timeframe

        # TSLA break scan
        labels.break_direction = self.break_direction
        labels.break_magnitude = self.break_magnitude
        labels.bars_to_first_break = self.bars_to_first_break
        labels.returned_to_channel = self.returned_to_channel
        labels.bounces_after_return = self.bounces_after_return
        labels.round_trip_bounces = self.round_trip_bounces
        labels.channel_continued = self.channel_continued
        labels.permanent_break_direction = self.permanent_break_direction
        labels.permanent_break_magnitude = self.permanent_break_magnitude
        labels.bars_to_permanent_break = self.bars_to_permanent_break
        labels.duration_to_permanent = self.duration_to_permanent
        labels.avg_bars_outside = self.avg_bars_outside
        labels.total_bars_outside = self.total_bars_outside
        labels.durability_score = self.durability_score
        labels.first_break_returned = self.first_break_returned
        labels.exit_return_rate = self.exit_return_rate
        labels.exits_returned_count = self.exits_returned_count
        labels.exits_stayed_out_count = self.exits_stayed_out_count
        labels.scan_timed_out = self.scan_timed_out
        labels.bars_verified_permanent = self.bars_verified_permanent

        # TSLA exit events
        labels.exit_bars = self.exit_bars.copy()
        labels.exit_magnitudes = self.exit_magnitudes.copy()
        labels.exit_durations = self.exit_durations.copy()
        labels.exit_types = self.exit_types.copy()
        labels.exit_returned = self.exit_returned.copy()

        # TSLA source channel
        labels.source_channel_slope = self.source_channel_slope
        labels.source_channel_intercept = self.source_channel_intercept
        labels.source_channel_std_dev = self.source_channel_std_dev
        labels.source_channel_r_squared = self.source_channel_r_squared
        labels.source_channel_direction = self.source_channel_direction
        labels.source_channel_bounce_count = self.source_channel_bounce_count
        labels.source_channel_start_ts = self.source_channel_start_ts
        labels.source_channel_end_ts = self.source_channel_end_ts

        # TSLA next channel
        labels.best_next_channel_direction = self.best_next_channel_direction
        labels.best_next_channel_bars_away = self.best_next_channel_bars_away
        labels.best_next_channel_duration = self.best_next_channel_duration
        labels.best_next_channel_r_squared = self.best_next_channel_r_squared
        labels.best_next_channel_bounce_count = self.best_next_channel_bounce_count
        labels.shortest_next_channel_direction = self.shortest_next_channel_direction
        labels.shortest_next_channel_bars_away = self.shortest_next_channel_bars_away
        labels.shortest_next_channel_duration = self.shortest_next_channel_duration
        labels.small_channels_before_best = self.small_channels_before_best

        # TSLA RSI
        labels.rsi_at_first_break = self.rsi_at_first_break
        labels.rsi_at_permanent_break = self.rsi_at_permanent_break
        labels.rsi_at_channel_end = self.rsi_at_channel_end
        labels.rsi_overbought_at_break = self.rsi_overbought_at_break
        labels.rsi_oversold_at_break = self.rsi_oversold_at_break
        labels.rsi_divergence_at_break = self.rsi_divergence_at_break
        labels.rsi_trend_in_channel = self.rsi_trend_in_channel
        labels.rsi_range_in_channel = self.rsi_range_in_channel

        # Validity flags (shared)
        labels.duration_valid = self.duration_valid
        labels.direction_valid = self.direction_valid
        labels.next_channel_valid = self.next_channel_valid
        labels.break_scan_valid = self.break_scan_valid

        return labels

    def to_spy_labels(self) -> ChannelLabels:
        """Extract SPY-specific labels into a standalone ChannelLabels object."""
        labels = ChannelLabels()

        # Core (shared)
        labels.duration_bars = self.duration_bars
        labels.next_channel_direction = self.next_channel_direction
        labels.permanent_break = self.permanent_break
        labels.timeframe = self.timeframe

        # SPY break scan (mapped from spy_ prefixed fields)
        labels.break_direction = self.spy_break_direction
        labels.break_magnitude = self.spy_break_magnitude
        labels.bars_to_first_break = self.spy_bars_to_first_break
        labels.returned_to_channel = self.spy_returned_to_channel
        labels.bounces_after_return = self.spy_bounces_after_return
        labels.round_trip_bounces = self.spy_round_trip_bounces
        labels.channel_continued = self.spy_channel_continued
        labels.permanent_break_direction = self.spy_permanent_break_direction
        labels.permanent_break_magnitude = self.spy_permanent_break_magnitude
        labels.bars_to_permanent_break = self.spy_bars_to_permanent_break
        labels.duration_to_permanent = self.spy_duration_to_permanent
        labels.avg_bars_outside = self.spy_avg_bars_outside
        labels.total_bars_outside = self.spy_total_bars_outside
        labels.durability_score = self.spy_durability_score
        labels.first_break_returned = self.spy_first_break_returned
        labels.exit_return_rate = self.spy_exit_return_rate
        labels.exits_returned_count = self.spy_exits_returned_count
        labels.exits_stayed_out_count = self.spy_exits_stayed_out_count
        labels.scan_timed_out = self.spy_scan_timed_out
        labels.bars_verified_permanent = self.spy_bars_verified_permanent

        # SPY exit events
        labels.exit_bars = self.spy_exit_bars.copy()
        labels.exit_magnitudes = self.spy_exit_magnitudes.copy()
        labels.exit_durations = self.spy_exit_durations.copy()
        labels.exit_types = self.spy_exit_types.copy()
        labels.exit_returned = self.spy_exit_returned.copy()

        # SPY source channel
        labels.source_channel_slope = self.spy_source_channel_slope
        labels.source_channel_intercept = self.spy_source_channel_intercept
        labels.source_channel_std_dev = self.spy_source_channel_std_dev
        labels.source_channel_r_squared = self.spy_source_channel_r_squared
        labels.source_channel_direction = self.spy_source_channel_direction
        labels.source_channel_bounce_count = self.spy_source_channel_bounce_count
        labels.source_channel_start_ts = self.spy_source_channel_start_ts
        labels.source_channel_end_ts = self.spy_source_channel_end_ts

        # SPY next channel
        labels.best_next_channel_direction = self.spy_best_next_channel_direction
        labels.best_next_channel_bars_away = self.spy_best_next_channel_bars_away
        labels.best_next_channel_duration = self.spy_best_next_channel_duration
        labels.best_next_channel_r_squared = self.spy_best_next_channel_r_squared
        labels.best_next_channel_bounce_count = self.spy_best_next_channel_bounce_count
        labels.shortest_next_channel_direction = self.spy_shortest_next_channel_direction
        labels.shortest_next_channel_bars_away = self.spy_shortest_next_channel_bars_away
        labels.shortest_next_channel_duration = self.spy_shortest_next_channel_duration
        labels.small_channels_before_best = self.spy_small_channels_before_best

        # SPY RSI
        labels.rsi_at_first_break = self.spy_rsi_at_first_break
        labels.rsi_at_permanent_break = self.spy_rsi_at_permanent_break
        labels.rsi_at_channel_end = self.spy_rsi_at_channel_end
        labels.rsi_overbought_at_break = self.spy_rsi_overbought_at_break
        labels.rsi_oversold_at_break = self.spy_rsi_oversold_at_break
        labels.rsi_divergence_at_break = self.spy_rsi_divergence_at_break
        labels.rsi_trend_in_channel = self.spy_rsi_trend_in_channel
        labels.rsi_range_in_channel = self.spy_rsi_range_in_channel

        # Validity flags (shared)
        labels.duration_valid = self.duration_valid
        labels.direction_valid = self.direction_valid
        labels.next_channel_valid = self.next_channel_valid
        labels.break_scan_valid = self.break_scan_valid

        return labels


class ChannelSample:
    """
    Python representation of ChannelSample struct.

    Structure matches what the inspector expects:
    - timestamp: pd.Timestamp (or int64 if pandas unavailable)
    - labels_per_window: Dict[window, Dict[asset, Dict[tf, ChannelLabels]]]
      where asset is 'tsla' or 'spy'
    """

    def __init__(self):
        self.timestamp = 0  # Will be converted to pd.Timestamp
        self.channel_end_idx = 0
        self.best_window = 0
        self.tf_features: Dict[str, float] = {}
        # Structure: {window: {'tsla': {tf: labels}, 'spy': {tf: labels}}}
        self.labels_per_window: Dict[int, Dict[str, Dict[str, ChannelLabels]]] = {}
        self.bar_metadata: Dict[str, Dict[str, float]] = {}


# =============================================================================
# BINARY READING HELPERS
# =============================================================================

def read_string(f) -> str:
    """Read length-prefixed string"""
    length_bytes = f.read(2)
    if len(length_bytes) != 2:
        raise EOFError("Unexpected end of file reading string length")
    length = struct.unpack('<H', length_bytes)[0]
    data = f.read(length)
    if len(data) != length:
        raise EOFError("Unexpected end of file reading string data")
    return data.decode('utf-8')


def read_bool(f) -> bool:
    """Read boolean (1 byte)"""
    data = f.read(1)
    if len(data) != 1:
        raise EOFError("Unexpected end of file reading bool")
    return struct.unpack('B', data)[0] != 0


def read_int_vector(f) -> List[int]:
    """Read vector of int32"""
    count_bytes = f.read(4)
    if len(count_bytes) != 4:
        raise EOFError("Unexpected end of file reading vector count")
    count = struct.unpack('<I', count_bytes)[0]

    result = []
    for _ in range(count):
        data = f.read(4)
        if len(data) != 4:
            raise EOFError("Unexpected end of file reading int vector element")
        result.append(struct.unpack('<i', data)[0])
    return result


def read_double_vector(f) -> List[float]:
    """Read vector of doubles"""
    count_bytes = f.read(4)
    if len(count_bytes) != 4:
        raise EOFError("Unexpected end of file reading vector count")
    count = struct.unpack('<I', count_bytes)[0]

    result = []
    for _ in range(count):
        data = f.read(8)
        if len(data) != 8:
            raise EOFError("Unexpected end of file reading double vector element")
        result.append(struct.unpack('<d', data)[0])
    return result


def read_bool_vector(f) -> List[bool]:
    """Read vector of bools"""
    count_bytes = f.read(4)
    if len(count_bytes) != 4:
        raise EOFError("Unexpected end of file reading vector count")
    count = struct.unpack('<I', count_bytes)[0]

    result = []
    for _ in range(count):
        data = f.read(1)
        if len(data) != 1:
            raise EOFError("Unexpected end of file reading bool vector element")
        result.append(struct.unpack('B', data)[0] != 0)
    return result


def read_combined_channel_labels(f) -> CombinedChannelLabels:
    """Read combined ChannelLabels from binary stream (internal format)"""
    labels = CombinedChannelLabels()

    # Read in the exact same order as C++ serialization
    # Core prediction targets
    labels.duration_bars = struct.unpack('<i', f.read(4))[0]
    labels.next_channel_direction = struct.unpack('<i', f.read(4))[0]
    labels.permanent_break = read_bool(f)
    labels.timeframe = struct.unpack('<i', f.read(4))[0]

    # TSLA break scan features
    labels.break_direction = struct.unpack('<i', f.read(4))[0]
    labels.break_magnitude = struct.unpack('<d', f.read(8))[0]
    labels.bars_to_first_break = struct.unpack('<i', f.read(4))[0]
    labels.returned_to_channel = read_bool(f)
    labels.bounces_after_return = struct.unpack('<i', f.read(4))[0]
    labels.round_trip_bounces = struct.unpack('<i', f.read(4))[0]
    labels.channel_continued = read_bool(f)
    labels.permanent_break_direction = struct.unpack('<i', f.read(4))[0]
    labels.permanent_break_magnitude = struct.unpack('<d', f.read(8))[0]
    labels.bars_to_permanent_break = struct.unpack('<i', f.read(4))[0]
    labels.duration_to_permanent = struct.unpack('<i', f.read(4))[0]
    labels.avg_bars_outside = struct.unpack('<d', f.read(8))[0]
    labels.total_bars_outside = struct.unpack('<i', f.read(4))[0]
    labels.durability_score = struct.unpack('<d', f.read(8))[0]
    labels.first_break_returned = read_bool(f)
    labels.exit_return_rate = struct.unpack('<d', f.read(8))[0]
    labels.exits_returned_count = struct.unpack('<i', f.read(4))[0]
    labels.exits_stayed_out_count = struct.unpack('<i', f.read(4))[0]
    labels.scan_timed_out = read_bool(f)
    labels.bars_verified_permanent = struct.unpack('<i', f.read(4))[0]

    # TSLA exit events
    labels.exit_bars = read_int_vector(f)
    labels.exit_magnitudes = read_double_vector(f)
    labels.exit_durations = read_int_vector(f)
    labels.exit_types = read_int_vector(f)
    labels.exit_returned = read_bool_vector(f)

    # SPY break scan features (same pattern as TSLA)
    labels.spy_break_direction = struct.unpack('<i', f.read(4))[0]
    labels.spy_break_magnitude = struct.unpack('<d', f.read(8))[0]
    labels.spy_bars_to_first_break = struct.unpack('<i', f.read(4))[0]
    labels.spy_returned_to_channel = read_bool(f)
    labels.spy_bounces_after_return = struct.unpack('<i', f.read(4))[0]
    labels.spy_round_trip_bounces = struct.unpack('<i', f.read(4))[0]
    labels.spy_channel_continued = read_bool(f)
    labels.spy_permanent_break_direction = struct.unpack('<i', f.read(4))[0]
    labels.spy_permanent_break_magnitude = struct.unpack('<d', f.read(8))[0]
    labels.spy_bars_to_permanent_break = struct.unpack('<i', f.read(4))[0]
    labels.spy_duration_to_permanent = struct.unpack('<i', f.read(4))[0]
    labels.spy_avg_bars_outside = struct.unpack('<d', f.read(8))[0]
    labels.spy_total_bars_outside = struct.unpack('<i', f.read(4))[0]
    labels.spy_durability_score = struct.unpack('<d', f.read(8))[0]
    labels.spy_first_break_returned = read_bool(f)
    labels.spy_exit_return_rate = struct.unpack('<d', f.read(8))[0]
    labels.spy_exits_returned_count = struct.unpack('<i', f.read(4))[0]
    labels.spy_exits_stayed_out_count = struct.unpack('<i', f.read(4))[0]
    labels.spy_scan_timed_out = read_bool(f)
    labels.spy_bars_verified_permanent = struct.unpack('<i', f.read(4))[0]

    # SPY exit events
    labels.spy_exit_bars = read_int_vector(f)
    labels.spy_exit_magnitudes = read_double_vector(f)
    labels.spy_exit_durations = read_int_vector(f)
    labels.spy_exit_types = read_int_vector(f)
    labels.spy_exit_returned = read_bool_vector(f)

    # Source channel parameters
    labels.source_channel_slope = struct.unpack('<d', f.read(8))[0]
    labels.source_channel_intercept = struct.unpack('<d', f.read(8))[0]
    labels.source_channel_std_dev = struct.unpack('<d', f.read(8))[0]
    labels.source_channel_r_squared = struct.unpack('<d', f.read(8))[0]
    labels.source_channel_direction = struct.unpack('<i', f.read(4))[0]
    labels.source_channel_bounce_count = struct.unpack('<i', f.read(4))[0]
    labels.source_channel_start_ts = struct.unpack('<q', f.read(8))[0]
    labels.source_channel_end_ts = struct.unpack('<q', f.read(8))[0]

    labels.spy_source_channel_slope = struct.unpack('<d', f.read(8))[0]
    labels.spy_source_channel_intercept = struct.unpack('<d', f.read(8))[0]
    labels.spy_source_channel_std_dev = struct.unpack('<d', f.read(8))[0]
    labels.spy_source_channel_r_squared = struct.unpack('<d', f.read(8))[0]
    labels.spy_source_channel_direction = struct.unpack('<i', f.read(4))[0]
    labels.spy_source_channel_bounce_count = struct.unpack('<i', f.read(4))[0]
    labels.spy_source_channel_start_ts = struct.unpack('<q', f.read(8))[0]
    labels.spy_source_channel_end_ts = struct.unpack('<q', f.read(8))[0]

    # Next channel labels
    labels.best_next_channel_direction = struct.unpack('<i', f.read(4))[0]
    labels.best_next_channel_bars_away = struct.unpack('<i', f.read(4))[0]
    labels.best_next_channel_duration = struct.unpack('<i', f.read(4))[0]
    labels.best_next_channel_r_squared = struct.unpack('<d', f.read(8))[0]
    labels.best_next_channel_bounce_count = struct.unpack('<i', f.read(4))[0]
    labels.shortest_next_channel_direction = struct.unpack('<i', f.read(4))[0]
    labels.shortest_next_channel_bars_away = struct.unpack('<i', f.read(4))[0]
    labels.shortest_next_channel_duration = struct.unpack('<i', f.read(4))[0]
    labels.small_channels_before_best = struct.unpack('<i', f.read(4))[0]

    labels.spy_best_next_channel_direction = struct.unpack('<i', f.read(4))[0]
    labels.spy_best_next_channel_bars_away = struct.unpack('<i', f.read(4))[0]
    labels.spy_best_next_channel_duration = struct.unpack('<i', f.read(4))[0]
    labels.spy_best_next_channel_r_squared = struct.unpack('<d', f.read(8))[0]
    labels.spy_best_next_channel_bounce_count = struct.unpack('<i', f.read(4))[0]
    labels.spy_shortest_next_channel_direction = struct.unpack('<i', f.read(4))[0]
    labels.spy_shortest_next_channel_bars_away = struct.unpack('<i', f.read(4))[0]
    labels.spy_shortest_next_channel_duration = struct.unpack('<i', f.read(4))[0]
    labels.spy_small_channels_before_best = struct.unpack('<i', f.read(4))[0]

    # RSI labels
    labels.rsi_at_first_break = struct.unpack('<d', f.read(8))[0]
    labels.rsi_at_permanent_break = struct.unpack('<d', f.read(8))[0]
    labels.rsi_at_channel_end = struct.unpack('<d', f.read(8))[0]
    labels.rsi_overbought_at_break = read_bool(f)
    labels.rsi_oversold_at_break = read_bool(f)
    labels.rsi_divergence_at_break = struct.unpack('<i', f.read(4))[0]
    labels.rsi_trend_in_channel = struct.unpack('<i', f.read(4))[0]
    labels.rsi_range_in_channel = struct.unpack('<d', f.read(8))[0]

    labels.spy_rsi_at_first_break = struct.unpack('<d', f.read(8))[0]
    labels.spy_rsi_at_permanent_break = struct.unpack('<d', f.read(8))[0]
    labels.spy_rsi_at_channel_end = struct.unpack('<d', f.read(8))[0]
    labels.spy_rsi_overbought_at_break = read_bool(f)
    labels.spy_rsi_oversold_at_break = read_bool(f)
    labels.spy_rsi_divergence_at_break = struct.unpack('<i', f.read(4))[0]
    labels.spy_rsi_trend_in_channel = struct.unpack('<i', f.read(4))[0]
    labels.spy_rsi_range_in_channel = struct.unpack('<d', f.read(8))[0]

    # Validity flags
    labels.duration_valid = read_bool(f)
    labels.direction_valid = read_bool(f)
    labels.next_channel_valid = read_bool(f)
    labels.break_scan_valid = read_bool(f)

    return labels


def read_feature_name_table(f) -> List[str]:
    """Read feature name table from binary stream (v3 format)"""
    count_bytes = f.read(4)
    if len(count_bytes) != 4:
        raise EOFError("Unexpected end of file reading feature table count")
    count = struct.unpack('<I', count_bytes)[0]

    names = []
    for _ in range(count):
        name = read_string(f)
        names.append(name)
    return names


def read_channel_sample(f, version: int = 2, feature_table: Optional[List[str]] = None) -> ChannelSample:
    """Read ChannelSample from binary stream with inspector-compatible format

    Args:
        f: File handle
        version: Format version (2 or 3)
        feature_table: Feature name table for v3 format (required if version=3)
    """
    sample = ChannelSample()

    # Core sample data
    timestamp_ms = struct.unpack('<q', f.read(8))[0]
    # Convert to pd.Timestamp if pandas available
    if HAS_PANDAS:
        sample.timestamp = pd.Timestamp(timestamp_ms, unit='ms')
    else:
        sample.timestamp = timestamp_ms

    sample.channel_end_idx = struct.unpack('<i', f.read(4))[0]
    sample.best_window = struct.unpack('<i', f.read(4))[0]

    # Features - version-dependent reading
    feature_count = struct.unpack('<I', f.read(4))[0]
    if version >= 3 and feature_table is not None:
        # v3: index-based features
        for _ in range(feature_count):
            index = struct.unpack('<H', f.read(2))[0]
            value = struct.unpack('<d', f.read(8))[0]
            if index < len(feature_table):
                key = feature_table[index]
                sample.tf_features[key] = value
            else:
                raise ValueError(f"Feature index {index} out of range (table size: {len(feature_table)})")
    else:
        # v2: string keys
        for _ in range(feature_count):
            key = read_string(f)
            value = struct.unpack('<d', f.read(8))[0]
            sample.tf_features[key] = value

    # Labels per window - read and transform to inspector format
    window_count = struct.unpack('<I', f.read(4))[0]
    for _ in range(window_count):
        window_size = struct.unpack('<i', f.read(4))[0]
        tf_count = struct.unpack('<I', f.read(4))[0]

        if window_size not in sample.labels_per_window:
            sample.labels_per_window[window_size] = {'tsla': {}, 'spy': {}}

        for _ in range(tf_count):
            tf_key = read_string(f)
            combined_labels = read_combined_channel_labels(f)

            # Split into TSLA and SPY labels
            sample.labels_per_window[window_size]['tsla'][tf_key] = combined_labels.to_tsla_labels()
            sample.labels_per_window[window_size]['spy'][tf_key] = combined_labels.to_spy_labels()

    # Bar metadata
    metadata_tf_count = struct.unpack('<I', f.read(4))[0]
    for _ in range(metadata_tf_count):
        tf_key = read_string(f)
        meta_count = struct.unpack('<I', f.read(4))[0]

        if tf_key not in sample.bar_metadata:
            sample.bar_metadata[tf_key] = {}

        for _ in range(meta_count):
            meta_key = read_string(f)
            meta_value = struct.unpack('<d', f.read(8))[0]
            sample.bar_metadata[tf_key][meta_key] = meta_value

    return sample


def load_samples(filename: str, max_samples: Optional[int] = None) -> Tuple[int, int, int, List[ChannelSample]]:
    """
    Load samples from binary file.

    Supports both v2 (string keys) and v3 (index-based) formats.

    Args:
        filename: Path to the binary sample file
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        (version, num_samples, num_features, samples)
    """
    with open(filename, 'rb') as f:
        # Read and validate header
        magic = f.read(8)
        if magic != b'V15SAMP\x00':
            raise ValueError(f"Invalid magic bytes: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported format version: {version}")

        num_samples = struct.unpack('<Q', f.read(8))[0]
        num_features = struct.unpack('<I', f.read(4))[0]

        # v3: Read feature name table
        feature_table = None
        if version >= 3:
            feature_table = read_feature_name_table(f)

        # Determine how many samples to load
        samples_to_load = num_samples
        if max_samples is not None and max_samples < num_samples:
            samples_to_load = max_samples

        # Read samples
        samples = []
        for i in range(samples_to_load):
            try:
                sample = read_channel_sample(f, version=version, feature_table=feature_table)
                samples.append(sample)
            except EOFError as e:
                raise EOFError(f"Error reading sample {i}: {e}")

        return version, num_samples, num_features, samples


def load_samples_simple(filename: str, max_samples: Optional[int] = None) -> List[ChannelSample]:
    """
    Load samples from binary file (simplified API).

    Args:
        filename: Path to the binary sample file
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of ChannelSample objects ready for use with inspector.
    """
    _, _, _, samples = load_samples(filename, max_samples=max_samples)
    return samples


def main():
    """Test loading a sample file"""
    if len(sys.argv) < 2:
        print("Usage: python load_samples.py <sample_file.bin>")
        sys.exit(1)

    filename = sys.argv[1]

    if not Path(filename).exists():
        print(f"Error: File '{filename}' not found")
        sys.exit(1)

    print(f"Loading samples from {filename}...")
    print("=" * 60)

    version, num_samples, num_features, samples = load_samples(filename)

    print(f"Format version: {version}")
    print(f"Number of samples: {num_samples}")
    print(f"Average features per sample: {num_features}")
    print(f"Loaded: {len(samples)} samples")
    print()

    if samples:
        sample = samples[0]
        print("First sample:")
        print(f"  Timestamp: {sample.timestamp} (type: {type(sample.timestamp).__name__})")
        print(f"  Channel end idx: {sample.channel_end_idx}")
        print(f"  Best window: {sample.best_window}")
        print(f"  Features: {len(sample.tf_features)}")
        print(f"  Label windows: {list(sample.labels_per_window.keys())}")
        print(f"  Bar metadata TFs: {list(sample.bar_metadata.keys())}")

        # Show labels structure
        if sample.labels_per_window:
            print("\n  Labels structure (inspector-compatible):")
            for window in sorted(sample.labels_per_window.keys())[:1]:
                print(f"    Window {window}:")
                for asset in ['tsla', 'spy']:
                    if asset in sample.labels_per_window[window]:
                        tfs = list(sample.labels_per_window[window][asset].keys())
                        print(f"      {asset}: {tfs[:3]}{'...' if len(tfs) > 3 else ''}")

        if sample.tf_features:
            print("\n  Sample features (first 5):")
            for i, (key, value) in enumerate(sample.tf_features.items()):
                if i >= 5:
                    break
                print(f"    {key}: {value:.6f}")

        if sample.labels_per_window:
            for window, assets_dict in sample.labels_per_window.items():
                if 'tsla' in assets_dict:
                    for tf, labels in assets_dict['tsla'].items():
                        print(f"\n  TSLA Labels (window={window}, tf={tf}):")
                        print(f"    Duration bars: {labels.duration_bars}")
                        print(f"    Break direction: {labels.break_direction}")
                        print(f"    Break magnitude: {labels.break_magnitude:.4f}")
                        print(f"    Source channel slope: {labels.source_channel_slope:.6f}")
                        print(f"    Source channel r_squared: {labels.source_channel_r_squared:.4f}")
                        break
                if 'spy' in assets_dict:
                    for tf, labels in assets_dict['spy'].items():
                        print(f"\n  SPY Labels (window={window}, tf={tf}):")
                        print(f"    Break direction: {labels.break_direction}")
                        print(f"    Break magnitude: {labels.break_magnitude:.4f}")
                        print(f"    Source channel slope: {labels.source_channel_slope:.6f}")
                        break
                break

    print("\n" + "=" * 60)
    print("SUCCESS - Inspector-compatible format!")


if __name__ == '__main__':
    main()
