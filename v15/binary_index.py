"""
Binary file offset index for streaming data loading.

This module provides functions to build and manage byte-offset indices
for V15 binary sample files. The index enables random access to samples
without loading the entire file into memory.

Index format:
- NumPy array of int64 offsets, one per sample
- ~8 bytes per sample (996K samples = ~8MB index)

Usage:
    # Build or load index
    offsets = get_or_build_index('/path/to/samples.bin')

    # Access specific sample
    with open('/path/to/samples.bin', 'rb') as f:
        f.seek(offsets[42])
        sample = read_channel_sample_at_offset(f, version, feature_table)
"""

import struct
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


def skip_feature_name_table(f) -> int:
    """
    Skip v3 feature name table and return count.

    Returns:
        Number of features in the table.
    """
    count_bytes = f.read(4)
    if len(count_bytes) != 4:
        raise EOFError("Unexpected end of file reading feature table count")
    count = struct.unpack('<I', count_bytes)[0]

    for _ in range(count):
        name_len_bytes = f.read(2)
        if len(name_len_bytes) != 2:
            raise EOFError("Unexpected end of file reading feature name length")
        name_len = struct.unpack('<H', name_len_bytes)[0]
        f.read(name_len)

    return count


def skip_string(f):
    """Skip a length-prefixed string."""
    name_len_bytes = f.read(2)
    if len(name_len_bytes) != 2:
        raise EOFError("Unexpected end of file reading string length")
    name_len = struct.unpack('<H', name_len_bytes)[0]
    f.read(name_len)


def skip_int_vector(f):
    """Skip a vector of int32."""
    count_bytes = f.read(4)
    if len(count_bytes) != 4:
        raise EOFError("Unexpected end of file reading vector count")
    count = struct.unpack('<I', count_bytes)[0]
    f.read(count * 4)  # Each int32 is 4 bytes


def skip_double_vector(f):
    """Skip a vector of doubles."""
    count_bytes = f.read(4)
    if len(count_bytes) != 4:
        raise EOFError("Unexpected end of file reading vector count")
    count = struct.unpack('<I', count_bytes)[0]
    f.read(count * 8)  # Each double is 8 bytes


def skip_bool_vector(f):
    """Skip a vector of bools."""
    count_bytes = f.read(4)
    if len(count_bytes) != 4:
        raise EOFError("Unexpected end of file reading vector count")
    count = struct.unpack('<I', count_bytes)[0]
    f.read(count)  # Each bool is 1 byte


def skip_channel_labels(f):
    """Skip combined ChannelLabels without full deserialization."""
    # Core prediction targets (4 + 4 + 1 + 4 = 13 bytes)
    f.read(4 + 4 + 1 + 4)

    # TSLA break scan features (multiple fields)
    # break_direction(4) + break_magnitude(8) + bars_to_first_break(4) + returned(1) +
    # bounces_after_return(4) + round_trip_bounces(4) + channel_continued(1) +
    # permanent_break_direction(4) + permanent_break_magnitude(8) + bars_to_permanent_break(4) +
    # duration_to_permanent(4) + avg_bars_outside(8) + total_bars_outside(4) +
    # durability_score(8) + first_break_returned(1) + exit_return_rate(8) +
    # exits_returned_count(4) + exits_stayed_out_count(4) + scan_timed_out(1) +
    # bars_verified_permanent(4)
    # = 4+8+4+1+4+4+1+4+8+4+4+8+4+8+1+8+4+4+1+4 = 88 bytes
    f.read(88)

    # TSLA exit events (5 vectors)
    skip_int_vector(f)     # exit_bars
    skip_double_vector(f)  # exit_magnitudes
    skip_int_vector(f)     # exit_durations
    skip_int_vector(f)     # exit_types
    skip_bool_vector(f)    # exit_returned

    # SPY break scan features (same pattern as TSLA = 88 bytes)
    f.read(88)

    # SPY exit events (5 vectors)
    skip_int_vector(f)     # spy_exit_bars
    skip_double_vector(f)  # spy_exit_magnitudes
    skip_int_vector(f)     # spy_exit_durations
    skip_int_vector(f)     # spy_exit_types
    skip_bool_vector(f)    # spy_exit_returned

    # TSLA source channel parameters
    # slope(8) + intercept(8) + std_dev(8) + r_squared(8) + direction(4) +
    # bounce_count(4) + start_ts(8) + end_ts(8) = 56 bytes
    f.read(56)

    # SPY source channel parameters (same = 56 bytes)
    f.read(56)

    # TSLA next channel labels
    # best_direction(4) + best_bars_away(4) + best_duration(4) + best_r2(8) +
    # best_bounce_count(4) + shortest_direction(4) + shortest_bars_away(4) +
    # shortest_duration(4) + small_channels_before_best(4) = 40 bytes
    f.read(40)

    # SPY next channel labels (same = 40 bytes)
    f.read(40)

    # TSLA RSI labels
    # rsi_at_first_break(8) + rsi_at_permanent_break(8) + rsi_at_channel_end(8) +
    # rsi_overbought(1) + rsi_oversold(1) + rsi_divergence(4) + rsi_trend(4) +
    # rsi_range(8) = 42 bytes
    f.read(42)

    # SPY RSI labels (same = 42 bytes)
    f.read(42)

    # Validity flags (4 bools = 4 bytes)
    f.read(4)


def skip_labels_per_window(f):
    """Skip labels_per_window section."""
    window_count_bytes = f.read(4)
    if len(window_count_bytes) != 4:
        raise EOFError("Unexpected end of file reading window count")
    window_count = struct.unpack('<I', window_count_bytes)[0]

    for _ in range(window_count):
        # window_size (4 bytes)
        f.read(4)

        # tf_count
        tf_count_bytes = f.read(4)
        if len(tf_count_bytes) != 4:
            raise EOFError("Unexpected end of file reading tf count")
        tf_count = struct.unpack('<I', tf_count_bytes)[0]

        for _ in range(tf_count):
            # tf_key string
            skip_string(f)
            # ChannelLabels
            skip_channel_labels(f)


def skip_bar_metadata(f):
    """Skip bar_metadata section."""
    tf_count_bytes = f.read(4)
    if len(tf_count_bytes) != 4:
        raise EOFError("Unexpected end of file reading metadata tf count")
    tf_count = struct.unpack('<I', tf_count_bytes)[0]

    for _ in range(tf_count):
        # tf_key string
        skip_string(f)

        # meta_count
        meta_count_bytes = f.read(4)
        if len(meta_count_bytes) != 4:
            raise EOFError("Unexpected end of file reading meta count")
        meta_count = struct.unpack('<I', meta_count_bytes)[0]

        for _ in range(meta_count):
            # meta_key string + double value
            skip_string(f)
            f.read(8)


def skip_channel_sample(f, version: int):
    """
    Skip a single ChannelSample without full deserialization.

    Args:
        f: File handle positioned at sample start
        version: Format version (2 or 3)
    """
    # timestamp (8) + channel_end_idx (4) + best_window (4) = 16 bytes
    f.read(16)

    # Features
    feature_count_bytes = f.read(4)
    if len(feature_count_bytes) != 4:
        raise EOFError("Unexpected end of file reading feature count")
    feature_count = struct.unpack('<I', feature_count_bytes)[0]

    if version >= 3:
        # v3: index (2) + value (8) = 10 bytes per feature
        f.read(feature_count * 10)
    else:
        # v2: string key + value per feature
        for _ in range(feature_count):
            skip_string(f)
            f.read(8)

    # Labels per window
    skip_labels_per_window(f)

    # Bar metadata
    skip_bar_metadata(f)


def build_offset_index(binary_path: str, progress_interval: int = 10000) -> Tuple[np.ndarray, int, Optional[List[str]]]:
    """
    Scan binary file and build offset index.

    Args:
        binary_path: Path to binary sample file
        progress_interval: How often to print progress (samples)

    Returns:
        Tuple of:
            - np.ndarray of int64: [num_samples] byte offsets
            - version: File format version
            - feature_table: Feature name table (v3 only) or None
    """
    offsets = []

    with open(binary_path, 'rb') as f:
        # Read and validate header
        magic = f.read(8)
        if magic != b'V15SAMP\x00':
            raise ValueError(f"Invalid magic bytes: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported format version: {version}")

        num_samples = struct.unpack('<Q', f.read(8))[0]
        num_features = struct.unpack('<I', f.read(4))[0]

        # Read feature table for v3
        feature_table = None
        if version >= 3:
            # Read feature name table
            count = struct.unpack('<I', f.read(4))[0]
            feature_table = []
            for _ in range(count):
                name_len = struct.unpack('<H', f.read(2))[0]
                name = f.read(name_len).decode('utf-8')
                feature_table.append(name)

        # Scan samples and record offsets
        for i in range(num_samples):
            offset = f.tell()
            offsets.append(offset)
            skip_channel_sample(f, version)

            if progress_interval and (i + 1) % progress_interval == 0:
                print(f"Indexed {i + 1:,}/{num_samples:,} samples...")

    return np.array(offsets, dtype=np.int64), version, feature_table


def save_index(index: np.ndarray, index_path: str):
    """Save offset index to file."""
    np.save(index_path, index)


def load_index(index_path: str) -> np.ndarray:
    """Load offset index from file."""
    return np.load(index_path)


def get_or_build_index(
    binary_path: str,
    force_rebuild: bool = False
) -> Tuple[np.ndarray, int, Optional[List[str]]]:
    """
    Load existing index or build if needed.

    Args:
        binary_path: Path to binary sample file
        force_rebuild: If True, rebuild even if index exists

    Returns:
        Tuple of:
            - np.ndarray of int64: [num_samples] byte offsets
            - version: File format version
            - feature_table: Feature name table (v3 only) or None
    """
    index_path = f"{binary_path}.idx.npy"
    meta_path = f"{binary_path}.idx.meta"

    binary_mtime = Path(binary_path).stat().st_mtime

    # Check if valid cached index exists
    if not force_rebuild and Path(index_path).exists() and Path(meta_path).exists():
        index_mtime = Path(index_path).stat().st_mtime

        if index_mtime > binary_mtime:
            print(f"Loading existing index from {index_path}")
            offsets = load_index(index_path)

            # Load metadata (version, feature_table)
            import pickle
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)

            return offsets, meta['version'], meta.get('feature_table')

    # Build new index
    print(f"Building index for {binary_path}...")
    offsets, version, feature_table = build_offset_index(binary_path)

    # Save index
    save_index(offsets, index_path)
    print(f"Index saved to {index_path}")

    # Save metadata
    import pickle
    meta = {
        'version': version,
        'feature_table': feature_table,
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    return offsets, version, feature_table


def get_header_info(binary_path: str) -> dict:
    """
    Read header info from binary file without building full index.

    Returns:
        Dict with keys: version, num_samples, num_features, feature_table
    """
    with open(binary_path, 'rb') as f:
        magic = f.read(8)
        if magic != b'V15SAMP\x00':
            raise ValueError(f"Invalid magic bytes: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        num_samples = struct.unpack('<Q', f.read(8))[0]
        num_features = struct.unpack('<I', f.read(4))[0]

        feature_table = None
        if version >= 3:
            count = struct.unpack('<I', f.read(4))[0]
            feature_table = []
            for _ in range(count):
                name_len = struct.unpack('<H', f.read(2))[0]
                name = f.read(name_len).decode('utf-8')
                feature_table.append(name)

        return {
            'version': version,
            'num_samples': num_samples,
            'num_features': num_features,
            'feature_table': feature_table,
        }
