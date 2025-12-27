#!/usr/bin/env python3
"""
Convert pickle-based label files to memory-mappable numpy format.

This conversion enables sharing label data across multiple DataLoader workers
without copying, significantly reducing RAM usage in multi-GPU training.

Before conversion (4 workers × 2 GPUs = 8 processes):
  continuation_labels: 2.3 GB × 8 = 18.4 GB RAM

After conversion:
  continuation_labels: 2.3 GB shared via mmap = 2.3 GB RAM

Usage:
    python tools/convert_labels_to_mmap.py
    python tools/convert_labels_to_mmap.py --labels-dir /path/to/feature_cache
    python tools/convert_labels_to_mmap.py --force  # Reconvert existing files
"""

import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


def flatten_price_sequences(series: pd.Series, dtype=np.float32, strict: bool = True):
    """
    Flatten variable-length price sequences into mmap-able arrays.

    Args:
        series: pandas Series of lists (variable length)
        dtype: numpy dtype for values
        strict: If True, raise error on NaN/scalar values (data bug). If False, treat as empty.

    Returns:
        flat_values: 1D array of all prices concatenated
        offsets: Start index for each sample (length = n_samples + 1)
        lengths: Length of each sample's sequence
    """
    offsets = [0]
    all_values = []
    lengths = []
    nan_count = 0

    for idx, seq in enumerate(series):
        # Handle various empty/invalid cases
        if seq is None:
            lengths.append(0)
        elif isinstance(seq, (int, float)):
            # Scalar value (NaN or placeholder) - this is a DATA BUG
            if strict and np.isnan(seq):
                nan_count += 1
            lengths.append(0)
        elif hasattr(seq, '__len__') and len(seq) == 0:
            lengths.append(0)
        else:
            try:
                seq_list = list(seq) if not isinstance(seq, list) else seq
                all_values.extend(seq_list)
                lengths.append(len(seq_list))
            except (TypeError, ValueError):
                if strict:
                    raise ValueError(f"Unexpected value type at index {idx}: {type(seq)}")
                lengths.append(0)
        offsets.append(len(all_values))

    if strict and nan_count > 0:
        print(f"    ⚠️  WARNING: Found {nan_count} NaN values in price_sequence (data generation bug)")
        print(f"       Run with --regenerate-labels to fix, or regenerate continuation labels")

    flat_values = np.array(all_values, dtype=dtype) if all_values else np.array([], dtype=dtype)
    offsets = np.array(offsets, dtype=np.int64)
    lengths = np.array(lengths, dtype=np.int32)

    return flat_values, offsets, lengths


def convert_continuation_labels(pkl_path: Path, output_dir: Path, force: bool = False) -> bool:
    """
    Convert a continuation labels pickle file to mmap format.

    Output structure (directory with .npy files):
        - timestamps.npy: int64 array of nanosecond timestamps
        - max_gain_pct.npy: float32 array
        - w{window}_duration.npy: float32 array per window
        - w{window}_hit_upper.npy: float32 array per window
        - ... (other window columns)
        - w{window}_price_sequence_flat.npy: float32 array (flattened)
        - w{window}_price_sequence_offsets.npy: int64 array
        - w{window}_price_sequence_lengths.npy: int32 array
    """
    # Determine output directory
    # continuation_labels_5min_v5.9.1_xxx.pkl -> continuation_labels_5min_v5.9.1_xxx.mmap/
    output_name = pkl_path.stem + ".mmap"
    output_path = output_dir / output_name

    if output_path.exists() and not force:
        print(f"  ⏭️  Skipping {pkl_path.name} (already converted)")
        return False

    print(f"  📦 Converting {pkl_path.name}...")

    # Load pickle
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)

    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        print(f"    ⚠️  Invalid DataFrame, skipping")
        return False

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    total_size = 0

    # Store timestamps from index as int64 nanoseconds
    timestamps = pd.to_datetime(df.index).astype(np.int64)
    np.save(output_path / 'timestamps.npy', timestamps)
    total_size += (output_path / 'timestamps.npy').stat().st_size

    # Store max_gain_pct
    if 'max_gain_pct' in df.columns:
        arr = df['max_gain_pct'].values.astype(np.float32)
        np.save(output_path / 'max_gain_pct.npy', arr)
        total_size += (output_path / 'max_gain_pct.npy').stat().st_size

    # Store per-window data
    windows = config.CHANNEL_WINDOW_SIZES
    for window in windows:
        prefix = f'w{window}_'

        # Numeric columns (directly convertible)
        numeric_cols = [
            'duration', 'hit_upper', 'hit_midline', 'hit_lower',
            'bars_until_hit_upper', 'bars_until_hit_midline', 'bars_until_hit_lower',
            'time_near_upper', 'time_near_midline', 'time_near_lower',
            'slope', 'confidence', 'valid'
        ]

        for col_suffix in numeric_cols:
            col_name = prefix + col_suffix
            if col_name in df.columns:
                arr = df[col_name].values.astype(np.float32)
                np.save(output_path / f'{col_name}.npy', arr)
                total_size += (output_path / f'{col_name}.npy').stat().st_size

        # Price sequence (variable length - flatten with offsets)
        price_seq_col = prefix + 'price_sequence'
        if price_seq_col in df.columns:
            flat_vals, offsets, lengths = flatten_price_sequences(df[price_seq_col])
            np.save(output_path / f'{price_seq_col}_flat.npy', flat_vals)
            np.save(output_path / f'{price_seq_col}_offsets.npy', offsets)
            np.save(output_path / f'{price_seq_col}_lengths.npy', lengths)
            total_size += (output_path / f'{price_seq_col}_flat.npy').stat().st_size
            total_size += (output_path / f'{price_seq_col}_offsets.npy').stat().st_size
            total_size += (output_path / f'{price_seq_col}_lengths.npy').stat().st_size

    # Report size
    orig_size = pkl_path.stat().st_size / 1024**2
    new_size = total_size / 1024**2
    print(f"    ✓ {len(df):,} samples | {orig_size:.1f} MB → {new_size:.1f} MB ({100*new_size/orig_size:.0f}%)")

    return True


def convert_transition_labels(pkl_path: Path, output_dir: Path, force: bool = False) -> bool:
    """
    Convert a transition labels pickle file to mmap format.

    Output structure (directory with .npy files):
        - timestamps.npy: int64 array of nanosecond timestamps
        - transition_type.npy: int64 array
        - current_direction.npy: int64 array
        - new_direction.npy: int64 array
        - new_slope.npy: float32 array
        - switch_to_tf.npy: int64 array (TF index, -1 for None)
    """
    output_name = pkl_path.stem + ".mmap"
    output_path = output_dir / output_name

    if output_path.exists() and not force:
        print(f"  ⏭️  Skipping {pkl_path.name} (already converted)")
        return False

    print(f"  📦 Converting {pkl_path.name}...")

    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)

    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        print(f"    ⚠️  Invalid DataFrame, skipping")
        return False

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    total_size = 0

    # Timestamps
    timestamps = pd.to_datetime(df.index).astype(np.int64)
    np.save(output_path / 'timestamps.npy', timestamps)
    total_size += (output_path / 'timestamps.npy').stat().st_size

    # Numeric columns
    for col, dtype in [('transition_type', np.int64), ('current_direction', np.int64),
                       ('new_direction', np.int64), ('new_slope', np.float32)]:
        arr = df[col].values.astype(dtype)
        np.save(output_path / f'{col}.npy', arr)
        total_size += (output_path / f'{col}.npy').stat().st_size

    # switch_to_tf: convert TF names to indices
    if 'switch_to_tf' in df.columns:
        from src.ml.features import HIERARCHICAL_TIMEFRAMES
        tf_to_idx = {tf: i for i, tf in enumerate(HIERARCHICAL_TIMEFRAMES)}
        switch_indices = []
        for val in df['switch_to_tf']:
            if val is None or pd.isna(val):
                switch_indices.append(-1)
            else:
                switch_indices.append(tf_to_idx.get(val, -1))
        arr = np.array(switch_indices, dtype=np.int64)
        np.save(output_path / 'switch_to_tf.npy', arr)
        total_size += (output_path / 'switch_to_tf.npy').stat().st_size

    orig_size = pkl_path.stat().st_size / 1024**2
    new_size = total_size / 1024**2
    print(f"    ✓ {len(df):,} samples | {orig_size:.1f} MB → {new_size:.1f} MB")

    return True


def main():
    parser = argparse.ArgumentParser(description='Convert label files to mmap format')
    parser.add_argument('--labels-dir', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'feature_cache'),
                        help='Directory containing label files')
    parser.add_argument('--force', action='store_true',
                        help='Reconvert even if output exists')
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        sys.exit(1)

    print(f"🔄 Converting label files to mmap format")
    print(f"   Source: {labels_dir}")
    print()

    # Convert continuation labels
    print("📂 Continuation labels:")
    cont_files = list(labels_dir.glob("continuation_labels_*.pkl"))
    cont_converted = 0
    for pkl_file in sorted(cont_files):
        if convert_continuation_labels(pkl_file, labels_dir, force=args.force):
            cont_converted += 1
    print(f"   Converted: {cont_converted}/{len(cont_files)}")
    print()

    # Convert transition labels
    print("📂 Transition labels:")
    trans_files = list(labels_dir.glob("transition_labels_*.pkl"))
    trans_converted = 0
    for pkl_file in sorted(trans_files):
        if convert_transition_labels(pkl_file, labels_dir, force=args.force):
            trans_converted += 1
    print(f"   Converted: {trans_converted}/{len(trans_files)}")
    print()

    # Summary
    total_converted = cont_converted + trans_converted
    if total_converted > 0:
        print(f"✅ Conversion complete! {total_converted} files converted")
        print()
        print("To use mmap loading, the dataset will automatically detect .mmap/ directories")
    else:
        print("ℹ️  No new files to convert (use --force to reconvert)")


if __name__ == '__main__':
    main()
