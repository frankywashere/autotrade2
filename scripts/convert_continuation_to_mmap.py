"""
Convert continuation_labels pickle files to mmap-friendly numpy format.

This eliminates the ~18 GB RAM waste with multi-GPU training (2 GPUs × 4 workers).

Usage:
    python scripts/convert_continuation_to_mmap.py

The script:
1. Reads each continuation_labels_*.pkl file
2. Extracts numeric columns into a single .npy file (mmap-able)
3. Saves timestamps as a separate .npy file
4. Saves price_sequence separately (variable-length, can't mmap)
5. Saves metadata JSON with column info

Original files are NOT deleted - the dataset will auto-detect and use mmap versions.
"""

import numpy as np
import pandas as pd
import pickle
import json
import sys
from pathlib import Path
from typing import Dict, List
import time

# Add parent to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import config
from src.ml.features import HIERARCHICAL_TIMEFRAMES


def convert_continuation_labels(cache_dir: Path):
    """Convert all continuation_labels pickle files to mmap format."""

    print("=" * 70)
    print("Convert Continuation Labels to Mmap Format")
    print("=" * 70)
    print(f"Cache directory: {cache_dir}")
    print()

    # Find all continuation labels files
    pkl_files = list(cache_dir.glob("continuation_labels_*.pkl"))
    if not pkl_files:
        print("ERROR: No continuation_labels_*.pkl files found")
        return

    print(f"Found {len(pkl_files)} pickle files to convert")
    print()

    # Define numeric columns that can be mmap'd
    WINDOW_SIZES = config.CHANNEL_WINDOW_SIZES  # [100, 90, 80, ..., 10]

    NUMERIC_FIELDS = ['duration', 'hit_upper', 'hit_midline', 'hit_lower',
                      'bars_until_hit_upper', 'bars_until_hit_midline', 'bars_until_hit_lower',
                      'time_near_upper', 'time_near_midline', 'time_near_lower',
                      'slope', 'confidence', 'valid']

    total_original_size = 0
    total_mmap_size = 0

    for pkl_path in sorted(pkl_files):
        print(f"\n{'─' * 60}")
        print(f"Converting: {pkl_path.name}")

        start_time = time.time()

        # Load pickle
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f)

        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            print(f"  ⚠️  Skipping: Not a valid DataFrame")
            continue

        n_rows = len(df)
        original_size = pkl_path.stat().st_size
        total_original_size += original_size
        print(f"  Rows: {n_rows:,}")
        print(f"  Original size: {original_size / 1e6:.1f} MB")

        # Build column list and collect data
        numeric_columns = []
        numeric_data = []

        # max_gain_pct is always present
        if 'max_gain_pct' in df.columns:
            numeric_columns.append('max_gain_pct')
            numeric_data.append(df['max_gain_pct'].values.astype(np.float32))

        # Window-specific columns
        for window in WINDOW_SIZES:
            for field in NUMERIC_FIELDS:
                col_name = f'w{window}_{field}'
                if col_name in df.columns:
                    numeric_columns.append(col_name)
                    numeric_data.append(df[col_name].values.astype(np.float32))

        if not numeric_data:
            print(f"  ⚠️  No numeric columns found, skipping")
            continue

        # Stack into single array [n_rows, n_columns]
        numeric_array = np.column_stack(numeric_data)
        print(f"  Numeric array: {numeric_array.shape} ({numeric_array.nbytes / 1e6:.1f} MB)")

        # Extract timestamps (index)
        timestamps = df.index.values.astype('datetime64[ns]').astype('int64')
        print(f"  Timestamps: {timestamps.shape} ({timestamps.nbytes / 1e6:.1f} MB)")

        # Handle price_sequence separately (variable-length lists, can't mmap)
        price_sequence_cols = [c for c in df.columns if 'price_sequence' in c]
        has_price_sequences = len(price_sequence_cols) > 0

        # Generate output paths
        base_name = pkl_path.stem  # e.g., "continuation_labels_5min_v5.9.1_..."

        numeric_path = cache_dir / f"{base_name}_mmap.npy"
        timestamps_path = cache_dir / f"{base_name}_timestamps.npy"
        meta_path = cache_dir / f"{base_name}_mmap_meta.json"
        price_seq_path = cache_dir / f"{base_name}_price_sequences.pkl"

        # Save numeric array (main mmap file)
        np.save(numeric_path, numeric_array)

        # Save timestamps
        np.save(timestamps_path, timestamps)

        # Save price sequences if present (still pickle, but much smaller)
        if has_price_sequences:
            price_seq_data = {}
            for col in price_sequence_cols:
                price_seq_data[col] = df[col].values
            with open(price_seq_path, 'wb') as f:
                pickle.dump(price_seq_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            price_seq_size = price_seq_path.stat().st_size
            print(f"  Price sequences: {len(price_sequence_cols)} columns ({price_seq_size / 1e6:.1f} MB)")
        else:
            price_seq_size = 0

        # Save metadata
        meta = {
            'version': '5.9.6',
            'source_file': pkl_path.name,
            'n_rows': n_rows,
            'numeric_columns': numeric_columns,
            'has_price_sequences': has_price_sequences,
            'price_sequence_columns': price_sequence_cols if has_price_sequences else [],
            'dtype': 'float32',
            'timestamp_dtype': 'int64',
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # Calculate sizes
        mmap_size = numeric_path.stat().st_size + timestamps_path.stat().st_size + price_seq_size
        total_mmap_size += mmap_size

        elapsed = time.time() - start_time
        print(f"  Mmap size: {mmap_size / 1e6:.1f} MB (ratio: {mmap_size / original_size:.2f}x)")
        print(f"  Converted in {elapsed:.1f}s")
        print(f"  ✓ Saved: {numeric_path.name}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files converted: {len(pkl_files)}")
    print(f"Original total: {total_original_size / 1e9:.2f} GB")
    print(f"Mmap total: {total_mmap_size / 1e9:.2f} GB")
    print()
    print("RAM savings with multi-GPU:")
    print(f"  2 GPUs × 4 workers: ~{total_original_size / 1e9 * 8:.1f} GB → ~{total_mmap_size / 1e9:.1f} GB (shared)")
    print(f"  4 GPUs × 4 workers: ~{total_original_size / 1e9 * 16:.1f} GB → ~{total_mmap_size / 1e9:.1f} GB (shared)")
    print()
    print("The dataset will automatically detect and use mmap files.")
    print("Original pickle files are preserved as fallback.")


def main():
    cache_dir = Path("data/feature_cache")

    if not cache_dir.exists():
        print(f"ERROR: Cache directory not found: {cache_dir}")
        sys.exit(1)

    convert_continuation_labels(cache_dir)


if __name__ == '__main__':
    main()
