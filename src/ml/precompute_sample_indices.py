"""
Pre-compute Sample Indices for Fast Training (v5.9.5)

Phase 1: Pre-computed Index Mapping
- Eliminates 11x np.searchsorted() calls per sample (~5μs savings)
- Pre-computes [start, end] slice indices for each sample across all timeframes

Run this ONCE after feature extraction. The pre-computed files are used by
HierarchicalDataset for faster __getitem__.

Usage:
    python -m src.ml.precompute_sample_indices --cache-dir data/feature_cache

The script reads existing cached files and creates NEW files alongside them.
No existing cache files are modified.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import sys

# Add parent to path for imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from src.ml.features import HIERARCHICAL_TIMEFRAMES


def load_tf_metadata(cache_dir: Path) -> Tuple[Dict, str]:
    """
    Load timeframe metadata needed for index computation.

    Returns:
        Tuple of (metadata dict, cache_key string)
    """
    print("Loading timeframe metadata...")

    # Find tf_meta file
    tf_meta_files = sorted(cache_dir.glob("tf_meta_*.json"))
    if not tf_meta_files:
        raise FileNotFoundError(f"No tf_meta_*.json found in {cache_dir}")

    tf_meta_path = tf_meta_files[-1]
    print(f"  Loading {tf_meta_path.name}")
    with open(tf_meta_path) as f:
        tf_meta = json.load(f)

    cache_key = tf_meta['cache_key']
    print(f"  Cache key: {cache_key}")

    return tf_meta, cache_key


def load_timestamps(cache_dir: Path, cache_key: str) -> Dict[str, np.ndarray]:
    """
    Load timestamps for all timeframes.

    Returns:
        Dict mapping timeframe -> timestamp array (int64 ns)
    """
    print("Loading timestamps...")

    timestamps = {}
    for tf in HIERARCHICAL_TIMEFRAMES:
        ts_path = cache_dir / f"tf_timestamps_{tf}_{cache_key}.npy"
        if ts_path.exists():
            timestamps[tf] = np.load(str(ts_path))
            print(f"  {tf}: {len(timestamps[tf]):,} timestamps")
        else:
            print(f"  {tf}: Not found, skipping")

    return timestamps


def compute_sample_indices(
    cache_dir: Path,
    cache_key: str,
    tf_meta: Dict,
    timestamps: Dict[str, np.ndarray],
    valid_indices: np.ndarray,
    sequence_lengths: Dict[str, int]
) -> Dict[str, np.ndarray]:
    """
    Pre-compute slice indices [start, end] for each sample across all timeframes.

    This eliminates np.searchsorted() calls in __getitem__.

    Args:
        cache_dir: Path to cache directory
        cache_key: Cache key string
        tf_meta: Timeframe metadata dict
        timestamps: Dict of timestamp arrays per timeframe
        valid_indices: Array of valid sample indices (into 5min array)
        sequence_lengths: Dict of sequence lengths per timeframe

    Returns:
        Dict with:
            - 'tf_indices': Dict[tf, ndarray[n_samples, 2]] with [start, end] for each sample
            - 'valid_indices': Copy of valid_indices for validation
            - 'sequence_lengths': Copy of sequence_lengths for validation
            - 'n_samples': Number of samples
    """
    print(f"\nPre-computing sample indices for {len(valid_indices):,} samples...")

    n_samples = len(valid_indices)
    ts_5min = timestamps['5min']

    result = {
        'tf_indices': {},
        'valid_indices': valid_indices.copy(),
        'sequence_lengths': sequence_lengths.copy(),
        'n_samples': n_samples,
    }

    start_time = time.time()

    for tf in HIERARCHICAL_TIMEFRAMES:
        if tf not in timestamps:
            print(f"  Skipping {tf} (no timestamps)")
            continue

        seq_len = sequence_lengths.get(tf, 75)  # Default to 75 if not specified
        tf_timestamps = timestamps[tf]

        # Pre-allocate index array: [n_samples, 2] for [start, end]
        tf_indices = np.zeros((n_samples, 2), dtype=np.int32)

        print(f"  Computing {tf} indices (seq_len={seq_len})...", end=" ", flush=True)
        tf_start = time.time()

        for i, data_idx_5min in enumerate(valid_indices):
            # Get 5min timestamp
            ts_5min_val = ts_5min[data_idx_5min]

            # Find index in this timeframe (same as _getitem_native_timeframe)
            tf_idx = np.searchsorted(tf_timestamps, ts_5min_val, side='right') - 1
            tf_idx = max(seq_len, tf_idx)  # Ensure we have enough history

            # Compute slice indices
            start = tf_idx - seq_len
            end = tf_idx

            tf_indices[i, 0] = start
            tf_indices[i, 1] = end

        tf_elapsed = time.time() - tf_start
        result['tf_indices'][tf] = tf_indices
        print(f"{tf_elapsed:.1f}s ({n_samples / tf_elapsed:.0f} samples/sec)")

    elapsed = time.time() - start_time
    print(f"  Total: {elapsed:.1f}s for {len(result['tf_indices'])} timeframes")

    return result


def save_sample_indices(cache_dir: Path, cache_key: str, data: Dict):
    """
    Save pre-computed sample indices to cache directory.

    Args:
        cache_dir: Path to cache directory
        cache_key: Cache key string
        data: Dict from compute_sample_indices()
    """
    print("\nSaving pre-computed sample indices...")

    # Save as .npz with tf_indices flattened
    save_data = {
        'valid_indices': data['valid_indices'],
        'n_samples': np.array([data['n_samples']], dtype=np.int32),
    }

    # Save sequence lengths as a structured format
    seq_len_keys = []
    seq_len_vals = []
    for tf, sl in data['sequence_lengths'].items():
        seq_len_keys.append(tf)
        seq_len_vals.append(sl)
    save_data['sequence_lengths_keys'] = np.array(seq_len_keys, dtype='U10')
    save_data['sequence_lengths_vals'] = np.array(seq_len_vals, dtype=np.int32)

    # Save each timeframe's indices
    for tf, indices in data['tf_indices'].items():
        save_data[f'indices_{tf}'] = indices

    indices_path = cache_dir / f"sample_indices_{cache_key}.npz"
    np.savez_compressed(indices_path, **save_data)
    size_mb = indices_path.stat().st_size / 1e6
    print(f"  Saved {indices_path.name} ({size_mb:.1f} MB)")

    # Save metadata JSON for easy inspection
    meta = {
        'version': '5.9.5',
        'cache_key': cache_key,
        'n_samples': int(data['n_samples']),
        'sequence_lengths': data['sequence_lengths'],
        'timeframes': list(data['tf_indices'].keys()),
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    meta_path = cache_dir / f"sample_indices_meta_{cache_key}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {meta_path.name}")


def load_sample_indices(cache_dir: Path, cache_key: str) -> Optional[Dict]:
    """
    Load pre-computed sample indices from cache directory.

    Args:
        cache_dir: Path to cache directory
        cache_key: Cache key string

    Returns:
        Dict with tf_indices, valid_indices, sequence_lengths, n_samples
        or None if file doesn't exist
    """
    indices_path = cache_dir / f"sample_indices_{cache_key}.npz"
    if not indices_path.exists():
        return None

    print(f"     Loading pre-computed sample indices...")
    data = dict(np.load(indices_path))

    # Reconstruct the structure
    result = {
        'valid_indices': data['valid_indices'],
        'n_samples': int(data['n_samples'][0]),
        'tf_indices': {},
        'sequence_lengths': {},
    }

    # Reconstruct sequence_lengths dict
    for key, val in zip(data['sequence_lengths_keys'], data['sequence_lengths_vals']):
        result['sequence_lengths'][str(key)] = int(val)

    # Extract timeframe indices
    for key, arr in data.items():
        if key.startswith('indices_'):
            tf = key.replace('indices_', '')
            result['tf_indices'][tf] = arr

    print(f"        Loaded {len(result['tf_indices'])} timeframes, {result['n_samples']:,} samples")
    return result


def validate_sample_indices(data: Dict, current_sequence_lengths: Dict[str, int]) -> bool:
    """
    Validate that pre-computed indices match current configuration.

    Args:
        data: Loaded sample indices dict
        current_sequence_lengths: Current sequence length configuration

    Returns:
        True if valid, False if regeneration needed
    """
    # Check sequence lengths match
    stored = data['sequence_lengths']
    for tf, sl in current_sequence_lengths.items():
        if tf in stored and stored[tf] != sl:
            print(f"     ⚠️  Sequence length mismatch for {tf}: stored={stored[tf]}, current={sl}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Pre-compute sample indices for fast training")
    parser.add_argument('--cache-dir', type=str, default='data/feature_cache',
                        help='Path to feature cache directory')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration even if indices exist')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"ERROR: Cache directory not found: {cache_dir}")
        sys.exit(1)

    print("=" * 70)
    print("Pre-compute Sample Indices for Fast Training (v5.9.5)")
    print("=" * 70)
    print(f"Cache directory: {cache_dir}")
    print()

    # Load metadata
    tf_meta, cache_key = load_tf_metadata(cache_dir)

    # Check if indices already exist
    indices_path = cache_dir / f"sample_indices_{cache_key}.npz"
    if indices_path.exists() and not args.force:
        print(f"\n⚠️  Sample indices already exist: {indices_path.name}")
        print("   Use --force to regenerate")
        return

    # Load timestamps
    timestamps = load_timestamps(cache_dir, cache_key)

    # Load valid indices (from precomputed targets or compute fresh)
    valid_indices_path = cache_dir / f"precomputed_valid_indices_{cache_key}.npy"
    if valid_indices_path.exists():
        valid_indices = np.load(valid_indices_path)
        print(f"\nLoaded valid indices: {len(valid_indices):,} samples")
    else:
        # Compute valid indices (same logic as HierarchicalDataset)
        seq_lengths = tf_meta.get('sequence_lengths', {})
        max_seq_len = max(seq_lengths.values()) if seq_lengths else 200
        n_5min_bars = len(timestamps.get('5min', []))
        prediction_horizon = 24
        horizon_5min = prediction_horizon // 5 + 1

        valid_start = max_seq_len
        valid_end = n_5min_bars - horizon_5min
        valid_indices = np.arange(valid_start, valid_end)
        print(f"\nComputed valid indices: {len(valid_indices):,} samples")

    # Get sequence lengths
    sequence_lengths = tf_meta.get('sequence_lengths', {})
    if not sequence_lengths:
        # Default to LOW preset
        sequence_lengths = {tf: 75 for tf in HIERARCHICAL_TIMEFRAMES}

    # Compute sample indices
    data = compute_sample_indices(
        cache_dir, cache_key, tf_meta,
        timestamps, valid_indices, sequence_lengths
    )

    # Save results
    save_sample_indices(cache_dir, cache_key, data)

    print()
    print("=" * 70)
    print("Sample indices pre-computation complete!")
    print("=" * 70)
    print()
    print("Expected speedup: ~5% faster __getitem__ (eliminates searchsorted)")
    print()
    print("The HierarchicalDataset will automatically detect and load these")
    print("indices when they exist in the cache directory.")


if __name__ == '__main__':
    main()
