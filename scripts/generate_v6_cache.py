#!/usr/bin/env python3
"""
v6.0 Cache Generation Script

Generates unified v6.0 cache with duration-primary labels.

This script:
1. Loads existing feature data (from v5.x cache)
2. Computes v6 labels (duration, break tracking, transitions)
3. Saves unified .npz files per timeframe

Usage:
    python scripts/generate_v6_cache.py --features-path data/feature_cache/tf_meta_*.json --output data/feature_cache_v6

Requirements:
    - Existing v5.x feature cache (with channel features)
    - Raw OHLC data (for break detection)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import config as project_config
from src.ml.cache_v6 import generate_v6_cache
from src.ml.features import TradingFeatureExtractor, load_vix_data


def load_v5_features_for_v6_generation(tf_meta_path: str) -> tuple:
    """
    Load existing v5.x features to use as input for v6 cache generation.

    Args:
        tf_meta_path: Path to v5.x tf_meta_*.json

    Returns:
        (features_df, raw_ohlc_df)
    """
    print(f"\n{'='*60}")
    print(f"Loading v5.x Feature Cache")
    print(f"{'='*60}")

    with open(tf_meta_path) as f:
        tf_meta = json.load(f)

    cache_dir = Path(tf_meta_path).parent
    cache_key = tf_meta['cache_key']

    # Load native TF sequences (we only need timestamps and basic features)
    print(f"  Loading 5min timeframe data...")
    tf_5min_path = cache_dir / f"tf_sequence_5min_{cache_key}.npy"
    tf_5min = np.load(str(tf_5min_path), mmap_mode='r')

    tf_5min_ts_path = cache_dir / f"tf_timestamps_5min_{cache_key}.npy"
    timestamps = np.load(str(tf_5min_ts_path), mmap_mode='r')

    print(f"     ✓ Loaded {len(timestamps):,} samples")

    # Create minimal features DataFrame (just for indexing)
    # We don't need all features, just timestamps
    features_df = pd.DataFrame(
        index=pd.to_datetime(timestamps, unit='ns')
    )

    print(f"     ✓ Created features DataFrame: {len(features_df):,} rows")

    # Load raw OHLC for break detection
    print(f"  Loading raw OHLC data...")
    ohlc_path = project_config.DATA_DIR / "TSLA_1min.csv"
    if not ohlc_path.exists():
        ohlc_path = project_config.DATA_DIR / "tsla_1min_data.csv"

    if ohlc_path.exists():
        raw_df = pd.read_csv(ohlc_path, parse_dates=['timestamp'], index_col='timestamp')
        # Rename columns if needed
        col_map = {
            'open': 'tsla_open',
            'high': 'tsla_high',
            'low': 'tsla_low',
            'close': 'tsla_close',
        }
        for old, new in col_map.items():
            if old in raw_df.columns and new not in raw_df.columns:
                raw_df.rename(columns={old: new}, inplace=True)

        print(f"     ✓ Loaded {len(raw_df):,} 1-min bars")
    else:
        raise FileNotFoundError(f"Raw OHLC file not found: {ohlc_path}")

    return features_df, raw_df


def main():
    parser = argparse.ArgumentParser(description='Generate v6.0 duration-primary cache')
    parser.add_argument('--features-path', type=str, required=True,
                       help='Path to v5.x tf_meta_*.json file (e.g., data/feature_cache/tf_meta_*.json)')
    parser.add_argument('--output', type=str, default='data/feature_cache_v6',
                       help='Output directory for v6 cache')
    parser.add_argument('--max-scan-bars', type=int, default=500,
                       help='Maximum bars to scan forward for breaks')
    parser.add_argument('--return-threshold', type=int, default=3,
                       help='Bars inside to count as returned')
    parser.add_argument('--validate', action='store_true',
                       help='Validate cache after generation')

    args = parser.parse_args()

    # FIX #4: Pre-flight validation checks
    print(f"\n{'='*60}")
    print(f"Pre-flight Validation Checks")
    print(f"{'='*60}")

    # Resolve paths
    features_path = Path(args.features_path)
    if not features_path.exists():
        print(f"✗ Features path not found: {features_path}")
        # Try to find it
        pattern = "tf_meta_*.json"
        matches = list(Path('data/feature_cache').glob(pattern))
        if matches:
            print(f"\nFound candidates:")
            for m in matches:
                print(f"  - {m}")
            print(f"\nUse: --features-path {matches[0]}")
        return 1
    print(f"✓ Features metadata found: {features_path}")

    # Check v5.9 cache directory
    v5_cache_dir = features_path.parent
    if not v5_cache_dir.exists():
        print(f"✗ v5.9 cache directory not found: {v5_cache_dir}")
        return 1
    print(f"✓ v5.9 cache directory: {v5_cache_dir}")

    # Check that v5.9 features exist for all timeframes
    timeframes = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']
    missing_tfs = []
    for tf in timeframes:
        pattern = f"tf_sequence_{tf}_v5.9*.npy"
        matches = list(v5_cache_dir.glob(pattern))
        if not matches:
            missing_tfs.append(tf)

    if missing_tfs:
        print(f"✗ Missing v5.9 features for timeframes: {missing_tfs}")
        print(f"  Please run feature extraction first!")
        return 1
    print(f"✓ v5.9 features exist for all {len(timeframes)} timeframes")

    # Check raw OHLC data
    ohlc_paths = [
        project_config.DATA_DIR / "TSLA_1min.csv",
        project_config.DATA_DIR / "tsla_1min_data.csv",
    ]
    ohlc_exists = any(p.exists() for p in ohlc_paths)
    if not ohlc_exists:
        print(f"✗ Raw OHLC data not found. Checked:")
        for p in ohlc_paths:
            print(f"  - {p}")
        return 1
    existing_ohlc = next(p for p in ohlc_paths if p.exists())
    print(f"✓ Raw OHLC data found: {existing_ohlc}")

    # Check disk space (estimate ~3GB needed)
    output_dir = Path(args.output)
    try:
        import shutil
        stat = shutil.disk_usage(output_dir.parent if output_dir.exists() else output_dir.parent.parent)
        free_gb = stat.free / (1024**3)
        if free_gb < 5:
            print(f"⚠️  Low disk space: {free_gb:.1f} GB free (recommend 5+ GB)")
        else:
            print(f"✓ Disk space: {free_gb:.1f} GB free")
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")

    print(f"\n{'='*60}")
    print(f"v6.0 Cache Generation")
    print(f"{'='*60}")
    print(f"Input: {features_path}")
    print(f"Output: {output_dir}")
    print(f"Max scan bars: {args.max_scan_bars}")
    print(f"Return threshold: {args.return_threshold} bars")

    # Load v5 features
    features_df, raw_ohlc_df = load_v5_features_for_v6_generation(str(features_path))

    # Generate v6 cache
    metadata = generate_v6_cache(
        features_df=features_df,
        raw_ohlc_df=raw_ohlc_df,
        output_dir=str(output_dir),
        v5_cache_dir=str(v5_cache_dir),  # FIX #3: Pass v5 cache dir to load features
        max_scan_bars=args.max_scan_bars,
        return_threshold_bars=args.return_threshold,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"✓ Cache Generation Complete!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Version: {metadata['version']}")
    print(f"Timeframes: {len(metadata['timeframes'])}")

    total_size_mb = sum(tf['file_size_mb'] for tf in metadata['timeframes'].values())
    print(f"Total size: {total_size_mb:.1f} MB")

    # Validate if requested
    if args.validate:
        from src.ml.cache_v6 import validate_v6_cache
        print(f"\nValidating cache...")
        valid = validate_v6_cache(str(output_dir))
        if valid:
            print(f"✓ Cache validation passed!")
            return 0
        else:
            print(f"✗ Cache validation failed!")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
