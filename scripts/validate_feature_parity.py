#!/usr/bin/env python3
"""
Validate feature parity between C++ and Python feature extractors.

Loads a .flat sample file, extracts features via both paths, and compares:
- Feature count match
- Feature name match
- Value match (max absolute difference per feature)

Usage:
    python scripts/validate_feature_parity.py [--flat PATH] [--checkpoint PATH]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def compare_features(cpp_features: dict, py_features: dict):
    """Compare C++ and Python feature dictionaries."""
    cpp_names = set(cpp_features.keys())
    py_names = set(py_features.keys())

    print(f"\n{'='*60}")
    print("FEATURE PARITY REPORT")
    print(f"{'='*60}")
    print(f"C++ features:    {len(cpp_names):,}")
    print(f"Python features: {len(py_names):,}")

    common = cpp_names & py_names
    only_cpp = cpp_names - py_names
    only_py = py_names - cpp_names

    print(f"Common features: {len(common):,}")
    print(f"Only in C++:     {len(only_cpp):,}")
    print(f"Only in Python:  {len(only_py):,}")

    if only_cpp:
        print(f"\n--- Features only in C++ (first 20) ---")
        for name in sorted(only_cpp)[:20]:
            print(f"  {name}: {cpp_features[name]:.6f}")

    if only_py:
        print(f"\n--- Features only in Python (first 20) ---")
        for name in sorted(only_py)[:20]:
            print(f"  {name}: {py_features[name]:.6f}")

    # Compare values for common features
    if common:
        diffs = {}
        for name in common:
            diff = abs(cpp_features[name] - py_features[name])
            if diff > 0:
                diffs[name] = diff

        if diffs:
            max_diff_name = max(diffs, key=diffs.get)
            max_diff = diffs[max_diff_name]
            avg_diff = sum(diffs.values()) / len(diffs)
            print(f"\n--- Value Differences ---")
            print(f"Features with non-zero diff: {len(diffs):,}/{len(common):,}")
            print(f"Max absolute diff: {max_diff:.6e} ({max_diff_name})")
            print(f"Avg absolute diff: {avg_diff:.6e}")

            # Show top 10 differences
            print(f"\nTop 10 largest differences:")
            for name in sorted(diffs, key=diffs.get, reverse=True)[:10]:
                print(f"  {name}: C++={cpp_features[name]:.6f} Py={py_features[name]:.6f} diff={diffs[name]:.6e}")
        else:
            print(f"\nAll {len(common):,} common features match exactly!")

    # Overall verdict
    print(f"\n{'='*60}")
    if len(only_cpp) == 0 and len(only_py) == 0:
        max_d = max(diffs.values()) if diffs else 0
        if max_d < 1e-6:
            print("PASS: Feature parity confirmed (all features match within 1e-6)")
        else:
            print(f"WARN: Features match by name but values differ (max diff: {max_d:.6e})")
    else:
        print(f"FAIL: Feature set mismatch ({len(only_cpp)} C++-only, {len(only_py)} Python-only)")
    print(f"{'='*60}\n")


def validate_with_live_data():
    """Validate using live yfinance data (no .flat file needed)."""
    import pandas as pd
    import numpy as np

    print("Fetching live data for validation...")

    try:
        import v15scanner_cpp
        print(f"C++ module loaded: {v15scanner_cpp.get_feature_count()} expected features")
    except ImportError:
        print("ERROR: C++ module not available. Build with: pip install -e .")
        return False

    from v15.features.tf_extractor import extract_all_tf_features

    # Try to fetch live data
    try:
        from v15.live_data import YFinanceLiveData
        data_feed = YFinanceLiveData(cache_ttl=60)
        tsla_df, spy_df, vix_df = data_feed.get_historical(period='60d', interval='5m')
        print(f"Fetched: TSLA={len(tsla_df)}, SPY={len(spy_df)}, VIX={len(vix_df)} bars")
    except Exception as e:
        print(f"Could not fetch live data: {e}")
        print("Generating synthetic data instead...")

        # Generate synthetic data
        n = 5000
        dates = pd.date_range('2024-01-01', periods=n, freq='5min')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
        tsla_df = pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(np.random.randn(n) * 0.5),
            'low': prices - np.abs(np.random.randn(n) * 0.5),
            'close': prices + np.random.randn(n) * 0.2,
            'volume': np.random.randint(100000, 10000000, n).astype(float),
        }, index=dates)
        spy_df = tsla_df.copy()
        spy_df['close'] = 450 + np.cumsum(np.random.randn(n) * 0.05)
        vix_df = tsla_df.copy()
        vix_df['close'] = 20 + np.random.randn(n) * 2

    timestamp = tsla_df.index[-1]
    source_bar_count = len(tsla_df)

    # Extract via C++
    print("\nExtracting features via C++...")
    timestamp_ms = int(timestamp.timestamp() * 1000) if hasattr(timestamp, 'timestamp') else int(timestamp)
    cpp_features = v15scanner_cpp.extract_features(
        tsla_df, spy_df, vix_df, timestamp_ms, source_bar_count
    )
    print(f"C++ extracted: {len(cpp_features)} features")

    # Extract via Python
    print("Extracting features via Python...")
    py_features = extract_all_tf_features(
        tsla_df=tsla_df,
        spy_df=spy_df,
        vix_df=vix_df,
        timestamp=timestamp,
        source_bar_count=source_bar_count,
        include_bar_metadata=True,
    )
    print(f"Python extracted: {len(py_features)} features")

    compare_features(cpp_features, py_features)
    return True


def validate_with_checkpoint(checkpoint_path: str):
    """Validate C++ features match what the checkpoint expects."""
    import torch

    try:
        import v15scanner_cpp
    except ImportError:
        print("ERROR: C++ module not available. Build with: pip install -e .")
        return False

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    feature_names = ckpt.get('feature_names', [])
    print(f"Checkpoint expects {len(feature_names)} features")

    cpp_names = v15scanner_cpp.get_feature_names()
    print(f"C++ produces {len(cpp_names)} feature names")

    ckpt_set = set(feature_names)
    cpp_set = set(cpp_names)

    common = ckpt_set & cpp_set
    only_ckpt = ckpt_set - cpp_set
    only_cpp = cpp_set - ckpt_set

    print(f"\nCommon:       {len(common):,}")
    print(f"Only in ckpt: {len(only_ckpt):,}")
    print(f"Only in C++:  {len(only_cpp):,}")

    if only_ckpt:
        print(f"\nFeatures model expects but C++ doesn't produce (first 20):")
        for name in sorted(only_ckpt)[:20]:
            print(f"  {name}")

    if only_cpp:
        print(f"\nFeatures C++ produces but model doesn't expect (first 20):")
        for name in sorted(only_cpp)[:20]:
            print(f"  {name}")

    match_pct = len(common) / len(feature_names) * 100 if feature_names else 0
    print(f"\nFeature name match: {match_pct:.1f}%")
    return match_pct > 99


def main():
    parser = argparse.ArgumentParser(description="Validate C++ vs Python feature parity")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint for name validation")
    parser.add_argument("--live", action="store_true", default=True, help="Use live/synthetic data (default)")
    args = parser.parse_args()

    if args.checkpoint:
        validate_with_checkpoint(args.checkpoint)

    validate_with_live_data()


if __name__ == "__main__":
    main()
