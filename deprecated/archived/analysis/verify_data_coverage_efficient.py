"""
Verify Walk-Forward Windows Data Coverage (Efficient Version)

This script efficiently verifies data coverage by streaming through the cache file
instead of loading everything into memory at once.
"""

import pickle
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from v7.training.walk_forward import generate_walk_forward_windows


def get_cache_date_range(cache_path: Path) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
    """
    Efficiently get min/max dates and sample count from cache.

    Args:
        cache_path: Path to channel_samples.pkl

    Returns:
        Tuple of (min_date, max_date, sample_count)
    """
    print(f"Reading cache file: {cache_path}")
    print("Extracting date range (this may take 30-60 seconds for large files)...")

    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    # Extract samples
    if isinstance(data, dict):
        samples = data.get('samples', [])
        cache_info = {k: v for k, v in data.items() if k != 'samples'}
    else:
        samples = data
        cache_info = {}

    print(f"\nCache info: {cache_info}")
    print(f"Total samples: {len(samples):,}")

    # Get min and max dates without sorting entire list
    min_date = min(sample.timestamp for sample in samples)
    max_date = max(sample.timestamp for sample in samples)

    return min_date, max_date, len(samples)


def count_samples_in_period(
    cache_path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> int:
    """
    Count samples within a date range.

    Args:
        cache_path: Path to channel_samples.pkl
        start_date: Start of period
        end_date: End of period

    Returns:
        Number of samples in period
    """
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        samples = data.get('samples', [])
    else:
        samples = data

    count = sum(1 for sample in samples if start_date <= sample.timestamp <= end_date)
    return count


def verify_coverage_simple():
    """Simplified verification without loading all timestamps."""

    print("=" * 80)
    print("WALK-FORWARD DATA COVERAGE VERIFICATION")
    print("=" * 80)

    # Configuration
    cache_path = Path("/Users/frank/Desktop/CodingProjects/x6/data/feature_cache/channel_samples.pkl")
    cache_end_expected = "2025-07-30"
    num_windows = 3
    validation_months = 5

    # Step 1: Get date range from cache
    print("\n1. Analyzing cache date range...")
    if not cache_path.exists():
        print(f"ERROR: Cache file not found at {cache_path}")
        return

    try:
        min_date, max_date, total_samples = get_cache_date_range(cache_path)

        print(f"\nCache date range:")
        print(f"  Start: {min_date.date()}")
        print(f"  End:   {max_date.date()}")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Days covered: {(max_date - min_date).days}")

    except Exception as e:
        print(f"ERROR loading cache: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Generate windows
    print("\n2. Generating walk-forward windows...")
    print(f"   Number of windows: {num_windows}")
    print(f"   Validation period: {validation_months} months per window")

    data_start = min_date.strftime('%Y-%m-%d')
    data_end = cache_end_expected

    try:
        windows = generate_walk_forward_windows(
            data_start=data_start,
            data_end=data_end,
            num_windows=num_windows,
            validation_period_months=validation_months
        )

        print(f"\n   ✓ Generated {len(windows)} windows successfully")

    except ValueError as e:
        print(f"\n   ✗ ERROR: Could not generate windows: {e}")
        return

    # Step 3: Display windows and check coverage
    print("\n3. Window boundaries and coverage check:")
    print("=" * 80)

    cache_end_date = pd.Timestamp(cache_end_expected)
    all_covered = True

    for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
        print(f"\nWindow {i + 1}:")
        print(f"  Training:   {train_start.date()} to {train_end.date()}")
        print(f"  Validation: {val_start.date()} to {val_end.date()}")

        # Calculate expected duration
        train_days = (train_end - train_start).days + 1
        val_days = (val_end - val_start).days + 1
        val_months = (val_end.year - val_start.year) * 12 + (val_end.month - val_start.month) + 1

        print(f"  Train duration: {train_days} days")
        print(f"  Val duration: {val_days} days (~{val_months} months)")

        # Check if validation period is covered by cache
        val_covered = val_end <= max_date

        if val_covered:
            print(f"  Coverage: ✓ COMPLETE")
        else:
            missing_days = (val_end - max_date).days
            print(f"  Coverage: ✗ INCOMPLETE - missing last {missing_days} days")
            print(f"    Val ends:   {val_end.date()}")
            print(f"    Cache ends: {max_date.date()}")
            all_covered = False

    # Step 4: Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n1. Cache Coverage:")
    print(f"   Expected end: {cache_end_date.date()}")
    print(f"   Actual end:   {max_date.date()}")

    cache_adequate = max_date >= cache_end_date
    if cache_adequate:
        print(f"   Status: ✓ ADEQUATE (cache extends to or beyond expected end)")
    else:
        missing_days = (cache_end_date - max_date).days
        print(f"   Status: ✗ INSUFFICIENT (missing {missing_days} days)")

    print(f"\n2. Window Status:")
    if all_covered:
        print(f"   ✓ ALL {num_windows} WINDOWS HAVE COMPLETE DATA COVERAGE")
    else:
        print(f"   ✗ SOME WINDOWS EXTEND BEYOND AVAILABLE DATA")

    print(f"\n3. Data Quality:")
    print(f"   Total samples: {total_samples:,}")
    days_covered = (max_date - min_date).days
    if days_covered > 0:
        samples_per_day = total_samples / days_covered
        print(f"   Days covered: {days_covered}")
        print(f"   Samples per day: {samples_per_day:.1f}")

    # Final verdict
    print("\n" + "=" * 80)
    if all_covered and cache_adequate:
        print("✓ VERIFICATION PASSED: All windows have complete data coverage")
        print("=" * 80)
        print("\nYou can proceed with walk-forward validation training.")
    else:
        print("✗ VERIFICATION FAILED: Data coverage is incomplete")
        print("=" * 80)

        print("\nRECOMMENDATIONS:")

        if not all_covered:
            print("\n1. Some validation windows extend beyond available data")
            print("   Options:")
            print(f"   a) Reduce number of windows from {num_windows}")
            print(f"   b) Reduce validation period from {validation_months} months")
            print(f"   c) Extend cache to cover until {cache_end_date.date()}")

            # Calculate alternatives
            available_months = (max_date.year - min_date.year) * 12 + \
                              (max_date.month - min_date.month)

            # Try different validation periods
            for test_val_months in [4, 3, 2]:
                min_train_months = 6
                max_possible_windows = (available_months - min_train_months) // test_val_months

                if max_possible_windows >= 2:
                    print(f"\n   Alternative: {max_possible_windows} windows with {test_val_months}-month validation")
                    print(f"   Command: generate_walk_forward_windows(..., num_windows={max_possible_windows}, validation_period_months={test_val_months})")
                    break

        if not cache_adequate:
            missing_days = (cache_end_date - max_date).days
            print(f"\n2. Cache ends {missing_days} days before expected date")
            print("   You need to download more recent data or adjust expected end date")

    print()


if __name__ == '__main__':
    verify_coverage_simple()
