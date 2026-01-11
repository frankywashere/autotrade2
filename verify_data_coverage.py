"""
Verify Walk-Forward Windows Data Coverage

This script verifies that the walk-forward windows have complete data coverage
by checking the actual data in the cache against the expected window boundaries.

Given:
- Window 1 val: 5 months validation
- Window 2 val: 5 months validation
- Window 3 val: 5 months validation
- Cache ends at: 2025-07-30

Check:
1. Calculate actual window boundaries for 3 windows with 5-month validation
2. Verify each window has complete data in the cache
3. Identify any gaps or missing periods
4. Check if cache end date (2025-07-30) covers all validation periods
"""

import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from v7.training.walk_forward import generate_walk_forward_windows


def load_sample_timestamps(cache_path: Path) -> List[pd.Timestamp]:
    """
    Load just the timestamps from the cache file.

    Args:
        cache_path: Path to channel_samples.pkl

    Returns:
        List of timestamps from all samples
    """
    print(f"Loading cache file: {cache_path}")
    print("(This may take a moment for large files...)")

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
    print(f"Total samples in cache: {len(samples):,}")

    # Extract timestamps
    timestamps = [sample.timestamp for sample in samples]
    timestamps.sort()

    return timestamps, cache_info


def analyze_date_coverage(
    timestamps: List[pd.Timestamp],
    start_date: str,
    end_date: str
) -> Dict:
    """
    Analyze date coverage in the dataset.

    Args:
        timestamps: List of sample timestamps
        start_date: Expected start date (YYYY-MM-DD)
        end_date: Expected end date (YYYY-MM-DD)

    Returns:
        Dictionary with coverage analysis
    """
    if not timestamps:
        return {
            'error': 'No timestamps found',
            'total_samples': 0
        }

    actual_start = timestamps[0]
    actual_end = timestamps[-1]

    expected_start = pd.Timestamp(start_date)
    expected_end = pd.Timestamp(end_date)

    # Count samples per month
    monthly_counts = defaultdict(int)
    for ts in timestamps:
        month_key = (ts.year, ts.month)
        monthly_counts[month_key] += 1

    # Identify gaps (months with < 10 samples as potential gaps)
    gaps = []
    current_month = expected_start
    while current_month <= expected_end:
        month_key = (current_month.year, current_month.month)
        count = monthly_counts.get(month_key, 0)
        if count < 10:
            gaps.append((current_month, count))
        current_month = current_month + pd.DateOffset(months=1)

    return {
        'total_samples': len(timestamps),
        'actual_start': actual_start,
        'actual_end': actual_end,
        'expected_start': expected_start,
        'expected_end': expected_end,
        'starts_before_expected': actual_start < expected_start,
        'ends_after_expected': actual_end >= expected_end,
        'monthly_counts': dict(monthly_counts),
        'gaps': gaps,
        'total_days': (actual_end - actual_start).days,
        'samples_per_day': len(timestamps) / max((actual_end - actual_start).days, 1)
    }


def verify_window_coverage(
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    timestamps: List[pd.Timestamp],
    cache_end: str
) -> Dict:
    """
    Verify that each window has complete data coverage.

    Args:
        windows: List of (train_start, train_end, val_start, val_end) tuples
        timestamps: List of sample timestamps
        cache_end: Expected cache end date

    Returns:
        Dictionary with window coverage analysis
    """
    cache_end_date = pd.Timestamp(cache_end)
    actual_end = max(timestamps)

    print("\n" + "=" * 80)
    print("WINDOW COVERAGE VERIFICATION")
    print("=" * 80)

    print(f"\nCache end date (expected): {cache_end}")
    print(f"Cache end date (actual):   {actual_end.date()}")
    print(f"Coverage adequate: {actual_end >= cache_end_date}")

    window_analysis = []

    for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
        print(f"\n{'=' * 80}")
        print(f"Window {i + 1}")
        print(f"{'=' * 80}")

        # Count samples in each period
        train_samples = [ts for ts in timestamps if train_start <= ts <= train_end]
        val_samples = [ts for ts in timestamps if val_start <= ts <= val_end]

        # Calculate expected duration
        train_days = (train_end - train_start).days + 1
        val_days = (val_end - val_start).days + 1
        val_months = (val_end.year - val_start.year) * 12 + (val_end.month - val_start.month) + 1

        # Check if we have data
        has_train_data = len(train_samples) > 0
        has_val_data = len(val_samples) > 0
        val_covered = val_end <= actual_end

        print(f"Training Period:   {train_start.date()} to {train_end.date()} ({train_days} days)")
        print(f"  Samples: {len(train_samples):,}")
        print(f"  Coverage: {'✓ Complete' if has_train_data else '✗ NO DATA'}")

        print(f"\nValidation Period: {val_start.date()} to {val_end.date()} ({val_days} days, ~{val_months} months)")
        print(f"  Samples: {len(val_samples):,}")
        print(f"  Coverage: {'✓ Complete' if has_val_data and val_covered else '✗ INCOMPLETE'}")

        if not val_covered:
            missing_days = (val_end - actual_end).days
            print(f"  ⚠ WARNING: Missing last {missing_days} days of validation data!")
            print(f"    Val ends:   {val_end.date()}")
            print(f"    Cache ends: {actual_end.date()}")

        # Monthly breakdown for validation period
        if len(val_samples) > 0:
            val_monthly = defaultdict(int)
            for ts in val_samples:
                month_key = f"{ts.year}-{ts.month:02d}"
                val_monthly[month_key] += 1

            print(f"\n  Monthly breakdown:")
            current = val_start
            while current <= val_end:
                month_key = f"{current.year}-{current.month:02d}"
                count = val_monthly.get(month_key, 0)
                status = "✓" if count > 0 else "✗"
                print(f"    {status} {month_key}: {count:,} samples")
                current = current + pd.DateOffset(months=1)

        window_analysis.append({
            'window_id': i + 1,
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end,
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'has_train_data': has_train_data,
            'has_val_data': has_val_data,
            'val_covered': val_covered,
            'val_months': val_months,
            'val_days': val_days
        })

    return {
        'cache_end_expected': cache_end_date,
        'cache_end_actual': actual_end,
        'cache_adequate': actual_end >= cache_end_date,
        'windows': window_analysis
    }


def print_summary(coverage_analysis: Dict, window_analysis: Dict):
    """Print summary of data coverage verification."""

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Overall cache status
    print("\n1. Cache Coverage:")
    cache_ok = coverage_analysis['ends_after_expected']
    print(f"   Expected end: {coverage_analysis['expected_end'].date()}")
    print(f"   Actual end:   {coverage_analysis['actual_end'].date()}")
    print(f"   Status: {'✓ ADEQUATE' if cache_ok else '✗ INSUFFICIENT'}")

    # Window status
    print("\n2. Window Status:")
    all_windows_ok = all(w['val_covered'] for w in window_analysis['windows'])

    for window in window_analysis['windows']:
        status = '✓' if window['val_covered'] else '✗'
        print(f"   {status} Window {window['window_id']}: ", end="")
        print(f"Val period {window['val_months']} months, {window['val_samples']:,} samples")

    print(f"\n   Overall: {'✓ ALL WINDOWS COVERED' if all_windows_ok else '✗ SOME WINDOWS INCOMPLETE'}")

    # Data quality
    print("\n3. Data Quality:")
    print(f"   Total samples: {coverage_analysis['total_samples']:,}")
    print(f"   Samples per day: {coverage_analysis['samples_per_day']:.1f}")

    if coverage_analysis['gaps']:
        print(f"   ⚠ Found {len(coverage_analysis['gaps'])} months with low sample counts:")
        for gap_date, count in coverage_analysis['gaps'][:5]:  # Show first 5
            print(f"     - {gap_date.strftime('%Y-%m')}: {count} samples")
        if len(coverage_analysis['gaps']) > 5:
            print(f"     ... and {len(coverage_analysis['gaps']) - 5} more")
    else:
        print(f"   ✓ No significant gaps detected")

    # Final verdict
    print("\n" + "=" * 80)
    if all_windows_ok and cache_ok:
        print("✓ VERIFICATION PASSED: All windows have complete data coverage")
    else:
        print("✗ VERIFICATION FAILED: Data coverage is incomplete")
        if not all_windows_ok:
            print("  - Some validation windows extend beyond available data")
        if not cache_ok:
            print("  - Cache does not cover the expected date range")
    print("=" * 80)


def main():
    """Main verification function."""

    print("=" * 80)
    print("WALK-FORWARD DATA COVERAGE VERIFICATION")
    print("=" * 80)

    # Configuration
    cache_path = Path("/Users/frank/Desktop/CodingProjects/x6/data/feature_cache/channel_samples.pkl")
    cache_end = "2025-07-30"
    num_windows = 3
    validation_months = 5

    # We need to determine data_start and data_end
    # For now, we'll load the cache first to see the actual range

    print("\n1. Loading cache timestamps...")
    if not cache_path.exists():
        print(f"ERROR: Cache file not found at {cache_path}")
        return

    timestamps, cache_info = load_sample_timestamps(cache_path)

    if not timestamps:
        print("ERROR: No timestamps found in cache")
        return

    data_start = timestamps[0].strftime('%Y-%m-%d')
    data_end = cache_end  # Use expected cache end

    print(f"\nData range: {data_start} to {data_end}")

    # Generate walk-forward windows
    print("\n2. Generating walk-forward windows...")
    print(f"   Number of windows: {num_windows}")
    print(f"   Validation period: {validation_months} months per window")

    try:
        windows = generate_walk_forward_windows(
            data_start=data_start,
            data_end=data_end,
            num_windows=num_windows,
            validation_period_months=validation_months
        )

        print(f"\n   Generated {len(windows)} windows successfully")

        # Print window boundaries
        print("\n   Window boundaries:")
        for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
            print(f"   Window {i + 1}:")
            print(f"     Train: {train_start.date()} to {train_end.date()}")
            print(f"     Val:   {val_start.date()} to {val_end.date()}")

    except ValueError as e:
        print(f"\n   ERROR: Could not generate windows: {e}")
        return

    # Analyze overall coverage
    print("\n3. Analyzing date coverage...")
    coverage = analyze_date_coverage(timestamps, data_start, data_end)

    # Verify window coverage
    print("\n4. Verifying window coverage...")
    window_verification = verify_window_coverage(windows, timestamps, cache_end)

    # Print summary
    print_summary(coverage, window_verification)

    # Additional recommendations
    if not window_verification['cache_adequate']:
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        actual_end = coverage['actual_end']
        expected_end = coverage['expected_end']
        missing_days = (expected_end - actual_end).days

        print(f"\n1. Cache ends on {actual_end.date()} but needs to extend to {expected_end.date()}")
        print(f"   Missing: {missing_days} days of data")
        print("\n2. Options to fix:")
        print(f"   a) Extend cache to {expected_end.date()} by downloading more recent data")
        print(f"   b) Reduce number of windows to fit available data")
        print(f"   c) Reduce validation period length (currently {validation_months} months)")

        # Calculate what would fit
        available_months = (actual_end.year - pd.Timestamp(data_start).year) * 12 + \
                          (actual_end.month - pd.Timestamp(data_start).month)
        max_windows_current_val = available_months // validation_months - 1  # -1 for training buffer

        print(f"\n3. With current data ({actual_end.date()}):")
        print(f"   - Can fit {max_windows_current_val} windows with {validation_months}-month validation")

        if validation_months > 3:
            alt_val_months = 3
            max_windows_alt = available_months // alt_val_months - 1
            print(f"   - Can fit {max_windows_alt} windows with {alt_val_months}-month validation")


if __name__ == '__main__':
    main()
