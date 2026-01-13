#!/usr/bin/env python3
"""
Test script to verify load_market_data() can successfully load all required data.
"""

import sys
from pathlib import Path
import pandas as pd

# Add v7 to path so we can import from it
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the function
from v7.training.dataset import load_market_data


def test_load_market_data():
    """Test loading market data with all required symbols."""

    print("=" * 80)
    print("TEST: load_market_data() - Loading all required data")
    print("=" * 80)

    # Set up data directory
    data_dir = Path(__file__).parent / "data"

    print(f"\nData directory: {data_dir}")
    print(f"Directory exists: {data_dir.exists()}")

    # Check for required files
    required_files = {
        "TSLA_1min.csv": data_dir / "TSLA_1min.csv",
        "SPY_1min.csv": data_dir / "SPY_1min.csv",
        "VIX_History.csv": data_dir / "VIX_History.csv"
    }

    print("\nChecking for required files:")
    all_files_exist = True
    for name, path in required_files.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path} ({exists})")
        if exists:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"      Size: {size_mb:.2f} MB")
        all_files_exist = all_files_exist and exists

    if not all_files_exist:
        print("\n✗ FAILED: Not all required files exist")
        return False

    # Test 1: Load without date filters
    print("\n" + "-" * 80)
    print("Test 1: Load all data (no date filters)")
    print("-" * 80)

    try:
        tsla_df, spy_df, vix_df = load_market_data(data_dir, verbose=True)

        print("\n✓ SUCCESS: All data loaded successfully")

        # Verify shapes
        print("\nData shapes:")
        print(f"  TSLA: {tsla_df.shape} (rows, cols)")
        print(f"  SPY:  {spy_df.shape} (rows, cols)")
        print(f"  VIX:  {vix_df.shape} (rows, cols)")

        # Verify alignment
        if len(tsla_df) == len(spy_df) == len(vix_df):
            print(f"\n✓ Alignment verified: All series have {len(tsla_df)} rows")
        else:
            print(f"\n✗ WARNING: Series have different lengths!")
            print(f"  TSLA: {len(tsla_df)}, SPY: {len(spy_df)}, VIX: {len(vix_df)}")

        # Check for NaN values
        tsla_nans = tsla_df.isna().sum().sum()
        spy_nans = spy_df.isna().sum().sum()
        vix_nans = vix_df.isna().sum().sum()

        print(f"\nNaN values:")
        print(f"  TSLA: {tsla_nans}")
        print(f"  SPY:  {spy_nans}")
        print(f"  VIX:  {vix_nans}")

        if tsla_nans + spy_nans + vix_nans > 0:
            print("  ✗ WARNING: NaN values detected!")
        else:
            print("  ✓ No NaN values")

        # Show date ranges
        print(f"\nDate ranges:")
        print(f"  TSLA: {tsla_df.index[0]} to {tsla_df.index[-1]}")
        print(f"  SPY:  {spy_df.index[0]} to {spy_df.index[-1]}")
        print(f"  VIX:  {vix_df.index[0]} to {vix_df.index[-1]}")

        # Check columns
        print(f"\nColumns:")
        print(f"  TSLA: {list(tsla_df.columns)}")
        print(f"  SPY:  {list(spy_df.columns)}")
        print(f"  VIX:  {list(vix_df.columns)}")

        test1_success = True

    except Exception as e:
        print(f"\n✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        test1_success = False

    # Test 2: Load with date filters
    print("\n" + "-" * 80)
    print("Test 2: Load with date filters (2023-01-01 to 2023-12-31)")
    print("-" * 80)

    try:
        tsla_df2, spy_df2, vix_df2 = load_market_data(
            data_dir,
            start_date="2023-01-01",
            end_date="2023-12-31",
            verbose=True
        )

        print("\n✓ SUCCESS: Filtered data loaded successfully")

        # Verify shapes
        print("\nFiltered data shapes:")
        print(f"  TSLA: {tsla_df2.shape} (rows, cols)")
        print(f"  SPY:  {spy_df2.shape} (rows, cols)")
        print(f"  VIX:  {vix_df2.shape} (rows, cols)")

        # Verify alignment
        if len(tsla_df2) == len(spy_df2) == len(vix_df2):
            print(f"\n✓ Alignment verified: All series have {len(tsla_df2)} rows")
        else:
            print(f"\n✗ WARNING: Series have different lengths!")

        # Show date ranges
        print(f"\nDate ranges:")
        print(f"  TSLA: {tsla_df2.index[0]} to {tsla_df2.index[-1]}")
        print(f"  SPY:  {spy_df2.index[0]} to {spy_df2.index[-1]}")
        print(f"  VIX:  {vix_df2.index[0]} to {vix_df2.index[-1]}")

        test2_success = True

    except Exception as e:
        print(f"\n✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        test2_success = False

    # Test 3: Load just TSLA and SPY (to verify independent loading works)
    print("\n" + "-" * 80)
    print("Test 3: Verify individual file existence and basic loading")
    print("-" * 80)

    try:
        # Check TSLA
        tsla_raw = pd.read_csv(data_dir / "TSLA_1min.csv", nrows=5)
        print(f"✓ TSLA file readable: {len(tsla_raw)} rows (sample)")
        print(f"  Columns: {list(tsla_raw.columns)}")

        # Check SPY
        spy_raw = pd.read_csv(data_dir / "SPY_1min.csv", nrows=5)
        print(f"✓ SPY file readable: {len(spy_raw)} rows (sample)")
        print(f"  Columns: {list(spy_raw.columns)}")

        # Check VIX
        vix_raw = pd.read_csv(data_dir / "VIX_History.csv", nrows=5)
        print(f"✓ VIX file readable: {len(vix_raw)} rows (sample)")
        print(f"  Columns: {list(vix_raw.columns)}")

        test3_success = True

    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        test3_success = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    results = {
        "Test 1 (Load all data)": test1_success,
        "Test 2 (Load with filters)": test2_success,
        "Test 3 (Individual files)": test3_success,
    }

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("load_market_data() can successfully load all required data (TSLA, SPY, VIX)")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = test_load_market_data()
    sys.exit(0 if success else 1)
