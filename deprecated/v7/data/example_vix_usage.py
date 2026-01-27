"""
VIX Fetcher Usage Examples

This script demonstrates how to use the VIX fetcher in different scenarios.
"""

import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v7.data import FREDVixFetcher, fetch_vix_data
from v7.features import extract_vix_features


def example_1_simple_usage():
    """Example 1: Simple usage with automatic fallback."""
    print("=" * 80)
    print("Example 1: Simple Usage")
    print("=" * 80)

    # Simplest usage - automatically tries all sources
    vix_df = fetch_vix_data(
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    print(f"\nFetched {len(vix_df)} VIX records")
    print(f"Date range: {vix_df.index.min().date()} to {vix_df.index.max().date()}")
    print(f"\nSample data:")
    print(vix_df.head())


def example_2_with_fred_api():
    """Example 2: Using FRED API (most reliable)."""
    print("\n" + "=" * 80)
    print("Example 2: Using FRED API")
    print("=" * 80)

    # Get FRED API key from environment
    fred_api_key = os.getenv('FRED_API_KEY')

    if not fred_api_key:
        print("SKIPPED: Set FRED_API_KEY environment variable to use FRED API")
        print("Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    # Use FRED API explicitly
    vix_df = fetch_vix_data(
        start_date="2023-01-01",
        end_date="2023-12-31",
        fred_api_key=fred_api_key
    )

    print(f"\nFetched {len(vix_df)} VIX records from FRED")
    print(vix_df.head())


def example_3_with_local_csv():
    """Example 3: Using local CSV file."""
    print("\n" + "=" * 80)
    print("Example 3: Using Local CSV")
    print("=" * 80)

    # Use local CSV file
    csv_path = Path(__file__).parent.parent.parent / "data" / "VIX_History.csv"

    if not csv_path.exists():
        print(f"SKIPPED: CSV file not found at {csv_path}")
        return

    vix_df = fetch_vix_data(
        start_date="2023-01-01",
        end_date="2023-12-31",
        csv_path=str(csv_path)
    )

    print(f"\nLoaded {len(vix_df)} VIX records from CSV")
    print(vix_df.head())


def example_4_with_feature_extraction():
    """Example 4: Fetch VIX and extract features for model."""
    print("\n" + "=" * 80)
    print("Example 4: Fetch VIX and Extract Features")
    print("=" * 80)

    # Fetch VIX data
    vix_df = fetch_vix_data(
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    print(f"\nFetched {len(vix_df)} VIX records")

    # Extract VIX features for model
    vix_features = extract_vix_features(vix_df)

    print("\nVIX Features:")
    print(f"  Current level: {vix_features.level:.2f}")
    print(f"  Normalized (0-1): {vix_features.level_normalized:.3f}")
    print(f"  5-day trend: {vix_features.trend_5d:+.2f}")
    print(f"  20-day trend: {vix_features.trend_20d:+.2f}")
    print(f"  Percentile (252d): {vix_features.percentile_252d:.1f}%")
    print(f"  Regime: {vix_features.regime} (0=low, 1=normal, 2=high, 3=extreme)")


def example_5_advanced_usage():
    """Example 5: Advanced usage with custom configuration."""
    print("\n" + "=" * 80)
    print("Example 5: Advanced Usage")
    print("=" * 80)

    # Create fetcher with custom configuration
    fetcher = FREDVixFetcher(
        fred_api_key=os.getenv('FRED_API_KEY'),
        csv_path=str(Path(__file__).parent.parent.parent / "data" / "VIX_History.csv")
    )

    # Fetch data without forward-fill
    vix_df_raw = fetcher.fetch(
        start_date="2023-01-01",
        end_date="2023-01-31",
        forward_fill=False
    )

    # Fetch data with forward-fill
    vix_df_filled = fetcher.fetch(
        start_date="2023-01-01",
        end_date="2023-01-31",
        forward_fill=True
    )

    print(f"\nWithout forward-fill: {len(vix_df_raw)} records")
    print(f"With forward-fill: {len(vix_df_filled)} records")
    print(f"Filled {len(vix_df_filled) - len(vix_df_raw)} missing dates")

    # Get source information
    source_info = fetcher.get_source_info()
    if source_info:
        print(f"\nData source: {source_info.source}")
        print(f"Date range: {source_info.date_range[0].date()} to {source_info.date_range[1].date()}")
        print(f"Records: {source_info.num_records}")
        print(f"Has gaps: {source_info.has_gaps}")


def example_6_handling_errors():
    """Example 6: Error handling and recovery."""
    print("\n" + "=" * 80)
    print("Example 6: Error Handling")
    print("=" * 80)

    try:
        # Try to fetch with invalid date range
        vix_df = fetch_vix_data(
            start_date="2025-01-01",  # Future date
            end_date="2025-12-31"
        )

        if len(vix_df) == 0:
            print("\nNo data available for future dates (as expected)")
        else:
            print(f"\nReceived {len(vix_df)} records")

    except Exception as e:
        print(f"\nCaught expected error: {str(e)}")
        print("This is normal - VIX data is not available for future dates")


def example_7_integration_with_training():
    """Example 7: Integration with training pipeline."""
    print("\n" + "=" * 80)
    print("Example 7: Integration with Training Pipeline")
    print("=" * 80)

    # This shows how to integrate VIX fetcher with the training pipeline
    from v7.training.dataset import load_market_data

    # Method 1: Use existing load_market_data (assumes VIX CSV exists)
    print("\nMethod 1: Using existing load_market_data function")
    try:
        tsla_df, spy_df, vix_df = load_market_data(
            data_dir=Path(__file__).parent.parent.parent / "data",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        print(f"  Loaded TSLA: {len(tsla_df)} records")
        print(f"  Loaded SPY: {len(spy_df)} records")
        print(f"  Loaded VIX: {len(vix_df)} records")
    except Exception as e:
        print(f"  Error: {str(e)}")

    # Method 2: Fetch VIX separately with API fallback
    print("\nMethod 2: Fetch VIX with API fallback")
    try:
        vix_df_api = fetch_vix_data(
            start_date="2023-01-01",
            end_date="2023-01-31",
            fred_api_key=os.getenv('FRED_API_KEY')
        )
        print(f"  Fetched VIX: {len(vix_df_api)} records")
        print(f"  Can now use this VIX data in training pipeline")
    except Exception as e:
        print(f"  Error: {str(e)}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("VIX FETCHER USAGE EXAMPLES")
    print("=" * 80)

    examples = [
        example_1_simple_usage,
        example_2_with_fred_api,
        example_3_with_local_csv,
        example_4_with_feature_extraction,
        example_5_advanced_usage,
        example_6_handling_errors,
        example_7_integration_with_training,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nExample failed: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
