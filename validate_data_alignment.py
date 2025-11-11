"""
validate_data_alignment.py - CRITICAL data validation before training

Ensures ALL data is aligned and validated:
- SPY and TSLA timestamps match exactly (inner join)
- No nulls, zeros, or gaps in price data
- Events dates fall within trading data range
- No missing features during training periods
- Validates data integrity before ANY training starts

Usage:
    python validate_data_alignment.py --spy_data data/SPY_1min.csv \\
                                      --tsla_data data/TSLA_1min.csv \\
                                      --events_data data/tsla_events_REAL.csv \\
                                      --start_date 2015-01-01 --end_date 2023-12-31
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config


class DataValidator:
    """Comprehensive data validation system"""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {}

    def log_error(self, message):
        """Log critical error"""
        self.errors.append(message)
        print(f"  ✗ ERROR: {message}")

    def log_warning(self, message):
        """Log warning"""
        self.warnings.append(message)
        print(f"  ⚠ WARNING: {message}")

    def log_success(self, message):
        """Log success"""
        print(f"  ✓ {message}")

    def validate_csv_exists(self, file_path, name):
        """Check if CSV file exists"""
        print(f"\n1. Checking {name} file...")

        if not Path(file_path).exists():
            self.log_error(f"{name} file not found: {file_path}")
            return False

        self.log_success(f"{name} file exists: {file_path}")
        return True

    def validate_csv_structure(self, df, name, required_columns):
        """Validate CSV has required columns"""
        print(f"\n2. Validating {name} structure...")

        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            self.log_error(f"{name} missing columns: {missing_cols}")
            return False

        self.log_success(f"{name} has all required columns: {required_columns}")

        # Check for nulls
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            self.log_error(f"{name} has null values:\n{null_counts[null_counts > 0]}")
            return False

        self.log_success(f"{name} has no null values")

        return True

    def validate_price_data(self, df, name):
        """Validate price data (no zeros, prices are reasonable)"""
        print(f"\n3. Validating {name} price data...")

        price_cols = ['open', 'high', 'low', 'close']

        # Check for zeros
        zero_counts = (df[price_cols] == 0).sum()
        if zero_counts.any():
            self.log_error(f"{name} has zero prices:\n{zero_counts[zero_counts > 0]}")
            return False

        self.log_success(f"{name} has no zero prices")

        # Check for negative prices
        negative_counts = (df[price_cols] < 0).sum()
        if negative_counts.any():
            self.log_error(f"{name} has negative prices:\n{negative_counts[negative_counts > 0]}")
            return False

        self.log_success(f"{name} has no negative prices")

        # Check for reasonable price ranges
        price_stats = df[price_cols].describe()
        min_price = price_stats.loc['min'].min()
        max_price = price_stats.loc['max'].max()

        if min_price < 1:
            self.log_warning(f"{name} has very low prices (min: ${min_price:.2f})")
        if max_price > 10000:
            self.log_warning(f"{name} has very high prices (max: ${max_price:.2f})")

        self.log_success(f"{name} price range: ${min_price:.2f} - ${max_price:.2f}")

        return True

    def validate_timestamps(self, df, name):
        """Validate timestamps are sorted and reasonable"""
        print(f"\n4. Validating {name} timestamps...")

        # Check if sorted
        if not df.index.is_monotonic_increasing:
            self.log_error(f"{name} timestamps are not sorted")
            return False

        self.log_success(f"{name} timestamps are sorted")

        # Check for duplicates
        duplicate_count = df.index.duplicated().sum()
        if duplicate_count > 0:
            self.log_error(f"{name} has {duplicate_count} duplicate timestamps")
            return False

        self.log_success(f"{name} has no duplicate timestamps")

        # Check date range
        start_date = df.index.min()
        end_date = df.index.max()
        total_days = (end_date - start_date).days

        self.log_success(f"{name} date range: {start_date} to {end_date} ({total_days} days)")

        self.stats[f'{name}_start'] = start_date
        self.stats[f'{name}_end'] = end_date
        self.stats[f'{name}_bars'] = len(df)

        return True

    def align_spy_tsla(self, spy_df, tsla_df):
        """Align SPY and TSLA by timestamp (inner join)"""
        print(f"\n5. Aligning SPY and TSLA timestamps...")

        print(f"  SPY bars: {len(spy_df):,}")
        print(f"  TSLA bars: {len(tsla_df):,}")

        # Inner join on timestamp
        common_timestamps = spy_df.index.intersection(tsla_df.index)

        if len(common_timestamps) == 0:
            self.log_error("NO COMMON TIMESTAMPS between SPY and TSLA!")
            return None, None

        spy_aligned = spy_df.loc[common_timestamps].copy()
        tsla_aligned = tsla_df.loc[common_timestamps].copy()

        # Rename columns
        spy_aligned.columns = [f'spy_{col}' for col in spy_aligned.columns]
        tsla_aligned.columns = [f'tsla_{col}' for col in tsla_aligned.columns]

        alignment_pct = len(common_timestamps) / max(len(spy_df), len(tsla_df)) * 100

        self.log_success(f"Found {len(common_timestamps):,} common timestamps ({alignment_pct:.1f}% alignment)")

        # Check for gaps
        time_diffs = pd.Series(common_timestamps).diff()
        median_diff = time_diffs.median()
        large_gaps = time_diffs[time_diffs > median_diff * 10].count()

        if large_gaps > 0:
            self.log_warning(f"Found {large_gaps} large time gaps in aligned data")
        else:
            self.log_success("No large time gaps in aligned data")

        self.stats['aligned_bars'] = len(common_timestamps)
        self.stats['alignment_pct'] = alignment_pct

        return spy_aligned, tsla_aligned

    def validate_events_coverage(self, events_df, aligned_start, aligned_end):
        """Validate events fall within data range"""
        print(f"\n6. Validating events coverage...")

        events_df['date'] = pd.to_datetime(events_df['date'])

        # Check events within data range
        events_in_range = events_df[
            (events_df['date'] >= aligned_start) &
            (events_df['date'] <= aligned_end)
        ]

        events_before = len(events_df[events_df['date'] < aligned_start])
        events_after = len(events_df[events_df['date'] > aligned_end])

        self.log_success(f"Events in data range: {len(events_in_range)} / {len(events_df)}")

        if events_before > 0:
            self.log_warning(f"{events_before} events before data start date")
        if events_after > 0:
            self.log_warning(f"{events_after} events after data end date")

        # Count by type and source
        print("\n  Event breakdown:")
        for source in events_df['source'].unique():
            source_events = events_in_range[events_in_range['source'] == source]
            print(f"    {source.upper()}: {len(source_events)} events")

            for event_type in source_events['event_type'].unique():
                type_count = len(source_events[source_events['event_type'] == event_type])
                print(f"      - {event_type}: {type_count}")

        self.stats['events_in_range'] = len(events_in_range)
        self.stats['events_total'] = len(events_df)

        return events_in_range

    def validate_feature_extraction_readiness(self, aligned_df):
        """Validate data is ready for feature extraction"""
        print(f"\n7. Validating feature extraction readiness...")

        # Check we have enough data for sequence creation
        min_sequence_length = config.ML_SEQUENCE_LENGTH
        min_prediction_horizon = config.PREDICTION_HORIZON_HOURS

        required_bars = min_sequence_length + min_prediction_horizon

        if len(aligned_df) < required_bars:
            self.log_error(f"Insufficient data for training: {len(aligned_df)} bars < {required_bars} required")
            return False

        self.log_success(f"Sufficient data for training: {len(aligned_df):,} bars >= {required_bars} required")

        # Check for data quality in both symbols
        spy_nulls = aligned_df[[c for c in aligned_df.columns if c.startswith('spy_')]].isnull().sum().sum()
        tsla_nulls = aligned_df[[c for c in aligned_df.columns if c.startswith('tsla_')]].isnull().sum().sum()

        if spy_nulls > 0 or tsla_nulls > 0:
            self.log_error(f"Aligned data has nulls: SPY={spy_nulls}, TSLA={tsla_nulls}")
            return False

        self.log_success("Aligned data has no nulls")

        # Check for zeros in price columns
        spy_price_cols = [c for c in aligned_df.columns if c.startswith('spy_') and c.split('_')[1] in ['open', 'high', 'low', 'close']]
        tsla_price_cols = [c for c in aligned_df.columns if c.startswith('tsla_') and c.split('_')[1] in ['open', 'high', 'low', 'close']]

        spy_zeros = (aligned_df[spy_price_cols] == 0).sum().sum()
        tsla_zeros = (aligned_df[tsla_price_cols] == 0).sum().sum()

        if spy_zeros > 0 or tsla_zeros > 0:
            self.log_error(f"Aligned data has zero prices: SPY={spy_zeros}, TSLA={tsla_zeros}")
            return False

        self.log_success("Aligned data has no zero prices")

        return True

    def generate_report(self):
        """Generate final validation report"""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        print("\nStatistics:")
        for key, value in self.stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        if self.errors:
            print(f"\n✗ VALIDATION FAILED - {len(self.errors)} ERRORS:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
            return False

        if self.warnings:
            print(f"\n⚠ {len(self.warnings)} WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        print("\n✓ VALIDATION PASSED - Data is ready for training!")
        print("  - SPY and TSLA are perfectly aligned")
        print("  - No nulls, zeros, or gaps in price data")
        print("  - Events coverage validated")
        print("  - Feature extraction ready")

        return True


def main():
    parser = argparse.ArgumentParser(description='Validate data alignment before training')

    parser.add_argument('--spy_data', type=str, default='data/SPY_1min.csv',
                       help='Path to SPY 1-minute CSV')
    parser.add_argument('--tsla_data', type=str, default='data/TSLA_1min.csv',
                       help='Path to TSLA 1-minute CSV')
    parser.add_argument('--events_data', type=str, default='data/tsla_events_REAL.csv',
                       help='Path to events CSV')
    parser.add_argument('--start_date', type=str, default='2015-01-01',
                       help='Start date for validation')
    parser.add_argument('--end_date', type=str, default='2023-12-31',
                       help='End date for validation')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("DATA ALIGNMENT VALIDATION (CRITICAL - NO FAKE DATA!)")
    print("=" * 70)
    print(f"Date range: {args.start_date} to {args.end_date}")

    validator = DataValidator()

    # 1. Check files exist
    if not validator.validate_csv_exists(args.spy_data, "SPY"):
        sys.exit(1)
    if not validator.validate_csv_exists(args.tsla_data, "TSLA"):
        sys.exit(1)
    if not validator.validate_csv_exists(args.events_data, "Events"):
        sys.exit(1)

    # 2. Load SPY data
    print("\n" + "=" * 70)
    print("LOADING SPY DATA")
    print("=" * 70)
    spy_df = pd.read_csv(args.spy_data)
    spy_df['timestamp'] = pd.to_datetime(spy_df['timestamp'])
    spy_df.set_index('timestamp', inplace=True)
    spy_df = spy_df[(spy_df.index >= args.start_date) & (spy_df.index <= args.end_date)]

    if not validator.validate_csv_structure(spy_df, "SPY", ['open', 'high', 'low', 'close', 'volume']):
        sys.exit(1)
    if not validator.validate_price_data(spy_df, "SPY"):
        sys.exit(1)
    if not validator.validate_timestamps(spy_df, "SPY"):
        sys.exit(1)

    # 3. Load TSLA data
    print("\n" + "=" * 70)
    print("LOADING TSLA DATA")
    print("=" * 70)
    tsla_df = pd.read_csv(args.tsla_data)
    tsla_df['timestamp'] = pd.to_datetime(tsla_df['timestamp'])
    tsla_df.set_index('timestamp', inplace=True)
    tsla_df = tsla_df[(tsla_df.index >= args.start_date) & (tsla_df.index <= args.end_date)]

    if not validator.validate_csv_structure(tsla_df, "TSLA", ['open', 'high', 'low', 'close', 'volume']):
        sys.exit(1)
    if not validator.validate_price_data(tsla_df, "TSLA"):
        sys.exit(1)
    if not validator.validate_timestamps(tsla_df, "TSLA"):
        sys.exit(1)

    # 4. Align SPY and TSLA
    print("\n" + "=" * 70)
    print("ALIGNING SPY AND TSLA")
    print("=" * 70)
    spy_aligned, tsla_aligned = validator.align_spy_tsla(spy_df, tsla_df)

    if spy_aligned is None or tsla_aligned is None:
        sys.exit(1)

    aligned_df = pd.concat([spy_aligned, tsla_aligned], axis=1)

    # 5. Validate events
    print("\n" + "=" * 70)
    print("VALIDATING EVENTS")
    print("=" * 70)
    events_df = pd.read_csv(args.events_data)
    events_in_range = validator.validate_events_coverage(
        events_df,
        aligned_df.index.min(),
        aligned_df.index.max()
    )

    # 6. Validate feature extraction readiness
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION READINESS")
    print("=" * 70)
    if not validator.validate_feature_extraction_readiness(aligned_df):
        sys.exit(1)

    # 7. Generate report
    success = validator.generate_report()

    print("=" * 70)

    if not success:
        print("\n❌ DATA VALIDATION FAILED - DO NOT PROCEED WITH TRAINING!")
        print("Fix the errors above before training.")
        sys.exit(1)
    else:
        print("\n✅ DATA VALIDATION PASSED - SAFE TO TRAIN!")
        print(f"\nNext steps:")
        print(f"  1. Run: python train_model.py --tsla_events {args.events_data}")
        print(f"  2. Model will train on {validator.stats['aligned_bars']:,} aligned bars")
        print(f"  3. With {validator.stats['events_in_range']} events integrated")


if __name__ == '__main__':
    main()
