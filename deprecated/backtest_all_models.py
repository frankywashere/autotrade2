#!/usr/bin/env python3
"""
Backtest all timeframe models automatically.

This script automatically finds and backtests all trained multi-scale models.
Runs backtests sequentially and shows combined summary.

Usage:
    python backtest_all_models.py --test_year 2023 --num_simulations 500
    python backtest_all_models.py --test_year 2024 --num_simulations 100
"""

import argparse
import subprocess
import sys
from pathlib import Path
import sqlite3
import pandas as pd
import torch
from datetime import datetime
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.data_feed import CSVDataFeed
from src.ml.features import TradingFeatureExtractor
from backtest import select_random_dates


# Expected model filenames
MODEL_FILENAMES = [
    'lnn_15min.pth',
    'lnn_1hour.pth',
    'lnn_4hour.pth',
    'lnn_daily.pth'
]


def find_models(models_dir='models'):
    """Find all trained timeframe models."""
    models_dir = Path(models_dir)

    found_models = []
    for model_name in MODEL_FILENAMES:
        model_path = models_dir / model_name
        if model_path.exists():
            found_models.append(str(model_path))

    return found_models


def run_backtest(model_path, test_year, num_simulations, dates_file=None):
    """Run backtest for a single model."""
    print(f"\n{'='*70}")
    print(f"BACKTESTING: {Path(model_path).name}")
    print(f"{'='*70}")

    cmd = [
        sys.executable,  # Use same Python interpreter
        'backtest.py',
        '--model_path', model_path,
        '--test_year', str(test_year),
        '--num_simulations', str(num_simulations),
        '--seed', '42'  # Same seed for all models
    ]

    # Add dates file if provided
    if dates_file:
        cmd.extend(['--dates_file', dates_file])
        print(f"  Using shared dates file: {dates_file}")

    # Debug: print full command
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error backtesting {model_path}: {e}")
        return False


def show_summary(db_path='data/predictions.db'):
    """Show comparison summary of all models."""
    try:
        conn = sqlite3.connect(db_path)

        query = """
            SELECT
                model_timeframe,
                COUNT(*) as num_predictions,
                AVG(absolute_error) as avg_error,
                AVG(confidence) as avg_confidence
            FROM predictions
            WHERE actual_high IS NOT NULL
              AND model_timeframe IN ('15min', '1hour', '4hour', 'daily', 'single')
            GROUP BY model_timeframe
            ORDER BY avg_error ASC
        """

        df = pd.read_sql(query, conn)
        conn.close()

        if len(df) > 0:
            print("\n" + "=" * 70)
            print("BACKTEST SUMMARY (All Models)")
            print("=" * 70)
            print(df.to_string(index=False))
            print("\n" + "=" * 70)
        else:
            print("\n⚠️  No predictions found in database yet")

    except Exception as e:
        print(f"\n⚠️  Could not load summary: {e}")


def main():
    parser = argparse.ArgumentParser(description='Backtest all timeframe models')

    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory containing model checkpoints')
    parser.add_argument('--test_year', type=int, default=2023,
                       help='Year to backtest on')
    parser.add_argument('--num_simulations', type=int, default=500,
                       help='Number of simulations per model')
    parser.add_argument('--db_path', type=str, default='data/predictions.db',
                       help='Predictions database path')

    args = parser.parse_args()

    print("=" * 70)
    print("BACKTEST ALL MODELS")
    print("=" * 70)
    print(f"Test year: {args.test_year}")
    print(f"Simulations per model: {args.num_simulations}")
    print()

    # Find models
    print("Finding trained models...")
    found_models = find_models(args.models_dir)

    if not found_models:
        print(f"\n❌ No models found in {args.models_dir}/")
        print(f"\nExpected model files:")
        for model_name in MODEL_FILENAMES:
            print(f"  - {model_name}")
        print(f"\nTrain models first:")
        print(f"  python train_model_lazy.py --input_timeframe 15min ...")
        sys.exit(1)

    print(f"Found {len(found_models)} models:")
    for model_path in found_models:
        print(f"  ✓ {model_path}")

    # Confirm
    print(f"\nThis will run {len(found_models)} backtests × {args.num_simulations} simulations")
    print(f"Estimated time: ~{len(found_models) * 10}-{len(found_models) * 15} minutes")

    confirm = input("\nProceed? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Ask about clearing existing database
    db_path = Path(args.db_path)
    if db_path.exists():
        print(f"\nFound existing database: {args.db_path}")
        clear_db = input("Clear old predictions before running? [y/N]: ").strip().lower()
        if clear_db == 'y':
            db_path.unlink()
            print(f"✓ Cleared {args.db_path}")
        else:
            print(f"✓ Appending new predictions to {args.db_path}")

    # Pre-select test dates ONCE for all models
    print("\n" + "=" * 70)
    print("PRE-SELECTING TEST DATES (multi-timeframe validation)")
    print("=" * 70)

    # Load ALL 4 models to get metadata for each timeframe
    print("Loading all models to extract metadata...")
    all_metadata = {}
    timeframe_order = ['15min', '1hour', '4hour', 'daily']

    for model_path in found_models:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        metadata = checkpoint.get('metadata', {})
        timeframe = metadata.get('input_timeframe', '1min')
        all_metadata[timeframe] = metadata
        print(f"  ✓ {Path(model_path).name}: {timeframe} (seq={metadata.get('sequence_length', '?')})")

    # Create data feeds AND feature extractor for all timeframes
    print("\nInitializing data feeds for all timeframes...")
    data_feeds = {}
    for tf in timeframe_order:
        if tf in all_metadata:
            data_feeds[tf] = CSVDataFeed(timeframe=tf)
            print(f"  ✓ {tf} data feed ready")

    print("\nInitializing feature extractor for validation...")
    feature_extractor = TradingFeatureExtractor()
    print(f"  ✓ Feature extractor ready")

    # Validate dates against ALL timeframes
    from backtest import get_safe_date_range, validate_date_has_data
    import random

    print(f"\nFinding dates valid for ALL {len(timeframe_order)} timeframes...")
    print("=" * 70)

    # Get safe date range
    data_start_date = datetime(2015, 1, 2)
    data_end_date = datetime(2025, 9, 27)
    safe_start, safe_end = get_safe_date_range(args.test_year, data_start_date, data_end_date)

    print(f"Test year: {args.test_year}")
    print(f"Safe range: {safe_start.strftime('%Y-%m-%d')} to {safe_end.strftime('%Y-%m-%d')}")

    # Generate candidate pool
    from datetime import timedelta
    candidate_dates = []
    current = safe_start
    while current <= safe_end:
        if current.weekday() < 5:  # Weekdays only
            candidate_dates.append(current)
        current += timedelta(days=1)

    print(f"Candidate pool: {len(candidate_dates)} weekdays")
    print(f"\nValidating dates with COMPLETE checks (all 7 failure points)...")
    print("This mirrors run_simulation() exactly: data loading + feature extraction + actuals")
    print("=" * 70)

    random.seed(42)
    random.shuffle(candidate_dates)

    validated_dates = []
    attempts = 0
    max_attempts = min(args.num_simulations * 5, len(candidate_dates))  # Try up to 5x

    # Track failure reasons for statistics
    failure_stats = {}

    for candidate in candidate_dates:
        if len(validated_dates) >= args.num_simulations:
            break

        if attempts >= max_attempts:
            break

        attempts += 1

        # Check if this date works for ALL timeframes
        valid_for_all_timeframes = True
        failed_timeframe = None
        failed_reason = None

        for tf in timeframe_order:
            if tf not in all_metadata or tf not in data_feeds:
                continue

            # Enhanced validation with feature extraction
            is_valid, reason = validate_date_has_data(
                candidate,
                data_feeds[tf],
                all_metadata[tf],
                feature_extractor=feature_extractor,  # Now testing feature extraction!
                verbose=False
            )

            if not is_valid:
                valid_for_all_timeframes = False
                failed_timeframe = tf
                failed_reason = reason

                # Track failure statistics
                failure_key = f"{tf}:{reason}"
                failure_stats[failure_key] = failure_stats.get(failure_key, 0) + 1

                break  # Skip this date if any timeframe fails

        if valid_for_all_timeframes:
            validated_dates.append(candidate)

            # Progress feedback
            if len(validated_dates) % 10 == 0:
                progress_pct = (len(validated_dates) / args.num_simulations) * 100
                print(f"  [{int(progress_pct):3d}%] Found {len(validated_dates)}/{args.num_simulations} valid dates (tested {attempts} candidates)")

    validated_dates.sort()
    test_dates = validated_dates

    # Display comprehensive validation summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    success_rate = (len(test_dates) / attempts * 100) if attempts > 0 else 0
    print(f"✓ Found {len(test_dates)}/{args.num_simulations} dates valid for ALL timeframes")
    print(f"  Success rate: {success_rate:.1f}% ({len(test_dates)}/{attempts} candidates tested)")
    print(f"  These dates will work for: 15min, 1hour, 4hour, daily")

    # Show failure breakdown
    if failure_stats:
        print(f"\nFailure breakdown (why dates were rejected):")
        sorted_failures = sorted(failure_stats.items(), key=lambda x: x[1], reverse=True)
        for failure_key, count in sorted_failures[:10]:  # Top 10
            timeframe, reason = failure_key.split(':', 1)
            print(f"  {timeframe:6s} - {reason:40s}: {count:3d} dates")

    print("=" * 70)

    if len(test_dates) == 0:
        print("\n❌ No valid dates found for backtesting!")
        print("   Try: (1) Earlier test year, (2) Fewer simulations")
        sys.exit(1)

    # Write dates to temporary file AND permanent debug file
    dates_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', prefix='backtest_dates_')
    dates_file_path = dates_file.name

    # Also write to debug file for inspection
    debug_dates_path = 'models/last_validated_dates_debug.txt'

    print(f"\n✓ Selected {len(test_dates)} validated dates")
    print(f"  Writing to: {dates_file_path}")
    print(f"  Debug copy: {debug_dates_path}")

    with open(debug_dates_path, 'w') as debug_file:
        for date in test_dates:
            date_str = date.strftime('%Y-%m-%d')
            dates_file.write(f"{date_str}\n")
            debug_file.write(f"{date_str}\n")

    dates_file.close()

    print(f"  ✓ All models will use these same {len(test_dates)} dates")
    print(f"  (Dates saved to {debug_dates_path} for inspection)")
    print("=" * 70)

    # Run backtests
    successful = 0
    failed = 0

    for i, model_path in enumerate(found_models, 1):
        print(f"\n\n{'#'*70}")
        print(f"# BACKTEST {i}/{len(found_models)}")
        print(f"{'#'*70}")

        if run_backtest(model_path, args.test_year, args.num_simulations, dates_file=dates_file_path):
            successful += 1
        else:
            failed += 1

    # Show summary
    print("\n\n" + "=" * 70)
    print("ALL BACKTESTS COMPLETE")
    print("=" * 70)
    print(f"Successful: {successful}/{len(found_models)}")
    print(f"Failed: {failed}/{len(found_models)}")

    # Show comparison
    show_summary(args.db_path)

    # Clean up temporary dates file
    try:
        Path(dates_file_path).unlink()
        print(f"\n✓ Cleaned up temporary dates file")
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Analyze results in predictions.db")
    print("  2. Train Meta-LNN coach:")
    print("     python train_meta_lnn.py --mode backtest_no_news")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
