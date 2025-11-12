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


def run_backtest(model_path, test_year, num_simulations):
    """Run backtest for a single model."""
    print(f"\n{'='*70}")
    print(f"BACKTESTING: {Path(model_path).name}")
    print(f"{'='*70}")

    cmd = [
        sys.executable,  # Use same Python interpreter
        'backtest.py',
        '--model_path', model_path,
        '--test_year', str(test_year),
        '--num_simulations', str(num_simulations)
    ]

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

    # Run backtests
    successful = 0
    failed = 0

    for i, model_path in enumerate(found_models, 1):
        print(f"\n\n{'#'*70}")
        print(f"# BACKTEST {i}/{len(found_models)}")
        print(f"{'#'*70}")

        if run_backtest(model_path, args.test_year, args.num_simulations):
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

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Analyze results in predictions.db")
    print("  2. Train Meta-LNN coach:")
    print("     python train_meta_lnn.py --mode backtest_no_news")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
