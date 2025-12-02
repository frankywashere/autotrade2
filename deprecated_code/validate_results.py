"""
validate_results.py - Post-training validation script

Analyzes model performance, generates reports, and validates quality
User runs training, then runs this to check results

Usage:
    python validate_results.py --model_path models/lnn_model.pth \\
                               --test_data data/test_data.csv --db_path data/predictions.db
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.ml.database import SQLitePredictionDB


def load_model_metadata(model_path):
    """Load and display model metadata"""
    print("\n" + "=" * 70)
    print("MODEL METADATA")
    print("=" * 70)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    metadata = checkpoint.get('metadata', {})

    print(f"\nModel Type: {metadata.get('model_type', 'Unknown')}")
    print(f"Input Size: {metadata.get('input_size', 'Unknown')}")
    print(f"Hidden Size: {metadata.get('hidden_size', 'Unknown')}")
    print(f"Training Period: {metadata.get('train_start_year', '?')}-{metadata.get('train_end_year', '?')}")
    print(f"Training Date: {metadata.get('training_date', 'Unknown')}")
    print(f"Epochs: {metadata.get('epochs', 'Unknown')}")
    print(f"Pretrain Epochs: {metadata.get('pretrain_epochs', 'Unknown')}")
    print(f"Final Train Loss: {metadata.get('final_train_loss', 'N/A')}")
    print(f"Final Val Loss: {metadata.get('final_val_loss', 'N/A')}")

    if 'update_count' in metadata:
        print(f"\nOnline Updates: {metadata['update_count']}")
        print(f"Last Update: {metadata.get('last_update', 'Unknown')}")

    return metadata


def analyze_database_metrics(db_path):
    """Analyze prediction accuracy from database"""
    print("\n" + "=" * 70)
    print("DATABASE METRICS")
    print("=" * 70)

    db = SQLitePredictionDB(db_path)

    # Overall metrics
    print("\nOverall Metrics:")
    metrics = db.get_accuracy_metrics()

    if metrics['num_predictions'] == 0:
        print("  No predictions with actuals in database")
        return None

    print(f"  Total predictions: {metrics['num_predictions']}")
    print(f"  Mean absolute error: {metrics['mean_absolute_error']:.2f}%")
    print(f"  Median absolute error: {metrics['median_absolute_error']:.2f}%")
    print(f"  Std dev error: {metrics['std_absolute_error']:.2f}%")
    print(f"  Mean error (high): {metrics['mean_error_high']:.2f}%")
    print(f"  Mean error (low): {metrics['mean_error_low']:.2f}%")
    print(f"  Mean confidence: {metrics['mean_confidence']:.2f}")

    # Accuracy by confidence bins
    print("\nAccuracy by Confidence:")
    for key in sorted(metrics.keys()):
        if key.startswith('error_confidence_'):
            parts = key.split('_')
            low, high = parts[2], parts[3]
            print(f"  Confidence {low}-{high}: {metrics[key]:.2f}% error")

    # Get error patterns
    error_df = db.get_error_patterns(limit=20)

    if not error_df.empty:
        print("\nTop 10 Worst Predictions:")
        for i, row in error_df.head(10).iterrows():
            print(f"  {i+1}. {row['timestamp'].strftime('%Y-%m-%d')} | "
                  f"Error: {row['absolute_error']:.2f}% | "
                  f"Confidence: {row['confidence']:.2f} | "
                  f"Events: {'Earnings' if row['has_earnings'] else ''} "
                  f"{'Macro' if row['has_macro_event'] else ''}")

    return metrics, error_df


def generate_visualizations(db_path, output_dir):
    """Generate visualization plots"""
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    db = SQLitePredictionDB(db_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get data
    error_df = db.get_error_patterns(limit=1000)

    if error_df.empty:
        print("  No data for visualizations")
        return

    # 1. Error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(error_df['absolute_error'], bins=50, edgecolor='black')
    plt.xlabel('Absolute Error (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.axvline(error_df['absolute_error'].mean(), color='red', linestyle='--',
                label=f'Mean: {error_df["absolute_error"].mean():.2f}%')
    plt.axvline(error_df['absolute_error'].median(), color='green', linestyle='--',
                label=f'Median: {error_df["absolute_error"].median():.2f}%')
    plt.legend()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'error_distribution.png'}")
    plt.close()

    # 2. Confidence vs Error scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(error_df['confidence'], error_df['absolute_error'], alpha=0.5)
    plt.xlabel('Confidence')
    plt.ylabel('Absolute Error (%)')
    plt.title('Prediction Confidence vs Error')
    plt.savefig(output_dir / 'confidence_vs_error.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'confidence_vs_error.png'}")
    plt.close()

    # 3. Error by event type
    if 'has_earnings' in error_df.columns and 'has_macro_event' in error_df.columns:
        plt.figure(figsize=(10, 6))
        categories = []
        errors = []

        no_event_errors = error_df[~(error_df['has_earnings'] | error_df['has_macro_event'])]['absolute_error']
        if len(no_event_errors) > 0:
            categories.append('No Events')
            errors.append(no_event_errors.mean())

        earnings_errors = error_df[error_df['has_earnings']]['absolute_error']
        if len(earnings_errors) > 0:
            categories.append('Earnings')
            errors.append(earnings_errors.mean())

        macro_errors = error_df[error_df['has_macro_event']]['absolute_error']
        if len(macro_errors) > 0:
            categories.append('Macro')
            errors.append(macro_errors.mean())

        plt.bar(categories, errors, edgecolor='black')
        plt.ylabel('Mean Absolute Error (%)')
        plt.title('Prediction Error by Event Type')
        plt.savefig(output_dir / 'error_by_event.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / 'error_by_event.png'}")
        plt.close()

    # 4. Error over time
    if 'timestamp' in error_df.columns:
        plt.figure(figsize=(12, 6))
        error_df_sorted = error_df.sort_values('timestamp')
        plt.plot(error_df_sorted['timestamp'], error_df_sorted['absolute_error'], alpha=0.6)
        plt.xlabel('Date')
        plt.ylabel('Absolute Error (%)')
        plt.title('Prediction Error Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'error_over_time.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / 'error_over_time.png'}")
        plt.close()


def generate_report(model_path, db_path, output_dir):
    """Generate comprehensive validation report"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / 'validation_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("STAGE 2 ML MODEL VALIDATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Database: {db_path}\n")
        f.write("=" * 70 + "\n\n")

        # Model metadata
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        metadata = checkpoint.get('metadata', {})

        f.write("MODEL METADATA\n")
        f.write("-" * 70 + "\n")
        for key, value in metadata.items():
            if key != 'feature_names':  # Skip long feature list
                f.write(f"{key}: {value}\n")
        f.write("\n")

        # Database metrics
        db = SQLitePredictionDB(db_path)
        metrics = db.get_accuracy_metrics()

        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n")

        if metrics['num_predictions'] < 100:
            f.write("• Run more backtests to gather sufficient validation data\n")

        if metrics['mean_absolute_error'] > 15.0:
            f.write("• Model accuracy needs improvement - consider retraining\n")
            f.write("• Check for overfitting or underfitting\n")

        if metrics.get('mean_confidence', 0.5) < 0.6:
            f.write("• Low confidence scores - model may be uncertain\n")

        f.write("\n")
        f.write("=" * 70 + "\n")

    print(f"\n✓ Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate Stage 2 ML model results')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--db_path', type=str, default=str(config.ML_DB_PATH),
                       help='Path to prediction database')
    parser.add_argument('--output_dir', type=str, default='reports',
                       help='Directory for validation outputs (default: reports/)')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("STAGE 2: MODEL VALIDATION")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_path}")
    print("=" * 70)

    # 1. Load and display model metadata
    metadata = load_model_metadata(args.model_path)

    # 2. Analyze database metrics
    metrics, error_df = analyze_database_metrics(args.db_path)

    # 3. Generate visualizations
    if metrics and metrics['num_predictions'] > 0:
        generate_visualizations(args.db_path, args.output_dir)

    # 4. Generate report
    generate_report(args.model_path, args.db_path, args.output_dir)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Reports saved to: {args.output_dir}/")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
