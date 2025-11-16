#!/usr/bin/env python3
"""
Hierarchical Model Backtester

Simulates trading on historical data to evaluate model performance.

Usage:
    python backtest_hierarchical.py --interactive
    python backtest_hierarchical.py --model models/hierarchical_lnn.pth --year 2023
"""

import argparse
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.hierarchical_model import load_hierarchical_model
from src.ml.features import TradingFeatureExtractor
from src.ml.data_feed import CSVDataFeed
from src.ml.events import CombinedEventsHandler


def interactive_setup():
    """Interactive CLI for backtest configuration"""
    try:
        from InquirerPy import inquirer
        from InquirerPy.base.control import Choice
    except ImportError:
        print("InquirerPy not installed. Using command-line args only.")
        return None

    print("\n" + "="*70)
    print("🔬 HIERARCHICAL MODEL BACKTESTER")
    print("="*70)

    # Model selection
    model_path = inquirer.filepath(
        message="Select trained model:",
        default="models/hierarchical_lnn.pth"
    ).execute()

    # Year selection
    year = inquirer.select(
        message="Test year:",
        choices=[
            Choice(2023, "2023"),
            Choice(2024, "2024"),
            Choice(2022, "2022"),
            Choice("custom", "Custom date range")
        ]
    ).execute()

    start_date = None
    end_date = None

    if year == "custom":
        start_date = inquirer.text(
            message="Start date (YYYY-MM-DD):",
            default="2023-01-01"
        ).execute()

        end_date = inquirer.text(
            message="End date (YYYY-MM-DD):",
            default="2023-12-31"
        ).execute()
    else:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

    # Confidence threshold
    confidence_threshold = inquirer.number(
        message="Minimum confidence for trades:",
        default=0.75,
        min_allowed=0.0,
        max_allowed=1.0,
        float_allowed=True
    ).execute()

    # Output
    save_results = inquirer.confirm(
        message="Save results to CSV?",
        default=True
    ).execute()

    return {
        'model_path': model_path,
        'start_date': start_date,
        'end_date': end_date,
        'confidence_threshold': float(confidence_threshold),
        'save_results': save_results
    }


def run_backtest(model_path, start_date, end_date, confidence_threshold=0.75, save_results=True):
    """Run backtest on historical data"""

    print("\n" + "="*70)
    print("📊 BACKTESTING HIERARCHICAL MODEL")
    print("="*70)

    print(f"\nModel: {model_path}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Confidence threshold: {confidence_threshold}")

    # Load model
    print("\n1. Loading model...")
    model = load_hierarchical_model(model_path, device='cpu')
    model.eval()
    print(f"   ✓ Model loaded (input_size={model.input_size})")

    # Load data
    print("\n2. Loading historical data...")
    data_feed = CSVDataFeed(timeframe='1min')
    df = data_feed.load_aligned_data(start_date=start_date, end_date=end_date)
    print(f"   ✓ Loaded {len(df):,} bars")

    # Extract features
    print("\n3. Extracting features...")
    extractor = TradingFeatureExtractor()

    try:
        events_handler = CombinedEventsHandler()
    except:
        events_handler = None

    features_df = extractor.extract_features(df, use_cache=True, events_handler=events_handler)
    print(f"   ✓ Extracted {len(features_df.columns)} features")

    # Check compatibility
    if features_df.shape[1] != model.input_size:
        print(f"\n❌ ERROR: Feature mismatch!")
        print(f"   Model expects: {model.input_size} features")
        print(f"   System extracts: {features_df.shape[1]} features")
        print(f"\n   Solution: Retrain model with current feature set")
        return 1

    # Backtest
    print("\n4. Running backtest...")
    print(f"   Testing {len(features_df) - 200:,} predictions...")

    results = []
    sequence_length = 200
    prediction_horizon = 24  # bars (24 minutes for 1-min data)

    with tqdm(total=len(features_df) - 200 - prediction_horizon, desc="   Backtesting") as pbar:
        for i in range(sequence_length, len(features_df) - prediction_horizon):
            # Get sequence
            sequence = features_df.iloc[i-sequence_length:i].values
            x_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

            # Current price
            current_price = float(features_df.iloc[i]['tsla_close'])

            # Predict
            with torch.no_grad():
                pred = model.predict(x_tensor)

            pred_high = float(pred['predicted_high'])
            pred_low = float(pred['predicted_low'])
            confidence = float(pred['confidence'])

            # Get actual outcome (next prediction_horizon bars)
            future_prices = features_df.iloc[i:i+prediction_horizon]['tsla_close'].values
            actual_high_price = np.max(future_prices)
            actual_low_price = np.min(future_prices)

            actual_high_pct = ((actual_high_price - current_price) / current_price) * 100
            actual_low_pct = ((actual_low_price - current_price) / current_price) * 100

            # Calculate errors
            error_high = abs(pred_high - actual_high_pct)
            error_low = abs(pred_low - actual_low_pct)
            avg_error = (error_high + error_low) / 2

            # Trade simulation (if confidence > threshold)
            trade_result = None
            if confidence >= confidence_threshold:
                # Simulate trade
                target_price = current_price * (1 + pred_high / 100)
                stop_price = current_price * (1 + pred_low / 100)

                # Did we hit target?
                hit_target = actual_high_price >= target_price
                # Did we hit stop?
                hit_stop = actual_low_price <= stop_price

                trade_result = {
                    'hit_target': hit_target,
                    'hit_stop': hit_stop,
                    'win': hit_target and not hit_stop
                }

            # Store result
            results.append({
                'timestamp': features_df.index[i],
                'current_price': current_price,
                'predicted_high': pred_high,
                'predicted_low': pred_low,
                'actual_high': actual_high_pct,
                'actual_low': actual_low_pct,
                'error_high': error_high,
                'error_low': error_low,
                'avg_error': avg_error,
                'confidence': confidence,
                'trade_taken': confidence >= confidence_threshold,
                'trade_win': trade_result['win'] if trade_result else None
            })

            pbar.update(1)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate metrics
    print("\n" + "="*70)
    print("📈 BACKTEST RESULTS")
    print("="*70)

    overall_mape = results_df['avg_error'].mean()
    high_conf_results = results_df[results_df['confidence'] >= confidence_threshold]

    print(f"\nOverall Performance:")
    print(f"  Total predictions: {len(results_df):,}")
    print(f"  Average error (MAPE): {overall_mape:.2f}%")
    print(f"  High prediction error: {results_df['error_high'].mean():.2f}%")
    print(f"  Low prediction error: {results_df['error_low'].mean():.2f}%")

    if len(high_conf_results) > 0:
        print(f"\nHigh Confidence Trades (>{confidence_threshold:.0%}):")
        print(f"  Trade count: {len(high_conf_results):,}")
        print(f"  Average error: {high_conf_results['avg_error'].mean():.2f}%")

        trades_taken = high_conf_results[high_conf_results['trade_taken']]
        if len(trades_taken) > 0:
            wins = trades_taken['trade_win'].sum()
            total_trades = len(trades_taken)
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

            print(f"  Win rate: {win_rate:.1f}% ({int(wins)}/{total_trades})")

    # Best/worst predictions
    print(f"\nBest Predictions:")
    best = results_df.nsmallest(5, 'avg_error')
    for _, row in best.iterrows():
        print(f"  {row['timestamp']}: {row['avg_error']:.2f}% error (conf: {row['confidence']:.2f})")

    print(f"\nWorst Predictions:")
    worst = results_df.nlargest(5, 'avg_error')
    for _, row in worst.iterrows():
        print(f"  {row['timestamp']}: {row['avg_error']:.2f}% error (conf: {row['confidence']:.2f})")

    # Save results
    if save_results:
        output_file = f"backtest_results_{start_date}_{end_date}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "="*70)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Backtest hierarchical model on historical data")
    parser.add_argument('--interactive', action='store_true', help='Interactive CLI mode')
    parser.add_argument('--model', '--model_path', dest='model_path', default='models/hierarchical_lnn.pth')
    parser.add_argument('--year', type=int, help='Year to backtest (2023, 2024, etc.)')
    parser.add_argument('--start_date', help='Custom start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', help='Custom end date (YYYY-MM-DD)')
    parser.add_argument('--confidence_threshold', type=float, default=0.75)
    parser.add_argument('--no_save', action='store_true', help='Don\'t save results to CSV')

    args = parser.parse_args()

    # Interactive mode
    if args.interactive:
        config = interactive_setup()
        if config:
            return run_backtest(**config)

    # Command-line mode
    if args.year:
        start_date = f"{args.year}-01-01"
        end_date = f"{args.year}-12-31"
    elif args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        print("Error: Specify --year or --start_date/--end_date")
        return 1

    return run_backtest(
        model_path=args.model_path,
        start_date=start_date,
        end_date=end_date,
        confidence_threshold=args.confidence_threshold,
        save_results=not args.no_save
    )


if __name__ == '__main__':
    sys.exit(main())
