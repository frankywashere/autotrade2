"""
backtest.py - Backtesting script for Stage 2 ML model

Simulates forward testing by picking random days/weeks in test year
Provides prior context and evaluates predictions against actuals

Usage:
    python backtest.py --model_path models/lnn_model.pth --test_year 2024 \\
                       --num_simulations 100 --db_path data/predictions.db
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.ml.data_feed import CSVDataFeed
from src.ml.features import TradingFeatureExtractor
from src.ml.events import CombinedEventsHandler
from src.ml.model import LNNTradingModel, LSTMTradingModel
from src.ml.database import SQLitePredictionDB


def load_model(model_path):
    """Load trained model from checkpoint"""
    print(f"Loading model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    metadata = checkpoint.get('metadata', {})

    model_type = metadata.get('model_type', 'LNN')
    input_size = metadata['input_size']
    hidden_size = metadata.get('hidden_size', config.LNN_HIDDEN_SIZE)

    if model_type == 'LNN':
        model = LNNTradingModel(input_size, hidden_size)
    else:
        model = LSTMTradingModel(input_size, hidden_size)

    model.load_checkpoint(model_path)
    model.eval()

    print(f"Model loaded: {model_type}, {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Trained on: {metadata.get('train_start_year')}-{metadata.get('train_end_year')}")
    print(f"Input timeframe: {metadata.get('input_timeframe', '1min')}")
    print(f"Sequence length: {metadata.get('sequence_length', config.ML_SEQUENCE_LENGTH)}")
    print(f"Prediction horizon: {metadata.get('prediction_horizon', config.PREDICTION_HORIZON_HOURS)} bars")
    print(f"Prediction mode: {metadata.get('prediction_mode', 'uniform_bars')}")

    return model, metadata


def select_random_dates(test_year, num_simulations, seed=None):
    """
    Select random dates throughout the test year for simulation
    Ensures dates are trading days (Monday-Friday)
    """
    if seed:
        random.seed(seed)

    start_date = datetime(test_year, 1, 1)
    end_date = datetime(test_year, 12, 31)

    # Generate all potential dates
    all_dates = []
    current = start_date
    while current <= end_date:
        # Only include weekdays (Monday=0 to Friday=4)
        if current.weekday() < 5:
            all_dates.append(current)
        current += timedelta(days=1)

    # Randomly select dates
    selected_dates = random.sample(all_dates, min(num_simulations, len(all_dates)))
    selected_dates.sort()

    return selected_dates


def run_simulation(date, model, feature_extractor, data_feed, events_handler, db, ensemble=None, mode='backtest_no_news', metadata=None):
    """
    Run a single backtest simulation for a specific date

    Args:
        metadata: Model metadata containing input_timeframe and sequence_length

    Returns: (prediction_dict, actual_dict, error_dict)
    """
    # Get model configuration from metadata
    if metadata is None:
        metadata = {}

    input_timeframe = metadata.get('input_timeframe', '1min')
    sequence_length = metadata.get('sequence_length', config.ML_SEQUENCE_LENGTH)
    prediction_horizon_bars = metadata.get('prediction_horizon', config.PREDICTION_HORIZON_HOURS)
    prediction_mode = metadata.get('prediction_mode', 'uniform_bars')

    # Define timeframe to minutes conversion
    timeframe_minutes = {
        '1min': 1,
        '5min': 5,
        '15min': 15,
        '30min': 30,
        '1hour': 60,
        '2hour': 120,
        '3hour': 180,
        '4hour': 240,
        'daily': 1440,  # 24 * 60
        'weekly': 10080,  # 7 * 24 * 60
        'monthly': 43200,  # 30 * 24 * 60 (approximation)
        '3month': 129600  # 90 * 24 * 60 (approximation)
    }

    # Convert prediction horizon bars to actual time
    minutes_per_bar = timeframe_minutes.get(input_timeframe, 60)
    prediction_horizon_minutes = prediction_horizon_bars * minutes_per_bar
    prediction_horizon_hours = prediction_horizon_minutes / 60

    # Log the actual prediction window
    print(f"  Prediction window: {prediction_horizon_bars} bars = {prediction_horizon_hours:.1f} hours")

    # 1. Load historical context (e.g., 1 week before the date)
    context_days = 7
    start_context = date - timedelta(days=context_days)
    end_context = date

    try:
        aligned_df = data_feed.load_aligned_data(
            start_context.strftime('%Y-%m-%d'),
            end_context.strftime('%Y-%m-%d')
        )

        if len(aligned_df) < sequence_length:
            print(f"   ⚠ Insufficient data for {date.strftime('%Y-%m-%d')}, need {sequence_length} bars, skipping...")
            return None

        # 2. Extract features
        features_df = feature_extractor.extract_features(aligned_df)

        # Get last sequence (use model's sequence_length from metadata)
        sequence = features_df.tail(sequence_length).values
        sequence_tensor = torch.tensor([sequence], dtype=torch.float32)  # (1, seq_len, features)

        # 3. Get events for this date
        events = events_handler.get_events_for_date(date.strftime('%Y-%m-%d'))
        has_earnings = any(e.get('event_type') in ['earnings', 'delivery', 'production']
                          for e in events)
        has_macro = any(e.get('event_type') in ['fomc', 'cpi', 'nfp']
                       for e in events)

        # 4. Make prediction
        if ensemble is not None:
            # Ensemble mode: Use multi-scale ensemble
            # TODO: For now, use simplified approach - full ensemble integration
            # requires loading data at multiple timeframes
            # For backtest purposes, we'll use single model approach until
            # ensemble data pipeline is complete
            print("   ⚠ Ensemble mode not yet integrated with backtest pipeline")
            print("   Use single-model mode for now: python backtest.py --model_path models/lnn_15min.pth")
            return None
        else:
            # Single model prediction
            predictions = model.predict(sequence_tensor)

            pred_high = predictions['predicted_high'][0]
            pred_low = predictions['predicted_low'][0]
            pred_center = predictions['predicted_center'][0]
            pred_range = predictions['predicted_range'][0]
            confidence = predictions['confidence'][0]

            # Add model metadata to predictions dict
            predictions['is_ensemble'] = False
            predictions['news_enabled'] = False
            predictions['model_timeframe'] = 'single'

        # 5. Get actuals (load data for prediction window after date)
        target_start = date
        target_end = date + timedelta(minutes=prediction_horizon_minutes)

        actual_df = data_feed.load_aligned_data(
            target_start.strftime('%Y-%m-%d'),
            target_end.strftime('%Y-%m-%d')
        )

        if len(actual_df) == 0:
            print(f"   ⚠ No actual data for {date.strftime('%Y-%m-%d')}, skipping...")
            return None

        actual_high = actual_df['tsla_close'].max()
        actual_low = actual_df['tsla_close'].min()
        actual_center = (actual_high + actual_low) / 2

        # 6. Calculate errors
        error_high = abs(pred_high - actual_high) / actual_high * 100
        error_low = abs(pred_low - actual_low) / actual_low * 100
        error_center = abs(pred_center - actual_center) / actual_center * 100
        avg_error = (error_high + error_low) / 2

        # 7. Log to database
        prediction_record = {
            'prediction_timestamp': datetime.now(),
            'target_timestamp': target_end,
            'symbol': 'TSLA',
            'timeframe': '24h',
            'predicted_high': float(pred_high),
            'predicted_low': float(pred_low),
            'predicted_center': float(pred_center),
            'predicted_range': float(pred_range),
            'confidence': float(confidence),
            'has_earnings': has_earnings,
            'has_macro_event': has_macro,
            'event_type': events[0].get('event_type') if events else None,
            'model_version': 'backtest_v1',
            'feature_dim': feature_extractor.get_feature_dim(),
            # Multi-scale ensemble fields (populated if ensemble mode)
            'model_timeframe': prediction.get('model_timeframe', 'single'),
            'is_ensemble': prediction.get('is_ensemble', False),
            'news_enabled': prediction.get('news_enabled', False),
        }

        # Add sub-predictions if ensemble
        if prediction.get('sub_predictions'):
            for tf, sub_pred in prediction['sub_predictions'].items():
                prediction_record[f'sub_pred_{tf}_high'] = float(sub_pred['predicted_high'])
                prediction_record[f'sub_pred_{tf}_low'] = float(sub_pred['predicted_low'])
                prediction_record[f'sub_pred_{tf}_conf'] = float(sub_pred['confidence'])

        pred_id = db.log_prediction(prediction_record)
        db.update_actual(pred_id, float(actual_high), float(actual_low))

        return {
            'date': date,
            'pred_high': pred_high,
            'pred_low': pred_low,
            'actual_high': actual_high,
            'actual_low': actual_low,
            'error_high': error_high,
            'error_low': error_low,
            'avg_error': avg_error,
            'confidence': confidence,
            'has_earnings': has_earnings,
            'has_macro': has_macro
        }

    except Exception as e:
        print(f"   ✗ Error in simulation for {date.strftime('%Y-%m-%d')}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Backtest Stage 2 ML model')

    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint (for single-model mode)')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use multi-scale ensemble mode')
    parser.add_argument('--mode', type=str, default='backtest_no_news',
                       choices=['backtest_no_news', 'live_with_news'],
                       help='Prediction mode (default: backtest_no_news)')
    parser.add_argument('--test_year', type=int, default=config.ML_TEST_YEAR,
                       help='Year to backtest on (default: 2024)')
    parser.add_argument('--num_simulations', type=int, default=config.BACKTEST_NUM_SIMULATIONS,
                       help='Number of random simulations (default: 100)')
    parser.add_argument('--db_path', type=str, default=str(config.ML_DB_PATH),
                       help='Path to prediction database')
    parser.add_argument('--seed', type=int, default=config.BACKTEST_RANDOM_SEED,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Validate arguments
    if not args.ensemble and args.model_path is None:
        parser.error("--model_path required when not using --ensemble mode")

    print("\n" + "=" * 70)
    print("STAGE 2: MODEL BACKTESTING")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'ENSEMBLE' if args.ensemble else 'SINGLE MODEL'}")
    if not args.ensemble:
        print(f"Model: {args.model_path}")
    print(f"Test year: {args.test_year}")
    print(f"Simulations: {args.num_simulations}")
    print(f"News mode: {args.mode}")
    print("=" * 70)

    # 1. Load model(s)
    if args.ensemble:
        # Load ensemble system
        from src.ml.ensemble import load_ensemble

        print("\nLoading multi-scale ensemble...")
        ensemble = load_ensemble(
            mode=args.mode,
            device='cpu',  # Can be made configurable
            models_dir='models',
            events_csv='data/tsla_events_REAL.csv'
        )
        model = None
        metadata = {'model_type': 'ensemble', 'mode': args.mode}
    else:
        # Load single model
        model, metadata = load_model(args.model_path)
        ensemble = None

    # 2. Initialize components
    print("\nInitializing components...")

    # Get timeframe from metadata (for correct CSV loading)
    input_timeframe = metadata.get('input_timeframe', '1min')

    data_feed = CSVDataFeed(timeframe=input_timeframe)  # Load correct timeframe CSVs
    feature_extractor = TradingFeatureExtractor()
    events_handler = CombinedEventsHandler()
    db = SQLitePredictionDB(args.db_path)

    # 3. Select random dates
    print(f"\nSelecting {args.num_simulations} random dates in {args.test_year}...")
    test_dates = select_random_dates(args.test_year, args.num_simulations, args.seed)
    print(f"Date range: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")

    # 4. Run simulations
    print("\n" + "=" * 70)
    print("RUNNING SIMULATIONS")
    print("=" * 70)

    results = []
    for i, date in enumerate(test_dates, 1):
        print(f"\n[{i}/{len(test_dates)}] Simulating {date.strftime('%Y-%m-%d')}...")

        result = run_simulation(date, model, feature_extractor, data_feed, events_handler, db,
                               ensemble=ensemble, mode=args.mode, metadata=metadata)

        if result:
            results.append(result)
            print(f"   ✓ Predicted: [{result['pred_low']:.2f} - {result['pred_high']:.2f}]")
            print(f"   ✓ Actual: [{result['actual_low']:.2f} - {result['actual_high']:.2f}]")
            print(f"   ✓ Error: {result['avg_error']:.2f}% | Confidence: {result['confidence']:.2f}")

    # 5. Summarize results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 70)

    if results:
        results_df = pd.DataFrame(results)

        print(f"\nCompleted simulations: {len(results)}/{args.num_simulations}")
        print(f"\nAverage Metrics:")
        print(f"  Mean Error (High): {results_df['error_high'].mean():.2f}%")
        print(f"  Mean Error (Low): {results_df['error_low'].mean():.2f}%")
        print(f"  Mean Absolute Error: {results_df['avg_error'].mean():.2f}%")
        print(f"  Median Absolute Error: {results_df['avg_error'].median():.2f}%")
        print(f"  Std Dev Error: {results_df['avg_error'].std():.2f}%")
        print(f"  Mean Confidence: {results_df['confidence'].mean():.2f}")

        print(f"\nError by Event Type:")
        print(f"  With Earnings: {results_df[results_df['has_earnings']]['avg_error'].mean():.2f}% "
              f"({results_df['has_earnings'].sum()} cases)")
        print(f"  With Macro Event: {results_df[results_df['has_macro']]['avg_error'].mean():.2f}% "
              f"({results_df['has_macro'].sum()} cases)")
        print(f"  No Events: {results_df[~(results_df['has_earnings'] | results_df['has_macro'])]['avg_error'].mean():.2f}% "
              f"({(~(results_df['has_earnings'] | results_df['has_macro'])).sum()} cases)")

        print(f"\nDatabase Summary:")
        metrics = db.get_accuracy_metrics()
        print(f"  Total predictions logged: {metrics['num_predictions']}")
        print(f"  Mean absolute error: {metrics['mean_absolute_error']:.2f}%")

        # Save results
        results_file = Path(args.model_path).parent / f"backtest_results_{args.test_year}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")

    else:
        print("\n⚠ No successful simulations completed")

    print(f"\nBacktesting completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
