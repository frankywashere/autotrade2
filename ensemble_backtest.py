#!/usr/bin/env python3
"""
Ensemble Backtesting Script

Tests Meta-LNN coach performance by combining predictions from all 4 timeframe models.
Creates predictions with model_timeframe='ensemble' for comparison with individual models.

Usage:
    python3 ensemble_backtest.py --test_year 2024 --num_simulations 50
"""

import argparse
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.ml.data_feed import CSVDataFeed
from src.ml.features import TradingFeatureExtractor
from src.ml.events import CombinedEventsHandler
from src.ml.database import SQLitePredictionDB
from src.ml.ensemble import load_ensemble


def validate_models():
    """Validate all required model files exist and have correct structure."""
    print("\n" + "="*70)
    print("VALIDATING MODELS")
    print("="*70)

    required_models = {
        '15min': 'models/lnn_15min.pth',
        '1hour': 'models/lnn_1hour.pth',
        '4hour': 'models/lnn_4hour.pth',
        'daily': 'models/lnn_daily.pth',
        'meta': 'models/meta_lnn.pth'
    }

    for name, path in required_models.items():
        if not Path(path).exists():
            print(f"  ✗ Missing {name} model: {path}")
            return False

        # Check metadata
        try:
            ckpt = torch.load(path, weights_only=False)
            metadata = ckpt.get('metadata', {})

            if name != 'meta':
                input_size = metadata.get('input_size', 0)
                timeframe = metadata.get('input_timeframe', 'unknown')

                if input_size != 245:
                    print(f"  ✗ {name} model has wrong input_size: {input_size} (expected 245)")
                    return False

                print(f"  ✓ {name} model: {timeframe}, 245 features")
            else:
                print(f"  ✓ Meta-LNN coach loaded")

        except Exception as e:
            print(f"  ✗ Error loading {name} model: {e}")
            return False

    print("="*70)
    return True


def calculate_minimum_context_days(min_bars_per_timeframe=20):
    """
    Calculate minimum historical lookback needed for complete feature extraction.
    Longest timeframe (3month) requires ~1848 days for 20 bars.
    """
    timeframe_requirements = {
        '5min': min_bars_per_timeframe * 0.0003 * 1.4,      # ~0.008 days
        '15min': min_bars_per_timeframe * 0.001 * 1.4,      # ~0.028 days
        '30min': min_bars_per_timeframe * 0.002 * 1.4,      # ~0.056 days
        '1h': min_bars_per_timeframe * 0.15 * 1.4,          # ~4.2 days
        '2h': min_bars_per_timeframe * 0.31 * 1.4,          # ~8.7 days
        '3h': min_bars_per_timeframe * 0.46 * 1.4,          # ~13 days
        '4h': min_bars_per_timeframe * 0.62 * 1.4,          # ~17 days
        'daily': min_bars_per_timeframe * 1.4,              # ~28 days
        'weekly': min_bars_per_timeframe * 5 * 1.4,         # ~140 days
        'monthly': min_bars_per_timeframe * 22 * 1.4,       # ~616 days
        '3month': min_bars_per_timeframe * 66 * 1.4,        # ~1848 days
    }
    return int(max(timeframe_requirements.values()))


def run_ensemble_simulation(
    date: datetime,
    ensemble,
    data_feeds: Dict[str, CSVDataFeed],
    feature_extractor: TradingFeatureExtractor,
    events_handler: CombinedEventsHandler,
    db: SQLitePredictionDB
) -> Optional[Dict]:
    """
    Run one ensemble backtest simulation for a specific date.

    Returns:
        Dict with prediction results, or None if failed
    """
    print(f"\n[{date.strftime('%Y-%m-%d')}] Running ensemble simulation...")

    try:
        # Step 1: Load data at all 4 timeframes
        print(f"  Step 1: Loading multi-timeframe data...")
        data_dict = {}
        context_days = calculate_minimum_context_days(min_bars_per_timeframe=20)

        # Load model metadata directly from checkpoint files
        model_metadata = {}
        for tf in ['15min', '1hour', '4hour', 'daily']:
            ckpt = torch.load(f'models/lnn_{tf}.pth', weights_only=False)
            model_metadata[tf] = ckpt['metadata']

        for tf in ['15min', '1hour', '4hour', 'daily']:
            try:
                # Get this sub-model's sequence length from metadata
                seq_len = model_metadata[tf]['sequence_length']

                # Load data
                context_start = date - timedelta(days=context_days)
                hist_df = data_feeds[tf].load_aligned_data(
                    context_start.strftime('%Y-%m-%d'),
                    date.strftime('%Y-%m-%d')
                )

                if len(hist_df) < seq_len:
                    print(f"    ✗ {tf}: Insufficient data ({len(hist_df)}/{seq_len} bars)")
                    return None

                # Extract features
                features_df = feature_extractor.extract_features(hist_df)

                if len(features_df) < seq_len:
                    print(f"    ✗ {tf}: Insufficient features after extraction ({len(features_df)}/{seq_len})")
                    return None

                # Create input tensor
                sequence = features_df.tail(seq_len).values
                data_dict[tf] = torch.tensor(sequence, dtype=torch.float32)

                print(f"    ✓ {tf}: {len(hist_df)} bars → {len(features_df)} features")

            except Exception as e:
                print(f"    ✗ {tf}: Error loading data: {e}")
                return None

        # Step 2: Get current price and market state
        print(f"  Step 2: Extracting current price and market state...")
        try:
            # Use 1hour data for current price
            main_df = data_feeds['1hour'].load_aligned_data(
                context_start.strftime('%Y-%m-%d'),
                date.strftime('%Y-%m-%d')
            )
            main_features = feature_extractor.extract_features(main_df)
            current_price = main_features.iloc[-1]['tsla_close']
            current_idx = len(main_features) - 1

            if pd.isna(current_price) or current_price <= 0:
                print(f"    ✗ Invalid current price: {current_price}")
                return None

            print(f"    ✓ Current price: ${current_price:.2f}")

        except Exception as e:
            print(f"    ✗ Error getting current price: {e}")
            return None

        # Step 3: Get ensemble prediction
        print(f"  Step 3: Getting ensemble prediction...")
        try:
            predictions = ensemble.predict(
                data=data_dict,
                features_df=main_features,
                current_idx=current_idx,
                timestamp=date
            )

            pred_high = predictions['predicted_high']
            pred_low = predictions['predicted_low']
            confidence = predictions['confidence']
            sub_predictions = predictions.get('sub_predictions', {})

            print(f"    ✓ Ensemble: high={pred_high:+.2f}%, low={pred_low:+.2f}%, conf={confidence:.2f}")

            # Show sub-predictions
            for tf in ['15min', '1hour', '4hour', 'daily']:
                if tf in sub_predictions:
                    sp = sub_predictions[tf]
                    print(f"      - {tf}: high={sp['predicted_high']:+.2f}%, low={sp['predicted_low']:+.2f}%, conf={sp['confidence']:.2f}")

        except Exception as e:
            print(f"    ✗ Error getting ensemble prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Step 4: Get actuals
        print(f"  Step 4: Loading actual future prices...")
        try:
            future_start = date
            future_end = date + timedelta(hours=24)

            actual_df = data_feeds['1hour'].load_aligned_data(
                future_start.strftime('%Y-%m-%d'),
                future_end.strftime('%Y-%m-%d')
            )

            if len(actual_df) == 0:
                print(f"    ✗ No future data available")
                return None

            actual_high = actual_df['tsla_close'].max()
            actual_low = actual_df['tsla_close'].min()

            if pd.isna(actual_high) or pd.isna(actual_low):
                print(f"    ✗ Actuals are NaN")
                return None

            # Convert to percentages
            actual_high_pct = (actual_high - current_price) / current_price * 100
            actual_low_pct = (actual_low - current_price) / current_price * 100

            print(f"    ✓ Actuals: high={actual_high_pct:+.2f}%, low={actual_low_pct:+.2f}%")

        except Exception as e:
            print(f"    ✗ Error loading actuals: {e}")
            return None

        # Step 5: Calculate errors
        error_high = abs(pred_high - actual_high_pct)
        error_low = abs(pred_low - actual_low_pct)
        avg_error = (error_high + error_low) / 2

        print(f"    ✓ Errors: high={error_high:.2f}pp, low={error_low:.2f}pp, avg={avg_error:.2f}pp")

        # Step 6: Log to database
        print(f"  Step 5: Logging to database...")
        try:
            prediction_record = {
                'prediction_timestamp': datetime.now(),
                'target_timestamp': future_end,
                'simulation_date': date,
                'symbol': 'TSLA',
                'timeframe': '24h',
                'model_timeframe': 'ensemble',
                'is_ensemble': True,
                'news_enabled': False,
                'predicted_high': float(pred_high),
                'predicted_low': float(pred_low),
                'predicted_center': float((pred_high + pred_low) / 2),
                'predicted_range': float(pred_high - pred_low),
                'confidence': float(confidence),
                'current_price': float(current_price),
                'model_version': 'ensemble_v3.4',
                'feature_dim': 245,
            }

            # Add sub-predictions
            for tf in ['15min', '1hour', '4hour', 'daily']:
                if tf in sub_predictions:
                    sp = sub_predictions[tf]
                    prediction_record[f'sub_pred_{tf}_high'] = float(sp['predicted_high'])
                    prediction_record[f'sub_pred_{tf}_low'] = float(sp['predicted_low'])
                    prediction_record[f'sub_pred_{tf}_conf'] = float(sp['confidence'])

            pred_id = db.log_prediction(prediction_record)
            db.update_actual(pred_id, float(actual_high), float(actual_low))

            print(f"    ✓ Logged to database (pred_id={pred_id})")

        except Exception as e:
            print(f"    ✗ Error logging to database: {e}")
            import traceback
            traceback.print_exc()
            return None

        return {
            'date': date,
            'pred_high': pred_high,
            'pred_low': pred_low,
            'actual_high_pct': actual_high_pct,
            'actual_low_pct': actual_low_pct,
            'error': avg_error,
            'confidence': confidence
        }

    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def select_random_dates(test_year: int, num_simulations: int, seed: int = 42):
    """Select random test dates from test year."""
    print("\n" + "="*70)
    print("SELECTING TEST DATES")
    print("="*70)

    # Generate candidate dates (weekdays only)
    start_date = datetime(test_year, 1, 1)
    end_date = datetime(test_year, 12, 31)

    candidate_dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday=0, Friday=4
            candidate_dates.append(current)
        current += timedelta(days=1)

    print(f"  Test year: {test_year}")
    print(f"  Candidate pool: {len(candidate_dates)} weekdays")

    # Select random sample
    random.seed(seed)
    selected = random.sample(candidate_dates, min(num_simulations, len(candidate_dates)))
    selected.sort()

    print(f"  ✓ Selected {len(selected)} dates")
    print(f"  Date range: {selected[0].strftime('%Y-%m-%d')} to {selected[-1].strftime('%Y-%m-%d')}")
    print("="*70)

    return selected


def main():
    parser = argparse.ArgumentParser(description='Backtest Meta-LNN Ensemble')

    parser.add_argument('--test_year', type=int, default=2024,
                       help='Year to test on (default: 2024)')
    parser.add_argument('--num_simulations', type=int, default=50,
                       help='Number of test dates (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--db_path', type=str, default='data/predictions.db',
                       help='Predictions database path')

    args = parser.parse_args()

    print("="*70)
    print("ENSEMBLE BACKTESTING - META-LNN COACH")
    print("="*70)
    print(f"Test year: {args.test_year}")
    print(f"Simulations: {args.num_simulations}")
    print(f"Random seed: {args.seed}")
    print(f"Database: {args.db_path}")
    print("="*70)

    # Validate models
    if not validate_models():
        print("\n❌ Model validation failed. Please ensure all 4 sub-models + Meta-LNN are trained.")
        sys.exit(1)

    # Load ensemble
    print("\n" + "="*70)
    print("LOADING ENSEMBLE")
    print("="*70)

    try:
        ensemble = load_ensemble(
            mode='backtest_no_news',
            device='cpu',
            models_dir='models',
            events_csv='data/tsla_events_REAL.csv'
        )
        print(f"  ✓ Loaded 4 sub-models + Meta-LNN coach")
        print("="*70)
    except Exception as e:
        print(f"  ✗ Error loading ensemble: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Initialize components
    print("\n" + "="*70)
    print("INITIALIZING COMPONENTS")
    print("="*70)

    try:
        data_feeds = {
            '15min': CSVDataFeed(timeframe='15min'),
            '1hour': CSVDataFeed(timeframe='1hour'),
            '4hour': CSVDataFeed(timeframe='4hour'),
            'daily': CSVDataFeed(timeframe='daily')
        }
        print(f"  ✓ Created data feeds for all 4 timeframes")

        feature_extractor = TradingFeatureExtractor()
        print(f"  ✓ Feature extractor ready (245 features)")

        events_handler = CombinedEventsHandler()
        print(f"  ✓ Events handler ready")

        db = SQLitePredictionDB(args.db_path)
        print(f"  ✓ Database ready: {args.db_path}")

        print("="*70)

    except Exception as e:
        print(f"  ✗ Error initializing components: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Select test dates
    test_dates = select_random_dates(args.test_year, args.num_simulations, args.seed)

    # Run simulations
    print("\n" + "="*70)
    print("RUNNING ENSEMBLE SIMULATIONS")
    print("="*70)

    results = []
    successful = 0
    failed = 0

    for i, date in enumerate(test_dates, 1):
        print(f"\n[{i}/{len(test_dates)}] {date.strftime('%Y-%m-%d')}")

        result = run_ensemble_simulation(
            date=date,
            ensemble=ensemble,
            data_feeds=data_feeds,
            feature_extractor=feature_extractor,
            events_handler=events_handler,
            db=db
        )

        if result:
            results.append(result)
            successful += 1
            print(f"  ✅ Success (error: {result['error']:.2f}%)")
        else:
            failed += 1
            print(f"  ❌ Failed")

    # Summary
    print("\n" + "="*70)
    print("ENSEMBLE BACKTEST SUMMARY")
    print("="*70)

    print(f"\nSimulations:")
    print(f"  Successful: {successful}/{len(test_dates)}")
    print(f"  Failed: {failed}/{len(test_dates)}")

    if results:
        errors = [r['error'] for r in results]
        confidences = [r['confidence'] for r in results]

        print(f"\nEnsemble Performance:")
        print(f"  Mean error: {np.mean(errors):.2f}%")
        print(f"  Median error: {np.median(errors):.2f}%")
        print(f"  Min error: {np.min(errors):.2f}%")
        print(f"  Max error: {np.max(errors):.2f}%")
        print(f"  Mean confidence: {np.mean(confidences):.2f}")

        print(f"\nComparison Query:")
        print(f"  Run this to compare ensemble vs individual models:")
        print(f'')
        print(f'  sqlite3 {args.db_path} "')
        print(f'    SELECT ')
        print(f'      model_timeframe,')
        print(f'      COUNT(*) as n,')
        print(f'      ROUND(AVG(absolute_error), 2) as error,')
        print(f'      ROUND(AVG(confidence), 2) as conf')
        print(f'    FROM predictions')
        print(f"    WHERE simulation_date >= '{test_dates[0].strftime('%Y-%m-%d')}'")
        print(f'    GROUP BY model_timeframe')
        print(f'    ORDER BY error ASC')
        print(f'  "')
        print(f'')

    print("="*70)
    print(f"✅ Ensemble backtesting complete!")
    print("="*70)


if __name__ == '__main__':
    main()
