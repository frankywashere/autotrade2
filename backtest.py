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
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.ml.data_feed import CSVDataFeed
from src.ml.features import TradingFeatureExtractor
from src.ml.events import CombinedEventsHandler
from src.ml.model import LNNTradingModel, LSTMTradingModel
from src.ml.database import SQLitePredictionDB


def calculate_minimum_context_days(min_bars_per_timeframe=20):
    """
    Calculate minimum context days needed to extract all features.

    Feature extraction resamples to multiple timeframes:
    1min, 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3month

    Each timeframe needs min_bars_per_timeframe bars for channel calculations
    (see src/ml/features.py:193 for the 20-bar minimum).

    The longest timeframe (3month) determines the minimum data requirement,
    since we need 20+ bars of 3-month data to calculate meaningful features.

    Args:
        min_bars_per_timeframe: Minimum bars needed per timeframe (default: 20)

    Returns:
        Minimum calendar days needed (includes weekend/holiday buffer)

    Example:
        If 3month = 66 trading days, and we need 20 bars:
        20 bars × 66 days/bar × 1.4 buffer = 1848 calendar days ≈ 5 years
    """
    # Trading days per timeframe (approximate, assuming 252 trading days/year)
    trading_days_per_timeframe = {
        '3month': 66,      # ~1/6 year = 66 trading days
        'monthly': 22,     # ~1/12 year = 22 trading days
        'weekly': 5,       # 1 week = 5 trading days
        'daily': 1,
        '4hour': 0.25,
        '3hour': 0.167,
        '2hour': 0.111,
        '1hour': 0.056,
        '30min': 0.028,
        '15min': 0.014,
        '5min': 0.0028,
        '1min': 0.00056,
    }

    # Find the longest timeframe (3month = 66 trading days)
    longest_tf_trading_days = max(trading_days_per_timeframe.values())

    # Calculate minimum trading days needed
    min_trading_days = min_bars_per_timeframe * longest_tf_trading_days

    # Convert to calendar days (multiply by 1.4 to account for weekends/holidays)
    # 252 trading days per year, 365 calendar days per year
    # 365/252 ≈ 1.45, so use 1.4 as conservative estimate
    calendar_to_trading_ratio = 365 / 252
    min_calendar_days = int(min_trading_days * calendar_to_trading_ratio)

    return min_calendar_days


def get_safe_date_range(test_year, data_start_date, data_end_date):
    """
    Calculate safe date range for backtesting based on data availability.

    Safe dates must have:
    - Sufficient historical data (5 years back for 3month feature extraction)
    - Sufficient future data (30 days forward for prediction actuals)

    Args:
        test_year: Year to test (e.g., 2024)
        data_start_date: Earliest date in dataset (datetime)
        data_end_date: Latest date in dataset (datetime)

    Returns:
        (safe_start, safe_end): Tuple of datetime objects
    """
    # Calculate minimum context needed for feature extraction
    context_days = calculate_minimum_context_days(min_bars_per_timeframe=20)

    # Conservative buffer for prediction window (longest model needs ~24 days)
    prediction_buffer_days = 30

    # Calculate absolute safe boundaries
    absolute_safe_start = data_start_date + timedelta(days=context_days)
    absolute_safe_end = data_end_date - timedelta(days=prediction_buffer_days)

    # Intersect with requested test year
    test_year_start = datetime(test_year, 1, 1)
    test_year_end = datetime(test_year, 12, 31)

    safe_start = max(absolute_safe_start, test_year_start)
    safe_end = min(absolute_safe_end, test_year_end)

    return safe_start, safe_end


def validate_date_has_data(date, data_feed, metadata, feature_extractor=None, verbose=False):
    """
    Pre-validate that a date has sufficient aligned data for backtesting.

    Tests ALL 7 failure points that run_simulation() checks:
    1. Historical bar count
    2. Feature extraction success
    3. Features length after extraction
    4. Current price extraction
    5. Future data availability
    6. Actuals column existence
    7. Actuals NaN check

    Args:
        date: Date to validate (datetime)
        data_feed: CSVDataFeed instance
        metadata: Model metadata dict with sequence_length, input_timeframe, prediction_horizon
        feature_extractor: TradingFeatureExtractor instance (required for full validation)
        verbose: If True, print detailed validation info

    Returns:
        (success: bool, failure_reason: str)
    """
    try:
        # Extract model configuration
        sequence_length = metadata.get('sequence_length', config.ML_SEQUENCE_LENGTH)
        input_timeframe = metadata.get('input_timeframe', '1min')
        prediction_horizon_bars = metadata.get('prediction_horizon', config.PREDICTION_HORIZON_HOURS)

        # === TEST 1: Historical bar count ===
        context_days = calculate_minimum_context_days(min_bars_per_timeframe=20)
        context_days = max(context_days, 7)

        start_context = date - timedelta(days=context_days)
        end_context = date

        historical_df = data_feed.load_aligned_data(
            start_context.strftime('%Y-%m-%d'),
            end_context.strftime('%Y-%m-%d')
        )

        if len(historical_df) < sequence_length:
            return False, f"insufficient_historical_bars ({len(historical_df)}/{sequence_length})"

        # === TEST 2 & 3: Feature extraction ===
        if feature_extractor is not None:
            try:
                features_df = feature_extractor.extract_features(historical_df)

                if len(features_df) < sequence_length:
                    return False, f"insufficient_features_after_extraction ({len(features_df)}/{sequence_length})"

            except Exception as e:
                return False, f"feature_extraction_error: {str(e)[:50]}"

        # === TEST 4: Current price extraction ===
        if 'tsla_close' not in historical_df.columns:
            return False, "tsla_close_column_missing"

        try:
            cp_raw = historical_df['tsla_close'].iloc[-1]
            cp_float = float(cp_raw)
            if np.isnan(cp_float) or cp_float <= 0:
                return False, "current_price_invalid"
        except Exception:
            return False, "current_price_extraction_error"

        # === TEST 5: Future data availability ===
        timeframe_minutes = {
            '1min': 1, '5min': 5, '15min': 15, '30min': 30,
            '1hour': 60, '2hour': 120, '3hour': 180, '4hour': 240,
            'daily': 1440, 'weekly': 10080, 'monthly': 43200, '3month': 129600
        }

        minutes_per_bar = timeframe_minutes.get(input_timeframe, 60)
        prediction_horizon_minutes = prediction_horizon_bars * minutes_per_bar

        future_start = date
        future_end = date + timedelta(minutes=prediction_horizon_minutes)

        future_df = data_feed.load_aligned_data(
            future_start.strftime('%Y-%m-%d'),
            future_end.strftime('%Y-%m-%d')
        )

        if len(future_df) == 0:
            return False, "no_future_data"

        # === TEST 6 & 7: Actuals calculation ===
        if 'tsla_close' not in future_df.columns:
            return False, "tsla_close_missing_in_future"

        try:
            actual_high = future_df['tsla_close'].max()
            actual_low = future_df['tsla_close'].min()

            if pd.isna(actual_high) or pd.isna(actual_low):
                return False, "actuals_are_nan"
        except Exception:
            return False, "actuals_calculation_error"

        # All checks passed!
        return True, "valid"

    except Exception as e:
        return False, f"unexpected_error: {str(e)[:50]}"


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


def select_random_dates(test_year, num_simulations, seed=None, data_feed=None, metadata=None, validate=True):
    """
    Select random dates throughout the test year for simulation with two-phase validation.

    Phase 1: Safe date range filtering (calendar time availability)
    Phase 2: Pre-validation of aligned bar count (optional but recommended)

    Args:
        test_year: Year to test (e.g., 2024)
        num_simulations: Number of dates to select
        seed: Random seed for reproducibility
        data_feed: CSVDataFeed instance (required if validate=True)
        metadata: Model metadata dict (required if validate=True)
        validate: If True, pre-validate each date for sufficient aligned data

    Returns:
        List of validated datetime objects (sorted)
    """
    if seed:
        random.seed(seed)

    # Phase 1: Calculate safe date range based on data availability
    # Hardcoded data boundaries from CSV inspection
    data_start_date = datetime(2015, 1, 2)
    data_end_date = datetime(2025, 9, 27)

    safe_start, safe_end = get_safe_date_range(test_year, data_start_date, data_end_date)

    print(f"\n📅 Date selection:")
    print(f"  Test year: {test_year}")
    print(f"  Safe range: {safe_start.strftime('%Y-%m-%d')} to {safe_end.strftime('%Y-%m-%d')}")
    print(f"  Validation: {'Enabled' if validate else 'Disabled'}")

    if safe_start > safe_end:
        print(f"\n⚠️  WARNING: No safe dates available for {test_year}!")
        print(f"  Data range: {data_start_date.strftime('%Y-%m-%d')} to {data_end_date.strftime('%Y-%m-%d')}")
        print(f"  Try an earlier year (e.g., 2023)")
        return []

    # Generate candidate pool from safe range (weekdays only)
    candidate_dates = []
    current = safe_start
    while current <= safe_end:
        if current.weekday() < 5:  # Monday-Friday
            candidate_dates.append(current)
        current += timedelta(days=1)

    print(f"  Candidate pool: {len(candidate_dates)} weekdays")

    # Phase 2: Pre-validation (optional but recommended)
    if validate and data_feed is not None and metadata is not None:
        print(f"  Validating candidates...")

        validated_dates = []
        attempts = 0
        max_attempts = min(num_simulations * 3, len(candidate_dates))  # Try up to 3x or pool size

        # Shuffle candidates to test random order
        random.shuffle(candidate_dates)

        for candidate in candidate_dates:
            if len(validated_dates) >= num_simulations:
                break

            if attempts >= max_attempts:
                print(f"  ⚠️  Reached max attempts ({max_attempts}), stopping validation")
                break

            attempts += 1

            # Pre-validate this date
            is_valid, reason = validate_date_has_data(candidate, data_feed, metadata)
            if is_valid:
                validated_dates.append(candidate)

                # Progress feedback every 10 valid dates
                if len(validated_dates) % 10 == 0:
                    print(f"    Validated {len(validated_dates)}/{num_simulations} dates...")

        validated_dates.sort()

        success_rate = (len(validated_dates) / attempts * 100) if attempts > 0 else 0
        print(f"  ✓ Validated {len(validated_dates)}/{num_simulations} dates ({success_rate:.1f}% success rate)")

        if len(validated_dates) < num_simulations:
            print(f"  ⚠️  WARNING: Only found {len(validated_dates)} valid dates (requested {num_simulations})")
            print(f"  Consider: (1) Using earlier test_year, (2) Reducing num_simulations")

        return validated_dates

    else:
        # No validation - just randomly sample from candidate pool
        if not validate:
            print(f"  ⚠️  Validation disabled - dates may fail during backtesting")

        selected_dates = random.sample(candidate_dates, min(num_simulations, len(candidate_dates)))
        selected_dates.sort()

        print(f"  ✓ Selected {len(selected_dates)} dates (unvalidated)")

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

    # 1. Calculate required lookback dynamically based on model metadata
    # Need enough calendar days to get sequence_length bars, accounting for:
    # - Market hours (6.5 hours/day for US stocks)
    # - Weekends (no trading Saturday/Sunday)
    # - Holidays (various market closures)

    bars_per_trading_day = {
        '1min': 390,      # 6.5 hours × 60 minutes
        '5min': 78,       # 6.5 hours × 12 (5-min bars per hour)
        '15min': 26,      # 6.5 hours × 4
        '30min': 13,      # 6.5 hours × 2
        '1hour': 6.5,
        '2hour': 3.25,
        '3hour': 2.17,
        '4hour': 1.625,
        'daily': 1,
        'weekly': 0.2,    # ~1 bar per week
        'monthly': 0.05,  # ~1 bar per month
        '3month': 0.017   # ~1 bar per 3 months
    }.get(input_timeframe, 6.5)

    # Calculate calendar days needed
    # Use the longer of:
    # 1. Days needed for the model's sequence (based on input timeframe)
    # 2. Days needed to extract all features (requires 20+ bars of each timeframe including 3month)
    sequence_context_days = int((sequence_length / bars_per_trading_day) * 1.5) + 10
    feature_context_days = calculate_minimum_context_days(min_bars_per_timeframe=20)

    context_days = max(sequence_context_days, feature_context_days)

    # Sanity check: ensure at least 7 days
    context_days = max(context_days, 7)

    print(f"  Loading {context_days} calendar days to get {sequence_length} bars of {input_timeframe} data...")

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

        # Get current price (last bar in sequence)
        current_price = None
        try:
            if 'tsla_close' in aligned_df.columns and len(aligned_df) > 0:
                cp_raw = aligned_df['tsla_close'].iloc[-1]

                # Convert to Python float
                if cp_raw is not None:
                    cp_float = float(cp_raw)
                    if not np.isnan(cp_float) and cp_float > 0:
                        current_price = cp_float

            if current_price is None:
                print(f"   ⚠ Could not extract valid current price from data")
                return None

        except Exception as e:
            print(f"   ⚠ Error getting current price: {e}")
            return None

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
            predictions['model_timeframe'] = input_timeframe  # Use extracted timeframe instead of hardcoded 'single'

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

        # Check if tsla_close column exists and has data
        if 'tsla_close' not in actual_df.columns:
            print(f"   ⚠ tsla_close column not found in actual_df")
            return None

        actual_high = actual_df['tsla_close'].max()
        actual_low = actual_df['tsla_close'].min()

        # Check for NaN in actuals
        if pd.isna(actual_high) or pd.isna(actual_low):
            print(f"   ⚠ actual_high or actual_low is NaN: high={actual_high}, low={actual_low}")
            return None
        actual_center = (actual_high + actual_low) / 2

        # 6. Convert actuals to percentage changes and calculate errors
        # Predictions are already in percentage terms, so convert actuals to match
        # Debug: Check values before calculation
        if current_price is None or pd.isna(current_price):
            print(f"   ⚠ current_price is invalid: {current_price}")
            print(f"      actual_high={actual_high}, actual_low={actual_low}")
            return None

        actual_high_pct = (actual_high - current_price) / current_price * 100
        actual_low_pct = (actual_low - current_price) / current_price * 100
        actual_center_pct = (actual_center - current_price) / current_price * 100

        # Errors are in percentage points (e.g., predicted +2.5%, actual +3.2% = 0.7pp error)
        error_high = abs(pred_high - actual_high_pct)
        error_low = abs(pred_low - actual_low_pct)
        error_center = abs(pred_center - actual_center_pct)
        avg_error = (error_high + error_low) / 2

        # 7. Log to database
        prediction_record = {
            'prediction_timestamp': datetime.now(),
            'target_timestamp': target_end,
            'simulation_date': date,  # Historical date being backtested (for multi-model alignment)
            'symbol': 'TSLA',
            'timeframe': '24h',
            'predicted_high': float(pred_high),  # Now in percentage terms
            'predicted_low': float(pred_low),    # Now in percentage terms
            'predicted_center': float(pred_center),
            'predicted_range': float(pred_range),
            'confidence': float(confidence),
            'current_price': float(current_price),  # Needed for percentage → absolute conversion
            'has_earnings': has_earnings,
            'has_macro_event': has_macro,
            'event_type': events[0].get('event_type') if events else None,
            'model_version': 'backtest_v1',
            'feature_dim': feature_extractor.get_feature_dim(),
            # Multi-scale ensemble fields (populated if ensemble mode)
            'model_timeframe': predictions.get('model_timeframe', 'single'),
            'is_ensemble': predictions.get('is_ensemble', False),
            'news_enabled': predictions.get('news_enabled', False),
        }

        # Add sub-predictions if ensemble
        if predictions.get('sub_predictions'):
            for tf, sub_pred in predictions['sub_predictions'].items():
                prediction_record[f'sub_pred_{tf}_high'] = float(sub_pred['predicted_high'])
                prediction_record[f'sub_pred_{tf}_low'] = float(sub_pred['predicted_low'])
                prediction_record[f'sub_pred_{tf}_conf'] = float(sub_pred['confidence'])

        # Defensive check: ensure current_price is a valid number before storing
        if current_price is None or not isinstance(current_price, (int, float)):
            print(f"   ⚠ Current price is invalid type: {type(current_price)} = {current_price}, skipping prediction")
            return None

        if np.isnan(current_price) or current_price <= 0:
            print(f"   ⚠ Current price is invalid value: {current_price}, skipping prediction")
            return None

        pred_id = db.log_prediction(prediction_record)

        # Only update actuals if we have a valid current_price
        try:
            db.update_actual(pred_id, float(actual_high), float(actual_low))
        except Exception as e:
            print(f"   ⚠ Error updating actuals: {e}")

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
        traceback.print_exc()
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
    parser.add_argument('--dates_file', type=str, default=None,
                       help='Path to file containing predefined dates (one per line, YYYY-MM-DD format)')

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

    # 3. Select or load test dates
    if args.dates_file:
        # Load predefined dates from file
        print(f"\n📅 Loading predefined dates from: {args.dates_file}")
        try:
            with open(args.dates_file, 'r') as f:
                date_strings = [line.strip() for line in f if line.strip()]

            test_dates = [datetime.strptime(d, '%Y-%m-%d') for d in date_strings]
            test_dates.sort()

            print(f"✓ Loaded {len(test_dates)} dates from file")
            print(f"  Date range: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")

        except FileNotFoundError:
            print(f"\n❌ Dates file not found: {args.dates_file}")
            return
        except Exception as e:
            print(f"\n❌ Error loading dates file: {e}")
            return
    else:
        # Select random dates with two-phase validation
        test_dates = select_random_dates(
            args.test_year,
            args.num_simulations,
            args.seed,
            data_feed=data_feed,
            metadata=metadata,
            validate=True  # Enable pre-validation for guaranteed success
        )

        if len(test_dates) == 0:
            print("\n❌ No valid dates found for backtesting!")
            print("   Try: (1) Earlier test year, (2) Fewer simulations, (3) Check data files")
            return

        print(f"\n✓ Selected {len(test_dates)} validated dates")
        print(f"  Date range: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")

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
