#!/usr/bin/env python3
"""
AutoTrade2 - ML Predictions Dashboard

All-in-one dashboard for live ML predictions:
- Loads trained models (15min, 1hour, 4hour, daily, ensemble)
- Fetches live data via yfinance
- Runs predictions at correct bar-close times
- Sends Telegram alerts for high-confidence predictions
- Displays predictions with countdown timers
- Shows recent performance metrics

Usage:
    streamlit run ml_dashboard.py
"""

import streamlit as st
import pandas as pd
import torch
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import threading
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.ml.live_data_loader import LiveDataLoader
from src.ml.prediction_cache import PredictionCache
from src.ml.features import TradingFeatureExtractor
from src.ml.database import SQLitePredictionDB
from src.ml.model import LNNTradingModel, LSTMTradingModel
import requests  # For Telegram API


# Global state (for background thread - can't use st.session_state in threads)
GLOBAL_PREDICTION_CACHE = PredictionCache()
GLOBAL_MODELS = {}
GLOBAL_TELEGRAM_TOKEN = None
GLOBAL_TELEGRAM_CHAT_ID = None
GLOBAL_SCHEDULER_RUNNING = False

# Streamlit session state (for UI state only)
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()


def send_telegram_message(token: str, chat_id: str, message: str) -> bool:
    """Simple Telegram message sender."""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def load_telegram_config():
    """Load Telegram credentials from config/api_keys.json."""
    try:
        api_keys_path = Path('config/api_keys.json')
        if api_keys_path.exists():
            with open(api_keys_path) as f:
                keys = json.load(f)
                telegram_config = keys.get('telegram', {})
                return telegram_config.get('bot_token'), telegram_config.get('chat_id')
    except Exception as e:
        st.warning(f"Could not load Telegram config: {e}")

    return None, None


def load_model(model_name: str):
    """Load a trained model from disk."""
    model_path = f'models/lnn_{model_name}.pth'

    if not Path(model_path).exists():
        st.error(f"Model not found: {model_path}")
        return None, None

    try:
        checkpoint = torch.load(model_path, weights_only=False)
        metadata = checkpoint['metadata']

        input_size = metadata['input_size']
        hidden_size = metadata.get('hidden_size', 128)
        model_type = metadata.get('model_type', 'LNN')

        if model_type == 'LNN':
            model = LNNTradingModel(input_size, hidden_size)
        else:
            model = LSTMTradingModel(input_size, hidden_size)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, metadata

    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None, None


def make_prediction(model_name: str, model, metadata, data_loader: LiveDataLoader, feature_extractor):
    """Make a prediction with error handling."""
    try:
        # Load live data
        sequence_length = metadata['sequence_length']
        lookback_days = 1911  # 5 years for 3month features

        aligned_df, data_status = data_loader.load_live_data(lookback_days=lookback_days)

        if len(aligned_df) < sequence_length:
            return None, f"Insufficient data: {len(aligned_df)}/{sequence_length} bars"

        # Extract features
        features_df = feature_extractor.extract_features(aligned_df)

        if len(features_df) < sequence_length:
            return None, f"Insufficient features: {len(features_df)}/{sequence_length}"

        # Create input sequence (numpy first for performance)
        sequence = features_df.tail(sequence_length).values
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).float()

        # Get current price
        current_price = features_df.iloc[-1]['tsla_close']

        # Make prediction
        with torch.no_grad():
            predictions = model.predict(sequence_tensor)

        pred_high = float(predictions['predicted_high'][0])
        pred_low = float(predictions['predicted_low'][0])
        confidence = float(predictions['confidence'][0])

        # Calculate center and range
        pred_center = (pred_high + pred_low) / 2
        pred_range = pred_high - pred_low

        # Convert to absolute prices
        from src.ml.model import percentage_to_absolute
        pred_high_price = float(percentage_to_absolute(pred_high, current_price))
        pred_low_price = float(percentage_to_absolute(pred_low, current_price))
        pred_center_price = float(percentage_to_absolute(pred_center, current_price))

        result = {
            'model': model_name,
            'timestamp': datetime.now(),
            'data_status': data_status,
            'current_price': float(current_price),
            'predicted_high_pct': pred_high,
            'predicted_low_pct': pred_low,
            'predicted_center_pct': pred_center,
            'predicted_range_pct': pred_range,
            'predicted_high_price': pred_high_price,
            'predicted_low_price': pred_low_price,
            'predicted_center_price': pred_center_price,
            'confidence': confidence,
            'prediction_window': '24 hours'
        }

        return result, None

    except Exception as e:
        return None, str(e)


def calculate_minimum_context_days(min_bars_per_timeframe=20):
    """
    Calculate minimum historical lookback needed for complete feature extraction.
    Longest timeframe (3month) requires ~1848 days for 20 bars.
    """
    # 3month timeframe needs the most context
    return 1911  # ~5 years for safe 3month channel features


def load_ensemble_for_dashboard():
    """Load ensemble system (4 sub-models + Meta-LNN coach)."""
    try:
        # Check all required models exist
        required_models = {
            '15min': 'models/lnn_15min.pth',
            '1hour': 'models/lnn_1hour.pth',
            '4hour': 'models/lnn_4hour.pth',
            'daily': 'models/lnn_daily.pth',
            'meta': 'models/meta_lnn.pth'
        }

        for name, path in required_models.items():
            if not Path(path).exists():
                return None, f"Missing {name} model: {path}"

        # Load ensemble
        from src.ml.ensemble import load_ensemble

        ensemble = load_ensemble(
            mode='backtest_no_news',
            device='cpu',
            models_dir='models',
            events_csv='data/tsla_events_REAL.csv'
        )

        return ensemble, None

    except Exception as e:
        return None, f"Error loading ensemble: {e}"


def load_hierarchical_for_dashboard():
    """Load hierarchical LNN system (single 3-layer model)."""
    try:
        # Check model exists
        model_path = Path('models/hierarchical_lnn.pth')
        if not model_path.exists():
            return None, f"Missing hierarchical model: {model_path}"

        # Load via ensemble wrapper (ensemble.py already supports hierarchical mode!)
        from src.ml.ensemble import MultiScaleEnsemble
        from src.ml.events import CombinedEventsHandler

        # Create events handler
        events_handler = CombinedEventsHandler('data/tsla_events_REAL.csv')

        # Load hierarchical model via ensemble interface
        hierarchical_ensemble = MultiScaleEnsemble(
            model_paths={'hierarchical': str(model_path)},
            meta_model_path=None,  # Not needed for hierarchical
            mode='backtest_no_news',
            device='cpu',
            events_handler=events_handler
        )

        return hierarchical_ensemble, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error loading hierarchical model: {e}"


def make_ensemble_prediction(ensemble, feature_extractor):
    """
    Make ensemble prediction using all 4 timeframe models + Meta-LNN.

    Returns:
        (prediction_dict, error_message)
    """
    try:
        # Step 1: Load model metadata for sequence lengths
        model_metadata = {}
        for tf in ['15min', '1hour', '4hour', 'daily']:
            ckpt = torch.load(f'models/lnn_{tf}.pth', weights_only=False)
            model_metadata[tf] = ckpt['metadata']

        # Step 2: Create data loaders for all timeframes
        data_feeds = {
            '15min': LiveDataLoader(timeframe='15min'),
            '1hour': LiveDataLoader(timeframe='1hour'),
            '4hour': LiveDataLoader(timeframe='4hour'),
            'daily': LiveDataLoader(timeframe='daily')
        }

        # Step 3: Load data at all 4 timeframes
        data_dict = {}
        context_days = calculate_minimum_context_days(min_bars_per_timeframe=20)

        for tf in ['15min', '1hour', '4hour', 'daily']:
            seq_len = model_metadata[tf]['sequence_length']

            # Load live data
            aligned_df, status = data_feeds[tf].load_live_data(lookback_days=context_days)

            if len(aligned_df) < seq_len:
                return None, f"Insufficient data for {tf}: {len(aligned_df)}/{seq_len} bars"

            # Extract features
            features_df = feature_extractor.extract_features(aligned_df)

            if len(features_df) < seq_len:
                return None, f"Insufficient features for {tf}: {len(features_df)}/{seq_len}"

            # Create input tensor
            sequence = features_df.tail(seq_len).values
            data_dict[tf] = torch.tensor(sequence, dtype=torch.float32)

        # Step 4: Get current price and market state
        main_df, _ = data_feeds['1hour'].load_live_data(lookback_days=context_days)
        main_features = feature_extractor.extract_features(main_df)
        current_price = float(main_features.iloc[-1]['tsla_close'])
        current_idx = len(main_features) - 1

        # Step 5: Get ensemble prediction
        predictions = ensemble.predict(
            data=data_dict,
            features_df=main_features,
            current_idx=current_idx,
            timestamp=datetime.now()
        )

        # Step 6: Format result
        from src.ml.model import percentage_to_absolute

        pred_high = float(predictions['predicted_high'])
        pred_low = float(predictions['predicted_low'])
        pred_center = (pred_high + pred_low) / 2
        pred_range = pred_high - pred_low

        result = {
            'model': 'ensemble',
            'timestamp': datetime.now(),
            'data_status': 'LIVE',
            'current_price': current_price,
            'predicted_high_pct': pred_high,
            'predicted_low_pct': pred_low,
            'predicted_center_pct': pred_center,
            'predicted_range_pct': pred_range,
            'predicted_high_price': float(percentage_to_absolute(pred_high, current_price)),
            'predicted_low_price': float(percentage_to_absolute(pred_low, current_price)),
            'predicted_center_price': float(percentage_to_absolute(pred_center, current_price)),
            'confidence': float(predictions['confidence']),
            'sub_predictions': predictions.get('sub_predictions', {}),
            'prediction_window': '24 hours'
        }

        return result, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)


def make_hierarchical_prediction(hierarchical_ensemble, feature_extractor):
    """
    Make hierarchical LNN prediction using 1-min data.

    Returns:
        (prediction_dict, error_message)
    """
    try:
        # Step 1: Load 1-min data
        data_feed = LiveDataLoader(timeframe='1min')
        context_days = calculate_minimum_context_days(min_bars_per_timeframe=20)

        aligned_df, status = data_feed.load_live_data(lookback_days=context_days)

        # Step 2: Extract features (473 features with events)
        features_df = feature_extractor.extract_features(aligned_df)

        # Step 3: Get sequence (200 1-min bars)
        sequence_length = 200
        if len(features_df) < sequence_length:
            return None, f"Insufficient data: {len(features_df)}/{sequence_length} bars"

        sequence = features_df.tail(sequence_length).values
        x_tensor = torch.tensor(sequence, dtype=torch.float32)

        # Step 4: Prepare data dict (hierarchical expects {'1min': tensor})
        data_dict = {'1min': x_tensor}

        # Step 5: Get current index and price
        current_idx = len(features_df) - 1
        current_price = float(features_df.iloc[-1]['tsla_close'])

        # Step 6: Call hierarchical ensemble predict
        predictions = hierarchical_ensemble.predict(
            data=data_dict,
            features_df=features_df,
            current_idx=current_idx,
            timestamp=datetime.now()
        )

        # Step 7: Format result for dashboard
        from src.ml.model import percentage_to_absolute

        pred_high = float(predictions['predicted_high'])
        pred_low = float(predictions['predicted_low'])
        pred_center = (pred_high + pred_low) / 2
        pred_range = pred_high - pred_low

        result = {
            'model': 'hierarchical',
            'timestamp': datetime.now(),
            'data_status': status,
            'current_price': current_price,
            'predicted_high_pct': pred_high,
            'predicted_low_pct': pred_low,
            'predicted_center_pct': pred_center,
            'predicted_range_pct': pred_range,
            'predicted_high_price': float(percentage_to_absolute(pred_high, current_price)),
            'predicted_low_price': float(percentage_to_absolute(pred_low, current_price)),
            'predicted_center_price': float(percentage_to_absolute(pred_center, current_price)),
            'confidence': float(predictions['confidence']),
            'sub_predictions': predictions.get('sub_predictions', {}),  # Layer predictions
            'fusion_weights': predictions.get('fusion_weights', {}),
            'features_dict': features_df.iloc[-1].to_dict(),  # For channel context
            'prediction_window': '30 minutes'
        }

        return result, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)


def should_update_prediction(model_name: str, cache: PredictionCache) -> bool:
    """Check if it's time to update this model's prediction."""
    now = datetime.now()

    # Check if cached prediction exists and is still valid
    cached = cache.get(model_name)
    if cached is not None:
        return False  # Still valid, don't update

    # No cache or expired - check if we're at a bar close time
    if model_name == '15min':
        return now.minute % 15 == 0
    elif model_name == '1hour':
        return now.minute == 0
    elif model_name == '4hour':
        return now.hour % 4 == 0 and now.minute == 0
    elif model_name == 'daily':
        return now.hour == 16 and now.minute == 0
    elif model_name == 'ensemble':
        # Ensemble updates hourly (same as 1-hour model)
        return now.minute == 0
    elif model_name == 'hierarchical':
        # Hierarchical updates every 30 minutes (fast layer schedule)
        return now.minute % 30 == 0
    else:
        return False


def send_telegram_alert(prediction: Dict, token: str, chat_id: str):
    """Send Telegram alert for high-confidence prediction."""
    try:
        model = prediction['model']
        current = prediction['current_price']
        high_pct = prediction['predicted_high_pct']
        low_pct = prediction['predicted_low_pct']
        high_price = prediction['predicted_high_price']
        low_price = prediction['predicted_low_price']
        conf = prediction['confidence']

        # Determine signal
        center = (high_pct + low_pct) / 2
        if center > 1.0:
            signal = "BULLISH"
            emoji = "📈"
        elif center < -1.0:
            signal = "BEARISH"
            emoji = "📉"
        else:
            signal = "NEUTRAL"
            emoji = "➡️"

        message = f"""🤖 <b>AutoTrade2 ML Prediction</b>

<b>Model:</b> {model} (0.99% avg error)
<b>Time:</b> {prediction['timestamp'].strftime('%Y-%m-%d %H:%M ET')}

<b>Current:</b> ${current:.2f}

<b>Prediction (Next 24h):</b>
  High: ${high_price:.2f} ({high_pct:+.1f}%) ▲
  Low: ${low_price:.2f} ({low_pct:+.1f}%) ▼

<b>Confidence:</b> {conf:.0%} {'🟢' if conf > 0.75 else '🟡' if conf > 0.5 else '🔴'}

<b>Signal:</b> {emoji} {signal}
<b>Bias:</b> {center:+.1f}% center

<i>Powered by 245-feature LNN
(SPY+TSLA multi-scale analysis)</i>"""

        return send_telegram_message(token, chat_id, message)

    except Exception as e:
        print(f"Telegram alert failed: {e}")
        return False


def prediction_scheduler_thread(selected_models, alert_threshold):
    """
    Background thread that runs predictions at correct bar times.
    Uses global variables (not st.session_state) for thread safety.
    """
    global GLOBAL_SCHEDULER_RUNNING, GLOBAL_PREDICTION_CACHE, GLOBAL_MODELS
    global GLOBAL_TELEGRAM_TOKEN, GLOBAL_TELEGRAM_CHAT_ID

    GLOBAL_SCHEDULER_RUNNING = True

    data_loaders = {}
    feature_extractor = TradingFeatureExtractor()
    db = SQLitePredictionDB()

    # Load Telegram config if configured
    token, chat_id = load_telegram_config()
    GLOBAL_TELEGRAM_TOKEN = token
    GLOBAL_TELEGRAM_CHAT_ID = chat_id

    while GLOBAL_SCHEDULER_RUNNING:
        try:
            for model_name in selected_models:
                # Check if we should update this model
                if should_update_prediction(model_name, GLOBAL_PREDICTION_CACHE):

                    if model_name == 'ensemble':
                        # Special handling for ensemble
                        # Load ensemble if not already loaded
                        if 'ensemble' not in GLOBAL_MODELS:
                            ensemble, error = load_ensemble_for_dashboard()
                            if ensemble:
                                GLOBAL_MODELS['ensemble'] = ensemble
                                print(f"✓ Loaded ensemble for background predictions")
                            else:
                                print(f"✗ Failed to load ensemble: {error}")
                                continue

                        # Make ensemble prediction
                        ensemble = GLOBAL_MODELS['ensemble']
                        prediction, error = make_ensemble_prediction(ensemble, feature_extractor)

                    elif model_name == 'hierarchical':
                        # Special handling for hierarchical
                        # Load hierarchical if not already loaded
                        if 'hierarchical' not in GLOBAL_MODELS:
                            hierarchical_ensemble, error = load_hierarchical_for_dashboard()
                            if hierarchical_ensemble:
                                GLOBAL_MODELS['hierarchical'] = hierarchical_ensemble
                                print(f"✓ Loaded hierarchical model for background predictions")
                            else:
                                print(f"✗ Failed to load hierarchical: {error}")
                                continue

                        # Make hierarchical prediction
                        hierarchical_ensemble = GLOBAL_MODELS['hierarchical']
                        prediction, error = make_hierarchical_prediction(hierarchical_ensemble, feature_extractor)

                    else:
                        # Regular model handling
                        # Load model if not already loaded
                        if model_name not in GLOBAL_MODELS:
                            model, metadata = load_model(model_name)
                            if model:
                                GLOBAL_MODELS[model_name] = {'model': model, 'metadata': metadata}

                        if model_name in GLOBAL_MODELS:
                            model_info = GLOBAL_MODELS[model_name]

                            # Create data loader for this timeframe if needed
                            timeframe = model_info['metadata']['input_timeframe']
                            if timeframe not in data_loaders:
                                data_loaders[timeframe] = LiveDataLoader(timeframe=timeframe)

                            # Make prediction
                            prediction, error = make_prediction(
                                model_name,
                                model_info['model'],
                                model_info['metadata'],
                                data_loaders[timeframe],
                                feature_extractor
                            )
                        else:
                            prediction = None
                            error = "Model not loaded"

                    # Cache and log prediction (works for both ensemble and regular models)
                    if prediction:
                        # Cache the prediction (use global for thread safety)
                        GLOBAL_PREDICTION_CACHE.set(model_name, prediction)

                        # Log to database
                        try:
                            log_data = {
                                'prediction_timestamp': prediction['timestamp'],
                                'target_timestamp': prediction['timestamp'] + timedelta(hours=24),
                                'simulation_date': None,  # Live prediction
                                'symbol': 'TSLA',
                                'timeframe': '24h',
                                'model_timeframe': model_name,
                                'is_ensemble': (model_name == 'ensemble'),
                                'predicted_high': prediction['predicted_high_pct'],
                                'predicted_low': prediction['predicted_low_pct'],
                                'predicted_center': prediction['predicted_center_pct'],
                                'predicted_range': prediction['predicted_range_pct'],
                                'confidence': prediction['confidence'],
                                'current_price': prediction['current_price'],
                                'feature_dim': 245
                            }

                            # Add sub-predictions if ensemble
                            if model_name == 'ensemble' and 'sub_predictions' in prediction:
                                for tf, sp in prediction['sub_predictions'].items():
                                    log_data[f'sub_pred_{tf}_high'] = float(sp['predicted_high'])
                                    log_data[f'sub_pred_{tf}_low'] = float(sp['predicted_low'])
                                    log_data[f'sub_pred_{tf}_conf'] = float(sp['confidence'])

                            db.log_prediction(log_data)
                        except Exception as e:
                            print(f"Error logging to DB: {e}")

                        # Send alert if high confidence
                        if token and chat_id and prediction['confidence'] > alert_threshold:
                            send_telegram_alert(prediction, token, chat_id)

            # Sleep for 1 minute before next check
            time.sleep(60)

        except Exception as e:
            print(f"Error in scheduler: {e}")
            time.sleep(60)


def main():
    st.set_page_config(
        page_title="AutoTrade2 - ML Predictions",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 AutoTrade2 - ML Predictions Dashboard")
    st.markdown("*Live predictions from trained Liquid Neural Networks*")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Model selection
    st.sidebar.subheader("Models")
    model_15min = st.sidebar.checkbox("15min Model (Best: 0.99% error)", value=True)
    model_1hour = st.sidebar.checkbox("1hour Model (1.41% error)", value=True)
    model_4hour = st.sidebar.checkbox("4hour Model (2.21% error)", value=False)
    model_daily = st.sidebar.checkbox("Daily Model (11.96% error)", value=False)
    model_ensemble = st.sidebar.checkbox("Ensemble (4 models + Meta-LNN)", value=False)
    model_hierarchical = st.sidebar.checkbox("Hierarchical LNN (3-layer: Fast/Medium/Slow) ⭐", value=False)

    selected_models = []
    if model_15min:
        selected_models.append('15min')
    if model_1hour:
        selected_models.append('1hour')
    if model_4hour:
        selected_models.append('4hour')
    if model_daily:
        selected_models.append('daily')
    if model_ensemble:
        selected_models.append('ensemble')
    if model_hierarchical:
        selected_models.append('hierarchical')

    # Alert settings
    st.sidebar.subheader("Alerts")
    alert_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.70, 0.05)
    alerts_enabled = st.sidebar.checkbox("Enable Telegram Alerts", value=True)

    # Check Telegram config
    token, chat_id = load_telegram_config()
    if alerts_enabled:
        if token and chat_id:
            st.sidebar.success(f"✓ Telegram configured")
        else:
            st.sidebar.warning("⚠️ Telegram not configured")

    # Auto-refresh
    st.sidebar.subheader("Display")
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10)

    # Manual controls
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

    if st.sidebar.button("🗑️ Clear Cache"):
        GLOBAL_PREDICTION_CACHE.invalidate_all()
        st.success("Cache cleared!")

    # Start prediction scheduler if not running
    global GLOBAL_SCHEDULER_RUNNING
    if not GLOBAL_SCHEDULER_RUNNING and len(selected_models) > 0:
        thread = threading.Thread(
            target=prediction_scheduler_thread,
            args=(selected_models, alert_threshold),
            daemon=True
        )
        thread.start()
        st.sidebar.success("✓ Prediction scheduler started")

    # Main content
    if len(selected_models) == 0:
        st.warning("⚠️ Please select at least one model from the sidebar")
        return

    # Current market status
    col1, col2, col3 = st.columns(3)

    with col1:
        # Try to get current price from any loaded model's data
        try:
            loader = LiveDataLoader(timeframe='1min')
            df, status = loader.load_live_data(lookback_days=1)
            current_price = df.iloc[-1]['tsla_close']
            latest_time = df.index[-1]
            st.metric("Current TSLA Price", f"${current_price:.2f}")
        except:
            st.metric("Current TSLA Price", "Loading...")
            current_price = None
            latest_time = None

    with col2:
        if latest_time:
            data_age = (datetime.now() - latest_time).total_seconds() / 60
            st.metric("Data Age", f"{data_age:.1f} minutes")
        else:
            st.metric("Data Age", "...")

    with col3:
        loader_temp = LiveDataLoader()
        market_open = loader_temp.is_market_open()
        st.metric("Market Status", "OPEN 🟢" if market_open else "CLOSED 🔴")

    st.markdown("---")

    # Display predictions for each selected model
    for model_name in selected_models:
        render_model_prediction(model_name, alert_threshold)

    st.markdown("---")

    # Recent performance
    st.subheader("📊 Recent Performance (Last 30 Days)")
    render_recent_performance()

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


def render_ensemble_prediction(alert_threshold: float):
    """Render ensemble prediction card with sub-model breakdown."""
    st.subheader("🔮 ENSEMBLE Model (Meta-LNN Coach)")

    # Get cached prediction
    cached = GLOBAL_PREDICTION_CACHE.get('ensemble')

    if cached is None:
        # No prediction yet - trigger one
        with st.spinner("Loading ensemble (4 models + Meta-LNN coach) and making prediction..."):
            # Load ensemble if not already loaded
            if 'ensemble' not in GLOBAL_MODELS:
                ensemble, error = load_ensemble_for_dashboard()
                if ensemble:
                    GLOBAL_MODELS['ensemble'] = ensemble
                else:
                    st.error(f"Failed to load ensemble: {error}")
                    return

            ensemble = GLOBAL_MODELS['ensemble']
            feature_extractor = TradingFeatureExtractor()

            # Make ensemble prediction
            prediction, error = make_ensemble_prediction(ensemble, feature_extractor)

            if prediction:
                GLOBAL_PREDICTION_CACHE.set('ensemble', prediction)
                cached = prediction
            else:
                st.error(f"Error: {error}")
                return

    # Display main ensemble prediction
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown(f"**Predicted High:** ${cached['predicted_high_price']:.2f} ({cached['predicted_high_pct']:+.1f}%)")
        st.markdown(f"**Predicted Low:** ${cached['predicted_low_price']:.2f} ({cached['predicted_low_pct']:+.1f}%)")

    with col2:
        conf_pct = cached['confidence'] * 100
        st.markdown(f"**Confidence:** {conf_pct:.0f}%")

        # Confidence bar
        if conf_pct > 75:
            st.progress(conf_pct / 100, "🟢 High")
        elif conf_pct > 50:
            st.progress(conf_pct / 100, "🟡 Medium")
        else:
            st.progress(conf_pct / 100, "🔴 Low")

    with col3:
        # Countdown to next update
        seconds_until = GLOBAL_PREDICTION_CACHE.get_time_until_update('ensemble')
        if seconds_until:
            minutes = int(seconds_until // 60)
            secs = int(seconds_until % 60)
            st.markdown(f"**Next update:**")
            st.markdown(f"`{minutes:02d}:{secs:02d}`")

    # Additional info
    st.caption(f"Last updated: {cached['timestamp'].strftime('%H:%M:%S')} | "
               f"Data: {cached['data_status']} | "
               f"Window: {cached['prediction_window']}")

    # Show sub-predictions breakdown
    with st.expander("📊 View Sub-Model Predictions"):
        sub_preds = cached.get('sub_predictions', {})

        if sub_preds:
            st.markdown("**Individual model predictions that went into ensemble:**")

            for tf in ['15min', '1hour', '4hour', 'daily']:
                if tf in sub_preds:
                    sp = sub_preds[tf]
                    col1, col2, col3 = st.columns([1, 2, 1])

                    with col1:
                        st.markdown(f"**{tf}:**")

                    with col2:
                        st.markdown(f"High: {sp['predicted_high']:+.2f}%, Low: {sp['predicted_low']:+.2f}%")

                    with col3:
                        st.markdown(f"Conf: {sp['confidence']:.2f}")
        else:
            st.info("No sub-predictions available")

    # Alert indicator
    if cached['confidence'] > alert_threshold:
        st.success(f"✅ Alert sent (confidence > {alert_threshold:.0%})")

    st.markdown("---")


def render_hierarchical_prediction(alert_threshold: float):
    """Render hierarchical LNN prediction card with layer breakdown."""
    st.subheader("🔮 HIERARCHICAL LNN (3-Layer: Fast/Medium/Slow)")

    # Get cached prediction
    cached = GLOBAL_PREDICTION_CACHE.get('hierarchical')

    if cached is None:
        st.info("⏳ Waiting for first prediction... (updates every 30 min)")
        return

    # Display main prediction
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown(f"**Predicted High:** ${cached['predicted_high_price']:.2f} ({cached['predicted_high_pct']:+.1f}%)")
        st.markdown(f"**Predicted Low:** ${cached['predicted_low_price']:.2f} ({cached['predicted_low_pct']:+.1f}%)")

    with col2:
        conf_pct = cached['confidence'] * 100
        st.markdown(f"**Confidence:** {conf_pct:.0f}%")

        if conf_pct > 75:
            st.progress(conf_pct / 100, "🟢 High")
        elif conf_pct > 50:
            st.progress(conf_pct / 100, "🟡 Medium")
        else:
            st.progress(conf_pct / 100, "🔴 Low")

    with col3:
        seconds_until = GLOBAL_PREDICTION_CACHE.get_time_until_update('hierarchical')
        if seconds_until:
            minutes = int(seconds_until // 60)
            secs = int(seconds_until % 60)
            st.markdown(f"**Next update:**")
            st.markdown(f"`{minutes:02d}:{secs:02d}`")

    st.caption(f"Last updated: {cached['timestamp'].strftime('%H:%M:%S')} | "
               f"Data: {cached['data_status']} | "
               f"Window: {cached['prediction_window']}")

    # Alert indicator
    if cached['confidence'] > alert_threshold:
        st.success(f"✅ Alert sent (confidence > {alert_threshold:.0%})")

    # Show layer predictions breakdown
    with st.expander("📊 Layer Predictions (Fast/Medium/Slow)"):
        layer_preds = cached.get('sub_predictions', {})

        if layer_preds:
            st.markdown("**Individual layer predictions:**")

            for layer in ['fast', 'medium', 'slow']:
                if layer in layer_preds:
                    lp = layer_preds[layer]
                    col1, col2, col3 = st.columns([1, 2, 1])

                    with col1:
                        emoji = "⚡" if layer == 'fast' else "🔄" if layer == 'medium' else "🐢"
                        st.markdown(f"**{emoji} {layer.capitalize()}:**")

                    with col2:
                        st.markdown(f"High: {lp['predicted_high']:+.2f}%, Low: {lp['predicted_low']:+.2f}%")

                    with col3:
                        st.markdown(f"Conf: {lp['confidence']:.2f}")

            # Show fusion weights
            fusion_weights = cached.get('fusion_weights', {})
            if fusion_weights and len(fusion_weights) == 3:
                st.markdown("\n**Fusion weights (how much each layer contributes):**")
                st.markdown(f"- ⚡ Fast: {fusion_weights[0]:.1%}")
                st.markdown(f"- 🔄 Medium: {fusion_weights[1]:.1%}")
                st.markdown(f"- 🐢 Slow: {fusion_weights[2]:.1%}")
        else:
            st.info("No layer predictions available")

    # Channel context (already added in earlier commit)
    with st.expander("📊 Channel Context (1H Timeframe)"):
        features_dict = cached.get('features_dict', {})

        channel_slope_pct = features_dict.get('tsla_channel_1h_slope_pct', 0.0)
        channel_position = features_dict.get('tsla_channel_1h_position', 0.5)
        ping_pongs = features_dict.get('tsla_channel_1h_ping_pongs', 0)
        r_squared = features_dict.get('tsla_channel_1h_r_squared', 0.0)

        # Direction indicator
        if abs(channel_slope_pct) > 0.2:
            emoji = "📈" if channel_slope_pct > 0 else "📉"
            strength = "Strong"
        elif abs(channel_slope_pct) > 0.1:
            emoji = "↗️" if channel_slope_pct > 0 else "↘️"
            strength = "Moderate"
        else:
            emoji = "➡️"
            strength = "Sideways"

        direction_text = "Bullish" if channel_slope_pct > 0 else "Bearish" if channel_slope_pct < 0 else "Ranging"

        st.markdown(f"{emoji} **{strength} {direction_text} Channel**")
        st.markdown(f"- Slope: `{channel_slope_pct:+.2f}% per bar`")
        st.markdown(f"- Position in channel: `{channel_position:.2f}` (0=bottom, 1=top)")
        st.markdown(f"- Ping-pongs: `{ping_pongs}` bounces")
        st.markdown(f"- Channel quality (R²): `{r_squared:.3f}`")

    st.markdown("---")


def render_model_prediction(model_name: str, alert_threshold: float):
    """Render prediction card for one model."""

    # Route ensemble to special renderer
    if model_name == 'ensemble':
        render_ensemble_prediction(alert_threshold)
        return

    # Route hierarchical to special renderer
    if model_name == 'hierarchical':
        render_hierarchical_prediction(alert_threshold)
        return

    st.subheader(f"🔮 {model_name.upper()} Model")

    # Get cached prediction
    cached = GLOBAL_PREDICTION_CACHE.get(model_name)

    if cached is None:
        # No prediction yet - trigger one
        with st.spinner(f"Loading {model_name} model and making initial prediction..."):
            # Load model
            if model_name not in GLOBAL_MODELS:
                model, metadata = load_model(model_name)
                if model:
                    GLOBAL_MODELS[model_name] = {'model': model, 'metadata': metadata}

            if model_name in GLOBAL_MODELS:
                model_info = GLOBAL_MODELS[model_name]
                timeframe = model_info['metadata']['input_timeframe']
                data_loader = LiveDataLoader(timeframe=timeframe)
                feature_extractor = TradingFeatureExtractor()

                prediction, error = make_prediction(
                    model_name,
                    model_info['model'],
                    model_info['metadata'],
                    data_loader,
                    feature_extractor
                )

                if prediction:
                    GLOBAL_PREDICTION_CACHE.set(model_name, prediction)
                    cached = prediction
                else:
                    st.error(f"Error: {error}")
                    return
            else:
                st.error(f"Could not load {model_name} model")
                return

    # Display prediction
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown(f"**Predicted High:** ${cached['predicted_high_price']:.2f} ({cached['predicted_high_pct']:+.1f}%)")
        st.markdown(f"**Predicted Low:** ${cached['predicted_low_price']:.2f} ({cached['predicted_low_pct']:+.1f}%)")

    with col2:
        conf_pct = cached['confidence'] * 100
        st.markdown(f"**Confidence:** {conf_pct:.0f}%")

        # Confidence bar
        if conf_pct > 75:
            st.progress(conf_pct / 100, "🟢 High")
        elif conf_pct > 50:
            st.progress(conf_pct / 100, "🟡 Medium")
        else:
            st.progress(conf_pct / 100, "🔴 Low")

    with col3:
        # Countdown to next update
        seconds_until = GLOBAL_PREDICTION_CACHE.get_time_until_update(model_name)
        if seconds_until:
            minutes = int(seconds_until // 60)
            secs = int(seconds_until % 60)
            st.markdown(f"**Next update:**")
            st.markdown(f"`{minutes:02d}:{secs:02d}`")

    # Additional info
    st.caption(f"Last updated: {cached['timestamp'].strftime('%H:%M:%S')} | "
               f"Data: {cached['data_status']} | "
               f"Window: {cached['prediction_window']}")

    # Alert indicator
    if cached['confidence'] > alert_threshold:
        st.success(f"✅ Alert sent (confidence > {alert_threshold:.0%})")

    # Channel context (v3.8)
    with st.expander("📊 Channel Context (1H Timeframe)"):
        # Extract channel metrics from cached features if available
        features_dict = cached.get('features_dict', {})

        channel_slope_pct = features_dict.get('tsla_channel_1h_slope_pct', 0.0)
        channel_position = features_dict.get('tsla_channel_1h_position', 0.5)
        ping_pongs = features_dict.get('tsla_channel_1h_ping_pongs', 0)
        r_squared = features_dict.get('tsla_channel_1h_r_squared', 0.0)

        # Direction indicator
        if abs(channel_slope_pct) > 0.2:
            emoji = "📈" if channel_slope_pct > 0 else "📉"
            strength = "Strong"
        elif abs(channel_slope_pct) > 0.1:
            emoji = "↗️" if channel_slope_pct > 0 else "↘️"
            strength = "Moderate"
        else:
            emoji = "➡️"
            strength = "Sideways"

        direction_text = "Bullish" if channel_slope_pct > 0 else "Bearish" if channel_slope_pct < 0 else "Ranging"

        st.markdown(f"{emoji} **{strength} {direction_text} Channel**")
        st.markdown(f"- Slope: `{channel_slope_pct:+.2f}% per bar`")
        st.markdown(f"- Position in channel: `{channel_position:.2f}` (0=bottom, 1=top)")
        st.markdown(f"- Ping-pongs: `{ping_pongs}` bounces")
        st.markdown(f"- Channel quality (R²): `{r_squared:.3f}`")

    st.markdown("---")


def render_recent_performance():
    """Show recent prediction performance from database."""
    try:
        db = SQLitePredictionDB()

        # Get predictions from last 30 days with actuals
        query = """
            SELECT
                model_timeframe,
                COUNT(*) as num_predictions,
                ROUND(AVG(absolute_error), 2) as avg_error,
                ROUND(AVG(confidence), 2) as avg_confidence,
                ROUND(MIN(absolute_error), 2) as best_error,
                ROUND(MAX(absolute_error), 2) as worst_error
            FROM predictions
            WHERE timestamp >= datetime('now', '-30 days')
              AND has_actuals = 1
              AND simulation_date IS NULL
            GROUP BY model_timeframe
            ORDER BY avg_error ASC
        """

        df = pd.read_sql(query, db.session.bind)

        if len(df) > 0:
            st.dataframe(
                df,
                column_config={
                    "model_timeframe": "Model",
                    "num_predictions": "Predictions",
                    "avg_error": "Avg Error (%)",
                    "avg_confidence": "Avg Confidence",
                    "best_error": "Best (%)",
                    "worst_error": "Worst (%)"
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No live predictions with actuals yet. Predictions need 24 hours to get actuals.")

    except Exception as e:
        st.warning(f"Could not load performance metrics: {e}")


def check_api_keys():
    """
    Check for optional API keys on startup (runs in terminal before Streamlit).

    These are for auto-updating event data (earnings, FOMC).
    Not required for operation - system works without them.
    """
    import os

    print("\n" + "="*70)
    print("  AutoTrade2 - API Key Configuration Check")
    print("="*70)

    keys_configured = True

    # Check Alpha Vantage (for TSLA earnings)
    if not os.getenv("ALPHA_VANTAGE_API_KEY"):
        print("\n📋 Alpha Vantage API Key (Optional - for auto-updating TSLA earnings)")
        print("   Purpose: Fetch future TSLA earnings dates automatically")
        print("   Get free key: https://www.alphavantage.co/support/#api-key")
        print("   Or: Set environment variable ALPHA_VANTAGE_API_KEY")
        key = input("\n   Enter Alpha Vantage API key (or press Enter to skip): ").strip()
        if key:
            os.environ['ALPHA_VANTAGE_API_KEY'] = key
            print("   ✓ Alpha Vantage key configured for this session")
        else:
            keys_configured = False
            print("   ⚠️  Skipped - will use existing CSV data only")

    else:
        print("\n✅ Alpha Vantage API key found")

    # Check FRED (for FOMC, CPI, NFP)
    if not os.getenv("FRED_API_KEY"):
        print("\n📋 FRED API Key (Optional - for auto-updating macro events)")
        print("   Purpose: Fetch future FOMC, CPI, NFP dates automatically")
        print("   Get free key: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("   Or: Set environment variable FRED_API_KEY")
        key = input("\n   Enter FRED API key (or press Enter to skip): ").strip()
        if key:
            os.environ['FRED_API_KEY'] = key
            print("   ✓ FRED key configured for this session")
        else:
            keys_configured = False
            print("   ⚠️  Skipped - will use existing CSV data only")
    else:
        print("\n✅ FRED API key found")

    if not keys_configured:
        print("\n" + "="*70)
        print("  ℹ️  API keys are optional - system will work without them")
        print("  ℹ️  To update event data, run: python update_events_from_api.py")
        print("  ℹ️  Or manually update data/tsla_events_REAL.csv")
        print("="*70)

    print("\n🚀 Starting AutoTrade2 Dashboard...\n")
    time.sleep(1)  # Brief pause so user sees the messages


if __name__ == '__main__':
    check_api_keys()  # Prompt for API keys in terminal (before Streamlit starts)
    main()            # Start Streamlit dashboard
