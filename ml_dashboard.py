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


# Global state
if 'prediction_cache' not in st.session_state:
    st.session_state.prediction_cache = PredictionCache()

if 'models' not in st.session_state:
    st.session_state.models = {}

if 'telegram_token' not in st.session_state:
    st.session_state.telegram_token = None

if 'telegram_chat_id' not in st.session_state:
    st.session_state.telegram_chat_id = None

if 'scheduler_running' not in st.session_state:
    st.session_state.scheduler_running = False


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

        # Create input sequence
        sequence = features_df.tail(sequence_length).values
        sequence_tensor = torch.tensor([sequence], dtype=torch.float32)

        # Get current price
        current_price = features_df.iloc[-1]['tsla_close']

        # Make prediction
        with torch.no_grad():
            predictions = model.predict(sequence_tensor)

        pred_high = predictions['predicted_high'][0]
        pred_low = predictions['predicted_low'][0]
        confidence = predictions['confidence'][0]

        # Convert to absolute prices
        from src.ml.model import percentage_to_absolute
        pred_high_price = percentage_to_absolute(pred_high, current_price)
        pred_low_price = percentage_to_absolute(pred_low, current_price)

        result = {
            'model': model_name,
            'timestamp': datetime.now(),
            'data_status': data_status,
            'current_price': current_price,
            'predicted_high_pct': pred_high,
            'predicted_low_pct': pred_low,
            'predicted_high_price': pred_high_price,
            'predicted_low_price': pred_low_price,
            'confidence': confidence,
            'prediction_window': '24 hours'
        }

        return result, None

    except Exception as e:
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
    """
    st.session_state.scheduler_running = True

    data_loaders = {}
    feature_extractor = TradingFeatureExtractor()
    db = SQLitePredictionDB()

    # Load Telegram config if configured
    token, chat_id = load_telegram_config()
    st.session_state.telegram_token = token
    st.session_state.telegram_chat_id = chat_id

    while st.session_state.scheduler_running:
        try:
            for model_name in selected_models:
                # Check if we should update this model
                if should_update_prediction(model_name, st.session_state.prediction_cache):
                    # Load model if not already loaded
                    if model_name not in st.session_state.models:
                        model, metadata = load_model(model_name)
                        if model:
                            st.session_state.models[model_name] = {'model': model, 'metadata': metadata}

                    if model_name in st.session_state.models:
                        model_info = st.session_state.models[model_name]

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

                        if prediction:
                            # Cache the prediction
                            st.session_state.prediction_cache.set(model_name, prediction)

                            # Log to database
                            try:
                                db.log_prediction({
                                    'prediction_timestamp': prediction['timestamp'],
                                    'target_timestamp': prediction['timestamp'] + timedelta(hours=24),
                                    'simulation_date': None,  # Live prediction
                                    'symbol': 'TSLA',
                                    'timeframe': '24h',
                                    'model_timeframe': model_name,
                                    'is_ensemble': False,
                                    'predicted_high': prediction['predicted_high_pct'],
                                    'predicted_low': prediction['predicted_low_pct'],
                                    'confidence': prediction['confidence'],
                                    'current_price': prediction['current_price'],
                                    'feature_dim': 245
                                })
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
    model_ensemble = st.sidebar.checkbox("Ensemble (preliminary)", value=False)

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
        st.session_state.prediction_cache.invalidate_all()
        st.success("Cache cleared!")

    # Start prediction scheduler if not running
    if not st.session_state.scheduler_running and len(selected_models) > 0:
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


def render_model_prediction(model_name: str, alert_threshold: float):
    """Render prediction card for one model."""
    st.subheader(f"🔮 {model_name.upper()} Model")

    # Get cached prediction
    cached = st.session_state.prediction_cache.get(model_name)

    if cached is None:
        # No prediction yet - trigger one
        with st.spinner(f"Loading {model_name} model and making initial prediction..."):
            # Load model
            if model_name not in st.session_state.models:
                model, metadata = load_model(model_name)
                if model:
                    st.session_state.models[model_name] = {'model': model, 'metadata': metadata}

            if model_name in st.session_state.models:
                model_info = st.session_state.models[model_name]
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
                    st.session_state.prediction_cache.set(model_name, prediction)
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
        seconds_until = st.session_state.prediction_cache.get_time_until_update(model_name)
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


if __name__ == '__main__':
    main()
