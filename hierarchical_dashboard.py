#!/usr/bin/env python3
"""
Hierarchical LNN Dashboard - Simplified

Shows BEST trading opportunity based on highest confidence layer prediction.

Usage:
    streamlit run hierarchical_dashboard.py

Features:
- Loads only hierarchical model (fast/medium/slow layers)
- Displays best opportunity (highest confidence)
- Shows time-to-target estimates
- Channel context and warnings
- Alternative opportunities
- Auto-refresh every 30 minutes
"""

import streamlit as st
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.hierarchical_model import load_hierarchical_model
from src.ml.features import TradingFeatureExtractor
from src.ml.data_feed import CSVDataFeed
from src.ml.events import CombinedEventsHandler
import config


# Page config
st.set_page_config(
    page_title="AutoTrade2 - Hierarchical LNN",
    page_icon="🎯",
    layout="wide"
)


def load_hierarchical():
    """Load hierarchical model"""
    model_path = Path('models/hierarchical_lnn.pth')

    if not model_path.exists():
        st.error(f"❌ Model not found: {model_path}")
        st.info("Train the model first: `python train_hierarchical.py --interactive`")
        st.stop()

    try:
        model = load_hierarchical_model(str(model_path), device='cpu')
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()


def estimate_time_to_target(layer_name):
    """Estimate time to reach target based on layer"""
    estimates = {
        'fast': ('30-60 minutes', 0.5),      # Fast layer: 30min-1h
        'medium': ('4-8 hours', 6),          # Medium layer: 4-8h
        'slow': ('2-4 weeks', 21)            # Slow layer: weeks
    }
    return estimates.get(layer_name, ('Unknown', 1))


def detect_channel_warning(features_dict):
    """Detect if prediction involves channel break/resistance"""
    warnings = []

    # Check 1H channel position
    position_1h = features_dict.get('tsla_channel_1h_position', 0.5)
    r_squared_1h = features_dict.get('tsla_channel_1h_r_squared', 0.0)

    if position_1h > 0.85 and r_squared_1h > 0.7:
        warnings.append("⚠️ Near 1H channel top (resistance)")
    elif position_1h < 0.15 and r_squared_1h > 0.7:
        warnings.append("⚠️ Near 1H channel bottom (support)")

    # Check 4H channel
    position_4h = features_dict.get('tsla_channel_4h_position', 0.5)
    r_squared_4h = features_dict.get('tsla_channel_4h_r_squared', 0.0)

    if position_4h > 0.9 and r_squared_4h > 0.7:
        warnings.append("⚠️ Near 4H channel resistance - may break through")

    # Check event proximity
    days_to_earnings = features_dict.get('days_until_earnings', 0)
    if abs(days_to_earnings) <= 3:
        direction = "before" if days_to_earnings < 0 else "after"
        warnings.append(f"📅 Earnings {abs(days_to_earnings)} days {direction} - expect volatility")

    return warnings


def make_prediction(model, features_df):
    """Make prediction and get layer breakdown"""

    # Get sequence (200 bars)
    sequence_length = 200
    if len(features_df) < sequence_length:
        return None, f"Need {sequence_length} bars, have {len(features_df)}"

    sequence = features_df.tail(sequence_length).values
    x_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

    # Current price
    current_price = float(features_df.iloc[-1]['tsla_close'])

    # Predict
    with torch.no_grad():
        result = model.predict(x_tensor)

    # Format results
    prediction = {
        'timestamp': datetime.now(),
        'current_price': current_price,
        'fusion': {
            'predicted_high': float(result['predicted_high']),
            'predicted_low': float(result['predicted_low']),
            'confidence': float(result['confidence']),
            'layer': 'fusion'
        }
    }

    # Extract layer predictions
    layers = {}
    for layer_name in ['fast', 'medium', 'slow']:
        if f'{layer_name}_pred_high' in result:
            layers[layer_name] = {
                'predicted_high': float(result[f'{layer_name}_pred_high']),
                'predicted_low': float(result[f'{layer_name}_pred_low']),
                'confidence': float(result[f'{layer_name}_pred_conf']),
                'layer': layer_name
            }

    prediction['layers'] = layers
    prediction['fusion_weights'] = result.get('fusion_weights', [0.33, 0.33, 0.33])
    prediction['features_dict'] = features_df.iloc[-1].to_dict()

    return prediction, None


def find_best_opportunity(prediction):
    """Find layer with highest confidence"""
    opportunities = []

    # Add fusion
    opportunities.append(prediction['fusion'])

    # Add layers
    for layer_name, layer_pred in prediction['layers'].items():
        opportunities.append(layer_pred)

    # Sort by confidence (descending)
    opportunities.sort(key=lambda x: x['confidence'], reverse=True)

    return opportunities


# Main app
def main():
    st.title("🎯 AutoTrade2 - Hierarchical LNN Dashboard")
    st.caption("Multi-timeframe predictions with adaptive fusion")

    # Sidebar config
    st.sidebar.header("⚙️ Configuration")

    # Auto-refresh disabled for now (will add proper implementation)
    # auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)

    alert_threshold = st.sidebar.slider("Alert Threshold", 0.0, 1.0, 0.75, 0.05)

    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

    # Load model
    try:
        with st.spinner("Loading hierarchical model..."):
            model = load_hierarchical()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

    st.sidebar.success(f"✅ Model loaded")
    st.sidebar.caption(f"Model: Hierarchical LNN v3.10")

    # Load data and extract features
    try:
        with st.spinner("Fetching live data from yfinance..."):
            # Use HybridLiveDataFeed for live predictions
            from src.ml.live_data_feed import HybridLiveDataFeed

            feed = HybridLiveDataFeed(symbols=['TSLA', 'SPY'])
            df = feed.fetch_for_prediction()  # Gets 7 days of 1-min data

            if len(df) < 200:
                st.error(f"❌ Insufficient data: {len(df)} bars (need 200)")
                st.info("Market may be closed or data unavailable")
                st.stop()

        with st.spinner("Extracting 473 features..."):
            # Extract features
            extractor = TradingFeatureExtractor()

            # Create events handler
            try:
                events_handler = CombinedEventsHandler()
            except:
                events_handler = None

            features_df = extractor.extract_features(df, use_cache=True, events_handler=events_handler)

            current_price = float(features_df.iloc[-1]['tsla_close'])

    except Exception as e:
        st.error(f"❌ Data loading failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

    st.sidebar.info(f"💰 Current: ${current_price:.2f}")
    st.sidebar.caption(f"Data: {len(features_df):,} bars")

    # Make prediction
    with st.spinner("Running prediction..."):
        prediction, error = make_prediction(model, features_df)

    if error:
        st.error(f"❌ Prediction failed: {error}")
        st.stop()

    # Find best opportunity
    opportunities = find_best_opportunity(prediction)
    best = opportunities[0]
    alternatives = opportunities[1:]

    # Detect warnings
    warnings = detect_channel_warning(prediction['features_dict'])

    # Display BEST OPPORTUNITY (Main Card)
    st.header("🎯 BEST OPPORTUNITY")

    layer_emoji = {"fast": "⚡", "medium": "🔄", "slow": "🐢", "fusion": "🎯"}
    layer_name = best['layer'].capitalize()
    emoji = layer_emoji.get(best['layer'], "🎯")

    conf_pct = best['confidence'] * 100

    # Confidence badge
    if conf_pct >= 80:
        conf_badge = f"🟢 {conf_pct:.0f}% (High Confidence)"
    elif conf_pct >= 60:
        conf_badge = f"🟡 {conf_pct:.0f}% (Medium Confidence)"
    else:
        conf_badge = f"🔴 {conf_pct:.0f}% (Low Confidence)"

    st.subheader(f"{emoji} {layer_name} Layer - {conf_badge}")

    # Calculate target prices
    pred_high = best['predicted_high']
    pred_low = best['predicted_low']
    target_high_price = current_price * (1 + pred_high / 100)
    target_low_price = current_price * (1 + pred_low / 100)

    # Time estimate
    time_desc, _ = estimate_time_to_target(best['layer'])

    # Main prediction
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label=f"📈 Predicted High (in {time_desc})",
            value=f"${target_high_price:.2f}",
            delta=f"{pred_high:+.2f}%"
        )

    with col2:
        st.metric(
            label=f"📉 Predicted Low (in {time_desc})",
            value=f"${target_low_price:.2f}",
            delta=f"{pred_low:+.2f}%"
        )

    # Warnings
    if warnings:
        st.warning("\n\n".join(warnings))

    # Explanation
    st.info(f"💡 **Setup:** Price will likely reach **${target_high_price:.2f}** ({pred_high:+.2f}%) "
            f"in the next **{time_desc}**, with downside risk to ${target_low_price:.2f} ({pred_low:+.2f}%)")

    # Alert if high confidence
    if conf_pct >= alert_threshold * 100:
        st.success(f"🔔 HIGH CONFIDENCE ALERT (>{alert_threshold:.0%}) - Consider this trade!")

    st.markdown("---")

    # Alternative Opportunities
    if alternatives:
        with st.expander("📊 Alternative Opportunities (Other Timeframes)"):
            for opp in alternatives:
                layer_name = opp['layer'].capitalize()
                emoji = layer_emoji.get(opp['layer'], "📊")
                conf = opp['confidence'] * 100
                time_desc, _ = estimate_time_to_target(opp['layer'])

                high_price = current_price * (1 + opp['predicted_high'] / 100)
                low_price = current_price * (1 + opp['predicted_low'] / 100)

                st.markdown(f"**{emoji} {layer_name}** - {conf:.0f}% confidence")
                st.markdown(f"- Target: ${high_price:.2f} ({opp['predicted_high']:+.2f}%) in {time_desc}")
                st.markdown(f"- Risk: ${low_price:.2f} ({opp['predicted_low']:+.2f}%)")
                st.markdown("")

    # Fusion Weights (How model is weighting each layer)
    with st.expander("⚖️ Fusion Weights (Layer Trust Levels)"):
        weights = prediction['fusion_weights']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("⚡ Fast Layer", f"{weights[0]:.1%}")
        with col2:
            st.metric("🔄 Medium Layer", f"{weights[1]:.1%}")
        with col3:
            st.metric("🐢 Slow Layer", f"{weights[2]:.1%}")

        max_weight_idx = weights.index(max(weights))
        layer_names = ['Fast', 'Medium', 'Slow']
        st.caption(f"Model is currently trusting **{layer_names[max_weight_idx]}** layer most ({weights[max_weight_idx]:.1%})")

    # Channel Context
    with st.expander("📊 Channel Analysis"):
        features = prediction['features_dict']

        # 1H channel
        st.markdown("**1-Hour Channel:**")
        slope_1h = features.get('tsla_channel_1h_slope_pct', 0.0)
        pos_1h = features.get('tsla_channel_1h_position', 0.5)
        pp_1h = features.get('tsla_channel_1h_ping_pongs', 0)
        r2_1h = features.get('tsla_channel_1h_r_squared', 0.0)

        direction = "📈 Bullish" if slope_1h > 0.1 else "📉 Bearish" if slope_1h < -0.1 else "➡️ Sideways"
        st.markdown(f"- Direction: {direction} ({slope_1h:+.2f}% per bar)")
        st.markdown(f"- Position: {pos_1h:.2f} (0=bottom, 1=top)")
        st.markdown(f"- Strength: {pp_1h} ping-pongs, R²={r2_1h:.2f}")

        st.markdown("")

        # 4H channel
        st.markdown("**4-Hour Channel:**")
        slope_4h = features.get('tsla_channel_4h_slope_pct', 0.0)
        pos_4h = features.get('tsla_channel_4h_position', 0.5)

        direction = "📈 Bullish" if slope_4h > 0.1 else "📉 Bearish" if slope_4h < -0.1 else "➡️ Sideways"
        st.markdown(f"- Direction: {direction} ({slope_4h:+.2f}% per bar)")
        st.markdown(f"- Position: {pos_4h:.2f}")

    # Event Context
    with st.expander("📅 Upcoming Events"):
        days_to_earnings = features.get('days_until_earnings', 0)
        days_to_fomc = features.get('days_until_fomc', 0)
        is_event_week = features.get('is_earnings_week', 0)

        if is_event_week:
            if days_to_earnings != 0:
                when = f"{abs(int(days_to_earnings))} days {'before' if days_to_earnings < 0 else 'after'}"
                st.warning(f"📊 Earnings: {when} - Expect increased volatility")
            if days_to_fomc != 0:
                when = f"{abs(int(days_to_fomc))} days {'before' if days_to_fomc < 0 else 'after'}"
                st.info(f"🏦 FOMC: {when}")
        else:
            st.success("✅ No major events in next 14 days")

    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {prediction['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption(f"Model: Hierarchical LNN v3.10 | Features: 473 | Parameters: ~2.8M")


# Run main app (Streamlit executes script directly)
main()
