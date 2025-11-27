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
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.hierarchical_model import load_hierarchical_model
from src.ml.features import TradingFeatureExtractor
from src.ml.data_feed import CSVDataFeed
from src.ml.events import CombinedEventsHandler
from src.linear_regression import LinearRegressionChannel
import config

# Layer mapping for adaptive projection display
layer_map = {
    0: "Fast (intraday ripples: hours)",
    1: "Medium (swings: days)",
    2: "Slow (tides: weeks+)"
}


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


def create_channel_chart(price_df, features_dict, timeframe='1h', lookback=168):
    """
    Create channel visualization showing price + upper/lower bounds.

    RECALCULATES channel using same method as feature extraction for accuracy.

    Args:
        price_df: DataFrame with tsla_close column
        features_dict: Feature dictionary with channel metrics (for display only)
        timeframe: Which timeframe channel to display
        lookback: Number of bars to show

    Returns:
        plotly figure
    """
    # Get channel metrics from features (for display)
    r_squared = features_dict.get(f'tsla_channel_{timeframe}_r_squared', 0.0)
    ping_pongs = features_dict.get(f'tsla_channel_{timeframe}_ping_pongs', 0)
    slope_pct = features_dict.get(f'tsla_channel_{timeframe}_slope_pct', 0.0)

    # Resample price data to target timeframe
    timeframe_rules = {
        '5min': '5min', '15min': '15min', '30min': '30min',
        '1h': '1h', '2h': '2h', '3h': '3h', '4h': '4h',
        'daily': '1D', 'weekly': '1W'
    }

    tf_rule = timeframe_rules.get(timeframe, '1h')

    # Prepare data for resampling
    tsla_df = price_df[['tsla_open', 'tsla_high', 'tsla_low', 'tsla_close', 'tsla_volume']].copy()
    tsla_df.columns = ['open', 'high', 'low', 'close', 'volume']

    # Resample to target timeframe
    resampled = tsla_df.resample(tf_rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Use last N bars
    window = resampled.tail(lookback)

    # RECALCULATE channel using same method as feature extraction
    channel_calc = LinearRegressionChannel()
    channel = channel_calc.calculate_channel(window, min(lookback, len(window)), timeframe)

    # Get actual channel lines
    upper_line = channel.upper_line
    lower_line = channel.lower_line
    center_line = channel.center_line
    dates = window.index

    # Get prices from window
    prices = window['close'].values

    # Create figure
    fig = go.Figure()

    # Add price line
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='TSLA Price',
        line=dict(color='blue', width=2)
    ))

    # Add regression line (center)
    fig.add_trace(go.Scatter(
        x=dates,
        y=center_line,
        mode='lines',
        name='Regression Line',
        line=dict(color='yellow', width=1, dash='dot')
    ))

    # Add upper bound
    fig.add_trace(go.Scatter(
        x=dates,
        y=upper_line,
        mode='lines',
        name='Upper Bound (+2σ)',
        line=dict(color='green', width=1)
    ))

    # Add lower bound
    fig.add_trace(go.Scatter(
        x=dates,
        y=lower_line,
        mode='lines',
        name='Lower Bound (-2σ)',
        line=dict(color='red', width=1)
    ))

    # Mark current position
    fig.add_trace(go.Scatter(
        x=[dates[-1]],
        y=[prices[-1]],
        mode='markers',
        name='Current Price',
        marker=dict(color='orange', size=12, symbol='star')
    ))

    # Layout
    direction = "📈 Bullish" if slope_pct > 0.1 else "📉 Bearish" if slope_pct < -0.1 else "➡️ Sideways"

    fig.update_layout(
        title=f"{timeframe.upper()} Channel - {direction} (R²={r_squared:.2f}, {ping_pongs} ping-pongs)",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=400,
        showlegend=True
    )

    return fig


def make_prediction(model, features_df):
    """Make prediction and get layer breakdown"""

    # Get sequence (200 bars)
    sequence_length = 200
    if len(features_df) < sequence_length:
        return None, f"Need {sequence_length} bars, have {len(features_df)}"

    # Check feature count compatibility
    expected_features = model.input_size
    actual_features = features_df.shape[1]

    if expected_features != actual_features:
        return None, (f"Model expects {expected_features} features, but system extracts {actual_features}. "
                     f"Please retrain model with: python train_hierarchical.py --interactive")

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

    # Validate model outputs (untrained models may output nonsense)
    if pred_high < pred_low:
        st.error(f"⚠️ MODEL OUTPUT INVALID: High ({pred_high:+.2f}%) < Low ({pred_low:+.2f}%)")
        st.warning(f"🔧 Model needs more training! Current: 1 epoch. Recommended: 100 epochs.")
        st.caption("With only 1 epoch, model hasn't learned constraints. Retrain with more epochs.")

    # Use adaptive projection if available, otherwise fallback to traditional
    if 'multi_task' in prediction and 'price_change_pct' in prediction['multi_task']:
        # Adaptive projection mode
        adaptive_data = prediction['multi_task']
        price_change_pct = adaptive_data['price_change_pct'].item()
        horizon_bars = adaptive_data['horizon_bars'].item()
        adaptive_confidence = adaptive_data['adaptive_confidence'].item()
        # Decode dominant layer index to string (0=fast, 1=medium, 2=slow)
        dominant_layer_idx = adaptive_data.get('dominant_layer_idx', None)
        if dominant_layer_idx is not None:
            layer_names = {0: 'fast', 1: 'medium', 2: 'slow'}
            idx = dominant_layer_idx.item() if hasattr(dominant_layer_idx, 'item') else dominant_layer_idx
            dominant_layer = layer_names.get(idx, best['layer'])
        else:
            dominant_layer = best['layer']

        # Calculate target prices using adaptive projection
        target_price = current_price * (1 + price_change_pct / 100)

        # Convert horizon to readable format
        horizon_hours = horizon_bars / 12  # 1-min bars to hours
        if horizon_hours < 24:
            time_desc = f"{horizon_hours:.1f} hours"
        else:
            time_desc = f"{horizon_hours/24:.1f} days"

        # Main prediction - Adaptive
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label=f"🎯 Adaptive Projection ({dominant_layer.upper()})",
                value=f"${target_price:.2f}",
                delta=f"{price_change_pct:+.2f}%"
            )

        with col2:
            st.metric(
                label=f"⏱️ Time Horizon",
                value=time_desc,
                delta=f"{adaptive_confidence:.0%} confidence"
            )

        # Adaptive explanation
        direction = "📈 UPWARD" if price_change_pct > 0 else "📉 DOWNWARD"
        st.info(f"💡 **Adaptive Projection:** Price expected to move **{direction}** by **{abs(price_change_pct):.2f}%** "
                f"within **{time_desc}** ({adaptive_confidence:.0%} confidence). "
                f"Dominant layer: **{dominant_layer.upper()}**")

    else:
        # Fallback to traditional prediction
        target_high_price = current_price * (1 + pred_high / 100)
        target_low_price = current_price * (1 + pred_low / 100)

        # Time estimate
        time_desc, _ = estimate_time_to_target(best['layer'])

        # Main prediction - Traditional
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

        # Traditional explanation
        st.info(f"💡 **Setup:** Price will likely reach **${target_high_price:.2f}** ({pred_high:+.2f}%) "
                f"in the next **{time_desc}**, with downside risk to ${target_low_price:.2f} ({pred_low:+.2f}%)")

    # Warnings
    if warnings:
        st.warning("\n\n".join(warnings))

    # Alert if high confidence
    if conf_pct >= alert_threshold * 100:
        st.success(f"🔔 HIGH CONFIDENCE ALERT (>{alert_threshold:.0%}) - Consider this trade!")

    st.markdown("---")

    # Multi-Timeframe Channel Visualization
    st.subheader("📈 Channel Analysis - All Timeframes")
    st.caption("Visual proof of channel detection across multiple timeframes")

    # Find best VALID channel (R² AND ping-pongs >= 3)
    timeframes_to_check = ['5min', '15min', '1h', '4h', 'daily', 'weekly', 'monthly']
    channel_quality = []

    features = prediction['features_dict']
    for tf in timeframes_to_check:
        r2 = features.get(f'tsla_channel_{tf}_r_squared', 0.0)
        pp = features.get(f'tsla_channel_{tf}_ping_pongs', 0)

        # Only consider channels with at least 3 ping-pongs (real channels bounce!)
        if pp >= 3:
            channel_quality.append((tf, r2, pp))

    if channel_quality:
        # Sort by R² (descending)
        channel_quality.sort(key=lambda x: x[1], reverse=True)
        best_tf, best_r2, best_pp = channel_quality[0]

        # Show best channel first
        if best_r2 > 0.7:
            st.success(f"🎯 BEST CHANNEL: {best_tf.upper()} (R²={best_r2:.2f}, {best_pp} ping-pongs) ✓")
        elif best_r2 > 0.4:
            st.info(f"📊 BEST CHANNEL: {best_tf.upper()} (R²={best_r2:.2f}, {best_pp} ping-pongs)")
        else:
            st.warning(f"⚠️ BEST CHANNEL: {best_tf.upper()} (R²={best_r2:.2f}, {best_pp} ping-pongs) - Weak")
    else:
        # NO valid channels found
        st.error("❌ NO VALID CHANNELS - All timeframes have <3 ping-pongs")
        st.caption("Price may be in breakout/breakdown mode or trending without bounces")
        best_tf = '1h'  # Default fallback

    # Create tabs for different timeframes
    tabs = st.tabs(["🏆 Best", "⚡ 5min", "📊 15min", "🔄 1H", "📈 4H", "📅 Daily", "📆 Weekly", "📊 Monthly"])

    timeframes_display = [best_tf, '5min', '15min', '1h', '4h', 'daily', 'weekly', 'monthly']

    for tab_idx, (tab, tf) in enumerate(zip(tabs, timeframes_display)):
        with tab:
            try:
                # Show channel for this timeframe
                r2 = features.get(f'tsla_channel_{tf}_r_squared', 0.0)
                position = features.get(f'tsla_channel_{tf}_position', 0.5)
                slope_pct = features.get(f'tsla_channel_{tf}_slope_pct', 0.0)
                ping_pongs = features.get(f'tsla_channel_{tf}_ping_pongs', 0)

                # Determine lookback (how many bars to show)
                lookback_map = {
                    '5min': 100,
                    '15min': 96,   # ~24 hours
                    '1h': 168,     # ~1 week
                    '4h': 180,     # ~1 month
                    'daily': 90,   # ~3 months
                    'weekly': 52,  # ~1 year
                    'monthly': 24  # ~2 years
                }
                lookback = lookback_map.get(tf, 168)

                # Create chart
                chart = create_channel_chart(df, features, timeframe=tf, lookback=min(lookback, len(df)))
                st.plotly_chart(chart, use_container_width=True, key=f"channel_chart_{tab_idx}_{tf}")

                # Quality assessment
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if r2 > 0.7:
                        st.metric("Channel Quality", "🟢 Strong", f"R²={r2:.2f}")
                    elif r2 > 0.4:
                        st.metric("Channel Quality", "🟡 Moderate", f"R²={r2:.2f}")
                    else:
                        st.metric("Channel Quality", "🔴 Weak", f"R²={r2:.2f}")

                with col2:
                    direction = "📈 Bull" if slope_pct > 0.1 else "📉 Bear" if slope_pct < -0.1 else "➡️ Side"
                    st.metric("Direction", direction, f"{slope_pct:+.2f}%/bar")

                with col3:
                    pos_label = "Top" if position > 0.7 else "Bottom" if position < 0.3 else "Middle"
                    st.metric("Position", pos_label, f"{position:.2f}")

                with col4:
                    st.metric("Ping-Pongs", f"{ping_pongs}", "bounces")

                # Interpretation with ping-pong validation
                if ping_pongs < 3:
                    st.error(f"❌ NOT A VALID CHANNEL - Only {ping_pongs} ping-pongs (need 3+)")
                    st.caption("Real channels require multiple bounces. This is just a trend line.")
                elif r2 > 0.7 and ping_pongs >= 5:
                    st.success(f"✅ Excellent {tf} channel - High confidence for trading")
                elif r2 > 0.5 and ping_pongs >= 3:
                    st.info(f"📊 Decent {tf} channel - Use with confirmation")
                elif ping_pongs >= 3:
                    st.warning(f"⚠️ Weak {tf} channel - Has {ping_pongs} bounces but low R²")
                else:
                    st.warning(f"⚠️ Insufficient data for {tf} channel")

            except Exception as e:
                st.error(f"Could not create {tf} chart: {e}")

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

    # Layer Interplay & Confidence Debate
    with st.expander("🧠 Layer Interplay & Confidence Debate"):
        # Get adaptive projection data if available
        multi_task = prediction.get('multi_task', {})
        # Decode dominant layer index (0=fast, 1=medium, 2=slow)
        dominant_layer_idx = multi_task.get('dominant_layer_idx', None)
        if dominant_layer_idx is not None:
            dominant_layer = dominant_layer_idx.item() if hasattr(dominant_layer_idx, 'item') else dominant_layer_idx
        else:
            dominant_layer = 1  # Default to medium
        weights = prediction.get('fusion_weights', [0.33, 0.33, 0.33])

        # Adaptive projection data
        change_pct = multi_task.get('price_change_pct', 0.0)
        horizon_bars = multi_task.get('horizon_bars', 576.0)  # Default ~1 day
        adaptive_conf = multi_task.get('adaptive_confidence', 0.8)

        # Convert horizon to readable format
        horizon_days = horizon_bars / 1440  # 1-min bars to days (24h=1440 bars)
        if horizon_days < 1:
            time_desc = f"{horizon_days*24:.1f} hours"
        else:
            time_desc = f"{horizon_days:.1f} days"

        # Dominant layer display
        dominant_name = layer_map.get(dominant_layer, "Medium (swings: days)")
        st.markdown(f"**Dominant Layer: {dominant_name}** (Overall Confidence: {adaptive_conf:.0%})")

        # Fusion weights with explanations
        st.markdown("**Fusion Weights (how much each layer influences the prediction):**")
        st.markdown(f"- **Fast**: {weights[0]*100:.1f}% (Short-term volatility check—e.g., high 1h RSI warns of quick breaks)")
        st.markdown(f"- **Medium**: {weights[1]*100:.1f}% (Swing confirmation—e.g., 4h channel alignment boosts hold time)")
        st.markdown(f"- **Slow**: {weights[2]*100:.1f}% (Macro tide support—e.g., weekly RSI low allows multi-day rides)")

        # Adaptive projection display
        if abs(change_pct) > 0.1:  # Only show if meaningful projection
            direction = "📈 UPWARD" if change_pct > 0 else "📉 DOWNWARD"
            st.markdown(f"**Projected Move:** {direction} by **{abs(change_pct):.2f}%** within **{time_desc}**")
            st.caption("Horizon adapts based on layer consensus—short if RSI risks detected, long with multi-frame confirmation.")
        else:
            st.caption("No significant directional bias detected - layers in disagreement.")

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
                days = int(days_to_earnings)
                if days == 0:
                    when = "TODAY (Earnings Day!)"
                else:
                    when = f"{abs(days)} days {'before' if days < 0 else 'after'}"
                st.warning(f"📊 Earnings: {when} - Expect increased volatility")

            if days_to_fomc != 0:
                days = int(days_to_fomc)
                if days == 0:
                    when = "TODAY (FOMC Meeting!)"
                else:
                    when = f"{abs(days)} days {'before' if days < 0 else 'after'}"
                st.info(f"🏦 FOMC: {when}")
        else:
            st.success("✅ No major events in next 14 days")

    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {prediction['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption(f"Model: Hierarchical LNN v3.10 | Features: 473 | Parameters: ~2.8M")


# Run main app (Streamlit executes script directly)
main()
