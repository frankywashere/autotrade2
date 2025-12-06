#!/usr/bin/env python3
"""
Hierarchical LNN Dashboard v5.0

Live prediction dashboard for the 11-timeframe hierarchical model.

Usage:
    streamlit run dashboard.py

Features:
- Live predictions (predicted high/low)
- Auto-fetch data from yfinance (SPY, TSLA, VIX)
- Data buffer status for all 11 timeframes
- VIX regime display
- Auto-refresh capability
- Prediction history tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Hierarchical LNN v5.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import after page config
from predict import LivePredictor, fetch_live_vix


# ═══════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize session state variables."""
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(model_path: str):
    """Load the hierarchical model (cached)."""
    try:
        # Find the TRAINING tf_meta (not live-generated ones)
        # Training tf_meta has the full training date range and largest row count
        import glob
        tf_meta_files = sorted(glob.glob('data/feature_cache/tf_meta_*.json'))

        # Prefer files with earlier start dates (training data) and with 'ev' hash (events)
        training_tf_meta = None
        for f in tf_meta_files:
            if '_ev' in f:  # Has events hash = from training
                training_tf_meta = f
                break

        # Fallback: use first one found (earliest date)
        if training_tf_meta is None and tf_meta_files:
            training_tf_meta = tf_meta_files[0]

        predictor = LivePredictor(model_path, tf_meta_path=training_tf_meta, device='auto')
        return predictor, None
    except Exception as e:
        return None, str(e)


def get_model_path():
    """Find the model file."""
    default_path = Path('models/hierarchical_lnn.pth')
    if default_path.exists():
        return str(default_path)

    # Look for any .pth file in models/
    models_dir = Path('models')
    if models_dir.exists():
        pth_files = list(models_dir.glob('*.pth'))
        if pth_files:
            return str(pth_files[0])

    return None


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

def render_sidebar():
    """Render the sidebar with controls."""
    st.sidebar.title("🎛️ Controls")

    # Model status
    st.sidebar.subheader("Model")
    model_path = get_model_path()

    if model_path:
        st.sidebar.success(f"✓ Found: {Path(model_path).name}")

        if st.sidebar.button("🔄 Load/Reload Model", use_container_width=True):
            with st.spinner("Loading model..."):
                predictor, error = load_model(model_path)
                if predictor:
                    st.session_state.predictor = predictor
                    st.session_state.model_loaded = True
                    st.sidebar.success("Model loaded!")
                else:
                    st.sidebar.error(f"Failed: {error}")
    else:
        st.sidebar.error("❌ No model found")
        st.sidebar.info("Train a model first or place .pth in models/")

    st.sidebar.divider()

    # Data controls
    st.sidebar.subheader("Data Fetch Settings")
    st.sidebar.caption("Native intervals from yfinance (no resampling)")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        intraday_days = st.number_input("Intraday (days)", 1, 60, 60, help="Days of intraday data (5m, 15m, 30m, 1h). Max 60 days from yfinance.")
    with col2:
        daily_days = st.number_input("Daily+ (days)", 30, 1000, 400, help="Days of daily/weekly/monthly data")

    if st.sidebar.button("📡 Fetch Live Data", use_container_width=True):
        if st.session_state.predictor:
            with st.spinner("Fetching SPY, TSLA, VIX..."):
                try:
                    st.session_state.predictor.fetch_live_data(
                        intraday_days=intraday_days,
                        daily_days=daily_days
                    )
                    st.session_state.data_loaded = True
                    st.session_state.last_refresh = datetime.now()
                    st.sidebar.success("Data loaded!")
                except Exception as e:
                    st.sidebar.error(f"Failed: {e}")
        else:
            st.sidebar.warning("Load model first")

    st.sidebar.divider()

    # Auto-refresh
    st.sidebar.subheader("Auto-Refresh")
    st.session_state.auto_refresh = st.sidebar.toggle(
        "Enable auto-refresh",
        value=st.session_state.auto_refresh
    )

    if st.session_state.auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Interval (minutes)", 1, 60, 5
        )
        st.sidebar.info(f"Refreshing every {refresh_interval} min")

    st.sidebar.divider()

    # Info
    st.sidebar.subheader("Info")
    if st.session_state.last_refresh:
        st.sidebar.text(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

    if st.session_state.predictor:
        vix_status = st.session_state.predictor.get_vix_status()
        if vix_status.get('loaded'):
            st.sidebar.text(f"VIX: {vix_status.get('latest_value', 0):.2f}")
            st.sidebar.text(f"VIX date: {vix_status.get('latest_date')}")


# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════

def render_header():
    """Render the header section."""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.title("📈 Hierarchical LNN v5.0")
        st.caption("11-Timeframe Prediction System")

    with col2:
        if st.session_state.model_loaded:
            st.success("🟢 Model Ready")
        else:
            st.warning("🟡 Model Not Loaded")

    with col3:
        if st.session_state.data_loaded:
            st.success("🟢 Data Ready")
        else:
            st.warning("🟡 No Data")


def render_prediction():
    """Render the main prediction display."""
    st.subheader("🎯 Current Prediction")

    if not st.session_state.predictor:
        st.info("👈 Load model from sidebar")
        return

    if not st.session_state.data_loaded:
        st.info("👈 Fetch live data from sidebar")
        return

    # Make prediction button
    if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
        with st.spinner("Running inference..."):
            try:
                result = st.session_state.predictor.predict()
                st.session_state.last_prediction = result

                # Add to history
                st.session_state.prediction_history.append({
                    'timestamp': result['timestamp'],
                    'predicted_high': result['predicted_high'],
                    'predicted_low': result['predicted_low']
                })

                # Keep last 100 predictions
                if len(st.session_state.prediction_history) > 100:
                    st.session_state.prediction_history = st.session_state.prediction_history[-100:]

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Display prediction
    if st.session_state.last_prediction:
        pred = st.session_state.last_prediction

        # Show selected timeframe (v5.1: channel selection)
        selected_tf = pred.get('selected_tf', 'unknown')
        confidence = pred.get('confidence')

        st.markdown(f"### Selected Channel: **{selected_tf}**" +
                   (f" (confidence: {confidence:.2f})" if confidence else ""))

        col1, col2, col3 = st.columns(3)

        with col1:
            high_val = pred['predicted_high']
            st.metric(
                label="📈 Predicted High",
                value=f"{high_val:.2f}%",
            )

        with col2:
            low_val = pred['predicted_low']
            st.metric(
                label="📉 Predicted Low",
                value=f"{low_val:.2f}%",
            )

        with col3:
            st.metric(
                label="🕐 Timestamp",
                value=pred['timestamp'].strftime("%H:%M:%S")
            )

        # Show all channels sorted by confidence
        if 'all_channels' in pred:
            st.divider()
            st.markdown("#### All Channel Predictions (by confidence)")

            # Create a nice table
            channels_data = []
            for ch in pred['all_channels']:
                channels_data.append({
                    'Timeframe': ch['timeframe'],
                    'High %': f"{ch['high']:.2f}%",
                    'Low %': f"{ch['low']:.2f}%",
                    'Confidence': f"{ch['confidence']:.3f}",
                    'Valid': "✓" if ch['high'] >= ch['low'] else "⚠️"
                })

            st.dataframe(
                pd.DataFrame(channels_data),
                hide_index=True,
                use_container_width=True
            )

        # Interpretation
        st.divider()

        if high_val > 0 and low_val < 0:
            st.success(f"✓ **Normal Range**: {selected_tf} channel predicts price will reach +{high_val:.2f}% high and {low_val:.2f}% low")
        elif high_val > 0 and low_val > 0:
            st.info(f"📈 **Bullish**: {selected_tf} channel predicts both high ({high_val:.2f}%) and low ({low_val:.2f}%) above current")
        elif high_val < 0 and low_val < 0:
            st.warning(f"📉 **Bearish**: {selected_tf} channel predicts both high ({high_val:.2f}%) and low ({low_val:.2f}%) below current")
        elif high_val < low_val:
            st.error(f"⚠️ **Inverted**: High ({high_val:.2f}%) < Low ({low_val:.2f}%) - model may need more training")


def render_buffer_status():
    """Render data buffer status."""
    st.subheader("📊 Data Buffer Status")

    if not st.session_state.predictor:
        st.info("Load model to see buffer status")
        return

    buffer_status = st.session_state.predictor.get_buffer_status()

    # Create columns for each timeframe group
    col1, col2, col3 = st.columns(3)

    intraday = ['1min', '5min', '15min', '30min']
    hourly = ['1hour', '2h', '3h', '4h']
    longer = ['daily', 'weekly', 'monthly']

    with col1:
        st.caption("**Intraday**")
        for interval in intraday:
            if interval in buffer_status:
                status = buffer_status[interval]
                count = status.get('count', 0)
                required = status.get('required', 1)
                pct = min(100, count / required * 100) if required > 0 else 0

                if count > 0:
                    st.progress(pct / 100, text=f"{interval}: {count:,}")
                else:
                    st.progress(0, text=f"{interval}: empty")

    with col2:
        st.caption("**Hourly**")
        for interval in hourly:
            if interval in buffer_status:
                status = buffer_status[interval]
                count = status.get('count', 0)
                required = status.get('required', 1)
                pct = min(100, count / required * 100) if required > 0 else 0

                if count > 0:
                    st.progress(pct / 100, text=f"{interval}: {count:,}")
                else:
                    st.progress(0, text=f"{interval}: empty")

    with col3:
        st.caption("**Longer TFs**")
        for interval in longer:
            if interval in buffer_status:
                status = buffer_status[interval]
                count = status.get('count', 0)
                required = status.get('required', 1)
                pct = min(100, count / required * 100) if required > 0 else 0

                if count > 0:
                    st.progress(pct / 100, text=f"{interval}: {count:,}")
                else:
                    st.progress(0, text=f"{interval}: empty")


def render_vix_status():
    """Render VIX status panel."""
    st.subheader("📊 VIX Status")

    if not st.session_state.predictor:
        st.info("Load model to see VIX status")
        return

    vix_status = st.session_state.predictor.get_vix_status()

    if not vix_status.get('loaded'):
        st.warning("VIX data not loaded")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        vix_val = vix_status.get('latest_value', 0)
        st.metric("VIX Level", f"{vix_val:.2f}")

    with col2:
        # Determine regime
        if vix_val < 15:
            regime = "🟢 Low"
        elif vix_val < 20:
            regime = "🟡 Normal"
        elif vix_val < 30:
            regime = "🟠 Elevated"
        else:
            regime = "🔴 High"
        st.metric("Regime", regime)

    with col3:
        st.metric("History Days", vix_status.get('total_days', 0))

    with col4:
        st.metric("Latest Date", str(vix_status.get('latest_date', 'N/A')))


def render_prediction_history():
    """Render prediction history chart."""
    st.subheader("📈 Prediction History")

    if not st.session_state.prediction_history:
        st.info("Make predictions to see history")
        return

    df = pd.DataFrame(st.session_state.prediction_history)

    # Simple line chart
    chart_data = df[['predicted_high', 'predicted_low']].rename(columns={
        'predicted_high': 'High',
        'predicted_low': 'Low'
    })

    st.line_chart(chart_data)


def render_timeframe_table():
    """Render sequence lengths table."""
    st.subheader("⏱️ Timeframe Configuration")

    if not st.session_state.predictor:
        return

    seq_lengths = st.session_state.predictor.sequence_lengths

    data = []
    for tf, seq_len in seq_lengths.items():
        # Calculate approximate time span
        time_spans = {
            '5min': f"{seq_len * 5 / 60:.1f} hours",
            '15min': f"{seq_len * 15 / 60:.1f} hours",
            '30min': f"{seq_len * 30 / 60:.1f} hours",
            '1h': f"{seq_len} hours",
            '2h': f"{seq_len * 2} hours",
            '3h': f"{seq_len * 3} hours",
            '4h': f"{seq_len * 4} hours",
            'daily': f"{seq_len} days",
            'weekly': f"{seq_len} weeks",
            'monthly': f"{seq_len} months",
            '3month': f"{seq_len * 3} months",
        }

        data.append({
            'Timeframe': tf,
            'Sequence Length': seq_len,
            'Time Span': time_spans.get(tf, 'N/A')
        })

    st.dataframe(
        pd.DataFrame(data),
        hide_index=True
    )


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    """Main dashboard entry point."""
    init_session_state()

    # Sidebar
    render_sidebar()

    # Main content
    render_header()

    st.divider()

    # Prediction section
    render_prediction()

    st.divider()

    # Two columns for status panels
    col1, col2 = st.columns(2)

    with col1:
        render_buffer_status()

    with col2:
        render_vix_status()

    st.divider()

    # History and config
    tab1, tab2 = st.tabs(["📈 History", "⚙️ Configuration"])

    with tab1:
        render_prediction_history()

    with tab2:
        render_timeframe_table()

    # Auto-refresh logic
    if st.session_state.auto_refresh and st.session_state.data_loaded:
        time.sleep(0.1)  # Small delay
        st.rerun()


if __name__ == "__main__":
    main()
