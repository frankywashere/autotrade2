#!/usr/bin/env python3
"""
AutoTrade v5.3.1 Dashboard

Comprehensive live prediction dashboard with:
- v5.2: Duration prediction (conservative/expected/aggressive)
- v5.2: Validity scores (forward-looking channel assessment)
- v5.2: Transition forecasts (Phase 2 informational)
- v5.3: Hierarchical containment analysis
- v5.3: Calibrated confidence (dual signal system)
- v5.3.1: 4-way information flow support
- Training history visualization

Usage:
    streamlit run dashboard_v531.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="AutoTrade v5.3.1",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import after page config
from predict import LivePredictor


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'predictor': None,
        'last_prediction': None,
        'prediction_history': [],
        'last_refresh': None,
        'model_loaded': False,
        'data_loaded': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(model_path: str):
    """Load the v5.3.1 model (cached)."""
    try:
        import glob

        # Find training tf_meta (with events hash)
        tf_meta_files = sorted(glob.glob('data/feature_cache/tf_meta_*.json'))
        training_tf_meta = None

        for f in tf_meta_files:
            if '_ev' in f:  # Has events hash = from training
                training_tf_meta = f
                break

        if training_tf_meta is None and tf_meta_files:
            training_tf_meta = tf_meta_files[0]

        predictor = LivePredictor(model_path, tf_meta_path=training_tf_meta, device='auto')
        return predictor, None
    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

def render_sidebar():
    """Render sidebar with controls and event timeline."""
    st.sidebar.title("🎛️ Controls")

    # Model section
    st.sidebar.subheader("Model")
    model_path = Path('models/hierarchical_lnn.pth')

    if model_path.exists():
        st.sidebar.success(f"✓ Found: {model_path.name}")

        if st.sidebar.button("🔄 Load Model", use_container_width=True):
            with st.spinner("Loading v5.3.1 model..."):
                predictor, error = load_model(str(model_path))
                if predictor:
                    st.session_state.predictor = predictor
                    st.session_state.model_loaded = True
                    st.sidebar.success("Model loaded!")
                else:
                    st.sidebar.error(f"Failed: {error}")
    else:
        st.sidebar.error("❌ No model found")
        st.sidebar.info("Train a model first")

    st.sidebar.divider()

    # Data fetching
    st.sidebar.subheader("Data Fetch")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        intraday_days = st.number_input("Intraday (days)", 1, 60, 60, help="5m/15m/30m/1h data")
    with col2:
        daily_days = st.number_input("Daily (days)", 30, 2000, 400)

    longer_days = st.sidebar.number_input("Weekly/Monthly (days)", 365, 7300, 5475, help="~15 years for w168")

    if st.sidebar.button("📡 Fetch Live Data", use_container_width=True):
        if st.session_state.predictor:
            with st.spinner("Fetching from yfinance..."):
                try:
                    st.session_state.predictor.fetch_live_data(
                        intraday_days=intraday_days,
                        daily_days=daily_days,
                        longer_days=longer_days
                    )
                    st.session_state.data_loaded = True
                    st.session_state.last_refresh = datetime.now()
                    st.sidebar.success("Data loaded!")
                except Exception as e:
                    st.sidebar.error(f"Failed: {e}")
        else:
            st.sidebar.warning("Load model first")

    st.sidebar.divider()

    # Events timeline (if available in last prediction)
    st.sidebar.subheader("📅 Events")
    if st.session_state.last_prediction and 'v52_events' in st.session_state.last_prediction:
        events = st.session_state.last_prediction['v52_events']
        if events:
            for event in events[:5]:  # Show top 5
                days = event.get('days_until', 0)
                event_type = event.get('type', 'unknown').upper()
                if days >= 0:
                    st.sidebar.text(f"{event_type}: {days} days")
                else:
                    st.sidebar.text(f"{event_type}: {abs(days)} days ago")
        else:
            st.sidebar.text("No upcoming events")
    else:
        st.sidebar.text("Make prediction to see events")

    st.sidebar.divider()

    # Info
    st.sidebar.subheader("Info")
    if st.session_state.last_refresh:
        st.sidebar.text(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

def render_header():
    """Render header with status."""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.title("🤖 AutoTrade v5.3.1")
        st.caption("Hierarchical Duration Predictor")

    with col2:
        if st.session_state.model_loaded:
            st.success("🟢 Model Ready")
        else:
            st.warning("🟡 No Model")

    with col3:
        if st.session_state.data_loaded:
            st.success("🟢 Data Ready")
        else:
            st.warning("🟡 No Data")


# ═══════════════════════════════════════════════════════════════
# MAIN PREDICTION CARD
# ═══════════════════════════════════════════════════════════════

def render_main_prediction():
    """Render primary prediction card."""
    st.subheader("📈 Current Prediction")

    if not st.session_state.predictor:
        st.info("👈 Load model from sidebar")
        return

    if not st.session_state.data_loaded:
        st.info("👈 Fetch live data from sidebar")
        return

    # Make prediction button
    if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
        with st.spinner("Running v5.3.1 inference..."):
            try:
                result = st.session_state.predictor.predict()
                st.session_state.last_prediction = result

                # Add to history
                st.session_state.prediction_history.append({
                    'timestamp': result['timestamp'],
                    'high': result['predicted_high'],
                    'low': result['predicted_low'],
                    'selected_tf': result.get('selected_tf', 'unknown')
                })

                # Keep last 100
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
        selected_tf = pred.get('selected_tf', 'unknown')

        # Get validity and confidence
        validity = None
        if 'v52_validity' in pred and selected_tf in pred['v52_validity']:
            validity = pred['v52_validity'][selected_tf]

        confidence = pred.get('confidence', 0)

        # Main prediction card
        st.markdown(f"### 📊 SELECTED: **{selected_tf.upper()}**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("High", f"{pred['predicted_high']:.2f}%")

        with col2:
            st.metric("Low", f"{pred['predicted_low']:.2f}%")

        with col3:
            if validity is not None:
                st.metric("Validity", f"{validity:.0%}", help="Will channel hold?")
            else:
                st.metric("Validity", "N/A")

        with col4:
            if confidence:
                st.metric("Confidence", f"{confidence:.0%}", help="Prediction accuracy (calibrated)")
            else:
                st.metric("Confidence", "N/A")

        # Duration display (if available)
        if 'v52_duration' in pred and selected_tf in pred['v52_duration']:
            dur = pred['v52_duration'][selected_tf]

            st.markdown("#### Duration Forecast")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Conservative", f"{dur['conservative']:.0f} bars",
                         delta=f"{dur['conservative'] - dur['expected']:.0f}")

            with col2:
                st.metric("Expected", f"{dur['expected']:.0f} bars")

            with col3:
                st.metric("Aggressive", f"{dur['aggressive']:.0f} bars",
                         delta=f"{dur['aggressive'] - dur['expected']:.0f}")

            # Duration confidence
            dur_conf = dur.get('confidence', 0.5)
            st.progress(dur_conf, text=f"Duration Confidence: {dur_conf:.0%}")

        # Agreement indicator
        if validity is not None and confidence is not None:
            agreement_diff = abs(validity - confidence)
            if agreement_diff < 0.15:
                st.success("✓ Validity and Confidence AGREE - Strong signal")
            elif agreement_diff < 0.30:
                st.info("○ Validity and Confidence MODERATE agreement")
            else:
                st.warning("⚠️ Validity and Confidence DIVERGE - Exercise caution")

                # Explain divergence
                if validity > confidence + 0.15:
                    st.write("Channel is strong but prediction uncertain")
                elif confidence > validity + 0.15:
                    st.write("Prediction confident but channel may break")


# ═══════════════════════════════════════════════════════════════
# ALL CHANNELS TABLE
# ═══════════════════════════════════════════════════════════════

def render_all_channels():
    """Render all 11 timeframe predictions."""
    st.subheader("📊 All Timeframes")

    if not st.session_state.last_prediction:
        st.info("Make a prediction to see all channels")
        return

    pred = st.session_state.last_prediction

    # Build table data
    table_data = []

    all_channels = pred.get('all_channels', [])
    validities = pred.get('v52_validity', {})
    durations = pred.get('v52_duration', {})

    for channel in all_channels:
        tf = channel['timeframe']
        high = channel['high']
        low = channel['low']
        conf = channel['confidence']

        # Get validity
        validity = validities.get(tf, 0.5)

        # Get duration
        dur_data = durations.get(tf, {})
        dur_expected = dur_data.get('expected', 0)
        dur_conf = dur_data.get('confidence', 0.5)

        # Agreement
        agreement = "✓" if abs(validity - conf) < 0.15 else "⚠️"

        # Is this selected?
        is_selected = (tf == pred.get('selected_tf'))

        table_data.append({
            '': '⭐' if is_selected else '',
            'TF': tf.upper(),
            'High %': f"{high:.2f}",
            'Low %': f"{low:.2f}",
            'Validity': f"{validity:.2f}",
            'Confidence': f"{conf:.2f}",
            'Duration': f"{dur_expected:.0f}±{dur_expected*(1-dur_conf):.0f}" if dur_expected > 0 else "N/A",
            'Agreement': agreement
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Legend
    with st.expander("ℹ️ Column Explanations"):
        st.write("""
        - **Validity**: Forward-looking assessment (will channel hold?)
        - **Confidence**: Calibrated prediction accuracy (historical performance)
        - **Duration**: Expected bars ± uncertainty
        - **Agreement**: ✓ if validity and confidence agree (<15% diff)
        """)


# ═══════════════════════════════════════════════════════════════
# PHASE 2 FORECAST (EXPANDABLE)
# ═══════════════════════════════════════════════════════════════

def render_phase2_forecast():
    """Render Phase 2 transition forecast (informational)."""
    if not st.session_state.last_prediction:
        st.info("Make a prediction first")
        return

    pred = st.session_state.last_prediction

    if 'v52_compositor' not in pred:
        st.info("No compositor data available")
        return

    compositor = pred['v52_compositor']

    st.markdown("**After current channel ends:**")
    st.caption("⚠️ Informational only - NOT included in final prediction")

    # Transition probabilities
    st.markdown("#### Transition Type")
    trans = compositor['transition']

    fig = go.Figure(data=[
        go.Bar(
            x=['Continue', 'Switch TF', 'Reverse', 'Sideways'],
            y=[trans['continue'], trans['switch_tf'], trans['reverse'], trans['sideways']],
            marker_color=['green', 'blue', 'orange', 'gray']
        )
    ])
    fig.update_layout(
        yaxis_title="Probability",
        height=300,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Direction forecast
    st.markdown("#### Phase 2 Direction")
    direction = compositor['direction']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Bull", f"{direction['bull']:.0%}")
    with col2:
        st.metric("Bear", f"{direction['bear']:.0%}")
    with col3:
        st.metric("Sideways", f"{direction['sideways']:.0%}")

    # Phase 2 slope
    phase2_slope = compositor.get('phase2_slope', 0)
    st.metric("Phase 2 Slope", f"{phase2_slope:.3f}", help="Predicted new channel slope")

    # Most likely scenario
    likely_trans = max(trans.keys(), key=lambda k: trans[k])
    likely_dir = max(direction.keys(), key=lambda k: direction[k])

    st.info(f"**Most Likely**: {likely_trans.upper()} → {likely_dir.upper()} direction")


# ═══════════════════════════════════════════════════════════════
# HIERARCHICAL CONTAINMENT (EXPANDABLE)
# ═══════════════════════════════════════════════════════════════

def render_containment():
    """Render hierarchical containment analysis."""
    if not st.session_state.last_prediction:
        st.info("Make a prediction first")
        return

    pred = st.session_state.last_prediction

    if 'v53_containment' not in pred:
        st.info("No containment data available")
        return

    containment = pred['v53_containment']
    selected_tf = pred.get('selected_tf', 'unknown')

    st.markdown(f"**{selected_tf.upper()} fits within parent timeframes:**")

    if not containment:
        st.info(f"{selected_tf} has no larger parent TFs to check")
        return

    # Table of parent TFs
    table_data = []
    for parent_tf, result in containment.items():
        table_data.append({
            'Parent TF': parent_tf.upper(),
            'Contained': f"{result['containment_score']:.0%}",
            'Violation High': f"{result['violation_high']:.2f}%",
            'Violation Low': f"{result['violation_low']:.2f}%",
            'Parent Validity': f"{result.get('parent_validity', 0.5):.2f}",
            'Fits': "✓" if result['fits'] else "✗"
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Interpretation
    all_fit = all(r['fits'] for r in containment.values())
    if all_fit:
        st.success("✓ Prediction fits within all parent channels")
    else:
        violating = [tf for tf, r in containment.items() if not r['fits']]
        st.warning(f"⚠️ May violate: {', '.join(violating)}")


# ═══════════════════════════════════════════════════════════════
# TRAINING HISTORY (EXPANDABLE)
# ═══════════════════════════════════════════════════════════════

def render_training_history():
    """Render training history visualization."""
    history_path = Path('models/hierarchical_training_history.json')

    if not history_path.exists():
        st.info("No training history available")
        st.caption("Train the model to generate history")
        return

    try:
        with open(history_path) as f:
            history = json.load(f)
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return

    total_epochs = history.get('total_epochs', 0)
    best_val = history.get('best_val_loss', 0)

    st.markdown(f"**Training Summary**: {total_epochs} epochs, Best Val Loss: {best_val:.4f}")

    # Loss evolution chart
    if 'loss_components' in history:
        st.markdown("#### Loss Component Evolution")

        components = history['loss_components']
        epochs = list(range(1, len(components['primary']) + 1))

        fig = go.Figure()

        # Add each component
        for comp_name, values in components.items():
            if len(values) > 0:
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=values,
                    name=comp_name,
                    mode='lines+markers'
                ))

        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis_type="log",  # Log scale for wide range
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Validation loss
        if 'val_losses' in history and len(history['val_losses']) > 0:
            st.markdown("#### Validation Performance")

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=epochs,
                y=history['val_losses'],
                name='Val Loss',
                mode='lines+markers'
            ))

            if 'val_errors' in history:
                fig2.add_trace(go.Scatter(
                    x=epochs,
                    y=history['val_errors'],
                    name='Val MAE (%)',
                    mode='lines+markers',
                    yaxis='y2'
                ))

                fig2.update_layout(
                    yaxis_title="Val Loss",
                    yaxis2=dict(title="Val MAE (%)", overlaying='y', side='right'),
                    height=400
                )

            st.plotly_chart(fig2, use_container_width=True)

    else:
        # Fallback: basic chart
        st.line_chart({
            'Train Loss': history.get('train_losses', []),
            'Val Loss': history.get('val_losses', [])
        })


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    """Main dashboard entry point."""
    init_session_state()

    # Sidebar
    render_sidebar()

    # Header
    render_header()

    st.divider()

    # Main prediction
    render_main_prediction()

    st.divider()

    # All channels
    render_all_channels()

    st.divider()

    # Expandable diagnostics
    with st.expander("📊 Phase 2 Forecast (Informational Only)"):
        render_phase2_forecast()

    with st.expander("🔗 Hierarchical Containment"):
        render_containment()

    with st.expander("📈 Training History"):
        render_training_history()


if __name__ == "__main__":
    main()
