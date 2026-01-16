"""
V15 Streamlit Dashboard

Run with: streamlit run v15/dashboard.py

Features:
- Live predictions with current market data
- Feature importance visualization
- Model performance metrics
- Correlation analysis display
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from v15.config import TIMEFRAMES, STANDARD_WINDOWS, TOTAL_FEATURES
from v15.types import ChannelSample

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="V15 Channel Predictor",
    page_icon="📈",
    layout="wide"
)


@st.cache_data(ttl=300)
def load_market_data(data_dir: str):
    """Load and cache market data."""
    from v15.data import load_market_data as _load
    return _load(data_dir)


@st.cache_resource
def load_predictor(checkpoint_path: str):
    """Load and cache predictor."""
    from v15.inference import Predictor
    return Predictor.load(checkpoint_path)


def show_prediction_card(prediction):
    """Display prediction in a card format."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Duration",
            f"{prediction.duration_mean:.0f} bars",
            f"±{prediction.duration_std:.0f}"
        )

    with col2:
        direction_color = "green" if prediction.direction == 'up' else "red"
        st.metric(
            "Direction",
            prediction.direction.upper(),
            f"{prediction.direction_prob:.1%}"
        )

    with col3:
        st.metric(
            "New Channel",
            prediction.new_channel.title(),
            f"{prediction.new_channel_probs[prediction.new_channel]:.1%}"
        )

    with col4:
        st.metric(
            "Confidence",
            f"{prediction.confidence:.1%}",
            f"Window: {prediction.best_window}"
        )


def show_feature_importance(predictor, top_k: int = 50):
    """Show feature importance from model weights."""
    if predictor.model.feature_weights is None:
        st.warning("Model doesn't have explicit feature weights")
        return

    importance = predictor.model.feature_weights.get_feature_importance()
    importance = importance.cpu().numpy()

    # Get top features
    top_indices = np.argsort(importance)[-top_k:][::-1]
    top_names = [predictor.feature_names[i] for i in top_indices]
    top_values = importance[top_indices]

    # Create chart
    fig = px.bar(
        x=top_values,
        y=top_names,
        orientation='h',
        title=f"Top {top_k} Feature Importance",
        labels={'x': 'Importance', 'y': 'Feature'}
    )
    fig.update_layout(height=800, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


def show_tf_attention(prediction_output: Dict[str, Any]):
    """Show timeframe attention weights."""
    if 'aggregation_weights' not in prediction_output:
        return

    weights = prediction_output['aggregation_weights']
    if weights is None:
        return

    weights = weights.squeeze().cpu().numpy()

    fig = px.bar(
        x=TIMEFRAMES,
        y=weights,
        title="Timeframe Attention Weights",
        labels={'x': 'Timeframe', 'y': 'Weight'}
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📈 V15 Channel Break Predictor")

    # Sidebar config
    st.sidebar.header("Configuration")

    data_dir = st.sidebar.text_input("Data Directory", value="data")
    checkpoint_path = st.sidebar.text_input(
        "Model Checkpoint",
        value="checkpoints/best.pt"
    )

    # Check if paths exist
    if not Path(data_dir).exists():
        st.error(f"Data directory not found: {data_dir}")
        st.info("Please provide path to directory containing TSLA_1min.csv, SPY_1min.csv, VIX_History.csv")
        return

    # Load data
    with st.spinner("Loading market data..."):
        try:
            tsla, spy, vix = load_market_data(data_dir)
            st.sidebar.success(f"Loaded {len(tsla):,} bars")
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

    # Load model if available
    predictor = None
    if Path(checkpoint_path).exists():
        with st.spinner("Loading model..."):
            try:
                predictor = load_predictor(checkpoint_path)
                st.sidebar.success("Model loaded")
            except Exception as e:
                st.sidebar.warning(f"Model not loaded: {e}")
    else:
        st.sidebar.info("No model checkpoint found. Train a model first.")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Live Prediction",
        "Feature Analysis",
        "Model Info",
        "Data Explorer"
    ])

    with tab1:
        st.header("Live Prediction")

        if predictor is None:
            st.warning("Load a trained model to make predictions")
        else:
            # Select time range
            col1, col2 = st.columns(2)
            with col1:
                lookback = st.slider(
                    "Lookback bars",
                    min_value=1000,
                    max_value=50000,
                    value=35000
                )

            if st.button("Make Prediction", type="primary"):
                with st.spinner("Extracting features and predicting..."):
                    try:
                        # Use recent data
                        tsla_slice = tsla.iloc[-lookback:]
                        spy_slice = spy.iloc[-lookback:]
                        vix_slice = vix.iloc[-lookback:]

                        prediction = predictor.predict(
                            tsla_slice, spy_slice, vix_slice
                        )

                        st.success("Prediction complete!")
                        show_prediction_card(prediction)

                        # Show details
                        with st.expander("Prediction Details"):
                            st.json({
                                'timestamp': str(prediction.timestamp),
                                'duration_mean': prediction.duration_mean,
                                'duration_std': prediction.duration_std,
                                'direction': prediction.direction,
                                'direction_prob': prediction.direction_prob,
                                'new_channel': prediction.new_channel,
                                'new_channel_probs': prediction.new_channel_probs,
                                'confidence': prediction.confidence,
                                'best_window': prediction.best_window,
                            })

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        logger.exception("Prediction error")

    with tab2:
        st.header("Feature Analysis")

        if predictor is not None:
            top_k = st.slider("Top K Features", 10, 100, 50)
            show_feature_importance(predictor, top_k)
        else:
            st.info("Load a model to see feature importance")

    with tab3:
        st.header("Model Information")

        if predictor is not None:
            model = predictor.model

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
                st.metric("Input Dimension", model.input_dim)
                st.metric("Timeframes", model.n_timeframes)

            with col2:
                st.metric("Features per TF", model.features_per_tf)
                st.metric("Has Explicit Weights", "Yes" if model.feature_weights else "No")
                st.metric("Device", str(predictor.device))
        else:
            st.info("Load a model to see information")

    with tab4:
        st.header("Data Explorer")

        st.subheader("TSLA")
        st.line_chart(tsla['close'].tail(1000))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Bars", f"{len(tsla):,}")
            st.metric("Start", str(tsla.index[0].date()))
        with col2:
            st.metric("End", str(tsla.index[-1].date()))
            st.metric("Last Close", f"${tsla['close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
