"""
V15 Streamlit Dashboard

Run with: streamlit run v15/dashboard.py

Features:
- Live predictions with current market data
- Feature importance visualization
- Model performance metrics
- Correlation analysis display
- Window selection analysis and comparison
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from v15.config import TIMEFRAMES, STANDARD_WINDOWS, TOTAL_FEATURES
from v15.dtypes import ChannelSample

logger = logging.getLogger(__name__)

# Import window selection strategies
try:
    from v7.core.window_strategy import (
        SelectionStrategy, get_strategy,
        BounceFirstStrategy, LabelValidityStrategy,
        BalancedScoreStrategy, QualityScoreStrategy
    )
    from v7.core.channel import detect_channels_multi_window, select_best_channel, STANDARD_WINDOWS as V7_WINDOWS
    WINDOW_STRATEGIES_AVAILABLE = True
except ImportError:
    WINDOW_STRATEGIES_AVAILABLE = False
    logger.warning("Window selection strategies not available - v7.core.window_strategy not found")


# =============================================================================
# Window Selection Analysis Functions
# =============================================================================

def analyze_windows(channels: Dict[int, Any]) -> Dict[str, Any]:
    """
    Analyze all windows and return comprehensive window selection data.

    Returns dict with:
        - validity: Dict[int, bool] - validity per window
        - scores: Dict[int, float] - quality scores per window
        - bounces: Dict[int, int] - bounce counts per window
        - r_squared: Dict[int, float] - r_squared per window
        - directions: Dict[int, str] - direction per window
        - strategy_picks: Dict[str, Tuple[int, float]] - best window per strategy
    """
    result = {
        'validity': {},
        'scores': {},
        'bounces': {},
        'r_squared': {},
        'directions': {},
        'slopes': {},
        'strategy_picks': {}
    }

    if not channels:
        return result

    # Extract per-window metrics
    for window, channel in channels.items():
        if channel is None:
            result['validity'][window] = False
            result['scores'][window] = 0.0
            result['bounces'][window] = 0
            result['r_squared'][window] = 0.0
            result['directions'][window] = 'N/A'
            result['slopes'][window] = 0.0
        else:
            is_valid = getattr(channel, 'valid', False)
            result['validity'][window] = is_valid

            if is_valid:
                bounce_count = getattr(channel, 'bounce_count', 0)
                r_squared = getattr(channel, 'r_squared', 0.0)
                result['scores'][window] = bounce_count * r_squared
                result['bounces'][window] = bounce_count
                result['r_squared'][window] = r_squared
                result['slopes'][window] = getattr(channel, 'slope', 0.0)

                direction = getattr(channel, 'direction', 1)
                dir_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
                result['directions'][window] = dir_names.get(int(direction), 'Unknown')
            else:
                result['scores'][window] = 0.0
                result['bounces'][window] = 0
                result['r_squared'][window] = 0.0
                result['directions'][window] = 'Invalid'
                result['slopes'][window] = 0.0

    # Run each selection strategy if available
    if WINDOW_STRATEGIES_AVAILABLE:
        strategies = {
            'Bounce First': BounceFirstStrategy(),
            'Quality Score': QualityScoreStrategy(),
            'Balanced (40/60)': BalancedScoreStrategy(bounce_weight=0.4, label_weight=0.6),
        }

        for name, strategy in strategies.items():
            try:
                best_window, confidence = strategy.select_window(channels)
                result['strategy_picks'][name] = (best_window, confidence)
            except Exception as e:
                result['strategy_picks'][name] = (None, 0.0)

    return result


def create_window_validity_chart(analysis: Dict[str, Any]) -> go.Figure:
    """Create a visual chart showing all 8 windows' validity and scores."""
    windows = STANDARD_WINDOWS

    # Prepare data
    validity = [analysis['validity'].get(w, False) for w in windows]
    scores = [analysis['scores'].get(w, 0.0) for w in windows]
    bounces = [analysis['bounces'].get(w, 0) for w in windows]
    r_squared = [analysis['r_squared'].get(w, 0.0) for w in windows]

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add bars for scores
    colors = ['green' if v else 'red' for v in validity]
    fig.add_trace(go.Bar(
        x=[str(w) for w in windows],
        y=scores,
        name='Quality Score',
        marker_color=colors,
        opacity=0.7,
        text=[f"{'Valid' if v else 'Invalid'}" for v in validity],
        textposition='outside'
    ))

    # Add line for bounce count
    fig.add_trace(go.Scatter(
        x=[str(w) for w in windows],
        y=bounces,
        name='Bounces',
        mode='lines+markers',
        line=dict(color='blue', width=2),
        yaxis='y2'
    ))

    # Add line for r_squared
    fig.add_trace(go.Scatter(
        x=[str(w) for w in windows],
        y=r_squared,
        name='R-squared',
        mode='lines+markers',
        line=dict(color='orange', width=2, dash='dash'),
        yaxis='y3'
    ))

    fig.update_layout(
        title='Window Analysis: Validity, Scores, and Metrics',
        xaxis_title='Window Size (bars)',
        yaxis=dict(title='Quality Score', side='left'),
        yaxis2=dict(title='Bounce Count', overlaying='y', side='right'),
        yaxis3=dict(title='R-squared', overlaying='y', side='right', position=0.95, showgrid=False),
        legend=dict(x=0.01, y=0.99),
        barmode='group',
        height=400
    )

    return fig


def create_strategy_comparison_chart(analysis: Dict[str, Any]) -> go.Figure:
    """Create chart comparing different selection strategies."""
    strategy_picks = analysis.get('strategy_picks', {})

    if not strategy_picks:
        fig = go.Figure()
        fig.add_annotation(text="No strategy data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig

    strategies = list(strategy_picks.keys())
    windows_picked = [strategy_picks[s][0] if strategy_picks[s][0] else 0 for s in strategies]
    confidences = [strategy_picks[s][1] for s in strategies]

    fig = go.Figure()

    # Bar for selected windows
    fig.add_trace(go.Bar(
        x=strategies,
        y=windows_picked,
        name='Selected Window',
        marker_color='steelblue',
        text=[f"W{w}" if w else "None" for w in windows_picked],
        textposition='outside'
    ))

    # Line for confidence
    fig.add_trace(go.Scatter(
        x=strategies,
        y=[c * 100 for c in confidences],  # Convert to percentage
        name='Confidence %',
        mode='lines+markers',
        line=dict(color='green', width=3),
        yaxis='y2'
    ))

    fig.update_layout(
        title='Strategy Comparison: Window Selection',
        xaxis_title='Selection Strategy',
        yaxis=dict(title='Selected Window Size', range=[0, 90]),
        yaxis2=dict(title='Confidence (%)', overlaying='y', side='right', range=[0, 110]),
        legend=dict(x=0.7, y=0.99),
        height=350
    )

    return fig


def show_window_selection_panel(
    analysis: Dict[str, Any],
    current_best_window: int,
    selection_mode: str = 'heuristic'
):
    """Display the main window selection information panel."""

    st.subheader("Current Window Selection")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Selected Window",
            f"{current_best_window} bars",
            help="The window size currently being used for predictions"
        )

    with col2:
        # Count valid windows
        valid_count = sum(1 for v in analysis['validity'].values() if v)
        st.metric(
            "Valid Windows",
            f"{valid_count}/8",
            help="Number of windows with valid channel detection"
        )

    with col3:
        # Get confidence from heuristic
        bounce_pick = analysis['strategy_picks'].get('Bounce First', (None, 0.0))
        confidence = bounce_pick[1] if bounce_pick[0] else 0.0
        st.metric(
            "Selection Confidence",
            f"{confidence:.0%}",
            help="Confidence in the current window selection"
        )

    with col4:
        st.metric(
            "Selection Mode",
            selection_mode.title(),
            help="Heuristic (rule-based) or Learned (model-based)"
        )


def show_window_details_table(analysis: Dict[str, Any]):
    """Show detailed table of all windows."""

    data = []
    for window in STANDARD_WINDOWS:
        row = {
            'Window': window,
            'Valid': 'Yes' if analysis['validity'].get(window, False) else 'No',
            'Bounces': analysis['bounces'].get(window, 0),
            'R-squared': f"{analysis['r_squared'].get(window, 0.0):.3f}",
            'Score': f"{analysis['scores'].get(window, 0.0):.2f}",
            'Direction': analysis['directions'].get(window, 'N/A'),
            'Slope': f"{analysis['slopes'].get(window, 0.0):.4f}"
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Style the dataframe
    def highlight_valid(row):
        if row['Valid'] == 'Yes':
            return ['background-color: #d4edda'] * len(row)
        else:
            return ['background-color: #f8d7da'] * len(row)

    styled_df = df.style.apply(highlight_valid, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def show_strategy_picks(analysis: Dict[str, Any]):
    """Show what each strategy would pick."""

    st.subheader("Strategy Comparison")

    strategy_picks = analysis.get('strategy_picks', {})

    if not strategy_picks:
        st.info("Window selection strategies not available")
        return

    cols = st.columns(len(strategy_picks))

    for i, (name, (window, confidence)) in enumerate(strategy_picks.items()):
        with cols[i]:
            if window is not None:
                st.metric(
                    name,
                    f"Window {window}",
                    f"{confidence:.0%} confident"
                )
            else:
                st.metric(
                    name,
                    "No valid window",
                    "0% confident"
                )

    # Check for disagreement
    picked_windows = [w for w, _ in strategy_picks.values() if w is not None]
    if len(set(picked_windows)) > 1:
        st.warning("Strategies disagree on best window - consider reviewing the data")
    elif len(picked_windows) > 0:
        st.success(f"All strategies agree: Window {picked_windows[0]} is best")


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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Live Prediction",
        "Window Selection",
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
        st.header("Window Selection Analysis")

        st.markdown("""
        This tab shows how window selection works across all 8 standard windows (10, 20, 30, 40, 50, 60, 70, 80 bars).
        Different selection strategies may pick different "best" windows based on channel quality metrics.
        """)

        # Window selection controls
        col1, col2 = st.columns(2)
        with col1:
            window_lookback = st.slider(
                "Analysis Lookback (bars)",
                min_value=500,
                max_value=20000,
                value=5000,
                key="window_lookback"
            )
        with col2:
            selection_mode = st.selectbox(
                "Selection Mode",
                ["Heuristic (Rule-based)", "Model-based (if trained)"],
                key="selection_mode"
            )

        if st.button("Analyze Windows", type="primary", key="analyze_windows_btn"):
            with st.spinner("Detecting channels at all window sizes..."):
                try:
                    # Get recent data slice
                    tsla_slice = tsla.iloc[-window_lookback:]

                    # Check if v7 channel detection is available
                    if WINDOW_STRATEGIES_AVAILABLE:
                        # Detect channels at all windows
                        channels = detect_channels_multi_window(tsla_slice, windows=STANDARD_WINDOWS)

                        # Select best channel using default heuristic
                        best_channel, best_window = select_best_channel(channels)
                        if best_window is None:
                            best_window = 50  # Default fallback

                        # Analyze windows
                        analysis = analyze_windows(channels)

                        # Store in session state for persistence
                        st.session_state['window_analysis'] = analysis
                        st.session_state['best_window'] = best_window
                        st.session_state['channels'] = channels

                        st.success(f"Analysis complete! Best window: {best_window} bars")
                    else:
                        st.error("Window selection strategies not available. Ensure v7.core.window_strategy is accessible.")

                except Exception as e:
                    st.error(f"Window analysis failed: {e}")
                    logger.exception("Window analysis error")

        # Display results if available
        if 'window_analysis' in st.session_state:
            analysis = st.session_state['window_analysis']
            best_window = st.session_state.get('best_window', 50)

            st.divider()

            # Show main panel
            show_window_selection_panel(
                analysis,
                best_window,
                selection_mode="heuristic" if "Heuristic" in selection_mode else "learned"
            )

            st.divider()

            # Strategy comparison
            show_strategy_picks(analysis)

            st.divider()

            # Visual charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Window Validity & Scores")
                fig = create_window_validity_chart(analysis)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Strategy Selection Comparison")
                fig = create_strategy_comparison_chart(analysis)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Detailed table
            st.subheader("All Windows Detail")
            show_window_details_table(analysis)

            # Additional insights
            with st.expander("Window Selection Insights"):
                st.markdown("""
                **Understanding Window Selection:**

                - **Bounce First Strategy**: Prioritizes windows with the most bounces (channel oscillations).
                  Higher bounce count = more validated channel structure.

                - **Quality Score Strategy**: Uses `bounce_count * r_squared` as a composite score.
                  Balances oscillation frequency with trend linearity.

                - **Balanced Strategy**: Combines bounce quality (40%) with label validity (60%).
                  Useful during training to ensure good downstream labels.

                **Confidence Levels:**
                - **100%**: Clear winner, significantly better than alternatives
                - **70-90%**: Good confidence, moderate gap to runner-up
                - **50-60%**: Close call, multiple windows are similar quality

                **When Strategies Disagree:**
                If different strategies pick different windows, consider:
                1. The market may be transitioning between timeframes
                2. Multiple valid channel interpretations exist
                3. Review the raw data to understand the structure
                """)

        else:
            st.info("Click 'Analyze Windows' to run the window selection analysis")

    with tab3:
        st.header("Feature Analysis")

        if predictor is not None:
            top_k = st.slider("Top K Features", 10, 100, 50)
            show_feature_importance(predictor, top_k)
        else:
            st.info("Load a model to see feature importance")

    with tab4:
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

    with tab5:
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
