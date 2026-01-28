"""
V15 Streamlit Dashboard

Run with: streamlit run v15/dashboard.py

Features:
- Live predictions with current market data
- Feature importance visualization
- Model performance metrics
- Correlation analysis display
- Window selection analysis and comparison
- Channel visualization across all timeframes
- Live data integration with yfinance
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

# Import new visualization modules
from v15.visualization.plotly_charts import (
    create_tf_channel_chart,
    create_candlestick_chart,
    add_channel_overlay,
    PLOTLY_AVAILABLE
)

# Import live data module
from v15.live_data import (
    YFinanceLiveData,
    should_refresh,
    fetch_live_data,
    get_market_status,
    YFINANCE_AVAILABLE
)

logger = logging.getLogger(__name__)

# Import window selection strategies
try:
    # Try to import v7 window strategies from deprecated folder
    import sys
    from pathlib import Path
    _deprecated_dir = Path(__file__).parent.parent / 'deprecated'
    if str(_deprecated_dir) not in sys.path:
        sys.path.insert(0, str(_deprecated_dir))
    from v7.core.window_strategy import (
        SelectionStrategy, get_strategy,
        BounceFirstStrategy, LabelValidityStrategy,
        BalancedScoreStrategy, QualityScoreStrategy
    )
    WINDOW_STRATEGIES_AVAILABLE = True
except ImportError:
    WINDOW_STRATEGIES_AVAILABLE = False
    logger.warning("Window selection strategies not available - v7.core.window_strategy not found")

# Import channel detection from v15 (was v7)
from v15.core.channel import detect_channels_multi_window, select_best_channel, STANDARD_WINDOWS as V7_WINDOWS

# Try to import streamlit-autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False
    logger.info("streamlit-autorefresh not available - using manual refresh")


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


# =============================================================================
# Per-TF Predictions Display
# =============================================================================

def show_per_tf_predictions_table(prediction):
    """Show per-timeframe prediction breakdown table."""
    if prediction.per_tf_predictions is None:
        st.info("Model doesn't support per-TF predictions. Train with per-TF heads enabled.")
        return

    data = []
    for tf_name in TIMEFRAMES:
        if tf_name in prediction.per_tf_predictions:
            tf_pred = prediction.per_tf_predictions[tf_name]
            data.append({
                'Timeframe': tf_name,
                'Duration': f"{tf_pred.duration_mean:.0f} +/- {tf_pred.duration_std:.0f}",
                'Confidence': f"{tf_pred.confidence:.0%}",
                'Window': tf_pred.best_window,
            })
        else:
            data.append({
                'Timeframe': tf_name,
                'Duration': 'N/A',
                'Confidence': 'N/A',
                'Window': 'N/A',
            })

    df = pd.DataFrame(data)

    # Style based on confidence
    def style_confidence(row):
        try:
            conf_str = row['Confidence']
            if conf_str == 'N/A':
                return [''] * len(row)
            conf = float(conf_str.rstrip('%')) / 100
            if conf >= 0.8:
                return ['background-color: #d4edda'] * len(row)  # Green
            elif conf >= 0.6:
                return ['background-color: #fff3cd'] * len(row)  # Yellow
            else:
                return ['background-color: #f8d7da'] * len(row)  # Red
        except:
            return [''] * len(row)

    styled_df = df.style.apply(style_confidence, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


# =============================================================================
# Channel Visualization Tab
# =============================================================================

def show_channel_visualization_tab(tsla_df: pd.DataFrame, prediction=None):
    """
    Display channel visualization for all 10 timeframes.

    Args:
        tsla_df: TSLA OHLCV DataFrame (5-min base)
        prediction: Optional prediction with per_tf_predictions
    """
    from v15.features.tf_extractor import resample_to_timeframe

    st.header("Channel Visualization")

    if not PLOTLY_AVAILABLE:
        st.error("Plotly is not available. Install with: pip install plotly")
        return

    st.markdown("""
    This tab shows channel detection results across all 10 timeframes.
    Each expander contains a candlestick chart with channel overlay and bounce markers.
    """)

    # Iterate through all timeframes
    for tf_name in TIMEFRAMES:
        # Get per-TF prediction info if available
        duration = 0.0
        confidence = 0.0
        tf_window = 50

        if prediction is not None and prediction.per_tf_predictions is not None:
            if tf_name in prediction.per_tf_predictions:
                tf_pred = prediction.per_tf_predictions[tf_name]
                duration = tf_pred.duration_mean
                confidence = tf_pred.confidence
                tf_window = tf_pred.best_window

        # Create expander with summary in header
        expander_header = f"{tf_name} - Duration: {duration:.0f} bars - Conf: {confidence:.0%}"

        with st.expander(expander_header, expanded=False):
            try:
                # Resample to this timeframe
                tf_df = resample_to_timeframe(tsla_df, tf_name)

                if len(tf_df) < 20:
                    st.warning(f"Insufficient data for {tf_name} ({len(tf_df)} bars)")
                    continue

                # Detect channel at this timeframe
                tf_channels = detect_channels_multi_window(tf_df, windows=STANDARD_WINDOWS)
                best_channel, best_window = select_best_channel(tf_channels)

                if best_window is None:
                    best_window = 50

                # Use the window from channel detection if prediction doesn't have it
                if tf_window == 50 and best_window != 50:
                    tf_window = best_window

                # Get the channel for the selected window
                channel = tf_channels.get(tf_window)

                # Slice data to show just the channel window
                window_bars = min(tf_window, len(tf_df))
                chart_df = tf_df.iloc[-window_bars:].copy()

                if channel is not None and getattr(channel, 'valid', False):
                    # Create chart with channel
                    fig = create_tf_channel_chart(
                        df=chart_df,
                        channel=channel,
                        tf_name=tf_name,
                        duration=duration,
                        confidence=confidence,
                        show_bounces=True,
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Show channel metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        bounce_count = getattr(channel, 'bounce_count', 0)
                        st.metric("Bounces", bounce_count)
                    with col2:
                        r_squared = getattr(channel, 'r_squared', 0.0)
                        st.metric("R-squared", f"{r_squared:.3f}")
                    with col3:
                        direction = getattr(channel, 'direction', 1)
                        dir_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
                        st.metric("Direction", dir_names.get(int(direction), 'Unknown'))
                    with col4:
                        st.metric("Window", f"{tf_window} bars")
                else:
                    # No valid channel - show placeholder
                    st.warning("No valid channel detected for this timeframe")

                    # Still show candlestick chart without channel
                    fig = create_candlestick_chart(chart_df)
                    fig.update_layout(
                        title=f"{tf_name} - No Valid Channel",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing {tf_name}: {e}")
                logger.exception(f"Channel visualization error for {tf_name}")


# =============================================================================
# Live Data Integration
# =============================================================================

def get_live_data_with_fallback(
    use_live: bool,
    loaded_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    lookback: int = 35000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, Optional[str]]:
    """
    Get market data, preferring live data if enabled and available.

    Args:
        use_live: Whether to attempt live data fetch
        loaded_data: Tuple of (tsla, spy, vix) loaded from files
        lookback: Number of bars to fetch for live data

    Returns:
        Tuple of (tsla, spy, vix, is_live, error_msg)
    """
    tsla, spy, vix = loaded_data
    is_live = False
    error_msg = None

    if use_live:
        if not YFINANCE_AVAILABLE:
            error_msg = "yfinance not installed. Install with: pip install yfinance"
            return tsla, spy, vix, False, error_msg

        try:
            data_feed = YFinanceLiveData(cache_ttl=60)
            tsla_live, spy_live, vix_live = data_feed.get_historical(
                period='60d',
                interval='5m'
            )

            # Take last N bars
            if len(tsla_live) > lookback:
                tsla_live = tsla_live.iloc[-lookback:]
                spy_live = spy_live.iloc[-lookback:]
                vix_live = vix_live.iloc[-lookback:]

            return tsla_live, spy_live, vix_live, True, None

        except Exception as e:
            error_msg = f"Live data fetch failed: {e}. Falling back to loaded data."
            logger.exception("Live data error")
            return tsla, spy, vix, False, error_msg

    return tsla, spy, vix, False, None


def show_live_data_sidebar():
    """Show live data configuration in sidebar. Returns config dict."""
    st.sidebar.divider()
    st.sidebar.header("Live Data")

    config = {
        'use_live': False,
        'auto_refresh': False,
        'refresh_interval': 300,
    }

    if YFINANCE_AVAILABLE:
        config['use_live'] = st.sidebar.checkbox(
            "Use Live Data (yfinance)",
            value=False,
            help="Fetch real-time data from Yahoo Finance instead of loaded files"
        )

        if config['use_live']:
            config['auto_refresh'] = st.sidebar.checkbox(
                "Auto-refresh",
                value=False,
                help="Automatically refresh data at specified interval"
            )

            if config['auto_refresh']:
                interval_options = {
                    "1 minute": 60,
                    "5 minutes": 300,
                    "15 minutes": 900,
                }
                interval_label = st.sidebar.selectbox(
                    "Refresh Interval",
                    options=list(interval_options.keys()),
                    index=1  # Default to 5 minutes
                )
                config['refresh_interval'] = interval_options[interval_label]

                if AUTOREFRESH_AVAILABLE:
                    # Auto-refresh using streamlit-autorefresh
                    count = st_autorefresh(
                        interval=config['refresh_interval'] * 1000,
                        limit=None,
                        key="live_data_refresh"
                    )
                    st.sidebar.caption(f"Auto-refresh count: {count}")
                else:
                    st.sidebar.info("Install streamlit-autorefresh for auto-refresh")
                    if st.sidebar.button("Manual Refresh"):
                        st.rerun()

            # Show market status
            try:
                market_status = get_market_status()
                status_text = "OPEN" if market_status['is_open'] else "CLOSED"
                status_color = "green" if market_status['is_open'] else "red"
                st.sidebar.markdown(f"Market: :{status_color}[{status_text}]")
                st.sidebar.caption(market_status['current_time_et'])
            except Exception:
                pass

    else:
        st.sidebar.warning("yfinance not installed")
        st.sidebar.caption("pip install yfinance")

    return config


def show_data_status_bar(is_live: bool, error_msg: Optional[str], last_update: Optional[datetime]):
    """Show data status bar at top of page."""
    if error_msg:
        st.warning(error_msg)

    col1, col2, col3 = st.columns(3)

    with col1:
        source = "Live (yfinance)" if is_live else "Loaded (CSV)"
        st.info(f"Data Source: {source}")

    with col2:
        if last_update:
            st.info(f"Last Update: {last_update.strftime('%H:%M:%S')}")
        else:
            st.info("Last Update: N/A")

    with col3:
        if is_live:
            try:
                market_status = get_market_status()
                if market_status['is_open']:
                    st.success("Market: OPEN")
                else:
                    st.info(f"Market: CLOSED (next open: {market_status.get('next_open', 'N/A')})")
            except Exception:
                st.info("Market Status: Unknown")
        else:
            st.info("Market Status: N/A (using loaded data)")


# =============================================================================
# Page Configuration & Caching
# =============================================================================

# Page config
st.set_page_config(
    page_title="V15 Channel Predictor",
    page_icon="",
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
            f"+/-{prediction.duration_std:.0f}"
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
    st.title("V15 Channel Break Predictor")

    # Sidebar config
    st.sidebar.header("Configuration")

    data_dir = st.sidebar.text_input("Data Directory", value="data")
    checkpoint_path = st.sidebar.text_input(
        "Model Checkpoint",
        value="checkpoints/best.pt"
    )

    # Live data sidebar configuration
    live_config = show_live_data_sidebar()

    # Check if paths exist
    if not Path(data_dir).exists():
        st.error(f"Data directory not found: {data_dir}")
        st.info("Please provide path to directory containing TSLA_1min.csv, SPY_1min.csv, VIX_History.csv")
        return

    # Load data from files (used as fallback)
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

    # Get data (live or loaded)
    current_tsla, current_spy, current_vix, is_live, error_msg = get_live_data_with_fallback(
        use_live=live_config['use_live'],
        loaded_data=(tsla, spy, vix),
        lookback=35000
    )

    # Track last update time
    if 'last_update' not in st.session_state:
        st.session_state['last_update'] = datetime.now()
    if is_live:
        st.session_state['last_update'] = datetime.now()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Live Prediction",
        "Channel Visualization",
        "Window Selection",
        "Feature Analysis",
        "Model Info",
        "Data Explorer"
    ])

    with tab1:
        st.header("Live Prediction")

        # Show data status
        show_data_status_bar(is_live, error_msg, st.session_state.get('last_update'))

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
                        # Use current data (live or loaded)
                        tsla_slice = current_tsla.iloc[-lookback:]
                        spy_slice = current_spy.iloc[-lookback:]
                        vix_slice = current_vix.iloc[-lookback:]

                        # Use predict_with_per_tf for per-TF breakdown
                        try:
                            prediction = predictor.predict_with_per_tf(
                                tsla_slice, spy_slice, vix_slice
                            )
                        except AttributeError:
                            # Fallback if predict_with_per_tf not available
                            prediction = predictor.predict(
                                tsla_slice, spy_slice, vix_slice
                            )

                        # Store prediction for channel visualization tab
                        st.session_state['last_prediction'] = prediction
                        st.session_state['prediction_data'] = (tsla_slice, spy_slice, vix_slice)

                        st.success("Prediction complete!")
                        show_prediction_card(prediction)

                        # Show per-TF breakdown table
                        st.divider()
                        st.subheader("Per-Timeframe Breakdown")
                        show_per_tf_predictions_table(prediction)

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
                                'learned_window': prediction.learned_window,
                                'used_learned_selection': prediction.used_learned_selection,
                            })

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        logger.exception("Prediction error")

    with tab2:
        # Channel Visualization tab
        prediction = st.session_state.get('last_prediction')
        pred_data = st.session_state.get('prediction_data')

        if pred_data is not None:
            tsla_slice, _, _ = pred_data
            show_channel_visualization_tab(tsla_slice, prediction)
        else:
            st.info("Make a prediction first to see channel visualization across timeframes")

            # Allow visualization without prediction
            if st.button("Visualize Channels (without prediction)"):
                lookback = st.slider(
                    "Lookback for visualization",
                    min_value=1000,
                    max_value=20000,
                    value=5000,
                    key="viz_lookback"
                )
                show_channel_visualization_tab(current_tsla.iloc[-lookback:], None)

    with tab3:
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
                    tsla_slice = current_tsla.iloc[-window_lookback:]

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

    with tab4:
        st.header("Feature Analysis")

        if predictor is not None:
            top_k = st.slider("Top K Features", 10, 100, 50)
            show_feature_importance(predictor, top_k)
        else:
            st.info("Load a model to see feature importance")

    with tab5:
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

            # Additional model info
            st.divider()
            st.subheader("Model Capabilities")

            col1, col2 = st.columns(2)
            with col1:
                has_window_sel = predictor.has_learned_window_selection
                st.metric("Learned Window Selection", "Yes" if has_window_sel else "No")

            with col2:
                # Check if model supports per-TF predictions
                has_per_tf = hasattr(model, 'forward_with_per_tf')
                st.metric("Per-TF Predictions", "Yes" if has_per_tf else "No")

        else:
            st.info("Load a model to see information")

    with tab6:
        st.header("Data Explorer")

        # Show data status
        show_data_status_bar(is_live, error_msg, st.session_state.get('last_update'))

        st.subheader("TSLA")
        st.line_chart(current_tsla['close'].tail(1000))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Bars", f"{len(current_tsla):,}")
            st.metric("Start", str(current_tsla.index[0].date()))
        with col2:
            st.metric("End", str(current_tsla.index[-1].date()))
            st.metric("Last Close", f"${current_tsla['close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
