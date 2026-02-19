"""
X23 Streamlit Dashboard

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
from v15.signals.bounce_signal import SignalStrategy

try:
    from v15.inference import get_cpp_status
except ImportError:
    def get_cpp_status():
        return {'available': False, 'feature_count': 0, 'expected': 15350}

# Import new visualization modules
from v15.visualization.plotly_charts import (
    create_tf_channel_chart,
    create_candlestick_chart,
    add_channel_overlay,
    add_duration_projection,
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

# Import live trading monitor
try:
    from v15.trading.live_monitor import TradingMonitor
    from v15.trading.signals import SignalType as RASignalType
    TRADING_MONITOR_AVAILABLE = True
except ImportError:
    TRADING_MONITOR_AVAILABLE = False

# Import native TF data loader for per-TF yfinance fetching
from v15.data.native_tf import load_native_tf_data as _load_native_tf_data

# Native TFs fetched from yfinance.
# Keep long-history TFs first so critical weekly/monthly data is available early.
# 5min/15min/30min are derived locally from the base 5-min feed.
NATIVE_TF_LIST = ['daily', 'weekly', 'monthly', '1h', '2h', '3h', '4h']

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
    st.dataframe(styled_df, width="stretch", hide_index=True)


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
    """Show per-timeframe prediction breakdown table with direction and horizon grouping."""
    from v15.config import TF_TO_HORIZON

    if prediction.per_tf_predictions is None:
        st.info("Model doesn't support per-TF predictions. Train with per-TF heads enabled.")
        return

    data = []
    for tf_name in TIMEFRAMES:
        if tf_name in prediction.per_tf_predictions:
            tf_pred = prediction.per_tf_predictions[tf_name]
            horizon = TF_TO_HORIZON.get(tf_name, '?')
            # Extract next_channel if available
            nc_str = 'N/A'
            nc_prob_str = 'N/A'
            if hasattr(tf_pred, 'next_channel') and hasattr(tf_pred, 'next_channel_probs'):
                nc_str = tf_pred.next_channel.upper()
                nc_prob = max(tf_pred.next_channel_probs.values())
                nc_prob_str = f"{nc_prob:.0%}"

            data.append({
                'Horizon': horizon.title(),
                'Timeframe': tf_name,
                'Direction': tf_pred.direction.upper(),
                'Dir Prob': f"{tf_pred.direction_prob:.0%}",
                'Next Channel': nc_str,
                'NC Prob': nc_prob_str,
                'Confidence': f"{tf_pred.confidence:.0%}",
                'Duration': f"{tf_pred.duration_mean:.0f} +/- {tf_pred.duration_std:.0f}",
                'Window': tf_pred.best_window,
                '_direction': tf_pred.direction,  # hidden, for styling
                '_confidence': tf_pred.confidence,  # hidden, for styling
            })
        else:
            data.append({
                'Horizon': TF_TO_HORIZON.get(tf_name, '?').title(),
                'Timeframe': tf_name,
                'Direction': 'N/A',
                'Dir Prob': 'N/A',
                'Next Channel': 'N/A',
                'NC Prob': 'N/A',
                'Confidence': 'N/A',
                'Duration': 'N/A',
                'Window': 'N/A',
                '_direction': None,
                '_confidence': 0.0,
            })

    df = pd.DataFrame(data)

    # Style rows: green for up, red for down, dim for low confidence
    def style_row(row):
        direction = row.get('_direction')
        confidence = row.get('_confidence', 0.0)
        if direction is None:
            return [''] * len(row)
        if confidence < 0.55:
            return ['background-color: #3a3a3a; color: #e0e0e0'] * len(row)
        if direction == 'up':
            return ['background-color: #d4edda'] * len(row)  # Green
        else:
            return ['background-color: #f8d7da'] * len(row)  # Red

    # Drop hidden columns for display
    display_df = df.drop(columns=['_direction', '_confidence'])
    styled_df = df.style.apply(style_row, axis=1)
    # Hide the internal columns in the styled output
    styled_df = styled_df.hide(subset=['_direction', '_confidence'], axis='columns')
    st.dataframe(styled_df, width="stretch", hide_index=True)


# =============================================================================
# Channel Visualization Tab
# =============================================================================

def show_channel_visualization_tab(tsla_df: pd.DataFrame, prediction=None, native_bars_by_tf=None):
    """
    Display channel visualization for all 10 timeframes.

    Args:
        tsla_df: TSLA OHLCV DataFrame (5-min base)
        prediction: Optional prediction with per_tf_predictions
        native_bars_by_tf: Optional dict of native TF data keyed by symbol then timeframe
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
        tf_direction = None
        tf_next_channel = None
        tf_pred = None

        if prediction is not None and prediction.per_tf_predictions is not None:
            if tf_name in prediction.per_tf_predictions:
                tf_pred = prediction.per_tf_predictions[tf_name]
                duration = tf_pred.duration_mean
                confidence = tf_pred.confidence
                tf_window = tf_pred.best_window
                tf_direction = tf_pred.direction
                if hasattr(tf_pred, 'next_channel'):
                    tf_next_channel = tf_pred.next_channel

        # Create expander with summary in header
        dir_str = tf_direction.upper() if tf_direction else '?'
        nc_str = f" → {tf_next_channel.upper()}" if tf_next_channel else ""
        expander_header = f"{tf_name} - {dir_str}{nc_str} - Duration: {duration:.0f} bars - Conf: {confidence:.0%}"

        with st.expander(expander_header, expanded=False):
            try:
                # Use native TF data if available for this timeframe
                used_native = False
                if (native_bars_by_tf
                        and native_bars_by_tf.get('TSLA', {}).get(tf_name) is not None
                        and len(native_bars_by_tf['TSLA'][tf_name]) >= 20):
                    tf_df = native_bars_by_tf['TSLA'][tf_name].copy()
                    used_native = True
                    print(f"[VIZ] {tf_name}: Using NATIVE data ({len(tf_df)} bars)")
                else:
                    tf_df = resample_to_timeframe(tsla_df, tf_name)
                    print(f"[VIZ] {tf_name}: Using RESAMPLED from 5min ({len(tf_df)} bars)")

                if len(tf_df) < 20:
                    st.warning(f"⚠️ Insufficient data for {tf_name}: {len(tf_df)} bars (used_native={used_native})")
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

                # Slice data: channel window + current bar for display.
                # detect_channel uses df.iloc[-(window+1):-1] internally,
                # so the first window_bars of chart_df align with the channel.
                # The extra bar at the end is the current bar (shown as a
                # candle but outside the channel overlay).
                # Ensure at least MIN_CHART_BARS so small best_window values
                # (e.g. 5-6 for 30min) don't produce tiny charts.
                MIN_CHART_BARS = 50
                display_bars = max(tf_window, MIN_CHART_BARS)
                window_bars = min(display_bars, len(tf_df) - 1)
                chart_df = tf_df.iloc[-(window_bars + 1):].copy()

                # DIAGNOSTIC: Log slicing for all timeframes
                print(f"[VIZ] {tf_name} SLICE: total_bars={len(tf_df)}, tf_window={tf_window}, display_bars={display_bars}, chart_bars={len(chart_df)}, used_native={used_native}")
                if len(chart_df) > 0:
                    print(f"[VIZ] {tf_name} DATE RANGE: {chart_df.index[0]} to {chart_df.index[-1]}")

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

                    # Add duration projection overlay
                    if duration > 0 and confidence > 0:
                        duration_std = 0.0
                        if (prediction is not None
                                and prediction.per_tf_predictions is not None
                                and tf_name in prediction.per_tf_predictions):
                            duration_std = prediction.per_tf_predictions[tf_name].duration_std

                        agg_direction = tf_pred.direction if tf_pred else (prediction.direction if prediction else None)

                        fig = add_duration_projection(
                            fig, channel, chart_df,
                            duration, duration_std, confidence, agg_direction,
                            tf_name=tf_name,
                        )

                    st.plotly_chart(fig, width="stretch")

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
                    fig = create_candlestick_chart(chart_df, tf_name=tf_name)
                    fig.update_layout(
                        title=f"{tf_name} - No Valid Channel",
                        height=350
                    )
                    st.plotly_chart(fig, width="stretch")

            except Exception as e:
                st.error(f"Error processing {tf_name}: {e}")
                logger.exception(f"Channel visualization error for {tf_name}")


# =============================================================================
# Live Data Integration
# =============================================================================

@st.cache_data(ttl=60)
def _fetch_live_5min() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cached 5-min yfinance fetch (TTL=60s to avoid redundant API calls)."""
    print("[DATA] Fetching live 5-min data from yfinance (cache miss)...")
    data_feed = YFinanceLiveData(cache_ttl=60)
    return data_feed.get_historical(period='60d', interval='5m')


@st.cache_data(ttl=60)
def fetch_all_market_data():
    """Single source of truth: fetch ALL yfinance data (5-min + native TFs).

    Returns:
        Tuple of (native_data, live_5min) where:
        - native_data: dict from _load_native_tf_data() or None
        - live_5min: tuple of (tsla, spy, vix) DataFrames or None
    """
    native_data = None
    live_5min = None

    if not YFINANCE_AVAILABLE:
        return native_data, live_5min

    # 1. Fetch native TFs (daily/weekly/monthly/1h-4h) — years of history
    try:
        print("[DATA] fetch_all_market_data: fetching native TFs...")
        native_data = _load_native_tf_data(
            symbols=['TSLA', 'SPY', '^VIX'],
            timeframes=NATIVE_TF_LIST,
            start_date='2015-01-01',
            use_cache=True,
            cache_max_age_hours=5 / 60,
            max_retries=2,
            retry_delay=0.75,
            yf_request_timeout=8.0,
            request_wall_timeout=20.0,
            inter_request_delay=0.25,
            verbose=True,
        )
        if native_data:
            for symbol in ['TSLA', 'SPY', '^VIX']:
                for tf in NATIVE_TF_LIST:
                    df = native_data.get(symbol, {}).get(tf)
                    if df is not None and not df.empty:
                        print(f"[DATA]   {symbol:5s} {tf:8s}: {len(df):4d} bars")
    except Exception as e:
        print(f"[DATA] Native TF fetch failed: {e}")
        logger.exception("Native TF data fetch failed")

    # 2. Fetch 5-min data (60 days, yfinance limit)
    try:
        print("[DATA] fetch_all_market_data: fetching 5-min data...")
        data_feed = YFinanceLiveData(cache_ttl=60)
        live_5min = data_feed.get_historical(period='60d', interval='5m')
    except Exception as e:
        print(f"[DATA] 5-min yfinance failed: {e}")

    return native_data, live_5min


def get_live_data_with_fallback(
    use_live: bool,
    loaded_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    live_5min: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None,
    lookback: int = 35000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, Optional[str]]:
    """
    Get market data, merging CSV history with fresh yfinance data.

    When use_live=True, uses provided live 5-min data (or fetches it)
    and appends to CSV historical data. This gives us years of history plus
    current market data — both are needed for all 10 timeframes.

    Args:
        use_live: Whether to attempt live data merge
        loaded_data: Tuple of (tsla, spy, vix) loaded from CSV files
        live_5min: Optional pre-fetched 5-min data tuple from fetch_all_market_data()
        lookback: Number of bars to return (from the end)

    Returns:
        Tuple of (tsla, spy, vix, is_live, error_msg)
    """
    tsla, spy, vix = loaded_data
    is_live = False
    error_msg = None

    print(f"[DATA] CSV data: TSLA={len(tsla):,} bars, SPY={len(spy):,} bars, "
          f"VIX={len(vix):,} bars")
    if len(tsla) > 0:
        print(f"[DATA] CSV range: {tsla.index[0]} to {tsla.index[-1]} "
              f"(tz={tsla.index.tz})")

    if use_live:
        if not YFINANCE_AVAILABLE:
            error_msg = "yfinance not installed. Install with: pip install yfinance"
            return tsla, spy, vix, False, error_msg

        try:
            # Use pre-fetched 5-min data or fall back to direct fetch
            if live_5min is not None:
                tsla_live, spy_live, vix_live = live_5min
                print(f"[DATA] Using pre-fetched 5-min data")
            else:
                print("[DATA] Fetching live data from yfinance...")
                tsla_live, spy_live, vix_live = _fetch_live_5min()
            print(f"[DATA] yfinance returned: TSLA={len(tsla_live):,} bars "
                  f"({tsla_live.index[0]} to {tsla_live.index[-1]}, "
                  f"tz={tsla_live.index.tz})")

            # Normalize timezones to UTC for consistent comparison
            # CSV data is tz-naive (represents America/New_York market time)
            # yfinance data is tz-aware (America/New_York)
            def _normalize_to_utc(df: pd.DataFrame) -> pd.DataFrame:
                if len(df) == 0 or not isinstance(df.index, pd.DatetimeIndex):
                    return df
                if df.index.tz is None:
                    # tz-naive CSV data: localize to NY then convert to UTC
                    df = df.copy()
                    df.index = df.index.tz_localize(
                        "America/New_York",
                        ambiguous="infer",
                        nonexistent="shift_forward"
                    ).tz_convert("UTC")
                else:
                    # tz-aware yfinance data: convert to UTC
                    df = df.copy()
                    df.index = df.index.tz_convert("UTC")
                return df

            tsla = _normalize_to_utc(tsla)
            spy = _normalize_to_utc(spy)
            vix = _normalize_to_utc(vix)
            tsla_live = _normalize_to_utc(tsla_live)
            spy_live = _normalize_to_utc(spy_live)
            vix_live = _normalize_to_utc(vix_live)
            print(f"[DATA] Normalized all data to UTC")

            # Merge: CSV history + fresh yfinance data
            # Find where yfinance data starts after CSV ends
            csv_end = tsla.index[-1] if len(tsla) > 0 else pd.Timestamp.min.tz_localize("UTC")
            fresh_tsla = tsla_live[tsla_live.index > csv_end]
            fresh_spy = spy_live[spy_live.index > csv_end]
            fresh_vix = vix_live[vix_live.index > csv_end]

            if len(fresh_tsla) > 0:
                tsla_merged = pd.concat([tsla, fresh_tsla])
                spy_merged = pd.concat([spy, fresh_spy])
                vix_merged = pd.concat([vix, fresh_vix])
                print(f"[DATA] Merged: {len(tsla):,} CSV + {len(fresh_tsla):,} fresh "
                      f"= {len(tsla_merged):,} total bars")
            else:
                # yfinance data is older than or overlaps with CSV — use CSV as-is
                # but update the last few bars with live values for freshness
                overlap_tsla = tsla_live[tsla_live.index >= tsla.index[0]]
                if len(overlap_tsla) > 0:
                    # Replace overlapping bars with live data (more current)
                    tsla_merged = tsla.copy()
                    spy_merged = spy.copy()
                    vix_merged = vix.copy()
                    tsla_merged.update(overlap_tsla)
                    spy_merged.update(spy_live[spy_live.index >= spy.index[0]])
                    vix_merged.update(vix_live[vix_live.index >= vix.index[0]])
                    print(f"[DATA] Updated {len(overlap_tsla):,} overlapping bars with live data")
                else:
                    tsla_merged = tsla
                    spy_merged = spy
                    vix_merged = vix
                    print("[DATA] No overlap — using CSV data as-is")

            # Take last N bars
            if len(tsla_merged) > lookback:
                tsla_merged = tsla_merged.iloc[-lookback:]
                spy_merged = spy_merged.iloc[-lookback:]
                vix_merged = vix_merged.iloc[-lookback:]

            print(f"[DATA] Final: {len(tsla_merged):,} bars (is_live=True)")
            return tsla_merged, spy_merged, vix_merged, True, None

        except Exception as e:
            error_msg = f"Live data fetch failed: {e}. Using CSV data only."
            logger.exception("Live data error")
            print(f"[DATA] yfinance failed: {e}")
            print(f"[DATA] Falling back to CSV: {len(tsla):,} bars")

    # Take last N bars from CSV
    if len(tsla) > lookback:
        tsla = tsla.iloc[-lookback:]
        spy = spy.iloc[-lookback:]
        vix = vix.iloc[-lookback:]

    print(f"[DATA] Final: {len(tsla):,} bars (is_live={is_live})")
    return tsla, spy, vix, is_live, error_msg


def show_live_data_sidebar():
    """Show live data configuration in sidebar. Returns config dict."""
    st.sidebar.divider()
    st.sidebar.header("Live Data")

    config = {
        'use_live': True,
        'auto_refresh': False,
        'refresh_interval': 300,
    }

    if not YFINANCE_AVAILABLE:
        st.sidebar.error("yfinance not installed — using CSV fallback")
        st.sidebar.caption("pip install yfinance")
        config['use_live'] = False
        return config

    # Auto-refresh options
    config['auto_refresh'] = st.sidebar.checkbox(
        "Auto-refresh",
        value=False,
        help="Automatically refresh live data at specified interval"
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
            index=1
        )
        config['refresh_interval'] = interval_options[interval_label]

        if AUTOREFRESH_AVAILABLE:
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

    # Market status
    try:
        market_status = get_market_status()
        status_text = "OPEN" if market_status['is_open'] else "CLOSED"
        status_color = "green" if market_status['is_open'] else "red"
        st.sidebar.markdown(f"Market: :{status_color}[{status_text}]")
        st.sidebar.caption(market_status['current_time_et'])
    except Exception:
        pass

    return config


def show_data_status_bar(is_live: bool, error_msg: Optional[str], last_update: Optional[datetime]):
    """Show data status bar at top of page."""
    if error_msg:
        st.error(error_msg)

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
    page_title="X23 Channel Predictor",
    page_icon="",
    layout="wide"
)


@st.cache_data(ttl=300)
def load_market_data(data_dir: str):
    """Load and cache market data."""
    from v15.data import load_market_data as _load
    return _load(data_dir)


CHECKPOINT_RELEASE_TAG = "v0.1-model"
CHECKPOINT_REPO = "frankywashere/autotrade2"
CHECKPOINT_ASSET_NAME = "x23_best_per_tf.pt"
CALIBRATION_ASSET_NAME = "temperature_calibration_x23.json"
DEFAULT_MODEL_PATH = f"models/{CHECKPOINT_ASSET_NAME}"


def _ensure_checkpoint(path: str = DEFAULT_MODEL_PATH) -> tuple:
    """Download checkpoint from GitHub Releases if not present locally.

    Returns:
        (path, asset_info) where asset_info is a dict with 'id', 'name',
        'uploaded_at' from GitHub, or None if loaded from local file.
    """
    p = Path(path)
    if p.exists():
        # Validate existing file isn't corrupt (e.g. from a failed previous download)
        if p.stat().st_size > 1_000_000:  # >1MB = plausible checkpoint
            return str(p), None
        else:
            print(f"[MODEL] Removing suspect file {p} ({p.stat().st_size} bytes)")
            p.unlink()

    p.parent.mkdir(parents=True, exist_ok=True)

    # For private repos, use token from Streamlit secrets or env
    import os
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        try:
            token = st.secrets["GITHUB_TOKEN"]
        except Exception:
            token = ""

    import requests as _requests

    # Fetch release by tag
    release_url = f"https://api.github.com/repos/{CHECKPOINT_REPO}/releases/tags/{CHECKPOINT_RELEASE_TAG}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    print(f"[MODEL] Fetching release info from {CHECKPOINT_RELEASE_TAG} ...")
    resp = _requests.get(release_url, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(
            f"GitHub API returned HTTP {resp.status_code} for release "
            f"{CHECKPOINT_RELEASE_TAG}: {resp.text[:500]}"
        )
    release = resp.json()

    assets = release.get("assets", [])
    if not assets:
        # Log the full response so we can debug if this is an auth/rate issue
        msg = release.get("message", "no error message")
        raise RuntimeError(
            f"No assets found in release {CHECKPOINT_RELEASE_TAG}. "
            f"GitHub response message: {msg}"
        )

    # Find the target .pt asset by name
    pt_assets = [a for a in assets if a["name"].endswith(".pt")]
    asset = None
    for a in pt_assets:
        if a["name"] == CHECKPOINT_ASSET_NAME:
            asset = a
            break
    if asset is None:
        asset = pt_assets[-1] if pt_assets else assets[-1]

    print(f"[MODEL] Downloading {asset['name']} ({asset['size'] / 1e6:.1f} MB) ...")

    def _download_asset(asset_dict, dest_path):
        """Download a GitHub release asset via browser_download_url."""
        # Use browser_download_url — works for public repos directly,
        # for private repos pass token in Authorization header
        url = asset_dict["browser_download_url"]
        dl_headers = {}
        if token:
            dl_headers["Authorization"] = f"token {token}"
        resp = _requests.get(url, headers=dl_headers, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        actual_size = dest_path.stat().st_size
        if asset_dict["size"] > 1000 and actual_size < asset_dict["size"] * 0.5:
            dest_path.unlink()
            raise RuntimeError(
                f"Download corrupt: got {actual_size} bytes, "
                f"expected ~{asset_dict['size']} bytes"
            )
        return actual_size

    # Download the checkpoint binary
    dl_size = _download_asset(asset, p)
    print(f"[MODEL] Downloaded checkpoint: {dl_size / 1e6:.1f} MB")

    # Also download temperature_calibration_x23.json if available
    cal_assets = [a for a in assets if a["name"] == CALIBRATION_ASSET_NAME]
    if cal_assets:
        cal_asset = cal_assets[0]
        cal_path = p.parent / CALIBRATION_ASSET_NAME
        print(f"[MODEL] Downloading {cal_asset['name']} ...")
        _download_asset(cal_asset, cal_path)
        print(f"[MODEL] Downloaded calibration: {cal_path}")

    asset_info = {
        'id': asset.get('id'),
        'name': asset.get('name'),
        'uploaded_at': asset.get('updated_at', asset.get('created_at', '')),
    }
    return str(p), asset_info


@st.cache_resource
def load_predictor(checkpoint_path: str):
    """Load and cache live predictor with channel history tracking."""
    from v15.live import LivePredictor
    cal_path = str(Path(checkpoint_path).parent / CALIBRATION_ASSET_NAME)
    return LivePredictor(checkpoint_path, track_channel_history=True,
                         calibration_path=cal_path)


## load_native_tf() removed — replaced by fetch_all_market_data()


def show_prediction_card(prediction):
    """Display prediction card with trade recommendations, conflicts, and aggregated summary."""
    # Trade Recommendations by horizon
    if prediction.trade_recommendations:
        st.subheader("Trade Recommendations")
        cols = st.columns(3)
        horizon_labels = {'short': 'Short Term', 'medium': 'Medium Term', 'long': 'Long Term'}
        for i, horizon in enumerate(['short', 'medium', 'long']):
            with cols[i]:
                rec = prediction.trade_recommendations.get(horizon)
                if rec:
                    dir_icon = "UP" if rec.direction == 'up' else "DOWN"
                    st.metric(
                        horizon_labels[horizon],
                        f"{dir_icon} ({rec.timeframe})",
                        f"Score: {rec.score:.2f} | Conf: {rec.confidence:.0%}"
                    )
                    st.caption(f"Duration: {rec.duration_mean:.0f} +/- {rec.duration_std:.0f} bars")
                else:
                    st.metric(horizon_labels[horizon], "N/A", "No data")

    # Conflict warnings
    if prediction.conflicts:
        for conflict in prediction.conflicts:
            st.warning(
                f"Conflict: {conflict.horizon_a} ({conflict.direction_a.upper()}) vs "
                f"{conflict.horizon_b} ({conflict.direction_b.upper()})"
            )

    # Aggregated summary
    st.subheader("Aggregated")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Break Direction",
            prediction.direction.upper(),
            f"{prediction.direction_prob:.1%}"
        )

    with col2:
        st.metric(
            "Duration",
            f"{prediction.duration_mean:.0f} bars",
            f"+/-{prediction.duration_std:.0f}"
        )

    with col3:
        # Next channel direction with top probability
        nc = prediction.new_channel.upper() if prediction.new_channel else "?"
        nc_probs = prediction.new_channel_probs or {}
        nc_conf = max(nc_probs.values()) if nc_probs else 0
        st.metric(
            "Next Channel",
            nc,
            f"{nc_conf:.1%}"
        )

    with col4:
        st.metric(
            "Confidence",
            f"{prediction.confidence:.1%}",
            f"Window: {prediction.best_window}"
        )


def show_channel_history_status(predictor):
    """Show channel history tracking status from LivePredictor."""
    if not hasattr(predictor, 'channel_history_by_tf'):
        return

    history = predictor.channel_history_by_tf
    total_entries = 0
    populated_tfs = 0

    for tf_name in TIMEFRAMES:
        tf_hist = history.get(tf_name, {})
        tsla_count = len(tf_hist.get('tsla', []))
        spy_count = len(tf_hist.get('spy', []))
        if tsla_count > 0 or spy_count > 0:
            populated_tfs += 1
        total_entries += tsla_count + spy_count

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Channel History",
            f"{total_entries} entries",
            help="Total channel history entries tracked across all TFs"
        )
    with col2:
        st.metric(
            "Active TFs",
            f"{populated_tfs}/{len(TIMEFRAMES)}",
            help="Timeframes with at least one channel history entry"
        )
    with col3:
        tracking = "Active" if predictor.track_channel_history else "Disabled"
        st.metric("History Tracking", tracking)


def show_feature_importance(predictor, top_k: int = 50):
    """Show feature importance from model weights."""
    if predictor.model.feature_weights is None:
        st.warning("Model doesn't have explicit feature weights")
        return

    if predictor.feature_names is None:
        st.error("Checkpoint is missing feature_names. Patch the checkpoint with "
                 "correct names from the flat file, or retrain with updated code.")
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
    st.plotly_chart(fig, width="stretch")


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
    st.plotly_chart(fig, width="stretch")


def _get_trading_monitor() -> 'TradingMonitor':
    """Get or create the TradingMonitor singleton in session state."""
    if 'trading_monitor' not in st.session_state:
        st.session_state['trading_monitor'] = TradingMonitor()
    return st.session_state['trading_monitor']


def _run_prediction_for_monitor(predictor, current_tsla, current_spy, current_vix):
    """Run prediction and cache result. Returns (prediction, live_pred) or (None, None)."""
    try:
        predictor.load_historical_data(current_tsla, current_spy, current_vix)
        live_pred = predictor.predict_with_per_tf()
        if live_pred is not None:
            prediction = live_pred.prediction
            st.session_state['last_prediction'] = prediction
            st.session_state['last_live_pred'] = live_pred
            st.session_state['prediction_data'] = (current_tsla, current_spy, current_vix)
            return prediction, live_pred
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    return None, None


def show_trading_monitor_tab(
    predictor,
    current_tsla,
    current_spy,
    current_vix,
    native_tf_data,
    live_config,
    is_live,
    missing_tfs,
):
    """Tab 7: Live Trading Monitor — signals, positions, risk, history."""
    st.header("Trading Monitor")

    if not TRADING_MONITOR_AVAILABLE:
        st.error("Trading monitor not available. Check v15/trading/ imports.")
        return

    if predictor is None:
        st.warning("Load a trained model to use the trading monitor.")
        return

    monitor = _get_trading_monitor()

    # --- Auto-predict on refresh ---
    auto_predict = live_config.get('auto_refresh', False)
    if auto_predict and not missing_tfs:
        prediction = st.session_state.get('last_prediction')
        # Always re-run prediction on auto-refresh to keep signals fresh
        with st.spinner("Evaluating signals..."):
            prediction, live_pred = _run_prediction_for_monitor(
                predictor, current_tsla, current_spy, current_vix
            )
    else:
        prediction = st.session_state.get('last_prediction')

    # Manual predict button when auto-refresh is off
    if not auto_predict:
        col_pred, col_status = st.columns([1, 3])
        with col_pred:
            if st.button("Evaluate Signals", type="primary", disabled=bool(missing_tfs)):
                with st.spinner("Evaluating signals..."):
                    prediction, live_pred = _run_prediction_for_monitor(
                        predictor, current_tsla, current_spy, current_vix
                    )
        with col_status:
            if missing_tfs:
                st.error(f"Missing data for {len(missing_tfs)} TFs — cannot evaluate.")

    # Get current price and VIX
    current_price = 0.0
    vix_level = 20.0
    if len(current_tsla) > 0 and 'close' in current_tsla.columns:
        current_price = float(current_tsla.iloc[-1]['close'])
    if len(current_vix) > 0 and 'close' in current_vix.columns:
        vix_level = float(current_vix.iloc[-1]['close'])

    # =====================================================================
    # Section A: Active Signal Alerts
    # =====================================================================
    st.subheader("Signal Alerts")

    signals = []
    if prediction is not None and prediction.per_tf_predictions:
        close_series = current_tsla['close'] if 'close' in current_tsla.columns else None
        signals = monitor.evaluate(
            prediction.per_tf_predictions,
            current_price,
            vix_level,
            tsla_close_series=close_series,
        )

    if signals:
        for fs in signals:
            sig = fs.signal
            pos = fs.position
            direction = sig.signal_type.value.upper()
            color = "#28a745" if sig.signal_type == RASignalType.LONG else "#dc3545"
            rr = f"{pos.risk_reward_ratio:.1f}" if pos else "?"
            shares = pos.shares if pos else 0
            sl_pct = f"{pos.stop_loss_pct:.1%}" if pos else "?"
            tp_pct = f"{pos.take_profit_pct:.1%}" if pos else "?"

            st.markdown(
                f"<div style='background-color:{color};color:white;padding:12px;"
                f"border-radius:8px;margin-bottom:8px;font-weight:bold'>"
                f"{direction} — {fs.strategy.upper()} | "
                f"Conf: {sig.confidence:.0%} | TF: {sig.primary_tf} | "
                f"Shares: {shares} | SL: {sl_pct} | TP: {tp_pct} | R:R {rr}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Enter position button
            with st.expander(f"Enter {fs.strategy} position", expanded=False):
                entry_price = st.number_input(
                    "Entry Price",
                    value=current_price,
                    step=0.01,
                    key=f"entry_price_{fs.strategy}",
                )
                shares_input = st.number_input(
                    "Shares",
                    value=shares,
                    min_value=1,
                    step=1,
                    key=f"entry_shares_{fs.strategy}",
                )
                if st.button(
                    f"Confirm Entry: {fs.strategy}",
                    key=f"confirm_entry_{fs.strategy}",
                ):
                    pos_id = monitor.enter_position(
                        fs,
                        actual_entry_price=entry_price,
                        shares_override=shares_input,
                    )
                    st.success(f"Position entered: {pos_id}")
                    st.rerun()
    else:
        if prediction is not None:
            st.info("No actionable signals at current evaluation.")
        else:
            st.info("Run 'Evaluate Signals' or enable auto-refresh to scan for signals.")

    # =====================================================================
    # Section B: Open Positions
    # =====================================================================
    st.subheader("Open Positions")

    if monitor.positions:
        # Check for exit alerts
        exit_alerts = []
        if current_price > 0:
            high = current_price * 1.001  # approximate intra-bar high
            low = current_price * 0.999
            if len(current_tsla) > 0:
                high = float(current_tsla.iloc[-1].get('high', current_price))
                low = float(current_tsla.iloc[-1].get('low', current_price))
            exit_alerts = monitor.check_exits(current_price, high, low)

        # Show exit alerts prominently
        alert_map = {pos_id: (reason, price) for pos_id, reason, price in exit_alerts}
        for strat_key, pos in monitor.positions.items():
            if pos.pos_id in alert_map:
                reason, alert_price = alert_map[pos.pos_id]
                st.markdown(
                    f"<div style='background-color:#dc3545;color:white;padding:10px;"
                    f"border-radius:8px;margin-bottom:4px;font-weight:bold'>"
                    f"EXIT ALERT: {strat_key.upper()} — {reason} "
                    f"@ ${alert_price:.2f}</div>",
                    unsafe_allow_html=True,
                )

        # Positions table
        pos_data = []
        for strat_key, pos in monitor.positions.items():
            if pos.direction == 'long':
                pnl = (current_price - pos.entry_price) * pos.shares
            else:
                pnl = (pos.entry_price - current_price) * pos.shares
            entry_value = pos.entry_price * pos.shares
            pnl_pct = pnl / entry_value if entry_value > 0 else 0.0

            entry_dt = datetime.fromisoformat(pos.entry_time)
            hold_td = datetime.now() - entry_dt
            hold_str = str(hold_td).split('.')[0]  # HH:MM:SS

            # Trail distance
            if pos.direction == 'long' and pos.best_price > pos.entry_price:
                trail_dist = (pos.best_price - current_price) / pos.best_price
            elif pos.direction == 'short' and pos.best_price > 0 and pos.best_price < pos.entry_price:
                trail_dist = (current_price - pos.best_price) / pos.best_price
            else:
                trail_dist = 0.0

            pos_data.append({
                'Strategy': strat_key,
                'Dir': pos.direction.upper(),
                'Entry': f"${pos.entry_price:.2f}",
                'Current': f"${current_price:.2f}",
                'P&L ($)': f"${pnl:+,.0f}",
                'P&L (%)': f"{pnl_pct:+.1%}",
                'Stop': f"${pos.stop_loss_price:.2f}",
                'TP': f"${pos.take_profit_price:.2f}",
                'Trail Dist': f"{trail_dist:.2%}",
                'Hold Time': hold_str,
            })

        st.dataframe(pd.DataFrame(pos_data), hide_index=True)

        # Close position buttons
        for strat_key, pos in list(monitor.positions.items()):
            with st.expander(f"Close {strat_key} position", expanded=False):
                exit_price = st.number_input(
                    "Exit Price",
                    value=current_price,
                    step=0.01,
                    key=f"exit_price_{strat_key}",
                )
                if st.button(
                    f"Confirm Exit: {strat_key}",
                    key=f"confirm_exit_{strat_key}",
                ):
                    reason = alert_map.get(pos.pos_id, ('manual', 0))[0] if pos.pos_id in alert_map else 'manual'
                    trade = monitor.exit_position(strat_key, exit_price, exit_reason=reason)
                    if trade:
                        st.success(
                            f"Closed {strat_key}: P&L ${trade.pnl:+,.2f} ({trade.pnl_pct:+.1%})"
                        )
                    st.rerun()
    else:
        st.info("No open positions. Enter a position from the Signal Alerts section above.")

    # =====================================================================
    # Section C: Risk Dashboard
    # =====================================================================
    with st.expander("Risk Dashboard", expanded=len(monitor.positions) > 0):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Equity", f"${monitor.equity:,.0f}")
        with c2:
            dd = monitor.current_drawdown
            dd_color = "red" if dd > 0.10 else ("orange" if dd > 0.05 else "green")
            st.metric("Drawdown", f"{dd:.1%}")
            if dd > 0.10:
                st.error("Drawdown > 10%")
        with c3:
            exp_pct = monitor.exposure_pct(current_price) if current_price > 0 else 0
            st.metric("Exposure", f"{exp_pct:.0%}")
            if exp_pct > 0.60:
                st.warning("Exposure > 60%")
        with c4:
            st.metric("Positions", len(monitor.positions))

        # Unrealized P&L
        if monitor.positions and current_price > 0:
            unreal = monitor.unrealized_pnl(current_price)
            st.metric("Unrealized P&L", f"${unreal:+,.0f}")

        # Equity curve from closed trades
        if monitor.closed_trades:
            equity_points = [monitor.closed_trades[0].entry_price]  # placeholder
            running_equity = monitor.equity - sum(t.pnl for t in monitor.closed_trades)
            eq_data = [{'Trade': 0, 'Equity': running_equity}]
            for i, t in enumerate(monitor.closed_trades):
                running_equity += t.pnl
                eq_data.append({'Trade': i + 1, 'Equity': running_equity})
            eq_df = pd.DataFrame(eq_data)
            st.line_chart(eq_df.set_index('Trade')['Equity'])

    # =====================================================================
    # Section D: Signal History
    # =====================================================================
    with st.expander("Signal History (last 50)"):
        history = monitor.signal_history[-50:]
        if history:
            history_reversed = list(reversed(history))
            hist_data = []
            for entry in history_reversed:
                hist_data.append({
                    'Time': entry.get('time', '?')[:19],
                    'Strategy': entry.get('strategy', '?'),
                    'Dir': entry.get('direction', '?').upper(),
                    'Conf': f"{entry.get('confidence', 0):.0%}",
                    'Urgency': f"{entry.get('urgency', 0):.0%}",
                    'TF': entry.get('primary_tf', '?'),
                    'Regime': entry.get('regime', '?'),
                    'Acted': 'Yes' if entry.get('acted') else '',
                })
            st.dataframe(pd.DataFrame(hist_data), hide_index=True)
        else:
            st.info("No signal history yet.")

    # =====================================================================
    # Section E: Closed Trades
    # =====================================================================
    if monitor.closed_trades:
        with st.expander("Closed Trades"):
            trade_data = []
            for t in reversed(monitor.closed_trades):
                trade_data.append({
                    'Strategy': t.strategy,
                    'Dir': t.direction.upper(),
                    'Entry': f"${t.entry_price:.2f}",
                    'Exit': f"${t.exit_price:.2f}",
                    'Shares': t.shares,
                    'P&L ($)': f"${t.pnl:+,.0f}",
                    'P&L (%)': f"{t.pnl_pct:+.1%}",
                    'Reason': t.exit_reason,
                    'Hold': f"{t.hold_minutes:.0f}m",
                })
            st.dataframe(pd.DataFrame(trade_data), hide_index=True)


def main():
    st.title("X23 Channel Break Predictor")

    # Sidebar config
    st.sidebar.header("Configuration")

    data_dir = st.sidebar.text_input("Data Directory", value="data")
    models_dir = Path("models")
    model_files = sorted(models_dir.glob("*.pt")) if models_dir.exists() else []

    def _model_label(p: Path) -> str:
        """Format model path with file modification timestamp."""
        try:
            import datetime
            mtime = p.stat().st_mtime
            ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            return f"{p.name}  ({ts})"
        except OSError:
            return str(p)

    model_options = [str(f) for f in model_files]
    model_labels = [_model_label(f) for f in model_files]

    if model_options:
        selected_idx = st.sidebar.selectbox(
            "Model Checkpoint",
            options=range(len(model_options)),
            format_func=lambda i: model_labels[i],
            index=0,
        )
        checkpoint_path = model_options[selected_idx]
    else:
        checkpoint_path = st.sidebar.text_input(
            "Model Checkpoint",
            value=DEFAULT_MODEL_PATH
        )
        st.sidebar.warning("No .pt files found in models/")

    # Live data sidebar configuration
    live_config = show_live_data_sidebar()

    # Load data from files (used as fallback; empty DFs if no local data)
    tsla = spy = vix = pd.DataFrame()
    if Path(data_dir).exists():
        with st.spinner("Loading market data..."):
            try:
                tsla, spy, vix = load_market_data(data_dir)
                st.sidebar.success(f"Loaded {len(tsla):,} bars")
            except Exception as e:
                st.sidebar.warning(f"CSV data not loaded: {e}")
    else:
        st.sidebar.info("No local data — using live yfinance data")

    # Force re-download button
    if st.sidebar.button("Re-download model from GitHub"):
        p = Path(checkpoint_path)
        if p.exists():
            p.unlink()
        cal_path = p.parent / CALIBRATION_ASSET_NAME
        if cal_path.exists():
            cal_path.unlink()
        load_predictor.clear()
        fetch_all_market_data.clear()  # Clear unified data cache
        _fetch_live_5min.clear()  # Clear legacy 5-min cache
        st.rerun()

    # Load model (auto-download from GitHub Releases if not present)
    predictor = None
    with st.spinner("Loading model..."):
        try:
            checkpoint_path, asset_info = _ensure_checkpoint(checkpoint_path)
            predictor = load_predictor(checkpoint_path)

            # Log checkpoint details to terminal
            cp = Path(checkpoint_path)
            import torch as _torch
            _ckpt = _torch.load(cp, map_location='cpu', weights_only=False)
            _epoch = _ckpt.get('epoch', '?')
            _mae = _ckpt.get('best_per_tf_mae', '?')
            if isinstance(_mae, float):
                _mae = f"{_mae:.3f}"
            _cal_path = cp.parent / CALIBRATION_ASSET_NAME
            if _cal_path.exists():
                import json as _json
                _cal = _json.loads(_cal_path.read_text())
                _temp = f"T={_cal['temperature']:.4f}, ECE={_cal['ece_after']:.4f}"
            else:
                _temp = "none"
            print(f"[MODEL] Loaded: {cp.name} | epoch {_epoch} | per-TF MAE {_mae} | calibration: {_temp}")
            del _ckpt

            if asset_info:
                asset_id = str(asset_info['id'])
                uploaded = asset_info['uploaded_at'][:10] if asset_info['uploaded_at'] else '?'
                st.sidebar.success(f"Model loaded — {asset_info['name']} (RA_...{asset_id[-4:]}, uploaded {uploaded})")
            else:
                st.sidebar.success("Model loaded")
        except Exception as e:
            st.sidebar.warning(f"Model not loaded: {e}")
            logger.exception("Model loading error")

    # Show C++ feature extraction status
    cpp_status = get_cpp_status()
    if cpp_status['available']:
        st.sidebar.success(
            f"C++ features: {cpp_status['feature_count']:,}/{cpp_status['expected']:,}"
        )
    else:
        st.sidebar.error(
            "C++ features: UNAVAILABLE — predictions will fail. "
            "Build with: pip install -e ."
        )

    # Unified data fetch: native TFs + 5-min in one call
    native_tf_data = None
    live_5min = None
    if YFINANCE_AVAILABLE:
        with st.spinner("Fetching market data..."):
            native_tf_data, live_5min = fetch_all_market_data()
            if native_tf_data:
                st.session_state['native_tf_data'] = native_tf_data
                if predictor is not None:
                    predictor.load_native_tf_data(native_tf_data)

    # Get data (live or loaded), passing pre-fetched 5-min data
    current_tsla, current_spy, current_vix, is_live, error_msg = get_live_data_with_fallback(
        use_live=live_config['use_live'],
        loaded_data=(tsla, spy, vix),
        live_5min=live_5min,
        lookback=35000
    )

    # Track last update time
    if 'last_update' not in st.session_state:
        st.session_state['last_update'] = datetime.now()
    if is_live:
        st.session_state['last_update'] = datetime.now()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Live Prediction",
        "Channel Visualization",
        "Window Selection",
        "Feature Analysis",
        "Model Info",
        "Data Explorer",
        "Trading Monitor",
    ])

    # Validate native TF data completeness (shared across tabs)
    missing_tfs = []
    if native_tf_data is None:
        missing_tfs = NATIVE_TF_LIST.copy()
    else:
        required_symbols = ['TSLA', 'SPY', '^VIX']
        for tf in NATIVE_TF_LIST:
            for sym in required_symbols:
                sym_data = native_tf_data.get(sym, {})
                tf_df = sym_data.get(tf)
                if tf_df is None or len(tf_df) < 10:
                    missing_tfs.append(f"{sym} {tf}")

    with tab1:
        st.header("Live Prediction")

        # Show data status
        show_data_status_bar(is_live, error_msg, st.session_state.get('last_update'))

        if predictor is None:
            st.warning("Load a trained model to make predictions")
        else:
            st.caption(f"Available: {len(current_tsla):,} bars")

            if missing_tfs:
                st.error(
                    f"Cannot predict: missing data for {len(missing_tfs)} timeframe(s). "
                    f"Missing: {', '.join(missing_tfs)}"
                )
                st.info("Refresh the page to re-fetch. Rate-limited fetches may need a few minutes.")

            if st.button("Make Prediction", type="primary", disabled=bool(missing_tfs)):
                with st.spinner("Loading data and predicting..."):
                    try:
                        # Use all available data (LivePredictor trims internally)
                        tsla_slice = current_tsla
                        spy_slice = current_spy
                        vix_slice = current_vix

                        print(f"[DASHBOARD] Prediction requested: "
                              f"available TSLA={len(current_tsla):,} bars")
                        if len(tsla_slice) > 0:
                            print(f"[DASHBOARD] Slice range: {tsla_slice.index[0]} to {tsla_slice.index[-1]}")

                        # Load data into LivePredictor (updates rolling window + bar count)
                        predictor.load_historical_data(tsla_slice, spy_slice, vix_slice)

                        # Use predict_with_per_tf for per-TF breakdown + channel history
                        live_pred = predictor.predict_with_per_tf()

                        if live_pred is None:
                            bar_count = len(predictor.tsla_data) if predictor.tsla_data is not None else 0
                            st.error(
                                f"Cannot predict: need at least {predictor.min_bars:,} bars "
                                f"(have {bar_count:,})"
                            )
                        else:
                            prediction = live_pred.prediction

                            # Store prediction for channel visualization tab
                            st.session_state['last_prediction'] = prediction
                            st.session_state['prediction_data'] = (tsla_slice, spy_slice, vix_slice)
                            st.session_state['last_live_pred'] = live_pred

                            st.success(
                                f"Prediction complete! "
                                f"(latency: {live_pred.latency_ms:.0f}ms, "
                                f"bars: {live_pred.source_bar_count:,})"
                            )

                            # Signal strategy selection
                            signal_strategy_map = {
                                'Most Confident TF': SignalStrategy.MOST_CONFIDENT,
                                'Shortest Confident TF (>70%)': SignalStrategy.SHORTEST_CONFIDENT,
                                'Consensus (7/10 TFs agree)': SignalStrategy.CONSENSUS,
                            }
                            selected_strategy_name = st.selectbox(
                                "Signal Selection Strategy",
                                options=list(signal_strategy_map.keys()),
                                index=0,
                                help="How to select which timeframe to use for trading signals"
                            )
                            selected_strategy = signal_strategy_map[selected_strategy_name]

                            # Recompute signal with selected strategy
                            if prediction.per_tf_predictions:
                                from v15.signals.bounce_signal import BounceSignalEngine
                                engine = BounceSignalEngine()
                                prediction.bounce_signal = engine.generate_signal(
                                    per_tf_predictions=prediction.per_tf_predictions,
                                    strategy=selected_strategy,
                                )

                            show_prediction_card(prediction)

                            # Bounce Signal Alert
                            st.divider()
                            st.subheader("Bounce Signal")
                            if prediction.bounce_signal and prediction.bounce_signal.actionable:
                                signal = prediction.bounce_signal

                                if signal.signal_type.value == 'buy':
                                    st.success(f"🟢 BUY SIGNAL - {signal.strength}")
                                elif signal.signal_type.value == 'sell':
                                    st.error(f"🔴 SELL SIGNAL - {signal.strength}")

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Primary TF", signal.primary_tf)
                                with col2:
                                    st.metric("Confidence", f"{signal.primary_confidence:.0%}")
                                with col3:
                                    st.metric("Time to Breach", f"{signal.time_to_breach_bars:.1f} bars")
                                with col4:
                                    st.metric("Boundary", signal.trigger_boundary.upper())

                                # Risk warnings
                                if signal.risk_warnings:
                                    st.warning("⚠️ " + " | ".join(signal.risk_warnings))

                                # Expandable per-TF breakdown
                                with st.expander("📊 Signal Breakdown by Timeframe"):
                                    signal_data = []
                                    for tf in TIMEFRAMES:
                                        if tf in signal.per_tf_scores:
                                            score = signal.per_tf_scores[tf]
                                            pred = prediction.per_tf_predictions[tf]
                                            signal_data.append({
                                                'TF': tf,
                                                'Signal Score': f"{score:.0%}",
                                                'Direction': pred.direction.upper(),
                                                'Confidence': f"{pred.confidence:.0%}",
                                                'Duration': f"{pred.duration_mean:.1f}±{pred.duration_std:.1f}",
                                                '_score': score,  # hidden, for styling
                                            })

                                    signal_df = pd.DataFrame(signal_data)

                                    def style_signal_row(row):
                                        score = row.get('_score', 0.0)
                                        if score >= 0.70:
                                            return ['background-color: #d4edda'] * len(row)  # Green
                                        elif score >= 0.60:
                                            return ['background-color: #fff3cd'] * len(row)  # Yellow
                                        else:
                                            return [''] * len(row)

                                    display_df = signal_df.drop(columns=['_score'])
                                    styled = signal_df.style.apply(style_signal_row, axis=1)
                                    styled = styled.hide(subset=['_score'], axis='columns')
                                    st.dataframe(styled, hide_index=True)
                            elif prediction.bounce_signal:
                                st.info(f"ℹ️ {selected_strategy_name}: Signal not actionable (confidence={prediction.bounce_signal.primary_confidence:.0%})")

                            # === REGIME-ADAPTIVE TRADING ENGINE (c3) ===
                            st.divider()
                            st.subheader("Regime-Adaptive Trading Engine")
                            try:
                                from v15.trading.signals import RegimeAdaptiveSignalEngine, SignalType as RASignalType
                                from v15.trading.position_sizer import PositionSizer
                                from v15.trading.meta_strategy import MetaStrategy

                                if prediction.per_tf_predictions:
                                    ra_engine = RegimeAdaptiveSignalEngine()
                                    ra_signal = ra_engine.generate_signal(prediction.per_tf_predictions)

                                    # Regime indicator
                                    regime = ra_signal.regime
                                    regime_colors = {
                                        'trending_bull': ('TRENDING BULL', '#28a745'),
                                        'trending_bear': ('TRENDING BEAR', '#dc3545'),
                                        'ranging': ('RANGING', '#ffc107'),
                                        'transitioning': ('TRANSITIONING', '#6c757d'),
                                    }
                                    regime_label, regime_color = regime_colors.get(
                                        regime.regime.value, ('UNKNOWN', '#999')
                                    )
                                    st.markdown(
                                        f"<div style='background-color:{regime_color};color:white;padding:10px;border-radius:5px;text-align:center;font-size:1.2em;font-weight:bold'>"
                                        f"{regime_label} (conf: {regime.confidence:.0%})</div>",
                                        unsafe_allow_html=True
                                    )

                                    # Signal
                                    if ra_signal.signal_type == RASignalType.LONG:
                                        st.success(f"LONG - {ra_signal.strength} (conf: {ra_signal.confidence:.0%})")
                                    elif ra_signal.signal_type == RASignalType.SHORT:
                                        st.error(f"SHORT - {ra_signal.strength} (conf: {ra_signal.confidence:.0%})")
                                    else:
                                        st.info(f"FLAT - No trade ({ra_signal.confidence:.0%})")

                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Primary TF", ra_signal.primary_tf)
                                    with col2:
                                        st.metric("Edge Est.", f"{ra_signal.edge_estimate:.2%}")
                                    with col3:
                                        st.metric("Entry Urgency", f"{ra_signal.entry_urgency:.0%}")
                                    with col4:
                                        hazard_pct = ra_signal.hazard.aggregate_hazard
                                        st.metric("Break Risk", f"{hazard_pct:.0%}")

                                    # Position sizing
                                    if ra_signal.actionable:
                                        current_price = float(current_tsla.iloc[-1]['close']) if 'close' in current_tsla.columns else float(current_tsla.iloc[-1]['Close'])
                                        sizer = PositionSizer(capital=100000)
                                        pos = sizer.size_position(ra_signal, current_price)
                                        if pos.should_trade:
                                            st.markdown(
                                                f"**Position:** {pos.shares} shares (${pos.dollar_amount:,.0f}) | "
                                                f"SL: {pos.stop_loss_pct:.1%} | TP: {pos.take_profit_pct:.1%} | "
                                                f"R:R {pos.risk_reward_ratio:.1f}"
                                            )

                                    # Warnings
                                    if ra_signal.risk_warnings:
                                        st.warning(" | ".join(ra_signal.risk_warnings))

                                    # Regime detail
                                    with st.expander("Regime & Strategy Detail"):
                                        st.markdown(f"**Bull:** {regime.bull_score:.0%} | **Bear:** {regime.bear_score:.0%} | **Sideways:** {regime.sideways_score:.0%}")
                                        st.markdown(f"**TF Agreement:** {regime.tf_agreement:.0%} | **Dominant Horizon:** {regime.dominant_horizon}")
                                        st.markdown(f"**Hazard Velocity:** {ra_signal.hazard.hazard_velocity:+.3f}")

                                        # Meta-strategy
                                        meta = MetaStrategy()
                                        best_meta, all_meta = meta.evaluate(
                                            prediction.per_tf_predictions, regime, ra_signal.hazard
                                        )
                                        meta_data = []
                                        for name, sig in all_meta.items():
                                            meta_data.append({
                                                'Strategy': name,
                                                'Signal': sig.signal_type.value.upper(),
                                                'Confidence': f"{sig.confidence:.0%}",
                                                'Edge': f"{sig.edge_estimate:.2%}",
                                            })
                                        st.dataframe(pd.DataFrame(meta_data), hide_index=True)
                                        st.markdown(f"**Meta Winner:** {best_meta.name} ({best_meta.signal_type.value})")
                                        st.markdown(f"**Reasoning:** {best_meta.reasoning}")
                                else:
                                    st.info("Per-TF predictions required for regime engine")
                            except ImportError as e:
                                st.warning(f"Trading engine not available: {e}")
                            except Exception as e:
                                st.error(f"Trading engine error: {e}")

                            # Daily channel chart with aggregate projection
                            st.divider()
                            st.subheader("Daily Channel")
                            try:
                                from v15.features.tf_extractor import resample_to_timeframe

                                # Get daily data (prefer native TF)
                                daily_df = None
                                if (native_tf_data
                                        and native_tf_data.get('TSLA', {}).get('daily') is not None
                                        and len(native_tf_data['TSLA']['daily']) >= 20):
                                    daily_df = native_tf_data['TSLA']['daily'].copy()
                                elif len(tsla_slice) >= 20:
                                    daily_df = resample_to_timeframe(tsla_slice, 'daily')

                                if daily_df is not None and len(daily_df) >= 20:
                                    # Detect channel
                                    daily_channels = detect_channels_multi_window(
                                        daily_df, windows=STANDARD_WINDOWS
                                    )
                                    daily_best, daily_best_window = select_best_channel(daily_channels)

                                    # Use prediction's window if available
                                    daily_window = 50
                                    if (prediction.per_tf_predictions is not None
                                            and 'daily' in prediction.per_tf_predictions):
                                        daily_window = prediction.per_tf_predictions['daily'].best_window
                                    elif daily_best_window is not None:
                                        daily_window = daily_best_window

                                    daily_channel = daily_channels.get(daily_window)
                                    window_bars = min(daily_window, len(daily_df) - 1)
                                    daily_chart_df = daily_df.iloc[-(window_bars + 1):].copy()

                                    if daily_channel is not None and getattr(daily_channel, 'valid', False):
                                        fig = create_tf_channel_chart(
                                            df=daily_chart_df,
                                            channel=daily_channel,
                                            tf_name='daily',
                                            duration=prediction.duration_mean,
                                            confidence=prediction.confidence,
                                            show_bounces=True,
                                            height=400
                                        )

                                        if prediction.duration_mean > 0 and prediction.confidence > 0:
                                            fig = add_duration_projection(
                                                fig, daily_channel, daily_chart_df,
                                                prediction.duration_mean,
                                                prediction.duration_std,
                                                prediction.confidence,
                                                prediction.direction,
                                                tf_name='daily',
                                            )

                                        st.plotly_chart(fig, width="stretch")

                                        # Metrics row
                                        c1, c2, c3, c4 = st.columns(4)
                                        with c1:
                                            st.metric("Bounces", getattr(daily_channel, 'bounce_count', 0))
                                        with c2:
                                            st.metric("R-squared", f"{getattr(daily_channel, 'r_squared', 0.0):.3f}")
                                        with c3:
                                            dir_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
                                            ch_dir = getattr(daily_channel, 'direction', 1)
                                            st.metric("Channel Dir", dir_names.get(int(ch_dir), 'Unknown'))
                                        with c4:
                                            st.metric("Window", f"{daily_window} bars")
                                    else:
                                        fig = create_candlestick_chart(daily_chart_df, tf_name='daily')
                                        fig.update_layout(title="Daily - No Valid Channel", height=400)
                                        st.plotly_chart(fig, width="stretch")
                                else:
                                    st.info("Insufficient daily data for channel detection")
                            except Exception as e:
                                st.warning(f"Could not render daily channel: {e}")
                                logger.exception("Daily channel chart error")

                            # Show learned window selection if model uses it
                            if prediction.used_learned_selection:
                                st.info(
                                    f"Model selected window {prediction.learned_window} "
                                    f"(learned selection)"
                                )

                            # Show channel history status
                            show_channel_history_status(predictor)

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

                            # Show bar completion by TF
                            with st.expander("Bar Completion by Timeframe"):
                                completion = live_pred.bar_completion_by_tf
                                if completion:
                                    comp_data = [
                                        {'Timeframe': tf, 'Completion': f"{pct:.0%}"}
                                        for tf, pct in completion.items()
                                    ]
                                    st.dataframe(
                                        pd.DataFrame(comp_data),
                                        width="stretch",
                                        hide_index=True
                                    )

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        logger.exception("Prediction error")

    with tab2:
        # Channel Visualization tab
        prediction = st.session_state.get('last_prediction')
        pred_data = st.session_state.get('prediction_data')

        native_tf = st.session_state.get('native_tf_data')
        if pred_data is not None:
            tsla_slice, _, _ = pred_data
            show_channel_visualization_tab(tsla_slice, prediction, native_bars_by_tf=native_tf)
        else:
            show_channel_visualization_tab(current_tsla, None, native_bars_by_tf=native_tf)

    with tab3:
        st.header("Window Selection Analysis")

        st.markdown("""
        This tab shows how window selection works across all 8 standard windows (10, 20, 30, 40, 50, 60, 70, 80 bars).
        Different selection strategies may pick different "best" windows based on channel quality metrics.
        """)

        # Window selection controls
        window_lookback = st.slider(
            "Analysis Lookback (bars)",
            min_value=500,
            max_value=20000,
            value=5000,
            key="window_lookback"
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
                selection_mode="heuristic"
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
                st.plotly_chart(fig, width="stretch")

            with col2:
                st.subheader("Strategy Selection Comparison")
                fig = create_strategy_comparison_chart(analysis)
                st.plotly_chart(fig, width="stretch")

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

            # LivePredictor stats
            st.divider()
            st.subheader("Live Predictor Status")

            stats = predictor.get_stats()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predictions Made", stats['prediction_count'])
                st.metric("Data Bars", f"{stats['data_bars']:,}")
            with col2:
                st.metric("Total Bars Received", f"{stats['total_bars_received']:,}")
                st.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")
            with col3:
                st.metric("Channel History", "Active" if stats['tracking_channel_history'] else "Disabled")
                st.metric("Can Predict", "Yes" if stats['can_predict'] else "No")

            # Channel history detail
            if hasattr(predictor, 'channel_history_by_tf'):
                with st.expander("Channel History Detail"):
                    history = predictor.channel_history_by_tf
                    hist_data = []
                    for tf_name in TIMEFRAMES:
                        tf_hist = history.get(tf_name, {})
                        tsla_count = len(tf_hist.get('tsla', []))
                        spy_count = len(tf_hist.get('spy', []))
                        hist_data.append({
                            'Timeframe': tf_name,
                            'TSLA Entries': tsla_count,
                            'SPY Entries': spy_count,
                            'Total': tsla_count + spy_count,
                        })
                    st.dataframe(
                        pd.DataFrame(hist_data),
                        width="stretch",
                        hide_index=True
                    )

        else:
            st.info("Load a model to see information")

    with tab6:
        st.header("Data Explorer")

        # Show data status
        show_data_status_bar(is_live, error_msg, st.session_state.get('last_update'))

        st.subheader("TSLA")
        if len(current_tsla) > 0 and 'close' in current_tsla.columns:
            st.line_chart(current_tsla['close'].tail(1000))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Bars", f"{len(current_tsla):,}")
                st.metric("Start", str(current_tsla.index[0].date()))
            with col2:
                st.metric("End", str(current_tsla.index[-1].date()))
                st.metric("Last Close", f"${current_tsla['close'].iloc[-1]:.2f}")
        else:
            st.info("No TSLA data available")

    with tab7:
        show_trading_monitor_tab(
            predictor=predictor,
            current_tsla=current_tsla,
            current_spy=current_spy,
            current_vix=current_vix,
            native_tf_data=native_tf_data,
            live_config=live_config,
            is_live=is_live,
            missing_tfs=missing_tfs,
        )


if __name__ == "__main__":
    main()
