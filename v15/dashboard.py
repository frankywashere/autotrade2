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

# Import native TF data loader for per-TF yfinance fetching
from v15.data.native_tf import load_native_tf_data as _load_native_tf_data

# Higher TFs that benefit from native yfinance fetching
# (daily/weekly/monthly have unlimited history vs 60d for 5-min)
NATIVE_TF_LIST = ['1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly']

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
            data.append({
                'Horizon': horizon.title(),
                'Timeframe': tf_name,
                'Direction': tf_pred.direction.upper(),
                'Dir Prob': f"{tf_pred.direction_prob:.0%}",
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
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


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
        tf_pred = None

        if prediction is not None and prediction.per_tf_predictions is not None:
            if tf_name in prediction.per_tf_predictions:
                tf_pred = prediction.per_tf_predictions[tf_name]
                duration = tf_pred.duration_mean
                confidence = tf_pred.confidence
                tf_window = tf_pred.best_window
                tf_direction = tf_pred.direction

        # Create expander with summary in header
        dir_str = tf_direction.upper() if tf_direction else '?'
        expander_header = f"{tf_name} - {dir_str} - Duration: {duration:.0f} bars - Conf: {confidence:.0%}"

        with st.expander(expander_header, expanded=False):
            try:
                # Use native TF data if available for this timeframe
                if (native_bars_by_tf
                        and native_bars_by_tf.get('TSLA', {}).get(tf_name) is not None
                        and len(native_bars_by_tf['TSLA'][tf_name]) >= 20):
                    tf_df = native_bars_by_tf['TSLA'][tf_name].copy()
                else:
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

                # Slice data: channel window + current bar for display.
                # detect_channel uses df.iloc[-(window+1):-1] internally,
                # so the first window_bars of chart_df align with the channel.
                # The extra bar at the end is the current bar (shown as a
                # candle but outside the channel overlay).
                window_bars = min(tf_window, len(tf_df) - 1)
                chart_df = tf_df.iloc[-(window_bars + 1):].copy()

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
                    fig = create_candlestick_chart(chart_df, tf_name=tf_name)
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
    Get market data, merging CSV history with fresh yfinance data.

    When use_live=True, fetches recent data from yfinance and appends it
    to the CSV historical data. This gives us years of history plus
    current market data — both are needed for all 10 timeframes.

    Args:
        use_live: Whether to attempt live data fetch
        loaded_data: Tuple of (tsla, spy, vix) loaded from CSV files
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
            print("[DATA] Fetching live data from yfinance...")
            data_feed = YFinanceLiveData(cache_ttl=60)
            tsla_live, spy_live, vix_live = data_feed.get_historical(
                period='60d',
                interval='5m'
            )
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
    page_title="V15 Channel Predictor",
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
DEFAULT_MODEL_PATH = "models/best.pt"


def _ensure_checkpoint(path: str = DEFAULT_MODEL_PATH) -> tuple:
    """Download checkpoint from GitHub Releases if not present locally.

    Returns:
        (path, asset_info) where asset_info is a dict with 'id', 'name',
        'uploaded_at' from GitHub, or None if loaded from local file.
    """
    p = Path(path)
    if p.exists():
        return str(p), None

    p.parent.mkdir(parents=True, exist_ok=True)

    # For private repos, use token from Streamlit secrets or env
    import os
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        try:
            token = st.secrets["GITHUB_TOKEN"]
        except (KeyError, AttributeError):
            token = ""

    import urllib.request
    import json as _json

    # Fetch release by tag to get the latest asset URL (no hardcoded asset ID)
    release_url = f"https://api.github.com/repos/{CHECKPOINT_REPO}/releases/tags/{CHECKPOINT_RELEASE_TAG}"
    req = urllib.request.Request(release_url)
    if token:
        req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")

    print(f"[MODEL] Fetching release info from {CHECKPOINT_RELEASE_TAG} ...")
    with urllib.request.urlopen(req) as resp:
        release = _json.loads(resp.read())

    assets = release.get("assets", [])
    if not assets:
        raise RuntimeError(f"No assets found in release {CHECKPOINT_RELEASE_TAG}")

    asset = assets[0]
    asset_url = asset["url"]
    print(f"[MODEL] Downloading {asset['name']} ({asset['size'] / 1e6:.1f} MB) ...")

    # Download the actual asset binary
    req = urllib.request.Request(asset_url)
    if token:
        req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/octet-stream")

    with urllib.request.urlopen(req) as resp, open(p, "wb") as f:
        import shutil
        shutil.copyfileobj(resp, f)
    print(f"[MODEL] Downloaded {p.stat().st_size / 1e6:.1f} MB")

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
    return LivePredictor(checkpoint_path, track_channel_history=True)


@st.cache_data(ttl=300)
def load_native_tf():
    """Fetch and cache native TF data from yfinance for higher timeframes."""
    print("[DATA] Fetching native TF data (daily/weekly/monthly) from yfinance...")
    try:
        data = _load_native_tf_data(
            symbols=['TSLA', 'SPY', '^VIX'],
            timeframes=NATIVE_TF_LIST,
            start_date='2015-01-01',
            use_cache=True,
            cache_max_age_hours=5 / 60,  # ~5 min, matches Streamlit TTL
            verbose=True,
        )
        for symbol in data:
            for tf in data[symbol]:
                df = data[symbol][tf]
                print(f"[DATA] Native TF: {symbol} {tf} = {len(df)} bars")
        return data
    except Exception as e:
        print(f"[DATA] Native TF fetch failed: {e}")
        logger.exception("Native TF data fetch failed")
        return None


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
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Direction",
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
            value="models/best.pt"
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
        load_predictor.clear()
        st.rerun()

    # Load model (auto-download from GitHub Releases if not present)
    predictor = None
    with st.spinner("Loading model..."):
        try:
            checkpoint_path, asset_info = _ensure_checkpoint(checkpoint_path)
            predictor = load_predictor(checkpoint_path)
            if asset_info:
                asset_id = str(asset_info['id'])
                uploaded = asset_info['uploaded_at'][:10] if asset_info['uploaded_at'] else '?'
                st.sidebar.success(f"Model loaded — {asset_info['name']} (RA_...{asset_id[-4:]}, uploaded {uploaded})")
            else:
                st.sidebar.success("Model loaded")
        except Exception as e:
            st.sidebar.warning(f"Model not loaded: {e}")
            logger.exception("Model loading error")

    # Fetch native TF data for higher timeframes (daily/weekly/monthly)
    # These have unlimited yfinance history vs 60-day limit for 5-min
    native_tf_data = None
    if YFINANCE_AVAILABLE:
        with st.spinner("Fetching native TF data..."):
            native_tf_data = load_native_tf()
            if native_tf_data:
                st.session_state['native_tf_data'] = native_tf_data
                if predictor is not None:
                    predictor.load_native_tf_data(native_tf_data)

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
            st.caption(f"Available: {len(current_tsla):,} bars")

            if st.button("Make Prediction", type="primary"):
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
                            show_prediction_card(prediction)

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

                                        st.plotly_chart(fig, use_container_width=True)

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
                                        st.plotly_chart(fig, use_container_width=True)
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
                                        use_container_width=True,
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
                        use_container_width=True,
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


if __name__ == "__main__":
    main()
