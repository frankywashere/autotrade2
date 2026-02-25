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
    YFINANCE_AVAILABLE,
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

# Import Channel Surfer
try:
    from v15.core.channel_surfer import (
        prepare_multi_tf_analysis,
        ChannelAnalysis,
        SIGNAL_TFS,
        ZONE_OVERSOLD,
        ZONE_LOWER,
        ZONE_UPPER,
        ZONE_OVERBOUGHT,
    )
    CHANNEL_SURFER_AVAILABLE = True
except ImportError:
    CHANNEL_SURFER_AVAILABLE = False
    logger.warning("Channel Surfer not available")

# Try to import streamlit-autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False
    logger.info("streamlit-autorefresh not available - using manual refresh")

# Channel Surfer live scanner
try:
    from v15.trading.surfer_live_scanner import SurferLiveScanner, ScannerConfig, ScannerAlert
    SURFER_SCANNER_AVAILABLE = True
except ImportError:
    SURFER_SCANNER_AVAILABLE = False
    logger.warning("SurferLiveScanner not available")


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

def _normalize_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame's DatetimeIndex to UTC.

    CSV data is tz-naive (America/New_York market time).
    yfinance data is tz-aware (America/New_York).
    Both get converted to UTC.
    """
    if len(df) == 0 or not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize(
            "America/New_York",
            ambiguous="infer",
            nonexistent="shift_forward"
        ).tz_convert("UTC")
    else:
        tz_str = str(df.index.tz)
        if tz_str != "UTC":
            df = df.copy()
            df.index = df.index.tz_convert("UTC")
    return df


def _normalize_tuple_to_utc(
    data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize a (tsla, spy, vix) tuple to UTC."""
    return tuple(_normalize_to_utc(df) for df in data)


def _append_bars(
    old: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    new: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Concat old + new 5-min data, dedup on index (newer wins)."""
    result = []
    for o, n in zip(old, new):
        merged = pd.concat([o, n])
        merged = merged[~merged.index.duplicated(keep='last')]
        merged = merged.sort_index()
        result.append(merged)
    return tuple(result)


def _refresh_5min_data(
    force_full: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Session-state accumulator for 5-min data.

    Cold start (or force_full): fetch full 60d of 5-min data.
    Subsequent refreshes: fetch only period='1d' and append/update.

    Returns:
        Tuple of (tsla, spy, vix) DataFrames.
    """
    cached = st.session_state.get('_5min_data')
    is_cold = cached is None or force_full

    # Check staleness: full re-fetch if data is >2 hours old
    if cached is not None and not force_full:
        age_s = (datetime.now() - cached['ts']).total_seconds()
        if age_s > 7200:  # 2 hours
            print(f"[DATA] 5-min data is {age_s/60:.0f}min old — forcing full re-fetch")
            is_cold = True

    if is_cold:
        print("[DATA] 5-min FULL fetch (cold start / forced)...")
        data_feed = YFinanceLiveData(cache_ttl=5)
        data = _normalize_tuple_to_utc(
            data_feed.get_historical(period='60d', interval='5m')
        )
        st.session_state['_5min_data'] = {'data': data, 'ts': datetime.now()}
        print(f"[DATA] 5-min full fetch: {len(data[0]):,} TSLA bars (UTC)")
        return data
    else:
        print("[DATA] 5-min INCREMENTAL fetch (period='1d')...")
        data_feed = YFinanceLiveData(cache_ttl=5)
        delta = _normalize_tuple_to_utc(
            data_feed.get_historical(period='1d', interval='5m')
        )
        merged = _append_bars(cached['data'], delta)
        st.session_state['_5min_data'] = {'data': merged, 'ts': datetime.now()}
        print(f"[DATA] 5-min incremental: +{len(delta[0])} new/updated bars, "
              f"total={len(merged[0]):,}")
        return merged


@st.cache_data(ttl=55)
def _get_realtime_prices() -> Dict[str, Optional[float]]:
    """Get real-time prices via yfinance ticker.info.

    Works in all sessions (market, premarket, afterhours).
    Includes backoff: after 3 consecutive failures (rate-limited),
    stops trying for 5 minutes to avoid burning API budget.
    """
    import time as _time
    backoff = st.session_state.get('_rt_price_backoff', {})
    fails = backoff.get('consecutive_fails', 0)
    last_fail = backoff.get('last_fail_ts', 0)
    if fails >= 3:
        elapsed = _time.time() - last_fail
        if elapsed < 300:  # 5-minute cooldown
            print(f"[PRICE] ticker.info backed off ({fails} fails, {300-elapsed:.0f}s remaining)")
            return st.session_state.get('_rt_price_last', {})
        else:
            print(f"[PRICE] ticker.info backoff expired, retrying...")

    try:
        data_feed = YFinanceLiveData(cache_ttl=55)
        prices = data_feed.get_realtime_prices()
        all_none = all(v is None for v in prices.values())
        if all_none:
            st.session_state['_rt_price_backoff'] = {
                'consecutive_fails': fails + 1,
                'last_fail_ts': _time.time(),
            }
            print(f"[PRICE] ticker.info all None — rate-limited (fail #{fails+1})")
        else:
            st.session_state['_rt_price_backoff'] = {'consecutive_fails': 0, 'last_fail_ts': 0}
            print(f"[PRICE] ticker.info: { {k: f'${v:.2f}' if v else 'None' for k, v in prices.items()} }")
        st.session_state['_rt_price_last'] = prices
        return prices
    except Exception as e:
        st.session_state['_rt_price_backoff'] = {
            'consecutive_fails': fails + 1,
            'last_fail_ts': _time.time(),
        }
        print(f"[PRICE] ticker.info FAILED: {type(e).__name__}: {e} (fail #{fails+1})")
        return st.session_state.get('_rt_price_last', {})


def _resample_5min_to_tf(
    df_5min: pd.DataFrame,
    tf: str,
) -> pd.DataFrame:
    """Resample 5-min bars to a single current-period OHLCV bar for `tf`.

    Used to keep native TF data fresh between yfinance re-fetches by
    locally aggregating the 5-min data we already have.
    """
    if df_5min.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    now = datetime.now()

    # Select 5-min bars for the CURRENT period of this TF
    if tf == 'monthly':
        mask = (df_5min.index.year == now.year) & (df_5min.index.month == now.month)
    elif tf == 'weekly':
        # ISO week: Monday=0
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        mask = df_5min.index >= pd.Timestamp(week_start, tz=getattr(df_5min.index, 'tz', None))
    elif tf == 'daily':
        mask = df_5min.index.date == now.date()
    elif tf in ('1h', '2h', '3h', '4h'):
        hours = int(tf.replace('h', ''))
        current_hour_block = (now.hour // hours) * hours
        period_start = now.replace(hour=current_hour_block, minute=0, second=0, microsecond=0)
        mask = df_5min.index >= pd.Timestamp(period_start, tz=getattr(df_5min.index, 'tz', None))
    else:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    period_bars = df_5min[mask]
    if period_bars.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    # Aggregate to single bar
    agg = pd.DataFrame([{
        'open': period_bars['open'].iloc[0],
        'high': period_bars['high'].max(),
        'low': period_bars['low'].min(),
        'close': period_bars['close'].iloc[-1],
        'volume': period_bars['volume'].sum(),
    }], index=[period_bars.index[0]])
    agg.index.name = df_5min.index.name
    return agg


def _detect_tf_boundary_crossings(last_ts: datetime) -> List[str]:
    """Return list of TFs whose boundary has been crossed since `last_ts`."""
    now = datetime.now()
    crossed = []
    if now.date() != last_ts.date():
        crossed.extend(['daily', 'monthly'])
    if now.isocalendar()[1] != last_ts.isocalendar()[1]:
        crossed.append('weekly')
    if now.hour != last_ts.hour:
        crossed.append('1h')
    # 2h/3h/4h derived from 1h, so if 1h boundary crossed, they might be too
    if '1h' in crossed:
        for multi_h in ['2h', '3h', '4h']:
            h = int(multi_h.replace('h', ''))
            if (now.hour // h) != (last_ts.hour // h):
                crossed.append(multi_h)
    return crossed


def _update_native_tf_current_bar(
    native_data: Dict,
    live_5min: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> Dict:
    """Update the last bar of each native TF with local resample from 5-min.

    Modifies native_data in-place and returns it.
    """
    symbol_map = {'TSLA': 0, 'SPY': 1, '^VIX': 2}
    for symbol, idx in symbol_map.items():
        df_5m = live_5min[idx]
        if df_5m.empty:
            continue
        for tf in NATIVE_TF_LIST:
            tf_df = native_data.get(symbol, {}).get(tf)
            if tf_df is None or tf_df.empty:
                continue
            resampled = _resample_5min_to_tf(df_5m, tf)
            if resampled.empty:
                continue
            # Replace/append the current-period bar
            # If the resampled bar's timestamp matches the last bar in the
            # native data, update it; otherwise append it as a new bar.
            last_native_ts = tf_df.index[-1]
            # Align resampled index tz to match native TF data
            native_tz = getattr(tf_df.index, 'tz', None)
            resamp_tz = getattr(resampled.index, 'tz', None)
            if native_tz is not None and resamp_tz is not None:
                # Both tz-aware: convert resampled to native tz
                if str(native_tz) != str(resamp_tz):
                    resampled.index = resampled.index.tz_convert(native_tz)
            elif native_tz is not None and resamp_tz is None:
                # Native tz-aware, resamp naive: localize
                resampled.index = resampled.index.tz_localize(native_tz)
            elif native_tz is None and resamp_tz is not None:
                # Native naive, resamp tz-aware: strip tz
                resampled.index = resampled.index.tz_convert(None)
            resamp_ts = resampled.index[0]

            if resamp_ts >= last_native_ts:
                # Drop the last bar and append the resampled one
                updated = pd.concat([tf_df.iloc[:-1], resampled])
                updated = updated[~updated.index.duplicated(keep='last')]
                native_data[symbol][tf] = updated
    return native_data


def fetch_all_market_data(force_full: bool = False):
    """Single source of truth: fetch ALL yfinance data (5-min + native TFs).

    Uses session-state accumulators so subsequent calls are incremental:
    - 5-min: full 60d on cold start, period='1d' on refreshes
    - Native TFs: full fetch on cold start or boundary crossing, local
      resample from 5-min between boundaries to keep current bar fresh

    Args:
        force_full: If True, bypass accumulators and re-fetch everything.

    Returns:
        Tuple of (native_data, live_5min) where:
        - native_data: dict from _load_native_tf_data() or None
        - live_5min: tuple of (tsla, spy, vix) DataFrames or None
    """
    native_data = None
    live_5min = None

    if not YFINANCE_AVAILABLE:
        return native_data, live_5min

    # 1. Native TFs (daily/weekly/monthly/1h-4h) — years of history
    #    Cold start: full yfinance fetch.
    #    Warm: skip yfinance, update current bar from 5-min resample.
    #    Boundary crossing: re-fetch only the crossed TFs from yfinance.
    cached_ntf = st.session_state.get('_native_tf_data')
    need_full_ntf = cached_ntf is None or force_full

    if not need_full_ntf:
        native_data = cached_ntf['data']
        age_s = (datetime.now() - cached_ntf['ts']).total_seconds()

        # Check for TF boundary crossings → selective re-fetch
        crossed = _detect_tf_boundary_crossings(cached_ntf['ts'])
        # Filter to TFs we actually fetch natively
        crossed_native = [tf for tf in crossed if tf in NATIVE_TF_LIST]

        if crossed_native:
            print(f"[DATA] TF boundary crossed: {crossed_native} — re-fetching those TFs")
            try:
                fresh = _load_native_tf_data(
                    symbols=['TSLA', 'SPY', '^VIX'],
                    timeframes=crossed_native,
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
                if fresh:
                    for symbol in ['TSLA', 'SPY', '^VIX']:
                        for tf in crossed_native:
                            fdf = fresh.get(symbol, {}).get(tf)
                            if fdf is not None and not fdf.empty:
                                native_data[symbol][tf] = fdf
                    print(f"[DATA] Updated {len(crossed_native)} TFs from yfinance")
            except Exception as e:
                print(f"[DATA] Boundary re-fetch failed: {e}")
        else:
            print(f"[DATA] Native TF data from session state ({age_s:.0f}s old, no boundary crossed)")
    else:
        try:
            print("[DATA] fetch_all_market_data: fetching native TFs (cold start)...")
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

    # 2. Fetch 5-min data via session-state accumulator (incremental after warm start)
    try:
        live_5min = _refresh_5min_data(force_full=force_full)
    except Exception as e:
        print(f"[DATA] 5-min yfinance failed: {e}")

    # 3. Update native TF current bars from 5-min resample
    if native_data and live_5min:
        try:
            native_data = _update_native_tf_current_bar(native_data, live_5min)
            print("[DATA] Updated native TF current bars from 5-min resample")
        except Exception as e:
            print(f"[DATA] Native TF current-bar update failed: {e}")

    # Persist to session state
    if native_data:
        st.session_state['_native_tf_data'] = {
            'data': native_data,
            'ts': datetime.now(),
        }

    return native_data, live_5min


def get_live_data_with_fallback(
    use_live: bool,
    loaded_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    live_5min: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None,
    lookback: int = 35000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, Optional[str]]:
    """
    Get market data, merging CSV history with fresh yfinance data.

    When use_live=True, uses provided live 5-min data (already UTC-normalized
    from _refresh_5min_data) and appends to CSV historical data.

    Caches the merged result in st.session_state['_merged_5min'] so
    subsequent reruns only re-merge the delta bars, not the full 35K.

    Args:
        use_live: Whether to attempt live data merge
        loaded_data: Tuple of (tsla, spy, vix) loaded from CSV files
        live_5min: Optional pre-fetched 5-min data (already UTC) from fetch_all_market_data()
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
            # Use pre-fetched 5-min data or fall back to accumulator
            if live_5min is not None:
                tsla_live, spy_live, vix_live = live_5min
                print(f"[DATA] Using pre-fetched 5-min data")
            else:
                print("[DATA] Fetching live data via accumulator...")
                tsla_live, spy_live, vix_live = _refresh_5min_data()
            print(f"[DATA] yfinance: TSLA={len(tsla_live):,} bars "
                  f"({tsla_live.index[0]} to {tsla_live.index[-1]}, "
                  f"tz={tsla_live.index.tz})")

            # 5-min data is already UTC (normalized in _refresh_5min_data).
            # CSV data only needs normalizing once on first merge.
            cached_merge = st.session_state.get('_merged_5min')
            if cached_merge is not None:
                # Incremental: use previously merged CSV base, just update live tail
                tsla_base, spy_base, vix_base = cached_merge['csv_utc']
            else:
                # First merge in this session — normalize CSV to UTC once
                tsla_base = _normalize_to_utc(tsla)
                spy_base = _normalize_to_utc(spy)
                vix_base = _normalize_to_utc(vix)
                print(f"[DATA] Normalized CSV data to UTC (one-time)")

            # Merge: CSV history + fresh yfinance data
            csv_end = tsla_base.index[-1] if len(tsla_base) > 0 else pd.Timestamp.min.tz_localize("UTC")
            fresh_tsla = tsla_live[tsla_live.index > csv_end]
            fresh_spy = spy_live[spy_live.index > csv_end]
            fresh_vix = vix_live[vix_live.index > csv_end]

            if len(fresh_tsla) > 0:
                tsla_merged = pd.concat([tsla_base, fresh_tsla])
                spy_merged = pd.concat([spy_base, fresh_spy])
                vix_merged = pd.concat([vix_base, fresh_vix])
                print(f"[DATA] Merged: {len(tsla_base):,} CSV + {len(fresh_tsla):,} fresh "
                      f"= {len(tsla_merged):,} total bars")
            else:
                # yfinance data overlaps with CSV — update last bars
                overlap_tsla = tsla_live[tsla_live.index >= tsla_base.index[0]]
                if len(overlap_tsla) > 0:
                    tsla_merged = tsla_base.copy()
                    spy_merged = spy_base.copy()
                    vix_merged = vix_base.copy()
                    tsla_merged.update(overlap_tsla)
                    spy_merged.update(spy_live[spy_live.index >= spy_base.index[0]])
                    vix_merged.update(vix_live[vix_live.index >= vix_base.index[0]])
                    print(f"[DATA] Updated {len(overlap_tsla):,} overlapping bars")
                else:
                    tsla_merged = tsla_base
                    spy_merged = spy_base
                    vix_merged = vix_base
                    print("[DATA] No overlap — using CSV data as-is")

            # Take last N bars
            if len(tsla_merged) > lookback:
                tsla_merged = tsla_merged.iloc[-lookback:]
                spy_merged = spy_merged.iloc[-lookback:]
                vix_merged = vix_merged.iloc[-lookback:]

            # Cache: store the UTC-normalized CSV base + final merged result
            st.session_state['_merged_5min'] = {
                'csv_utc': (tsla_base, spy_base, vix_base),
            }

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
    # Prefer yfinance last 5-min bar (prepost=True, includes pre/post market).
    current_price = 0.0
    vix_level = 20.0
    rt_prices = _get_realtime_prices()
    if len(current_tsla) > 0 and 'close' in current_tsla.columns:
        current_price = float(current_tsla.iloc[-1]['close'])
        _price_source = "yfinance (live)"
    elif rt_prices.get('TSLA'):
        current_price = rt_prices['TSLA']
        _price_source = "yfinance (rt)"
    else:
        _price_source = "unavailable"
    if len(current_vix) > 0 and 'close' in current_vix.columns:
        vix_level = float(current_vix.iloc[-1]['close'])
    st.caption(f"TSLA ${current_price:.2f} ({_price_source}) | VIX {vix_level:.1f}")

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


# =============================================================================
# Channel Surfer Tab
# =============================================================================

def _render_position_gauge(position_pct: float, width_px: int = 300) -> str:
    """Render a horizontal gauge showing price position in channel as HTML."""
    pos = max(0, min(1, position_pct))
    pct = pos * 100

    # Color based on zone
    if pos <= ZONE_OVERSOLD:
        color = '#00c853'  # Strong buy green
        zone = 'STRONG BUY'
    elif pos <= ZONE_LOWER:
        color = '#69f0ae'  # Light buy green
        zone = 'BUY ZONE'
    elif pos >= ZONE_OVERBOUGHT:
        color = '#ff1744'  # Strong sell red
        zone = 'STRONG SELL'
    elif pos >= ZONE_UPPER:
        color = '#ff8a80'  # Light sell red
        zone = 'SELL ZONE'
    else:
        color = '#ffab40'  # Neutral orange
        zone = 'NEUTRAL'

    return f"""
    <div style="position:relative;height:32px;background:linear-gradient(90deg,#00c853 0%,#00c853 15%,#69f0ae 15%,#69f0ae 30%,#555 30%,#555 70%,#ff8a80 70%,#ff8a80 85%,#ff1744 85%,#ff1744 100%);border-radius:6px;width:{width_px}px;margin:2px 0;">
        <div style="position:absolute;left:{pct}%;top:-2px;transform:translateX(-50%);width:4px;height:36px;background:white;border-radius:2px;box-shadow:0 0 4px rgba(255,255,255,0.8);"></div>
        <div style="position:absolute;left:{pct}%;top:34px;transform:translateX(-50%);font-size:11px;color:{color};font-weight:bold;white-space:nowrap;">{pct:.0f}% {zone}</div>
    </div>
    """


def _render_market_insights(analysis) -> None:
    """Render rules-based market insights derived from channel analysis fields.

    Translates raw numeric fields (momentum_is_turning, position_pct, ou_half_life,
    break_prob, channel_health) into plain-English observations. No LLM needed.
    """
    insights = []

    tf_hours = {'5min': 1/12, '15min': 0.25, '30min': 0.5, '1h': 1, '2h': 2, '3h': 3,
                '4h': 4, 'daily': 24, 'weekly': 168, 'monthly': 720}

    for tf, state in analysis.tf_states.items():
        if not state.valid:
            continue
        hrs = tf_hours.get(tf, 1)

        # Momentum exhaustion / turning
        if getattr(state, 'momentum_is_turning', False):
            direction = 'sell-off' if getattr(state, 'momentum_direction', 0) < 0 else 'rally'
            insights.append(('⚠️', tf, f'{tf} momentum turning — {direction} may be near exhaustion'))

        # At channel boundary
        pos = getattr(state, 'position_pct', 0.5)
        if pos is not None:
            pos_f = float(pos)
            if pos_f <= 0.05:
                hl = getattr(state, 'ou_half_life', None)
                rs = getattr(state, 'ou_reversion_score', 0)
                hl_str = f', bounce expected ~{hl * hrs:.0f}h' if hl is not None and rs > 0.2 else ''
                insights.append(('📍', tf, f'{tf} price AT channel bottom (position={pos_f:.1%}){hl_str}'))
            elif pos_f >= 0.95:
                hl = getattr(state, 'ou_half_life', None)
                rs = getattr(state, 'ou_reversion_score', 0)
                hl_str = f', pullback expected ~{hl * hrs:.0f}h' if hl is not None and rs > 0.2 else ''
                insights.append(('📍', tf, f'{tf} price AT channel top (position={pos_f:.1%}){hl_str}'))

        # High break probability
        bp_dn = getattr(state, 'break_prob_down', 0)
        bp_up = getattr(state, 'break_prob_up', 0)
        if float(bp_dn) > 0.55:
            insights.append(('🔴', tf, f'{tf} high breakdown probability: {float(bp_dn):.0%}'))
        elif float(bp_up) > 0.55:
            insights.append(('🟢', tf, f'{tf} high breakout probability: {float(bp_up):.0%}'))

        # Weak channel health
        ch = getattr(state, 'channel_health', 1.0)
        if float(ch) < 0.35:
            insights.append(('🟡', tf, f'{tf} channel health weak ({float(ch):.2f}) — signal less reliable'))

        # High total energy (channel about to break)
        te = getattr(state, 'total_energy', 0)
        be = getattr(state, 'binding_energy', 1)
        if float(be) > 0 and float(te) / float(be) > 2.5:
            insights.append(('⚡', tf, f'{tf} energy ratio {float(te)/float(be):.1f}x — channel under stress'))

    # Confluence note
    cf = getattr(analysis, 'confluence_matrix', {})
    if cf:
        all_agree = all(abs(v - list(cf.values())[0]) < 0.01 for v in cf.values())
        if all_agree and len(cf) >= 3:
            first_v = list(cf.values())[0]
            direction = 'bearish' if getattr(analysis.signal, 'action', '') == 'SELL' else 'bullish'
            insights.append(('✅', 'all TFs', f'All {len(cf)} timeframes in full consensus ({direction}) — high conviction signal'))

    if not insights:
        return

    with st.expander('Market Insights', expanded=True):
        for icon, tf, msg in insights:
            st.markdown(f'{icon} {msg}')


def _render_signal_banner(signal, current_price: float = 0.0) -> None:
    """Render a prominent BUY/SELL/HOLD banner with signal type indicator."""
    sig_type = getattr(signal, 'signal_type', 'bounce')
    type_label = 'BREAKOUT' if sig_type == 'break' else 'BOUNCE'
    type_icon = '&#x26A1;' if sig_type == 'break' else '&#x21C4;'  # ⚡ vs ⇄

    if signal.action == 'BUY':
        # Breakout = orange-green gradient, Bounce = green gradient
        if sig_type == 'break':
            bg = "linear-gradient(135deg,#1a3300,#336600)"
            border = "#76ff03"
            glow = "#76ff03"
        else:
            bg = "linear-gradient(135deg,#004d1a,#00802b)"
            border = "#00c853"
            glow = "#00ff55"
        # Compute entry instructions
        entry_info = ""
        if current_price > 0:
            stop = current_price * (1 - signal.suggested_stop_pct)
            tp = current_price * (1 + signal.suggested_tp_pct)
            rr = signal.suggested_tp_pct / max(signal.suggested_stop_pct, 0.001)
            entry_info = (
                f"<div style='font-size:16px;color:#ccffcc;margin-top:8px;font-family:monospace;'>"
                f"Entry: ${current_price:.2f} &nbsp; Stop: ${stop:.2f} &nbsp; TP: ${tp:.2f} &nbsp; R:R {rr:.1f}:1"
                f"</div>"
            )

        st.markdown(
            f"""<div style="background:{bg};border:2px solid {border};
            border-radius:12px;padding:20px;text-align:center;margin:10px 0;">
            <div style="font-size:12px;font-weight:700;color:#ffcc00;letter-spacing:2px;margin-bottom:4px;">
            {type_icon} {type_label}</div>
            <div style="font-size:48px;font-weight:900;color:{glow};text-shadow:0 0 20px {glow};">
            BUY</div>
            <div style="font-size:18px;color:#aaffcc;margin-top:8px;">
            Confidence: {signal.confidence:.0%} | {signal.primary_tf} | Stop: {signal.suggested_stop_pct:.2%} | TP: {signal.suggested_tp_pct:.2%}
            </div>
            {entry_info}
            <div style="font-size:14px;color:#88cc99;margin-top:4px;">{signal.reason}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    elif signal.action == 'SELL':
        if sig_type == 'break':
            bg = "linear-gradient(135deg,#330000,#661a00)"
            border = "#ff6d00"
            glow = "#ff6d00"
        else:
            bg = "linear-gradient(135deg,#4d0000,#800000)"
            border = "#ff1744"
            glow = "#ff4444"
        # Compute entry instructions for SELL
        entry_info = ""
        if current_price > 0:
            stop = current_price * (1 + signal.suggested_stop_pct)
            tp = current_price * (1 - signal.suggested_tp_pct)
            rr = signal.suggested_tp_pct / max(signal.suggested_stop_pct, 0.001)
            entry_info = (
                f"<div style='font-size:16px;color:#ffcccc;margin-top:8px;font-family:monospace;'>"
                f"Entry: ${current_price:.2f} &nbsp; Stop: ${stop:.2f} &nbsp; TP: ${tp:.2f} &nbsp; R:R {rr:.1f}:1"
                f"</div>"
            )

        st.markdown(
            f"""<div style="background:{bg};border:2px solid {border};
            border-radius:12px;padding:20px;text-align:center;margin:10px 0;">
            <div style="font-size:12px;font-weight:700;color:#ffcc00;letter-spacing:2px;margin-bottom:4px;">
            {type_icon} {type_label}</div>
            <div style="font-size:48px;font-weight:900;color:{glow};text-shadow:0 0 20px {glow};">
            SELL</div>
            <div style="font-size:18px;color:#ffaaaa;margin-top:8px;">
            Confidence: {signal.confidence:.0%} | {signal.primary_tf} | Stop: {signal.suggested_stop_pct:.2%} | TP: {signal.suggested_tp_pct:.2%}
            </div>
            {entry_info}
            <div style="font-size:14px;color:#cc8888;margin-top:4px;">{signal.reason}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div style="background:linear-gradient(135deg,#1a1a2e,#2a2a4e);border:2px solid #555;
            border-radius:12px;padding:16px;text-align:center;margin:10px 0;">
            <div style="font-size:36px;font-weight:700;color:#aaa;">
            HOLD</div>
            <div style="font-size:14px;color:#888;margin-top:4px;">
            {signal.reason} (conf: {signal.confidence:.0%})</div>
            </div>""",
            unsafe_allow_html=True,
        )


def _play_alert_sound(action: str) -> None:
    """Inject HTML5 audio to play a BUY/SELL alert sound."""
    if action == 'BUY':
        # Rising tone sequence
        freq_start, freq_end = 400, 800
    elif action == 'SELL':
        # Falling tone sequence
        freq_start, freq_end = 800, 400
    else:
        return

    st.markdown(
        f"""<script>
        (function() {{
            try {{
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.type = 'sine';
                osc.frequency.setValueAtTime({freq_start}, ctx.currentTime);
                osc.frequency.linearRampToValueAtTime({freq_end}, ctx.currentTime + 0.3);
                gain.gain.setValueAtTime(0.3, ctx.currentTime);
                gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.5);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.5);
            }} catch(e) {{}}
        }})();
        </script>""",
        unsafe_allow_html=True,
    )


def _play_exit_alert_sound(exit_reason: str) -> None:
    """Play an audible alert when a position exit triggers."""
    if exit_reason == 'take_profit':
        # Celebratory ascending chime: E4 → G#4 → B4
        js = """(function(){
            try {
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                [330, 415, 494].forEach(function(f, i) {
                    var o = ctx.createOscillator(), g = ctx.createGain();
                    o.connect(g); g.connect(ctx.destination);
                    o.type = 'sine'; o.frequency.value = f;
                    var t = ctx.currentTime + i * 0.18;
                    g.gain.setValueAtTime(0.35, t);
                    g.gain.linearRampToValueAtTime(0, t + 0.25);
                    o.start(t); o.stop(t + 0.25);
                });
            } catch(e) {}
        })();"""
    elif exit_reason in ('stop_loss', 'trailing_stop'):
        # Urgent alarm: 3 descending square-wave pulses
        js = """(function(){
            try {
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                [600, 500, 420].forEach(function(f, i) {
                    var o = ctx.createOscillator(), g = ctx.createGain();
                    o.connect(g); g.connect(ctx.destination);
                    o.type = 'square'; o.frequency.value = f;
                    var t = ctx.currentTime + i * 0.14;
                    g.gain.setValueAtTime(0.18, t);
                    g.gain.linearRampToValueAtTime(0, t + 0.12);
                    o.start(t); o.stop(t + 0.12);
                });
            } catch(e) {}
        })();"""
    else:
        # timeout / other: single neutral beep
        js = """(function(){
            try {
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                var o = ctx.createOscillator(), g = ctx.createGain();
                o.connect(g); g.connect(ctx.destination);
                o.type = 'sine'; o.frequency.value = 440;
                g.gain.setValueAtTime(0.2, ctx.currentTime);
                g.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.4);
                o.start(ctx.currentTime); o.stop(ctx.currentTime + 0.4);
            } catch(e) {}
        })();"""
    st.markdown(f"<script>{js}</script>", unsafe_allow_html=True)


def _render_scanner_exit_alert(ea) -> None:
    """Render a prominent exit alert banner for a closed position."""
    if ea.exit_reason == 'take_profit':
        bg, border, icon, label = '#0a3320', '#00e676', '🎯', 'TAKE PROFIT HIT'
    elif ea.exit_reason == 'stop_loss':
        bg, border, icon, label = '#3a0a0a', '#ff1744', '🛑', 'STOP LOSS HIT'
    elif ea.exit_reason == 'trailing_stop':
        bg, border, icon, label = '#2a1a0a', '#ff9100', '📉', 'TRAILING STOP HIT'
    else:
        bg, border, icon, label = '#1a1a2e', '#888888', '⏱', ea.exit_reason.upper()
    pnl_color = '#00e676' if ea.pnl >= 0 else '#ff5252'
    st.markdown(
        f'<div style="background:{bg};padding:12px 16px;border-radius:8px;margin:6px 0;'
        f'border:2px solid {border};">'
        f'<span style="font-size:18px">{icon}</span> '
        f'<b style="color:{border};font-size:15px"> {label}</b> '
        f'<span style="color:#aaa">[{ea.pos_id}]</span> @ '
        f'<b>${ea.price:.2f}</b> — '
        f'P&L: <b style="color:{pnl_color}">${ea.pnl:+,.0f}</b> ({ea.pnl_pct:+.2%})'
        f'</div>',
        unsafe_allow_html=True,
    )


def _show_surfer_chart(tsla_df, analysis):
    """Show interactive 5min candlestick chart with channel overlay and signal markers."""
    from v15.core.channel import detect_channels_multi_window, select_best_channel
    import pytz

    # Filter to today's trading session only; fall back to most recent trading day
    try:
        et_tz = pytz.timezone('America/New_York')
        if tsla_df.index.tz is not None:
            dates_et = tsla_df.index.tz_convert(et_tz).date
        else:
            dates_et = tsla_df.index.date
        today_et = pd.Timestamp.now(tz=et_tz).date()
        df_today = tsla_df[dates_et == today_et]
        # If fewer than 5 bars today (market closed / weekend), use most recent trading day
        if len(df_today) < 5 and len(tsla_df) > 0:
            most_recent = dates_et[-1]
            df_today = tsla_df[dates_et == most_recent]
        df_chart = df_today.copy() if len(df_today) > 0 else tsla_df.tail(100).copy()
    except Exception:
        df_chart = tsla_df.tail(100).copy()

    # Detect the 5min channel
    windows = [10, 15, 20, 30, 40]
    try:
        multi_ch = detect_channels_multi_window(df_chart, windows=windows)
        best_ch, best_w = select_best_channel(multi_ch)
    except Exception:
        best_ch = None

    # Build integer x-axis with ET tick labels (consistent with multi-TF charts)
    x_values = list(range(len(df_chart)))

    # Convert index to ET for tick labels
    disp_idx = df_chart.index
    if disp_idx.tz is not None:
        disp_idx = disp_idx.tz_convert('America/New_York')

    # Generate tick positions — every ~30 min for 5-min bars
    n = len(df_chart)
    tick_step = max(1, n // 10)  # ~10 ticks
    tick_positions = list(range(0, n, tick_step))
    if tick_positions[-1] != n - 1:
        tick_positions.append(n - 1)
    tick_labels = [disp_idx[p].strftime('%H:%M') for p in tick_positions]

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=x_values,
        open=df_chart['open'],
        high=df_chart['high'],
        low=df_chart['low'],
        close=df_chart['close'],
        name='TSLA 5min',
        increasing_line_color='#00c853',
        decreasing_line_color='#ff1744',
    ))

    # Channel overlay
    if best_ch and best_ch.valid:
        ch_len = len(best_ch.center_line)
        ch_start = max(0, len(df_chart) - ch_len)
        ch_x = x_values[ch_start:ch_start + ch_len]

        fig.add_trace(go.Scatter(
            x=ch_x, y=best_ch.upper_line[:len(ch_x)],
            mode='lines', name='Upper',
            line=dict(color='rgba(255,100,100,0.6)', width=1, dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=ch_x, y=best_ch.lower_line[:len(ch_x)],
            mode='lines', name='Lower',
            line=dict(color='rgba(100,255,100,0.6)', width=1, dash='dash'),
            fill='tonexty', fillcolor='rgba(100,100,255,0.05)',
        ))
        fig.add_trace(go.Scatter(
            x=ch_x, y=best_ch.center_line[:len(ch_x)],
            mode='lines', name='Center',
            line=dict(color='rgba(200,200,200,0.4)', width=1, dash='dot'),
        ))

    # Signal marker
    sig = analysis.signal
    if sig.action in ('BUY', 'SELL') and len(df_chart) > 0:
        last_x = x_values[-1]
        last_price = float(df_chart['close'].iloc[-1])
        color = '#00ff55' if sig.action == 'BUY' else '#ff4444'
        symbol = 'triangle-up' if sig.action == 'BUY' else 'triangle-down'
        fig.add_trace(go.Scatter(
            x=[last_x], y=[last_price],
            mode='markers+text',
            marker=dict(size=18, color=color, symbol=symbol),
            text=[sig.action],
            textposition='top center' if sig.action == 'BUY' else 'bottom center',
            textfont=dict(size=14, color=color),
            name=sig.action,
        ))

    # Get 5min state for position annotation
    state_5m = analysis.tf_states.get('5min')
    title_extra = ""
    if state_5m and state_5m.valid:
        title_extra = f" | Pos: {state_5m.position_pct:.0%} | Health: {state_5m.channel_health:.0%} | Break: {state_5m.break_prob:.0%}"

    fig.update_layout(
        title=f"TSLA 5min Channel{title_extra}",
        template='plotly_dark',
        height=450,
        xaxis_rangeslider_visible=False,
        xaxis=dict(tickvals=tick_positions, ticktext=tick_labels),
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, width="stretch")


def _render_break_predictor(analysis, native_tf_data, current_spy=None, current_vix=None):
    """Render the Phase 4 evolved channel break direction predictor panel."""
    _bp_unavail = (
        '<div style="background:#1a1a2e;border:1px solid #333;border-radius:6px;'
        'padding:8px 12px;margin:4px 0;display:flex;align-items:center;gap:12px;">'
        '<span style="color:#aaa;font-size:11px;font-weight:600;white-space:nowrap;">'
        'BREAK PREDICTOR</span>'
        '<span style="color:#555;font-size:14px;font-weight:700;">UNAVAILABLE</span>'
        '<span style="color:#555;font-size:11px;">{reason}</span></div>'
    )

    try:
        from v15.core.break_predictor import extract_break_features, predict_break
    except ImportError as e:
        print(f"[BREAK_PRED] IMPORT FAILED: {e}")
        st.markdown(_bp_unavail.format(reason=f"import failed: {e}"), unsafe_allow_html=True)
        return

    if analysis is None:
        st.markdown(_bp_unavail.format(reason="no channel analysis"), unsafe_allow_html=True)
        return

    try:
        features = extract_break_features(analysis, native_tf_data, current_spy, current_vix)
        if features is None:
            print("[BREAK_PRED] extract_break_features returned None")
            st.markdown(_bp_unavail.format(reason="insufficient data for features"), unsafe_allow_html=True)
            return
        result = predict_break(features)
    except Exception as e:
        print(f"[BREAK_PRED] FAILED: {type(e).__name__}: {e}")
        st.markdown(_bp_unavail.format(reason=f"{type(e).__name__}: {e}"), unsafe_allow_html=True)
        return

    direction   = result['direction']
    confidence  = result['confidence']
    will_break  = result['will_break']
    position    = features['position']

    # Position label
    if position > 0.80:
        pos_label = "near upper boundary"
    elif position < 0.20:
        pos_label = "near lower boundary"
    else:
        pos_label = f"mid-channel ({position:.0%})"

    # Direction badge
    if will_break:
        dir_color = '#4caf50' if direction == 'UP' else '#ef5350'
        dir_arrow = '↑' if direction == 'UP' else '↓'
        dir_text  = f"{dir_arrow} {direction}"
        status_text = "Break predicted"
    else:
        dir_color = '#888888'
        dir_arrow = '→'
        dir_text  = "HOLD"
        status_text = "Channel holds"

    # Alignment with current channel signal (if available from tf_states)
    signal_action = 'HOLD'
    if analysis.tf_states:
        for s in analysis.tf_states.values():
            if s.valid:
                if s.position_pct < 0.25 and direction == 'UP':
                    signal_action = 'ALIGNED'  # near lower, predicts UP = bounce holds
                elif s.position_pct < 0.25 and direction == 'DOWN':
                    signal_action = 'CAUTION'  # near lower, predicts DOWN = bounce fails
                elif s.position_pct > 0.75 and direction == 'UP':
                    signal_action = 'ALIGNED'  # near upper, predicts UP = breakout confirmed
                elif s.position_pct > 0.75 and direction == 'DOWN':
                    signal_action = 'CAUTION'  # near upper, predicts DOWN = false breakout
                break

    align_html = ''
    if signal_action == 'ALIGNED':
        align_html = '<span style="color:#4caf50;font-size:11px;margin-left:8px;">✓ aligned</span>'
    elif signal_action == 'CAUTION':
        align_html = '<span style="color:#ff9800;font-size:11px;margin-left:8px;">⚠ counter-signal</span>'

    st.markdown(
        f"""<div style="background:#1a1a2e;border:1px solid #333;border-radius:6px;
                        padding:8px 12px;margin:4px 0;display:flex;align-items:center;gap:12px;">
            <span style="color:#aaa;font-size:11px;font-weight:600;white-space:nowrap;">
                BREAK PREDICTOR</span>
            <span style="color:{dir_color};font-size:18px;font-weight:700;">{dir_text}</span>
            <span style="color:#888;font-size:11px;">{status_text}</span>
            <span style="color:#666;font-size:11px;">·</span>
            <span style="color:#aaa;font-size:11px;">{pos_label}</span>
            <span style="color:#666;font-size:11px;">·</span>
            <span style="color:#888;font-size:11px;">conf {confidence:.0%}</span>
            {align_html}
        </div>""",
        unsafe_allow_html=True,
    )


def _show_ml_predictions(analysis, current_tsla, native_tf_data):
    """Show ML model predictions for channel lifetime, break direction, and action."""
    try:
        from v15.core.surfer_ml import (
            GBTModel, extract_tf_features, extract_cross_tf_features,
            extract_context_features, extract_correlation_features,
            extract_temporal_features,
            get_feature_names, ML_TFS, PER_TF_FEATURES,
            CROSS_TF_FEATURES, CONTEXT_FEATURES, CORRELATION_FEATURES,
            TEMPORAL_FEATURES,
        )
    except ImportError:
        return  # ML module not available

    # Load model (cached in session state)
    if 'surfer_ml_model' not in st.session_state:
        import os
        model_path = os.path.join(os.path.dirname(__file__), '..', 'surfer_models', 'gbt_model.pkl')
        if not os.path.exists(model_path):
            model_path = 'surfer_models/gbt_model.pkl'
        if os.path.exists(model_path):
            try:
                st.session_state['surfer_ml_model'] = GBTModel.load(model_path)
            except Exception:
                st.session_state['surfer_ml_model'] = None
        else:
            st.session_state['surfer_ml_model'] = None

    model = st.session_state.get('surfer_ml_model')
    if model is None:
        return  # No ML model available

    if analysis is None or not analysis.tf_states:
        return

    # Extract features
    try:
        feature_names = get_feature_names()
        num_features = len(feature_names)
        feature_vec = np.zeros(num_features, dtype=np.float32)
        offset = 0

        for tf in ML_TFS:
            state = analysis.tf_states.get(tf)
            if state:
                tf_feats = extract_tf_features(state)
            else:
                tf_feats = np.zeros(len(PER_TF_FEATURES), dtype=np.float32)
            feature_vec[offset:offset + len(PER_TF_FEATURES)] = tf_feats
            offset += len(PER_TF_FEATURES)

        cross_feats = extract_cross_tf_features(analysis.tf_states)
        feature_vec[offset:offset + len(CROSS_TF_FEATURES)] = cross_feats
        offset += len(CROSS_TF_FEATURES)

        # Context features from current data
        ctx_feats = np.zeros(len(CONTEXT_FEATURES), dtype=np.float32)
        if current_tsla is not None and len(current_tsla) > 20:
            bar_idx = len(current_tsla) - 1
            ctx_feats = extract_context_features(current_tsla, bar_idx)
            feature_vec[offset:offset + len(CONTEXT_FEATURES)] = ctx_feats
        offset += len(CONTEXT_FEATURES)

        # Temporal features (use session state for history buffer)
        if 'ml_history_buffer' not in st.session_state:
            st.session_state['ml_history_buffer'] = []
        dash_snapshot = {}
        for tf in ML_TFS:
            state = analysis.tf_states.get(tf)
            if state and state.valid:
                for feat_name in PER_TF_FEATURES:
                    val = getattr(state, feat_name, 0.0)
                    if isinstance(val, (int, float)):
                        dash_snapshot[f'{tf}_{feat_name}'] = float(val)
        dash_snapshot['rsi_14'] = float(ctx_feats[0])
        dash_snapshot['volume_ratio_20'] = float(ctx_feats[2])

        closes_arr = current_tsla['close'].values if current_tsla is not None else None
        temporal_feats = extract_temporal_features(
            dash_snapshot, st.session_state['ml_history_buffer'],
            closes=closes_arr,
            bar_idx=len(current_tsla) - 1 if current_tsla is not None else 0,
            eval_interval=1,
        )
        feature_vec[offset:offset + len(TEMPORAL_FEATURES)] = temporal_feats
        offset += len(TEMPORAL_FEATURES)

        st.session_state['ml_history_buffer'].append(dash_snapshot)
        if len(st.session_state['ml_history_buffer']) > 20:
            st.session_state['ml_history_buffer'].pop(0)

        # Correlation features (zeros for now — dashboard doesn't have SPY/VIX 5min)
        offset += len(CORRELATION_FEATURES)

        # Predict
        prediction = model.predict(feature_vec.reshape(1, -1))

    except Exception as e:
        logger.warning(f"ML prediction failed: {e}")
        return

    # --- Display ML Predictions ---
    st.markdown("---")
    st.subheader("ML Predictions")

    col1, col2, col3, col4 = st.columns(4)

    # Channel Lifetime
    with col1:
        lifetime = float(prediction.get('lifetime', [0])[0])
        hours = lifetime * 5 / 60  # Convert 5min bars to hours
        if lifetime > 100:
            lt_color = "#00c853"
            lt_label = "LONG"
        elif lifetime > 30:
            lt_color = "#ffab40"
            lt_label = "MEDIUM"
        else:
            lt_color = "#ff1744"
            lt_label = "SHORT"
        st.metric("Channel Life", f"~{lifetime:.0f} bars", help=f"~{hours:.1f} hours remaining")
        st.markdown(f"<div style='text-align:center;color:{lt_color};font-weight:bold;'>{lt_label}</div>",
                   unsafe_allow_html=True)

    # Break Direction
    with col2:
        bd = int(prediction.get('break_dir', [0])[0])
        bd_labels = {0: "SURVIVE", 1: "BREAK UP", 2: "BREAK DOWN"}
        bd_colors = {0: "#4fc3f7", 1: "#00c853", 2: "#ff1744"}
        bd_emojis = {0: "~", 1: "+", 2: "-"}
        bd_label = bd_labels.get(bd, "?")
        bd_color = bd_colors.get(bd, "#888")

        # Show probabilities if available
        if 'break_dir_probs' in prediction:
            probs = prediction['break_dir_probs'][0]
            conf_str = f"{probs[bd]:.0%}"
        else:
            conf_str = ""

        st.metric("Break Direction", f"{bd_emojis.get(bd, '?')} {bd_label}")
        st.markdown(f"<div style='text-align:center;color:{bd_color};font-weight:bold;'>{conf_str}</div>",
                   unsafe_allow_html=True)

    # ML Action
    with col3:
        action = int(prediction.get('action', [0])[0])
        action_labels = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_colors = {0: "#888", 1: "#00c853", 2: "#ff1744"}
        action_label = action_labels.get(action, "?")
        action_color = action_colors.get(action, "#888")

        if 'action_probs' in prediction:
            probs = prediction['action_probs'][0]
            conf_str = f"{probs[action]:.0%}"
        else:
            conf_str = ""

        st.metric("ML Signal", action_label)
        st.markdown(f"<div style='text-align:center;color:{action_color};font-weight:bold;font-size:18px;'>"
                   f"{conf_str}</div>", unsafe_allow_html=True)

    # Future Return Prediction
    with col4:
        ret_20 = float(prediction.get('future_return_20', [0])[0])
        ret_color = "#00c853" if ret_20 > 0 else "#ff1744"
        st.metric("20-bar Return", f"{ret_20:+.2%}")
        st.markdown(f"<div style='text-align:center;color:{ret_color};font-size:12px;'>Predicted</div>",
                   unsafe_allow_html=True)

    # Agreement indicator
    physics_action = analysis.signal.action
    ml_action = action_labels.get(action, "HOLD")

    if physics_action == ml_action and physics_action != 'HOLD':
        st.success(f"PHYSICS + ML AGREE: {physics_action} (high confidence setup)")
    elif physics_action != 'HOLD' and ml_action != 'HOLD' and physics_action != ml_action:
        st.warning(f"CONFLICT: Physics says {physics_action}, ML says {ml_action} (proceed with caution)")
    elif physics_action == 'HOLD' and ml_action != 'HOLD':
        st.info(f"ML sees opportunity: {ml_action} (but physics says HOLD)")


def _get_tod_dow_multipliers(signal_type: str = 'bounce'):
    """
    Compute current TOD + DOW position sizing multipliers for bounce signals.
    Mirrors the logic in surfer_backtest.py (Arch 410 + Arch 406 committed values).
    Only bounce signals get TOD/DOW boosts.
    Returns (tod_mult, tod_label, dow_mult, dow_label).
    """
    import datetime as _dt
    import pytz
    now_utc = _dt.datetime.now(_dt.timezone.utc)
    hour_utc = now_utc.hour
    dow = now_utc.weekday()  # 0=Mon, 4=Fri, 5/6=Weekend

    if signal_type != 'bounce':
        return 1.0, f'N/A (not bounce)', 1.0, 'N/A (not bounce)'

    # TOD table (UTC hour → multiplier, label) — keep in sync with surfer_backtest.py (Arch410)
    tod_table = {
        8:  (1.15, '3am ET'),
        9:  (1.15, '4am ET'),
        10: (1.15, '5am ET'),
        11: (1.15, '6am ET'),
        12: (1.05, '7am ET'),
        13: (1.50, '8am ET ⭐'),
        14: (1.40, '9am ET ⭐'),
        15: (1.40, '10am ET ⭐'),
        16: (1.40, '11am ET ⭐'),
        17: (1.40, '12pm ET ⭐'),
        18: (1.50, '1pm ET ⭐'),
        19: (1.50, '2pm ET ⭐'),
        20: (1.20, '3pm ET'),
        21: (1.20, '4pm ET'),
    }
    # DOW table (weekday → multiplier, label) — keep in sync with surfer_backtest.py (Arch415)
    dow_table = {
        0: (1.35, 'Monday'),
        1: (1.35, 'Tuesday'),
        2: (1.35, 'Wednesday'),
        3: (1.45, 'Thursday ⭐'),
        4: (1.35, 'Friday'),
    }
    tod_mult, tod_label = tod_table.get(hour_utc, (1.0, f'UTC{hour_utc} (no boost)'))
    dow_mult, dow_label = dow_table.get(dow, (1.0, 'Weekend'))
    return tod_mult, tod_label, dow_mult, dow_label


def _load_signal_quality_model():
    """Load the signal quality model into session state (once)."""
    if 'signal_quality_model' not in st.session_state:
        _load_log = []
        # Prefer c10 arch2 model (177 features), then tuned (169 features), then base
        base_dir = Path(__file__).parent / 'validation'
        model_path = base_dir / 'signal_quality_model_c10_arch2.pkl'
        if not model_path.exists():
            model_path = base_dir / 'signal_quality_model_tuned.pkl'
            if not model_path.exists():
                model_path = base_dir / 'signal_quality_model.pkl'
                _load_log.append(f"Tuned model not found, falling back to base: {model_path.name}")
        if model_path.exists():
            try:
                from v15.validation.signal_quality_model import SignalQualityModel
                model = SignalQualityModel.load(str(model_path))
                st.session_state['signal_quality_model'] = model
                cv = model.cv_metrics or {}
                auc = cv.get('overall_auc', 'N/A')
                calibrated = 'yes' if model.calibrator is not None else 'no'
                n_feat = len(model.feature_names) if model.feature_names else '?'
                msg = f"Model: {model_path.name} | AUC={auc} | calibrated={calibrated} | {n_feat} features"
                _load_log.append(msg)
                print(f"[SQ] Loaded {msg}")
            except Exception as e:
                st.session_state['signal_quality_model'] = None
                _load_log.append(f"LOAD FAILED: {model_path.name}: {e}")
                print(f"[SQ] LOAD FAILED: {model_path.name}: {e}")
        else:
            st.session_state['signal_quality_model'] = None
            _load_log.append("No signal quality model file found")
            print("[SQ] No signal quality model file found")
        st.session_state['_sq_load_log'] = _load_log
    return st.session_state['signal_quality_model']


def _render_ml_signal_quality(analysis, sig, current_tsla, spy_df=None, vix_df=None):
    """Render ML signal quality panel below the signal banner (BUY/SELL only)."""
    if sig.action == 'HOLD':
        return

    import datetime as _dt
    log = []  # Fresh log each render
    log.append(f"--- ML Signal Quality @ {_dt.datetime.now().strftime('%H:%M:%S')} ---")

    model = _load_signal_quality_model()
    # Include model load info
    for line in st.session_state.get('_sq_load_log', []):
        log.append(line)
    if model is None:
        log.append("NO MODEL LOADED — skipping")
        st.session_state['_sq_log'] = log
        for line in log:
            print(f"[SQ] {line}")
        load_info = '\n'.join(st.session_state.get('_sq_load_log', ['(no load log)']))
        st.warning(f"ML Signal Quality: model not loaded.\n{load_info}")
        return

    # Log the incoming signal
    sig_type = getattr(sig, 'signal_type', 'unknown')
    log.append(f"Signal: {sig.action} {sig_type} | TF={sig.primary_tf} | conf={sig.confidence:.0%}")
    log.append(f"  stop={sig.suggested_stop_pct:.2%} | tp={sig.suggested_tp_pct:.2%} | "
               f"R:R={sig.suggested_tp_pct/max(sig.suggested_stop_pct, 1e-6):.1f}:1")
    log.append(f"  scores: pos={sig.position_score:.2f} energy={sig.energy_score:.2f} "
               f"entropy={sig.entropy_score:.2f} confluence={sig.confluence_score:.2f} "
               f"timing={sig.timing_score:.2f} health={sig.channel_health:.2f}")

    # Extract feature vector
    try:
        from v15.core.surfer_backtest import _extract_signal_features
        from v15.core.surfer_ml import get_feature_names

        feature_names = get_feature_names()
        history_buffer = st.session_state.get('_sq_history_buffer', [])
        closes = current_tsla['close'].values if current_tsla is not None else None
        bar = len(current_tsla) - 1 if current_tsla is not None else 0

        feature_vec, _ = _extract_signal_features(
            analysis, current_tsla, bar, closes,
            spy_df=spy_df, vix_df=vix_df,
            feature_names=feature_names,
            history_buffer=history_buffer,
            eval_interval=6,  # Must match training default (signal_quality_model.py)
        )
        st.session_state['_sq_history_buffer'] = history_buffer
        nz = int(np.count_nonzero(feature_vec))
        log.append(f"Features: {len(feature_vec)}-dim base | {nz} non-zero | "
                   f"range [{np.min(feature_vec):.2f}, {np.max(feature_vec):.2f}]")

        # Append signal meta features (extended=True for tuned model)
        from v15.validation.signal_quality_model import _append_signal_meta
        class _SigProxy:
            pass
        _sig_proxy = _SigProxy()
        _sig_proxy.signal_type = getattr(sig, 'signal_type', 'bounce')
        _sig_proxy.direction = sig.action
        _sig_proxy.stop_pct = sig.suggested_stop_pct
        _sig_proxy.tp_pct = sig.suggested_tp_pct
        _sig_proxy.primary_tf = sig.primary_tf
        _sig_proxy.entry_time = _dt.datetime.now().isoformat()
        sig_data = {
            'position_score': sig.position_score,
            'energy_score': sig.energy_score,
            'entropy_score': sig.entropy_score,
            'confluence_score': sig.confluence_score,
            'timing_score': sig.timing_score,
            'channel_health': sig.channel_health,
            'confidence': sig.confidence,
        }
        full_features = _append_signal_meta(feature_vec, _sig_proxy, sig_data, extended=True)
        log.append(f"Meta appended: {len(feature_vec)} + {len(full_features) - len(feature_vec)} → {len(full_features)}-dim")

        pred = model.predict(full_features)
    except Exception as e:
        log.append(f"ERROR: {type(e).__name__}: {e}")
        st.warning(f"ML Signal Quality unavailable: {e}")
        st.session_state['_sq_log'] = log
        for line in log:
            print(f"[SQ] {line}")
        with st.expander("ML Signal Quality — Debug Log"):
            st.code('\n'.join(log), language='text')
        return

    win_prob = pred['win_prob']
    expected_pnl = pred['expected_pnl_pct']
    quality_score = pred['quality_score']
    risk_rating = pred['risk_rating']

    # Compute TOD/DOW multipliers for the current time
    sig_type = getattr(sig, 'signal_type', 'bounce')
    tod_mult, tod_label, dow_mult, dow_label = _get_tod_dow_multipliers(sig_type)

    # Compute VIX regime boost (Arch417: VIX≥20 ×1.10, VIX≥30 ×1.25, bounces only)
    vix_mult = 1.0
    vix_label = 'N/A'
    if sig_type == 'bounce' and vix_df is not None:
        try:
            import datetime as _dt2
            _vix_ts = _dt2.datetime.now()
            _vix_avail = vix_df[vix_df.index <= _vix_ts]
            if len(_vix_avail) > 0:
                _vix_val = float(_vix_avail['close'].iloc[-1])
                if _vix_val >= 30:
                    vix_mult = 1.25
                    vix_label = f'VIX={_vix_val:.0f} (HIGH ×1.25)'
                elif _vix_val >= 20:
                    vix_mult = 1.10
                    vix_label = f'VIX={_vix_val:.0f} (MID ×1.10)'
                else:
                    vix_label = f'VIX={_vix_val:.0f} (LOW, no boost)'
        except Exception:
            pass
    elif sig_type != 'bounce':
        vix_label = 'N/A (not bounce)'

    # Log prediction results
    _sq_size = ('FULL+' if quality_score >= 80 else 'FULL' if quality_score >= 60
                else 'REDUCED' if quality_score >= 40 else 'MINIMAL')
    _sq_mult = 1.3 if quality_score >= 80 else 1.0 if quality_score >= 60 else 0.7 if quality_score >= 40 else 0.4
    combined_mult = _sq_mult * tod_mult * dow_mult * vix_mult
    log.append(f"Result: win={win_prob:.0%} pnl={expected_pnl:+.2%} quality={quality_score:.0f} "
               f"risk={risk_rating} → ML={_sq_mult:.2f}x × TOD={tod_mult:.2f}x ({tod_label}) "
               f"× DOW={dow_mult:.2f}x ({dow_label}) × VIX={vix_mult:.2f}x ({vix_label}) "
               f"= {combined_mult:.2f}x combined")
    st.session_state['_sq_log'] = log
    # Print to terminal
    for line in log:
        print(f"[SQ] {line}")

    # Position sizing tier
    if quality_score >= 80:
        size_mult = 1.3
        size_label = 'FULL+'
        size_color = '#00c853'
    elif quality_score >= 60:
        size_mult = 1.0
        size_label = 'FULL'
        size_color = '#4caf50'
    elif quality_score >= 40:
        size_mult = 0.7
        size_label = 'REDUCED'
        size_color = '#ff9800'
    else:
        size_mult = 0.4
        size_label = 'MINIMAL'
        size_color = '#ff1744'

    # Risk badge colors
    risk_colors = {'LOW': '#00c853', 'MEDIUM': '#ff9800', 'HIGH': '#ff1744'}
    risk_color = risk_colors.get(risk_rating, '#888')

    # Quality bar color
    if quality_score >= 70:
        q_color = '#00c853'
    elif quality_score >= 40:
        q_color = '#ff9800'
    else:
        q_color = '#ff1744'

    # Pre-compute dollar estimate for caption (full sizing block computed below)
    _base_cap_preview = st.session_state.get('backtest_capital', 100_000.0)
    _base_trade_preview = _base_cap_preview / 10.0
    _ml_adj = min(_base_trade_preview * size_mult, 1_000_000.0)
    _tod_adj = min(_ml_adj * tod_mult, 1_000_000.0)
    _dow_adj = min(_tod_adj * dow_mult * vix_mult, 1_000_000.0)
    _exp_dollar_preview = expected_pnl * _dow_adj

    # Plain-text summary line (always visible regardless of HTML rendering)
    action_icon = '🟢' if sig.action == 'BUY' else '🔴'
    vix_str = f" | VIX: {vix_mult:.2f}x" if vix_mult != 1.0 else ""
    st.caption(
        f"**ML Quality** {action_icon} {sig.action} | Win: {win_prob:.0%} | "
        f"E.PnL: {expected_pnl:+.2%} | Score: {quality_score:.0f}/100 | "
        f"Size: {size_mult:.1f}x {size_label} | TOD: {tod_mult:.2f}x | DOW: {dow_mult:.2f}x{vix_str} | "
        f"**Combined: {combined_mult:.2f}x | Est. $: ${_exp_dollar_preview:+,.0f} on ${_dow_adj:,.0f} trade**"
    )

    st.markdown(
        f"""<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
        border:1px solid #334;border-radius:8px;padding:12px 16px;margin:8px 0;">
        <div style="text-align:center;font-size:12px;color:#888;letter-spacing:1px;margin-bottom:8px;">
        ML SIGNAL QUALITY</div>
        <div style="display:flex;justify-content:space-around;text-align:center;">
            <div>
                <div style="font-size:11px;color:#aaa;">Win Probability</div>
                <div style="font-size:28px;font-weight:700;color:{'#00c853' if win_prob >= 0.65 else '#ff9800' if win_prob >= 0.50 else '#ff1744'};">
                {win_prob:.0%}</div>
                <div style="font-size:11px;"><span style="background:{risk_color};color:#fff;
                padding:2px 8px;border-radius:10px;font-weight:600;">{risk_rating} RISK</span></div>
            </div>
            <div>
                <div style="font-size:11px;color:#aaa;">Expected P&L</div>
                <div style="font-size:28px;font-weight:700;color:{'#00c853' if expected_pnl > 0 else '#ff1744'};">
                {expected_pnl:+.2%}</div>
            </div>
            <div>
                <div style="font-size:11px;color:#aaa;">Quality Score</div>
                <div style="font-size:28px;font-weight:700;color:{q_color};">
                {quality_score:.0f}</div>
                <div style="background:#333;border-radius:4px;height:6px;width:80px;margin:4px auto;">
                <div style="background:{q_color};border-radius:4px;height:6px;width:{quality_score:.0f}%;"></div></div>
            </div>
            <div>
                <div style="font-size:11px;color:#aaa;">Position Size</div>
                <div style="font-size:28px;font-weight:700;color:{size_color};">
                {size_mult:.1f}x</div>
                <div style="font-size:11px;"><span style="background:{size_color};color:#fff;
                padding:2px 8px;border-radius:10px;font-weight:600;">{size_label}</span></div>
            </div>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Position Sizing Breakdown panel (TOD + DOW + VIX + ML combined, Arch418)
    base_capital = st.session_state.get('backtest_capital', 100_000.0)
    base_trade_usd = base_capital / 10.0  # Standard: capital/10
    max_trade_usd = 1_000_000.0  # Arch418: $1M cap
    # After bounce_cap (12x), base could range up to 12x. Use base_trade_usd as floor.
    # TOD/DOW/VIX applied post-cap, so multiply against max_trade_usd for worst case
    ml_adjusted_usd = min(base_trade_usd * _sq_mult, max_trade_usd)
    tod_adjusted_usd = min(ml_adjusted_usd * tod_mult, max_trade_usd)
    dow_adjusted_usd = min(tod_adjusted_usd * dow_mult * vix_mult, max_trade_usd)
    # Get current TSLA price for share count estimate (prefer real-time over stale bar)
    tsla_price = None
    if current_tsla is not None and len(current_tsla) > 0:
        tsla_price = float(current_tsla['close'].iloc[-1])
    try:
        _rt = _get_realtime_prices()
        if _rt.get('TSLA'):
            tsla_price = float(_rt['TSLA'])
    except Exception:
        pass
    est_shares = int(dow_adjusted_usd / tsla_price) if tsla_price and tsla_price > 0 else None

    # Price targets from signal stop/TP percentages
    if tsla_price and tsla_price > 0:
        if sig.action == 'BUY':
            pt_entry = tsla_price
            pt_tp = tsla_price * (1 + sig.suggested_tp_pct)
            pt_sl = tsla_price * (1 - sig.suggested_stop_pct)
            pt_tp_pct = sig.suggested_tp_pct
            pt_sl_pct = -sig.suggested_stop_pct
            pt_entry_label = 'Buy Long @'
            pt_tp_label = 'Take Profit (sell)'
            pt_sl_label = 'Stop Loss (sell)'
        else:
            pt_entry = tsla_price
            pt_tp = tsla_price * (1 - sig.suggested_tp_pct)
            pt_sl = tsla_price * (1 + sig.suggested_stop_pct)
            pt_tp_pct = -sig.suggested_tp_pct
            pt_sl_pct = sig.suggested_stop_pct
            pt_entry_label = 'Sell Short @'
            pt_tp_label = 'Take Profit (cover)'
            pt_sl_label = 'Stop Loss (cover)'
        pt_rr = sig.suggested_tp_pct / max(sig.suggested_stop_pct, 0.001)
        price_targets_html = (
            f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid #2a4a6a;'
            f'display:flex;justify-content:center;gap:28px;text-align:center;">'
            f'<div><div style="font-size:10px;color:#aaa;">{pt_entry_label}</div>'
            f'<div style="font-size:17px;font-weight:600;color:#ccc;">${pt_entry:.2f}</div></div>'
            f'<div style="font-size:18px;color:#555;align-self:center;">→</div>'
            f'<div><div style="font-size:10px;color:#aaa;">{pt_tp_label}</div>'
            f'<div style="font-size:17px;font-weight:600;color:#00c853;">${pt_tp:.2f}</div>'
            f'<div style="font-size:10px;color:#666;">{pt_tp_pct:+.2%}</div></div>'
            f'<div style="font-size:16px;color:#555;align-self:center;">|</div>'
            f'<div><div style="font-size:10px;color:#aaa;">{pt_sl_label}</div>'
            f'<div style="font-size:17px;font-weight:600;color:#ff4444;">${pt_sl:.2f}</div>'
            f'<div style="font-size:10px;color:#666;">{pt_sl_pct:+.2%}</div></div>'
            f'<div style="font-size:16px;color:#555;align-self:center;">|</div>'
            f'<div><div style="font-size:10px;color:#aaa;">R:R</div>'
            f'<div style="font-size:17px;font-weight:600;color:#fff;">{pt_rr:.1f}:1</div></div>'
            f'</div>'
        )
    else:
        price_targets_html = ''

    # Expected dollar P&L = expected_pnl_pct × estimated trade size
    expected_dollar = expected_pnl * dow_adjusted_usd
    expected_dollar_str = f"${expected_dollar:+,.0f}"

    tod_color = '#00c853' if tod_mult >= 1.40 else '#ff9800' if tod_mult >= 1.20 else '#888'
    dow_color = '#00c853' if dow_mult >= 1.40 else '#ff9800' if dow_mult >= 1.20 else '#888'
    vix_color = '#00c853' if vix_mult >= 1.20 else '#ff9800' if vix_mult > 1.0 else '#888'
    combined_color = '#00c853' if combined_mult >= 1.5 else '#ff9800' if combined_mult >= 1.2 else '#888'
    shares_str = f"~{est_shares:,} shares @ ${tsla_price:.0f}" if est_shares else "N/A"

    st.markdown(
        f"""<div style="background:linear-gradient(135deg,#0d1b2a,#1b2838);
        border:1px solid #2a4a6a;border-radius:8px;padding:12px 16px;margin:4px 0;">
        <div style="text-align:center;font-size:12px;color:#888;letter-spacing:1px;margin-bottom:8px;">
        POSITION SIZING BREAKDOWN</div>
        <div style="display:flex;justify-content:space-around;text-align:center;flex-wrap:wrap;gap:8px;">
            <div>
                <div style="font-size:10px;color:#aaa;">Base Trade</div>
                <div style="font-size:20px;font-weight:600;color:#ccc;">${base_trade_usd:,.0f}</div>
                <div style="font-size:10px;color:#666;">capital÷10</div>
            </div>
            <div style="font-size:18px;color:#555;align-self:center;">×</div>
            <div>
                <div style="font-size:10px;color:#aaa;">ML Score</div>
                <div style="font-size:20px;font-weight:600;color:{size_color};">{_sq_mult:.2f}x</div>
                <div style="font-size:10px;color:#666;">{_sq_size}</div>
            </div>
            <div style="font-size:18px;color:#555;align-self:center;">×</div>
            <div>
                <div style="font-size:10px;color:#aaa;">TOD Boost</div>
                <div style="font-size:20px;font-weight:600;color:{tod_color};">{tod_mult:.2f}x</div>
                <div style="font-size:10px;color:#666;">{tod_label}</div>
            </div>
            <div style="font-size:18px;color:#555;align-self:center;">×</div>
            <div>
                <div style="font-size:10px;color:#aaa;">DOW Boost</div>
                <div style="font-size:20px;font-weight:600;color:{dow_color};">{dow_mult:.2f}x</div>
                <div style="font-size:10px;color:#666;">{dow_label}</div>
            </div>
            <div style="font-size:18px;color:#555;align-self:center;">×</div>
            <div>
                <div style="font-size:10px;color:#aaa;">VIX Boost</div>
                <div style="font-size:20px;font-weight:600;color:{vix_color};">{vix_mult:.2f}x</div>
                <div style="font-size:10px;color:#666;">{vix_label}</div>
            </div>
            <div style="font-size:18px;color:#555;align-self:center;">=</div>
            <div>
                <div style="font-size:10px;color:#aaa;">Est. Trade Size</div>
                <div style="font-size:20px;font-weight:700;color:{combined_color};">${dow_adjusted_usd:,.0f}</div>
                <div style="font-size:10px;color:#888;">{shares_str}</div>
            </div>
            <div style="font-size:18px;color:#555;align-self:center;">→</div>
            <div>
                <div style="font-size:10px;color:#aaa;">Est. Profit</div>
                <div style="font-size:20px;font-weight:700;color:{'#00c853' if expected_dollar > 0 else '#ff1744'};">{expected_dollar_str}</div>
                <div style="font-size:10px;color:#888;">{expected_pnl:+.2%} return</div>
            </div>
        </div>
        {price_targets_html}
        </div>""",
        unsafe_allow_html=True,
    )

    # Expandable CV performance
    cv = model.cv_metrics
    if cv:
        with st.expander("Model CV Performance"):
            col1, col2, col3 = st.columns(3)
            col1.metric("AUC-ROC", f"{cv.get('overall_auc', 0):.3f}")
            col2.metric("Brier Score", f"{cv.get('overall_brier', 0):.3f}")
            col3.metric("PnL MAE", f"{cv.get('overall_pnl_mae', 0):.4f}")

            per_year = cv.get('per_year', {})
            if per_year:
                rows = []
                for yr, m in sorted(per_year.items()):
                    rows.append({
                        'Year': yr,
                        'Trades': m['n_trades'],
                        'WR': f"{m['win_rate']:.0%}",
                        'AUC': f"{m['auc']:.3f}",
                        'Brier': f"{m['brier']:.3f}",
                    })
                st.dataframe(rows, width="stretch", hide_index=True)

    # Debug log expander
    if log:
        with st.expander("ML Signal Quality — Debug Log"):
            st.code('\n'.join(log), language='text')


def _get_surfer_scanner(initial_capital: float = 100_000.0) -> 'SurferLiveScanner':
    """Get or create the SurferLiveScanner instance (persisted in session state)."""
    key = 'surfer_live_scanner'
    if key not in st.session_state:
        config = ScannerConfig(initial_capital=initial_capital)
        # Read Gist credentials for cloud persistence (Streamlit secrets or env vars)
        gist_id, github_token = '', ''
        try:
            gist_id = st.secrets.get('GIST_ID', '')
            github_token = st.secrets.get('GITHUB_TOKEN', '')
        except Exception:
            pass
        if not gist_id:
            import os
            gist_id = os.environ.get('GIST_ID', '')
            github_token = os.environ.get('GITHUB_TOKEN', '')
        st.session_state[key] = SurferLiveScanner(config, gist_id=gist_id, github_token=github_token)
    return st.session_state[key]


def _render_surfer_live_section(scanner, analysis, sig, current_price: float,
                                 current_tsla, run_analysis: bool):
    """Render the live scanner panel: positions, alerts, trade history."""
    st.subheader("Live Scanner")

    # Auto-refresh every 60s when positions are open for real-time stop/TP monitoring
    if scanner.positions:
        if AUTOREFRESH_AVAILABLE:
            st_autorefresh(interval=60_000, key="scanner_position_monitor")
            st.caption("🔄 Auto-checking stop/TP every 60s")
        else:
            st.info("Install streamlit-autorefresh for automatic 1-min stop/TP monitoring")

    col_cap, col_eq, col_unrealized = st.columns(3)
    unrealized = scanner.get_unrealized_pnl(current_price)
    col_cap.metric("Starting Capital", f"${scanner.config.initial_capital:,.0f}")
    col_eq.metric("Equity", f"${scanner.equity:,.0f}",
                  delta=f"${scanner.equity - scanner.config.initial_capital:+,.0f}")
    col_unrealized.metric("Unrealized P&L", f"${unrealized:+,.0f}")

    # Kill switch
    kill = st.checkbox("Kill Switch (suppress all entries)", value=scanner.config.kill_switch,
                       key="surfer_kill_switch")
    scanner.config.kill_switch = kill

    # --- Real-time exit monitoring (runs on EVERY render, not just analysis runs) ---
    if current_tsla is not None and len(current_tsla) > 0:
        # Get best available price — prefer real-time over stale bar
        rt_price = current_price
        try:
            _rt = _get_realtime_prices()
            if _rt.get('TSLA'):
                rt_price = float(_rt['TSLA'])
        except Exception:
            pass

        if run_analysis:
            # Full OHLC bar: catches intrabar wicks
            bar = current_tsla.iloc[-1]
            bar_high = float(bar.get('high', rt_price))
            bar_low = float(bar.get('low', rt_price))
        else:
            # Auto-refresh: use single-point current price
            bar_high = rt_price
            bar_low = rt_price

        # Check exits and fire audible+visual alert on hit
        if scanner.positions:
            exit_alerts = scanner.check_exits(rt_price, bar_high, bar_low)
            for ea in exit_alerts:
                _render_scanner_exit_alert(ea)
                _play_exit_alert_sound(ea.exit_reason)

        # Evaluate new entry signal (only on explicit analysis run)
        if run_analysis and sig.action != 'HOLD':
            entry_alert = scanner.evaluate_signal(analysis, rt_price)
            if entry_alert and entry_alert.alert_type == 'ENTRY':
                action_color = "#00e676" if entry_alert.action == 'BUY' else "#ff5252"
                st.markdown(
                    f'<div style="background:#1a2233;padding:10px;border-radius:6px;margin:4px 0;'
                    f'border:2px solid {action_color};">'
                    f'<b style="color:{action_color};font-size:16px">'
                    f'{entry_alert.action} ENTRY [{entry_alert.pos_id}]</b>  '
                    f'{entry_alert.shares} shares @ ${entry_alert.price:.2f} | '
                    f'Stop: ${entry_alert.stop_price:.2f} | '
                    f'TP: ${entry_alert.tp_price:.2f} | '
                    f'Notional: ${entry_alert.notional:,.0f}'
                    f'</div>', unsafe_allow_html=True,
                )
            elif entry_alert and entry_alert.alert_type == 'RISK_WARNING':
                st.warning(f"Scanner: {entry_alert.warning_msg}")

    # Open positions — show distance to stop and TP
    if scanner.positions:
        st.markdown("**Open Positions**")
        for pos in scanner.positions.values():
            if pos.direction == 'long':
                upnl = (current_price - pos.entry_price) * pos.shares
                dist_stop = (current_price - pos.stop_price) / current_price
                dist_tp = (pos.tp_price - current_price) / current_price
            else:
                upnl = (pos.entry_price - current_price) * pos.shares
                dist_stop = (pos.stop_price - current_price) / current_price
                dist_tp = (current_price - pos.tp_price) / current_price
            upnl_color = "#00e676" if upnl >= 0 else "#ff5252"
            stop_color = '#ff5252' if dist_stop < 0.003 else ('#ff9800' if dist_stop < 0.01 else '#888')
            st.markdown(
                f'<div style="background:#1a2233;padding:8px;border-radius:6px;margin:3px 0;">'
                f'[{pos.pos_id}] <b>{pos.direction.upper()}</b> {pos.shares}sh '
                f'@ ${pos.entry_price:.2f} | '
                f'<span style="color:{stop_color}">SL: ${pos.stop_price:.2f} ({dist_stop:.1%} away)</span> | '
                f'TP: ${pos.tp_price:.2f} ({dist_tp:.1%} away) | '
                f'Unrealized: <b style="color:{upnl_color}">${upnl:+,.0f}</b>'
                f'</div>', unsafe_allow_html=True,
            )
    else:
        st.caption("No open positions.")

    # Closed trade history
    if scanner.closed_trades:
        total_trades = len(scanner.closed_trades)
        total_pnl = sum(t.pnl for t in scanner.closed_trades)
        wins = sum(1 for t in scanner.closed_trades if t.pnl > 0)
        wr = wins / total_trades if total_trades > 0 else 0
        with st.expander(
            f"Trade History: {total_trades} trades | WR {wr:.0%} | Total P&L ${total_pnl:+,.0f}",
            expanded=False
        ):
            hist_rows = [{
                'ID': t.pos_id,
                'Dir': t.direction.upper(),
                'Entry $': f"${t.entry_price:.2f}",
                'Exit $': f"${t.exit_price:.2f}",
                'Shares': t.shares,
                'P&L': f"${t.pnl:+.0f}",
                'Hold (min)': f"{t.hold_minutes:.0f}",
                'Reason': t.exit_reason,
            } for t in reversed(scanner.closed_trades[-50:])]
            st.dataframe(pd.DataFrame(hist_rows), hide_index=True, width="stretch")

    # Reset button
    if st.button("Reset Scanner (clear all positions/history)", key="surfer_scanner_reset"):
        scanner.reset()
        st.rerun()


def show_channel_surfer_tab(
    current_tsla,
    native_tf_data,
    live_config,
    is_live,
    current_spy=None,
    current_vix=None,
):
    """Channel Surfer tab — physics-inspired multi-TF channel trading."""
    st.header("Channel Surfer")

    if not CHANNEL_SURFER_AVAILABLE:
        st.error("Channel Surfer module not available. Check v15/core/channel_surfer.py")
        return

    # yfinance pull log (sidebar expander for data transparency)
    try:
        from v15.data.native_tf import PULL_LOG as _yf_pull_log
        if _yf_pull_log:
            with st.sidebar.expander("yfinance Pull Log", expanded=False):
                st.code('\n'.join(list(_yf_pull_log)[-20:]), language='text')
    except Exception:
        pass

    # Auto-run or manual analysis
    auto_refresh = live_config.get('auto_refresh', False)

    # Staleness check: auto-re-run if last analysis is older than 5 minutes
    _last = st.session_state.get('surfer_last_update')
    _stale = _last is None or (datetime.now() - _last).total_seconds() > 300

    # Analysis button
    run_analysis = False
    if auto_refresh or _stale:
        run_analysis = True
        if _stale and not auto_refresh:
            st.caption("Auto-analyzing (analysis >5 min old)...")
        elif auto_refresh:
            st.caption("Auto-analyzing on refresh...")
    else:
        if st.button("Analyze Channels", type="primary", key="surfer_analyze"):
            run_analysis = True

    # Get cached result or run new analysis
    analysis = st.session_state.get('surfer_analysis')

    if run_analysis:
        with st.spinner("Running Channel Surfer analysis..."):
            try:
                analysis = prepare_multi_tf_analysis(
                    native_data=native_tf_data,
                    live_5min_tsla=current_tsla,
                    target_tfs=['5min', '1h', '4h', 'daily', 'weekly'],
                )
                st.session_state['surfer_analysis'] = analysis
                st.session_state['surfer_last_update'] = datetime.now()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if analysis is None:
        st.info("Click 'Analyze Channels' to run the Channel Surfer engine, or enable auto-refresh.")
        return

    # Show last update time
    last_update = st.session_state.get('surfer_last_update')
    if last_update:
        st.caption(f"Last analysis: {last_update.strftime('%H:%M:%S')}")

    sig = analysis.signal

    # --- Section 1: Signal Banner ---
    # Primary price: last 5-min bar close (already fetched, zero extra API calls).
    # Supplement: ticker.info for premarket/afterhours prices when bar is stale.
    _bar_price = float(current_tsla['close'].iloc[-1]) if current_tsla is not None and len(current_tsla) > 0 else 0.0
    current_price = _bar_price
    _price_source = "5m bar"

    # Detect premarket session
    _market_session = get_market_status()
    _is_premarket = False
    if _market_session and not _market_session.get('is_open', True):
        _time_et = _market_session.get('current_time_et', '')
        try:
            import re as _re
            _hm = _re.search(r'(\d{2}):(\d{2}):\d{2} ET', _time_et)
            if _hm:
                _h, _m = int(_hm.group(1)), int(_hm.group(2))
                _is_premarket = 4 <= _h < 9 or (_h == 9 and _m < 30)
        except Exception:
            pass

    # Try ticker.info for a fresher price (especially premarket/afterhours)
    try:
        _rt_prices = _get_realtime_prices()
        _rt_tsla = _rt_prices.get('TSLA')
        if _rt_tsla and _rt_tsla > 0:
            current_price = float(_rt_tsla)
            _price_source = "premarket" if _is_premarket else "live"
            print(f"[PRICE] Channel Surfer using ticker.info: ${current_price:.2f} ({_price_source})")
        else:
            print(f"[PRICE] Channel Surfer: ticker.info returned None for TSLA, using 5m bar ${_bar_price:.2f}")
    except Exception as e:
        print(f"[PRICE] Channel Surfer: ticker.info failed ({e}), using 5m bar ${_bar_price:.2f}")

    # Annotate bar source with freshness
    if _price_source == "5m bar" and current_tsla is not None and len(current_tsla) > 0:
        _last_bar_ts = current_tsla.index[-1]
        if hasattr(_last_bar_ts, 'date') and _last_bar_ts.date() == datetime.now().date():
            _price_source = "5m bar (premarket)" if _is_premarket else "5m bar"

    # Current price display
    if current_price > 0:
        prev_price = float(current_tsla['close'].iloc[-2]) if current_tsla is not None and len(current_tsla) > 1 else current_price
        price_delta = current_price - prev_price
        _premarket_note = " [PREMARKET]" if _is_premarket else ""
        st.metric("TSLA", f"${current_price:.2f}{_premarket_note}", delta=f"{price_delta:+.2f} ({price_delta/prev_price*100:+.2f}%)",
                  help=f"Source: {_price_source}. Signal analysis uses completed bars (last session close).")
        st.caption(f"Price updated: {datetime.now().strftime('%H:%M:%S')} ({_price_source})")

    _render_signal_banner(sig, current_price=current_price)

    # --- Market Insights Panel ---
    try:
        _render_market_insights(analysis)
    except Exception:
        pass

    # --- ML Signal Quality Panel ---
    _render_ml_signal_quality(analysis, sig, current_tsla, spy_df=current_spy, vix_df=current_vix)

    # --- Live Scanner Panel ---
    if SURFER_SCANNER_AVAILABLE:
        scanner_capital = st.number_input(
            "Scanner capital ($)", value=100_000, step=10_000,
            min_value=10_000, key="surfer_scanner_capital",
            help="Starting capital for hypothetical position tracking",
        )
        scanner = _get_surfer_scanner(initial_capital=float(scanner_capital))
        _render_surfer_live_section(
            scanner=scanner,
            analysis=analysis,
            sig=sig,
            current_price=current_price,
            current_tsla=current_tsla,
            run_analysis=run_analysis,
        )
        st.divider()

    # Play audio for BUY/SELL signals (only if new)
    prev_action = st.session_state.get('surfer_prev_action', 'HOLD')
    if sig.action != 'HOLD' and sig.action != prev_action:
        _play_alert_sound(sig.action)
    st.session_state['surfer_prev_action'] = sig.action

    # --- Regime indicator ---
    regime = getattr(analysis, 'regime', None)
    if regime is not None:
        regime_colors = {'ranging': '#4fc3f7', 'trending': '#ff9800', 'transitioning': '#ce93d8'}
        r_color = regime_colors.get(regime.regime, '#888')
        trend_arrow = '&#x2191;' if regime.trend_direction > 0 else ('&#x2193;' if regime.trend_direction < 0 else '&#x2194;')
        st.markdown(
            f"""<div style="text-align:center;margin:5px 0;font-size:13px;">
            <span style="color:{r_color};font-weight:700;">Market: {regime.regime.upper()}</span>
            <span style="color:#888;margin:0 10px;">|</span>
            <span style="color:#aaa;">Health: {regime.avg_health:.0%}</span>
            <span style="color:#888;margin:0 10px;">|</span>
            <span style="color:#aaa;">Trend: {trend_arrow}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # --- ML Predictions Section ---
    _show_ml_predictions(analysis, current_tsla, native_tf_data)

    # --- Break Direction Predictor (Phase 4 evolved heuristic) ---
    _render_break_predictor(analysis, native_tf_data, current_spy=current_spy, current_vix=current_vix)

    # --- Section 1b: 5min Channel Chart ---
    if current_tsla is not None and len(current_tsla) > 0 and 'close' in current_tsla.columns:
        _show_surfer_chart(current_tsla, analysis)

    # --- Section 2: Signal Score Breakdown ---
    sig_type_label = getattr(sig, 'signal_type', 'bounce').upper()
    st.subheader(f"Signal Components ({sig_type_label})")
    cols = st.columns(7)
    components = [
        ("Type", sig_type_label, "bounce = mean-reversion, break = breakout"),
        ("Position", sig.position_score, "Where price sits in channel"),
        ("Energy", sig.energy_score, "Momentum/energy ratio"),
        ("Entropy", sig.entropy_score, "Channel predictability"),
        ("Confluence", sig.confluence_score, "Multi-TF agreement"),
        ("Timing", sig.timing_score, "Oscillation phase alignment"),
        ("Health", sig.channel_health, "Channel structural integrity"),
    ]
    for col, (name, val, help_text) in zip(cols, components):
        with col:
            if isinstance(val, str):
                color = "#4fc3f7" if val == "BOUNCE" else "#ff9800"
                st.markdown(f"**{name}**")
                st.markdown(f"<span style='color:{color};font-weight:bold;font-size:20px;'>{val}</span>",
                           unsafe_allow_html=True)
            else:
                st.metric(name, f"{val:.0%}", help=help_text)

    # --- Section 3: Per-TF Channel Positions ---
    st.subheader("Channel Positions by Timeframe")

    if analysis.tf_states:
        # Sort by our preferred TF order
        tf_order = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly']
        sorted_tfs = sorted(
            analysis.tf_states.items(),
            key=lambda x: tf_order.index(x[0]) if x[0] in tf_order else 99,
        )

        # Visual position gauges
        for tf, state in sorted_tfs:
            if not state.valid:
                continue

            col1, col2, col3 = st.columns([1, 3, 2])
            with col1:
                dir_emoji = {'bull': '+', 'bear': '-', 'sideways': '~'}.get(state.channel_direction, '?')
                st.markdown(f"**{tf}** ({dir_emoji})")

            with col2:
                st.markdown(
                    _render_position_gauge(state.position_pct),
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            with col3:
                health_color = '#00c853' if state.channel_health > 0.6 else '#ffab40' if state.channel_health > 0.3 else '#ff1744'
                break_color = '#ff1744' if state.break_prob > 0.5 else '#ffab40' if state.break_prob > 0.3 else '#888'
                st.markdown(
                    f"<span style='color:{health_color};font-weight:bold;'>Health: {state.channel_health:.0%}</span> "
                    f"| <span style='color:{break_color}'>Break: {state.break_prob:.0%}</span> "
                    f"| OU: {state.ou_half_life:.0f}bars "
                    f"| R2: {state.r_squared:.2f}",
                    unsafe_allow_html=True,
                )

        # --- Section 4: Detailed TF Table ---
        with st.expander("Detailed Per-TF Analysis", expanded=False):
            table_data = []
            for tf, state in sorted_tfs:
                if not state.valid:
                    continue
                table_data.append({
                    'TF': tf,
                    'Pos': f"{state.position_pct:.0%}",
                    'Dir': state.channel_direction,
                    'Health': f"{state.channel_health:.0%}",
                    'OU theta': f"{state.ou_theta:.3f}",
                    'OU t1/2': f"{state.ou_half_life:.0f}",
                    'Revert': f"{state.ou_reversion_score:.0%}",
                    'Break%': f"{state.break_prob:.0%}",
                    'PE': f"{state.potential_energy:.2f}",
                    'KE': f"{state.kinetic_energy:.2f}",
                    'Bind': f"{state.binding_energy:.2f}",
                    'Entropy': f"{state.entropy:.2f}",
                    'R2': f"{state.r_squared:.2f}",
                    'Bounces': state.bounce_count,
                    'Width%': f"{state.width_pct:.2f}",
                })
            if table_data:
                st.dataframe(pd.DataFrame(table_data), hide_index=True, width="stretch")

        # --- Section 5: Confluence Matrix ---
        with st.expander("Multi-TF Confluence", expanded=False):
            conf_data = []
            for tf, score in analysis.confluence_matrix.items():
                if score > 0:
                    conf_data.append({
                        'Timeframe': tf,
                        'Alignment': f"{score:.0%}",
                        'Direction': ('Bullish' if score > 0.6 else 'Bearish' if score < 0.4 else 'Neutral'),
                    })
            if conf_data:
                st.dataframe(pd.DataFrame(conf_data), hide_index=True)
            else:
                st.info("No confluence data available")

        # --- Section 6: Energy Diagram ---
        with st.expander("Energy State Diagram", expanded=False):
            valid_states = [(tf, s) for tf, s in sorted_tfs if s.valid]
            if valid_states:
                import plotly.graph_objects as go

                tfs = [tf for tf, _ in valid_states]
                pe_vals = [s.potential_energy for _, s in valid_states]
                ke_vals = [s.kinetic_energy for _, s in valid_states]
                bind_vals = [s.binding_energy for _, s in valid_states]

                fig = go.Figure()
                fig.add_trace(go.Bar(name='Potential Energy', x=tfs, y=pe_vals, marker_color='#ff6b35'))
                fig.add_trace(go.Bar(name='Kinetic Energy', x=tfs, y=ke_vals, marker_color='#00b4d8'))
                fig.add_trace(go.Scatter(
                    name='Binding Energy', x=tfs, y=bind_vals,
                    mode='lines+markers', marker=dict(size=10, color='#e63946'),
                    line=dict(width=3, dash='dash'),
                ))
                fig.update_layout(
                    title='Channel Energy vs Binding Energy',
                    yaxis_title='Energy (0-1)',
                    barmode='stack',
                    template='plotly_dark',
                    height=350,
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "When total energy (PE+KE) exceeds binding energy, channel breakout is likely."
                )

    else:
        st.warning("No valid channels detected at any timeframe.")

    # --- Section 7: Signal History ---
    if 'surfer_history' not in st.session_state:
        st.session_state['surfer_history'] = []

    # Append current signal to history
    if analysis and analysis.signal.action != 'HOLD':
        history = st.session_state['surfer_history']
        entry = {
            'time': analysis.timestamp,
            'action': analysis.signal.action,
            'type': getattr(analysis.signal, 'signal_type', 'bounce'),
            'confidence': analysis.signal.confidence,
            'primary_tf': analysis.signal.primary_tf,
            'reason': analysis.signal.reason,
            'health': analysis.signal.channel_health,
        }
        # Don't add duplicate timestamps
        if not history or history[-1]['time'] != entry['time']:
            history.append(entry)
            # Keep last 100
            st.session_state['surfer_history'] = history[-100:]

    with st.expander("Signal History", expanded=False):
        history = st.session_state.get('surfer_history', [])
        if history:
            hist_df = pd.DataFrame(reversed(history))
            st.dataframe(hist_df, hide_index=True, width="stretch")
        else:
            st.info("No BUY/SELL signals recorded yet.")

    # --- Section 8: Quick Backtest ---
    with st.expander("Quick Backtest", expanded=False):
        st.caption("Run a quick backtest of Channel Surfer on recent 5min data")
        bt_col1, bt_col2 = st.columns(2)
        with bt_col1:
            bt_days = st.selectbox("Backtest period", [30, 60], index=1, key="bt_days")
        with bt_col2:
            bt_pos_size = st.number_input("Position size ($)", value=10000, step=5000, key="bt_pos")

        if st.button("Run Backtest", key="run_bt"):
            with st.spinner("Running Channel Surfer backtest..."):
                try:
                    from v15.core.surfer_backtest import run_backtest as surfer_backtest
                    metrics, trades, eq_curve = surfer_backtest(
                        days=bt_days,
                        eval_interval=6,  # Must match training default
                        max_hold_bars=60,
                        position_size=bt_pos_size,
                        min_confidence=0.45,
                        use_multi_tf=True,
                    )
                    st.session_state['bt_results'] = (metrics, trades, eq_curve)
                except Exception as e:
                    st.error(f"Backtest failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        bt_data = st.session_state.get('bt_results')
        if bt_data:
            metrics, trades, eq_curve = bt_data
            m_cols = st.columns(5)
            m_cols[0].metric("Trades", metrics.total_trades)
            m_cols[1].metric("Win Rate", f"{metrics.win_rate:.0%}")
            m_cols[2].metric("Profit Factor", f"{metrics.profit_factor:.1f}")
            m_cols[3].metric("Total P&L", f"${metrics.total_pnl:,.2f}")
            m_cols[4].metric("$/Trade", f"${metrics.expectancy:,.2f}")

            if eq_curve:
                eq_df = pd.DataFrame(eq_curve, columns=['bar', 'equity'])
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=eq_df['bar'], y=eq_df['equity'],
                    mode='lines+markers',
                    line=dict(color='#00c853', width=2),
                    marker=dict(size=6),
                    fill='tozeroy',
                    fillcolor='rgba(0,200,83,0.1)',
                ))
                fig_eq.update_layout(
                    title='Equity Curve',
                    xaxis_title='Bar', yaxis_title='Equity ($)',
                    template='plotly_dark', height=300,
                    margin=dict(l=50, r=20, t=40, b=30),
                )
                st.plotly_chart(fig_eq, width="stretch")

            # Signal type breakdown
            if trades:
                bounce_t = [t for t in trades if getattr(t, 'signal_type', 'bounce') == 'bounce']
                break_t = [t for t in trades if getattr(t, 'signal_type', 'bounce') == 'break']
                b_cols = st.columns(2)
                if bounce_t:
                    bwr = sum(1 for t in bounce_t if t.pnl > 0) / len(bounce_t)
                    bpnl = sum(t.pnl for t in bounce_t)
                    b_cols[0].metric("Bounce", f"{len(bounce_t)} trades | {bwr:.0%} WR | ${bpnl:,.0f}")
                if break_t:
                    kwr = sum(1 for t in break_t if t.pnl > 0) / len(break_t)
                    kpnl = sum(t.pnl for t in break_t)
                    b_cols[1].metric("Breakout", f"{len(break_t)} trades | {kwr:.0%} WR | ${kpnl:,.0f}")

                trade_data = [{
                    'Type': getattr(t, 'signal_type', 'bounce')[:3].upper(),
                    'Dir': t.direction,
                    'Entry$': f"${t.entry_price:.2f}",
                    'Exit$': f"${t.exit_price:.2f}",
                    'P&L': f"${t.pnl:+.2f}",
                    '%': f"{t.pnl_pct:+.2%}",
                    'Hold': f"{t.hold_bars}b",
                    'Reason': t.exit_reason,
                    'Conf': f"{t.confidence:.2f}",
                    'Size': f"${getattr(t, 'trade_size', 10000):,.0f}",
                } for t in trades]
                st.dataframe(pd.DataFrame(trade_data), hide_index=True, width="stretch")

                # --- MAE/MFE Scatter Plot ---
                maes = [getattr(t, 'mae_pct', 0) for t in trades]
                mfes = [getattr(t, 'mfe_pct', 0) for t in trades]
                if any(m > 0 for m in maes) and any(m > 0 for m in mfes):
                    with st.expander("Trade Quality (MAE/MFE)", expanded=False):
                        winners = [t for t in trades if t.pnl > 0]
                        losers = [t for t in trades if t.pnl <= 0]

                        q_cols = st.columns(4)
                        avg_mae = np.mean([m for m in maes if m > 0]) if any(m > 0 for m in maes) else 0
                        avg_mfe = np.mean([m for m in mfes if m > 0]) if any(m > 0 for m in mfes) else 0
                        win_eff_vals = [t.pnl_pct / max(t.mfe_pct, 1e-6) for t in winners if getattr(t, 'mfe_pct', 0) > 0]
                        win_eff = np.mean(win_eff_vals) if win_eff_vals else 0
                        q_cols[0].metric("Avg MAE", f"{avg_mae:.2%}")
                        q_cols[1].metric("Avg MFE", f"{avg_mfe:.2%}")
                        q_cols[2].metric("Win Efficiency", f"{win_eff:.0%}")
                        q_cols[3].metric("Max DD", f"{metrics.max_drawdown_pct:.1%}")

                        fig_mfe = go.Figure()
                        for label, group, color in [('Winners', winners, '#00c853'), ('Losers', losers, '#ff1744')]:
                            fig_mfe.add_trace(go.Scatter(
                                x=[getattr(t, 'mae_pct', 0) * 100 for t in group],
                                y=[getattr(t, 'mfe_pct', 0) * 100 for t in group],
                                mode='markers',
                                name=label,
                                marker=dict(color=color, size=8, opacity=0.7),
                                text=[f"{t.signal_type} {t.direction} {t.pnl_pct:+.2%}" for t in group],
                            ))
                        fig_mfe.update_layout(
                            title='MAE vs MFE (closer to top-left = better)',
                            xaxis_title='MAE % (worst drawdown)',
                            yaxis_title='MFE % (best unrealized)',
                            template='plotly_dark', height=350,
                            margin=dict(l=50, r=20, t=40, b=30),
                        )
                        st.plotly_chart(fig_mfe, width="stretch")


def show_model_comparisons_tab():
    """Read all model tags from the shared Gist and show a side-by-side performance comparison."""
    import json
    import urllib.request
    import os

    st.header("Model Comparisons")
    st.caption("Reads the shared Gist to compare live performance across all branches. Refreshes every hour.")

    # --- Load Gist credentials ---
    gist_id, github_token = '', ''
    try:
        gist_id = st.secrets.get('GIST_ID', '')
        github_token = st.secrets.get('GITHUB_TOKEN', '')
    except Exception:
        pass
    if not gist_id:
        gist_id = os.environ.get('GIST_ID', '')
        github_token = os.environ.get('GITHUB_TOKEN', '')

    if not gist_id or not github_token:
        st.warning("No Gist credentials found (GIST_ID / GITHUB_TOKEN). Add them to .streamlit/secrets.toml or environment variables.")
        return

    # --- Fetch full Gist ---
    full_data = {}
    try:
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json',
        }
        url = f'https://api.github.com/gists/{gist_id}'
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            gist_data = json.loads(resp.read().decode())
        content = gist_data.get('files', {}).get('surfer_scanner_state.json', {}).get('content', '')
        if content:
            full_data = json.loads(content)
    except Exception as e:
        st.error(f"Failed to load Gist: {e}")
        return

    # Filter to model keys only (skip metadata keys like _last_updated)
    model_keys = [k for k in full_data if not k.startswith('_')]
    if not model_keys:
        st.info("No model data in Gist yet. Start a live scanner session to populate it.")
        return

    last_updated = full_data.get('_last_updated', 'unknown')
    st.caption(f"Gist last updated: {last_updated}")

    # --- Compute stats per model ---
    def compute_stats(mdata: dict) -> dict:
        trades = mdata.get('closed_trades', [])
        equity = mdata.get('equity', 100_000.0)
        positions = mdata.get('positions', {})
        if not trades:
            return {
                'equity': equity, 'total_pnl': 0.0, 'n_trades': 0,
                'win_rate': 0.0, 'avg_pnl': 0.0, 'best_trade': 0.0,
                'worst_trade': 0.0, 'avg_hold_min': 0.0,
                'open_positions': len(positions), 'trades': [],
            }
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        holds = [t.get('hold_minutes', 0) for t in trades]
        return {
            'equity': equity,
            'total_pnl': sum(pnls),
            'n_trades': len(trades),
            'win_rate': len(wins) / len(pnls),
            'avg_pnl': sum(pnls) / len(pnls),
            'best_trade': max(pnls),
            'worst_trade': min(pnls),
            'avg_hold_min': sum(holds) / len(holds) if holds else 0.0,
            'open_positions': len(positions),
            'trades': trades,
        }

    stats = {k: compute_stats(full_data[k]) for k in model_keys}

    # --- Summary table ---
    st.subheader("Summary")
    rows = []
    for tag, s in stats.items():
        rows.append({
            'Model': tag,
            'Equity': f"${s['equity']:,.0f}",
            'Total P&L': f"${s['total_pnl']:+,.0f}",
            'Trades': s['n_trades'],
            'Win Rate': f"{s['win_rate']:.1%}" if s['n_trades'] > 0 else '—',
            'Avg P&L': f"${s['avg_pnl']:+,.0f}" if s['n_trades'] > 0 else '—',
            'Best Trade': f"${s['best_trade']:+,.0f}" if s['n_trades'] > 0 else '—',
            'Worst Trade': f"${s['worst_trade']:+,.0f}" if s['n_trades'] > 0 else '—',
            'Avg Hold': f"{s['avg_hold_min']:.0f} min" if s['n_trades'] > 0 else '—',
            'Open': s['open_positions'],
        })
    st.dataframe(rows, width="stretch", hide_index=True)

    # --- Equity curves ---
    any_trades = any(s['n_trades'] > 0 for s in stats.values())
    if any_trades:
        st.subheader("Equity Curves")
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            colors = ['#00c853', '#2196f3', '#ff9800', '#e91e63', '#9c27b0']
            for i, (tag, s) in enumerate(stats.items()):
                trades = sorted(s['trades'], key=lambda t: t.get('exit_time', ''))
                if not trades:
                    continue
                initial = 100_000.0
                times, equity_vals = [], []
                running = initial
                for t in trades:
                    running += t['pnl']
                    times.append(t.get('exit_time', ''))
                    equity_vals.append(running)
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=times, y=equity_vals, name=tag,
                    mode='lines+markers', line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{tag}</b><br>%{{x}}<br>Equity: $%{{y:,.0f}}<extra></extra>',
                ))
            fig.update_layout(
                template='plotly_dark',
                xaxis_title='Exit Time',
                yaxis_title='Equity ($)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                margin=dict(l=60, r=20, t=40, b=40),
                height=400,
            )
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"Equity chart unavailable: {e}")

    # --- Recent trades per model ---
    st.subheader("Recent Trades")
    cols = st.columns(len(model_keys))
    for col, tag in zip(cols, model_keys):
        s = stats[tag]
        with col:
            st.markdown(f"**{tag}**")
            trades = sorted(s['trades'], key=lambda t: t.get('exit_time', ''), reverse=True)[:10]
            if not trades:
                st.caption("No trades yet")
                continue
            for t in trades:
                pnl = t['pnl']
                color = '#00c853' if pnl >= 0 else '#ff5252'
                col.markdown(
                    f"<div style='font-size:12px;border-left:3px solid {color};"
                    f"padding:3px 8px;margin:2px 0;'>"
                    f"<b style='color:{color}'>${pnl:+,.0f}</b> "
                    f"<span style='color:#888'>{t.get('exit_reason','?')} · "
                    f"{t.get('hold_minutes',0):.0f}m</span></div>",
                    unsafe_allow_html=True,
                )

    # --- Open positions ---
    open_models = [(tag, full_data[tag].get('positions', {})) for tag in model_keys
                   if full_data[tag].get('positions')]
    if open_models:
        st.subheader("Open Positions")
        for tag, positions in open_models:
            st.markdown(f"**{tag}** — {len(positions)} open")
            for pos_id, pos in positions.items():
                direction_icon = '🟢' if pos.get('direction') == 'long' else '🔴'
                st.caption(
                    f"{direction_icon} {pos_id} | Entry ${pos.get('entry_price', 0):.2f} | "
                    f"TP ${pos.get('tp_price', 0):.2f} | Stop ${pos.get('stop_price', 0):.2f} | "
                    f"Notional ${pos.get('notional', 0):,.0f}"
                )


def main():
    st.title("X23 Channel Break Predictor — c12")

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
        # Clear session-state data caches so next rerun does full fetch
        for _k in ['_5min_data', '_native_tf_data', '_merged_5min']:
            st.session_state.pop(_k, None)
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
    # Uses session-state accumulators: full fetch on cold start, incremental on refreshes.
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

    # Navigation — radio buttons instead of st.tabs() to preserve active tab on rerun
    _TAB_NAMES = [
        "Live Prediction",
        "Channel Visualization",
        "Window Selection",
        "Feature Analysis",
        "Model Info",
        "Data Explorer",
        "Trading Monitor",
        "Channel Surfer",
        "Model Comparisons",
    ]
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = _TAB_NAMES[0]
    selected_tab = st.radio(
        "Navigation",
        _TAB_NAMES,
        index=_TAB_NAMES.index(st.session_state.get('active_tab', _TAB_NAMES[0])),
        horizontal=True,
        label_visibility='collapsed',
        key='main_tab_selector',
    )
    st.session_state['active_tab'] = selected_tab

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

    if selected_tab == "Live Prediction":
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

    elif selected_tab == "Channel Visualization":
        # Channel Visualization tab
        prediction = st.session_state.get('last_prediction')
        pred_data = st.session_state.get('prediction_data')

        native_tf = st.session_state.get('native_tf_data')
        if pred_data is not None:
            tsla_slice, _, _ = pred_data
            show_channel_visualization_tab(tsla_slice, prediction, native_bars_by_tf=native_tf)
        else:
            show_channel_visualization_tab(current_tsla, None, native_bars_by_tf=native_tf)

    elif selected_tab == "Window Selection":
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

    elif selected_tab == "Feature Analysis":
        st.header("Feature Analysis")

        if predictor is not None:
            top_k = st.slider("Top K Features", 10, 100, 50)
            show_feature_importance(predictor, top_k)
        else:
            st.info("Load a model to see feature importance")

    elif selected_tab == "Model Info":
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

    elif selected_tab == "Data Explorer":
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

    elif selected_tab == "Trading Monitor":
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

    elif selected_tab == "Channel Surfer":
        st_autorefresh(interval=60_000, key="channel_surfer_price_monitor")
        show_channel_surfer_tab(
            current_tsla=current_tsla,
            native_tf_data=native_tf_data,
            live_config=live_config,
            is_live=is_live,
            current_spy=current_spy,
            current_vix=current_vix,
        )

    elif selected_tab == "Model Comparisons":
        st_autorefresh(interval=3_600_000, key="model_comparisons_refresh")
        show_model_comparisons_tab()


if __name__ == "__main__":
    main()
