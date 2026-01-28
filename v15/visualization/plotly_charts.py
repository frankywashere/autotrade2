"""
v15/visualization/plotly_charts.py - Plotly-based visualizations for Streamlit dashboard.

Provides interactive chart components for channel analysis visualization.
Converted from matplotlib implementation in inspector.py and utils.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # type: ignore

if TYPE_CHECKING:
    from v15.core.channel import Channel, Touch


# =============================================================================
# CONSTANTS
# =============================================================================

CANDLE_UP_COLOR = '#26a69a'  # Green for bullish
CANDLE_DOWN_COLOR = '#ef5350'  # Red for bearish
CHANNEL_LINE_COLOR = 'rgba(255, 165, 0, 0.8)'  # Orange
CHANNEL_FILL_COLOR = 'rgba(255, 165, 0, 0.15)'  # Semi-transparent orange
CENTER_LINE_COLOR = 'rgba(0, 100, 255, 0.6)'  # Blue
BOUNCE_UPPER_COLOR = 'rgba(255, 0, 0, 0.8)'  # Red for upper touches
BOUNCE_LOWER_COLOR = 'rgba(0, 128, 0, 0.8)'  # Green for lower touches

# Direction labels
DIR_NAMES = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _check_plotly_available() -> None:
    """Raise ImportError if Plotly is not available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for these visualizations. "
            "Install it with: pip install plotly"
        )


def _validate_ohlcv_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has required OHLCV columns.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        return False
    required_cols = {'open', 'high', 'low', 'close'}
    return required_cols.issubset(set(df.columns))


def _validate_channel(channel: Optional['Channel']) -> bool:
    """
    Validate that channel object is usable.

    Args:
        channel: Channel object to validate

    Returns:
        True if valid and usable, False otherwise
    """
    if channel is None:
        return False
    if not getattr(channel, 'valid', False):
        return False
    # Check required attributes
    required_attrs = ['slope', 'intercept', 'std_dev', 'upper_line', 'lower_line', 'center_line']
    for attr in required_attrs:
        if not hasattr(channel, attr) or getattr(channel, attr) is None:
            return False
    return True


# =============================================================================
# CANDLESTICK CHART
# =============================================================================

def create_candlestick_chart(df: pd.DataFrame) -> 'go.Figure':
    """
    Create a Plotly candlestick chart from OHLCV DataFrame.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns.
            Optionally includes 'volume'. Index can be datetime or integer.

    Returns:
        Plotly Figure with candlestick chart

    Raises:
        ImportError: If Plotly is not installed
        ValueError: If DataFrame is empty or missing required columns

    Example:
        >>> df = pd.DataFrame({
        ...     'open': [100, 102, 101],
        ...     'high': [103, 104, 103],
        ...     'low': [99, 101, 100],
        ...     'close': [102, 101, 102]
        ... })
        >>> fig = create_candlestick_chart(df)
        >>> fig.show()  # In Streamlit: st.plotly_chart(fig)
    """
    _check_plotly_available()

    if not _validate_ohlcv_dataframe(df):
        raise ValueError(
            "DataFrame must have 'open', 'high', 'low', 'close' columns and be non-empty"
        )

    # Use index as x-axis (handles both datetime and integer indices)
    x_values = df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df)))

    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=x_values,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing=dict(line=dict(color=CANDLE_UP_COLOR), fillcolor=CANDLE_UP_COLOR),
        decreasing=dict(line=dict(color=CANDLE_DOWN_COLOR), fillcolor=CANDLE_DOWN_COLOR),
        name='Price'
    ))

    # Update layout for better appearance
    fig.update_layout(
        xaxis_rangeslider_visible=False,  # Hide range slider for cleaner look
        xaxis_title='Bar Index' if not isinstance(df.index, pd.DatetimeIndex) else 'Time',
        yaxis_title='Price',
        hovermode='x unified',
        showlegend=False,
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Disable autoscale on x-axis for consistent display
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)

    return fig


# =============================================================================
# CHANNEL OVERLAY
# =============================================================================

def add_channel_overlay(
    fig: 'go.Figure',
    channel: 'Channel',
    start_idx: int,
    project_forward: int = 0,
    line_color: str = CHANNEL_LINE_COLOR,
    fill_color: str = CHANNEL_FILL_COLOR,
    center_color: str = CENTER_LINE_COLOR,
) -> 'go.Figure':
    """
    Add channel boundary lines (upper/lower/center) to an existing Plotly figure.

    The channel's pre-computed line arrays (upper_line, lower_line, center_line)
    are plotted starting at start_idx. Optionally projects bounds forward using
    slope and intercept.

    Args:
        fig: Existing Plotly Figure to add channel overlay to
        channel: Channel object with:
            - slope: Regression slope
            - intercept: Regression intercept
            - std_dev: Standard deviation for bounds
            - upper_line: Upper boundary values (numpy array)
            - lower_line: Lower boundary values (numpy array)
            - center_line: Center regression line (numpy array)
        start_idx: Starting x-axis index for the channel lines
        project_forward: Number of bars to project bounds forward (default: 0)
        line_color: Color for upper/lower lines (default: orange)
        fill_color: Color for fill between bounds (default: semi-transparent orange)
        center_color: Color for center line (default: blue)

    Returns:
        The modified Figure with channel overlay added

    Raises:
        ImportError: If Plotly is not installed

    Note:
        Returns the figure unchanged if channel is None or invalid.

    Example:
        >>> fig = create_candlestick_chart(df)
        >>> fig = add_channel_overlay(fig, channel, start_idx=0)
    """
    _check_plotly_available()

    if not _validate_channel(channel):
        return fig

    # Get channel line arrays
    upper_line = np.asarray(channel.upper_line)
    lower_line = np.asarray(channel.lower_line)
    center_line = np.asarray(channel.center_line)

    channel_len = len(center_line)
    if channel_len == 0:
        return fig

    # X-axis values for the channel region
    x_channel = list(range(start_idx, start_idx + channel_len))

    # Add filled region between upper and lower bounds
    # Use fill='tonexty' by adding lower then upper with fill
    fig.add_trace(go.Scatter(
        x=x_channel,
        y=lower_line,
        mode='lines',
        line=dict(color=line_color, width=2, dash='dash'),
        name='Lower Bound',
        hoverinfo='skip',
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=x_channel,
        y=upper_line,
        mode='lines',
        line=dict(color=line_color, width=2, dash='dash'),
        name='Upper Bound',
        fill='tonexty',
        fillcolor=fill_color,
        hoverinfo='skip',
        showlegend=False,
    ))

    # Add center line (dashed)
    fig.add_trace(go.Scatter(
        x=x_channel,
        y=center_line,
        mode='lines',
        line=dict(color=center_color, width=1.5, dash='dot'),
        name='Center Line',
        hoverinfo='skip',
        showlegend=False,
    ))

    # Project forward if requested
    if project_forward > 0:
        proj_start = start_idx + channel_len
        proj_x = list(range(proj_start, proj_start + project_forward))

        # Calculate projection using slope and intercept
        # x values for regression continue from where channel ended
        regression_x = np.arange(channel_len, channel_len + project_forward)
        proj_center = channel.slope * regression_x + channel.intercept
        proj_upper = proj_center + 2 * channel.std_dev
        proj_lower = proj_center - 2 * channel.std_dev

        # Ensure prices don't go negative
        proj_center = np.maximum(proj_center, 0.01)
        proj_upper = np.maximum(proj_upper, 0.01)
        proj_lower = np.maximum(proj_lower, 0.01)

        # Add projected bounds with lighter styling
        proj_line_color = 'rgba(255, 165, 0, 0.5)'
        proj_fill_color = 'rgba(255, 165, 0, 0.05)'

        fig.add_trace(go.Scatter(
            x=proj_x,
            y=proj_lower,
            mode='lines',
            line=dict(color=proj_line_color, width=1.5, dash='dot'),
            name='Projected Lower',
            hoverinfo='skip',
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=proj_x,
            y=proj_upper,
            mode='lines',
            line=dict(color=proj_line_color, width=1.5, dash='dot'),
            name='Projected Upper',
            fill='tonexty',
            fillcolor=proj_fill_color,
            hoverinfo='skip',
            showlegend=False,
        ))

    return fig


# =============================================================================
# BOUNCE MARKERS
# =============================================================================

def add_bounce_markers(
    fig: 'go.Figure',
    channel: 'Channel',
    df: pd.DataFrame,
    start_idx: int = 0,
    upper_color: str = BOUNCE_UPPER_COLOR,
    lower_color: str = BOUNCE_LOWER_COLOR,
    marker_size: int = 12,
) -> 'go.Figure':
    """
    Add markers at bounce points where price touched channel boundaries.

    Markers are placed at the actual touch prices (HIGH for upper touches,
    LOW for lower touches) from the channel's touches list.

    Args:
        fig: Existing Plotly Figure to add markers to
        channel: Channel object with touches: List[Touch], where each Touch has:
            - bar_index: int - Index within the channel window
            - touch_type: TouchType - UPPER (1) or LOWER (0)
            - price: float - The touch price
        df: DataFrame with price data (used to verify indices)
        start_idx: Starting index offset for the channel in the plot (default: 0)
        upper_color: Color for upper boundary touch markers (default: red)
        lower_color: Color for lower boundary touch markers (default: green)
        marker_size: Size of touch markers (default: 12)

    Returns:
        The modified Figure with bounce markers added

    Raises:
        ImportError: If Plotly is not installed

    Note:
        Returns the figure unchanged if channel is None/invalid or has no touches.

    Example:
        >>> fig = create_candlestick_chart(df)
        >>> fig = add_channel_overlay(fig, channel, start_idx=0)
        >>> fig = add_bounce_markers(fig, channel, df)
    """
    _check_plotly_available()

    if not _validate_channel(channel):
        return fig

    touches = getattr(channel, 'touches', None)
    if not touches or len(touches) == 0:
        return fig

    # Separate upper and lower touches
    upper_x = []
    upper_y = []
    lower_x = []
    lower_y = []

    n_bars = len(df) if df is not None else float('inf')

    for touch in touches:
        # Calculate plot x position (touch.bar_index is relative to channel window)
        plot_x = start_idx + touch.bar_index

        # Skip if outside valid range
        if plot_x < 0 or plot_x >= n_bars:
            continue

        # TouchType.UPPER = 1, TouchType.LOWER = 0
        touch_type = touch.touch_type
        if hasattr(touch_type, 'value'):
            touch_type = touch_type.value

        if touch_type == 1:  # Upper touch
            upper_x.append(plot_x)
            upper_y.append(touch.price)
        else:  # Lower touch
            lower_x.append(plot_x)
            lower_y.append(touch.price)

    # Add upper touch markers (triangles pointing down - price hit ceiling)
    if upper_x:
        fig.add_trace(go.Scatter(
            x=upper_x,
            y=upper_y,
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=marker_size,
                color=upper_color,
                line=dict(color='white', width=1)
            ),
            name='Upper Touches',
            hovertemplate='Upper Touch<br>Bar: %{x}<br>Price: %{y:.2f}<extra></extra>',
            showlegend=False,
        ))

    # Add lower touch markers (triangles pointing up - price hit floor)
    if lower_x:
        fig.add_trace(go.Scatter(
            x=lower_x,
            y=lower_y,
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=marker_size,
                color=lower_color,
                line=dict(color='white', width=1)
            ),
            name='Lower Touches',
            hovertemplate='Lower Touch<br>Bar: %{x}<br>Price: %{y:.2f}<extra></extra>',
            showlegend=False,
        ))

    return fig


# =============================================================================
# COMPLETE TIMEFRAME CHART
# =============================================================================

def create_tf_channel_chart(
    df: pd.DataFrame,
    channel: 'Channel',
    tf_name: str,
    duration: float,
    confidence: float,
    show_bounces: bool = True,
    project_forward: int = 0,
    height: int = 400,
) -> 'go.Figure':
    """
    Create a complete chart for one timeframe showing candlesticks + channel + bounces.

    This is a convenience function that combines create_candlestick_chart,
    add_channel_overlay, and add_bounce_markers into a single call with
    appropriate title and formatting.

    Args:
        df: DataFrame with OHLCV data for the timeframe
        channel: Channel object with all required attributes
        tf_name: Timeframe name for title (e.g., '5min', '1h', 'daily')
        duration: Predicted channel duration in bars (for title display)
        confidence: Model confidence score 0-1 (for title display)
        show_bounces: Whether to show bounce markers (default: True)
        project_forward: Number of bars to project channel forward (default: 0)
        height: Chart height in pixels (default: 400)

    Returns:
        Complete Plotly Figure with candlesticks, channel, and optionally bounces

    Raises:
        ImportError: If Plotly is not installed
        ValueError: If DataFrame is empty or invalid

    Note:
        If channel is None or invalid, returns just the candlestick chart
        with an indication in the title.

    Example:
        >>> fig = create_tf_channel_chart(
        ...     df=df_hourly,
        ...     channel=detected_channel,
        ...     tf_name='1h',
        ...     duration=45.5,
        ...     confidence=0.82
        ... )
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    _check_plotly_available()

    if not _validate_ohlcv_dataframe(df):
        raise ValueError(
            "DataFrame must have 'open', 'high', 'low', 'close' columns and be non-empty"
        )

    # Create base candlestick chart
    fig = create_candlestick_chart(df)

    # Determine channel validity and direction for title
    channel_valid = _validate_channel(channel)

    if channel_valid:
        # Get direction name
        direction_val = channel.direction
        if hasattr(direction_val, 'value'):
            direction_val = direction_val.value
        direction_str = DIR_NAMES.get(direction_val, 'UNKNOWN')

        # Get quality metrics
        r_squared = getattr(channel, 'r_squared', 0.0)
        bounce_count = getattr(channel, 'bounce_count', 0)
        window = getattr(channel, 'window', len(df))

        # Build title
        title = (
            f"<b>{tf_name}</b> | {direction_str} | "
            f"Duration: {duration:.1f} bars | Confidence: {confidence:.1%}<br>"
            f"<span style='font-size:12px'>R2: {r_squared:.3f} | "
            f"Bounces: {bounce_count} | Window: {window}</span>"
        )

        # Add channel overlay
        # Channel starts at bar 0 (assuming df is already sliced to show channel period)
        fig = add_channel_overlay(fig, channel, start_idx=0, project_forward=project_forward)

        # Add bounce markers
        if show_bounces:
            fig = add_bounce_markers(fig, channel, df, start_idx=0)
    else:
        # No valid channel
        title = (
            f"<b>{tf_name}</b> | NO VALID CHANNEL<br>"
            f"<span style='font-size:12px'>Duration: {duration:.1f} bars | "
            f"Confidence: {confidence:.1%}</span>"
        )

    # Update layout with title and sizing
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=14)
        ),
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


# =============================================================================
# MULTI-TIMEFRAME GRID
# =============================================================================

def create_multi_tf_chart(
    tf_data: List[dict],
    cols: int = 2,
    height_per_row: int = 350,
) -> 'go.Figure':
    """
    Create a grid of charts for multiple timeframes.

    Args:
        tf_data: List of dicts, each with keys:
            - df: DataFrame with OHLCV data
            - channel: Channel object (can be None)
            - tf_name: Timeframe name string
            - duration: Predicted duration
            - confidence: Confidence score
        cols: Number of columns in grid (default: 2)
        height_per_row: Height per row in pixels (default: 350)

    Returns:
        Plotly Figure with subplots grid

    Raises:
        ImportError: If Plotly is not installed
        ValueError: If tf_data is empty

    Example:
        >>> tf_data = [
        ...     {'df': df_5min, 'channel': ch_5min, 'tf_name': '5min',
        ...      'duration': 20, 'confidence': 0.8},
        ...     {'df': df_1h, 'channel': ch_1h, 'tf_name': '1h',
        ...      'duration': 45, 'confidence': 0.75},
        ... ]
        >>> fig = create_multi_tf_chart(tf_data)
    """
    _check_plotly_available()

    if not tf_data:
        raise ValueError("tf_data cannot be empty")

    n_charts = len(tf_data)
    rows = (n_charts + cols - 1) // cols  # Ceiling division

    # Create subplot grid
    subplot_titles = [d.get('tf_name', f'TF {i+1}') for i, d in enumerate(tf_data)]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    for i, data in enumerate(tf_data):
        row = i // cols + 1
        col = i % cols + 1

        df = data.get('df')
        channel = data.get('channel')

        if df is None or not _validate_ohlcv_dataframe(df):
            continue

        # Use index as x-axis
        x_values = list(range(len(df)))

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=x_values,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                increasing=dict(line=dict(color=CANDLE_UP_COLOR)),
                decreasing=dict(line=dict(color=CANDLE_DOWN_COLOR)),
                showlegend=False,
            ),
            row=row,
            col=col
        )

        # Add channel if valid
        if _validate_channel(channel):
            upper_line = np.asarray(channel.upper_line)
            lower_line = np.asarray(channel.lower_line)
            center_line = np.asarray(channel.center_line)
            channel_len = len(center_line)
            x_channel = list(range(channel_len))

            # Lower bound
            fig.add_trace(
                go.Scatter(
                    x=x_channel,
                    y=lower_line,
                    mode='lines',
                    line=dict(color=CHANNEL_LINE_COLOR, width=1.5, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip',
                ),
                row=row,
                col=col
            )

            # Upper bound with fill
            fig.add_trace(
                go.Scatter(
                    x=x_channel,
                    y=upper_line,
                    mode='lines',
                    line=dict(color=CHANNEL_LINE_COLOR, width=1.5, dash='dash'),
                    fill='tonexty',
                    fillcolor=CHANNEL_FILL_COLOR,
                    showlegend=False,
                    hoverinfo='skip',
                ),
                row=row,
                col=col
            )

    # Update layout
    total_height = rows * height_per_row
    fig.update_layout(
        height=total_height,
        showlegend=False,
        template='plotly_white',
    )

    # Hide range sliders for all x-axes
    for i in range(1, n_charts + 1):
        fig.update_xaxes(rangeslider_visible=False, row=(i-1)//cols + 1, col=(i-1)%cols + 1)

    return fig
