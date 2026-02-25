"""
v15/visualization/plotly_charts.py - Plotly-based visualizations for Streamlit dashboard.

Uses integer bar indices for the x-axis (like the inspector) to avoid all
datetime gap / rangebreak / timezone complexity.  Date labels are shown as
x-axis tick text at a few positions including the last bar.
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

# Price label font size
PRICE_LABEL_SIZE = 9


def _add_price_labels(
    fig: 'go.Figure',
    x_vals: list,
    y_upper: np.ndarray,
    y_lower: np.ndarray,
    color: str = 'rgba(120,120,120,0.8)',
    max_labels: int = 6,
) -> None:
    """Add price annotations along upper/lower channel lines at regular intervals."""
    n = len(x_vals)
    if n == 0:
        return

    # Pick evenly spaced indices, always including last
    if n <= max_labels:
        indices = list(range(n))
    else:
        step = max(1, (n - 1) // (max_labels - 1))
        indices = list(range(0, n, step))
        if indices[-1] != n - 1:
            indices.append(n - 1)

    lbl_x = [x_vals[i] for i in indices]
    lbl_upper = [float(y_upper[i]) for i in indices]
    lbl_lower = [float(y_lower[i]) for i in indices]

    fmt = lambda p: f"${p:,.2f}"

    fig.add_trace(go.Scatter(
        x=lbl_x, y=lbl_upper,
        mode='text',
        text=[fmt(p) for p in lbl_upper],
        textposition='top center',
        textfont=dict(size=PRICE_LABEL_SIZE, color=color),
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=lbl_x, y=lbl_lower,
        mode='text',
        text=[fmt(p) for p in lbl_lower],
        textposition='bottom center',
        textfont=dict(size=PRICE_LABEL_SIZE, color=color),
        showlegend=False, hoverinfo='skip',
    ))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_ts(ts: pd.Timestamp) -> str:
    """Format a timestamp for tick label display."""
    if hasattr(ts, 'hour') and (ts.hour != 0 or ts.minute != 0):
        return ts.strftime('%b %d %H:%M')
    return ts.strftime('%b %d')


def _make_date_ticks(df: pd.DataFrame, n_ticks: int = 8):
    """Build tick positions and labels from a DataFrame with DatetimeIndex.

    Returns (tickvals, ticktext) for use with fig.update_xaxes().
    If the index is not a DatetimeIndex, returns (None, None).
    """
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) == 0:
        return None, None

    n = len(df)
    if n <= n_ticks:
        positions = list(range(n))
    else:
        step = (n - 1) / (n_ticks - 1)
        positions = [round(i * step) for i in range(n_ticks)]
        # Always include the last bar
        if positions[-1] != n - 1:
            positions[-1] = n - 1

    # Convert tz-aware to ET for display
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert('America/New_York')

    labels = [_format_ts(idx[p]) for p in positions]
    return positions, labels


def _check_plotly_available() -> None:
    """Raise ImportError if Plotly is not available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for these visualizations. "
            "Install it with: pip install plotly"
        )


def _validate_ohlcv_dataframe(df: pd.DataFrame) -> bool:
    """Validate that DataFrame has required OHLCV columns."""
    if df is None or df.empty:
        return False
    required_cols = {'open', 'high', 'low', 'close'}
    return required_cols.issubset(set(df.columns))


def _validate_channel(channel: Optional['Channel']) -> bool:
    """Validate that channel object is usable."""
    if channel is None:
        return False
    if not getattr(channel, 'valid', False):
        return False
    required_attrs = ['slope', 'intercept', 'std_dev', 'upper_line', 'lower_line', 'center_line']
    for attr in required_attrs:
        if not hasattr(channel, attr) or getattr(channel, attr) is None:
            return False
    return True


# =============================================================================
# CANDLESTICK CHART
# =============================================================================

def create_candlestick_chart(df: pd.DataFrame, tf_name: Optional[str] = None) -> 'go.Figure':
    """
    Create a Plotly candlestick chart from OHLCV DataFrame.

    Uses integer bar indices for the x-axis so bars are always evenly spaced
    with no gaps.  A few date labels are shown as tick text.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns.
        tf_name: Timeframe name (unused, kept for API compat).

    Returns:
        Plotly Figure with candlestick chart
    """
    _check_plotly_available()

    if not _validate_ohlcv_dataframe(df):
        raise ValueError(
            "DataFrame must have 'open', 'high', 'low', 'close' columns and be non-empty"
        )

    x_values = list(range(len(df)))

    fig = go.Figure()

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

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        yaxis_title='Price',
        hovermode='x unified',
        showlegend=False,
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Date tick labels
    tickvals, ticktext = _make_date_ticks(df)
    if tickvals is not None:
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)

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
    **_kwargs,
) -> 'go.Figure':
    """
    Add channel boundary lines (upper/lower/center) to an existing Plotly figure.

    Uses integer bar indices for x-axis alignment (no datetime mapping needed).

    Args:
        fig: Existing Plotly Figure to add channel overlay to
        channel: Channel object with pre-computed line arrays
        start_idx: Starting x-axis index for the channel lines
        project_forward: Number of bars to project bounds forward (default: 0)
        line_color: Color for upper/lower lines
        fill_color: Color for fill between bounds
        center_color: Color for center line
    """
    _check_plotly_available()

    if not _validate_channel(channel):
        return fig

    upper_line = np.asarray(channel.upper_line)
    lower_line = np.asarray(channel.lower_line)
    center_line = np.asarray(channel.center_line)

    channel_len = len(center_line)
    if channel_len == 0:
        return fig

    x_channel = list(range(start_idx, start_idx + channel_len))

    # Lower bound
    fig.add_trace(go.Scatter(
        x=x_channel, y=lower_line,
        mode='lines',
        line=dict(color=line_color, width=2, dash='dash'),
        name='Lower Bound',
        hoverinfo='skip', showlegend=False,
    ))

    # Upper bound with fill
    fig.add_trace(go.Scatter(
        x=x_channel, y=upper_line,
        mode='lines',
        line=dict(color=line_color, width=2, dash='dash'),
        name='Upper Bound',
        fill='tonexty', fillcolor=fill_color,
        hoverinfo='skip', showlegend=False,
    ))

    # Center line
    fig.add_trace(go.Scatter(
        x=x_channel, y=center_line,
        mode='lines',
        line=dict(color=center_color, width=1.5, dash='dot'),
        name='Center Line',
        hoverinfo='skip', showlegend=False,
    ))

    # Price labels on main channel
    _add_price_labels(fig, x_channel, upper_line, lower_line,
                      color=line_color, max_labels=5)

    # Project forward if requested
    if project_forward > 0:
        proj_start = start_idx + channel_len
        proj_x = list(range(proj_start, proj_start + project_forward))

        regression_x = np.arange(channel_len, channel_len + project_forward)
        proj_center = channel.slope * regression_x + channel.intercept
        proj_upper = proj_center + 2 * channel.std_dev
        proj_lower = proj_center - 2 * channel.std_dev

        proj_center = np.maximum(proj_center, 0.01)
        proj_upper = np.maximum(proj_upper, 0.01)
        proj_lower = np.maximum(proj_lower, 0.01)

        proj_line_color = 'rgba(255, 165, 0, 0.5)'
        proj_fill_color = 'rgba(255, 165, 0, 0.05)'

        fig.add_trace(go.Scatter(
            x=proj_x, y=proj_lower,
            mode='lines',
            line=dict(color=proj_line_color, width=1.5, dash='dot'),
            name='Projected Lower',
            hoverinfo='skip', showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=proj_x, y=proj_upper,
            mode='lines',
            line=dict(color=proj_line_color, width=1.5, dash='dot'),
            name='Projected Upper',
            fill='tonexty', fillcolor=proj_fill_color,
            hoverinfo='skip', showlegend=False,
        ))

        # Price labels on projection
        _add_price_labels(fig, proj_x, proj_upper, proj_lower,
                          color=proj_line_color, max_labels=4)

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
    **_kwargs,
) -> 'go.Figure':
    """
    Add markers at bounce points where price touched channel boundaries.

    Uses integer bar indices for x-axis alignment.
    """
    _check_plotly_available()

    if not _validate_channel(channel):
        return fig

    touches = getattr(channel, 'touches', None)
    if not touches or len(touches) == 0:
        return fig

    upper_x, upper_y = [], []
    lower_x, lower_y = [], []

    n_bars = len(df) if df is not None else float('inf')

    for touch in touches:
        plot_x = start_idx + touch.bar_index

        if plot_x < 0 or plot_x >= n_bars:
            continue

        touch_type = touch.touch_type
        if hasattr(touch_type, 'value'):
            touch_type = touch_type.value

        if touch_type == 1:  # Upper
            upper_x.append(plot_x)
            upper_y.append(touch.price)
        else:  # Lower
            lower_x.append(plot_x)
            lower_y.append(touch.price)

    if upper_x:
        fig.add_trace(go.Scatter(
            x=upper_x, y=upper_y,
            mode='markers',
            marker=dict(symbol='triangle-down', size=marker_size,
                        color=upper_color, line=dict(color='white', width=1)),
            name='Upper Touches',
            hovertemplate='Upper Touch<br>Bar: %{x}<br>Price: %{y:.2f}<extra></extra>',
            showlegend=False,
        ))

    if lower_x:
        fig.add_trace(go.Scatter(
            x=lower_x, y=lower_y,
            mode='markers',
            marker=dict(symbol='triangle-up', size=marker_size,
                        color=lower_color, line=dict(color='white', width=1)),
            name='Lower Touches',
            hovertemplate='Lower Touch<br>Bar: %{x}<br>Price: %{y:.2f}<extra></extra>',
            showlegend=False,
        ))

    return fig


# =============================================================================
# DURATION PROJECTION
# =============================================================================

def add_duration_projection(
    fig: 'go.Figure',
    channel: 'Channel',
    chart_df: pd.DataFrame,
    duration_mean: float,
    duration_std: float,
    confidence: float,
    direction: Optional[str] = None,
    **_kwargs,
) -> 'go.Figure':
    """
    Add a duration projection overlay to a timeframe chart.

    Extends channel bounds forward using slope extrapolation, colored by
    predicted break direction.  Uses integer x-axis indices starting after
    the last candle bar.
    """
    _check_plotly_available()

    if not _validate_channel(channel):
        return fig

    n_forward = max(1, round(duration_mean))
    window = len(channel.center_line)

    # Future x: integers starting after the last bar
    start = len(chart_df)
    future_x = list(range(start, start + n_forward))

    # Project channel bounds forward
    regression_x = np.arange(window, window + n_forward)
    proj_center = channel.slope * regression_x + channel.intercept
    proj_upper = proj_center + 2 * channel.std_dev
    proj_lower = proj_center - 2 * channel.std_dev

    proj_upper = np.maximum(proj_upper, 0.01)
    proj_lower = np.maximum(proj_lower, 0.01)

    # Direction-based colors
    if direction == 'up':
        line_clr = 'rgba(0,180,0,0.5)'
        fill_clr = 'rgba(0,180,0,0.07)'
        vline_clr = 'green'
    elif direction == 'down':
        line_clr = 'rgba(200,0,0,0.5)'
        fill_clr = 'rgba(200,0,0,0.07)'
        vline_clr = 'red'
    else:
        line_clr = 'rgba(255,165,0,0.5)'
        fill_clr = 'rgba(255,165,0,0.05)'
        vline_clr = 'gray'

    fig.add_trace(go.Scatter(
        x=future_x, y=proj_lower,
        mode='lines',
        line=dict(color=line_clr, width=1.5, dash='dot'),
        hoverinfo='skip', showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=future_x, y=proj_upper,
        mode='lines',
        line=dict(color=line_clr, width=1.5, dash='dot'),
        fill='tonexty', fillcolor=fill_clr,
        hoverinfo='skip', showlegend=False,
    ))

    # Price labels along projection
    _add_price_labels(fig, future_x, proj_upper, proj_lower,
                      color=line_clr, max_labels=5)

    # Vertical marker at mean break point
    mean_idx = min(round(duration_mean) - 1, n_forward - 1)
    mean_x = future_x[max(0, mean_idx)]

    fig.add_vline(
        x=mean_x,
        line_dash='dash', line_color=vline_clr,
        line_width=1.5, opacity=min(confidence + 0.3, 1.0),
    )

    std_label = f" +/- {duration_std:.0f}" if duration_std > 0 else ""
    fig.add_annotation(
        x=mean_x, y=1.02, yref='paper',
        text=f"Break: {duration_mean:.0f}{std_label}",
        showarrow=False,
        font=dict(size=11, color=vline_clr),
    )

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

    Uses integer bar indices for the x-axis with date tick labels.
    """
    _check_plotly_available()

    if not _validate_ohlcv_dataframe(df):
        raise ValueError(
            "DataFrame must have 'open', 'high', 'low', 'close' columns and be non-empty"
        )

    # Create base candlestick chart (uses integer x internally)
    fig = create_candlestick_chart(df, tf_name=tf_name)

    channel_valid = _validate_channel(channel)

    if channel_valid:
        direction_val = channel.direction
        if hasattr(direction_val, 'value'):
            direction_val = direction_val.value
        direction_str = DIR_NAMES.get(direction_val, 'UNKNOWN')

        r_squared = getattr(channel, 'r_squared', 0.0)
        bounce_count = getattr(channel, 'bounce_count', 0)
        window = getattr(channel, 'window', len(df))

        title = (
            f"<b>{tf_name}</b> | {direction_str} | "
            f"Duration: {duration:.1f} bars | Confidence: {confidence:.1%}<br>"
            f"<span style='font-size:12px'>R2: {r_squared:.3f} | "
            f"Bounces: {bounce_count} | Window: {window}</span>"
        )

        # Right-align: channel was detected on the LAST `window` bars,
        # but chart may show more bars due to MIN_CHART_BARS.
        # The channel's bar 0 corresponds to chart bar (n - 1 - channel_len),
        # where n-1 is the last candle (current bar sits after the channel).
        channel_len = len(channel.center_line)
        channel_start = max(0, len(df) - 1 - channel_len)

        fig = add_channel_overlay(
            fig, channel, start_idx=channel_start, project_forward=project_forward,
        )

        if show_bounces:
            fig = add_bounce_markers(fig, channel, df, start_idx=channel_start)
    else:
        title = (
            f"<b>{tf_name}</b> | NO VALID CHANNEL<br>"
            f"<span style='font-size:12px'>Duration: {duration:.1f} bars | "
            f"Confidence: {confidence:.1%}</span>"
        )

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
    """
    _check_plotly_available()

    if not tf_data:
        raise ValueError("tf_data cannot be empty")

    n_charts = len(tf_data)
    rows = (n_charts + cols - 1) // cols

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

        x_values = list(range(len(df)))

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

        if _validate_channel(channel):
            upper_line = np.asarray(channel.upper_line)
            lower_line = np.asarray(channel.lower_line)
            center_line = np.asarray(channel.center_line)
            channel_len = len(center_line)
            x_channel = list(range(channel_len))

            fig.add_trace(
                go.Scatter(
                    x=x_channel, y=lower_line,
                    mode='lines',
                    line=dict(color=CHANNEL_LINE_COLOR, width=1.5, dash='dash'),
                    showlegend=False, hoverinfo='skip',
                ),
                row=row, col=col
            )

            fig.add_trace(
                go.Scatter(
                    x=x_channel, y=upper_line,
                    mode='lines',
                    line=dict(color=CHANNEL_LINE_COLOR, width=1.5, dash='dash'),
                    fill='tonexty', fillcolor=CHANNEL_FILL_COLOR,
                    showlegend=False, hoverinfo='skip',
                ),
                row=row, col=col
            )

    total_height = rows * height_per_row
    fig.update_layout(
        height=total_height,
        showlegend=False,
        template='plotly_white',
    )

    for i in range(1, n_charts + 1):
        fig.update_xaxes(rangeslider_visible=False, row=(i-1)//cols + 1, col=(i-1)%cols + 1)

    return fig
