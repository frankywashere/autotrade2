"""
v15/inspector_utils.py - Reusable visualization functions for channel inspectors.

Helper functions for use in dual inspector and other visualization tools.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v15.dtypes import (
    ChannelLabels,
    CrossCorrelationLabels,
    TIMEFRAMES,
    STANDARD_WINDOWS,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# =============================================================================
# CONSTANTS
# =============================================================================

# Direction labels
DIR_NAMES = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
BREAK_NAMES = {0: 'DOWN', 1: 'UP'}

# Colors
CANDLE_UP_COLOR = '#44ff44'
CANDLE_DOWN_COLOR = '#ff4444'
DEFAULT_CHANNEL_COLOR = 'blue'
BREAK_MARKER_COLOR = 'purple'

# Direction colors (for BEAR=0, SIDEWAYS=1, BULL=2)
DIR_COLORS = {0: '#ff4444', 1: '#ffaa00', 2: '#44ff44'}

# Break direction colors (for DOWN=0, UP=1)
BREAK_COLORS = {0: '#ff4444', 1: '#44ff44'}


# =============================================================================
# CANDLESTICK PLOTTING
# =============================================================================

def plot_candlesticks(
    ax: Axes,
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
) -> None:
    """
    Plot OHLC candlesticks on given axes.

    Args:
        ax: Matplotlib axes to plot on
        df: DataFrame with 'open', 'high', 'low', 'close' columns
        start_idx: Starting index in DataFrame
        end_idx: Ending index in DataFrame (exclusive)

    Green candles for up (close >= open), red for down (close < open).
    Bar width is 0.6 to provide appropriate spacing.
    """
    df_plot = df.iloc[start_idx:end_idx]
    n_bars = len(df_plot)

    if n_bars == 0:
        print(f"DEBUG plot_candlesticks: EARLY RETURN - n_bars=0")
        return

    print(f"DEBUG plot_candlesticks: Plotting {n_bars} candlesticks")

    opens = df_plot['open'].values
    highs = df_plot['high'].values
    lows = df_plot['low'].values
    closes = df_plot['close'].values

    for i in range(n_bars):
        color = CANDLE_UP_COLOR if closes[i] >= opens[i] else CANDLE_DOWN_COLOR

        # Wick (high-low line)
        ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.5, alpha=0.7)

        # Body (open-close rectangle)
        body_bottom = min(opens[i], closes[i])
        body_top = max(opens[i], closes[i])
        ax.add_patch(plt.Rectangle(
            (i - 0.3, body_bottom), 0.6, body_top - body_bottom,
            facecolor=color, edgecolor=color, alpha=0.8
        ))


# =============================================================================
# CHANNEL BOUNDS PLOTTING
# =============================================================================

def plot_channel_bounds(
    ax: Axes,
    channel: Any,
    start_idx: int,
    end_idx: int,
    color: str = 'blue',
    project_forward: int = 0,
) -> None:
    """
    Plot upper and lower channel bounds with center line.

    Args:
        ax: Matplotlib axes to plot on
        channel: Channel object with upper_line, lower_line, center_line,
                 slope, intercept, std_dev, and window attributes
        start_idx: Starting x position for the channel
        end_idx: Ending x position for the channel
        color: Color for channel lines (default: blue)
        project_forward: Number of bars to project bounds forward (default: 0)

    Plots solid lines for upper/lower bounds, dashed line for center.
    If project_forward > 0, projects bounds forward using slope/intercept.
    """
    print(f"DEBUG plot_channel_bounds: Called with start_idx={start_idx}, end_idx={end_idx}, project_forward={project_forward}")

    if channel is None:
        print(f"DEBUG plot_channel_bounds: EARLY RETURN - channel is None")
        return

    if not getattr(channel, 'valid', True):
        print(f"DEBUG plot_channel_bounds: EARLY RETURN - channel.valid={getattr(channel, 'valid', True)}")
        return

    # Calculate the actual plot window length
    plot_window = end_idx - start_idx

    # Get the channel's native window size
    channel_window = getattr(channel, 'window', plot_window)

    # Determine the actual length to use - take minimum to avoid index errors
    actual_length = min(plot_window, channel_window)

    # Slice channel line arrays to match the actual plot window
    def slice_channel_line(line, length):
        """Slice channel line array to match target length."""
        if line is None:
            return None
        line_array = np.asarray(line)
        if len(line_array) < length:
            # If channel line is shorter than needed, only use what we have
            return line_array
        # Otherwise, take the first 'length' elements
        return line_array[:length]

    # Create x-axis positions for the actual line length
    channel_x = np.arange(start_idx, start_idx + actual_length)

    # Plot channel lines with proper slicing
    if hasattr(channel, 'center_line') and channel.center_line is not None:
        center_sliced = slice_channel_line(channel.center_line, actual_length)
        if center_sliced is not None and len(center_sliced) == len(channel_x):
            ax.plot(channel_x, center_sliced, '--', color=color,
                    linewidth=1.5, alpha=0.7, label='Center')

    if hasattr(channel, 'upper_line') and channel.upper_line is not None:
        upper_sliced = slice_channel_line(channel.upper_line, actual_length)
        if upper_sliced is not None and len(upper_sliced) == len(channel_x):
            ax.plot(channel_x, upper_sliced, '-', color=color,
                    linewidth=2, alpha=0.9, label='Upper')

    if hasattr(channel, 'lower_line') and channel.lower_line is not None:
        lower_sliced = slice_channel_line(channel.lower_line, actual_length)
        if lower_sliced is not None and len(lower_sliced) == len(channel_x):
            ax.plot(channel_x, lower_sliced, '-', color=color,
                    linewidth=2, alpha=0.9, label='Lower')

    # Fill between bounds
    if (hasattr(channel, 'upper_line') and hasattr(channel, 'lower_line') and
            channel.upper_line is not None and channel.lower_line is not None):
        upper_sliced = slice_channel_line(channel.upper_line, actual_length)
        lower_sliced = slice_channel_line(channel.lower_line, actual_length)
        if (upper_sliced is not None and lower_sliced is not None and
                len(upper_sliced) == len(channel_x) and len(lower_sliced) == len(channel_x)):
            ax.fill_between(channel_x, lower_sliced, upper_sliced,
                            color=color, alpha=0.1)

    # Project forward if requested
    if project_forward > 0 and _can_project(channel):
        # Use actual_length for projection calculation to match what was actually plotted
        upper_proj, lower_proj = _project_channel_bounds(channel, project_forward, actual_length)
        proj_x = np.arange(start_idx + actual_length, start_idx + actual_length + project_forward)

        ax.plot(proj_x, upper_proj, '--', color=color, linewidth=1.5, alpha=0.5)
        ax.plot(proj_x, lower_proj, '--', color=color, linewidth=1.5, alpha=0.5)
        ax.fill_between(proj_x, lower_proj, upper_proj, color=color, alpha=0.05)


def _can_project(channel: Any) -> bool:
    """Check if channel has required attributes for projection."""
    return (
        hasattr(channel, 'slope') and
        hasattr(channel, 'intercept') and
        hasattr(channel, 'std_dev') and
        hasattr(channel, 'window')
    )


def _project_channel_bounds(
    channel: Any,
    num_bars: int,
    actual_length: Optional[int] = None,
    std_multiplier: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project channel bounds forward using slope/intercept.

    Args:
        channel: Channel object with slope, intercept, std_dev, window
        num_bars: Number of bars to project forward
        actual_length: Actual length of the plotted channel line (if None, uses channel.window)
        std_multiplier: Number of standard deviations for bounds (default: 2.0)

    Returns:
        Tuple of (upper_projection, lower_projection) arrays
    """
    # Use actual_length if provided, otherwise fall back to channel.window
    base_length = actual_length if actual_length is not None else channel.window
    future_x = np.arange(base_length, base_length + num_bars)
    center_projection = channel.slope * future_x + channel.intercept
    upper_projection = center_projection + std_multiplier * channel.std_dev
    lower_projection = center_projection - std_multiplier * channel.std_dev
    return upper_projection, lower_projection


# =============================================================================
# BREAK MARKER PLOTTING
# =============================================================================

def plot_break_marker(
    ax: Axes,
    break_bar: int,
    break_direction: int,
    color: str = 'red',  # CHANGED: purple -> red for visibility testing
    price_y: Optional[float] = None,
    price_range: Optional[float] = None,
) -> None:
    """
    Plot vertical line at break point with direction arrow.

    Args:
        ax: Matplotlib axes to plot on
        break_bar: X position of the break
        break_direction: Direction of break (0=DOWN, 1=UP)
        color: Color for the marker (default: red - TESTING)
        price_y: Y position for arrow (if None, arrow not drawn)
        price_range: Price range for calculating arrow size

    Plots a vertical dashed line at break_bar and an arrow indicating
    the break direction (up or down).
    """
    print(f"DEBUG plot_break_marker: Called with break_bar={break_bar}, break_direction={break_direction}, price_y={price_y}, price_range={price_range}, color={color}")

    # Vertical line at break point - TESTING: increased linewidth and alpha
    ax.axvline(break_bar, color=color, linestyle='--', linewidth=5, alpha=1.0)
    print(f"DEBUG plot_break_marker: Vertical line plotted with color={color}, linewidth=5, alpha=1.0")

    # Direction arrow
    if price_y is not None and price_range is not None:
        # Handle enum values
        direction = break_direction
        if hasattr(direction, 'value'):
            direction = direction.value

        # TESTING: increased arrow offset for better visibility
        arrow_offset = price_range * 0.1  # increased from 0.05 to 0.1
        arrow_color = CANDLE_UP_COLOR if direction == 1 else CANDLE_DOWN_COLOR

        if direction == 1:  # UP
            ax.annotate('', xy=(break_bar, price_y + arrow_offset),
                        xytext=(break_bar, price_y - arrow_offset),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, lw=5))  # increased from 3 to 5
            print(f"DEBUG plot_break_marker: UP arrow plotted at ({break_bar}, {price_y}) with offset={arrow_offset}")
        else:  # DOWN
            ax.annotate('', xy=(break_bar, price_y - arrow_offset),
                        xytext=(break_bar, price_y + arrow_offset),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, lw=5))  # increased from 3 to 5
            print(f"DEBUG plot_break_marker: DOWN arrow plotted at ({break_bar}, {price_y}) with offset={arrow_offset}")
    else:
        print(f"DEBUG plot_break_marker: *** WARNING *** Arrow NOT plotted - price_y={price_y}, price_range={price_range}")


# =============================================================================
# TEXT FORMATTING
# =============================================================================

def format_labels_text(labels: Optional[ChannelLabels], asset: str = 'tsla') -> str:
    """
    Format ChannelLabels info as multi-line string for display.

    Args:
        labels: ChannelLabels object or None
        asset: Asset name for display (default: 'tsla')

    Returns:
        Formatted multi-line string with label information

    Includes: duration_bars, break_direction, permanent_break,
    bars_to_first_break, break_magnitude, returned_to_channel
    """
    if labels is None:
        return f"{asset.upper()}: No labels"

    lines = [f"{asset.upper()} Labels:"]

    # Duration and permanent break
    lines.append(f"  Duration: {labels.duration_bars} bars")
    lines.append(f"  Permanent: {labels.permanent_break}")

    # Break direction
    break_dir = labels.break_direction
    if hasattr(break_dir, 'value'):
        break_dir = break_dir.value
    break_dir_str = BREAK_NAMES.get(break_dir, '?')
    lines.append(f"  Break Dir: {break_dir_str}")

    # New fields from break scan
    if labels.break_scan_valid:
        lines.append(f"  Bars to 1st Break: {labels.bars_to_first_break}")
        lines.append(f"  Break Magnitude: {labels.break_magnitude:.2f} std")
        lines.append(f"  Returned: {labels.returned_to_channel}")

        if labels.returned_to_channel:
            lines.append(f"  Bounces After: {labels.bounces_after_return}")
            lines.append(f"  Continued: {labels.channel_continued}")

    # Validity flags
    validity = []
    if labels.duration_valid:
        validity.append('dur')
    if labels.direction_valid:
        validity.append('dir')
    if labels.break_scan_valid:
        validity.append('scan')
    if validity:
        lines.append(f"  Valid: [{', '.join(validity)}]")

    return '\n'.join(lines)


def format_cross_correlation_text(cross_labels: Optional[CrossCorrelationLabels]) -> str:
    """
    Format CrossCorrelationLabels as multi-line text for display.

    Args:
        cross_labels: CrossCorrelationLabels object or None

    Returns:
        Formatted multi-line string with cross-correlation metrics
    """
    if cross_labels is None:
        return "Cross-Correlation: N/A"

    if not cross_labels.cross_valid:
        return "Cross-Correlation: Invalid (missing break data)"

    lines = ["Cross-Correlation:"]

    # Direction alignment
    dir_aligned = "Yes" if cross_labels.break_direction_aligned else "No"
    lines.append(f"  Dir Aligned: {dir_aligned}")

    # Who broke first
    if cross_labels.tsla_broke_first:
        lines.append(f"  Leader: TSLA (by {cross_labels.break_lag_bars} bars)")
    elif cross_labels.spy_broke_first:
        lines.append(f"  Leader: SPY (by {cross_labels.break_lag_bars} bars)")
    else:
        lines.append(f"  Leader: Simultaneous")

    # Magnitude spread
    lines.append(f"  Mag Spread: {cross_labels.magnitude_spread:+.2f}")

    # Return patterns
    if cross_labels.both_returned:
        lines.append(f"  Pattern: Both returned")
    elif cross_labels.both_permanent:
        lines.append(f"  Pattern: Both permanent")
    else:
        lines.append(f"  Pattern: Mixed")

    # Alignment flags
    lines.append(f"  Return Aligned: {cross_labels.return_pattern_aligned}")
    lines.append(f"  Cont. Aligned: {cross_labels.continuation_aligned}")

    return '\n'.join(lines)


# =============================================================================
# SAMPLE LABEL EXTRACTION
# =============================================================================

def get_labels_from_sample(
    sample: Any,
    window: int,
    tf: str,
    asset: str = 'tsla',
) -> Optional[ChannelLabels]:
    """
    Helper to extract labels from a sample, handling both old and new structure.

    Args:
        sample: ChannelSample object
        window: Window size to get labels for
        tf: Timeframe to get labels for
        asset: Asset name ('tsla' or 'spy', default: 'tsla')

    Returns:
        ChannelLabels object or None if not found

    Handles both structures:
    - New: labels_per_window[window][asset][tf]
    - Old: labels_per_window[window][tf] (assumes TSLA)
    """
    if sample is None:
        return None

    if not hasattr(sample, 'labels_per_window') or not sample.labels_per_window:
        return None

    if window not in sample.labels_per_window:
        return None

    window_labels = sample.labels_per_window[window]

    # Try new structure: {window: {asset: {tf: labels}}}
    if asset in window_labels:
        asset_labels = window_labels[asset]
        if isinstance(asset_labels, dict):
            return asset_labels.get(tf)

    # Try old structure: {window: {tf: labels}} (TSLA only)
    if asset == 'tsla' and tf in window_labels:
        tf_labels = window_labels[tf]
        # Make sure it's actually ChannelLabels, not another dict
        if isinstance(tf_labels, ChannelLabels):
            return tf_labels
        if hasattr(tf_labels, 'duration_bars'):
            return tf_labels

    return None


def get_cross_labels_from_sample(
    sample: Any,
    window: int,
    tf: str,
) -> Optional[CrossCorrelationLabels]:
    """
    Extract cross-correlation labels from a sample.

    Args:
        sample: ChannelSample object
        window: Window size
        tf: Timeframe

    Returns:
        CrossCorrelationLabels or None if not found

    Expected structure: labels_per_window[window]['cross'][tf]
    """
    if sample is None:
        return None

    if not hasattr(sample, 'labels_per_window') or not sample.labels_per_window:
        return None

    if window not in sample.labels_per_window:
        return None

    window_labels = sample.labels_per_window[window]

    if 'cross' in window_labels:
        cross_labels = window_labels['cross']
        if isinstance(cross_labels, dict):
            return cross_labels.get(tf)

    return None
