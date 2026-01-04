"""
Label Inspector - Visual inspection of generated labels

Provides a multi-timeframe visualization of channel samples, showing:
- OHLC price data with channel bounds (2x2 grid: 5min, 15min, 1h, daily)
- Channel bounds projected forward from detection window
- Break point markers with vertical lines at duration_bars forward
- Break direction arrows (UP=green, DOWN=red)
- Label annotations (duration, direction, trigger_tf, new_channel, validity flags)

Usage:
    python label_inspector.py                          # Interactive mode (browse samples)
    python label_inspector.py --sample 0               # Show specific sample
    python label_inspector.py --save output.png        # Save current view
    python label_inspector.py --list                   # List samples and exit
"""

import sys
sys.path.insert(0, '.')
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
import numpy as np
import pandas as pd
import pickle
import argparse

from v7.training.dataset import load_cached_samples, ChannelSample, get_cache_metadata
from v7.training.labels import project_channel_bounds, decode_trigger_tf, ChannelLabels
from v7.core.channel import detect_channel, detect_channels_multi_window, select_best_channel, Channel, Direction, STANDARD_WINDOWS
from v7.core.timeframe import resample_ohlc, TIMEFRAMES, BARS_PER_TF
from v7.tools.label_inspector import detect_suspicious_sample, detect_suspicious_samples, SuspiciousResult


# Constants
DATA_DIR = Path(__file__).parent / 'data'
CACHE_PATH = DATA_DIR / 'feature_cache' / 'channel_samples.pkl'
TSLA_CSV = DATA_DIR / 'TSLA_1min.csv'

# Colors
DIR_COLORS = {0: '#ff4444', 1: '#ffaa00', 2: '#44ff44'}  # BEAR, SIDEWAYS, BULL
DIR_NAMES = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
BREAK_COLORS = {0: '#ff4444', 1: '#44ff44'}  # DOWN, UP
BREAK_NAMES = {0: 'DOWN', 1: 'UP'}

# Timeframes for 2x2 grid
DISPLAY_TIMEFRAMES = ['5min', '15min', '1h', 'daily']

# Forward look bars per TF for visualization
FORWARD_BARS_PER_TF = {
    '5min': 100,    # ~8 hours
    '15min': 100,   # ~25 hours
    '30min': 50,    # ~25 hours
    '1h': 50,       # ~50 hours (~2 days)
    '2h': 50,       # ~100 hours (~4 days)
    '3h': 50,       # ~150 hours (~6 days)
    '4h': 50,       # ~200 hours (~8 days)
    'daily': 50,    # ~50 trading days (~2.5 months)
    'weekly': 50,   # ~50 weeks (~1 year)
    'monthly': 10,  # ~10 months
    '3month': 10,   # ~30 months (~2.5 years)
}

def get_forward_bars_for_tf(tf: str) -> int:
    """Get the number of forward bars to display for a timeframe."""
    return FORWARD_BARS_PER_TF.get(tf, 50)


def load_tsla_data() -> pd.DataFrame:
    """Load and prepare TSLA 5min data."""
    print(f"Loading TSLA data from {TSLA_CSV}...")

    tsla = pd.read_csv(TSLA_CSV, parse_dates=['timestamp'])
    tsla.set_index('timestamp', inplace=True)
    tsla.columns = tsla.columns.str.lower()

    # Resample to 5min
    tsla_5min = tsla.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    print(f"  Loaded {len(tsla_5min)} bars ({tsla_5min.index[0]} to {tsla_5min.index[-1]})")
    return tsla_5min


def get_sample_data_window(
    tsla_df: pd.DataFrame,
    sample: ChannelSample,
    tf: str,
    window: int = 50,
    min_cycles: int = 1,
    use_multi_window: bool = True
) -> tuple:
    """
    Get the data window for a sample at a specific timeframe.

    Uses HISTORICAL-ONLY resampling for channel detection (matches cache generation)
    to avoid future data leakage. Forward bars are still shown for visualization.

    Args:
        tsla_df: Full TSLA 5min DataFrame
        sample: The ChannelSample being inspected
        tf: Timeframe name (e.g., '5min', '1h')
        window: Channel detection window size (used if not multi-window)
        min_cycles: Minimum bounces for valid channel (from cache metadata)
        use_multi_window: Whether to use multi-window detection

    Returns:
        Tuple of (df_tf, channel_end_idx_tf, df_channel, df_forward, channels_dict)
        - df_tf: Full timeframe DataFrame for plotting
        - channel_end_idx_tf: Index where channel ends
        - df_channel: DataFrame for channel detection window
        - df_forward: Forward data for visualization
        - channels_dict: Dict[int, Channel] mapping window sizes to channels (if multi-window)
    """
    # Use timestamp to find position in raw data, NOT the cached index.
    # The cached index was computed against SPY/VIX-aligned data which may have
    # different row counts than raw TSLA data loaded by the inspector.
    try:
        channel_end_idx_5min = tsla_df.index.get_loc(sample.timestamp)
    except KeyError:
        # Exact timestamp not found, find closest
        idx = tsla_df.index.searchsorted(sample.timestamp)
        channel_end_idx_5min = min(idx, len(tsla_df) - 1)

    if channel_end_idx_5min < 0 or channel_end_idx_5min >= len(tsla_df):
        return None, None, None, None, None

    # Historical data (up to sample time) - for channel detection
    df_historical = tsla_df.iloc[:channel_end_idx_5min + 1]

    # Forward bars for visualization - scale by TF (need more 5min bars for longer TFs)
    tf_forward_bars = get_forward_bars_for_tf(tf)
    bars_per_tf = BARS_PER_TF.get(tf, 1)
    forward_bars_5min = tf_forward_bars * bars_per_tf  # Convert TF bars to 5min bars
    forward_end_5min = min(channel_end_idx_5min + forward_bars_5min + 1, len(tsla_df))

    if tf == '5min':
        df_tf = tsla_df.iloc[:forward_end_5min].copy()
        df_channel_tf = df_historical.copy()
        channel_end_idx_tf = channel_end_idx_5min
    else:
        # Resample HISTORICAL-ONLY for channel detection (matches cache generation)
        df_channel_tf = resample_ohlc(df_historical, tf)

        # Resample full data including forward bars for visualization
        df_subset = tsla_df.iloc[:forward_end_5min]
        df_tf = resample_ohlc(df_subset, tf)

        # Channel ends at last bar of historical resampled data
        channel_end_idx_tf = len(df_channel_tf) - 1

    # Check minimum window size for multi-window detection
    max_window = max(STANDARD_WINDOWS) if use_multi_window else window
    if channel_end_idx_tf < max_window:
        # Not enough data for largest window - use what we can
        pass

    # Get channel data for the largest window we can use
    usable_window = min(channel_end_idx_tf + 1, max_window)
    if usable_window < min(STANDARD_WINDOWS):
        return df_tf, channel_end_idx_tf, None, None, None

    channel_start_idx = channel_end_idx_tf - usable_window + 1
    df_channel = df_channel_tf.iloc[max(0, channel_start_idx):channel_end_idx_tf + 1]
    df_forward = df_tf.iloc[channel_end_idx_tf + 1:] if channel_end_idx_tf < len(df_tf) - 1 else None

    # Detect channels at multiple windows if requested
    channels_dict = None
    if use_multi_window and len(df_channel) >= min(STANDARD_WINDOWS):
        channels_dict = detect_channels_multi_window(
            df_channel_tf.iloc[:channel_end_idx_tf + 1],
            windows=STANDARD_WINDOWS,
            min_cycles=min_cycles
        )

    return df_tf, channel_end_idx_tf, df_channel, df_forward, channels_dict


def plot_tf_panel(
    ax: plt.Axes,
    df_tf: pd.DataFrame,
    channel: Channel,
    labels: ChannelLabels,
    tf_name: str,
    channel_end_idx: int,
    window: int = 50,
    channels_dict: dict = None,
    best_window: int = None,
    display_window: int = None,
    labels_in_cache: bool = False
):
    """
    Plot a single timeframe panel with channel, bounds, and labels.

    Args:
        ax: Matplotlib axis to plot on
        df_tf: Full timeframe DataFrame (includes forward data)
        channel: Detected channel object (the one to display)
        labels: Labels for this timeframe (or None)
        tf_name: Timeframe name (e.g., '5min', '1h')
        channel_end_idx: Index in df_tf where channel ends
        window: Channel detection window size
        channels_dict: Dict mapping window sizes to Channel objects (for multi-window display)
        best_window: Which window was selected as best (to highlight)
        display_window: Which window is currently being displayed (for 'w' cycling)
    """
    if df_tf is None or channel is None:
        ax.text(0.5, 0.5, f'{tf_name}\nNo Data', ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
        ax.set_title(tf_name)
        return

    # Determine plot range - use TF-specific forward bars
    channel_start_idx = max(0, channel_end_idx - window + 1)
    tf_forward_bars = get_forward_bars_for_tf(tf_name)
    forward_bars = min(tf_forward_bars, len(df_tf) - channel_end_idx - 1)
    plot_end_idx = channel_end_idx + forward_bars + 1

    # Get data slice for plotting
    df_plot = df_tf.iloc[channel_start_idx:plot_end_idx]
    n_bars = len(df_plot)
    x = np.arange(n_bars)

    # Channel window ends at this x position
    channel_window_end_x = window - 1

    # Plot candlesticks or line
    closes = df_plot['close'].values
    highs = df_plot['high'].values
    lows = df_plot['low'].values
    opens = df_plot['open'].values

    # Simple candlestick representation
    for i in range(n_bars):
        color = '#44ff44' if closes[i] >= opens[i] else '#ff4444'
        # Draw wick
        ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.5, alpha=0.7)
        # Draw body
        body_bottom = min(opens[i], closes[i])
        body_top = max(opens[i], closes[i])
        ax.add_patch(plt.Rectangle(
            (i - 0.3, body_bottom), 0.6, body_top - body_bottom,
            facecolor=color, edgecolor=color, alpha=0.8
        ))

    # Plot channel lines in the detection window
    if channel.valid:
        channel_x = np.arange(window)
        color = DIR_COLORS[channel.direction]

        # Historical channel lines
        ax.plot(channel_x, channel.center_line, '--', color=color, linewidth=1.5, alpha=0.7, label='Center')
        ax.plot(channel_x, channel.upper_line, '-', color=color, linewidth=2, alpha=0.9, label='Upper')
        ax.plot(channel_x, channel.lower_line, '-', color=color, linewidth=2, alpha=0.9, label='Lower')
        ax.fill_between(channel_x, channel.lower_line, channel.upper_line, color=color, alpha=0.1)

        # Project channel bounds forward
        if forward_bars > 0:
            upper_proj, lower_proj = project_channel_bounds(channel, forward_bars)
            proj_x = np.arange(window, window + len(upper_proj))

            ax.plot(proj_x, upper_proj, '--', color=color, linewidth=1.5, alpha=0.5)
            ax.plot(proj_x, lower_proj, '--', color=color, linewidth=1.5, alpha=0.5)
            ax.fill_between(proj_x, lower_proj, upper_proj, color=color, alpha=0.05)

        # Mark channel touches
        for touch in channel.touches:
            marker = '^' if touch.touch_type == 0 else 'v'  # Lower=up arrow, Upper=down arrow
            touch_color = '#00ff00' if touch.touch_type == 0 else '#ff0000'
            ax.plot(touch.bar_index, touch.price, marker, color=touch_color, markersize=8, alpha=0.8)

    # Draw vertical line at channel end (sample point)
    ax.axvline(channel_window_end_x, color='blue', linestyle='-', linewidth=2, alpha=0.7, label='Sample Point')

    # Draw break point and direction arrow if labels exist
    if labels is not None and labels.permanent_break:
        break_bar = labels.duration_bars
        if break_bar < forward_bars:
            break_x = window + break_bar

            # Vertical line at break
            ax.axvline(break_x, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='Break Point')

            # Direction arrow
            break_dir = labels.break_direction
            arrow_color = BREAK_COLORS.get(break_dir, 'gray')

            # Get price at break point for arrow positioning
            break_data_idx = channel_start_idx + int(break_x)
            if break_data_idx < len(df_tf):
                arrow_y = df_tf.iloc[break_data_idx]['close']
            else:
                arrow_y = closes[-1]

            price_range = max(highs) - min(lows)
            arrow_offset = price_range * 0.05

            # Draw arrow indicating break direction
            if break_dir == 1:  # UP
                ax.annotate('', xy=(break_x, arrow_y + arrow_offset),
                           xytext=(break_x, arrow_y - arrow_offset),
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=3))
            else:  # DOWN
                ax.annotate('', xy=(break_x, arrow_y - arrow_offset),
                           xytext=(break_x, arrow_y + arrow_offset),
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=3))

    # Build label annotation text with multi-window info
    label_text = f"{tf_name}"
    if channel.valid:
        label_text += f" | {DIR_NAMES[channel.direction]}"
        # Show best window if available
        if best_window is not None:
            label_text += f" | Best Window: {best_window}"

    # Build window scores line if multi-window data available
    if channels_dict and len(channels_dict) > 0:
        window_parts = []
        for w in sorted(channels_dict.keys()):
            ch = channels_dict[w]
            if ch is not None:
                bounce_str = f"{w}:{ch.bounce_count}b"
                # Mark best window with asterisk
                if w == best_window:
                    bounce_str += "*"
                # Mark currently displayed window with brackets
                if display_window is not None and w == display_window:
                    bounce_str = f"[{bounce_str}]"
                window_parts.append(bounce_str)
        if window_parts:
            label_text += f"\nWindow scores: {' '.join(window_parts)}"

    if labels is not None:
        label_text += f"\nDuration: {labels.duration_bars} bars"
        if labels.direction_valid:
            label_text += f" | Break: {BREAK_NAMES.get(labels.break_direction, '?')}"
        if labels.trigger_tf_valid:
            trigger_str = decode_trigger_tf(labels.break_trigger_tf)
            label_text += f"\nTrigger: {trigger_str}"
        if labels.new_channel_valid:
            label_text += f" | Next: {DIR_NAMES.get(labels.new_channel_direction, '?')}"

        # Validity flags
        validity_parts = []
        if labels.duration_valid:
            validity_parts.append('dur')
        if labels.direction_valid:
            validity_parts.append('dir')
        if labels.trigger_tf_valid:
            validity_parts.append('trig')
        if labels.new_channel_valid:
            validity_parts.append('next')
        if validity_parts:
            label_text += f"\nValid: [{', '.join(validity_parts)}]"
    else:
        # Distinguish between: cache has None vs key missing vs invalid channel
        if channel.valid:
            if labels_in_cache:
                # Labels were computed but are None (resampled channel failed or insufficient data)
                label_text += f"\nNo labels (TF channel invalid)"
            else:
                label_text += f"\nNo cached labels (cache stale?)"
        else:
            label_text += f"\nNo labels (bounces < min_cycles)"

    # Add text box with label info
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, label_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')

    # Title
    title = f"{tf_name}"
    if channel.valid:
        title += f" - R2: {channel.r_squared:.3f}, Width: {channel.width_pct:.2f}%"
    ax.set_title(title, fontsize=11, fontweight='bold')

    # Axis labels
    ax.set_xlabel('Bars')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)


def create_multi_tf_figure(
    sample: ChannelSample,
    tsla_df: pd.DataFrame,
    sample_idx: int = 0,
    total_samples: int = 1,
    window: int = 50,
    min_cycles: int = 1
) -> plt.Figure:
    """
    Create a 2x2 grid figure showing the same sample across different timeframes.

    Args:
        sample: The ChannelSample to visualize
        tsla_df: Full TSLA 5min DataFrame
        sample_idx: Current sample index (for title)
        total_samples: Total number of samples (for title)
        window: Channel detection window
        min_cycles: Minimum bounces for valid channel (from cache metadata)

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Sample {sample_idx + 1}/{total_samples} | Timestamp: {sample.timestamp}",
        fontsize=14, fontweight='bold'
    )

    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    for idx, tf in enumerate(DISPLAY_TIMEFRAMES):
        ax = axes_flat[idx]

        # Get data for this timeframe with multi-window detection
        df_tf, channel_end_idx, df_channel, df_forward, channels_dict = get_sample_data_window(
            tsla_df, sample, tf, window=window, min_cycles=min_cycles, use_multi_window=True
        )

        # Select best channel from multi-window detection
        channel = None
        best_window = None
        if channels_dict:
            channel, best_window = select_best_channel(channels_dict)
        elif df_channel is not None and len(df_channel) >= window:
            # Fallback: single window detection
            channel = detect_channel(df_channel, window=window, min_cycles=min_cycles)
            best_window = window

        # Get labels for this timeframe
        # IMPORTANT: labels_per_window is keyed by the 5min window (sample.best_window),
        # NOT by the TF-specific window we just detected. The cache was generated using
        # the 5min best_window to key the outer dict, then generate_labels_per_tf
        # internally does its own per-TF multi-window detection.
        labels = None
        labels_in_cache = False
        cache_window = sample.best_window  # Use the 5min window from cache, not TF-specific
        if sample.labels_per_window and cache_window and cache_window in sample.labels_per_window:
            window_labels = sample.labels_per_window[cache_window]
            labels_in_cache = tf in window_labels
            labels = window_labels.get(tf)
        elif isinstance(sample.labels, dict):
            labels_in_cache = tf in sample.labels
            labels = sample.labels.get(tf)

        # Plot panel with multi-window info
        plot_tf_panel(
            ax, df_tf, channel, labels, tf, channel_end_idx,
            window=best_window or window,
            channels_dict=channels_dict,
            best_window=best_window,
            display_window=best_window,  # Initially display the best window
            labels_in_cache=labels_in_cache
        )

    plt.tight_layout()
    return fig


class SampleBrowser:
    """Interactive browser for samples using matplotlib buttons and keyboard."""

    def __init__(self, samples: list, tsla_df: pd.DataFrame, window: int = 50, min_cycles: int = 1):
        self.samples = samples
        self.tsla_df = tsla_df
        self.window = window
        self.min_cycles = min_cycles
        self.current_idx = 0
        self.fig = None
        self.axes = None

        # Window cycling state: index into STANDARD_WINDOWS, or None for "best"
        self.display_window_idx = None  # None = show best window

        # Detect suspicious samples at startup
        print("Analyzing samples for suspicious patterns...")
        self.suspicious_results = detect_suspicious_samples(samples)
        self.suspicious_indices = [r.sample_idx for r in self.suspicious_results]
        print(f"  Found {len(self.suspicious_indices)} suspicious samples out of {len(samples)}")

        # Map sample idx to suspicious result for quick lookup
        self.suspicious_map = {r.sample_idx: r for r in self.suspicious_results}

    def show(self, start_idx: int = 0):
        """Show the browser starting at a specific sample."""
        self.current_idx = start_idx
        self._print_controls()
        self._create_figure()
        plt.show()

    def _print_controls(self):
        """Print keyboard controls."""
        print("\n" + "="*50)
        print("LABEL INSPECTOR - Keyboard Controls")
        print("="*50)
        print("  LEFT/RIGHT : Previous/Next sample")
        print("  r          : Random sample")
        print("  f          : Next flagged (suspicious) sample")
        print("  F          : Previous flagged sample")
        print("  w          : Cycle through window sizes (best -> 10 -> 20 -> ... -> 80 -> best)")
        print("  q/ESC      : Quit")
        print("="*50 + "\n")

    def _create_figure(self):
        """Create the figure with navigation buttons and keyboard handler."""
        self.fig = plt.figure(figsize=(16, 14))

        # Add navigation buttons
        ax_prev = plt.axes([0.1, 0.01, 0.1, 0.04])
        ax_next = plt.axes([0.8, 0.01, 0.1, 0.04])
        ax_goto = plt.axes([0.35, 0.01, 0.12, 0.04])
        ax_flagged = plt.axes([0.52, 0.01, 0.12, 0.04])

        self.btn_prev = Button(ax_prev, '← Previous')
        self.btn_next = Button(ax_next, 'Next →')
        self.btn_goto = Button(ax_goto, 'Jump to...')
        self.btn_flagged = Button(ax_flagged, '⚠ Flagged')

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_goto.on_clicked(self._on_goto)
        self.btn_flagged.on_clicked(self._on_next_flagged)

        # Connect keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._update_plot()

    def _on_key(self, event):
        """Handle keyboard navigation."""
        if event.key == 'right':
            self._navigate(1)
        elif event.key == 'left':
            self._navigate(-1)
        elif event.key == 'r':
            self.current_idx = np.random.randint(len(self.samples))
            self.display_window_idx = None  # Reset to best window on sample change
            self._update_plot()
        elif event.key == 'f':
            self._jump_to_flagged(forward=True)
        elif event.key == 'F':
            self._jump_to_flagged(forward=False)
        elif event.key == 'w':
            self._cycle_window()
        elif event.key in ('q', 'escape'):
            plt.close('all')

    def _cycle_window(self):
        """Cycle through window sizes: best -> 10 -> 20 -> ... -> 80 -> best."""
        if self.display_window_idx is None:
            # Currently showing best, start with first standard window
            self.display_window_idx = 0
        else:
            self.display_window_idx += 1
            if self.display_window_idx >= len(STANDARD_WINDOWS):
                # Wrap back to "best"
                self.display_window_idx = None

        # Print current window state
        if self.display_window_idx is None:
            print("Displaying: Best window (auto-selected)")
        else:
            print(f"Displaying: Window {STANDARD_WINDOWS[self.display_window_idx]}")

        self._update_plot()

    def _navigate(self, delta: int):
        """Navigate by delta samples."""
        self.current_idx = (self.current_idx + delta) % len(self.samples)
        self.display_window_idx = None  # Reset to best window on sample change
        self._update_plot()

    def _jump_to_flagged(self, forward: bool = True):
        """Jump to next/previous flagged sample."""
        if not self.suspicious_indices:
            print("No suspicious samples found!")
            return

        if forward:
            # Find next suspicious after current
            for idx in self.suspicious_indices:
                if idx > self.current_idx:
                    self.current_idx = idx
                    self._update_plot()
                    return
            # Wrap around to first
            self.current_idx = self.suspicious_indices[0]
        else:
            # Find previous suspicious before current
            for idx in reversed(self.suspicious_indices):
                if idx < self.current_idx:
                    self.current_idx = idx
                    self._update_plot()
                    return
            # Wrap around to last
            self.current_idx = self.suspicious_indices[-1]

        self._update_plot()

    def _update_plot(self):
        """Update the plot with current sample."""
        # Clear previous subplots (keep buttons - first 4 axes)
        for ax in self.fig.axes[4:]:  # Skip button axes
            ax.remove()

        sample = self.samples[self.current_idx]

        # Check if this sample is flagged
        suspicious_result = self.suspicious_map.get(self.current_idx)
        is_flagged = suspicious_result is not None

        # Determine which window to display
        if self.display_window_idx is not None:
            display_window = STANDARD_WINDOWS[self.display_window_idx]
        else:
            display_window = None  # Will use best window

        # Create 2x2 grid
        gs = self.fig.add_gridspec(2, 2, left=0.06, right=0.98, top=0.88, bottom=0.08,
                                   hspace=0.25, wspace=0.15)

        for idx, tf in enumerate(DISPLAY_TIMEFRAMES):
            row, col = idx // 2, idx % 2
            ax = self.fig.add_subplot(gs[row, col])

            # Get data for this timeframe with multi-window detection
            df_tf, channel_end_idx, df_channel, df_forward, channels_dict = get_sample_data_window(
                self.tsla_df, sample, tf, window=self.window,
                min_cycles=self.min_cycles, use_multi_window=True
            )

            # Select which channel to display
            channel = None
            best_window = None
            actual_display_window = display_window

            if channels_dict:
                # Get the best window for reference
                best_channel, best_window = select_best_channel(channels_dict)

                if display_window is not None and display_window in channels_dict:
                    # User selected a specific window
                    channel = channels_dict[display_window]
                    actual_display_window = display_window
                else:
                    # Show the best window
                    channel = best_channel
                    actual_display_window = best_window
            elif df_channel is not None and len(df_channel) >= self.window:
                # Fallback: single window detection
                channel = detect_channel(df_channel, window=self.window, min_cycles=self.min_cycles)
                best_window = self.window
                actual_display_window = self.window

            # Get labels for this timeframe
            # IMPORTANT: labels_per_window is keyed by the 5min window (sample.best_window),
            # NOT by the TF-specific window we just detected. The cache was generated using
            # the 5min best_window to key the outer dict, then generate_labels_per_tf
            # internally does its own per-TF multi-window detection.
            labels = None
            labels_in_cache = False
            cache_window = sample.best_window  # Use the 5min window from cache, not TF-specific
            if sample.labels_per_window and cache_window and cache_window in sample.labels_per_window:
                window_labels = sample.labels_per_window[cache_window]
                labels_in_cache = tf in window_labels
                labels = window_labels.get(tf)
            elif isinstance(sample.labels, dict):
                labels_in_cache = tf in sample.labels
                labels = sample.labels.get(tf)

            # Plot panel with multi-window info
            plot_tf_panel(
                ax, df_tf, channel, labels, tf, channel_end_idx,
                window=actual_display_window or self.window,
                channels_dict=channels_dict,
                best_window=best_window,
                display_window=actual_display_window,
                labels_in_cache=labels_in_cache
            )

            # Add red border if this TF has flags
            if suspicious_result:
                tf_flags = [f for f in suspicious_result.flags if f.tf == tf]
                if tf_flags:
                    for spine in ax.spines.values():
                        spine.set_color('red')
                        spine.set_linewidth(3)

        # Build title with flag status and window mode
        title = f"Sample {self.current_idx + 1}/{len(self.samples)} | {sample.timestamp}"
        if display_window is not None:
            title += f" | Window: {display_window}"
        else:
            title += " | Window: Best"

        if is_flagged:
            n_flags = len(suspicious_result.flags)
            title = f"[!] {title} | {n_flags} flag(s)"
            title_color = 'red'
        else:
            title = f"[OK] {title}"
            title_color = 'green'

        self.fig.suptitle(title, fontsize=14, fontweight='bold', color=title_color)

        # Add flag details if any
        if suspicious_result and suspicious_result.flags:
            flag_text = "Flags: " + " | ".join(f.message for f in suspicious_result.flags[:3])
            if len(suspicious_result.flags) > 3:
                flag_text += f" ... (+{len(suspicious_result.flags) - 3} more)"
            self.fig.text(0.5, 0.94, flag_text, ha='center', fontsize=10, color='red',
                         transform=self.fig.transFigure)

        self.fig.canvas.draw_idle()

    def _on_next_flagged(self, event):
        """Button handler for next flagged sample."""
        self._jump_to_flagged(forward=True)

    def _on_prev(self, event):
        """Go to previous sample."""
        self._navigate(-1)

    def _on_next(self, event):
        """Go to next sample."""
        self._navigate(1)

    def _on_goto(self, event):
        """Jump to a specific sample."""
        try:
            idx = int(input(f"Enter sample index (0-{len(self.samples)-1}): "))
            if 0 <= idx < len(self.samples):
                self.current_idx = idx
                self._update_plot()
            else:
                print(f"Index out of range. Must be 0-{len(self.samples)-1}")
        except ValueError:
            print("Invalid input. Enter a number.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Label Inspector - Visual inspection of generated labels')
    parser.add_argument('--sample', type=int, default=None, help='Show specific sample index')
    parser.add_argument('--save', type=str, default=None, help='Save figure to file')
    parser.add_argument('--cache', type=str, default=str(CACHE_PATH), help='Path to cache file')
    parser.add_argument('--window', type=int, default=50, help='Channel detection window (fallback)')
    parser.add_argument('--list', action='store_true', help='List samples and exit')
    args = parser.parse_args()

    cache_path = Path(args.cache)

    # Check cache exists
    if not cache_path.exists():
        print(f"Error: Cache file not found at {cache_path}")
        print("\nTo generate the cache, run:")
        print("  python -c \"from v7.training.dataset import prepare_dataset_from_scratch; ...")
        print("\nOr use train.py to build the cache first.")
        sys.exit(1)

    # Load cache metadata to get min_cycles and other params
    cache_metadata = get_cache_metadata(cache_path)
    if cache_metadata:
        min_cycles = cache_metadata.get('min_cycles', 1)
        cached_window = cache_metadata.get('window', args.window)
        print(f"Cache metadata: min_cycles={min_cycles}, window={cached_window}")
    else:
        min_cycles = 1
        cached_window = args.window
        print(f"No cache metadata found, using defaults: min_cycles={min_cycles}")

    # Load samples
    print(f"Loading samples from {cache_path}...")
    samples, load_info = load_cached_samples(cache_path, migrate_labels=True)
    print(f"Loaded {len(samples)} samples")
    print(f"  Cache version: {load_info.get('cached_version', 'unknown')}")

    # List mode
    if args.list:
        print("\nSample listing:")
        print("-" * 80)
        for i, sample in enumerate(samples[:20]):  # Show first 20
            tf_labels = list(sample.labels.keys()) if isinstance(sample.labels, dict) else []
            # Show best_window if available
            best_win_str = f"BestWin:{sample.best_window}" if sample.best_window else ""
            print(f"  [{i:4d}] {sample.timestamp} | TFs: {len(tf_labels)} | {best_win_str} | Channel: {sample.channel.direction.name if sample.channel else 'N/A'}")
        if len(samples) > 20:
            print(f"  ... and {len(samples) - 20} more samples")
        sys.exit(0)

    # Load TSLA data
    tsla_df = load_tsla_data()

    # Single sample mode
    if args.sample is not None:
        if args.sample >= len(samples):
            print(f"Error: Sample index {args.sample} out of range (0-{len(samples)-1})")
            sys.exit(1)

        sample = samples[args.sample]
        fig = create_multi_tf_figure(
            sample, tsla_df, args.sample, len(samples),
            window=cached_window, min_cycles=min_cycles
        )

        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"Saved to {args.save}")
        else:
            plt.show()

    # Interactive browser mode
    else:
        print("\nStarting interactive browser...")
        print("  Use 'Previous' and 'Next' buttons to navigate")
        print("  Use 'Jump to...' to go to a specific sample index")
        print("  Use 'w' to cycle through window sizes")

        browser = SampleBrowser(samples, tsla_df, window=cached_window, min_cycles=min_cycles)
        browser.show()


if __name__ == '__main__':
    main()
