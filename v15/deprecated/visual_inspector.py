#!/usr/bin/env python3
"""
Visual Label Inspector for v15 Cache System

A matplotlib-based visual inspection tool for ChannelSample objects.
Displays multi-timeframe visualization with channel bounds, break points,
and direction arrows in a 2x2 grid layout.

Works with the v15 ChannelSample structure:
    - sample.tf_features: Flat dict of TF-prefixed features ({tf}_{feature_name})
    - sample.labels_per_window[window][tf]: Labels for each window/timeframe
    - 10 timeframes: 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly
    - 8 standard windows: 10, 20, 30, 40, 50, 60, 70, 80 bars

Features:
    - Multi-timeframe 2x2 grid visualization (4 timeframes at once)
    - Timeframe view swapping ('t' key): mixed -> intraday -> multiday -> mixed
    - Window cycling ('w' key): best -> 10 -> 20 -> ... -> 80 -> best
    - Sample navigation: LEFT/RIGHT arrows, 'r' for random
    - Channel visualization: OHLC candlesticks with channel bounds projected forward
    - Two-pass channel detection support

Usage:
    python -m v15.visual_inspector samples.pkl
    python -m v15.visual_inspector --samples samples.pkl
    python -m v15.visual_inspector --cache data/feature_cache/test_v15.pkl

Keyboard Controls:
    LEFT/RIGHT : Navigate samples
    UP/DOWN    : Jump 10 samples
    r          : Random sample
    w          : Cycle window sizes (best -> 10 -> 20 -> ... -> 80 -> best)
    t          : Swap timeframe views (mixed -> intraday -> multiday -> mixed)
    i          : Print detailed sample info to console
    q/ESC      : Quit
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import v15 types and data loading
from v15.dtypes import ChannelSample, TIMEFRAMES, STANDARD_WINDOWS
from v15.data import load_market_data


# =============================================================================
# Constants
# =============================================================================

# Timeframe view sets for 2x2 grid
TF_VIEW_SETS = {
    'mixed': ['5min', '15min', '1h', 'daily'],
    'intraday': ['5min', '1h', '2h', '4h'],
    'multiday': ['daily', 'weekly', 'monthly', '4h'],
}
TF_VIEW_NAMES = ['mixed', 'intraday', 'multiday']

# Color schemes
DIR_COLORS = {0: '#ff4444', 1: '#ffaa00', 2: '#44ff44'}  # BEAR, SIDEWAYS, BULL
DIR_NAMES = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
BREAK_COLORS = {0: '#ff4444', 1: '#44ff44'}  # DOWN, UP
BREAK_NAMES = {0: 'DOWN', 1: 'UP'}

# Forward look bars per TF for visualization
FORWARD_BARS_PER_TF = {
    '5min': 100,
    '15min': 100,
    '30min': 50,
    '1h': 50,
    '2h': 50,
    '3h': 50,
    '4h': 50,
    'daily': 50,
    'weekly': 50,
    'monthly': 10,
}

# Bars per timeframe (for resampling)
BARS_PER_TF = {
    '5min': 1,
    '15min': 3,
    '30min': 6,
    '1h': 12,
    '2h': 24,
    '3h': 36,
    '4h': 48,
    'daily': 78,
    'weekly': 390,
    'monthly': 1638,
}


# =============================================================================
# Resampling Utility
# =============================================================================

def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 5min OHLCV data to target timeframe."""
    if timeframe == '5min':
        return df

    rule_map = {
        '15min': '15min', '30min': '30min', '1h': '1h',
        '2h': '2h', '3h': '3h', '4h': '4h',
        'daily': '1D', 'weekly': '1W', 'monthly': '1ME'
    }

    rule = rule_map.get(timeframe)
    if not rule:
        return df

    return df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


# =============================================================================
# Plotting Utilities
# =============================================================================

def project_channel_bounds(
    channel: Any,
    num_bars: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Project channel bounds forward using slope/intercept."""
    future_x = np.arange(channel.window, channel.window + num_bars)
    center_projection = channel.slope * future_x + channel.intercept
    std_multiplier = 2.0
    upper_projection = center_projection + std_multiplier * channel.std_dev
    lower_projection = center_projection - std_multiplier * channel.std_dev
    return upper_projection, lower_projection


def plot_timeframe_panel(
    ax: plt.Axes,
    df_tf: Optional[pd.DataFrame],
    channel: Optional[Any],
    labels: Optional[Any],
    tf_name: str,
    channel_end_idx: int,
    window: int,
    channels_dict: Optional[Dict[int, Any]] = None,
    best_window: Optional[int] = None,
    display_window: Optional[int] = None,
) -> None:
    """
    Plot a single timeframe panel with channel, bounds, and labels.
    """
    # Handle no data
    if df_tf is None:
        ax.text(0.5, 0.5, f'{tf_name}\nNo Data', ha='center', va='center',
                fontsize=14, transform=ax.transAxes, color='gray')
        ax.set_title(tf_name, color='gray')
        return

    # Handle unavailable channel
    if channel is None:
        ax.set_facecolor('#f5f5f5')
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(1.5)
            spine.set_linestyle('--')

        if display_window is not None:
            reason = f'Window {display_window} unavailable'
        else:
            reason = 'Channel detection failed'

        props = dict(boxstyle='round', facecolor='#e0e0e0', alpha=0.7,
                     edgecolor='#999999', linewidth=2, linestyle='--')
        ax.text(0.5, 0.5,
                f'{tf_name}\n\nN/A\n\n{reason}',
                ha='center', va='center',
                fontsize=11,
                color='#666666',
                transform=ax.transAxes,
                bbox=props,
                family='monospace')

        ax.set_title(f"{tf_name} - N/A", fontsize=11, fontweight='bold', color='gray')
        ax.grid(True, alpha=0.1, linestyle=':', color='gray')
        return

    # Determine plot range
    channel_start_idx = max(0, channel_end_idx - window + 1)
    tf_forward_bars = FORWARD_BARS_PER_TF.get(tf_name, 50)
    forward_bars = min(tf_forward_bars, len(df_tf) - channel_end_idx - 1)
    plot_end_idx = channel_end_idx + forward_bars + 1

    # Get data slice
    df_plot = df_tf.iloc[channel_start_idx:plot_end_idx]
    n_bars = len(df_plot)
    x = np.arange(n_bars)

    channel_window_end_x = window - 1

    # Extract OHLC data
    closes = df_plot['close'].values
    highs = df_plot['high'].values
    lows = df_plot['low'].values
    opens = df_plot['open'].values

    # Plot candlesticks
    for i in range(n_bars):
        color = '#44ff44' if closes[i] >= opens[i] else '#ff4444'
        ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.5, alpha=0.7)
        body_bottom = min(opens[i], closes[i])
        body_top = max(opens[i], closes[i])
        ax.add_patch(plt.Rectangle(
            (i - 0.3, body_bottom), 0.6, body_top - body_bottom,
            facecolor=color, edgecolor=color, alpha=0.8
        ))

    # Plot channel lines if valid
    if channel.valid:
        channel_x = np.arange(window)
        color = DIR_COLORS.get(channel.direction, '#888888')

        ax.plot(channel_x, channel.center_line, '--', color=color, linewidth=1.5, alpha=0.7)
        ax.plot(channel_x, channel.upper_line, '-', color=color, linewidth=2, alpha=0.9)
        ax.plot(channel_x, channel.lower_line, '-', color=color, linewidth=2, alpha=0.9)
        ax.fill_between(channel_x, channel.lower_line, channel.upper_line, color=color, alpha=0.1)

        # Project forward
        if forward_bars > 0:
            upper_proj, lower_proj = project_channel_bounds(channel, forward_bars)
            proj_x = np.arange(window, window + len(upper_proj))

            ax.plot(proj_x, upper_proj, '--', color=color, linewidth=1.5, alpha=0.5)
            ax.plot(proj_x, lower_proj, '--', color=color, linewidth=1.5, alpha=0.5)
            ax.fill_between(proj_x, lower_proj, upper_proj, color=color, alpha=0.05)

        # Mark channel touches
        if hasattr(channel, 'touches'):
            for touch in channel.touches:
                marker = '^' if touch.touch_type == 0 else 'v'
                touch_color = '#00ff00' if touch.touch_type == 0 else '#ff0000'
                ax.plot(touch.bar_index, touch.price, marker, color=touch_color, markersize=8, alpha=0.8)

    # Draw vertical line at channel end
    ax.axvline(channel_window_end_x, color='blue', linestyle='-', linewidth=2, alpha=0.7)

    # Draw break point and direction arrow if labels exist
    if labels is not None and getattr(labels, 'permanent_break', False):
        break_bar = labels.duration_bars
        if break_bar < forward_bars:
            break_x = window + break_bar

            ax.axvline(break_x, color='purple', linestyle='--', linewidth=2, alpha=0.7)

            # Get break direction (handle both enum and int)
            break_dir = labels.break_direction
            if hasattr(break_dir, 'value'):
                break_dir = break_dir.value
            arrow_color = BREAK_COLORS.get(break_dir, 'gray')

            break_data_idx = channel_start_idx + int(break_x)
            if break_data_idx < len(df_tf):
                arrow_y = df_tf.iloc[break_data_idx]['close']
            else:
                arrow_y = closes[-1]

            price_range = max(highs) - min(lows)
            arrow_offset = price_range * 0.05

            if break_dir == 1:  # UP
                ax.annotate('', xy=(break_x, arrow_y + arrow_offset),
                           xytext=(break_x, arrow_y - arrow_offset),
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=3))
            else:  # DOWN
                ax.annotate('', xy=(break_x, arrow_y - arrow_offset),
                           xytext=(break_x, arrow_y + arrow_offset),
                           arrowprops=dict(arrowstyle='->', color=arrow_color, lw=3))

    # Build label annotation text
    label_text = f"{tf_name}"
    if channel.valid:
        label_text += f" | {DIR_NAMES.get(channel.direction, '?')}"

    # Show which window is displayed
    if display_window is not None and best_window is not None:
        if display_window == best_window:
            label_text += f" | W:{display_window}*"  # * indicates best
        else:
            label_text += f" | W:{display_window} (best:{best_window})"
    elif best_window is not None:
        label_text += f" | W:{best_window}*"

    # Add label details
    if labels is not None:
        # Get break direction name (handle enum)
        break_dir = labels.break_direction
        if hasattr(break_dir, 'name'):
            break_dir_str = break_dir.name
        elif break_dir in BREAK_NAMES:
            break_dir_str = BREAK_NAMES[break_dir]
        else:
            break_dir_str = '?'

        label_text += f"\nDuration: {labels.duration_bars} bars"
        if getattr(labels, 'direction_valid', False):
            label_text += f" | Break: {break_dir_str}"

        # Get new channel direction (handle enum)
        new_dir = labels.next_channel_direction
        if hasattr(new_dir, 'name'):
            new_dir_str = new_dir.name
        elif new_dir in DIR_NAMES:
            new_dir_str = DIR_NAMES[new_dir]
        else:
            new_dir_str = '?'
        label_text += f" | Next: {new_dir_str}"

        validity_parts = []
        if getattr(labels, 'duration_valid', False):
            validity_parts.append('dur')
        if getattr(labels, 'direction_valid', False):
            validity_parts.append('dir')
        if validity_parts:
            label_text += f"\nValid: [{', '.join(validity_parts)}]"
    else:
        label_text += f"\nNo labels"

    # Add text box with color coding for non-best windows
    is_non_best = (display_window is not None and best_window is not None
                   and display_window != best_window)
    if is_non_best:
        props = dict(boxstyle='round', facecolor='#ffe6cc', alpha=0.9,
                     edgecolor='#cc6600', linewidth=1.5)  # Orange-tinted for non-best
    else:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, label_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')

    # Title with clear window indication
    title = f"{tf_name}"
    if channel.valid:
        title += f" - R2: {channel.r_squared:.3f}, Width: {channel.width_pct:.2f}%"

    # Show window info with clear best/non-best indication
    if display_window is not None and best_window is not None:
        if display_window == best_window:
            title += f" | Win: {display_window} (best)"
            title_color = 'black'
        else:
            title += f" | Win: {display_window}"
            title_color = '#cc6600'  # Orange for non-best
    elif best_window is not None:
        title += f" | Win: {best_window} (best)"
        title_color = 'black'
    else:
        title_color = 'black'

    ax.set_title(title, fontsize=11, fontweight='bold', color=title_color)

    ax.set_xlabel('Bars')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)


# =============================================================================
# Data Loading and Sample Processing
# =============================================================================

def get_sample_data_window(
    tsla_df: pd.DataFrame,
    sample: ChannelSample,
    tf: str,
    window: int = 50,
    min_cycles: int = 1
) -> Tuple[Optional[pd.DataFrame], int, Optional[Dict[int, Any]]]:
    """
    Get data window for a sample at a specific timeframe.
    """
    from v7.core.channel import detect_channels_multi_window

    # Use timestamp to find position in raw data
    try:
        channel_end_idx_5min = tsla_df.index.get_loc(sample.timestamp)
    except KeyError:
        idx = tsla_df.index.searchsorted(sample.timestamp)
        channel_end_idx_5min = min(idx, len(tsla_df) - 1)

    if channel_end_idx_5min < 0 or channel_end_idx_5min >= len(tsla_df):
        return None, 0, None

    # Historical data
    df_historical = tsla_df.iloc[:channel_end_idx_5min + 1]

    # Forward bars for visualization
    tf_forward_bars = FORWARD_BARS_PER_TF.get(tf, 50)
    bars_per_tf = BARS_PER_TF.get(tf, 1)
    forward_bars_5min = tf_forward_bars * bars_per_tf
    forward_end_5min = min(channel_end_idx_5min + forward_bars_5min + 1, len(tsla_df))

    if tf == '5min':
        df_tf = tsla_df.iloc[:forward_end_5min].copy()
        df_channel_tf = df_historical.copy()
        channel_end_idx_tf = channel_end_idx_5min
    else:
        df_channel_tf = resample_ohlc(df_historical, tf)
        df_subset = tsla_df.iloc[:forward_end_5min]
        df_tf = resample_ohlc(df_subset, tf)
        channel_end_idx_tf = len(df_channel_tf) - 1

    # Detect channels at multiple windows
    channels_dict = None
    if channel_end_idx_tf >= min(STANDARD_WINDOWS) and len(df_channel_tf) >= min(STANDARD_WINDOWS):
        try:
            channels_dict = detect_channels_multi_window(
                df_channel_tf.iloc[:channel_end_idx_tf + 1],
                windows=STANDARD_WINDOWS,
                min_cycles=min_cycles
            )
        except Exception:
            channels_dict = None

    return df_tf, channel_end_idx_tf, channels_dict


# =============================================================================
# Main Inspector Class
# =============================================================================

class VisualInspector:
    """
    Interactive visual inspector for v15 cache samples.
    """

    def __init__(
        self,
        samples: List[ChannelSample],
        tsla_df: pd.DataFrame
    ) -> None:
        """
        Initialize the inspector.

        Args:
            samples: List of ChannelSample objects from v15 cache
            tsla_df: TSLA 5min DataFrame for visualization
        """
        self.samples = samples
        self.tsla_df = tsla_df
        self.current_idx = 0

        # UI state
        self.fig = None
        self.display_window_idx = None  # None = show best window
        self.tf_view_idx = 0  # 0 = mixed

    def show(self, start_idx: int = 0) -> None:
        """Show the inspector starting at a specific sample."""
        self.current_idx = start_idx
        self._print_controls()
        self._create_figure()
        plt.show()

    def _print_controls(self) -> None:
        """Print keyboard controls to console."""
        print("\n" + "=" * 60)
        print("V15 VISUAL INSPECTOR - Keyboard Controls")
        print("=" * 60)
        print("  LEFT/RIGHT : Previous/Next sample")
        print("  UP/DOWN    : Jump 10 samples")
        print("  r          : Random sample")
        print("  w          : Cycle window sizes (best -> 10 -> 20 -> ... -> 80)")
        print("  t          : Swap timeframe views (mixed -> intraday -> multiday)")
        print("  i          : Print detailed sample info")
        print("  q/ESC      : Quit")
        print("=" * 60 + "\n")

    def _create_figure(self) -> None:
        """Create the matplotlib figure with keyboard handler."""
        self.fig = plt.figure(figsize=(16, 14))

        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._update_plot()

    def _on_key(self, event) -> None:
        """Handle keyboard navigation."""
        if event.key == 'right':
            self._navigate(1)
        elif event.key == 'left':
            self._navigate(-1)
        elif event.key == 'up':
            self._navigate(-10)
        elif event.key == 'down':
            self._navigate(10)
        elif event.key == 'r':
            self.current_idx = np.random.randint(len(self.samples))
            self.display_window_idx = None
            self._update_plot()
        elif event.key == 'w':
            self._cycle_window()
        elif event.key == 't':
            self._cycle_timeframe_view()
        elif event.key == 'i':
            self._print_sample_info()
        elif event.key in ('q', 'escape'):
            plt.close('all')

    def _navigate(self, delta: int) -> None:
        """Navigate by delta samples."""
        self.current_idx = (self.current_idx + delta) % len(self.samples)
        self.display_window_idx = None
        self._update_plot()

    def _cycle_window(self) -> None:
        """Cycle through window sizes: best -> 10 -> 20 -> ... -> 80 -> best."""
        sample = self.samples[self.current_idx]
        best_window = getattr(sample, 'best_window', None)

        if self.display_window_idx is None:
            self.display_window_idx = 0
        else:
            self.display_window_idx += 1
            if self.display_window_idx >= len(STANDARD_WINDOWS):
                self.display_window_idx = None

        # Build informative console output
        windows_str = " ".join(
            f"[{w}]" if (self.display_window_idx is not None and STANDARD_WINDOWS[self.display_window_idx] == w)
            else f"({w})" if w == best_window
            else str(w)
            for w in STANDARD_WINDOWS
        )

        if self.display_window_idx is None:
            print(f"\n>>> Window: BEST ({best_window}) <<<")
            print(f"    Available: {windows_str}")
            print(f"    Labels shown are from the cached best window.\n")
        else:
            current_window = STANDARD_WINDOWS[self.display_window_idx]
            is_best = current_window == best_window
            marker = " (BEST)" if is_best else ""
            print(f"\n>>> Window: {current_window}{marker} <<<")
            print(f"    Available: {windows_str}")
            if not is_best:
                print(f"    NOTE: Viewing non-best window. Labels may differ from cache.\n")
            else:
                print(f"    This is the best window for this sample.\n")

        self._update_plot()

    def _cycle_timeframe_view(self) -> None:
        """Cycle through timeframe views: mixed -> intraday -> multiday."""
        self.tf_view_idx = (self.tf_view_idx + 1) % len(TF_VIEW_NAMES)
        view_name = TF_VIEW_NAMES[self.tf_view_idx]

        tf_desc = {
            'mixed': 'Mixed (5min, 15min, 1h, daily)',
            'intraday': 'Intraday (5min, 1h, 2h, 4h)',
            'multiday': 'Multi-day (daily, weekly, monthly, 4h)'
        }
        print(f"Timeframe View: {tf_desc[view_name]}")

        self._update_plot()

    def _update_plot(self) -> None:
        """Update the plot with current sample."""
        from v7.core.channel import select_best_channel

        # Clear figure
        self.fig.clear()

        sample = self.samples[self.current_idx]

        # Determine display window
        if self.display_window_idx is not None:
            display_window = STANDARD_WINDOWS[self.display_window_idx]
        else:
            display_window = None

        # Create 2x2 grid
        gs = self.fig.add_gridspec(2, 2, left=0.06, right=0.98, top=0.88, bottom=0.08,
                                   hspace=0.25, wspace=0.15)

        # Get current timeframe view
        current_view_name = TF_VIEW_NAMES[self.tf_view_idx]
        display_timeframes = TF_VIEW_SETS[current_view_name]

        for idx, tf in enumerate(display_timeframes):
            row, col = idx // 2, idx % 2
            ax = self.fig.add_subplot(gs[row, col])

            # Get data for this timeframe
            df_tf, channel_end_idx, channels_dict = get_sample_data_window(
                self.tsla_df, sample, tf, window=50
            )

            # Select which channel to display
            channel = None
            best_window = None
            actual_display_window = display_window

            if channels_dict:
                best_channel, best_window = select_best_channel(channels_dict)

                if display_window is not None:
                    if display_window in channels_dict:
                        channel = channels_dict[display_window]
                        actual_display_window = display_window
                    else:
                        channel = None
                        actual_display_window = display_window
                else:
                    channel = best_channel
                    actual_display_window = best_window

            # Get labels for this timeframe from cache
            # IMPORTANT: Only use actual_display_window to ensure labels match the displayed channel
            # No fallback to cache_window - if labels don't exist for this window, show "No labels"
            labels = None

            if hasattr(sample, 'labels_per_window') and sample.labels_per_window:
                if actual_display_window and actual_display_window in sample.labels_per_window:
                    window_labels = sample.labels_per_window[actual_display_window]
                    labels = window_labels.get(tf)

            # Plot panel
            plot_timeframe_panel(
                ax, df_tf, channel, labels, tf, channel_end_idx,
                window=actual_display_window or 50,
                channels_dict=channels_dict,
                best_window=best_window,
                display_window=actual_display_window,
            )

        # Build title with clear window indication
        title = f"Sample {self.current_idx + 1}/{len(self.samples)} | {sample.timestamp}"
        best_win = getattr(sample, 'best_window', None)

        if display_window is not None:
            is_best = display_window == best_win
            if is_best:
                title += f" | Window: {display_window} (best)"
                title_color = 'black'
            else:
                title += f" | Window: {display_window} [non-best, best={best_win}]"
                title_color = '#cc6600'  # Orange to indicate non-best
        else:
            title += f" | Window: {best_win} (best)"
            title_color = 'black'

        self.fig.suptitle(title, fontsize=14, fontweight='bold', color=title_color)
        self.fig.canvas.draw_idle()

    def _print_sample_info(self) -> None:
        """Print detailed information about current sample."""
        sample = self.samples[self.current_idx]

        # Determine which window to show info for
        if self.display_window_idx is not None:
            info_window = STANDARD_WINDOWS[self.display_window_idx]
        else:
            info_window = sample.best_window

        print("\n" + "=" * 60)
        print(f"SAMPLE {self.current_idx} DETAILED INFO")
        print("=" * 60)

        print(f"\nTimestamp: {sample.timestamp}")
        print(f"Channel End Index: {sample.channel_end_idx}")
        print(f"Best Window: {sample.best_window}")
        if info_window != sample.best_window:
            print(f"Currently Viewing: Window {info_window}")

        # Features - now stored as flat tf_features dict with TF-prefixed names
        # Format: {tf}_{feature_name}, e.g., "5min_r_squared", "daily_slope"
        features = sample.tf_features
        print(f"\nFeatures ({len(features)} total, flat TF-prefixed format):")

        # Group features by timeframe for display
        tf_feature_counts = {}
        for k in features.keys():
            tf_prefix = k.split('_')[0]
            # Handle timeframes with underscores like "5min" vs feature names
            for tf in TIMEFRAMES:
                if k.startswith(f"{tf}_"):
                    tf_feature_counts[tf] = tf_feature_counts.get(tf, 0) + 1
                    break

        print("  Feature counts by timeframe:")
        for tf in TIMEFRAMES:
            count = tf_feature_counts.get(tf, 0)
            if count > 0:
                print(f"    {tf}: {count} features")

        # Show sample of features for current window's timeframes
        print(f"\n  Sample features (first 10):")
        for i, (k, v) in enumerate(list(features.items())[:10]):
            print(f"    {k}: {v:.4f}")
        if len(features) > 10:
            print(f"    ... and {len(features) - 10} more")

        # Labels - use the displayed window
        window_label = f"window {info_window}" if info_window != sample.best_window else "best window"
        print(f"\n--- Labels by Timeframe ({window_label}) ---")
        labels_dict = sample.labels_per_window.get(info_window, {})

        for tf in TIMEFRAMES:
            tf_label = labels_dict.get(tf)
            if tf_label is None:
                print(f"  {tf:10s}: None")
                continue

            # Handle enum values
            break_dir = tf_label.break_direction
            if hasattr(break_dir, 'name'):
                break_dir_str = break_dir.name
            else:
                break_dir_str = BREAK_NAMES.get(break_dir, '?')

            new_dir = tf_label.next_channel_direction
            if hasattr(new_dir, 'name'):
                new_dir_str = new_dir.name
            else:
                new_dir_str = DIR_NAMES.get(new_dir, '?')

            validity = (
                f"dur={'V' if getattr(tf_label, 'duration_valid', False) else '-'}"
                f" dir={'V' if getattr(tf_label, 'direction_valid', False) else '-'}"
            )

            perm_break = getattr(tf_label, 'permanent_break', False)
            print(f"  {tf:10s}: dur={tf_label.duration_bars:4d} break={str(perm_break):5s} "
                  f"dir={break_dir_str:4s} next={new_dir_str:8s} [{validity}]")

        print("=" * 60 + "\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """Command-line entry point for the visual inspector."""
    parser = argparse.ArgumentParser(
        description='Visual Label Inspector for v15 cache samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m v15.visual_inspector samples.pkl
    python -m v15.visual_inspector --samples path/to/cache.pkl
    python -m v15.visual_inspector -s cache.pkl --start 10
    python -m v15.visual_inspector --cache path/to/cache.pkl  # legacy alias
        """
    )
    # Positional argument (optional) - samples pickle file
    parser.add_argument(
        'samples_file',
        type=str,
        nargs='?',
        default=None,
        help='Path to the pickle cache file (positional argument)'
    )
    # Named argument (optional) - samples pickle file
    parser.add_argument(
        '--samples', '-s',
        type=str,
        default=None,
        help='Path to samples pickle file (.pkl)'
    )
    # Legacy alias for backwards compatibility
    parser.add_argument(
        '--cache', '-c',
        type=str,
        default=None,
        help='Alias for --samples (backwards compatibility)'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data',
        help='Directory containing market data CSVs (default: data)'
    )
    parser.add_argument(
        '--start', '-i',
        type=int,
        default=0,
        help='Starting sample index (default: 0)'
    )

    args = parser.parse_args()

    # Resolve cache path: positional > --samples > --cache
    cache_file = args.samples_file or args.samples or args.cache

    if not cache_file:
        parser.error("Please provide a samples pickle file as a positional argument or via --samples/-s")

    # Load cache
    cache_path = Path(cache_file)
    if not cache_path.exists():
        print(f"Error: Cache file not found: {cache_path}")
        sys.exit(1)

    print(f"Loading samples from {cache_path}...")
    with open(cache_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"  Loaded {len(samples)} samples")

    # Load market data
    print(f"Loading market data from {args.data_dir}...")
    tsla_df, _, _ = load_market_data(args.data_dir)
    print(f"  Loaded {len(tsla_df)} bars")

    # Create and run inspector
    inspector = VisualInspector(samples, tsla_df)
    inspector.show(start_idx=args.start)


if __name__ == '__main__':
    main()
