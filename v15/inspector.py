"""
v15/inspector.py - Clean Visual Inspector for Channel Analysis

Simple, focused visualizer that shows:
- Channel with projected bounds (FRESHLY DETECTED per panel)
- Price action (candlesticks)
- Break markers (first, biggest, permanent)
- Info panel with labels
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
import numpy as np
import pandas as pd

# Import v15 types
from v15.dtypes import ChannelSample, ChannelLabels, TIMEFRAMES, STANDARD_WINDOWS
from v15.labels import DetectedChannel
from v15.config import TF_MAX_SCAN, BREAK_DETECTION, BREAK_MARKER_COLORS, channel_sort_key
from v15.data import load_market_data

# Import channel detection from v7
from v7.core.channel import detect_channel, Channel


class Inspector:
    """Interactive channel inspector with keyboard navigation."""

    def __init__(self, samples: List[ChannelSample], tsla_df: pd.DataFrame,
                 channel_map: Optional[Dict[Tuple[str, int], List[DetectedChannel]]] = None):
        self.samples = samples
        self.tsla_df = tsla_df
        self.channel_map = channel_map
        self.current_idx = 0
        self.current_window = None  # None = best window
        self.fig = None
        self.axes = None

    def show(self, start_idx: int = 0):
        """Launch the inspector."""
        self.current_idx = start_idx
        self._setup_figure()
        self._connect_events()
        self._draw()
        plt.show()

    def _setup_figure(self):
        """Create figure with 2x2 grid for 4 timeframes."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.canvas.manager.set_window_title('V15 Channel Inspector')
        self.axes = self.axes.flatten()  # Make it easy to iterate

    def _connect_events(self):
        """Connect keyboard events."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        """Handle keyboard navigation."""
        if event.key == 'right':
            self.current_idx = min(self.current_idx + 1, len(self.samples) - 1)
            self._draw()
        elif event.key == 'left':
            self.current_idx = max(self.current_idx - 1, 0)
            self._draw()
        elif event.key == 'w':
            self._cycle_window()
            self._draw()
        elif event.key in ['q', 'escape']:
            plt.close(self.fig)

    def _cycle_window(self):
        """Cycle through window sizes."""
        windows = [None] + STANDARD_WINDOWS  # None = best
        if self.current_window is None:
            self.current_window = STANDARD_WINDOWS[0]
        else:
            try:
                idx = STANDARD_WINDOWS.index(self.current_window)
                self.current_window = STANDARD_WINDOWS[(idx + 1) % len(STANDARD_WINDOWS)]
            except ValueError:
                self.current_window = None

    def _get_best_window_for_tf(self, sample: ChannelSample, tf: str) -> int:
        """Get the best window for a specific timeframe.

        Selection priority (via channel_sort_key from config):
        1. Has valid labels with direction >= 0
        2. Meets minimum R-squared threshold (quality filter)
        3. Higher bounce_count (channel quality metric)
        4. Higher r_squared (better fit)
        5. Smaller window (tiebreaker - more stable)

        Uses channel_sort_key from config for consistent ranking across the system.
        """
        from v15.core.resample import resample_ohlc

        best_window = sample.best_window
        best_score = (-1, -1.0, 1000)  # (bounce_count, r2, window) - window is negative priority

        MIN_R2_THRESHOLD = BREAK_DETECTION['min_r2_threshold']  # From config

        # Get resampled data for this TF once (for fresh channel detection)
        BARS_PER_TF = {
            '5min': 1, '15min': 3, '30min': 6,
            '1h': 12, '2h': 24, '3h': 36, '4h': 48,
            'daily': 78, 'weekly': 390, 'monthly': 1638
        }
        bars_per_tf = BARS_PER_TF.get(tf, 1)
        max_window = max(STANDARD_WINDOWS)
        lookback_5min = (max_window + 50) * bars_per_tf

        channel_end_idx = sample.channel_end_idx
        start_idx = max(0, channel_end_idx - lookback_5min)
        end_idx = min(len(self.tsla_df), channel_end_idx + 10 * bars_per_tf)

        df_slice = self.tsla_df.iloc[start_idx:end_idx].copy()

        if tf != '5min':
            df_tf = resample_ohlc(df_slice, tf, keep_partial=False)
        else:
            df_tf = df_slice

        # Find sample position in resampled data
        sample_ts = sample.timestamp
        sample_idx = df_tf.index.searchsorted(sample_ts, side='right') - 1
        if sample_idx < 0:
            sample_idx = 0
        sample_idx = min(sample_idx, len(df_tf) - 1)

        for window, assets_dict in sample.labels_per_window.items():
            if 'tsla' not in assets_dict:
                continue
            labels = assets_dict['tsla'].get(tf)
            if labels is None:
                continue
            if labels.source_channel_direction < 0:
                continue  # Invalid channel

            r2 = labels.source_channel_r_squared
            if r2 < MIN_R2_THRESHOLD:
                continue  # Skip poor quality channels

            # Try to detect channel fresh to get bounce_count
            bounce_count = 0
            if len(df_tf) >= window:
                detection_start = max(0, sample_idx - window)
                detection_end = sample_idx + 1
                channel_df = df_tf.iloc[detection_start:detection_end + 1]

                if len(channel_df) >= window:
                    if not channel_df[['open', 'high', 'low', 'close']].isna().any().any():
                        try:
                            channel = detect_channel(channel_df, window=window, min_cycles=1)
                            if channel is not None and channel.valid:
                                # Use channel_sort_key for consistent ranking
                                bounce_count, _ = channel_sort_key(channel)
                        except Exception:
                            pass

            score = (bounce_count, r2, -window)  # -window so smaller is better

            if score > best_score:
                best_score = score
                best_window = window

        return best_window

    def _draw(self):
        """Draw the current sample."""
        # Clear all axes
        for ax in self.axes:
            ax.clear()

        sample = self.samples[self.current_idx]

        # Get timeframes to display (mixed: 5min, 1h, daily, weekly)
        display_tfs = ['5min', '1h', 'daily', 'weekly']

        # Update title based on mode
        if self.current_window is None:
            # Best per TF mode - each timeframe uses its own best window
            window_label = "Best per TF"
        else:
            # Fixed window mode - all timeframes use the same window
            window_label = f"Fixed W:{self.current_window}"

        self.fig.suptitle(
            f"Sample {self.current_idx + 1}/{len(self.samples)} | "
            f"{sample.timestamp} | {window_label}",
            fontsize=14, fontweight='bold'
        )

        # Draw each timeframe panel
        for i, tf in enumerate(display_tfs):
            if self.current_window is None:
                # Best per TF mode - determine best window for each timeframe
                best_window_for_tf = self._get_best_window_for_tf(sample, tf)
                self._draw_panel(self.axes[i], sample, tf, best_window_for_tf)
            else:
                # Fixed window mode - use the same window for all
                self._draw_panel(self.axes[i], sample, tf, self.current_window)

        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.fig.canvas.draw_idle()

    def _draw_panel(self, ax, sample: ChannelSample, tf: str, window: int):
        """Draw a single timeframe panel with channel, price, and breaks.

        FRESH CHANNEL DETECTION: Detects the channel fresh on the displayed data
        instead of using precomputed channel parameters from labels.
        """
        from v15.core.resample import resample_ohlc

        # 5min bars per TF bar (for calculating lookback)
        BARS_PER_TF = {
            '5min': 1, '15min': 3, '30min': 6,
            '1h': 12, '2h': 24, '3h': 36, '4h': 48,
            'daily': 78, 'weekly': 390, 'monthly': 1638
        }

        # Get labels for this tf/window (for break info, NOT channel parameters)
        labels = None
        if window in sample.labels_per_window:
            if 'tsla' in sample.labels_per_window[window]:
                labels = sample.labels_per_window[window]['tsla'].get(tf)

        # Get channel end position in 5min data
        channel_end_idx = sample.channel_end_idx

        # TF-specific forward bars for plot range (breaks can occur far forward)
        tf_forward_bars = {
            '5min': 120, '15min': 100, '30min': 80,
            '1h': 250, '2h': 200, '3h': 150, '4h': 100,
            'daily': 50, 'weekly': 30, 'monthly': 15
        }

        # Resample TSLA data to this timeframe
        # Calculate how many 5min bars we need based on TF
        bars_per_tf = BARS_PER_TF.get(tf, 1)
        lookback_5min = (window + 50) * bars_per_tf  # window + extra for resampling
        max_forward_tf = tf_forward_bars.get(tf, 50)
        forward_5min = (max_forward_tf + 20) * bars_per_tf  # Forward projection (with buffer)

        start_idx = max(0, channel_end_idx - lookback_5min)
        end_idx = min(len(self.tsla_df), channel_end_idx + forward_5min)

        df_slice = self.tsla_df.iloc[start_idx:end_idx].copy()

        # Resample to target timeframe (drop partial bars for cleaner channel detection)
        if tf != '5min':
            df_tf = resample_ohlc(df_slice, tf, keep_partial=False)
        else:
            df_tf = df_slice

        if len(df_tf) < window:
            ax.set_title(f"{tf} - Insufficient data")
            return

        # Find sample position in resampled data (by timestamp)
        sample_ts = sample.timestamp

        # Find closest index in resampled data
        sample_idx = df_tf.index.searchsorted(sample_ts, side='right') - 1
        if sample_idx < 0:
            sample_idx = 0
        sample_idx = min(sample_idx, len(df_tf) - 1)

        # === FRESH CHANNEL DETECTION ===
        # Detect channel on the exact data being displayed
        # Channel is `window` bars ending at sample position
        channel_start_idx = max(0, sample_idx - window + 1)
        channel_end_idx_tf = sample_idx

        # Get the slice for channel detection (need window+1 bars for detect_channel's internal slicing)
        # detect_channel uses df.iloc[-(window+1):-1], so we pass it window+1 bars ending at sample+1
        detection_start = max(0, sample_idx - window)
        detection_end = sample_idx + 1  # +1 because detect_channel excludes the last bar
        channel_df = df_tf.iloc[detection_start:detection_end + 1]

        # Detect channel fresh
        # Use min_cycles=1 consistently (matches scanner's detect_all_channels in labels.py)
        channel = None
        if len(channel_df) >= window:
            # Check for NaN before detection (resampled data can have NaN)
            if channel_df[['open', 'high', 'low', 'close']].isna().any().any():
                channel = None
            else:
                try:
                    channel = detect_channel(channel_df, window=window, min_cycles=1)
                except Exception as e:
                    channel = None

        # Calculate plot range
        plot_start = max(0, sample_idx - window + 1)
        max_forward = tf_forward_bars.get(tf, 50)
        forward_bars = min(max_forward, len(df_tf) - sample_idx - 1)
        plot_end = sample_idx + forward_bars + 1

        df_plot = df_tf.iloc[plot_start:plot_end]
        n_bars = len(df_plot)
        x = np.arange(n_bars)

        # Draw candlesticks
        self._draw_candles(ax, df_plot, x)

        # Ensure x-axis shows full range for markers (disable autoscale)
        ax.set_xlim(-1, n_bars)
        ax.autoscale(enable=False, axis='x')

        # Draw channel if detected
        if channel is not None and channel.valid:
            self._draw_channel_fresh(ax, channel, window, forward_bars, n_bars, df_plot, sample)

        # If we have a fresh channel but no labels, detect breaks fresh
        fresh_break_info = None
        # ALWAYS compute fresh break detection when we have a valid channel
        # This ensures breaks are relative to the DISPLAYED channel, not stored labels
        # which may come from a different time period
        if channel is not None and channel.valid:
            from v15.core.break_scanner import scan_for_break

            # Get TF-appropriate max scan
            max_scan = TF_MAX_SCAN.get(tf, 200)

            # Get forward data for break scanning (from sample position forward)
            forward_start = sample_idx + 1
            forward_end = min(len(df_tf), forward_start + max_scan)

            if forward_end > forward_start:
                forward_data = df_tf.iloc[forward_start:forward_end]

                try:
                    result = scan_for_break(
                        channel=channel,
                        forward_high=forward_data['high'].values,
                        forward_low=forward_data['low'].values,
                        forward_close=forward_data['close'].values,
                        min_break_magnitude=BREAK_DETECTION['min_break_magnitude'],
                        return_threshold_bars=BREAK_DETECTION['return_threshold_bars'],
                        max_scan_bars=max_scan  # Use TF-specific value
                    )
                    fresh_break_info = result
                except Exception as e:
                    fresh_break_info = None

        # ALWAYS use fresh break detection for accurate visualization
        # Stored labels may come from a different channel period than what's displayed
        if fresh_break_info is not None:
            self._draw_breaks_fresh(ax, fresh_break_info, window, n_bars, df_plot, sample)
            self._draw_info_fresh(ax, fresh_break_info, tf, window, sample, channel)
        elif labels is not None:
            # Fallback to stored labels if fresh detection failed
            self._draw_breaks(ax, labels, window, n_bars, df_plot, plot_start, sample)
            self._draw_info(ax, labels, tf, window, sample, channel)

        # Draw vertical line at SAMPLE position (the prediction point)
        sample_ts = sample.timestamp
        if sample_ts in df_plot.index:
            sample_x = df_plot.index.get_loc(sample_ts)
        else:
            sample_x = df_plot.index.searchsorted(sample_ts, side='right') - 1
            if sample_x < 0:
                sample_x = 0
            if sample_x >= len(df_plot):
                sample_x = len(df_plot) - 1
        ax.axvline(sample_x, color='blue', linestyle='-', linewidth=2, alpha=0.7)

        # Set title
        if channel is not None and channel.valid:
            r2 = channel.r_squared
            direction = ['BEAR', 'SIDE', 'BULL'][channel.direction]
        elif labels is not None:
            r2 = labels.source_channel_r_squared
            direction = ['BEAR', 'SIDE', 'BULL'][labels.source_channel_direction] if labels.source_channel_direction >= 0 else '?'
        else:
            r2 = 0
            direction = '?'
        ax.set_title(f"{tf} | {direction} | R2:{r2:.3f} | W:{window}")
        ax.set_xlabel('Bars')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)

    def _draw_candles(self, ax, df: pd.DataFrame, x: np.ndarray):
        """Draw candlestick chart."""
        for i, (idx, row) in enumerate(df.iterrows()):
            color = 'green' if row['close'] >= row['open'] else 'red'
            # Body
            ax.bar(x[i], abs(row['close'] - row['open']),
                   bottom=min(row['open'], row['close']),
                   color=color, width=0.6, alpha=0.8)
            # Wicks
            ax.vlines(x[i], row['low'], row['high'], color=color, linewidth=0.8)

    def _draw_channel_fresh(self, ax, channel: Channel, window: int, forward_bars: int,
                            n_bars: int, df_plot: pd.DataFrame, sample: ChannelSample):
        """Draw channel bounds using freshly detected channel object.

        The Channel object has pre-computed line arrays:
        - channel.upper_line: Upper boundary values
        - channel.lower_line: Lower boundary values
        - channel.center_line: Center regression line

        These are computed on the EXACT data being displayed.
        """
        # Find sample position in df_plot
        sample_ts = sample.timestamp
        if sample_ts in df_plot.index:
            sample_plot_idx = df_plot.index.get_loc(sample_ts)
        else:
            sample_plot_idx = df_plot.index.searchsorted(sample_ts, side='right') - 1
            sample_plot_idx = max(0, min(sample_plot_idx, len(df_plot) - 1))

        # === CHANNEL DRAWING using pre-computed lines ===
        # Channel spans `window` bars ending at sample position
        channel_start_plot = max(0, sample_plot_idx - window + 1)
        channel_end_plot = min(n_bars - 1, sample_plot_idx)

        if channel_end_plot <= channel_start_plot:
            return  # Nothing to draw

        # Plot x-coordinates for channel
        channel_x_plot = np.arange(channel_start_plot, channel_end_plot + 1)

        # Number of bars to draw
        n_channel_bars = len(channel_x_plot)

        # Use the pre-computed lines from the channel object
        # These arrays have length = window (or actual detected length)
        channel_len = len(channel.center_line)

        # Align the channel lines with our plot coordinates
        # We want the last `n_channel_bars` values from the channel lines
        if n_channel_bars <= channel_len:
            start_in_channel = channel_len - n_channel_bars
            center = channel.center_line[start_in_channel:]
            upper = channel.upper_line[start_in_channel:]
            lower = channel.lower_line[start_in_channel:]
        else:
            # Channel is shorter than our plot range - use all of it
            center = channel.center_line
            upper = channel.upper_line
            lower = channel.lower_line
            # Adjust x coordinates to match
            channel_x_plot = channel_x_plot[-len(center):]

        # Draw channel
        ax.fill_between(channel_x_plot, lower, upper, alpha=0.15, color='orange')
        ax.plot(channel_x_plot, center, 'b--', alpha=0.5, linewidth=1)
        ax.plot(channel_x_plot, upper, 'orange', linewidth=1.5, linestyle='--')
        ax.plot(channel_x_plot, lower, 'orange', linewidth=1.5, linestyle='--')

        # === PROJECTION (forward from sample/channel end) ===
        if forward_bars > 0:
            proj_start_plot = sample_plot_idx + 1
            proj_end_plot = min(proj_start_plot + forward_bars, n_bars)

            if proj_end_plot > proj_start_plot:
                proj_x_plot = np.arange(proj_start_plot, proj_end_plot)

                # Project using slope and intercept from the detected channel
                # Projection continues from where the channel ended
                # x=window is the first projection bar
                proj_regression_x = window + (proj_x_plot - proj_start_plot)

                # Project using slope
                proj_center = channel.slope * proj_regression_x + channel.intercept
                proj_upper = proj_center + 2 * channel.std_dev
                proj_lower = proj_center - 2 * channel.std_dev

                # Bounds checking: stock prices can't go negative
                proj_center = np.maximum(proj_center, 0.01)
                proj_upper = np.maximum(proj_upper, 0.01)
                proj_lower = np.maximum(proj_lower, 0.01)

                # Draw projected bounds
                ax.plot(proj_x_plot, proj_upper, 'orange', linewidth=1, linestyle=':', alpha=0.6)
                ax.plot(proj_x_plot, proj_lower, 'orange', linewidth=1, linestyle=':', alpha=0.6)
                ax.fill_between(proj_x_plot, proj_lower, proj_upper, alpha=0.05, color='orange')

    def _draw_breaks(self, ax, labels: ChannelLabels, window: int, n_bars: int,
                     df_plot: pd.DataFrame, plot_start: int, sample: ChannelSample):
        """Draw break markers: first (orange hollow), biggest (red), permanent (purple).

        Break info comes from labels (precomputed) - this is correct because breaks
        are events that happened AFTER the channel, and need to be tracked.
        """
        # Colors from config (single source of truth)
        FIRST_COLOR = BREAK_MARKER_COLORS['first']
        BIGGEST_COLOR = BREAK_MARKER_COLORS['biggest']
        PERM_COLOR = BREAK_MARKER_COLORS['permanent']

        # Find sample position in df_plot (this is also the channel end)
        sample_ts = sample.timestamp
        if sample_ts in df_plot.index:
            sample_plot_idx = df_plot.index.get_loc(sample_ts)
        else:
            sample_plot_idx = df_plot.index.searchsorted(sample_ts, side='right') - 1
            sample_plot_idx = max(0, min(sample_plot_idx, len(df_plot) - 1))

        # Helper to convert bars-after-sample to plot x coordinate
        # bars_after = 0 means the next bar after sample (sample_plot_idx + 1)
        def break_to_plot_x(bars_after):
            return sample_plot_idx + 1 + bars_after

        # FIRST BREAK - hollow orange triangle
        bars_to_first = getattr(labels, 'bars_to_first_break', -1)
        if bars_to_first >= 0:
            first_x = break_to_plot_x(bars_to_first)
            if 0 <= first_x < n_bars:
                break_dir = labels.break_direction
                marker = '^' if break_dir == 1 else 'v'
                bar_idx = int(first_x)
                if 0 <= bar_idx < len(df_plot):
                    price = df_plot.iloc[bar_idx]['high' if break_dir == 1 else 'low']
                    ax.scatter([first_x], [price], marker=marker, s=180,
                              c='none', edgecolors=FIRST_COLOR, linewidths=2.5, zorder=10)

        # BIGGEST BREAK - filled red triangle (max magnitude exit)
        exit_bars = getattr(labels, 'exit_bars', []) or []
        exit_mags = getattr(labels, 'exit_magnitudes', []) or []
        exit_types = getattr(labels, 'exit_types', []) or []

        if exit_mags and len(exit_mags) > 0 and any(m > 0 for m in exit_mags):
            max_idx = np.argmax(exit_mags)
            max_mag = exit_mags[max_idx]
            biggest_bar = exit_bars[max_idx] if max_idx < len(exit_bars) else -1

            # Only show if at a different bar than first break (avoid overlapping markers)
            if max_mag > 0.5 and biggest_bar >= 0 and biggest_bar != bars_to_first:
                biggest_x = break_to_plot_x(biggest_bar)
                if 0 <= biggest_x < n_bars:
                    biggest_dir = exit_types[max_idx] if max_idx < len(exit_types) else 0
                    marker = '^' if biggest_dir == 1 else 'v'
                    bar_idx = int(biggest_x)
                    if 0 <= bar_idx < len(df_plot):
                        price = df_plot.iloc[bar_idx]['high' if biggest_dir == 1 else 'low']
                        ax.scatter([biggest_x], [price], marker=marker, s=200,
                                  c=BIGGEST_COLOR, edgecolors='white', linewidths=1.5, zorder=11)

        # PERMANENT BREAK - filled purple triangle
        perm_dir = getattr(labels, 'permanent_break_direction', -1)
        perm_bar = getattr(labels, 'bars_to_permanent_break', -1)

        if perm_dir >= 0 and perm_bar >= 0:
            perm_x = break_to_plot_x(perm_bar)
            if 0 <= perm_x < n_bars:
                marker = '^' if perm_dir == 1 else 'v'
                bar_idx = int(perm_x)
                if 0 <= bar_idx < len(df_plot):
                    price = df_plot.iloc[bar_idx]['high' if perm_dir == 1 else 'low']
                    ax.scatter([perm_x], [price], marker=marker, s=220,
                              c=PERM_COLOR, edgecolors='black', linewidths=2, zorder=12)

    def _draw_info(self, ax, labels: ChannelLabels, tf: str, window: int,
                   sample: ChannelSample, channel: Optional[Channel] = None):
        """Draw info panel with label values."""
        # Use freshly detected channel for channel metrics if available
        if channel is not None and channel.valid:
            direction = ['BEAR', 'SIDE', 'BULL'][channel.direction]
            r_squared = channel.r_squared
            channel_source = "(fresh)"
        else:
            direction = ['BEAR', 'SIDE', 'BULL'][labels.source_channel_direction] if labels.source_channel_direction >= 0 else '?'
            r_squared = labels.source_channel_r_squared
            channel_source = "(labels)"

        # Break info from labels
        break_dir = 'UP' if labels.break_direction == 1 else 'DOWN'
        perm_dir_val = labels.permanent_break_direction
        perm_dir = 'UP' if perm_dir_val == 1 else ('DOWN' if perm_dir_val == 0 else 'NONE')

        returned = 'Yes' if labels.returned_to_channel else 'No'
        bounces = labels.bounces_after_return
        # Use fresh channel's bounce stats when available (fixes bug where STORED labels
        # from a DIFFERENT channel were shown for FRESHLY detected channels)
        if channel is not None and channel.valid:
            round_trips = channel.complete_cycles
            bounce_count = channel.bounce_count
        else:
            round_trips = getattr(labels, 'round_trip_bounces', 0)
            bounce_count = getattr(labels, 'source_channel_bounce_count', 0)

        first_mag = getattr(labels, 'break_magnitude', 0.0)
        perm_mag = getattr(labels, 'permanent_break_magnitude', 0.0)

        bars_to_first = getattr(labels, 'bars_to_first_break', -1)
        bars_to_perm = getattr(labels, 'bars_to_permanent_break', -1)

        durability = getattr(labels, 'durability_score', 0.0)

        # Best next channel info
        next_dir_val = getattr(labels, 'best_next_channel_direction', -1)
        next_bars_away = getattr(labels, 'best_next_channel_bars_away', -1)
        next_bounce_count = getattr(labels, 'best_next_channel_bounce_count', 0)
        next_dir_map = {-1: 'NONE', 0: 'BEAR', 1: 'SIDE', 2: 'BULL'}
        next_dir_str = next_dir_map.get(next_dir_val, 'NONE')

        # Determine best window for this TF
        best_window_for_tf = self._get_best_window_for_tf(sample, tf)
        is_best = (window == best_window_for_tf)
        window_label = f"W:{window} (best)" if is_best else f"W:{window}"

        info_lines = [
            f"{tf} | {direction} | {window_label} {channel_source}",
            f"R2: {r_squared:.4f}",
            f"Break: {break_dir} @ bar {bars_to_first} (mag: {first_mag:.2f})",
            f"Perm: {perm_dir} @ bar {bars_to_perm} (mag: {perm_mag:.2f})",
            f"Bounces: {bounce_count} | Return: {returned} | RoundTrips: {round_trips} (exits: {bounces})",
            f"Durability: {durability:.2f}",
            f"Next: {next_dir_str} @ +{next_bars_away} bars (bounces: {next_bounce_count})",
        ]

        info_text = '\n'.join(info_lines)

        # Position in top-left of plot
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _draw_breaks_fresh(self, ax, break_result, window: int, n_bars: int,
                           df_plot: pd.DataFrame, sample: ChannelSample):
        """Draw break markers from fresh BreakResult: first (orange hollow), permanent (purple).

        Uses BreakResult from scan_for_break() instead of precomputed labels.
        """
        from v15.core.break_scanner import BreakResult

        # Colors from config (single source of truth)
        FIRST_COLOR = BREAK_MARKER_COLORS['first']
        BIGGEST_COLOR = BREAK_MARKER_COLORS['biggest']
        PERM_COLOR = BREAK_MARKER_COLORS['permanent']

        # Find sample position in df_plot (this is also the channel end)
        sample_ts = sample.timestamp
        if sample_ts in df_plot.index:
            sample_plot_idx = df_plot.index.get_loc(sample_ts)
        else:
            sample_plot_idx = df_plot.index.searchsorted(sample_ts, side='right') - 1
            sample_plot_idx = max(0, min(sample_plot_idx, len(df_plot) - 1))

        # Helper to convert bars-after-sample to plot x coordinate
        # bars_after = 0 means the next bar after sample (sample_plot_idx + 1)
        def break_to_plot_x(bars_after):
            return sample_plot_idx + 1 + bars_after

        # FIRST BREAK - hollow orange triangle
        if break_result.break_detected and break_result.break_bar >= 0:
            first_x = break_to_plot_x(break_result.break_bar)
            if 0 <= first_x < n_bars:
                break_dir = break_result.break_direction
                marker = '^' if break_dir == 1 else 'v'
                bar_idx = int(first_x)
                if 0 <= bar_idx < len(df_plot):
                    price = df_plot.iloc[bar_idx]['high' if break_dir == 1 else 'low']
                    ax.scatter([first_x], [price], marker=marker, s=180,
                              c='none', edgecolors=FIRST_COLOR, linewidths=2.5, zorder=10)

        # PERMANENT BREAK - filled purple triangle
        perm_dir = break_result.permanent_break_direction
        perm_bar = break_result.permanent_break_bar

        if perm_dir >= 0 and perm_bar >= 0:
            perm_x = break_to_plot_x(perm_bar)
            if 0 <= perm_x < n_bars:
                marker = '^' if perm_dir == 1 else 'v'
                bar_idx = int(perm_x)
                if 0 <= bar_idx < len(df_plot):
                    price = df_plot.iloc[bar_idx]['high' if perm_dir == 1 else 'low']
                    ax.scatter([perm_x], [price], marker=marker, s=220,
                              c=PERM_COLOR, edgecolors='black', linewidths=2, zorder=12)

        # BIGGEST BREAK - filled red triangle (max magnitude exit)
        # Compute from all_exit_events since BreakResult doesn't store biggest directly
        biggest_bar = -1
        biggest_mag = 0.0
        biggest_dir = 0
        if break_result.all_exit_events:
            for evt in break_result.all_exit_events:
                if evt.magnitude > biggest_mag:
                    biggest_mag = evt.magnitude
                    biggest_bar = evt.bar_index
                    biggest_dir = 1 if evt.exit_type == 'upper' else 0

        first_bar = break_result.break_bar if break_result.break_detected else -1
        perm_bar = break_result.permanent_break_bar

        # Reference bar for distance check: permanent break if available, else first break
        reference_bar = perm_bar if perm_bar >= 0 else first_bar
        max_distance = BREAK_DETECTION['biggest_break_max_distance']

        # Only show if at a different bar than first break (avoid overlapping markers)
        # AND within max_distance bars of the reference break
        within_distance = (reference_bar < 0 or abs(biggest_bar - reference_bar) <= max_distance)
        if biggest_bar >= 0 and biggest_mag > 0.5 and biggest_bar != first_bar and within_distance:
            biggest_x = break_to_plot_x(biggest_bar)
            if 0 <= biggest_x < n_bars:
                marker = '^' if biggest_dir == 1 else 'v'
                bar_idx = int(biggest_x)
                if 0 <= bar_idx < len(df_plot):
                    price = df_plot.iloc[bar_idx]['high' if biggest_dir == 1 else 'low']
                    ax.scatter([biggest_x], [price], marker=marker, s=200,
                              c=BIGGEST_COLOR, edgecolors='white', linewidths=1.5, zorder=11)

    def _lookup_next_channel_from_map(
        self, tf: str, window: int, sample: ChannelSample
    ) -> Optional[Tuple[int, int, int]]:
        """
        Look up the best next channel from the channel map.

        Searches the channel_map for the current (tf, window), finds the current
        channel based on sample timestamp, and returns info about the best next
        channel (ranked by bounce_count, then r_squared).

        Args:
            tf: Timeframe string
            window: Window size
            sample: The current ChannelSample

        Returns:
            Tuple of (direction, bars_away, bounce_count) or None if not found.
            direction: 0=BEAR, 1=SIDE, 2=BULL
        """
        if self.channel_map is None:
            return None

        # Handle nested structure: channel_map may be {'tsla': {...}, 'spy': {...}}
        # or directly a ChannelMap Dict[(tf, window), List[DetectedChannel]]
        asset_map = self.channel_map
        if isinstance(self.channel_map, dict) and 'tsla' in self.channel_map:
            asset_map = self.channel_map.get('tsla', {})

        key = (tf, window)
        channels = asset_map.get(key, [])
        if not channels:
            return None

        sample_ts = sample.timestamp

        # Find the current channel index by matching end_timestamp to sample timestamp
        # The sample is created at the channel end position
        current_idx = -1
        for i, detected in enumerate(channels):
            if detected.end_timestamp is not None and detected.end_timestamp == sample_ts:
                current_idx = i
                break

        if current_idx < 0:
            # Fallback: find the channel whose end_timestamp is closest to (but <= ) sample_ts
            for i, detected in enumerate(channels):
                if detected.end_timestamp is not None and detected.end_timestamp <= sample_ts:
                    current_idx = i
                else:
                    break

        if current_idx < 0 or current_idx >= len(channels) - 1:
            # No current channel found or no next channels available
            return None

        # Look at next 1-2 channels
        next_channels_info = []
        for offset in range(1, 3):
            idx = current_idx + offset
            if idx < len(channels):
                next_ch = channels[idx]
                current_ch = channels[current_idx]
                bars_away = next_ch.start_idx - current_ch.end_idx
                next_channels_info.append((next_ch, bars_away))

        if not next_channels_info:
            return None

        # Find the "best" channel using channel_sort_key (bounce_count, then r_squared)
        best_ch, best_bars_away = next_channels_info[0]
        best_sort_key = channel_sort_key(best_ch.channel)

        for next_ch, bars_away in next_channels_info[1:]:
            sort_key = channel_sort_key(next_ch.channel)
            if sort_key > best_sort_key:
                best_sort_key = sort_key
                best_ch = next_ch
                best_bars_away = bars_away

        # Extract direction and bounce_count from best channel
        direction = best_ch.direction  # 0=BEAR, 1=SIDEWAYS, 2=BULL
        bounce_count = best_ch.channel.bounce_count if best_ch.channel.bounce_count is not None else 0

        return (direction, best_bars_away, bounce_count)

    def _draw_info_fresh(self, ax, break_result, tf: str, window: int,
                         sample: ChannelSample, channel: Optional[Channel] = None):
        """Draw info panel with fresh break result values."""
        from v15.core.break_scanner import BreakResult, compute_durability_from_result

        # Channel info from freshly detected channel
        if channel is not None and channel.valid:
            direction = ['BEAR', 'SIDE', 'BULL'][channel.direction]
            r_squared = channel.r_squared
        else:
            direction = '?'
            r_squared = 0.0

        # Break info from fresh BreakResult
        if break_result.break_detected:
            break_dir = 'UP' if break_result.break_direction == 1 else 'DOWN'
            bars_to_first = break_result.break_bar
            first_mag = break_result.break_magnitude
        else:
            break_dir = 'NONE'
            bars_to_first = -1
            first_mag = 0.0

        perm_dir_val = break_result.permanent_break_direction
        perm_dir = 'UP' if perm_dir_val == 1 else ('DOWN' if perm_dir_val == 0 else 'NONE')
        bars_to_perm = break_result.permanent_break_bar
        perm_mag = break_result.permanent_break_magnitude

        returned = 'Yes' if break_result.is_false_break else 'No'
        # Use fresh channel's bounce stats when available (for consistency with _draw_info)
        if channel is not None and channel.valid:
            round_trips = channel.complete_cycles
            bounce_count = channel.bounce_count
        else:
            round_trips = break_result.round_trip_bounces
            bounce_count = 0
        exits_count = len(break_result.all_exit_events) if break_result.all_exit_events else 0

        # Compute durability score
        _, _, durability = compute_durability_from_result(break_result)

        # Best next channel info - look up from channel_map if available
        next_channel_info = self._lookup_next_channel_from_map(tf, window, sample)
        if next_channel_info is not None:
            next_dir_val, next_bars_away, next_bounce_count = next_channel_info
            next_dir_map = {0: 'BEAR', 1: 'SIDE', 2: 'BULL'}
            next_dir_str = next_dir_map.get(next_dir_val, 'N/A')
        else:
            # Fallback to N/A display
            next_dir_str = 'N/A'
            next_bars_away = -1
            next_bounce_count = 0

        info_lines = [
            f"{tf} | {direction} | W:{window} (fresh)",
            f"R2: {r_squared:.4f}",
            f"Break: {break_dir} @ bar {bars_to_first} (mag: {first_mag:.2f})",
            f"Perm: {perm_dir} @ bar {bars_to_perm} (mag: {perm_mag:.2f})",
            f"Bounces: {bounce_count} | Return: {returned} | RoundTrips: {round_trips} (exits: {exits_count})",
            f"Durability: {durability:.2f}",
            f"Next: {next_dir_str} @ +{next_bars_away} bars (bounces: {next_bounce_count})",
        ]

        info_text = '\n'.join(info_lines)

        # Position in top-left of plot - use cyan background to indicate fresh detection
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))


def main():
    parser = argparse.ArgumentParser(description='V15 Channel Inspector')
    parser.add_argument('samples_file', nargs='?', help='Path to samples pickle file')
    parser.add_argument('--samples', '-s', help='Path to samples pickle file')
    parser.add_argument('--data-dir', '-d', default='data', help='Market data directory')
    parser.add_argument('--start', '-i', type=int, default=0, help='Starting sample index')
    parser.add_argument('--channel-map', '-c', help='Path to pickled channel map file (saved during scanning)')
    args = parser.parse_args()

    # Get samples file path
    samples_path = args.samples_file or args.samples
    if not samples_path:
        print("Error: Please provide a samples file")
        sys.exit(1)

    # Load samples
    print(f"Loading samples from {samples_path}...")
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples")

    # Load channel map if provided
    channel_map = None
    if args.channel_map:
        print(f"Loading channel map from {args.channel_map}...")
        with open(args.channel_map, 'rb') as f:
            channel_map = pickle.load(f)
        # Channel map structure: {'tsla': {...}, 'spy': {...}}
        if isinstance(channel_map, dict):
            total_channels = sum(
                len(channels) for asset_map in channel_map.values()
                for channels in asset_map.values()
            ) if channel_map else 0
            print(f"Loaded channel map with {total_channels} channels")

    # Load market data
    print(f"Loading market data from {args.data_dir}...")
    tsla_df, _, _ = load_market_data(args.data_dir)
    print(f"Loaded {len(tsla_df)} bars")

    # Launch inspector
    inspector = Inspector(samples, tsla_df, channel_map=channel_map)
    inspector.show(start_idx=args.start)


if __name__ == '__main__':
    main()
