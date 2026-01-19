#!/usr/bin/env python3
"""
Dual-Asset Visual Inspector for v15 Cache System

A matplotlib-based visual inspection tool for comparing TSLA and SPY channels
side-by-side. Displays both assets' channel behavior at the same timestamp
to analyze correlation and lead/lag relationships.

Works with the v15 ChannelSample structure:
    - sample.tf_features: Flat dict of TF-prefixed features
    - sample.labels_per_window[window]['tsla'][tf]: TSLA labels
    - sample.labels_per_window[window]['spy'][tf]: SPY labels
    - 10 timeframes: 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly
    - 8 standard windows: 10, 20, 30, 40, 50, 60, 70, 80 bars

Features:
    - Side-by-side TSLA/SPY visualization (2 columns)
    - Multi-timeframe grid (configurable rows)
    - Window cycling ('w' key)
    - Sample navigation: LEFT/RIGHT arrows, 'r' for random
    - Timeframe cycling ('t' key)

Usage:
    python -m v15.dual_inspector --cache samples.pkl --data-dir data
    python -m v15.dual_inspector --cache samples.pkl --data-dir data --sample 10

Keyboard Controls:
    LEFT/RIGHT or A/D : Previous/next sample
    UP/DOWN or W/S    : Previous/next window
    T                 : Cycle timeframe views (mixed/intraday/multiday)
    Q                 : Quit
    H                 : Show help overlay
    R                 : Reset to first sample
    G                 : Show goto sample info (use --sample flag to jump)
    I                 : Print detailed sample info
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

from v15.types import ChannelSample, ChannelLabels, CrossCorrelationLabels, TIMEFRAMES, STANDARD_WINDOWS
from v15.labels import compute_cross_correlation_labels
from v15.inspector_utils import (
    get_labels_from_sample,
    get_cross_labels_from_sample,
    format_labels_text,
    format_cross_correlation_text,
    plot_candlesticks,
    plot_channel_bounds,
    plot_break_marker,
    DIR_COLORS,
    DIR_NAMES,
    BREAK_COLORS,
    BREAK_NAMES,
    BREAK_MARKER_COLOR,
)


# =============================================================================
# Constants
# =============================================================================

# Timeframe view sets for visualization
TF_VIEW_SETS = {
    'mixed': ['5min', '1h', 'daily'],
    'intraday': ['5min', '15min', '1h'],
    'multiday': ['daily', 'weekly', 'monthly'],
}
TF_VIEW_NAMES = ['mixed', 'intraday', 'multiday']

# Bars per timeframe for resampling
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

# Panel colors for validity display
VALID_BG_COLOR = '#e8f5e9'      # Light green for valid data
INVALID_BG_COLOR = '#ffebee'    # Light red for invalid data
NEUTRAL_BG_COLOR = '#f5f5f5'    # Light gray for no data
ALIGNED_BG_COLOR = '#e0f7fa'    # Light cyan for aligned
DIVERGENT_BG_COLOR = '#fff3e0'  # Light orange for divergent


# =============================================================================
# Utility Functions
# =============================================================================

def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample 5min OHLCV data to target timeframe.

    Args:
        df: DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
        timeframe: Target timeframe (e.g., '5min', '15min', '1h', 'daily', etc.)

    Returns:
        Resampled DataFrame with same columns
    """
    if timeframe == '5min':
        return df

    rule_map = {
        '15min': '15min', '30min': '30min', '1h': '1h',
        '2h': '2h', '3h': '3h', '4h': '4h',
        'daily': '1D', 'weekly': '1W', 'monthly': 'ME'
    }

    rule = rule_map.get(timeframe)
    if not rule:
        return df

    # For monthly resampling, try 'ME' first (pandas 2.2+), fall back to 'M' (legacy)
    if timeframe == 'monthly':
        try:
            return df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        except ValueError:
            # Fall back to legacy 'M' for pandas < 2.2
            rule = 'M'

    return df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


# =============================================================================
# Main Inspector Class
# =============================================================================

class DualAssetInspector:
    """
    Interactive visual inspector for comparing TSLA and SPY channels side-by-side.

    Keyboard Controls:
        LEFT/RIGHT or A/D : Previous/next sample
        UP/DOWN or W/S    : Previous/next window
        T                 : Cycle timeframe views
        Q                 : Quit
        H                 : Show help overlay
        R                 : Reset to first sample
        G                 : Show goto sample info (use --sample flag to jump)
    """

    def __init__(
        self,
        cache_path: str,
        tsla_df: Optional[pd.DataFrame] = None,
        spy_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initialize the dual-asset inspector.

        Args:
            cache_path: Path to the samples pickle file
            tsla_df: TSLA 5min DataFrame for visualization (optional)
            spy_df: SPY 5min DataFrame for visualization (optional)
        """
        self.cache_path = cache_path
        self.samples: List[ChannelSample] = []
        self.current_idx: int = 0
        self.current_window: int = STANDARD_WINDOWS[4]  # Default to 50
        self.current_tf: str = 'mixed'

        # Market data for visualization
        self.tsla_df = tsla_df
        self.spy_df = spy_df

        # Matplotlib figure and axes
        self.fig: Optional[plt.Figure] = None
        self.axes: Optional[np.ndarray] = None

        # UI state
        self.window_idx: int = 4  # Index into STANDARD_WINDOWS (default 50)
        self.tf_view_idx: int = 0  # Index into TF_VIEW_NAMES

        # Help overlay state
        self.help_overlay = None
        self.help_visible: bool = False

        # Status bar text handles (to prevent accumulation)
        self.status_bar_text = None
        self.status_bar_nav_hints = None

        # Load the cache
        self.load_cache(cache_path)

    def load_cache(self, path: str) -> None:
        """
        Load samples from a pickle cache file.

        Handles both old structure {window: {tf: labels}} and new dual-asset
        structure {window: {'tsla': {tf: labels}, 'spy': {tf: labels}}}.

        Args:
            path: Path to the pickle file
        """
        cache_path = Path(path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        print(f"Loading samples from {cache_path}...")

        with open(cache_path, 'rb') as f:
            data = pickle.load(f)

        # Handle different cache structures
        if isinstance(data, list):
            # List of ChannelSample objects
            self.samples = data
        elif isinstance(data, dict):
            # Could be a dict wrapper around samples
            if 'samples' in data:
                self.samples = data['samples']
            else:
                raise ValueError(
                    f"Unrecognized cache structure. "
                    f"Expected list of ChannelSample or dict with 'samples' key."
                )
        else:
            raise ValueError(
                f"Unrecognized cache type: {type(data)}. "
                f"Expected list or dict."
            )

        print(f"  Loaded {len(self.samples)} samples")

        # Detect cache format (old vs new dual-asset)
        if self.samples:
            sample = self.samples[0]
            if hasattr(sample, 'labels_per_window') and sample.labels_per_window:
                # Check structure of first window's labels
                first_window = next(iter(sample.labels_per_window.keys()))
                window_data = sample.labels_per_window[first_window]

                if isinstance(window_data, dict):
                    if 'tsla' in window_data or 'spy' in window_data:
                        print("  Detected: Dual-asset format (TSLA + SPY)")
                    else:
                        # Old format: {tf: labels}
                        print("  Detected: Single-asset format (TSLA only)")
                        print("  Note: SPY panels will show 'No Data'")

    def setup_figure(self) -> None:
        """
        Create the matplotlib figure with subplots and connect keyboard handler.

        Layout: 3 columns (TSLA, SPY, Cross-Correlation) x N rows (timeframes + info)
        The cross-correlation column spans all rows for a comprehensive summary.
        """
        from matplotlib.gridspec import GridSpec

        # Get current timeframe view
        current_view = TF_VIEW_NAMES[self.tf_view_idx]
        n_timeframes = len(TF_VIEW_SETS[current_view])

        # Layout: n_timeframes rows + 1 info row
        n_rows = n_timeframes + 1

        # Create figure with GridSpec for flexible layout
        # Columns: TSLA (3), SPY (3), Cross-Correlation (2)
        self.fig = plt.figure(figsize=(20, 4 * n_rows))
        gs = GridSpec(
            n_rows, 3, figure=self.fig,
            width_ratios=[3, 3, 2],
            height_ratios=[3] * n_timeframes + [1.5],
            hspace=0.3, wspace=0.25
        )

        # Create axes array for TSLA and SPY panels
        self.axes = np.empty((n_rows, 2), dtype=object)
        for row in range(n_rows):
            self.axes[row, 0] = self.fig.add_subplot(gs[row, 0])  # TSLA
            self.axes[row, 1] = self.fig.add_subplot(gs[row, 1])  # SPY

        # Create single cross-correlation panel spanning all rows
        self.ax_cross = self.fig.add_subplot(gs[:, 2])

        # Connect keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Reserve space for status bar at bottom
        self.fig.subplots_adjust(bottom=0.12)

        # Initial plot using display_sample
        self.display_sample()

        plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave room for status bar

    def on_key(self, event) -> None:
        """
        Keyboard handler for navigation.

        Keys:
            left/right or a/d: Previous/next sample
            up/down or w/s: Previous/next window
            t: Cycle timeframe views
            q: Quit
            h: Show help overlay
            r: Reset to first sample
            g: Show goto sample info (use --sample flag to jump)
        """
        if event.key in ('right', 'd'):
            self.next_sample()
        elif event.key in ('left', 'a'):
            self.prev_sample()
        elif event.key in ('down', 's'):
            self.next_window()
        elif event.key in ('up', 'w'):
            self.prev_window()
        elif event.key == 'q':
            plt.close('all')
        elif event.key == 'h':
            self.show_help()
        elif event.key == 'r':
            self.current_idx = 0
            self.display_sample()
        elif event.key == 'g':
            self._goto_sample()
        elif event.key == 'i':
            self._print_sample_info()
        elif event.key == 't':
            self._cycle_timeframe_view()
        elif event.key == 'b':
            self._jump_to_best_window()

    def next_sample(self) -> None:
        """Increment current_idx, wrap around, call display_sample()."""
        self.current_idx = (self.current_idx + 1) % len(self.samples)
        self.display_sample()

    def prev_sample(self) -> None:
        """Decrement current_idx, wrap around, call display_sample()."""
        self.current_idx = (self.current_idx - 1) % len(self.samples)
        self.display_sample()

    def next_window(self) -> None:
        """Cycle to next window in STANDARD_WINDOWS."""
        self.window_idx = (self.window_idx + 1) % len(STANDARD_WINDOWS)
        self.current_window = STANDARD_WINDOWS[self.window_idx]
        print(f"Window: {self.current_window}")
        self.display_sample()

    def prev_window(self) -> None:
        """Cycle to previous window in STANDARD_WINDOWS."""
        self.window_idx = (self.window_idx - 1) % len(STANDARD_WINDOWS)
        self.current_window = STANDARD_WINDOWS[self.window_idx]
        print(f"Window: {self.current_window}")
        self.display_sample()


    def show_help(self) -> None:
        """Display help text overlay showing all keyboard shortcuts."""
        if self.help_visible:
            # Hide help overlay
            if self.help_overlay is not None:
                self.help_overlay.remove()
                self.help_overlay = None
            self.help_visible = False
            self.fig.canvas.draw_idle()
            return

        # Show help overlay
        help_text = """
KEYBOARD SHORTCUTS
==================

NAVIGATION
  LEFT / A     Previous sample
  RIGHT / D    Next sample
  R            Reset to first sample
  G            Show goto sample info

WINDOW / TIMEFRAME
  UP / W       Previous window size
  DOWN / S     Next window size
  T            Cycle timeframe view (mixed/intraday/multiday)

OTHER
  B            Jump to best window
  H            Toggle this help
  I            Print sample info to console
  Q            Quit

Current windows: {}
Current timeframes: {}
""".format(
            ', '.join(str(w) for w in STANDARD_WINDOWS),
            ', '.join(TIMEFRAMES)
        )

        # Create semi-transparent overlay
        self.help_overlay = self.fig.text(
            0.5, 0.5, help_text,
            transform=self.fig.transFigure,
            fontsize=12,
            family='monospace',
            verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(
                boxstyle='round,pad=1',
                facecolor='white',
                edgecolor='black',
                alpha=0.95,
                linewidth=2
            )
        )
        self.help_visible = True
        self.fig.canvas.draw_idle()

    def _goto_sample(self) -> None:
        """Print instructions for going to a specific sample index."""
        # Cannot use input() from matplotlib event loop due to readline conflict
        # Instead, print current info and instructions
        print(f"\n{'='*60}")
        print(f"GO TO SAMPLE")
        print(f"{'='*60}")
        print(f"Current sample: {self.current_idx} (showing {self.current_idx + 1}/{len(self.samples)})")
        print(f"Valid range: 0-{len(self.samples)-1}")
        print(f"\nTo jump to a sample, close this window and restart with:")
        print(f"  --sample <index>")
        print(f"\nExample:")
        print(f"  python -m v15.dual_inspector --cache <path> --sample 42")
        print(f"{'='*60}\n")

    def _jump_to_best_window(self) -> None:
        """Jump to the best window for the current sample."""
        sample = self.samples[self.current_idx]
        best_window = getattr(sample, 'best_window', None)

        if best_window is not None and best_window in STANDARD_WINDOWS:
            self.window_idx = STANDARD_WINDOWS.index(best_window)
            self.current_window = best_window
            print(f"Jumped to best window: {best_window}")
            self.display_sample()
        else:
            print(f"No best window set for this sample (current: {self.current_window})")

    def _get_ranked_windows(self, sample) -> List[Tuple[int, float]]:
        """
        Get windows ranked by quality score for a sample.

        Returns list of (window, score) tuples sorted by score descending.
        Score is based on: valid labels count, r_squared, bounce_count.
        """
        window_scores = []

        for window in STANDARD_WINDOWS:
            score = 0.0
            valid_tfs = 0

            # Check how many TFs have valid labels for this window
            if hasattr(sample, 'labels_per_window') and sample.labels_per_window:
                window_data = sample.labels_per_window.get(window, {})

                if isinstance(window_data, dict):
                    # Dual-asset format
                    for asset in ['tsla', 'spy']:
                        asset_data = window_data.get(asset, {})
                        if isinstance(asset_data, dict):
                            for tf, labels in asset_data.items():
                                if labels is not None:
                                    # Score based on validity
                                    if getattr(labels, 'break_scan_valid', False):
                                        score += 2.0
                                        valid_tfs += 1
                                    if getattr(labels, 'direction_valid', False):
                                        score += 1.0
                                    if getattr(labels, 'duration_valid', False):
                                        score += 0.5

            # Check for channel features (r_squared, bounce_count equivalent)
            if hasattr(sample, 'tf_features') and sample.tf_features:
                # Try to get quality metrics from tf_features
                for tf in ['5min', '1h', 'daily']:
                    r_sq_key = f"{tf}_w{window}_r_squared"
                    if r_sq_key in sample.tf_features:
                        r_sq = sample.tf_features[r_sq_key]
                        if isinstance(r_sq, (int, float)) and r_sq > 0:
                            score += r_sq * 5.0  # Weight r_squared heavily

            # Bonus if this is the pre-computed best window
            if getattr(sample, 'best_window', None) == window:
                score += 10.0

            window_scores.append((window, score))

        # Sort by score descending
        window_scores.sort(key=lambda x: x[1], reverse=True)
        return window_scores

    def display_sample(self) -> None:
        """Update the plot with current sample and status bar."""
        # Hide help if visible
        if self.help_visible and self.help_overlay is not None:
            self.help_overlay.remove()
            self.help_overlay = None
            self.help_visible = False

        self._update_plot()
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        """Add status bar at bottom showing navigation info."""
        # Remove old status bar text objects to prevent accumulation
        if self.status_bar_text is not None:
            self.status_bar_text.remove()
            self.status_bar_text = None
        if self.status_bar_nav_hints is not None:
            self.status_bar_nav_hints.remove()
            self.status_bar_nav_hints = None

        sample = self.samples[self.current_idx]
        current_window = STANDARD_WINDOWS[self.window_idx]
        current_view = TF_VIEW_NAMES[self.tf_view_idx]

        # Build status text
        status_parts = [
            f"Sample: {self.current_idx + 1}/{len(self.samples)}",
            f"Window: {current_window} ({self.window_idx + 1}/{len(STANDARD_WINDOWS)})",
            f"TF View: {current_view} ({self.tf_view_idx + 1}/{len(TF_VIEW_NAMES)})",
        ]

        # Add best window info and ranked windows
        best_window = getattr(sample, 'best_window', None)
        if best_window is not None:
            if current_window == best_window:
                status_parts.append("*BEST*")
            else:
                status_parts.append(f"(best: {best_window})")

        # Add top 3 ranked windows
        ranked = self._get_ranked_windows(sample)
        if ranked:
            top_3 = [str(w) + ('*' if w == best_window else '') for w, _ in ranked[:3]]
            status_parts.append(f"Top: {', '.join(top_3)}")

        status_text = "  |  ".join(status_parts)

        # Navigation hints
        nav_hints = "[A/D] Sample  [W/S] Window  [B] Best  [T] TF View  [H] Help  [Q] Quit"

        # Add status bar text and store handles
        self.status_bar_text = self.fig.text(
            0.5, 0.02, status_text,
            transform=self.fig.transFigure,
            fontsize=10,
            family='monospace',
            verticalalignment='bottom',
            horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8)
        )

        self.status_bar_nav_hints = self.fig.text(
            0.5, 0.05, nav_hints,
            transform=self.fig.transFigure,
            fontsize=9,
            family='monospace',
            verticalalignment='bottom',
            horizontalalignment='center',
            color='#666666'
        )

    def _cycle_timeframe_view(self) -> None:
        """Cycle through timeframe views (mixed/intraday/multiday)."""
        self.tf_view_idx = (self.tf_view_idx + 1) % len(TF_VIEW_NAMES)
        current_view = TF_VIEW_NAMES[self.tf_view_idx]
        print(f"Timeframe view: {current_view} - {TF_VIEW_SETS[current_view]}")

        # Recreate figure with new layout
        plt.close(self.fig)
        self.setup_figure()

    def _update_plot(self) -> None:
        """
        Update the plot with current sample.

        This is the main rendering method that:
        1. Clears all axes
        2. Gets current sample, window, and timeframe view
        3. Extracts TSLA labels using get_labels_from_sample()
        4. Extracts SPY labels using get_labels_from_sample()
        5. Calls plot methods for both assets
        6. Displays cross-correlation metrics
        7. Updates figure title with sample info
        8. Calls fig.canvas.draw_idle() to refresh display
        """
        if self.fig is None:
            return

        # Clear all axes
        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()

        # Clear cross-correlation panel if it exists
        if hasattr(self, 'ax_cross') and self.ax_cross is not None:
            self.ax_cross.clear()

        sample = self.samples[self.current_idx]

        # Determine display window
        if self.window_idx is not None:
            display_window = STANDARD_WINDOWS[self.window_idx]
        else:
            display_window = getattr(sample, 'best_window', 50)

        # Get current timeframe view
        current_view = TF_VIEW_NAMES[self.tf_view_idx]
        display_tfs = TF_VIEW_SETS[current_view]

        # Plot each timeframe row
        for row_idx, tf in enumerate(display_tfs):
            # Extract labels for both assets using helper function
            tsla_labels = get_labels_from_sample(sample, display_window, tf, asset='tsla')
            spy_labels = get_labels_from_sample(sample, display_window, tf, asset='spy')

            # TSLA panel (left column)
            self.plot_asset_panel(
                self.axes[row_idx, 0], 'tsla', tsla_labels,
                tf=tf, window=display_window, sample=sample
            )
            self.axes[row_idx, 0].set_title(f"TSLA - {tf}", fontsize=11, fontweight='bold')

            # SPY panel (right column)
            self.plot_asset_panel(
                self.axes[row_idx, 1], 'spy', spy_labels,
                tf=tf, window=display_window, sample=sample
            )
            self.axes[row_idx, 1].set_title(f"SPY - {tf}", fontsize=11, fontweight='bold')

        # Info panels (bottom row)
        first_tf = display_tfs[0]
        tsla_labels_info = get_labels_from_sample(sample, display_window, first_tf, asset='tsla')
        spy_labels_info = get_labels_from_sample(sample, display_window, first_tf, asset='spy')

        self.plot_info_panel(self.axes[-1, 0], tsla_labels_info, 'tsla')
        self.plot_info_panel(self.axes[-1, 1], spy_labels_info, 'spy')

        # Cross-correlation panel (right column, spans all rows)
        if hasattr(self, 'ax_cross') and self.ax_cross is not None:
            self.plot_cross_correlation_panel(self.ax_cross)

        # Update figure title with sample info
        best_window = getattr(sample, 'best_window', display_window)
        window_marker = '*' if display_window == best_window else ''
        title = (
            f"Sample {self.current_idx + 1}/{len(self.samples)} | "
            f"{sample.timestamp} | Window: {display_window}{window_marker}"
        )
        self.fig.suptitle(title, fontsize=14, fontweight='bold')

        self.fig.canvas.draw_idle()

    def plot_asset_panel(
        self,
        ax: plt.Axes,
        asset: str,
        labels: Optional[ChannelLabels],
        tf: str = '5min',
        window: int = 50,
        sample: Optional[ChannelSample] = None
    ) -> None:
        """
        Plot channel visualization for one asset with actual OHLC candlesticks.

        Shows:
        1. OHLC candlestick chart with actual price data
        2. Channel upper bound line (solid)
        3. Channel lower bound line (solid)
        4. Channel center line (dashed)
        5. Break point marker if applicable

        Args:
            ax: Matplotlib axes to plot on
            asset: Asset identifier ('tsla' or 'spy')
            labels: ChannelLabels for this asset/window/tf, or None if not available
            tf: Timeframe for this panel
            window: Window size for channel
            sample: ChannelSample object with timestamp and metadata
        """
        print(f"\nDEBUG [{asset} {tf} w={window}]: ===== plot_asset_panel ENTRY =====")
        print(f"DEBUG [{asset} {tf} w={window}]: labels={labels}, sample={'present' if sample else 'None'}")

        # Set background color based on validity
        if labels is None:
            bg_color = NEUTRAL_BG_COLOR
        elif labels.break_scan_valid or labels.direction_valid:
            bg_color = VALID_BG_COLOR
        else:
            bg_color = INVALID_BG_COLOR

        ax.set_facecolor(bg_color)

        # Handle no labels case
        if labels is None:
            print(f"DEBUG [{asset} {tf} w={window}]: EARLY RETURN - labels is None")
            ax.text(
                0.5, 0.5,
                f"{asset.upper()}\nNo labels available",
                ha='center', va='center',
                fontsize=12, color='gray',
                transform=ax.transAxes
            )
            ax.grid(True, alpha=0.3)
            return

        # Get the appropriate DataFrame
        asset_df = self.tsla_df if asset == 'tsla' else self.spy_df

        # If no data available, fall back to arrow display
        if asset_df is None or sample is None:
            # DEBUG: Print why taking fallback path
            if asset_df is None:
                print(f"DEBUG [{asset} {tf} w={window}]: Taking fallback - asset_df is None")
            if sample is None:
                print(f"DEBUG [{asset} {tf} w={window}]: Taking fallback - sample is None")
            self._plot_fallback_panel(ax, asset, labels)
            return

        # Get the timestamp for this sample
        timestamp = sample.timestamp

        # Find the index in the DataFrame
        try:
            # Find the position of the timestamp in the DataFrame
            if timestamp not in asset_df.index:
                # Find nearest timestamp
                idx_loc = asset_df.index.get_indexer([timestamp], method='nearest')[0]
            else:
                idx_loc = asset_df.index.get_loc(timestamp)

            # Resample data to the target timeframe
            df_resampled = resample_ohlc(asset_df, tf)

            # Find the corresponding position in resampled data
            if timestamp not in df_resampled.index:
                resample_idx = df_resampled.index.get_indexer([timestamp], method='nearest')[0]
            else:
                resample_idx = df_resampled.index.get_loc(timestamp)

            # Calculate forward_bars based on break or default
            if labels.break_scan_valid and labels.bars_to_first_break > 0:
                forward_bars = labels.bars_to_first_break + 5
            else:
                forward_bars = 20

            # Get the lookback slice (window bars before the timestamp) plus forward bars
            start_idx = max(0, resample_idx - window + 1)
            end_idx = min(len(df_resampled), resample_idx + forward_bars + 1)
            df_plot = df_resampled.iloc[start_idx:end_idx]

            if len(df_plot) < 2:
                # DEBUG: Print why taking fallback path
                print(f"DEBUG [{asset} {tf} w={window}]: Taking fallback - insufficient data (len={len(df_plot)})")
                self._plot_fallback_panel(ax, asset, labels)
                return

            # Plot candlesticks using inspector_utils
            print(f"DEBUG [{asset} {tf} w={window}]: BEFORE plot_candlesticks - df_plot shape={df_plot.shape}")
            plot_candlesticks(ax, df_plot, 0, len(df_plot))
            print(f"DEBUG [{asset} {tf} w={window}]: AFTER plot_candlesticks - SUCCESS")

            # Add vertical separator line at window boundary (marks end of channel window, start of projection)
            ax.axvline(window - 0.5, color='blue', linestyle='-', linewidth=2, alpha=0.3, zorder=10)

            # Reconstruct channel bounds from features if available
            # The channel should be for the last 'window' bars
            channel = self._reconstruct_channel_from_features(sample, tf, window, asset)

            # DEBUG: Print channel reconstruction result
            if channel is None:
                print(f"DEBUG [{asset} {tf} w={window}]: Channel is None - no features available")
            else:
                print(f"DEBUG [{asset} {tf} w={window}]: Channel valid - slope={channel.slope:.6f}, std_dev={channel.std_dev:.6f}")

            if channel is not None:
                # DEBUG: Print before calling plot_channel_bounds
                print(f"DEBUG [{asset} {tf} w={window}]: Calling plot_channel_bounds with channel (upper={channel.upper_line[0]:.2f}, lower={channel.lower_line[0]:.2f})")

                # Calculate forward projection distance
                # If there's a break, project to show it clearly (add 5 bars buffer)
                # Otherwise use a sensible default (20 bars)
                if labels.break_scan_valid and labels.bars_to_first_break > 0:
                    project_forward = labels.bars_to_first_break + 5
                else:
                    project_forward = 20

                print(f"DEBUG [{asset} {tf} w={window}]: project_forward={project_forward}")

                # Plot channel bounds with forward projection
                plot_channel_bounds(ax, channel, 0, window, color='blue', project_forward=project_forward)
                print(f"DEBUG [{asset} {tf} w={window}]: AFTER plot_channel_bounds - SUCCESS")

            # Plot break marker if break scan found a break AND channel is valid
            # (only show break marker if we have a valid channel - otherwise it's meaningless)
            # Note: bars_to_first_break is 0-indexed (0 = first bar after channel)
            if labels.break_scan_valid and labels.bars_to_first_break >= 0 and channel is not None:
                # Break happens AFTER the channel window at bars_to_first_break position
                # bars_to_first_break is 0-indexed: 0 = first bar after channel (visual index = window)
                # Visual position = window + bars_to_first_break
                break_bar = window + labels.bars_to_first_break
                break_dir = labels.break_direction
                if hasattr(break_dir, 'value'):
                    break_dir = break_dir.value

                # Get price range for arrow sizing
                price_range = df_plot['high'].max() - df_plot['low'].min()

                # Estimate price at break using channel projection
                if channel is not None:
                    # Use channel center line projected to break point
                    # Channel regression: center = slope * x + intercept
                    # At channel end: x = window - 1
                    # At first bar after (bars_to_first_break=0): x = window
                    # At bars_to_first_break bar: x = window + bars_to_first_break
                    future_x = window + labels.bars_to_first_break
                    price_y = channel.slope * future_x + channel.intercept
                else:
                    price_y = df_plot.iloc[-1]['close']

                # DEBUG: Print before calling plot_break_marker
                print(f"DEBUG [{asset} {tf} w={window}]: Calling plot_break_marker - break_bar={break_bar}, break_dir={break_dir}, price_y={price_y:.2f}")

                plot_break_marker(
                    ax, break_bar, break_dir,
                    color=BREAK_MARKER_COLOR,
                    price_y=price_y,
                    price_range=price_range
                )
                print(f"DEBUG [{asset} {tf} w={window}]: AFTER plot_break_marker - SUCCESS")

            # Add metrics overlay
            print(f"DEBUG [{asset} {tf} w={window}]: BEFORE _add_metrics_overlay")
            self._add_metrics_overlay(ax, labels, asset)
            print(f"DEBUG [{asset} {tf} w={window}]: AFTER _add_metrics_overlay - SUCCESS")

            # Update x-axis limits to show the full projection area
            # Calculate the maximum x position (window + projection)
            if labels.break_scan_valid and labels.bars_to_first_break >= 0:
                max_x = window + labels.bars_to_first_break + 5
            else:
                max_x = window + 20

            print(f"DEBUG [{asset} {tf} w={window}]: Setting x-axis limits to (-0.5, {max_x + 0.5})")
            ax.set_xlim(-0.5, max_x + 0.5)

            # Force Y-axis autoscale to ensure all plotted data is visible
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            print(f"DEBUG [{asset} {tf} w={window}]: plot_asset_panel COMPLETED SUCCESSFULLY")

        except Exception as e:
            # If any error occurs, fall back to arrow display
            import traceback
            print(f"ERROR [{asset} {tf} w={window}]: Exception in plot_asset_panel:")
            print(f"  Exception type: {type(e).__name__}")
            print(f"  Exception message: {e}")
            print(f"  Full traceback:")
            traceback.print_exc()
            self._plot_fallback_panel(ax, asset, labels)
            return

        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Bars')
        ax.set_ylabel('Price')

    def _plot_fallback_panel(self, ax: plt.Axes, asset: str, labels: ChannelLabels) -> None:
        """
        Fallback visualization using arrows when candlestick data is unavailable.

        Args:
            ax: Matplotlib axes to plot on
            asset: Asset identifier
            labels: ChannelLabels
        """
        # Build metrics text annotation
        lines = []

        # Duration and break status
        if labels.permanent_break:
            lines.append(f"Break: {labels.duration_bars} bars")
        else:
            lines.append(f"Duration: {labels.duration_bars} bars")
            lines.append("No permanent break")

        # Break direction (handle enum)
        break_dir = labels.break_direction
        if hasattr(break_dir, 'value'):
            break_dir = break_dir.value
        dir_str = BREAK_NAMES.get(break_dir, '?')
        dir_color = BREAK_COLORS.get(break_dir, 'gray')

        if labels.permanent_break:
            lines.append(f"Direction: {dir_str}")

        # Break scan metrics if available
        if labels.break_scan_valid:
            lines.append(f"1st Break: {labels.bars_to_first_break} bars")
            lines.append(f"Magnitude: {labels.break_magnitude:.2f} std")
            if labels.returned_to_channel:
                lines.append(f"Returned: Yes ({labels.bounces_after_return} bounces)")
            else:
                lines.append("Returned: No")

        # New channel direction
        new_dir = labels.next_channel_direction
        if hasattr(new_dir, 'value'):
            new_dir = new_dir.value
        new_dir_str = DIR_NAMES.get(new_dir, '?')
        lines.append(f"Next: {new_dir_str}")

        # Display metrics text
        metrics_text = '\n'.join(lines)
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        ax.text(
            0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=props, family='monospace'
        )

        # Draw break direction indicator in center
        if labels.permanent_break:
            arrow_symbol = '$\\uparrow$' if break_dir == 1 else '$\\downarrow$'
            ax.text(
                0.5, 0.5, arrow_symbol,
                ha='center', va='center',
                fontsize=48, color=dir_color, alpha=0.3,
                transform=ax.transAxes
            )

        # Add validity indicators
        validity_parts = []
        if labels.duration_valid:
            validity_parts.append('dur')
        if labels.direction_valid:
            validity_parts.append('dir')
        if labels.break_scan_valid:
            validity_parts.append('scan')
        validity_text = f"[{', '.join(validity_parts)}]" if validity_parts else "[none]"

        ax.text(
            0.98, 0.02, validity_text,
            ha='right', va='bottom',
            fontsize=8, color='gray',
            transform=ax.transAxes
        )

        ax.grid(True, alpha=0.3)

    def _add_metrics_overlay(self, ax: plt.Axes, labels: ChannelLabels, asset: str) -> None:
        """
        Add text overlay with key metrics on the chart.

        Args:
            ax: Matplotlib axes
            labels: ChannelLabels
            asset: Asset identifier
        """
        lines = []

        # Duration and break status
        if labels.permanent_break:
            lines.append(f"Break: {labels.duration_bars} bars")
        else:
            lines.append(f"Duration: {labels.duration_bars} bars")

        # Break scan metrics if available
        if labels.break_scan_valid and labels.permanent_break:
            break_dir = labels.break_direction
            if hasattr(break_dir, 'value'):
                break_dir = break_dir.value
            dir_str = BREAK_NAMES.get(break_dir, '?')
            lines.append(f"Dir: {dir_str}")
            lines.append(f"Mag: {labels.break_magnitude:.2f}σ")

        # Display metrics text
        if lines:
            metrics_text = '\n'.join(lines)
            props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
            ax.text(
                0.02, 0.98, metrics_text,
                transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=props, family='monospace'
            )

        # Add validity indicators
        validity_parts = []
        if labels.duration_valid:
            validity_parts.append('dur')
        if labels.direction_valid:
            validity_parts.append('dir')
        if labels.break_scan_valid:
            validity_parts.append('scan')
        validity_text = f"[{', '.join(validity_parts)}]" if validity_parts else "[none]"

        ax.text(
            0.98, 0.02, validity_text,
            ha='right', va='bottom',
            fontsize=7, color='gray',
            transform=ax.transAxes
        )

    def _reconstruct_channel_from_features(
        self,
        sample: ChannelSample,
        tf: str,
        window: int,
        asset: str
    ) -> Optional[Any]:
        """
        Reconstruct channel object from stored features.

        Uses the correct feature name patterns:
        - TSLA: {tf}_w{window}_channel_slope (no asset prefix)
        - SPY: {tf}_w{window}_spy_channel_slope (spy_ prefix in feature name)

        Args:
            sample: ChannelSample with tf_features
            tf: Timeframe
            window: Window size
            asset: Asset identifier ('tsla' or 'spy')

        Returns:
            Channel-like object with upper_line, lower_line, center_line, or None
        """
        # Build feature name prefix based on asset
        # TSLA: "{tf}_w{window}_channel_slope"
        # SPY: "{tf}_w{window}_spy_channel_slope"
        if asset == 'tsla':
            prefix = f"{tf}_w{window}_"
            slope_key = f"{prefix}channel_slope"
            intercept_key = f"{prefix}channel_intercept"
            std_dev_ratio_key = f"{prefix}std_dev_ratio"
        else:  # spy
            prefix = f"{tf}_w{window}_"
            slope_key = f"{prefix}spy_channel_slope"
            intercept_key = f"{prefix}spy_channel_intercept"
            std_dev_ratio_key = f"{prefix}spy_std_dev_ratio"

        # Check if required features exist
        if slope_key not in sample.tf_features:
            return None
        if intercept_key not in sample.tf_features:
            return None
        if std_dev_ratio_key not in sample.tf_features:
            return None

        slope = sample.tf_features[slope_key]
        intercept = sample.tf_features[intercept_key]
        std_dev_ratio = sample.tf_features[std_dev_ratio_key]

        # Validate that these are actually numbers, not functions or other objects
        if not isinstance(slope, (int, float, np.number)):
            print(f"Warning: slope is not a number for {asset} {tf} w{window}: {type(slope)}")
            return None
        if not isinstance(intercept, (int, float, np.number)):
            print(f"Warning: intercept is not a number for {asset} {tf} w{window}: {type(intercept)}")
            return None
        if not isinstance(std_dev_ratio, (int, float, np.number)):
            print(f"Warning: std_dev_ratio is not a number for {asset} {tf} w{window}: {type(std_dev_ratio)}")
            return None

        # Skip if any values are zero (indicates failed channel detection)
        if slope == 0.0 and intercept == 0.0 and std_dev_ratio == 0.0:
            return None

        # Get current price to compute actual std_dev from std_dev_ratio
        # std_dev = std_dev_ratio * avg_price
        # We'll use the intercept as a proxy for avg_price since it's the center line value
        # at position 0 (roughly the average price level)
        # Better: reconstruct from the asset DataFrame
        asset_df = self.tsla_df if asset == 'tsla' else self.spy_df
        if asset_df is None or sample is None:
            return None

        try:
            # Get timestamp and find position in DataFrame
            timestamp = sample.timestamp
            if timestamp not in asset_df.index:
                idx_loc = asset_df.index.get_indexer([timestamp], method='nearest')[0]
            else:
                idx_loc = asset_df.index.get_loc(timestamp)

            # Resample to target timeframe (use local function)
            df_resampled = resample_ohlc(asset_df, tf)

            # Validate resampling worked
            if df_resampled is None or not isinstance(df_resampled, pd.DataFrame):
                print(f"Warning: Resampling failed for {asset} {tf}")
                return None

            # Find position in resampled data
            if timestamp not in df_resampled.index:
                resample_idx = df_resampled.index.get_indexer([timestamp], method='nearest')[0]
            else:
                resample_idx = df_resampled.index.get_loc(timestamp)

            # Get the lookback slice
            start_idx = max(0, resample_idx - window + 1)
            end_idx = resample_idx + 1
            df_slice = df_resampled.iloc[start_idx:end_idx]

            if len(df_slice) < 2:
                return None

            # Calculate average price from the actual data
            avg_price = df_slice['close'].mean()

            # Compute actual std_dev
            std_dev = std_dev_ratio * avg_price

        except Exception as e:
            print(f"Warning: Could not compute std_dev for {asset} {tf}: {e}")
            return None

        # Reconstruct channel lines
        x = np.arange(window)
        center_line = slope * x + intercept
        upper_line = center_line + 2.0 * std_dev
        lower_line = center_line - 2.0 * std_dev

        # DEBUG: Print detailed channel reconstruction info
        print(f"\nCHANNEL RECONSTRUCTION [{asset.upper()} {tf} w={window}]:")
        print(f"  Features:")
        print(f"    slope = {slope:.6f}")
        print(f"    intercept = {intercept:.6f}")
        print(f"    std_dev_ratio = {std_dev_ratio:.6f}")
        print(f"  Derived:")
        print(f"    avg_price = {avg_price:.2f}")
        print(f"    std_dev = {std_dev:.6f}")
        print(f"  Channel lines (first 3 bars):")
        for i in range(min(3, len(x))):
            print(f"    Bar {i}: center={center_line[i]:.2f}, upper={upper_line[i]:.2f}, lower={lower_line[i]:.2f}")
        print(f"  Channel lines (last 3 bars):")
        for i in range(max(0, len(x) - 3), len(x)):
            print(f"    Bar {i}: center={center_line[i]:.2f}, upper={upper_line[i]:.2f}, lower={lower_line[i]:.2f}")

        # Create a simple object to hold the channel data
        class SimpleChannel:
            def __init__(self, upper, lower, center, slope, intercept, std_dev, window):
                self.upper_line = upper
                self.lower_line = lower
                self.center_line = center
                self.slope = slope
                self.intercept = intercept
                self.std_dev = std_dev
                self.window = window
                self.valid = True

        return SimpleChannel(upper_line, lower_line, center_line, slope, intercept, std_dev, window)

    def plot_info_panel(self, ax: plt.Axes, labels: Optional[ChannelLabels], asset: str) -> None:
        """
        Display formatted label text in the info panel.

        Shows:
        - Validity flags with color coding
        - Key label values
        - Color-code based on break direction

        Args:
            ax: Matplotlib axes to plot on
            labels: ChannelLabels or None
            asset: Asset name ('tsla' or 'spy')
        """
        ax.axis('off')

        # Format label text using helper function from inspector_utils
        info_text = format_labels_text(labels, asset)

        # Determine background color based on labels
        if labels is None:
            bg_color = NEUTRAL_BG_COLOR
        elif labels.break_scan_valid or labels.direction_valid:
            # Color based on break direction
            break_dir = labels.break_direction
            if hasattr(break_dir, 'value'):
                break_dir = break_dir.value
            if labels.permanent_break:
                # Green for up break, red for down break
                bg_color = '#e8f5e9' if break_dir == 1 else '#ffebee'
            else:
                bg_color = NEUTRAL_BG_COLOR
        else:
            bg_color = INVALID_BG_COLOR

        ax.set_facecolor(bg_color)

        # Display the formatted text
        ax.text(
            0.5, 0.5, info_text,
            ha='center', va='center',
            fontsize=9, family='monospace',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    def plot_cross_correlation_panel(self, ax: plt.Axes) -> None:
        """
        Display cross-correlation metrics between TSLA and SPY.

        Shows:
        - Alignment metrics (direction aligned, return pattern aligned)
        - Who broke first (TSLA vs SPY leader)
        - Lag in bars between breaks
        - Magnitude spread
        - Color-codes based on alignment (aligned = cyan, divergent = orange)

        Args:
            ax: Matplotlib axes to plot on (spans all rows in the figure)
        """
        ax.axis('off')

        # Get current sample and window
        sample = self.samples[self.current_idx]
        if self.window_idx is not None:
            window = STANDARD_WINDOWS[self.window_idx]
        else:
            window = getattr(sample, 'best_window', 50)

        # Get current timeframe view
        current_view = TF_VIEW_NAMES[self.tf_view_idx]
        display_tfs = TF_VIEW_SETS[current_view]

        # Build cross-correlation summary for all displayed TFs
        all_text = ["CROSS-CORRELATION", "=" * 25, ""]

        any_aligned = False
        any_divergent = False

        for tf in display_tfs:
            # Get or compute cross-correlation labels
            cross_labels = get_cross_labels_from_sample(sample, window, tf)

            # If not pre-computed, try to compute from TSLA/SPY labels
            if cross_labels is None:
                tsla_labels = get_labels_from_sample(sample, window, tf, asset='tsla')
                spy_labels = get_labels_from_sample(sample, window, tf, asset='spy')

                if tsla_labels is not None and spy_labels is not None:
                    cross_labels = compute_cross_correlation_labels(
                        tsla_labels, spy_labels, tf
                    )
                else:
                    cross_labels = CrossCorrelationLabels(cross_valid=False)

            # Format cross-correlation for this TF
            all_text.append(f"--- {tf} ---")

            if not cross_labels.cross_valid:
                all_text.append("  Invalid (missing data)")
                all_text.append("")
                continue

            # Track alignment
            if cross_labels.break_direction_aligned:
                any_aligned = True
            else:
                any_divergent = True

            # Direction alignment
            dir_status = "ALIGNED" if cross_labels.break_direction_aligned else "DIVERGENT"
            all_text.append(f"  Direction: {dir_status}")

            # Who broke first
            if cross_labels.tsla_broke_first:
                all_text.append(f"  Leader: TSLA (+{cross_labels.break_lag_bars} bars)")
            elif cross_labels.spy_broke_first:
                all_text.append(f"  Leader: SPY (+{cross_labels.break_lag_bars} bars)")
            else:
                all_text.append("  Leader: Simultaneous")

            # Magnitude spread
            mag_spread = cross_labels.magnitude_spread
            if abs(mag_spread) > 0.5:
                stronger = "TSLA" if mag_spread > 0 else "SPY"
                all_text.append(f"  Stronger: {stronger} ({mag_spread:+.2f})")
            else:
                all_text.append(f"  Magnitude: Similar ({mag_spread:+.2f})")

            # Return pattern
            if cross_labels.both_returned:
                all_text.append("  Pattern: Both returned")
            elif cross_labels.both_permanent:
                all_text.append("  Pattern: Both permanent")
            else:
                all_text.append("  Pattern: Mixed")

            all_text.append("")

        # Determine background color based on alignment
        if any_aligned and not any_divergent:
            bg_color = ALIGNED_BG_COLOR
        elif any_divergent:
            bg_color = DIVERGENT_BG_COLOR
        else:
            bg_color = NEUTRAL_BG_COLOR

        ax.set_facecolor(bg_color)

        # Display all text
        full_text = '\n'.join(all_text)
        ax.text(
            0.5, 0.98, full_text,
            ha='center', va='top',
            fontsize=9, family='monospace',
            transform=ax.transAxes
        )

        # Add legend at bottom
        ax.text(
            0.5, 0.02,
            "Cyan=Aligned | Orange=Divergent",
            ha='center', va='bottom',
            fontsize=8, color='gray',
            transform=ax.transAxes
        )

    def _print_sample_info(self) -> None:
        """Print detailed information about current sample to console."""
        sample = self.samples[self.current_idx]

        print("\n" + "=" * 70)
        print(f"SAMPLE {self.current_idx} - DUAL ASSET INFO")
        print("=" * 70)

        print(f"\nTimestamp: {sample.timestamp}")
        print(f"Channel End Index: {sample.channel_end_idx}")
        print(f"Best Window: {sample.best_window}")

        # Labels summary
        if hasattr(sample, 'labels_per_window') and sample.labels_per_window:
            print(f"\nLabels per window:")
            for window in STANDARD_WINDOWS:
                if window in sample.labels_per_window:
                    window_data = sample.labels_per_window[window]
                    if isinstance(window_data, dict):
                        if 'tsla' in window_data:
                            tsla_tfs = len(window_data.get('tsla', {}))
                            spy_tfs = len(window_data.get('spy', {}))
                            print(f"  Window {window}: TSLA={tsla_tfs} TFs, SPY={spy_tfs} TFs")
                        else:
                            # Old format
                            print(f"  Window {window}: {len(window_data)} TFs (single-asset)")

        print("=" * 70 + "\n")

    def run(self) -> None:
        """Run the interactive inspector."""
        if not self.samples:
            print("Error: No samples loaded")
            return

        print("\n" + "=" * 60)
        print("DUAL ASSET INSPECTOR - Keyboard Controls")
        print("=" * 60)
        print("  LEFT/RIGHT or A/D : Previous/next sample")
        print("  UP/DOWN or W/S    : Previous/next window")
        print("  B                 : Jump to best window")
        print("  T                 : Cycle timeframe views")
        print("  Q                 : Quit")
        print("  H                 : Show help overlay")
        print("  R                 : Reset to first sample")
        print("  G                 : Show goto sample info")
        print("  I                 : Print sample info to console")
        print("=" * 60 + "\n")

        self.setup_figure()
        plt.show()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """Command-line entry point for the dual-asset inspector."""
    parser = argparse.ArgumentParser(
        description='Dual-Asset Visual Inspector for v15 cache samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m v15.dual_inspector --cache samples.pkl --data-dir data
    python -m v15.dual_inspector --cache samples.pkl --data-dir data --sample 10
    python -m v15.dual_inspector -c path/to/cache.pkl -d data -s 0
        """
    )
    parser.add_argument(
        '--cache', '-c',
        type=str,
        required=True,
        help='Path to samples pickle file (.pkl)'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default=None,
        help='Path to data directory with TSLA/SPY CSV files (optional, for candlestick plotting)'
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=0,
        help='Starting sample index (default: 0)'
    )

    args = parser.parse_args()

    # Load market data if data_dir is provided
    tsla_df = None
    spy_df = None
    if args.data_dir:
        try:
            from v15.data import load_market_data
            print(f"Loading market data from {args.data_dir}...")
            tsla_df, spy_df, _ = load_market_data(args.data_dir)
            print(f"  TSLA: {len(tsla_df)} bars")
            print(f"  SPY: {len(spy_df)} bars")
        except Exception as e:
            print(f"Warning: Could not load market data: {e}")
            print("  Continuing without candlestick plotting...")

    # Create and run inspector
    try:
        inspector = DualAssetInspector(args.cache, tsla_df=tsla_df, spy_df=spy_df)
        inspector.current_idx = args.sample
        inspector.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
