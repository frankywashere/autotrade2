"""
Unified Channel Visualizer for V15

Combines the best features from:
- test_1h_w20.py: Hollow/filled triangle markers for first/permanent breaks
- dual_inspector.py: Side-by-side TSLA/SPY comparison, cross-correlation display

Usage:
    python -m v15.channel_visualizer samples.pkl --data-dir data
    python -m v15.channel_visualizer --samples cache.pkl --start 10
    python -m v15.channel_visualizer samples.pkl --single  # Single asset mode
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from v15.dtypes import (
    ChannelSample, ChannelLabels, CrossCorrelationLabels,
    TIMEFRAMES, STANDARD_WINDOWS, TF_MAX_SCAN,
    BREAK_UP, BREAK_DOWN, DIRECTION_BULL, DIRECTION_BEAR, DIRECTION_SIDEWAYS
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Direction colors
DIR_COLORS = {
    DIRECTION_BEAR: '#ff4444',      # Red
    DIRECTION_SIDEWAYS: '#ffaa00',  # Orange
    DIRECTION_BULL: '#44ff44',      # Green
}

# Break colors
BREAK_COLORS = {
    BREAK_DOWN: '#ff4444',  # Red
    BREAK_UP: '#44ff44',    # Green
    0: '#ff4444',           # DOWN = 0
    1: '#44ff44',           # UP = 1
}

# Background colors for validity
VALID_BG = '#e8f5e9'      # Light green
INVALID_BG = '#ffebee'    # Light red
NEUTRAL_BG = '#f5f5f5'    # Light gray
ALIGNED_BG = '#e0f7fa'    # Light cyan
DIVERGENT_BG = '#fff3e0'  # Light orange

# Timeframe view sets
TF_VIEWS = {
    'mixed': ['5min', '1h', 'daily'],
    'intraday': ['5min', '15min', '1h'],
    'multiday': ['daily', 'weekly', 'monthly'],
}

# Keyboard help
KEYBOARD_HELP = """
Navigation:
  LEFT/RIGHT  : Previous/Next sample
  UP/DOWN     : Jump 10 samples
  r           : Random sample

Display:
  w           : Cycle window (best → 10 → 20 → ... → 80 → best)
  t           : Cycle timeframe view (mixed/intraday/multiday)
  m           : Toggle single/dual asset mode
  i           : Print sample info to console
  h           : Toggle this help

Exit:
  q / ESC     : Quit
"""


# =============================================================================
# CANDLESTICK PLOTTING (from test_1h_w20.py)
# =============================================================================

def plot_candlesticks(ax, df: pd.DataFrame, start_idx: int = 0) -> None:
    """
    Plot OHLC candlesticks with black wicks (test_1h_w20.py style).
    """
    for i, (idx, row) in enumerate(df.iterrows()):
        x = start_idx + i
        open_price = row['open']
        high = row['high']
        low = row['low']
        close = row['close']

        # Determine color
        if close >= open_price:
            color = 'green'
            body_bottom = open_price
            body_height = close - open_price
        else:
            color = 'red'
            body_bottom = close
            body_height = open_price - close

        # Draw wick (high-low line) - BLACK like test_1h_w20.py
        ax.plot([x, x], [low, high], color='black', linewidth=0.5)

        # Draw body
        if body_height > 0:
            ax.add_patch(Rectangle(
                (x - 0.3, body_bottom), 0.6, body_height,
                facecolor=color, edgecolor='black', linewidth=0.5
            ))
        else:
            # Doji - just a horizontal line
            ax.plot([x - 0.3, x + 0.3], [close, close], color='black', linewidth=1)


# =============================================================================
# CHANNEL BOUNDS PLOTTING (from test_1h_w20.py)
# =============================================================================

def plot_channel_bounds(
    ax,
    slope: float,
    intercept: float,
    std_dev: float,
    channel_len: int,
    total_len: int
) -> None:
    """
    Plot channel bounds with solid lines in window, dashed projection.
    Uses green upper / red lower / blue center (test_1h_w20.py style).
    """
    extended_x = np.arange(total_len)
    center_line = slope * extended_x + intercept
    upper_line = center_line + 2 * std_dev
    lower_line = center_line - 2 * std_dev

    # Within window - SOLID lines
    ax.plot(extended_x[:channel_len], center_line[:channel_len],
            'b-', linewidth=1, alpha=0.7)
    ax.plot(extended_x[:channel_len], upper_line[:channel_len],
            'g-', linewidth=2, label='Upper')
    ax.plot(extended_x[:channel_len], lower_line[:channel_len],
            'r-', linewidth=2, label='Lower')

    # Beyond window - DASHED projection
    if total_len > channel_len:
        ax.plot(extended_x[channel_len-1:], center_line[channel_len-1:],
                'b--', linewidth=1, alpha=0.5)
        ax.plot(extended_x[channel_len-1:], upper_line[channel_len-1:],
                'g--', linewidth=1.5, alpha=0.7)
        ax.plot(extended_x[channel_len-1:], lower_line[channel_len-1:],
                'r--', linewidth=1.5, alpha=0.7)

    # Mark channel end
    ax.axvline(x=channel_len-1, color='blue', linestyle=':', alpha=0.8, label='Channel End')


# =============================================================================
# BREAK MARKERS (from test_1h_w20.py - THE KEY LOGIC)
# =============================================================================

def plot_break_markers(
    ax,
    labels: ChannelLabels,
    channel_len: int,
    data_slice: pd.DataFrame
) -> None:
    """
    Plot break markers using the test_1h_w20.py hollow/filled triangle logic.

    - First break: HOLLOW triangle + dashed vertical line
    - Permanent break: FILLED triangle + solid vertical line
    """
    if not labels.break_scan_valid:
        return

    # FIRST BREAK - hollow triangle, dashed line
    if labels.bars_to_first_break >= 0:
        break_x = channel_len - 1 + labels.bars_to_first_break
        if break_x < len(data_slice):
            break_dir = labels.break_direction
            break_color = 'green' if break_dir == 1 else 'red'

            # Dashed vertical line
            ax.axvline(x=break_x, color=break_color, linestyle='--',
                      linewidth=1.5, alpha=0.6, label='First Break')

            # HOLLOW triangle marker
            if break_dir == 1:  # UP
                break_price = data_slice['high'].iloc[break_x]
                marker = '^'
            else:  # DOWN
                break_price = data_slice['low'].iloc[break_x]
                marker = 'v'

            ax.scatter([break_x], [break_price], marker=marker, s=150,
                      c='none', edgecolors=break_color, linewidths=2, zorder=5)

    # PERMANENT BREAK - filled triangle, solid line
    perm_dir = labels.permanent_break_direction
    perm_bar = labels.bars_to_permanent_break

    if perm_dir >= 0 and perm_bar >= 0:
        perm_x = channel_len - 1 + perm_bar
        if perm_x < len(data_slice):
            perm_color = 'green' if perm_dir == 1 else 'red'

            # Solid vertical line
            ax.axvline(x=perm_x, color=perm_color, linestyle='-',
                      linewidth=2, alpha=0.8, label='Permanent Break')

            # FILLED triangle marker
            if perm_dir == 1:  # UP
                perm_price = data_slice['high'].iloc[perm_x]
                marker = '^'
            else:  # DOWN
                perm_price = data_slice['low'].iloc[perm_x]
                marker = 'v'

            ax.scatter([perm_x], [perm_price], marker=marker, s=200,
                      c=perm_color, edgecolors='black', linewidths=2, zorder=6)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_samples(path: str) -> List[ChannelSample]:
    """Load samples from pickle file."""
    with open(path, 'rb') as f:
        samples = pickle.load(f)

    if isinstance(samples, list):
        return samples
    elif hasattr(samples, 'samples'):
        return samples.samples
    else:
        raise ValueError(f"Unknown sample format: {type(samples)}")


def load_market_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load TSLA and SPY market data."""
    from v15.data import load_market_data as _load
    tsla, spy, _ = _load(data_dir, validate=False)
    return tsla, spy


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLC data to target timeframe."""
    tf_map = {
        '5min': '5min',
        '15min': '15min',
        '30min': '30min',
        '1h': '1h',
        '2h': '2h',
        '3h': '3h',
        '4h': '4h',
        'daily': '1D',
        'weekly': '1W',
        'monthly': '1ME',
    }

    rule = tf_map.get(timeframe, '1h')

    try:
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    except Exception:
        # Fallback for older pandas
        if rule == '1ME':
            rule = '1M'
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    return resampled


# =============================================================================
# LABEL EXTRACTION
# =============================================================================

def get_labels(sample: ChannelSample, window: int, tf: str, asset: str = 'tsla') -> Optional[ChannelLabels]:
    """Extract labels from sample for given window/tf/asset."""
    try:
        lpw = sample.labels_per_window
        if window not in lpw:
            return None

        window_data = lpw[window]

        # New format: labels_per_window[window][asset][tf]
        if asset in window_data:
            if tf in window_data[asset]:
                return window_data[asset][tf]

        # Old format: labels_per_window[window][tf]
        if tf in window_data and isinstance(window_data[tf], ChannelLabels):
            return window_data[tf]

        return None
    except Exception:
        return None


def get_cross_labels(sample: ChannelSample, window: int, tf: str) -> Optional[CrossCorrelationLabels]:
    """Extract cross-correlation labels from sample."""
    try:
        lpw = sample.labels_per_window
        if window not in lpw:
            return None

        window_data = lpw[window]

        # Check for cross_correlation key
        if 'cross_correlation' in window_data and tf in window_data['cross_correlation']:
            return window_data['cross_correlation'][tf]

        return None
    except Exception:
        return None


def get_channel_features(sample: ChannelSample, window: int, tf: str, asset: str = 'tsla') -> Dict[str, float]:
    """Extract channel features from sample."""
    features = {}
    prefix = f"{tf}_w{window}_"
    if asset == 'spy':
        prefix = f"{tf}_w{window}_spy_"

    if hasattr(sample, 'tf_features') and sample.tf_features:
        for key, val in sample.tf_features.items():
            if key.startswith(prefix):
                short_key = key[len(prefix):]
                features[short_key] = val

    return features


# =============================================================================
# MAIN VISUALIZER CLASS
# =============================================================================

class ChannelVisualizer:
    """Interactive dual-asset channel visualizer with break marker distinction."""

    def __init__(
        self,
        samples: List[ChannelSample],
        tsla_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        start_idx: int = 0,
        dual_mode: bool = True
    ):
        self.samples = samples
        self.tsla_df = tsla_df
        self.spy_df = spy_df
        self.sample_idx = start_idx
        self.dual_mode = dual_mode

        # Display state
        self.window_idx: Optional[int] = None  # None = best window
        self.tf_view_idx = 0  # Index into TF_VIEWS
        self.show_help = False

        # Figure and axes
        self.fig = None
        self.axes = {}

    @property
    def current_sample(self) -> ChannelSample:
        return self.samples[self.sample_idx]

    @property
    def current_window(self) -> int:
        if self.window_idx is None:
            return self.current_sample.best_window or STANDARD_WINDOWS[0]
        return STANDARD_WINDOWS[self.window_idx]

    @property
    def current_tf_view(self) -> str:
        views = list(TF_VIEWS.keys())
        return views[self.tf_view_idx % len(views)]

    @property
    def current_tfs(self) -> List[str]:
        return TF_VIEWS[self.current_tf_view]

    def run(self):
        """Start the interactive visualizer."""
        if not HAS_MATPLOTLIB:
            print("ERROR: matplotlib not available")
            return

        self._setup_figure()
        self._draw()
        plt.show()

    def _setup_figure(self):
        """Create the figure and connect event handlers."""
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._create_layout()

    def _create_layout(self):
        """Create the grid layout based on current mode."""
        self.fig.clear()
        self.axes = {}

        tfs = self.current_tfs
        n_tfs = len(tfs)

        if self.dual_mode:
            # Dual mode: TSLA | SPY | Cross-correlation
            gs = GridSpec(n_tfs + 1, 3, figure=self.fig,
                         width_ratios=[3, 3, 2],
                         height_ratios=[3] * n_tfs + [1],
                         hspace=0.3, wspace=0.2)

            for i, tf in enumerate(tfs):
                self.axes[('tsla', tf)] = self.fig.add_subplot(gs[i, 0])
                self.axes[('spy', tf)] = self.fig.add_subplot(gs[i, 1])

            self.axes['cross'] = self.fig.add_subplot(gs[:n_tfs, 2])
            self.axes['status'] = self.fig.add_subplot(gs[n_tfs, :])
        else:
            # Single mode: 2x2 grid of timeframes
            gs = GridSpec(3, 2, figure=self.fig,
                         height_ratios=[3, 3, 1],
                         hspace=0.3, wspace=0.2)

            for i, tf in enumerate(tfs[:4]):
                row = i // 2
                col = i % 2
                self.axes[('tsla', tf)] = self.fig.add_subplot(gs[row, col])

            self.axes['status'] = self.fig.add_subplot(gs[2, :])

    def _draw(self):
        """Redraw all panels."""
        sample = self.current_sample
        window = self.current_window
        tfs = self.current_tfs

        # Clear and redraw each panel
        for tf in tfs:
            # TSLA panel
            ax_tsla = self.axes.get(('tsla', tf))
            if ax_tsla:
                ax_tsla.clear()
                self._draw_asset_panel(ax_tsla, 'tsla', tf, window, sample)

            # SPY panel (dual mode only)
            if self.dual_mode:
                ax_spy = self.axes.get(('spy', tf))
                if ax_spy:
                    ax_spy.clear()
                    self._draw_asset_panel(ax_spy, 'spy', tf, window, sample)

        # Cross-correlation panel (dual mode only)
        if self.dual_mode and 'cross' in self.axes:
            self._draw_cross_panel(self.axes['cross'], window, tfs, sample)

        # Status bar
        self._draw_status(self.axes['status'])

        # Help overlay
        if self.show_help:
            self._draw_help()

        self.fig.canvas.draw_idle()

    def _draw_asset_panel(
        self,
        ax,
        asset: str,
        tf: str,
        window: int,
        sample: ChannelSample
    ):
        """Draw a single asset panel with channel and break markers."""
        # Get data
        df = self.tsla_df if asset == 'tsla' else self.spy_df
        df_tf = resample_ohlc(df, tf)

        # Get labels
        labels = get_labels(sample, window, tf, asset)
        features = get_channel_features(sample, window, tf, asset)

        # Find data slice around sample timestamp
        timestamp = sample.timestamp

        try:
            # Find nearest index
            idx = df_tf.index.get_indexer([timestamp], method='nearest')[0]
        except Exception:
            idx = len(df_tf) // 2

        # Calculate slice bounds
        channel_len = window
        forward_bars = 50
        if labels and labels.break_scan_valid:
            max_break = max(
                labels.bars_to_first_break,
                labels.bars_to_permanent_break if labels.permanent_break_direction >= 0 else 0
            )
            forward_bars = min(max_break + 30, TF_MAX_SCAN.get(tf, 300))

        start = max(0, idx - channel_len + 1)
        end = min(len(df_tf), idx + forward_bars + 1)
        data_slice = df_tf.iloc[start:end]

        if len(data_slice) < 5:
            ax.text(0.5, 0.5, f"Insufficient data for {tf}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{asset.upper()} - {tf} (W:{window})")
            return

        # Plot candlesticks
        plot_candlesticks(ax, data_slice, start_idx=0)

        # Plot channel bounds if we have features
        slope = features.get('channel_slope', 0)
        intercept = features.get('channel_intercept', data_slice['close'].iloc[0])
        std_dev = features.get('channel_std_dev', 0)

        if slope != 0 or std_dev != 0:
            plot_channel_bounds(ax, slope, intercept, std_dev,
                              min(channel_len, len(data_slice)), len(data_slice))

        # Plot break markers (THE KEY FEATURE)
        if labels:
            plot_break_markers(ax, labels, min(channel_len, len(data_slice)), data_slice)

        # Format title
        title = f"{asset.upper()} - {tf}"
        if self.window_idx is None:
            title += f" (W:{window}*)"  # * indicates best window
        else:
            title += f" (W:{window})"

        if labels and labels.break_scan_valid:
            first_dir = "UP" if labels.break_direction == 1 else "DOWN"
            perm_dir = "UP" if labels.permanent_break_direction == 1 else "DOWN" if labels.permanent_break_direction == 0 else "N/A"
            diverged = " DIVERGED!" if (labels.permanent_break_direction >= 0 and
                                        labels.permanent_break_direction != labels.break_direction) else ""
            title += f"\nFirst:{labels.bars_to_first_break}b {first_dir} | Perm:{labels.bars_to_permanent_break}b {perm_dir}{diverged}"

        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Bars')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=7)

    def _draw_cross_panel(self, ax, window: int, tfs: List[str], sample: ChannelSample):
        """Draw cross-correlation summary panel."""
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        ax.text(0.5, 0.95, "Cross-Correlation", ha='center', va='top',
               fontsize=12, fontweight='bold')

        y = 0.85
        for tf in tfs:
            cross = get_cross_labels(sample, window, tf)
            tsla_labels = get_labels(sample, window, tf, 'tsla')
            spy_labels = get_labels(sample, window, tf, 'spy')

            # Header for this TF
            ax.text(0.5, y, f"── {tf} ──", ha='center', va='top', fontsize=10, fontweight='bold')
            y -= 0.08

            if cross:
                # Direction alignment
                aligned = "ALIGNED" if cross.direction_aligned else "DIVERGENT"
                color = 'green' if cross.direction_aligned else 'red'
                ax.text(0.5, y, f"Direction: {aligned}", ha='center', va='top',
                       fontsize=9, color=color)
                y -= 0.06

                # Who broke first
                if cross.tsla_broke_first:
                    leader = f"TSLA first (+{cross.break_lag_bars}b)"
                elif cross.spy_broke_first:
                    leader = f"SPY first (+{cross.break_lag_bars}b)"
                else:
                    leader = "Simultaneous"
                ax.text(0.5, y, f"Leader: {leader}", ha='center', va='top', fontsize=9)
                y -= 0.06

                # Magnitude spread
                ax.text(0.5, y, f"Mag spread: {cross.magnitude_spread:.2f}",
                       ha='center', va='top', fontsize=9)
                y -= 0.08
            elif tsla_labels and spy_labels:
                # Compute basic comparison
                if tsla_labels.break_scan_valid and spy_labels.break_scan_valid:
                    aligned = tsla_labels.break_direction == spy_labels.break_direction
                    color = 'green' if aligned else 'red'
                    ax.text(0.5, y, f"Direction: {'ALIGNED' if aligned else 'DIVERGENT'}",
                           ha='center', va='top', fontsize=9, color=color)
                    y -= 0.06

                    lag = tsla_labels.bars_to_first_break - spy_labels.bars_to_first_break
                    if lag < 0:
                        leader = f"TSLA first ({-lag}b)"
                    elif lag > 0:
                        leader = f"SPY first ({lag}b)"
                    else:
                        leader = "Simultaneous"
                    ax.text(0.5, y, f"Leader: {leader}", ha='center', va='top', fontsize=9)
                    y -= 0.08
                else:
                    ax.text(0.5, y, "No valid break data", ha='center', va='top',
                           fontsize=9, color='gray')
                    y -= 0.08
            else:
                ax.text(0.5, y, "No data", ha='center', va='top', fontsize=9, color='gray')
                y -= 0.08

    def _draw_status(self, ax):
        """Draw status bar."""
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        sample = self.current_sample
        window = self.current_window
        best = sample.best_window or STANDARD_WINDOWS[0]

        mode = "DUAL" if self.dual_mode else "SINGLE"
        window_str = f"{window}*" if self.window_idx is None else str(window)

        status = (f"Sample: {self.sample_idx + 1}/{len(self.samples)} | "
                 f"Window: {window_str} (best={best}) | "
                 f"View: {self.current_tf_view} | "
                 f"Mode: {mode}")

        ax.text(0.5, 0.7, status, ha='center', va='center', fontsize=10)

        nav = "← → Navigate | ↑ ↓ Jump 10 | r Random | w Window | t TF | m Mode | h Help | q Quit"
        ax.text(0.5, 0.3, nav, ha='center', va='center', fontsize=8, color='gray')

    def _draw_help(self):
        """Draw help overlay."""
        ax = self.fig.add_axes([0.2, 0.2, 0.6, 0.6])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor('white')
        ax.patch.set_alpha(0.95)
        ax.axis('off')

        ax.text(0.5, 0.95, "KEYBOARD SHORTCUTS", ha='center', va='top',
               fontsize=14, fontweight='bold')
        ax.text(0.5, 0.5, KEYBOARD_HELP, ha='center', va='center',
               fontsize=10, family='monospace')
        ax.text(0.5, 0.05, "Press H to close", ha='center', va='bottom',
               fontsize=10, color='gray')

    def _on_key(self, event):
        """Handle keyboard events."""
        key = event.key

        if key in ('q', 'escape'):
            plt.close(self.fig)
            return

        if key == 'h':
            self.show_help = not self.show_help
            self._create_layout()
            self._draw()
            return

        if key == 'left':
            self.sample_idx = max(0, self.sample_idx - 1)
        elif key == 'right':
            self.sample_idx = min(len(self.samples) - 1, self.sample_idx + 1)
        elif key == 'up':
            self.sample_idx = max(0, self.sample_idx - 10)
        elif key == 'down':
            self.sample_idx = min(len(self.samples) - 1, self.sample_idx + 10)
        elif key == 'r':
            import random
            self.sample_idx = random.randint(0, len(self.samples) - 1)
        elif key == 'w':
            # Cycle window: None -> 0 -> 1 -> ... -> 7 -> None
            if self.window_idx is None:
                self.window_idx = 0
            else:
                self.window_idx += 1
                if self.window_idx >= len(STANDARD_WINDOWS):
                    self.window_idx = None
        elif key == 't':
            self.tf_view_idx = (self.tf_view_idx + 1) % len(TF_VIEWS)
            self._create_layout()
        elif key == 'm':
            self.dual_mode = not self.dual_mode
            self._create_layout()
        elif key == 'i':
            self._print_sample_info()
            return  # Don't redraw
        else:
            return  # Unknown key

        self._draw()

    def _print_sample_info(self):
        """Print detailed sample info to console."""
        sample = self.current_sample
        window = self.current_window

        print("\n" + "=" * 60)
        print(f"SAMPLE {self.sample_idx + 1}/{len(self.samples)}")
        print("=" * 60)
        print(f"Timestamp: {sample.timestamp}")
        print(f"Best Window: {sample.best_window}")
        print(f"Current Window: {window}")

        for tf in self.current_tfs:
            print(f"\n--- {tf} ---")
            for asset in ['tsla', 'spy']:
                labels = get_labels(sample, window, tf, asset)
                if labels:
                    print(f"  {asset.upper()}:")
                    print(f"    Break valid: {labels.break_scan_valid}")
                    print(f"    First break: {labels.bars_to_first_break}b, dir={labels.break_direction}")
                    print(f"    Perm break:  {labels.bars_to_permanent_break}b, dir={labels.permanent_break_direction}")
                    print(f"    Returned:    {labels.returned_to_channel}")
        print("=" * 60 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='V15 Channel Visualizer - Interactive dual-asset channel inspection'
    )
    parser.add_argument('samples_file', nargs='?', help='Path to samples pickle file')
    parser.add_argument('--samples', '-s', help='Path to samples pickle file')
    parser.add_argument('--cache', '-c', help='Legacy alias for --samples')
    parser.add_argument('--data-dir', '-d', default='data', help='Market data directory')
    parser.add_argument('--start', '-i', type=int, default=0, help='Starting sample index')
    parser.add_argument('--single', action='store_true', help='Single asset mode (TSLA only)')

    args = parser.parse_args()

    # Resolve samples path
    samples_path = args.samples_file or args.samples or args.cache
    if not samples_path:
        parser.error("Must provide samples file path")

    if not Path(samples_path).exists():
        print(f"ERROR: Samples file not found: {samples_path}")
        sys.exit(1)

    # Load samples
    print(f"Loading samples from {samples_path}...")
    samples = load_samples(samples_path)
    print(f"Loaded {len(samples)} samples")

    # Load market data
    print(f"Loading market data from {args.data_dir}...")
    try:
        tsla_df, spy_df = load_market_data(args.data_dir)
        print(f"Loaded TSLA: {len(tsla_df)} bars, SPY: {len(spy_df)} bars")
    except Exception as e:
        print(f"ERROR loading market data: {e}")
        sys.exit(1)

    # Start visualizer
    viz = ChannelVisualizer(
        samples=samples,
        tsla_df=tsla_df,
        spy_df=spy_df,
        start_idx=args.start,
        dual_mode=not args.single
    )

    print("\nStarting visualizer...")
    print("Press 'h' for help, 'q' to quit")
    viz.run()


if __name__ == '__main__':
    main()
