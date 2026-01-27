#!/usr/bin/env python3
"""
Label Inspector - Visual validation tool for channel labels with suspicious case detection.

Provides interactive inspection of generated labels across all timeframes, with
automatic detection and flagging of potentially problematic samples.

Features:
- Displays channel with projected bounds and break points
- Shows labels for all timeframes side-by-side
- Detects and flags suspicious samples:
  - Very short duration (channel breaks almost immediately)
  - NO_TRIGGER when permanent_break=True
  - All validity flags False for a TF with expected data
  - Inconsistent labels across TFs (e.g., 5min UP but 15min DOWN)
  - Very long duration (never broke, might indicate data issue)
  - Missing expected TFs when others have valid channels

Usage:
    python -m v7.tools.label_inspector
    python -m v7.tools.label_inspector --cache-path data/feature_cache/channel_samples.pkl
    python -m v7.tools.label_inspector --sample 100  # Jump to specific sample
    python -m v7.tools.label_inspector --suspicious-only  # Show only suspicious samples
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v7.core.timeframe import TIMEFRAMES, BARS_PER_TF
from v7.training.labels import (
    ChannelLabels, BreakDirection, BreakTriggerTF,
    NewChannelDirection, decode_trigger_tf
)
from v7.training.dataset import ChannelSample, get_cache_metadata


# =============================================================================
# Suspicious Sample Detection
# =============================================================================

@dataclass
class SuspiciousFlag:
    """A single flag indicating a potential issue with a sample."""
    flag_type: str      # Category of the issue
    tf: Optional[str]   # Which timeframe (None if sample-level)
    value: Any          # The problematic value
    message: str        # Human-readable description
    severity: str       # 'warning' or 'error'


@dataclass
class SuspiciousResult:
    """Result of suspicious sample detection for a single sample."""
    sample_idx: int
    flags: List[SuspiciousFlag]

    @property
    def is_suspicious(self) -> bool:
        return len(self.flags) > 0

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == 'error')

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == 'warning')


def detect_suspicious_sample(
    sample: ChannelSample,
    sample_idx: int,
    short_duration_thresholds: Optional[Dict[str, int]] = None,
    max_scan_thresholds: Optional[Dict[str, int]] = None
) -> SuspiciousResult:
    """
    Detect suspicious patterns in a single sample.

    Args:
        sample: The ChannelSample to analyze
        sample_idx: Index of this sample in the dataset
        short_duration_thresholds: Per-TF thresholds for "too short" (default: 5 for 5min, scaled for others)
        max_scan_thresholds: Per-TF thresholds for "too long" (default: 500 for 5min, scaled for others)

    Returns:
        SuspiciousResult with list of detected flags
    """
    flags = []

    # Default thresholds (scaled by timeframe)
    if short_duration_thresholds is None:
        short_duration_thresholds = {}
        base_short = 5  # 5 bars at 5min = 25 minutes
        for tf in TIMEFRAMES:
            # Scale inversely with bar size - shorter TFs need more bars to be meaningful
            short_duration_thresholds[tf] = max(2, base_short // max(1, BARS_PER_TF.get(tf, 1) // 3))

    if max_scan_thresholds is None:
        max_scan_thresholds = {}
        base_max = 500  # Default max_scan in 5min bars
        for tf in TIMEFRAMES:
            bars_per = BARS_PER_TF.get(tf, 1)
            max_scan_thresholds[tf] = max(10, base_max // bars_per)

    # Ensure labels is a dict
    labels_dict = sample.labels if isinstance(sample.labels, dict) else {'5min': sample.labels}

    # Track which TFs have valid labels for cross-TF consistency checks
    valid_tf_labels: Dict[str, ChannelLabels] = {}
    tfs_with_break: Dict[str, int] = {}  # tf -> break_direction

    for tf in TIMEFRAMES:
        tf_label = labels_dict.get(tf)

        if tf_label is None:
            # Check 1: Missing expected TF
            # Only flag as suspicious if nearby TFs have valid labels
            tf_idx = TIMEFRAMES.index(tf)
            has_neighbors = False

            # Check if adjacent TFs have valid labels
            if tf_idx > 0 and labels_dict.get(TIMEFRAMES[tf_idx - 1]) is not None:
                has_neighbors = True
            if tf_idx < len(TIMEFRAMES) - 1 and labels_dict.get(TIMEFRAMES[tf_idx + 1]) is not None:
                has_neighbors = True

            if has_neighbors and tf not in ['3month']:  # 3month often missing, don't flag
                flags.append(SuspiciousFlag(
                    flag_type='missing_tf',
                    tf=tf,
                    value=None,
                    message=f'{tf}: No valid channel when neighbors have data',
                    severity='warning'
                ))
            continue

        valid_tf_labels[tf] = tf_label

        # Check 2: Very short duration
        short_thresh = short_duration_thresholds.get(tf, 5)
        if tf_label.duration_valid and tf_label.permanent_break:
            if tf_label.duration_bars < short_thresh:
                flags.append(SuspiciousFlag(
                    flag_type='short_duration',
                    tf=tf,
                    value=tf_label.duration_bars,
                    message=f'{tf}: Duration only {tf_label.duration_bars} bars (threshold: {short_thresh})',
                    severity='warning'
                ))

        # Check 3: NO_TRIGGER when permanent_break=True
        if tf_label.permanent_break and tf_label.break_trigger_tf == 0:  # NO_TRIGGER
            # This means break happened but no longer TF boundary was found
            # Could indicate data issue or unusual market condition
            flags.append(SuspiciousFlag(
                flag_type='no_trigger',
                tf=tf,
                value='NO_TRIGGER',
                message=f'{tf}: Break occurred but no trigger TF found',
                severity='warning'
            ))

        # Check 4: All validity flags False when we have a label
        if not any([tf_label.duration_valid, tf_label.direction_valid,
                    tf_label.trigger_tf_valid, tf_label.new_channel_valid]):
            flags.append(SuspiciousFlag(
                flag_type='all_invalid',
                tf=tf,
                value=None,
                message=f'{tf}: All validity flags are False',
                severity='error'
            ))

        # Check 5: Very long duration (never broke within scan window)
        max_thresh = max_scan_thresholds.get(tf, 500)
        if tf_label.duration_valid and not tf_label.permanent_break:
            if tf_label.duration_bars >= max_thresh * 0.9:  # Within 10% of max
                flags.append(SuspiciousFlag(
                    flag_type='long_duration',
                    tf=tf,
                    value=tf_label.duration_bars,
                    message=f'{tf}: Duration {tf_label.duration_bars} bars (near max scan)',
                    severity='warning'
                ))

        # Track break directions for consistency check
        if tf_label.permanent_break and tf_label.direction_valid:
            tfs_with_break[tf] = tf_label.break_direction

    # Check 6: Inconsistent labels across TFs
    # Group consecutive TFs and check if break directions are consistent
    if len(tfs_with_break) >= 2:
        tf_list = sorted(tfs_with_break.keys(), key=lambda x: TIMEFRAMES.index(x))

        # Check adjacent TF pairs for direction inconsistency
        for i in range(len(tf_list) - 1):
            tf1, tf2 = tf_list[i], tf_list[i + 1]
            dir1, dir2 = tfs_with_break[tf1], tfs_with_break[tf2]

            # Only flag if TFs are close in the hierarchy
            idx1, idx2 = TIMEFRAMES.index(tf1), TIMEFRAMES.index(tf2)
            if idx2 - idx1 <= 2:  # Adjacent or one apart
                if dir1 != dir2:
                    dir1_str = 'UP' if dir1 == BreakDirection.UP else 'DOWN'
                    dir2_str = 'UP' if dir2 == BreakDirection.UP else 'DOWN'
                    flags.append(SuspiciousFlag(
                        flag_type='inconsistent_direction',
                        tf=f'{tf1}/{tf2}',
                        value=(dir1_str, dir2_str),
                        message=f'{tf1}={dir1_str} but {tf2}={dir2_str}',
                        severity='warning'
                    ))

    return SuspiciousResult(sample_idx=sample_idx, flags=flags)


def detect_suspicious_samples(
    samples: List[ChannelSample],
    short_duration_thresholds: Optional[Dict[str, int]] = None,
    max_scan_thresholds: Optional[Dict[str, int]] = None,
    progress: bool = True
) -> List[SuspiciousResult]:
    """
    Detect suspicious patterns across all samples.

    Args:
        samples: List of ChannelSample objects to analyze
        short_duration_thresholds: Per-TF thresholds for "too short"
        max_scan_thresholds: Per-TF thresholds for "too long"
        progress: Show progress indicator

    Returns:
        List of SuspiciousResult objects (only those with flags)
    """
    results = []

    for idx, sample in enumerate(samples):
        if progress and idx % 1000 == 0:
            print(f"  Scanning sample {idx}/{len(samples)}...", end='\r')

        result = detect_suspicious_sample(
            sample, idx,
            short_duration_thresholds=short_duration_thresholds,
            max_scan_thresholds=max_scan_thresholds
        )

        if result.is_suspicious:
            results.append(result)

    if progress:
        print(f"  Scanned {len(samples)} samples, found {len(results)} suspicious")

    return results


# =============================================================================
# Interactive Label Inspector
# =============================================================================

class LabelInspector:
    """
    Interactive inspector for channel labels with suspicious case detection.

    Keyboard controls:
    - Left/Right arrows: Navigate samples
    - Up/Down arrows: Jump 10 samples
    - f: Jump to next suspicious sample
    - F: Jump to previous suspicious sample
    - s: Toggle showing only suspicious samples
    - i: Print detailed info for current sample
    - q/Escape: Quit
    """

    def __init__(
        self,
        samples: List[ChannelSample],
        start_idx: int = 0,
        suspicious_only: bool = False
    ):
        """
        Initialize the label inspector.

        Args:
            samples: List of ChannelSample objects
            start_idx: Starting sample index
            suspicious_only: If True, only show suspicious samples
        """
        self.samples = samples
        self.current_idx = start_idx
        self.suspicious_only = suspicious_only

        # Detect suspicious samples at startup
        print("Detecting suspicious samples...")
        self.suspicious_results = detect_suspicious_samples(samples, progress=True)
        self.suspicious_indices = [r.sample_idx for r in self.suspicious_results]
        self.suspicious_map = {r.sample_idx: r for r in self.suspicious_results}

        print(f"Found {len(self.suspicious_indices)} suspicious samples out of {len(samples)}")

        # Create figure
        self.fig = None
        self.axes = None

    def find_next_suspicious(self, current_idx: int, forward: bool = True) -> int:
        """
        Find the next suspicious sample index.

        Args:
            current_idx: Current sample index
            forward: If True, search forward; if False, search backward

        Returns:
            Index of next suspicious sample (wraps around)
        """
        if not self.suspicious_indices:
            return current_idx

        if forward:
            # Find first suspicious index > current_idx
            for idx in self.suspicious_indices:
                if idx > current_idx:
                    return idx
            # Wrap around
            return self.suspicious_indices[0]
        else:
            # Find last suspicious index < current_idx
            for idx in reversed(self.suspicious_indices):
                if idx < current_idx:
                    return idx
            # Wrap around
            return self.suspicious_indices[-1]

    def get_sample_flags(self, sample_idx: int) -> List[SuspiciousFlag]:
        """Get suspicious flags for a sample."""
        result = self.suspicious_map.get(sample_idx)
        return result.flags if result else []

    def _create_figure(self):
        """Create the matplotlib figure layout."""
        self.fig = plt.figure(figsize=(18, 12))

        # Create grid: top row for summary, bottom for TF panels
        gs = GridSpec(3, 4, figure=self.fig, height_ratios=[0.8, 1, 1],
                      hspace=0.3, wspace=0.3)

        # Summary panel (top, spans all columns)
        self.ax_summary = self.fig.add_subplot(gs[0, :])
        self.ax_summary.axis('off')

        # TF panels (11 timeframes in 2 rows)
        self.tf_axes = {}
        tf_idx = 0
        for row in range(1, 3):
            for col in range(4):
                if tf_idx < len(TIMEFRAMES):
                    self.tf_axes[TIMEFRAMES[tf_idx]] = self.fig.add_subplot(gs[row, col])
                    tf_idx += 1

        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _plot_sample(self, sample_idx: int):
        """Plot a single sample with all TF labels."""
        if sample_idx < 0 or sample_idx >= len(self.samples):
            return

        sample = self.samples[sample_idx]
        flags = self.get_sample_flags(sample_idx)

        # Group flags by TF for panel highlighting
        tf_flags: Dict[str, List[SuspiciousFlag]] = {}
        for flag in flags:
            tf = flag.tf if flag.tf else 'summary'
            if tf not in tf_flags:
                tf_flags[tf] = []
            tf_flags[tf].append(flag)

        # Clear all axes
        self.ax_summary.clear()
        self.ax_summary.axis('off')
        for ax in self.tf_axes.values():
            ax.clear()

        # Draw summary panel
        self._draw_summary(sample, sample_idx, flags)

        # Draw TF panels
        labels_dict = sample.labels if isinstance(sample.labels, dict) else {'5min': sample.labels}

        for tf, ax in self.tf_axes.items():
            tf_label = labels_dict.get(tf)
            tf_flag_list = tf_flags.get(tf, [])

            # Draw panel border based on flags
            self._draw_tf_panel(ax, tf, tf_label, tf_flag_list)

        # Update title
        timestamp_str = sample.timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(sample, 'timestamp') else 'N/A'
        flag_status = f"[{len(flags)} FLAGS]" if flags else "[OK]"

        self.fig.suptitle(
            f"Sample {sample_idx + 1}/{len(self.samples)} | {timestamp_str} | {flag_status}",
            fontsize=14, fontweight='bold'
        )

        self.fig.canvas.draw()

    def _draw_summary(self, sample: ChannelSample, sample_idx: int, flags: List[SuspiciousFlag]):
        """Draw the summary panel at the top."""
        ax = self.ax_summary

        # Background color based on flags
        if any(f.severity == 'error' for f in flags):
            bg_color = '#ffcccc'  # Light red
        elif flags:
            bg_color = '#fff3cd'  # Light yellow
        else:
            bg_color = '#d4edda'  # Light green

        ax.set_facecolor(bg_color)

        # Status indicator
        if flags:
            status_text = f"WARNING: {len(flags)} suspicious flag(s) detected"
            status_color = 'red' if any(f.severity == 'error' for f in flags) else 'orange'
        else:
            status_text = "OK: No issues detected"
            status_color = 'green'

        ax.text(0.02, 0.85, status_text, transform=ax.transAxes,
                fontsize=14, fontweight='bold', color=status_color,
                verticalalignment='top')

        # Channel info
        channel = sample.channel
        channel_info = (
            f"Channel: {channel.direction.name} | "
            f"R2={channel.r_squared:.3f} | "
            f"Cycles={channel.complete_cycles} | "
            f"Width={channel.width_pct:.2f}%"
        )
        ax.text(0.02, 0.55, channel_info, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')

        # Flag messages (scrollable list on the right)
        if flags:
            flag_text = "Flags:\n"
            for i, flag in enumerate(flags[:6]):  # Limit to 6 flags
                severity_marker = "[E]" if flag.severity == 'error' else "[W]"
                flag_text += f"  {severity_marker} {flag.message}\n"
            if len(flags) > 6:
                flag_text += f"  ... and {len(flags) - 6} more\n"

            ax.text(0.55, 0.85, flag_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    color='darkred')

        # Navigation hints
        nav_text = "Navigation: [</>] prev/next | [f/F] next/prev suspicious | [s] toggle suspicious-only | [i] info | [q] quit"
        ax.text(0.02, 0.15, nav_text, transform=ax.transAxes,
                fontsize=9, color='gray', verticalalignment='top')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def _draw_tf_panel(
        self,
        ax: plt.Axes,
        tf: str,
        tf_label: Optional[ChannelLabels],
        flags: List[SuspiciousFlag]
    ):
        """Draw a single timeframe panel."""
        # Set background based on flags
        if any(f.severity == 'error' for f in flags):
            ax.set_facecolor('#ffcccc')
            border_color = 'red'
        elif flags:
            ax.set_facecolor('#fff3cd')
            border_color = 'orange'
        else:
            ax.set_facecolor('white')
            border_color = 'gray'

        # Draw border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2 if flags else 1)

        ax.set_title(tf, fontsize=10, fontweight='bold' if flags else 'normal',
                     color='red' if flags else 'black')

        if tf_label is None:
            ax.text(0.5, 0.5, 'No Label', ha='center', va='center',
                    fontsize=12, color='gray', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        # Build label info text
        info_lines = []

        # Duration
        duration_str = f"Duration: {tf_label.duration_bars}"
        if tf_label.duration_valid:
            duration_str += " [V]"
        info_lines.append(duration_str)

        # Break info
        if tf_label.permanent_break:
            direction_str = "UP" if tf_label.break_direction == BreakDirection.UP else "DOWN"
            direction_str = f"Break: {direction_str}"
            if tf_label.direction_valid:
                direction_str += " [V]"
            info_lines.append(direction_str)

            # Trigger TF
            trigger_str = decode_trigger_tf(tf_label.break_trigger_tf) or "NO_TRIGGER"
            trigger_line = f"Trigger: {trigger_str}"
            if tf_label.trigger_tf_valid:
                trigger_line += " [V]"
            info_lines.append(trigger_line)
        else:
            info_lines.append("Break: No")

        # New channel direction
        new_dir_map = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
        new_dir_str = new_dir_map.get(tf_label.new_channel_direction, '?')
        new_channel_line = f"Next: {new_dir_str}"
        if tf_label.new_channel_valid:
            new_channel_line += " [V]"
        info_lines.append(new_channel_line)

        # Display info
        y_pos = 0.85
        for line in info_lines:
            color = 'green' if '[V]' in line else 'gray'
            ax.text(0.1, y_pos, line, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    fontfamily='monospace', color=color)
            y_pos -= 0.18

        # Show flag messages at bottom
        if flags:
            flag_msgs = [f.message.split(':')[-1].strip() for f in flags[:2]]
            flag_text = '\n'.join(flag_msgs)
            ax.text(0.1, 0.1, flag_text, transform=ax.transAxes,
                    fontsize=7, color='red', verticalalignment='bottom')

        ax.set_xticks([])
        ax.set_yticks([])

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key in ['right', 'n']:
            self.current_idx = min(self.current_idx + 1, len(self.samples) - 1)
            self._plot_sample(self.current_idx)
        elif event.key in ['left', 'p']:
            self.current_idx = max(self.current_idx - 1, 0)
            self._plot_sample(self.current_idx)
        elif event.key == 'up':
            self.current_idx = max(self.current_idx - 10, 0)
            self._plot_sample(self.current_idx)
        elif event.key == 'down':
            self.current_idx = min(self.current_idx + 10, len(self.samples) - 1)
            self._plot_sample(self.current_idx)
        elif event.key == 'f':
            # Jump to next suspicious
            self.current_idx = self.find_next_suspicious(self.current_idx, forward=True)
            self._plot_sample(self.current_idx)
        elif event.key == 'F':
            # Jump to previous suspicious
            self.current_idx = self.find_next_suspicious(self.current_idx, forward=False)
            self._plot_sample(self.current_idx)
        elif event.key == 's':
            # Toggle suspicious-only mode
            self.suspicious_only = not self.suspicious_only
            mode_str = "ON" if self.suspicious_only else "OFF"
            print(f"Suspicious-only mode: {mode_str}")
            if self.suspicious_only and self.suspicious_indices:
                # Jump to first suspicious sample
                self.current_idx = self.suspicious_indices[0]
            self._plot_sample(self.current_idx)
        elif event.key == 'i':
            self._print_sample_info()
        elif event.key in ['q', 'escape']:
            plt.close(self.fig)

    def _print_sample_info(self):
        """Print detailed information about current sample."""
        sample = self.samples[self.current_idx]
        flags = self.get_sample_flags(self.current_idx)

        print("\n" + "=" * 60)
        print(f"SAMPLE {self.current_idx} DETAILED INFO")
        print("=" * 60)

        print(f"\nTimestamp: {sample.timestamp}")
        print(f"Channel End Index: {sample.channel_end_idx}")
        print(f"Channel Direction: {sample.channel.direction.name}")
        print(f"Channel Valid: {sample.channel.valid}")
        print(f"R-squared: {sample.channel.r_squared:.4f}")
        print(f"Complete Cycles: {sample.channel.complete_cycles}")
        print(f"Width: {sample.channel.width_pct:.2f}%")

        print("\n--- Labels by Timeframe ---")
        labels_dict = sample.labels if isinstance(sample.labels, dict) else {'5min': sample.labels}

        for tf in TIMEFRAMES:
            tf_label = labels_dict.get(tf)
            if tf_label is None:
                print(f"  {tf:10s}: No label")
                continue

            trigger_str = decode_trigger_tf(tf_label.break_trigger_tf) or "NO_TRIGGER"
            new_dir_map = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
            new_dir = new_dir_map.get(tf_label.new_channel_direction, '?')
            break_dir = 'UP' if tf_label.break_direction == 1 else 'DOWN'

            validity = (
                f"d={'V' if tf_label.duration_valid else '-'}"
                f"b={'V' if tf_label.direction_valid else '-'}"
                f"t={'V' if tf_label.trigger_tf_valid else '-'}"
                f"n={'V' if tf_label.new_channel_valid else '-'}"
            )

            print(f"  {tf:10s}: dur={tf_label.duration_bars:4d} break={tf_label.permanent_break} "
                  f"dir={break_dir:4s} trigger={trigger_str:15s} next={new_dir:8s} [{validity}]")

        print("\n--- Suspicious Flags ---")
        if flags:
            for flag in flags:
                severity_marker = "[ERROR]" if flag.severity == 'error' else "[WARN]"
                print(f"  {severity_marker} {flag.flag_type}: {flag.message}")
        else:
            print("  None")

        print("=" * 60 + "\n")

    def run(self):
        """Run the interactive inspector."""
        self._create_figure()
        self._plot_sample(self.current_idx)
        plt.show()


def print_suspicious_summary(results: List[SuspiciousResult]):
    """Print a summary of suspicious samples."""
    if not results:
        print("\nNo suspicious samples found!")
        return

    print(f"\n{'=' * 60}")
    print(f"SUSPICIOUS SAMPLES SUMMARY: {len(results)} total")
    print('=' * 60)

    # Count by flag type
    flag_counts: Dict[str, int] = {}
    for result in results:
        for flag in result.flags:
            flag_counts[flag.flag_type] = flag_counts.get(flag.flag_type, 0) + 1

    print("\nFlag type counts:")
    for flag_type, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        print(f"  {flag_type:25s}: {count:4d}")

    # Show first 10 suspicious samples
    print(f"\nFirst 10 suspicious samples:")
    for result in results[:10]:
        flags_summary = ', '.join([f.flag_type for f in result.flags])
        print(f"  Sample {result.sample_idx:5d}: {flags_summary}")

    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Label Inspector - Visual validation with suspicious detection')
    parser.add_argument('--cache-path', type=str, default=None,
                        help='Path to cached samples (.pkl file)')
    parser.add_argument('--sample', type=int, default=0,
                        help='Starting sample index')
    parser.add_argument('--suspicious-only', action='store_true',
                        help='Show only suspicious samples')
    parser.add_argument('--summary-only', action='store_true',
                        help='Print summary of suspicious samples and exit')

    args = parser.parse_args()

    # Find cache path
    if args.cache_path:
        cache_path = Path(args.cache_path)
    else:
        # Default location
        cache_path = Path(__file__).parent.parent.parent / "data" / "feature_cache" / "channel_samples.pkl"

    if not cache_path.exists():
        print(f"Cache file not found: {cache_path}")
        print("Please provide a valid cache path with --cache-path")
        sys.exit(1)

    # Load samples
    print(f"Loading samples from {cache_path}...")
    with open(cache_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples")

    # Load metadata if available
    meta_path = cache_path.with_suffix('.json')
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        print(f"Cache version: {metadata.get('cache_version', 'unknown')}")

    if args.summary_only:
        # Just print summary and exit
        results = detect_suspicious_samples(samples, progress=True)
        print_suspicious_summary(results)
    else:
        # Run interactive inspector
        inspector = LabelInspector(
            samples,
            start_idx=args.sample,
            suspicious_only=args.suspicious_only
        )

        # Print summary before starting interactive mode
        print_suspicious_summary(inspector.suspicious_results)

        print("\nStarting interactive inspector...")
        print("Use arrow keys to navigate, 'f' for next suspicious, 'q' to quit")
        inspector.run()


if __name__ == '__main__':
    main()
