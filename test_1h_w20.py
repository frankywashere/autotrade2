#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for 1h window-20 channel detection and labeling.

This script loads market data, runs channel detection for just the 1h timeframe
with window size 20, generates labels using the hybrid method, and prints
detailed debugging info about the results.

Usage:
    python test_1h_w20.py [--visualize] [--max-channels N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# V15 imports
from v15.data import load_market_data
from v15.labels import (
    detect_all_channels,
    generate_all_labels,
    DetectedChannel,
    LabeledChannel,
    channel_map_stats,
    labeled_map_stats,
)
from v15.config import TF_MAX_SCAN

# V7 channel detection
from v7.core.channel import detect_channel, Channel
from v7.core.timeframe import resample_ohlc


def print_separator(title: str = "", char: str = "=", width: int = 70):
    """Print a separator line with optional title."""
    if title:
        side_len = (width - len(title) - 2) // 2
        print(f"\n{char * side_len} {title} {char * side_len}")
    else:
        print(char * width)


def print_channel_info(channel: Channel, idx: int, tf: str, window: int):
    """Print detailed info about a detected channel."""
    print(f"\n  Channel #{idx}:")
    print(f"    Timeframe: {tf}, Window: {window}")
    print(f"    Valid: {channel.valid}")
    print(f"    Direction: {channel.direction.name if hasattr(channel.direction, 'name') else channel.direction}")
    print(f"    Slope: {channel.slope:.6f}")
    print(f"    Intercept: {channel.intercept:.2f}")
    print(f"    Std Dev: {channel.std_dev:.4f}")
    print(f"    R-squared: {channel.r_squared:.4f}")
    print(f"    Width %: {channel.width_pct:.2f}%")
    print(f"    Complete Cycles: {channel.complete_cycles}")
    print(f"    Bounce Count: {channel.bounce_count}")
    print(f"    Touches: {len(channel.touches)}")
    if channel.touches:
        print(f"    First touch: bar {channel.touches[0].bar_index}, type {channel.touches[0].touch_type.name}")
        print(f"    Last touch: bar {channel.touches[-1].bar_index}, type {channel.touches[-1].touch_type.name}")


def print_detected_channel_info(detected: DetectedChannel, idx: int):
    """Print info about a DetectedChannel from Pass 1."""
    print(f"\n  DetectedChannel #{idx}:")
    print(f"    TF: {detected.tf}, Window: {detected.window}")
    print(f"    Index range: {detected.start_idx} - {detected.end_idx}")
    print(f"    Timestamp range: {detected.start_timestamp} - {detected.end_timestamp}")
    print(f"    Direction: {detected.direction} (0=BEAR, 1=SIDEWAYS, 2=BULL)")

    ch = detected.channel
    print(f"    Slope: {ch.slope:.6f}")
    print(f"    Intercept: {ch.intercept:.2f}")
    print(f"    Std Dev: {ch.std_dev:.4f}")
    print(f"    Width %: {ch.width_pct:.2f}%")
    print(f"    Complete Cycles: {ch.complete_cycles}")


def print_labeled_channel_info(labeled: LabeledChannel, idx: int):
    """Print info about a LabeledChannel from Pass 2."""
    detected = labeled.detected
    labels = labeled.labels

    print(f"\n  LabeledChannel #{idx}:")
    print(f"    TF: {detected.tf}, Window: {detected.window}")
    print(f"    Index range: {detected.start_idx} - {detected.end_idx}")
    print(f"    Timestamp: {detected.start_timestamp} - {detected.end_timestamp}")
    print(f"    Next channel idx: {labeled.next_channel_idx}")

    print(f"\n    Labels:")
    print(f"      break_scan_valid: {labels.break_scan_valid}")
    print(f"      direction_valid: {labels.direction_valid}")
    print(f"      duration_valid: {labels.duration_valid}")

    if labels.break_scan_valid:
        print(f"\n      FIRST Break (initial, may be false break):")
        print(f"        bars_to_first_break: {labels.bars_to_first_break}")
        print(f"        break_direction: {labels.break_direction} (0=DOWN, 1=UP)")
        print(f"        break_magnitude: {labels.break_magnitude:.4f} std devs")
        print(f"        returned_to_channel: {labels.returned_to_channel}")
        print(f"        bounces_after_return: {labels.bounces_after_return}")
        print(f"        channel_continued: {labels.channel_continued}")

        print(f"\n      PERMANENT Break (final/lasting):")
        perm_dir = labels.permanent_break_direction
        perm_dir_str = "NONE" if perm_dir == -1 else ("DOWN" if perm_dir == 0 else "UP")
        print(f"        permanent_break_direction: {perm_dir} ({perm_dir_str})")
        print(f"        permanent_break_magnitude: {labels.permanent_break_magnitude:.4f} std devs")
        print(f"        bars_to_permanent_break: {labels.bars_to_permanent_break}")

        # Show if direction diverged (first != permanent)
        if perm_dir >= 0 and perm_dir != labels.break_direction:
            print(f"        *** DIRECTION DIVERGED: First={labels.break_direction}, Permanent={perm_dir} ***")

    if labels.duration_valid:
        print(f"\n      Duration info:")
        print(f"        duration_bars: {labels.duration_bars}")


def plot_candlesticks(ax, df, start_idx=0):
    """Plot OHLC candlesticks."""
    import matplotlib.pyplot as plt
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

        # Draw wick (high-low line)
        ax.plot([x, x], [low, high], color='black', linewidth=0.5)

        # Draw body
        if body_height > 0:
            ax.add_patch(plt.Rectangle((x - 0.3, body_bottom), 0.6, body_height,
                                       facecolor=color, edgecolor='black', linewidth=0.5))
        else:
            # Doji - just a line
            ax.plot([x - 0.3, x + 0.3], [close, close], color='black', linewidth=1)


def visualize_channel(
    df: pd.DataFrame,
    detected: DetectedChannel,
    labeled: LabeledChannel,
    sample_idx: int
):
    """Visualize a single channel with its forward scan results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    channel = detected.channel
    labels = labeled.labels
    # Get data slice for the channel window + some forward bars
    max_scan = TF_MAX_SCAN.get(detected.tf, 300)
    if labels.break_scan_valid:
        # Consider both first and permanent breaks for visible range
        max_break_bar = max(
            labels.bars_to_first_break,
            labels.bars_to_permanent_break if labels.permanent_break_direction >= 0 else 0
        )
        forward_bars = min(max_break_bar + 30, max_scan)
    else:
        forward_bars = 50

    start_idx = detected.start_idx
    end_idx = min(detected.end_idx + forward_bars, len(df) - 1)

    data_slice = df.iloc[start_idx:end_idx + 1]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot OHLC candlesticks
    plot_candlesticks(ax, data_slice, start_idx=0)

    # Plot channel bounds (extended forward)
    channel_len = detected.end_idx - detected.start_idx + 1

    # Extend channel lines forward
    extended_len = len(data_slice)
    extended_x = np.arange(extended_len)

    center_line = channel.slope * extended_x + channel.intercept
    upper_line = center_line + 2 * channel.std_dev
    lower_line = center_line - 2 * channel.std_dev

    # Channel bounds (solid within window, dashed projection)
    ax.plot(extended_x[:channel_len], center_line[:channel_len], 'b-', linewidth=1, alpha=0.7)
    ax.plot(extended_x[:channel_len], upper_line[:channel_len], 'g-', linewidth=2, label='Upper Bound')
    ax.plot(extended_x[:channel_len], lower_line[:channel_len], 'r-', linewidth=2, label='Lower Bound')

    # Projected bounds
    if extended_len > channel_len:
        ax.plot(extended_x[channel_len-1:], center_line[channel_len-1:], 'b--', linewidth=1, alpha=0.5)
        ax.plot(extended_x[channel_len-1:], upper_line[channel_len-1:], 'g--', linewidth=1.5, alpha=0.7)
        ax.plot(extended_x[channel_len-1:], lower_line[channel_len-1:], 'r--', linewidth=1.5, alpha=0.7)

    # Mark channel end
    ax.axvline(x=channel_len-1, color='blue', linestyle=':', alpha=0.8, label='Channel End')

    # Mark FIRST break point if valid
    if labels.break_scan_valid and labels.bars_to_first_break >= 0:
        break_x = channel_len - 1 + labels.bars_to_first_break
        if break_x < len(data_slice):
            break_color = 'green' if labels.break_direction == 1 else 'red'
            ax.axvline(x=break_x, color=break_color, linestyle='--', linewidth=1.5, alpha=0.6, label='First Break')

            # Add first break marker (hollow triangle)
            break_price = data_slice['high'].iloc[break_x] if labels.break_direction == 1 else data_slice['low'].iloc[break_x]
            marker = '^' if labels.break_direction == 1 else 'v'
            ax.scatter([break_x], [break_price], marker=marker, s=150, c='none',
                      edgecolors=break_color, linewidths=2, zorder=5)

    # Mark PERMANENT break point if valid and different from first
    perm_dir = labels.permanent_break_direction
    perm_bar = labels.bars_to_permanent_break
    if labels.break_scan_valid and perm_dir >= 0 and perm_bar >= 0:
        perm_x = channel_len - 1 + perm_bar
        if perm_x < len(data_slice):
            perm_color = 'green' if perm_dir == 1 else 'red'
            ax.axvline(x=perm_x, color=perm_color, linestyle='-', linewidth=2, alpha=0.8, label='Permanent Break')

            # Add permanent break marker (filled triangle)
            perm_price = data_slice['high'].iloc[perm_x] if perm_dir == 1 else data_slice['low'].iloc[perm_x]
            marker = '^' if perm_dir == 1 else 'v'
            ax.scatter([perm_x], [perm_price], marker=marker, s=200, c=perm_color,
                      edgecolors='black', linewidths=2, zorder=6)

    # Title and labels
    direction_name = ['BEAR', 'SIDEWAYS', 'BULL'][detected.direction]
    first_break_dir_name = 'UP' if labels.break_direction == 1 else 'DOWN'
    perm_break_dir_name = 'NONE' if perm_dir == -1 else ('UP' if perm_dir == 1 else 'DOWN')

    title_lines = [
        f"1h Window-20 Channel #{sample_idx}",
        f"Direction: {direction_name}, Cycles: {channel.complete_cycles}, Width: {channel.width_pct:.2f}%",
    ]
    if labels.break_scan_valid:
        title_lines.append(
            f"First Break: {labels.bars_to_first_break} bars, {first_break_dir_name}, "
            f"Returned: {labels.returned_to_channel}"
        )
        if perm_dir >= 0:
            diverged = " (DIVERGED!)" if perm_dir != labels.break_direction else ""
            title_lines.append(
                f"Permanent Break: {perm_bar} bars, {perm_break_dir_name}{diverged}"
            )
    else:
        title_lines.append("Break: No valid break scan")

    ax.set_title('\n'.join(title_lines), fontsize=12)
    ax.set_xlabel('Bars from Channel Start')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test 1h window-20 channel detection and labeling')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--max-channels', type=int, default=5, help='Max channels to print details for')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample channels')
    parser.add_argument('--num-viz', type=int, default=2, help='Number of channels to visualize')
    parser.add_argument('--diverged-only', action='store_true', help='Only show channels where first break != permanent break')
    args = parser.parse_args()

    print_separator("Loading Market Data")
    print(f"Data directory: {args.data_dir}")

    tsla_df, spy_df, vix_df = load_market_data(args.data_dir)
    print(f"TSLA: {len(tsla_df)} bars, {tsla_df.index[0]} to {tsla_df.index[-1]}")
    print(f"SPY: {len(spy_df)} bars")
    print(f"VIX: {len(vix_df)} bars")

    # Resample to 1h
    print_separator("Resampling to 1h")
    tsla_1h = resample_ohlc(tsla_df, '1h')
    spy_1h = resample_ohlc(spy_df, '1h')
    print(f"TSLA 1h: {len(tsla_1h)} bars")
    print(f"SPY 1h: {len(spy_1h)} bars")

    print_separator("PASS 1: Detecting Channels (1h, window=20)")

    # Detect channels for just 1h, window 20
    channel_map, resampled_dfs = detect_all_channels(
        df=tsla_df,
        timeframes=['1h'],
        windows=[20],
        step=1,  # Scan every bar for thorough detection
        min_cycles=1,
        min_gap_bars=5,
        verbose=True,
        workers=4
    )

    key = ('1h', 20)
    channels = channel_map.get(key, [])
    print(f"\nDetected {len(channels)} channels for 1h/w20")

    # Print stats
    stats = channel_map_stats(channel_map)
    print(f"\nChannel Map Stats:")
    print(f"  Total channels: {stats['total_channels']}")
    print(f"  Direction counts: BEAR={stats['direction_counts'][0]}, SIDEWAYS={stats['direction_counts'][1]}, BULL={stats['direction_counts'][2]}")

    # Print details for first few channels
    print_separator(f"Sample Channel Details (first {min(args.max_channels, len(channels))})")
    for i, detected in enumerate(channels[:args.max_channels]):
        print_detected_channel_info(detected, i)

    print_separator("PASS 2: Generating Labels (hybrid method)")

    labeled_map = generate_all_labels(
        channel_map=channel_map,
        resampled_dfs=resampled_dfs,
        labeling_method="hybrid",
        verbose=True
    )

    labeled_channels = labeled_map.get(key, [])
    print(f"\nLabeled {len(labeled_channels)} channels for 1h/w20")

    # Print labeled stats
    label_stats = labeled_map_stats(labeled_map)
    print(f"\nLabel Stats:")
    print(f"  Total labeled: {label_stats['total_labeled']}")
    print(f"  Valid direction: {label_stats['valid_direction_count']}")
    print(f"  Valid duration: {label_stats['valid_duration_count']}")
    print(f"  Break UP: {label_stats['break_up_count']}")
    print(f"  Break DOWN: {label_stats['break_down_count']}")

    # Print detailed label info for first few
    print_separator(f"Sample Label Details (first {min(args.max_channels, len(labeled_channels))})")
    for i, labeled in enumerate(labeled_channels[:args.max_channels]):
        print_labeled_channel_info(labeled, i)

    # Additional analysis
    print_separator("Break Scan Analysis")

    valid_break_scans = [lc for lc in labeled_channels if lc.labels.break_scan_valid]
    invalid_break_scans = [lc for lc in labeled_channels if not lc.labels.break_scan_valid]

    print(f"Valid break scans: {len(valid_break_scans)} ({100*len(valid_break_scans)/len(labeled_channels):.1f}%)")
    print(f"Invalid break scans: {len(invalid_break_scans)} ({100*len(invalid_break_scans)/len(labeled_channels):.1f}%)")

    if valid_break_scans:
        bars_to_break = [lc.labels.bars_to_first_break for lc in valid_break_scans]
        magnitudes = [lc.labels.break_magnitude for lc in valid_break_scans]
        returned = [lc.labels.returned_to_channel for lc in valid_break_scans]

        print(f"\nBars to first break:")
        print(f"  Min: {min(bars_to_break)}, Max: {max(bars_to_break)}, Mean: {np.mean(bars_to_break):.1f}")
        print(f"  Median: {np.median(bars_to_break):.1f}")

        print(f"\nBreak magnitude (std devs):")
        print(f"  Min: {min(magnitudes):.2f}, Max: {max(magnitudes):.2f}, Mean: {np.mean(magnitudes):.2f}")

        print(f"\nReturned to channel: {sum(returned)} ({100*sum(returned)/len(returned):.1f}%)")

        # Break direction distribution
        up_breaks = sum(1 for lc in valid_break_scans if lc.labels.break_direction == 1)
        down_breaks = len(valid_break_scans) - up_breaks
        print(f"\nBreak direction:")
        print(f"  UP: {up_breaks} ({100*up_breaks/len(valid_break_scans):.1f}%)")
        print(f"  DOWN: {down_breaks} ({100*down_breaks/len(valid_break_scans):.1f}%)")

    # Visualization
    if args.visualize and labeled_channels:
        print_separator("Visualization")

        # Filter for diverged samples if requested
        if args.diverged_only:
            viz_candidates = [
                lc for lc in labeled_channels
                if lc.labels.break_scan_valid
                and lc.labels.permanent_break_direction >= 0
                and lc.labels.permanent_break_direction != lc.labels.break_direction
            ]
            print(f"Found {len(viz_candidates)} diverged channels (first != permanent)")
        else:
            # Pick interesting samples (valid breaks with moderate bars_to_first_break)
            viz_candidates = [
                lc for lc in labeled_channels
                if lc.labels.break_scan_valid and 5 < lc.labels.bars_to_first_break < 100
            ]

        if not viz_candidates:
            viz_candidates = labeled_channels[:args.num_viz]
        else:
            viz_candidates = viz_candidates[:args.num_viz]

        print(f"Visualizing {len(viz_candidates)} channels...")

        for i, labeled in enumerate(viz_candidates):
            print(f"\nVisualizing channel {i+1}/{len(viz_candidates)}...")
            visualize_channel(tsla_1h, labeled.detected, labeled, i)

    print_separator("Done")


if __name__ == '__main__':
    main()
