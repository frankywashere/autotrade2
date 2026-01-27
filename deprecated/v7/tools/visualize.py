#!/usr/bin/env python3
"""
Channel Visualizer for v7

Generates clean visualizations of detected channels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import sys
import argparse
import random

# Add v7 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import detect_channel, find_best_channel, Direction, TouchType
from core.timeframe import resample_ohlc, TIMEFRAMES

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "reports"


def load_data(symbol: str = "TSLA", timeframe: str = "5min") -> pd.DataFrame:
    """Load and resample data."""
    csv_path = DATA_DIR / f"{symbol}_1min.csv"
    print(f"Loading {csv_path}...")

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()

    if timeframe != '1min':
        df = resample_ohlc(df, timeframe)
        print(f"  Resampled to {len(df):,} {timeframe} bars")

    return df


def plot_channel(df: pd.DataFrame, end_idx: int, window: int,
                symbol: str = "TSLA", timeframe: str = "5min",
                save_path: str = None, show: bool = True):
    """
    Plot a channel with all metrics.
    """
    # Get data slice and detect channel
    df_slice = df.iloc[:end_idx]
    channel = detect_channel(df_slice, window=window)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot window
    plot_df = df.iloc[max(0, end_idx - window):end_idx]
    timestamps = plot_df.index
    x = np.arange(len(timestamps))

    # Plot candlesticks
    for i, (ts, row) in enumerate(plot_df.iterrows()):
        color = 'green' if row['close'] >= row['open'] else 'red'
        # Wick
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=0.5)
        # Body
        body_bottom = min(row['open'], row['close'])
        body_height = abs(row['close'] - row['open'])
        rect = Rectangle((i - 0.3, body_bottom), 0.6, max(body_height, 0.01),
                         facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    # Plot channel lines
    ax.plot(x, channel.center_line, 'b-', linewidth=2, label='Center (regression)')
    ax.plot(x, channel.upper_line, 'r--', linewidth=1.5, label='Upper (2σ)')
    ax.plot(x, channel.lower_line, 'g--', linewidth=1.5, label='Lower (2σ)')

    # Fill channel
    ax.fill_between(x, channel.lower_line, channel.upper_line, alpha=0.1, color='blue')

    # Mark touch points
    for touch in channel.touches:
        if touch.touch_type == TouchType.UPPER:
            ax.scatter(touch.bar_index, touch.price, color='red', s=100,
                      marker='v', zorder=5, edgecolors='black', label='_nolegend_')
        else:
            ax.scatter(touch.bar_index, touch.price, color='green', s=100,
                      marker='^', zorder=5, edgecolors='black', label='_nolegend_')

    # Title
    end_time = timestamps[-1]
    title = f"{symbol} {timeframe} | Window: {window} bars | {end_time.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Metrics text
    valid_str = "VALID" if channel.valid else "INVALID"
    metrics_text = (
        f"Direction: {channel.direction.name}\n"
        f"Channel: {valid_str}\n"
        f"─────────────\n"
        f"R² = {channel.r_squared:.3f}\n"
        f"Complete Cycles = {channel.complete_cycles}\n"
        f"Bounces = {channel.bounce_count}\n"
        f"Touches = {len(channel.touches)}\n"
        f"Width = {channel.width_pct:.2f}%\n"
        f"Slope = {channel.slope_pct:.4f}%/bar\n"
        f"─────────────\n"
        f"Position = {channel.position_at():.2f}\n"
        f"Dist to Upper = {channel.distance_to_upper():.2f}%\n"
        f"Dist to Lower = {channel.distance_to_lower():.2f}%"
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace', bbox=props)

    ax.legend(loc='upper right')
    ax.set_xlabel('Bar Index')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return channel


def generate_samples(n_samples: int = 5, windows: list = [20, 50],
                    symbol: str = "TSLA", timeframe: str = "5min"):
    """Generate random sample visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(symbol, timeframe)

    max_window = max(windows)
    valid_start = max_window + 100
    valid_end = len(df) - 100

    print(f"\nGenerating {n_samples} samples...")

    results = []
    for i in range(n_samples):
        end_idx = random.randint(valid_start, valid_end)
        timestamp = df.index[end_idx]
        print(f"\n[Sample {i+1}/{n_samples}] {timestamp}")

        for w in windows:
            save_path = OUTPUT_DIR / f"v7_sample_{i+1:02d}_w{w}_{timestamp.strftime('%Y%m%d_%H%M')}.png"
            channel = plot_channel(df, end_idx, w, symbol, timeframe,
                                  save_path=str(save_path), show=False)
            results.append({
                'sample': i + 1,
                'timestamp': timestamp,
                'window': w,
                'valid': channel.valid,
                'direction': channel.direction.name,
                'cycles': channel.complete_cycles,
                'bounces': channel.bounce_count,
                'r_squared': channel.r_squared,
            })

    print(f"\n{'='*60}")
    print(f"Generated {len(results)} images in {OUTPUT_DIR}")
    print(f"{'='*60}")
    for r in results:
        status = "VALID" if r['valid'] else "INVALID"
        print(f"  Sample {r['sample']}, w{r['window']}: {status} | "
              f"{r['direction']}, cycles={r['cycles']}, R²={r['r_squared']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='v7 Channel Visualizer')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--windows', type=str, default='20,50')
    parser.add_argument('--symbol', type=str, default='TSLA')
    parser.add_argument('--timeframe', type=str, default='5min')
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    windows = [int(w) for w in args.windows.split(',')]
    generate_samples(args.samples, windows, args.symbol, args.timeframe)
