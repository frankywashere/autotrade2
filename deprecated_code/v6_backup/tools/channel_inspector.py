#!/usr/bin/env python3
"""
Channel Inspector - Visual validation of channel detection

Creates standalone images of detected channels with:
- Price candlesticks
- Regression line (center)
- Upper/Lower bounds (±2σ)
- Bounce points marked
- Metrics displayed

Usage:
    python tools/channel_inspector.py
    python tools/channel_inspector.py --samples 10
    python tools/channel_inspector.py --date 2024-06-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from scipy import stats
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import random

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "reports" / "channel_samples"


def load_and_resample_data(symbol: str = "TSLA", base_tf: str = "5min") -> pd.DataFrame:
    """Load 1-min data and resample to base timeframe (5min)."""
    csv_path = DATA_DIR / f"{symbol}_1min.csv"
    print(f"Loading {csv_path}...")

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()

    # Resample to 5-min bars
    df_5min = df.resample(base_tf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    print(f"  Loaded {len(df):,} 1-min bars → {len(df_5min):,} 5-min bars")
    print(f"  Date range: {df_5min.index[0]} to {df_5min.index[-1]}")

    return df_5min


def calculate_channel(prices_close: np.ndarray, prices_high: np.ndarray,
                     prices_low: np.ndarray, window: int) -> dict:
    """
    Calculate channel using linear regression.

    Returns dict with:
        - slope, intercept, r_squared (for close)
        - upper_line, lower_line, center_line (arrays)
        - std_dev, channel_width_pct
        - bounces at different thresholds
        - complete_cycles at different thresholds
    """
    if len(prices_close) < window:
        return None

    # Use last 'window' bars
    close = prices_close[-window:]
    high = prices_high[-window:]
    low = prices_low[-window:]

    x = np.arange(window)

    # Linear regression on close prices
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, close)
    r_squared = r_value ** 2

    # Fitted line
    center_line = slope * x + intercept

    # Residuals and standard deviation
    residuals = close - center_line
    std_dev = np.std(residuals)

    # Channel bounds (±2σ)
    upper_line = center_line + 2.0 * std_dev
    lower_line = center_line - 2.0 * std_dev

    # Also fit high/low for comparison
    slope_high, intercept_high, r_high, _, _ = stats.linregress(x, high)
    slope_low, intercept_low, r_low, _, _ = stats.linregress(x, low)
    high_line = slope_high * x + intercept_high
    low_line = slope_low * x + intercept_low

    # Take more conservative bounds
    upper_line = np.maximum(upper_line, high_line)
    lower_line = np.minimum(lower_line, low_line)

    # Channel width
    avg_price = np.mean(close)
    channel_width_pct = ((upper_line[-1] - lower_line[-1]) / avg_price) * 100

    # Detect bounces using HIGHS for upper touches, LOWS for lower touches
    # Threshold is % of channel width, not % of price
    channel_width = upper_line - lower_line

    # Thresholds as % of channel width (more intuitive)
    # 10% means "within 10% of the channel width from the boundary"
    thresholds = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20% of channel width

    bounces = {}
    complete_cycles = {}
    touch_points = {}  # For visualization

    for thresh in thresholds:
        touches = []
        for i in range(len(close)):
            # Use HIGH for upper touches, LOW for lower touches
            high_price = high[i]
            low_price = low[i]

            # Distance as fraction of channel width at this bar
            width_at_bar = channel_width[i]
            if width_at_bar <= 0:
                continue

            # Check if HIGH is near/above upper line
            upper_dist = (upper_line[i] - high_price) / width_at_bar
            # Check if LOW is near/below lower line
            lower_dist = (low_price - lower_line[i]) / width_at_bar

            # Touch upper if HIGH is within threshold of upper line (or above it)
            if upper_dist <= thresh:
                touches.append(('upper', i))
            # Touch lower if LOW is within threshold of lower line (or below it)
            elif lower_dist <= thresh:
                touches.append(('lower', i))

        # Count alternating touches (ping-pongs)
        ping_pong_count = 0
        last_touch = None
        for touch_type, idx in touches:
            if last_touch is not None and touch_type != last_touch:
                ping_pong_count += 1
            last_touch = touch_type

        bounces[thresh] = ping_pong_count

        # Count complete cycles (full round-trips)
        cycle_count = 0
        i = 0
        while i < len(touches) - 2:
            t1, t2, t3 = touches[i][0], touches[i+1][0], touches[i+2][0]
            # lower→upper→lower or upper→lower→upper
            if (t1 == 'lower' and t2 == 'upper' and t3 == 'lower') or \
               (t1 == 'upper' and t2 == 'lower' and t3 == 'upper'):
                cycle_count += 1
                i += 2
            else:
                i += 1

        complete_cycles[thresh] = cycle_count
        touch_points[thresh] = touches

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'std_dev': std_dev,
        'center_line': center_line,
        'upper_line': upper_line,
        'lower_line': lower_line,
        'channel_width_pct': channel_width_pct,
        'bounces': bounces,
        'complete_cycles': complete_cycles,
        'touch_points': touch_points,
        'close': close,
        'high': high,
        'low': low,
        'window': window,
    }


def plot_channel(df: pd.DataFrame, end_idx: int, window: int,
                 symbol: str = "TSLA", timeframe: str = "5min",
                 save_path: str = None, show: bool = True) -> dict:
    """
    Plot a channel ending at end_idx with the specified window.

    Returns channel metrics dict.
    """
    start_idx = max(0, end_idx - window)

    # Get data slice
    df_slice = df.iloc[start_idx:end_idx]
    if len(df_slice) < window:
        print(f"  Not enough data: {len(df_slice)} < {window}")
        return None

    # Calculate channel
    channel = calculate_channel(
        df_slice['close'].values,
        df_slice['high'].values,
        df_slice['low'].values,
        window
    )

    if channel is None:
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    timestamps = df_slice.index
    x = np.arange(len(timestamps))

    # Plot candlesticks (simplified as bars)
    colors = ['green' if c >= o else 'red'
              for o, c in zip(df_slice['open'], df_slice['close'])]

    # Plot OHLC as candlesticks
    for i, (ts, row) in enumerate(df_slice.iterrows()):
        color = 'green' if row['close'] >= row['open'] else 'red'
        # Wick
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=0.5)
        # Body
        body_bottom = min(row['open'], row['close'])
        body_height = abs(row['close'] - row['open'])
        rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                         facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    # Plot channel lines
    ax.plot(x, channel['center_line'], 'b-', linewidth=2, label='Center (regression)')
    ax.plot(x, channel['upper_line'], 'r--', linewidth=1.5, label='Upper (2σ)')
    ax.plot(x, channel['lower_line'], 'g--', linewidth=1.5, label='Lower (2σ)')

    # Fill channel
    ax.fill_between(x, channel['lower_line'], channel['upper_line'],
                    alpha=0.1, color='blue')

    # Mark touch points (using 10% of channel width threshold)
    touches = channel['touch_points'].get(0.10, [])
    for touch_type, idx in touches:
        if touch_type == 'upper':
            # Mark at the HIGH of the candle for upper touches
            ax.scatter(idx, channel['high'][idx], color='red', s=100,
                      marker='v', zorder=5, edgecolors='black')
        else:
            # Mark at the LOW of the candle for lower touches
            ax.scatter(idx, channel['low'][idx], color='green', s=100,
                      marker='^', zorder=5, edgecolors='black')

    # Title and labels
    end_time = timestamps[-1]
    r2 = channel['r_squared']
    cycles_10pct = channel['complete_cycles'].get(0.10, 0)
    bounces_10pct = channel['bounces'].get(0.10, 0)
    width_pct = channel['channel_width_pct']
    slope_pct = (channel['slope'] / channel['close'].mean()) * 100

    direction = "BULL" if slope_pct > 0.05 else "BEAR" if slope_pct < -0.05 else "SIDEWAYS"
    valid = "VALID" if cycles_10pct >= 1 else "INVALID"

    title = f"{symbol} {timeframe} | Window: {window} bars | {end_time.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Metrics text box
    metrics_text = (
        f"Direction: {direction}\n"
        f"Channel: {valid}\n"
        f"─────────────\n"
        f"R² = {r2:.3f}\n"
        f"Complete Cycles = {cycles_10pct}\n"
        f"Bounces = {bounces_10pct}\n"
        f"Width = {width_pct:.2f}%\n"
        f"Slope = {slope_pct:.3f}%/bar\n"
        f"─────────────\n"
        f"Cycles (% of width):\n"
        f"  5%:  {channel['complete_cycles'].get(0.05, 0)}\n"
        f"  10%: {channel['complete_cycles'].get(0.10, 0)}\n"
        f"  15%: {channel['complete_cycles'].get(0.15, 0)}\n"
        f"  20%: {channel['complete_cycles'].get(0.20, 0)}"
    )

    # Position text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace', bbox=props)

    # Legend
    ax.legend(loc='upper right')

    # Format x-axis with dates
    ax.set_xlabel('Bar Index')
    ax.set_ylabel('Price ($)')

    # Add grid
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return channel


def generate_random_samples(df: pd.DataFrame, n_samples: int = 5,
                           windows: list = [20, 50, 100],
                           symbol: str = "TSLA") -> list:
    """
    Generate random channel samples for inspection.

    Returns list of (timestamp, window, channel_dict) tuples.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    max_window = max(windows)

    # Get valid range (need enough bars before and after)
    valid_start = max_window + 100
    valid_end = len(df) - 100

    print(f"\nGenerating {n_samples} random samples...")
    print(f"  Windows: {windows}")
    print(f"  Valid index range: {valid_start:,} to {valid_end:,}")

    for i in range(n_samples):
        # Pick random end index
        end_idx = random.randint(valid_start, valid_end)
        timestamp = df.index[end_idx]

        print(f"\n[Sample {i+1}/{n_samples}] {timestamp}")

        for window in windows:
            save_path = OUTPUT_DIR / f"sample_{i+1:02d}_w{window}_{timestamp.strftime('%Y%m%d_%H%M')}.png"

            channel = plot_channel(
                df, end_idx, window,
                symbol=symbol,
                save_path=str(save_path),
                show=False
            )

            if channel:
                results.append({
                    'sample': i + 1,
                    'timestamp': timestamp,
                    'window': window,
                    'r_squared': channel['r_squared'],
                    'cycles_10pct': channel['complete_cycles'].get(0.10, 0),
                    'bounces_10pct': channel['bounces'].get(0.10, 0),
                    'width_pct': channel['channel_width_pct'],
                    'direction': 'BULL' if channel['slope'] > 0 else 'BEAR' if channel['slope'] < 0 else 'SIDEWAYS',
                    'save_path': str(save_path),
                })

    return results


def main():
    parser = argparse.ArgumentParser(description='Channel Inspector - Visual validation')
    parser.add_argument('--samples', type=int, default=5, help='Number of random samples')
    parser.add_argument('--date', type=str, default=None, help='Specific date (YYYY-MM-DD)')
    parser.add_argument('--windows', type=str, default='20,50,100', help='Window sizes (comma-separated)')
    parser.add_argument('--symbol', type=str, default='TSLA', help='Symbol (TSLA or SPY)')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')

    args = parser.parse_args()

    # Parse windows
    windows = [int(w) for w in args.windows.split(',')]

    # Load data
    df = load_and_resample_data(args.symbol)

    if args.date:
        # Find specific date
        target_date = pd.Timestamp(args.date)
        # Find closest bar after target date
        mask = df.index >= target_date
        if not mask.any():
            print(f"No data found for date {args.date}")
            return

        # Get first index where mask is True
        first_valid_idx = mask.argmax()  # Returns index of first True
        end_idx = first_valid_idx + max(windows)  # Get enough bars after
        timestamp = df.index[end_idx]

        print(f"\nPlotting channels for {timestamp}")
        for window in windows:
            save_path = OUTPUT_DIR / f"specific_{args.date}_w{window}.png"
            plot_channel(
                df, end_idx, window,
                symbol=args.symbol,
                save_path=str(save_path),
                show=args.show
            )
    else:
        # Generate random samples
        results = generate_random_samples(
            df,
            n_samples=args.samples,
            windows=windows,
            symbol=args.symbol
        )

        # Print summary
        print(f"\n" + "="*60)
        print(f"SUMMARY: Generated {len(results)} channel images")
        print(f"="*60)
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"\nSample summary:")
        for r in results:
            valid = "VALID" if r['cycles_10pct'] >= 1 else "INVALID"
            print(f"  Sample {r['sample']}, w{r['window']}: {valid} | "
                  f"R²={r['r_squared']:.2f}, cycles={r['cycles_10pct']}, "
                  f"bounces={r['bounces_10pct']}, width={r['width_pct']:.1f}%")


if __name__ == "__main__":
    main()
