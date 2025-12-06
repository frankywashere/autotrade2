"""
Live Channel Visualizer for AutoTrade v5.0

Visualize channels from live data buffer with:
- Current market state visualization
- Channel quality for all 11 timeframes
- Show which TF the model selected
- Compare multiple TFs side-by-side
- Interactive window selection

Usage:
    python tools/visualize_live_channels.py

    Or from within Python:
    from tools.visualize_live_channels import visualize_current_channels
    visualize_current_channels(predictor)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from predict import LivePredictor
from scipy import stats as scipy_stats


def calculate_channel(df: pd.DataFrame, symbol: str = 'tsla', window: int = 100) -> Dict:
    """
    Calculate linear regression channel for a symbol.

    Args:
        df: DataFrame with OHLC data (spy_*/tsla_* columns)
        symbol: 'spy' or 'tsla'
        window: Lookback window for regression

    Returns:
        Dict with channel metrics and lines
    """
    col_prefix = f'{symbol}_'

    # Take last window bars
    if len(df) < window:
        window = len(df)

    window_df = df.tail(window).copy()

    # Extract close prices
    close_col = f'{col_prefix}close'
    if close_col not in window_df.columns:
        raise ValueError(f"Column {close_col} not found")

    closes = window_df[close_col].values
    x = np.arange(len(closes))

    # Linear regression on close prices
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, closes)
    r_squared = r_value ** 2

    # Calculate channel lines
    center_line = slope * x + intercept
    residuals = closes - center_line
    residual_std = np.std(residuals)

    upper_line = center_line + (2.0 * residual_std)
    lower_line = center_line - (2.0 * residual_std)

    # Current price and position
    current_price = closes[-1]
    channel_width = upper_line[-1] - lower_line[-1]
    position = (current_price - lower_line[-1]) / channel_width if channel_width > 0 else 0.5
    position = np.clip(position, 0, 1)

    # Slope percentage (per bar)
    slope_pct = (slope / current_price * 100) if current_price > 0 else 0

    # Detect touches (within 2% of boundary)
    upper_touches = []
    lower_touches = []
    for i in range(len(closes)):
        upper_dist = abs(closes[i] - upper_line[i]) / upper_line[i] if upper_line[i] > 0 else 1.0
        lower_dist = abs(closes[i] - lower_line[i]) / abs(lower_line[i]) if lower_line[i] != 0 else 1.0

        if upper_dist <= 0.02:
            upper_touches.append(i)
        elif lower_dist <= 0.02:
            lower_touches.append(i)

    # Complete cycles (simplified: upper touch followed by lower touch)
    complete_cycles = 0
    last_was_upper = False
    for i in range(len(closes)):
        if i in upper_touches:
            last_was_upper = True
        elif i in lower_touches and last_was_upper:
            complete_cycles += 1
            last_was_upper = False

    # Project forward (24 bars as default horizon)
    forecast_horizon = 24
    future_x = np.arange(len(closes), len(closes) + forecast_horizon)
    future_center = slope * future_x + intercept
    future_upper = future_center + (2.0 * residual_std)
    future_lower = future_center - (2.0 * residual_std)

    # Projected high/low as percentage change
    projected_high = np.max(future_upper)
    projected_low = np.min(future_lower)
    projected_high_pct = (projected_high - current_price) / current_price * 100
    projected_low_pct = (projected_low - current_price) / current_price * 100

    # Quality score (simplified)
    quality = r_squared * (1.0 if complete_cycles >= 2 else 0.5)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'residual_std': residual_std,
        'center_line': center_line,
        'upper_line': upper_line,
        'lower_line': lower_line,
        'current_price': current_price,
        'position': position,
        'slope_pct': slope_pct,
        'channel_width_pct': (channel_width / current_price * 100) if current_price > 0 else 0,
        'upper_touches': upper_touches,
        'lower_touches': lower_touches,
        'complete_cycles': complete_cycles,
        'quality': quality,
        'projected_high': projected_high,
        'projected_low': projected_low,
        'projected_high_pct': projected_high_pct,
        'projected_low_pct': projected_low_pct,
        'future_x': future_x,
        'future_upper': future_upper,
        'future_lower': future_lower,
        'future_center': future_center,
        'window_df': window_df,
    }


def plot_channel(
    channel: Dict,
    symbol: str,
    timeframe: str,
    window: int,
    show_projection: bool = True,
    show_touches: bool = True,
    is_selected: bool = False
):
    """
    Plot channel with current state and projection.

    Args:
        channel: Channel metrics dict from calculate_channel()
        symbol: Stock symbol
        timeframe: Timeframe name
        window: Window size
        show_projection: Show 24-bar forward projection
        show_touches: Mark touch points
        is_selected: Whether this TF was selected by the model
    """
    window_df = channel['window_df']

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Plot price
    ax.plot(window_df.index, window_df[f'{symbol}_close'],
            label='Price', linewidth=2, color='black', zorder=3)

    # Plot channel lines
    ax.plot(window_df.index, channel['upper_line'],
            'r--', label='Upper (2σ)', alpha=0.7, linewidth=1.5)
    ax.plot(window_df.index, channel['lower_line'],
            'g--', label='Lower (2σ)', alpha=0.7, linewidth=1.5)
    ax.plot(window_df.index, channel['center_line'],
            'b-', label='Center (regression)', alpha=0.5, linewidth=1)

    # Mark current position
    current_price = channel['current_price']
    ax.axhline(current_price, color='purple', linestyle=':', alpha=0.6, linewidth=2, label='Current Price')

    # Show touches
    if show_touches:
        if channel['upper_touches']:
            ax.scatter(window_df.index[channel['upper_touches']],
                      window_df[f'{symbol}_close'].iloc[channel['upper_touches']],
                      color='red', s=100, zorder=5, marker='o', alpha=0.7,
                      label=f'Upper touches ({len(channel["upper_touches"])})')

        if channel['lower_touches']:
            ax.scatter(window_df.index[channel['lower_touches']],
                      window_df[f'{symbol}_close'].iloc[channel['lower_touches']],
                      color='green', s=100, zorder=5, marker='o', alpha=0.7,
                      label=f'Lower touches ({len(channel["lower_touches"])})')

    # Show projection
    if show_projection:
        # Create future timestamps (approximate)
        last_ts = window_df.index[-1]
        freq = pd.infer_freq(window_df.index)
        if freq:
            future_dates = pd.date_range(last_ts, periods=len(channel['future_x'])+1, freq=freq)[1:]
        else:
            # Fallback: extend x-axis numerically
            future_dates = range(len(window_df), len(window_df) + len(channel['future_x']))

        ax.plot(future_dates, channel['future_upper'],
                'r:', label='Projected Upper', alpha=0.5, linewidth=2)
        ax.plot(future_dates, channel['future_lower'],
                'g:', label='Projected Lower', alpha=0.5, linewidth=2)
        ax.plot(future_dates, channel['future_center'],
                'b:', label='Projected Center', alpha=0.3, linewidth=1)

        # Mark projected high/low
        ax.axhline(channel['projected_high'], color='red', linestyle='--', alpha=0.3)
        ax.axhline(channel['projected_low'], color='green', linestyle='--', alpha=0.3)

    # Title with selection indicator
    title_prefix = "⭐ SELECTED: " if is_selected else ""
    ax.set_title(
        f"{title_prefix}{symbol.upper()} - {timeframe.upper()} - Window {window} bars\n"
        f"R²={channel['r_squared']:.3f} | Position={channel['position']:.2f} | "
        f"Cycles={channel['complete_cycles']:.0f} | Quality={channel['quality']:.3f}\n"
        f"Slope={channel['slope_pct']:.3f}%/bar | Projection: [{channel['projected_low_pct']:.2f}%, {channel['projected_high_pct']:.2f}%]",
        fontsize=12, fontweight='bold'
    )

    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add metrics box
    metrics_text = (
        f"Current: ${current_price:.2f}\n"
        f"Position: {channel['position']:.2%}\n"
        f"Slope: {channel['slope_pct']:.3f}%/bar\n"
        f"Width: {channel['channel_width_pct']:.2f}%\n"
        f"Complete Cycles: {channel['complete_cycles']:.0f}\n"
        f"R²: {channel['r_squared']:.3f}\n"
        f"Quality: {channel['quality']:.3f}\n"
        f"\nProjected (24 bars):\n"
        f"High: {channel['projected_high_pct']:+.2f}%\n"
        f"Low: {channel['projected_low_pct']:+.2f}%"
    )

    ax.text(0.02, 0.98, metrics_text,
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top', family='monospace')

    plt.tight_layout()
    return fig


def visualize_all_timeframes(
    predictor: 'LivePredictor',
    symbol: str = 'tsla',
    window: int = 100,
    prediction_result: Dict = None
):
    """
    Show all 11 timeframe channels in a grid.

    Args:
        predictor: LivePredictor instance with loaded data
        symbol: 'spy' or 'tsla'
        window: Window size to use for all TFs
        prediction_result: Optional prediction result to show selected TF
    """
    timeframes = ['5min', '15min', '30min', '1hour', '2h', '3h', '4h',
                  'daily', 'weekly', 'monthly', '3month']

    selected_tf = prediction_result.get('selected_tf') if prediction_result else None

    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    axes = axes.flatten()

    for idx, tf in enumerate(timeframes):
        ax = axes[idx]

        # Get data for this TF
        tf_df = predictor.data_buffer.buffers.get(tf)

        if tf_df is None or len(tf_df) < 10:
            ax.text(0.5, 0.5, f'{tf.upper()}\nInsufficient Data',
                   ha='center', va='center', fontsize=14)
            ax.set_title(f"{tf.upper()}")
            continue

        # Calculate channel
        actual_window = min(window, len(tf_df))
        try:
            channel = calculate_channel(tf_df, symbol, actual_window)
        except Exception as e:
            ax.text(0.5, 0.5, f'{tf.upper()}\nError: {e}',
                   ha='center', va='center', fontsize=10)
            ax.set_title(f"{tf.upper()}")
            continue

        # Plot price and channel
        window_df = channel['window_df']
        ax.plot(range(len(window_df)), window_df[f'{symbol}_close'],
               label='Price', linewidth=1.5, color='black')
        ax.plot(range(len(window_df)), channel['upper_line'],
               'r--', alpha=0.6, linewidth=1)
        ax.plot(range(len(window_df)), channel['lower_line'],
               'g--', alpha=0.6, linewidth=1)
        ax.plot(range(len(window_df)), channel['center_line'],
               'b-', alpha=0.4, linewidth=0.8)

        # Current price marker
        ax.axhline(channel['current_price'], color='purple',
                  linestyle=':', alpha=0.5, linewidth=1.5)

        # Mark if selected
        title_prefix = "⭐ " if tf == selected_tf else ""
        bgcolor = 'yellow' if tf == selected_tf else 'white'

        ax.set_title(
            f"{title_prefix}{tf.upper()}\n"
            f"R²={channel['r_squared']:.2f} Pos={channel['position']:.2f} Cyc={channel['complete_cycles']:.0f}\n"
            f"Proj: [{channel['projected_low_pct']:+.1f}%, {channel['projected_high_pct']:+.1f}%]",
            fontsize=9, fontweight='bold' if tf == selected_tf else 'normal',
            bbox=dict(boxstyle='round', facecolor=bgcolor, alpha=0.3) if tf == selected_tf else None
        )

        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    # Hide unused subplot
    axes[-1].axis('off')

    plt.suptitle(
        f"{symbol.upper()} - All Timeframe Channels (window={window})\n"
        f"{'Selected: ' + selected_tf.upper() if selected_tf else 'No selection info'}",
        fontsize=16, fontweight='bold'
    )

    plt.tight_layout()
    return fig


def visualize_current_channels(
    predictor: 'LivePredictor',
    symbol: str = 'tsla',
    timeframes: List[str] = None,
    window: int = 100,
    prediction_result: Dict = None,
    show_projection: bool = True
):
    """
    Visualize current channel state for specified timeframes.

    Args:
        predictor: LivePredictor instance with loaded data
        symbol: 'spy' or 'tsla'
        timeframes: List of TFs to plot (None = top 3 by confidence)
        window: Window size for regression
        prediction_result: Optional prediction result from predictor.predict()
        show_projection: Show 24-bar forward projection (default: True)
    """
    # If prediction result given and no specific TFs, show top 3
    if timeframes is None and prediction_result and 'all_channels' in prediction_result:
        top_channels = prediction_result['all_channels'][:3]
        timeframes = [ch['timeframe'] for ch in top_channels]
        print(f"Showing top 3 channels by confidence: {timeframes}")
    elif timeframes is None:
        timeframes = ['5min', '30min', '1hour', 'daily']

    # Create subplot for each TF
    n_plots = len(timeframes)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 5*n_plots))

    if n_plots == 1:
        axes = [axes]

    selected_tf = prediction_result.get('selected_tf') if prediction_result else None

    for idx, tf in enumerate(timeframes):
        ax = axes[idx]

        # Get data for this TF
        tf_df = predictor.data_buffer.buffers.get(tf)

        if tf_df is None or len(tf_df) < 10:
            ax.text(0.5, 0.5, f'{tf.upper()}\nInsufficient Data ({len(tf_df) if tf_df is not None else 0} bars)',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title(f"{tf.upper()}")
            continue

        # Calculate channel
        actual_window = min(window, len(tf_df))
        channel = calculate_channel(tf_df, symbol, actual_window)
        window_df = channel['window_df']

        # Plot price
        ax.plot(window_df.index, window_df[f'{symbol}_close'],
               label='Price', linewidth=2, color='black', zorder=3)

        # Plot channel
        ax.plot(window_df.index, channel['upper_line'],
               'r--', label='Upper (2σ)', alpha=0.7, linewidth=1.5)
        ax.plot(window_df.index, channel['lower_line'],
               'g--', label='Lower (2σ)', alpha=0.7, linewidth=1.5)
        ax.plot(window_df.index, channel['center_line'],
               'b-', label='Center', alpha=0.5, linewidth=1)

        # Current price
        ax.axhline(channel['current_price'], color='purple',
                  linestyle=':', alpha=0.6, linewidth=2, label='Current')

        # Touches
        if channel['upper_touches']:
            ax.scatter(window_df.index[channel['upper_touches']],
                      window_df[f'{symbol}_close'].iloc[channel['upper_touches']],
                      color='red', s=100, zorder=5, marker='o', alpha=0.7)

        if channel['lower_touches']:
            ax.scatter(window_df.index[channel['lower_touches']],
                      window_df[f'{symbol}_close'].iloc[channel['lower_touches']],
                      color='green', s=100, zorder=5, marker='o', alpha=0.7)

        # Show projection
        if show_projection:
            last_ts = window_df.index[-1]
            freq = pd.infer_freq(window_df.index)
            if freq:
                future_dates = pd.date_range(last_ts, periods=len(channel['future_x'])+1, freq=freq)[1:]
                ax.plot(future_dates, channel['future_upper'],
                       'r:', label='Proj Upper', alpha=0.5, linewidth=2)
                ax.plot(future_dates, channel['future_lower'],
                       'g:', label='Proj Lower', alpha=0.5, linewidth=2)
                ax.plot(future_dates, channel['future_center'],
                       'b:', alpha=0.3, linewidth=1)

                ax.axhline(channel['projected_high'], color='red', linestyle='--', alpha=0.2)
                ax.axhline(channel['projected_low'], color='green', linestyle='--', alpha=0.2)

        # Get confidence from prediction result
        confidence = None
        if prediction_result and 'all_channels' in prediction_result:
            for ch in prediction_result['all_channels']:
                if ch['timeframe'] == tf:
                    confidence = ch['confidence']
                    break

        # Title
        title_prefix = "⭐ SELECTED: " if tf == selected_tf else ""
        conf_text = f" | Model Confidence: {confidence:.3f}" if confidence is not None else ""

        ax.set_title(
            f"{title_prefix}{symbol.upper()} - {tf.upper()} - Window {actual_window} bars{conf_text}\n"
            f"R²={channel['r_squared']:.3f} | Position={channel['position']:.2f} | "
            f"Cycles={channel['complete_cycles']:.0f} | Quality={channel['quality']:.3f} | "
            f"Slope={channel['slope_pct']:.3f}%/bar\n"
            f"Projected (24 bars): High={channel['projected_high_pct']:+.2f}%, Low={channel['projected_low_pct']:+.2f}%",
            fontsize=11, fontweight='bold' if tf == selected_tf else 'normal'
        )

        ax.set_ylabel('Price ($)', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def print_channel_summary(predictor: 'LivePredictor', symbol: str = 'tsla', window: int = 100):
    """
    Print text summary of all channels.

    Args:
        predictor: LivePredictor instance
        symbol: Stock symbol
        window: Window size
    """
    timeframes = ['5min', '15min', '30min', '1hour', '2h', '3h', '4h',
                  'daily', 'weekly', 'monthly', '3month']

    print(f"\n{'='*80}")
    print(f"CHANNEL SUMMARY - {symbol.upper()} (window={window})")
    print(f"{'='*80}\n")

    print(f"{'TF':<8} {'Bars':<6} {'R²':<6} {'Pos':<6} {'Cyc':<5} {'Qual':<6} {'Slope%':<8} {'Proj High':<10} {'Proj Low':<10}")
    print("-" * 80)

    for tf in timeframes:
        tf_df = predictor.data_buffer.buffers.get(tf)

        if tf_df is None or len(tf_df) < 10:
            print(f"{tf:<8} {'N/A':<6} {'N/A':<6} {'N/A':<6} {'N/A':<5} {'N/A':<6} {'N/A':<8} {'N/A':<10} {'N/A':<10}")
            continue

        actual_window = min(window, len(tf_df))
        try:
            channel = calculate_channel(tf_df, symbol, actual_window)
            print(f"{tf:<8} {len(tf_df):<6} {channel['r_squared']:<6.3f} {channel['position']:<6.2f} "
                  f"{channel['complete_cycles']:<5.0f} {channel['quality']:<6.3f} {channel['slope_pct']:<8.3f} "
                  f"{channel['projected_high_pct']:+9.2f}% {channel['projected_low_pct']:+9.2f}%")
        except Exception as e:
            print(f"{tf:<8} Error: {e}")

    print()


def main():
    """Interactive live channel visualization."""
    print("\n" + "="*80)
    print("LIVE CHANNEL VISUALIZER v5.0")
    print("="*80)

    # Load predictor
    print("\n1. Loading live predictor...")
    try:
        predictor = LivePredictor('models/hierarchical_lnn.pth', device='cpu')
    except Exception as e:
        print(f"❌ Failed to load predictor: {e}")
        print("\nMake sure you have:")
        print("  • models/hierarchical_lnn.pth")
        print("  • data/feature_cache/tf_meta_*.json")
        print("  • data/VIX_History.csv")
        return

    # Fetch live data
    print("\n2. Fetching live data...")
    try:
        predictor.fetch_live_data(intraday_days=60, daily_days=400)
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return

    # Make prediction
    print("\n3. Making prediction...")
    try:
        result = predictor.predict()
        print(f"\n✓ Prediction complete:")
        print(f"   Selected TF: {result.get('selected_tf', 'unknown')}")
        print(f"   Predicted High: {result['predicted_high']:+.2f}%")
        print(f"   Predicted Low: {result['predicted_low']:+.2f}%")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        result = None

    # Print channel summary
    print("\n4. Channel Analysis...")
    print_channel_summary(predictor, symbol='tsla', window=100)

    # Ask what to visualize
    print("\n5. Visualization options:")
    print("   1. All 11 timeframes (grid view)")
    print("   2. Top 3 confident channels (detailed)")
    print("   3. Specific timeframe (choose)")
    print("   4. Exit")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == '1':
        print("\nGenerating all timeframes grid...")
        fig = visualize_all_timeframes(predictor, symbol='tsla', window=100, prediction_result=result)
        plt.show()

    elif choice == '2':
        print("\nGenerating top 3 channels...")
        fig = visualize_current_channels(predictor, symbol='tsla', timeframes=None,
                                        window=100, prediction_result=result)
        plt.show()

    elif choice == '3':
        print("\nAvailable timeframes: 5min, 15min, 30min, 1hour, 2h, 3h, 4h, daily, weekly, monthly, 3month")
        tf = input("Enter timeframe: ").strip()
        if tf in predictor.data_buffer.buffers:
            channel = calculate_channel(predictor.data_buffer.buffers[tf], 'tsla', 100)
            fig = plot_channel(channel, 'tsla', tf, 100,
                             is_selected=(tf == result.get('selected_tf') if result else False))
            plt.show()
        else:
            print(f"❌ Timeframe {tf} not found in buffer")

    else:
        print("Exiting...")


if __name__ == '__main__':
    main()
