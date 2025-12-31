"""
Visual Dashboard for v7 Channel Prediction System (Matplotlib-based)

Creates a matplotlib-based visual dashboard with channel plots and predictions.
Better for static analysis and screenshot export.

Usage:
    python dashboard_visual.py                      # Show dashboard
    python dashboard_visual.py --save output.png    # Save to file
    python dashboard_visual.py --tf 1h 4h daily     # Show specific timeframes
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import torch

# Add v7 to path
sys.path.insert(0, str(Path(__file__).parent / 'v7'))

from v7.core.timeframe import TIMEFRAMES, resample_ohlc
from v7.core.channel import detect_channel, Direction
from v7.features.full_features import extract_full_features, features_to_tensor_dict
from v7.features.events import EventsHandler
from v7.models.hierarchical_cfc import HierarchicalCfCModel, FeatureConfig


# Constants
DATA_DIR = Path(__file__).parent / 'data'
TSLA_CSV = DATA_DIR / 'TSLA_1min.csv'
SPY_CSV = DATA_DIR / 'SPY_1min.csv'
VIX_CSV = DATA_DIR / 'VIX_History.csv'
EVENTS_CSV = DATA_DIR / 'events.csv'

# Colors
DIR_COLORS = {0: '#ff4444', 1: '#ffaa00', 2: '#44ff44'}
DIR_NAMES = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}


def load_data(lookback_days: int = 90):
    """Load data from CSV files."""
    print(f"Loading data (last {lookback_days} days)...")

    # Load TSLA
    tsla = pd.read_csv(TSLA_CSV, parse_dates=['Datetime'])
    tsla.set_index('Datetime', inplace=True)
    tsla.columns = tsla.columns.str.lower()

    # Load SPY
    spy = pd.read_csv(SPY_CSV, parse_dates=['Datetime'])
    spy.set_index('Datetime', inplace=True)
    spy.columns = spy.columns.str.lower()

    # Load VIX
    vix = pd.read_csv(VIX_CSV, parse_dates=['Date'])
    vix.set_index('Date', inplace=True)
    vix.columns = vix.columns.str.lower()

    # Resample to 5min
    tsla_5min = tsla.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    spy_5min = spy.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Filter to lookback
    cutoff = datetime.now() - timedelta(days=lookback_days)
    tsla_5min = tsla_5min[tsla_5min.index >= cutoff]
    spy_5min = spy_5min[spy_5min.index >= cutoff]
    vix = vix[vix.index >= cutoff.date()]

    print(f"  TSLA: {len(tsla_5min)} bars")
    print(f"  SPY:  {len(spy_5min)} bars")
    print(f"  VIX:  {len(vix)} bars")

    return tsla_5min, spy_5min, vix


def plot_channel(ax, df, channel, timeframe, show_touches=True):
    """Plot channel on axis."""
    # Get window data
    window_df = df.iloc[-channel.window:]
    x = np.arange(len(window_df))

    # Plot price
    ax.plot(x, channel.close, 'k-', linewidth=1, alpha=0.7, label='Close')

    # Plot channel lines
    color = DIR_COLORS[channel.direction]
    ax.plot(x, channel.center_line, '--', color=color, linewidth=1.5, alpha=0.8, label='Center')
    ax.plot(x, channel.upper_line, '-', color=color, linewidth=2, alpha=0.9, label='Upper')
    ax.plot(x, channel.lower_line, '-', color=color, linewidth=2, alpha=0.9, label='Lower')

    # Fill channel
    ax.fill_between(x, channel.lower_line, channel.upper_line, color=color, alpha=0.1)

    # Mark touches
    if show_touches and channel.touches:
        for touch in channel.touches:
            marker = '^' if touch.touch_type == 0 else 'v'  # Lower=up, Upper=down
            touch_color = '#00ff00' if touch.touch_type == 0 else '#ff0000'
            ax.plot(touch.bar_index, touch.price, marker, color=touch_color,
                   markersize=8, alpha=0.8)

    # Current position marker
    current_pos = channel.position_at()
    ax.axhline(channel.close[-1], color='blue', linestyle=':', alpha=0.5)
    ax.text(0.02, 0.98, f'Position: {current_pos:.2f}',
           transform=ax.transAxes, va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Title
    dir_name = DIR_NAMES[channel.direction]
    valid_str = "VALID" if channel.valid else "INVALID"
    ax.set_title(f'{timeframe} - {dir_name} ({valid_str})\n'
                f'Bounces: {channel.bounce_count} | Cycles: {channel.complete_cycles} | '
                f'R²: {channel.r_squared:.3f}',
                fontsize=10, fontweight='bold')

    ax.set_xlabel('Bars')
    ax.set_ylabel('Price')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def create_dashboard(
    tsla_df,
    spy_df,
    vix_df,
    timeframes_to_show=None,
    model=None
):
    """Create visual dashboard."""
    if timeframes_to_show is None:
        timeframes_to_show = ['5min', '15min', '1h', '4h', 'daily']

    print("\nDetecting channels...")
    channels = {}
    for tf in timeframes_to_show:
        if tf == '5min':
            df_tf = tsla_df
        else:
            df_tf = resample_ohlc(tsla_df, tf)

        if len(df_tf) >= 50:
            channels[tf] = detect_channel(df_tf, window=50)
            print(f"  {tf}: {channels[tf].direction.name} "
                  f"({channels[tf].bounce_count} bounces, "
                  f"{channels[tf].complete_cycles} cycles)")

    # Make predictions
    predictions = None
    if model is not None:
        print("\nMaking predictions...")
        full_features = extract_full_features(tsla_df, spy_df, vix_df, window=50, include_history=False)
        feature_arrays = features_to_tensor_dict(full_features)

        # Concatenate features
        feature_list = []
        for tf in TIMEFRAMES:
            if f'tsla_{tf}' in feature_arrays:
                feature_list.append(feature_arrays[f'tsla_{tf}'])
        for tf in TIMEFRAMES:
            if f'spy_{tf}' in feature_arrays:
                feature_list.append(feature_arrays[f'spy_{tf}'])
        for tf in TIMEFRAMES:
            if f'cross_{tf}' in feature_arrays:
                feature_list.append(feature_arrays[f'cross_{tf}'])

        feature_list.extend([
            feature_arrays['vix'],
            feature_arrays['tsla_history'],
            feature_arrays['spy_history'],
            feature_arrays['alignment']
        ])

        x = torch.from_numpy(np.concatenate(feature_list)).float().unsqueeze(0)

        with torch.no_grad():
            predictions = model.predict(x)

    # Create figure
    n_plots = len(timeframes_to_show)
    fig = plt.figure(figsize=(20, 4 * n_plots))

    if predictions is not None:
        gs = gridspec.GridSpec(n_plots + 1, 2, height_ratios=[1] + [3] * n_plots, hspace=0.3, wspace=0.3)
    else:
        gs = gridspec.GridSpec(n_plots, 2, hspace=0.3, wspace=0.3)

    # Header with predictions
    if predictions is not None:
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')

        # Extract prediction values
        dur_mean = float(predictions['duration_mean'][0, 0])
        dur_std = float(predictions['duration_std'][0, 0])
        break_dir = int(predictions['break_direction'][0, 0])
        break_probs = predictions['break_direction_probs'][0].numpy()
        next_dir = int(predictions['next_direction'][0, 0])
        next_probs = predictions['next_direction_probs'][0].numpy()
        conf = float(predictions['confidence'][0, 0])

        # Trading signal
        if conf > 0.75:
            signal = "LONG" if break_dir == 1 else "SHORT"
            signal_color = '#44ff44' if break_dir == 1 else '#ff4444'
        elif conf > 0.60:
            signal = "CAUTIOUS"
            signal_color = '#ffaa00'
        else:
            signal = "WAIT"
            signal_color = '#888888'

        header_text = (
            f"TSLA CHANNEL PREDICTION DASHBOARD\n"
            f"Time: {tsla_df.index[-1].strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Price: ${tsla_df['close'].iloc[-1]:.2f} | "
            f"SPY: ${spy_df['close'].iloc[-1]:.2f} | "
            f"VIX: {vix_df['close'].iloc[-1]:.2f}\n\n"
            f"SIGNAL: {signal} (Confidence: {conf*100:.0f}%) | "
            f"Duration: {dur_mean:.0f}±{dur_std:.0f} bars | "
            f"Break: {'UP' if break_dir == 1 else 'DOWN'} ({break_probs[break_dir]*100:.0f}%) | "
            f"Next: {DIR_NAMES[next_dir]} ({next_probs[next_dir]*100:.0f}%)"
        )

        ax_header.text(0.5, 0.5, header_text, ha='center', va='center',
                      fontsize=14, fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor=signal_color, alpha=0.3))

        start_row = 1
    else:
        start_row = 0

    # Plot channels
    for i, tf in enumerate(timeframes_to_show):
        if tf not in channels:
            continue

        # Get data
        if tf == '5min':
            df_tf = tsla_df
        else:
            df_tf = resample_ohlc(tsla_df, tf)

        channel = channels[tf]

        # Plot on left column
        ax = fig.add_subplot(gs[start_row + i, :])
        plot_channel(ax, df_tf, channel, tf)

    return fig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='v7 Visual Dashboard')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--save', type=str, help='Save to file instead of showing')
    parser.add_argument('--tf', nargs='+', default=['5min', '15min', '1h', '4h', 'daily'],
                       help='Timeframes to show')
    parser.add_argument('--lookback', type=int, default=90, help='Days of data to load')
    args = parser.parse_args()

    # Load model
    model = None
    if args.model and Path(args.model).exists():
        print(f"Loading model from {args.model}...")
        model = HierarchicalCfCModel(feature_config=FeatureConfig())
        checkpoint = torch.load(args.model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully")

    # Load data
    tsla_df, spy_df, vix_df = load_data(args.lookback)

    # Create dashboard
    fig = create_dashboard(tsla_df, spy_df, vix_df, args.tf, model)

    # Save or show
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"\nDashboard saved to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
