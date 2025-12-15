"""
Interactive Channel Visualizer for AutoTrade v3.17

Visualize channels from cached mmap shards with:
- Interactive shard location selection (local or external drive)
- Channel quality metrics (ping_pongs vs complete_cycles)
- Touch point visualization
- Multi-window comparison
- Browse by quality filters
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from tools.channel_loader import ChannelLoader
import config

try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice
    INQUIRER_AVAILABLE = True
except ImportError:
    print("⚠️  InquirerPy not available. Install with: pip install InquirerPy")
    INQUIRER_AVAILABLE = False


def select_shard_location() -> Path:
    """Interactive shard location selection with memory."""
    config_file = Path('.visualizer_config.json')

    # Load saved preference if exists
    saved_path = None
    if config_file.exists():
        try:
            cfg = json.load(open(config_file))
            saved_path = cfg.get('shard_path')
        except:
            pass

    # Default location
    default_path = Path('data/feature_cache')

    # Build choice list
    choices = []

    if default_path.exists() and any(default_path.glob('features_mmap_meta_*.json')):
        choices.append(Choice(str(default_path), f"📁 Default - {default_path}"))

    if saved_path and Path(saved_path).exists():
        if str(saved_path) != str(default_path):
            choices.append(Choice(saved_path, f"💾 Last used - {saved_path}"))

    choices.append(Choice('custom', "🔧 Custom path (enter manually)"))

    if not INQUIRER_AVAILABLE:
        print("\n📂 Shard Location Selection:")
        print(f"  1. Default: {default_path}")
        if saved_path:
            print(f"  2. Last used: {saved_path}")
        print(f"  {len(choices)}. Custom path")

        choice = input("Select option: ").strip()
        if choice == '1':
            return default_path
        elif choice == '2' and saved_path:
            return Path(saved_path)
        else:
            path = input("Enter shard path: ").strip()
            json.dump({'shard_path': path}, open(config_file, 'w'))
            return Path(path)

    # Use InquirerPy for nice menu
    selected = inquirer.select(
        message="📂 Select shard storage location:",
        choices=choices
    ).execute()

    if selected == 'custom':
        path = inquirer.text(
            message="Enter shard path:",
            default="/Volumes/NVME2/featureslabels"
        ).execute()

        # Save for next time
        json.dump({'shard_path': path}, open(config_file, 'w'))

        return Path(path)

    return Path(selected)


def reconstruct_channel_lines(metrics: Dict, window_df: pd.DataFrame) -> Dict:
    """
    Reconstruct channel regression lines from metrics.

    Args:
        metrics: Channel metrics dict from loader
        window_df: Raw OHLC window data

    Returns:
        Dict with upper_line, lower_line, center_line arrays
    """
    n = len(window_df)
    x = np.arange(n)

    # Use slope and current position to reconstruct
    # This is approximate - we don't store intercept in features!
    # We'll use the regression ourselves

    closes = window_df['close'].values

    # Simple linear regression
    from scipy import stats
    slope, intercept, r_value, _, _ = stats.linregress(x, closes)

    center_line = slope * x + intercept

    # Estimate std from r_squared and data
    residuals = closes - center_line
    residual_std = np.std(residuals)

    # Channel boundaries (2 std devs by default)
    upper_line = center_line + (2.0 * residual_std)
    lower_line = center_line - (2.0 * residual_std)

    return {
        'center_line': center_line,
        'upper_line': upper_line,
        'lower_line': lower_line,
        'slope': slope,
        'intercept': intercept,
        'residual_std': residual_std
    }


def detect_touch_points(prices: np.ndarray, upper: np.ndarray, lower: np.ndarray,
                        threshold: float = 0.02) -> Tuple[list, list]:
    """
    Detect which bars touched upper or lower channel.

    Args:
        prices: Price array
        upper: Upper channel line
        lower: Lower channel line
        threshold: Touch threshold (2% default)

    Returns:
        (upper_touches, lower_touches) - lists of indices
    """
    upper_touches = []
    lower_touches = []

    for i in range(len(prices)):
        upper_dist = abs(prices[i] - upper[i]) / upper[i] if upper[i] > 0 else 1.0
        lower_dist = abs(prices[i] - lower[i]) / abs(lower[i]) if lower[i] != 0 else 1.0

        if upper_dist <= threshold:
            upper_touches.append(i)
        elif lower_dist <= threshold:
            lower_touches.append(i)

    return upper_touches, lower_touches


def plot_channel(
    timestamp: pd.Timestamp,
    symbol: str,
    timeframe: str,
    window: int,
    metrics: Dict,
    window_df: pd.DataFrame,
    show_touches: bool = True,
    show_cycles: bool = True,
    continuation_label: Dict = None,
    transition_label: Dict = None,
    future_df: pd.DataFrame = None
):
    """
    Plot channel with price, regression lines, metrics, and labels.

    Args:
        timestamp: Timestamp of channel
        symbol: Stock symbol
        timeframe: Timeframe
        window: Window size
        metrics: Channel metrics dict
        window_df: Raw OHLC dataframe
        show_touches: Mark touch points
        show_cycles: Highlight complete cycles
        continuation_label: Dict with 'duration_bars', 'max_gain_pct'
        transition_label: Dict with 'transition_type', 'transition_name', 'new_direction'
        future_df: DataFrame with price data AFTER channel (to verify break)
    """
    # Reconstruct channel lines
    lines = reconstruct_channel_lines(metrics, window_df)

    # Determine figure size based on whether we have labels
    has_labels = continuation_label is not None or transition_label is not None
    height_ratios = [3, 1, 0.8] if has_labels else [3, 1]
    n_panels = 3 if has_labels else 2

    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 12 if has_labels else 10),
                             gridspec_kw={'height_ratios': height_ratios})

    if has_labels:
        ax_price, ax_metrics, ax_labels = axes
    else:
        ax_price, ax_metrics = axes
        ax_labels = None

    # =========================================================================
    # TOP PANEL: PRICE + CHANNEL + BREAK POINT
    # =========================================================================

    # Plot channel window
    ax_price.plot(window_df.index, window_df['close'], label='Price', linewidth=2, color='black')
    ax_price.plot(window_df.index, lines['upper_line'], 'r--', label='Upper', alpha=0.7, linewidth=1.5)
    ax_price.plot(window_df.index, lines['lower_line'], 'g--', label='Lower', alpha=0.7, linewidth=1.5)
    ax_price.plot(window_df.index, lines['center_line'], 'b-', label='Center', alpha=0.5, linewidth=1)

    # If we have future data, extend the chart
    if future_df is not None and len(future_df) > 0:
        # Extend channel lines into future
        n_future = len(future_df)
        future_x = np.arange(len(window_df), len(window_df) + n_future)

        slope = lines['slope']
        intercept = lines['intercept']
        residual_std = lines['residual_std']

        future_center = slope * future_x + intercept
        future_upper = future_center + (2.0 * residual_std)
        future_lower = future_center - (2.0 * residual_std)

        # Plot future price (different color)
        ax_price.plot(future_df.index, future_df['close'], label='Future Price',
                     linewidth=2, color='purple', alpha=0.7)

        # Extend channel lines (dashed)
        ax_price.plot(future_df.index, future_upper[:len(future_df)], 'r:', alpha=0.5, linewidth=1)
        ax_price.plot(future_df.index, future_lower[:len(future_df)], 'g:', alpha=0.5, linewidth=1)

        # Mark break point if we have duration label
        if continuation_label and 'duration_bars' in continuation_label:
            duration = continuation_label['duration_bars']
            if duration > 0 and duration < len(window_df) + len(future_df):
                # Find the break point
                if duration < len(window_df):
                    break_idx = duration
                    break_time = window_df.index[break_idx]
                    break_price = window_df['close'].iloc[break_idx]
                else:
                    future_idx = duration - len(window_df)
                    if future_idx < len(future_df):
                        break_time = future_df.index[future_idx]
                        break_price = future_df['close'].iloc[future_idx]
                    else:
                        break_time = None
                        break_price = None

                if break_time is not None:
                    # Draw vertical line at break
                    ax_price.axvline(x=break_time, color='orange', linestyle='--',
                                    linewidth=2, alpha=0.8, label=f'Break @ bar {duration}')
                    ax_price.scatter([break_time], [break_price], color='orange',
                                    s=150, zorder=10, marker='X', edgecolors='black')

    # Mark touches if requested
    if show_touches:
        upper_touches, lower_touches = detect_touch_points(
            window_df['close'].values,
            lines['upper_line'],
            lines['lower_line'],
            threshold=0.02
        )

        if upper_touches:
            ax_price.scatter(window_df.index[upper_touches],
                           window_df['close'].iloc[upper_touches],
                           color='red', s=80, zorder=5, marker='o', alpha=0.7,
                           label=f'Upper touches ({len(upper_touches)})')

        if lower_touches:
            ax_price.scatter(window_df.index[lower_touches],
                           window_df['close'].iloc[lower_touches],
                           color='green', s=80, zorder=5, marker='o', alpha=0.7,
                           label=f'Lower touches ({len(lower_touches)})')

    # Add transition type indicator
    if transition_label and 'transition_name' in transition_label:
        trans_name = transition_label['transition_name']
        trans_colors = {'CONTINUE': 'green', 'SWITCH_TF': 'blue', 'REVERSE': 'red', 'SIDEWAYS': 'gray'}
        trans_color = trans_colors.get(trans_name, 'black')

        ax_price.text(0.98, 0.98, f"⚡ {trans_name}",
                     transform=ax_price.transAxes, fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor=trans_color, alpha=0.3),
                     verticalalignment='top', horizontalalignment='right')

    # Complete cycles indicator
    if show_cycles and metrics.get('complete_cycles', 0) > 0:
        ax_price.text(0.02, 0.98, f"Cycles: {metrics['complete_cycles']:.0f}",
                     transform=ax_price.transAxes, fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                     verticalalignment='top')

    ax_price.set_title(
        f"{symbol.upper()} - {timeframe} - Window {window} bars - {timestamp.strftime('%Y-%m-%d %H:%M')}",
        fontsize=14, fontweight='bold'
    )
    ax_price.set_ylabel('Price ($)', fontsize=12)
    ax_price.legend(loc='upper left', fontsize=9)
    ax_price.grid(True, alpha=0.3)

    # =========================================================================
    # MIDDLE PANEL: CHANNEL METRICS
    # =========================================================================
    ax_metrics.axis('off')

    ratio_val = metrics.get('ping_pongs', 0)/metrics.get('complete_cycles', 1) if metrics.get('complete_cycles', 0) > 0 else 'N/A'
    ratio_str = f"{ratio_val:.1f}" if isinstance(ratio_val, float) else ratio_val

    metrics_text = f'''
╔════════════════════════════════════════════════════════════════════════════════╗
║  CHANNEL METRICS                                                               ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  Ping-pongs: {metrics.get('ping_pongs', 0):3.0f} (2%)  │  Complete Cycles: {metrics.get('complete_cycles', 0):3.0f} (2%)  │  Ratio: {ratio_str:<5}      ║
║  R² (fit):   {metrics.get('r_squared', 0):.3f}       │  Quality Score:   {metrics.get('quality_score', 0):.3f}       │  Valid: {'✅' if metrics.get('is_valid', 0) > 0.5 else '❌'}          ║
║  Slope:      {metrics.get('close_slope_pct', metrics.get('slope_pct', 0)):+.4f}%/bar │  Position: {metrics.get('position', 0):.2f} (0=low,1=up) │  Duration: {metrics.get('duration', 0):3.0f} bars ║
╚════════════════════════════════════════════════════════════════════════════════╝
    '''
    ax_metrics.text(0.02, 0.5, metrics_text, fontsize=10, family='monospace',
                    verticalalignment='center')

    # =========================================================================
    # BOTTOM PANEL: LABEL VERIFICATION
    # =========================================================================
    if ax_labels is not None:
        ax_labels.axis('off')

        # Build label info text
        label_lines = ["╔════════════════════════════════════════════════════════════════════════════════╗"]
        label_lines.append("║  LABEL VERIFICATION                                                           ║")
        label_lines.append("╠════════════════════════════════════════════════════════════════════════════════╣")

        if continuation_label:
            dur = continuation_label.get('duration_bars', 'N/A')
            gain = continuation_label.get('max_gain_pct', 0)
            gain_str = f"{gain:+.2f}%" if isinstance(gain, (int, float)) else 'N/A'
            label_lines.append(f"║  CONTINUATION:  Duration = {dur} bars  │  Max Gain = {gain_str:<20}       ║")
        else:
            label_lines.append("║  CONTINUATION:  [No label found]                                              ║")

        if transition_label:
            trans_type = transition_label.get('transition_name', 'N/A')
            direction = transition_label.get('direction_name', 'N/A')
            label_lines.append(f"║  TRANSITION:    Type = {trans_type:<12} │  New Direction = {direction:<15}       ║")
        else:
            label_lines.append("║  TRANSITION:    [No label found]                                              ║")

        # Visual verification status
        label_lines.append("╠════════════════════════════════════════════════════════════════════════════════╣")

        if continuation_label and future_df is not None and len(future_df) > 0:
            # Check if break occurred roughly where expected
            dur = continuation_label.get('duration_bars', 0)
            if dur > 0:
                # Check price at break point vs channel bounds
                check_str = "✓ Break point visible in future data" if dur < len(window_df) + len(future_df) else "⚠️ Break point beyond displayed range"
                label_lines.append(f"║  VISUAL CHECK:  {check_str:<62}║")
        else:
            label_lines.append("║  VISUAL CHECK:  [Load future data to verify break point]                      ║")

        label_lines.append("╚════════════════════════════════════════════════════════════════════════════════╝")

        label_text = '\n'.join(label_lines)
        ax_labels.text(0.02, 0.5, label_text, fontsize=10, family='monospace',
                      verticalalignment='center')

    plt.tight_layout()
    plt.show()


def main():
    """Interactive channel visualizer."""
    print("=" * 70)
    print("🎨 CHANNEL VISUALIZER - AutoTrade v3.17")
    print("=" * 70)
    print()

    # Step 1: Select shard location
    print("Step 1: Select shard storage location")
    print("-" * 70)
    shard_path = select_shard_location()
    print(f"\n✓ Selected: {shard_path}\n")

    # Step 2: Load shard data
    print("Step 2: Loading shard data...")
    print("-" * 70)
    try:
        loader = ChannelLoader(shard_path)
        print()
    except Exception as e:
        print(f"❌ Error loading shards: {e}")
        return

    # Step 2b: Load labels for verification
    print("Step 2b: Loading labels for verification...")
    print("-" * 70)
    loader._load_labels()
    print()

    # Step 3: Show summary stats
    print("Step 3: Channel Statistics")
    print("-" * 70)
    stats = loader.get_summary_stats()
    print(f"  Total timestamps: {stats['total_timestamps']:,}")
    print(f"  Sampled: {stats['sample_size']:,} for statistics\n")

    print("  Metrics (1h window=100 sample):")
    for metric, values in stats['metrics'].items():
        print(f"    {metric:20s}: mean={values['mean']:.2f}, "
              f"median={values['median']:.2f}, "
              f"range=[{values['min']:.2f}, {values['max']:.2f}]")
    print()

    # Step 4: Main visualization loop
    while True:
        print("\nStep 4: What would you like to visualize?")
        print("-" * 70)

        if not INQUIRER_AVAILABLE:
            print("  1. Specific channel (enter details)")
            print("  2. Random high-quality channels (quality > 0.8)")
            print("  3. Random low-quality channels (quality < 0.3)")
            print("  4. Compare ping_pongs vs complete_cycles")
            print("  5. Exit")

            action = input("Select option: ").strip()

            if action == '5' or action.lower() == 'exit':
                break
            # ... handle other options ...
        else:
            action = inquirer.select(
                message="Select visualization mode:",
                choices=[
                    Choice('specific', "🎯 Specific channel (enter details)"),
                    Choice('high_quality', "⭐ Random high-quality channels (quality > 0.8)"),
                    Choice('low_quality', "⚠️  Random low-quality channels (quality < 0.3)"),
                    Choice('by_transition', "🔄 Browse by transition type (CONTINUE/REVERSE/etc)"),
                    Choice('compare', "📊 Compare ping_pongs vs complete_cycles"),
                    Choice('browse', "🔍 Browse all windows for a timestamp"),
                    Choice('exit', "❌ Exit")
                ]
            ).execute()

            if action == 'exit':
                print("\n👋 Goodbye!")
                break

            # Handle specific channel visualization
            if action == 'specific':
                # Get user inputs
                symbol = inquirer.select(
                    message="Symbol:",
                    choices=[Choice('tsla', 'TSLA'), Choice('spy', 'SPY')]
                ).execute()

                timeframe = inquirer.select(
                    message="Timeframe:",
                    choices=[
                        Choice('1h', '1 Hour'),
                        Choice('4h', '4 Hour'),
                        Choice('daily', 'Daily'),
                        Choice('5min', '5 Minute'),
                        Choice('15min', '15 Minute'),
                    ]
                ).execute()

                window = inquirer.select(
                    message="Window size:",
                    choices=[
                        Choice(100, '100 bars (max available)'),
                        Choice(90, '90 bars'),
                        Choice(80, '80 bars'),
                        Choice(60, '60 bars'),
                        Choice(30, '30 bars')
                    ]
                ).execute()

                timestamp_str = inquirer.text(
                    message="Timestamp (YYYY-MM-DD HH:MM or 'random'):",
                    default="random"
                ).execute()

                if timestamp_str.lower() == 'random':
                    # Pick random high-quality timestamp
                    results = loader.find_high_quality_timestamps(
                        symbol=symbol, timeframe=timeframe, window=window,
                        min_quality=0.5, limit=100
                    )

                    if not results:
                        print("❌ No channels found with quality > 0.5")
                        continue

                    timestamp, quality = results[np.random.randint(len(results))]
                    print(f"  🎲 Random timestamp: {timestamp} (quality: {quality:.3f})")
                else:
                    timestamp = pd.Timestamp(timestamp_str)

                # Load and plot
                try:
                    print(f"\n  Loading channel data...")
                    metrics = loader.get_channel_metrics(timestamp, symbol, timeframe, window)

                    print(f"  Loading raw OHLC window...")
                    window_df = loader.get_raw_ohlc_window(timestamp, symbol, window, timeframe)

                    print(f"  Loading labels...")
                    cont_label = loader.get_continuation_label(timestamp, timeframe)
                    trans_label = loader.get_transition_label(timestamp, timeframe)

                    # Load future data to verify break point
                    future_df = None
                    if cont_label and 'duration_bars' in cont_label:
                        print(f"  Loading future price data for break verification...")
                        future_df = loader.get_future_price_data(
                            timestamp, symbol,
                            bars_forward=cont_label['duration_bars'] + 20,
                            timeframe=timeframe
                        )

                    print(f"  ✓ Loaded {len(window_df)} bars\n")

                    plot_channel(timestamp, symbol, timeframe, window, metrics, window_df,
                                continuation_label=cont_label, transition_label=trans_label,
                                future_df=future_df)

                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Handle high-quality random browsing
            elif action == 'high_quality':
                symbol = 'tsla'
                timeframe = '1h'
                window = 100  # v5.7: max available window

                print(f"  Finding high-quality {symbol} {timeframe} channels (window={window})...")

                results = loader.find_high_quality_timestamps(
                    symbol=symbol, timeframe=timeframe, window=window,
                    min_quality=0.8, limit=20
                )

                if not results:
                    print("❌ No high-quality channels found")
                    continue

                print(f"  ✓ Found {len(results)} high-quality channels\n")

                # Show random selection
                for i in range(min(5, len(results))):
                    ts, quality = results[i]

                    print(f"  [{i+1}] {ts} - Quality: {quality:.3f}")

                    try:
                        metrics = loader.get_channel_metrics(ts, symbol, timeframe, window)
                        window_df = loader.get_raw_ohlc_window(ts, symbol, window, timeframe)

                        # Load labels
                        cont_label = loader.get_continuation_label(ts, timeframe)
                        trans_label = loader.get_transition_label(ts, timeframe)

                        # Load future data if we have duration
                        future_df = None
                        if cont_label and 'duration_bars' in cont_label:
                            future_df = loader.get_future_price_data(
                                ts, symbol,
                                bars_forward=cont_label['duration_bars'] + 20,
                                timeframe=timeframe
                            )

                        plot_channel(ts, symbol, timeframe, window, metrics, window_df,
                                    continuation_label=cont_label, transition_label=trans_label,
                                    future_df=future_df)

                    except Exception as e:
                        print(f"    ⚠️  Could not plot: {e}")
                        continue

            # Handle low-quality browsing
            elif action == 'low_quality':
                # Similar to high_quality but with min_quality=0, filter for quality < 0.3
                print("  Finding low-quality channels...")

                symbol = 'tsla'
                timeframe = '1h'
                window = 100  # v5.7: max available window

                # Get all and filter
                all_results = loader.find_high_quality_timestamps(
                    symbol=symbol, timeframe=timeframe, window=window,
                    min_quality=0.0, limit=1000
                )

                low_quality = [(ts, q) for ts, q in all_results if q < 0.3]

                if not low_quality:
                    print("❌ No low-quality channels found")
                    continue

                print(f"  ✓ Found {len(low_quality)} low-quality channels\n")

                # Show random 5
                import random
                sample = random.sample(low_quality, min(5, len(low_quality)))

                for i, (ts, quality) in enumerate(sample):
                    print(f"  [{i+1}] {ts} - Quality: {quality:.3f}")

                    try:
                        metrics = loader.get_channel_metrics(ts, symbol, timeframe, window)
                        window_df = loader.get_raw_ohlc_window(ts, symbol, window, timeframe)

                        # Load labels
                        cont_label = loader.get_continuation_label(ts, timeframe)
                        trans_label = loader.get_transition_label(ts, timeframe)

                        # Load future data if we have duration
                        future_df = None
                        if cont_label and 'duration_bars' in cont_label:
                            future_df = loader.get_future_price_data(
                                ts, symbol,
                                bars_forward=cont_label['duration_bars'] + 20,
                                timeframe=timeframe
                            )

                        plot_channel(ts, symbol, timeframe, window, metrics, window_df,
                                    continuation_label=cont_label, transition_label=trans_label,
                                    future_df=future_df)

                    except Exception as e:
                        print(f"    ⚠️  Could not plot: {e}")
                        continue

            # Browse by transition type
            elif action == 'by_transition':
                print("  🔄 Browse by Transition Type")
                print("-" * 50)

                # Select transition type
                trans_type = inquirer.select(
                    message="Select transition type to browse:",
                    choices=[
                        Choice(0, "CONTINUE - Channel continues in same direction"),
                        Choice(1, "SWITCH_TF - Timeframe switches"),
                        Choice(2, "REVERSE - Direction reverses"),
                        Choice(3, "SIDEWAYS - Transitions to sideways")
                    ]
                ).execute()

                trans_names = ['CONTINUE', 'SWITCH_TF', 'REVERSE', 'SIDEWAYS']
                print(f"\n  Searching for {trans_names[trans_type]} transitions...\n")

                # Select timeframe
                timeframe = inquirer.select(
                    message="Timeframe:",
                    choices=[
                        Choice('1h', '1 Hour'),
                        Choice('4h', '4 Hour'),
                        Choice('daily', 'Daily'),
                        Choice('5min', '5 Minute'),
                        Choice('15min', '15 Minute'),
                    ]
                ).execute()

                # Find timestamps with this transition type
                timestamps = loader.find_timestamps_by_transition_type(
                    trans_type, timeframe, limit=50
                )

                if not timestamps:
                    print(f"  ❌ No {trans_names[trans_type]} transitions found for {timeframe}")
                    continue

                print(f"  ✓ Found {len(timestamps)} {trans_names[trans_type]} transitions\n")

                # Show random 5
                import random
                sample = random.sample(timestamps, min(5, len(timestamps)))

                symbol = 'tsla'
                window = 100  # Use 100-bar window for transitions

                for i, ts in enumerate(sample):
                    print(f"  [{i+1}] {ts}")

                    try:
                        metrics = loader.get_channel_metrics(ts, symbol, timeframe, window)
                        window_df = loader.get_raw_ohlc_window(ts, symbol, window, timeframe)

                        # Load labels
                        cont_label = loader.get_continuation_label(ts, timeframe)
                        trans_label = loader.get_transition_label(ts, timeframe)

                        # Load future data
                        future_df = None
                        if cont_label and 'duration_bars' in cont_label:
                            future_df = loader.get_future_price_data(
                                ts, symbol,
                                bars_forward=cont_label['duration_bars'] + 20,
                                timeframe=timeframe
                            )

                        plot_channel(ts, symbol, timeframe, window, metrics, window_df,
                                    continuation_label=cont_label, transition_label=trans_label,
                                    future_df=future_df)

                    except Exception as e:
                        print(f"    ⚠️  Could not plot: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

            # Compare metrics
            elif action == 'compare':
                print("  📊 Metric Comparison Mode")
                print("  This will show channels where ping_pongs and complete_cycles differ significantly\n")

                # Find channels with high ratio (many transitions, few cycles)
                # ... (implementation)
                print("  ⚠️  Feature coming soon!")

            # Browse all windows
            elif action == 'browse':
                print("  🔍 Browse All Windows Mode")
                print("  Shows same timestamp/timeframe with ALL window sizes\n")

                # ... (implementation)
                print("  ⚠️  Feature coming soon!")


if __name__ == '__main__':
    main()
