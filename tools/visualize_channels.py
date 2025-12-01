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
    show_cycles: bool = True
):
    """
    Plot channel with price, regression lines, and metrics.

    Args:
        timestamp: Timestamp of channel
        symbol: Stock symbol
        timeframe: Timeframe
        window: Window size
        metrics: Channel metrics dict
        window_df: Raw OHLC dataframe
        show_touches: Mark touch points
        show_cycles: Highlight complete cycles
    """
    # Reconstruct channel lines
    lines = reconstruct_channel_lines(metrics, window_df)

    fig, (ax_price, ax_metrics) = plt.subplots(2, 1, figsize=(16, 10),
                                                gridspec_kw={'height_ratios': [3, 1]})

    # Top panel: Price + Channel
    ax_price.plot(window_df.index, window_df['close'], label='Price', linewidth=2, color='black')
    ax_price.plot(window_df.index, lines['upper_line'], 'r--', label='Upper', alpha=0.7, linewidth=1.5)
    ax_price.plot(window_df.index, lines['lower_line'], 'g--', label='Lower', alpha=0.7, linewidth=1.5)
    ax_price.plot(window_df.index, lines['center_line'], 'b-', label='Center (regression)', alpha=0.5, linewidth=1)

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

    # Shade complete cycles if requested
    if show_cycles and metrics.get('complete_cycles', 0) > 0:
        # This is approximate - would need full touch sequence to be precise
        ax_price.text(0.02, 0.98, f"Complete Cycles: {metrics['complete_cycles']:.0f}",
                     transform=ax_price.transAxes, fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                     verticalalignment='top')

    ax_price.set_title(
        f"{symbol.upper()} - {timeframe} - Window {window} bars - {timestamp.strftime('%Y-%m-%d %H:%M')}",
        fontsize=14, fontweight='bold'
    )
    ax_price.set_ylabel('Price ($)', fontsize=12)
    ax_price.legend(loc='best')
    ax_price.grid(True, alpha=0.3)

    # Bottom panel: Metrics Table
    ax_metrics.axis('off')

    metrics_text = f'''
╔═══════════════════════════════════════════════════════════════╗
║  CHANNEL METRICS                                               ║
╠═══════════════════════════════════════════════════════════════╣
║  Legacy Transitions (ping_pongs):                              ║
║    ├─ 2.0% threshold: {metrics.get('ping_pongs', 0):.0f} transitions                        ║
║    ├─ 0.5% threshold: {metrics.get('ping_pongs_0_5pct', 0):.0f} transitions (strict)          ║
║    ├─ 1.0% threshold: {metrics.get('ping_pongs_1_0pct', 0):.0f} transitions                   ║
║    └─ 3.0% threshold: {metrics.get('ping_pongs_3_0pct', 0):.0f} transitions (loose)           ║
║                                                                ║
║  Complete Cycles (v3.17):                                      ║
║    ├─ 2.0% threshold: {metrics.get('complete_cycles', 0):.0f} full round-trips ⭐           ║
║    ├─ 0.5% threshold: {metrics.get('complete_cycles_0_5pct', 0):.0f} round-trips              ║
║    ├─ 1.0% threshold: {metrics.get('complete_cycles_1_0pct', 0):.0f} round-trips              ║
║    └─ 3.0% threshold: {metrics.get('complete_cycles_3_0pct', 0):.0f} round-trips              ║
║                                                                ║
║  Ratio: {metrics.get('ping_pongs', 0)/metrics.get('complete_cycles', 1) if metrics.get('complete_cycles', 0) > 0 else 'N/A':<6} transitions per complete cycle               ║
║                                                                ║
║  Quality Metrics:                                              ║
║    ├─ R² (fit quality): {metrics.get('r_squared', 0):.3f}                              ║
║    ├─ Quality score: {metrics.get('quality_score', 0):.3f}                                 ║
║    ├─ Is valid: {metrics.get('is_valid', 0):.1f} {'✅ YES' if metrics.get('is_valid', 0) > 0.5 else '❌ NO':<22}║
║    ├─ Slope: {metrics.get('slope', 0):+.4f} $/bar                               ║
║    ├─ Slope %: {metrics.get('slope_pct', 0):+.3f}% per bar                            ║
║    ├─ Position: {metrics.get('position', 0):.3f} (0=lower, 1=upper)                  ║
║    └─ Duration: {metrics.get('duration', 0):.0f} bars                                  ║
╚═══════════════════════════════════════════════════════════════╝
    '''

    ax_metrics.text(0.05, 0.5, metrics_text, fontsize=10, family='monospace',
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

    # Step 3: Show summary stats
    print("Step 3: Channel Statistics")
    print("-" * 70)
    stats = loader.get_summary_stats()
    print(f"  Total timestamps: {stats['total_timestamps']:,}")
    print(f"  Sampled: {stats['sample_size']:,} for statistics\n")

    print("  Metrics (1h window=168 sample):")
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
                        Choice(168, '168 bars (1 week for hourly)'),
                        Choice(120, '120 bars'),
                        Choice(90, '90 bars'),
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

                    print(f"  ✓ Loaded {len(window_df)} bars\n")

                    plot_channel(timestamp, symbol, timeframe, window, metrics, window_df)

                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Handle high-quality random browsing
            elif action == 'high_quality':
                symbol = 'tsla'
                timeframe = '1h'
                window = 168

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

                        plot_channel(ts, symbol, timeframe, window, metrics, window_df)

                    except Exception as e:
                        print(f"    ⚠️  Could not plot: {e}")
                        continue

            # Handle low-quality browsing
            elif action == 'low_quality':
                # Similar to high_quality but with min_quality=0, filter for quality < 0.3
                print("  Finding low-quality channels...")

                symbol = 'tsla'
                timeframe = '1h'
                window = 168

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

                        plot_channel(ts, symbol, timeframe, window, metrics, window_df)

                    except Exception as e:
                        print(f"    ⚠️  Could not plot: {e}")
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
