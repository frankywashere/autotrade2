"""
v15/inspector.py - Terminal-based label inspector for cache samples.

A simple CLI tool to visualize and validate cache samples interactively.

Usage:
    python -m v15.inspector --cache path/to/cache.pkl
"""

import argparse
import os
import pickle
import sys
from typing import List, Optional

from v15.types import (
    ChannelSample,
    ChannelLabels,
    TIMEFRAMES,
    STANDARD_WINDOWS,
    BREAK_DOWN,
    BREAK_UP,
    DIRECTION_BEAR,
    DIRECTION_SIDEWAYS,
    DIRECTION_BULL,
    INDEX_TO_TF,
)


def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_break_direction(direction: int) -> str:
    """Format break direction as human-readable string."""
    if direction == BREAK_DOWN:
        return "DOWN"
    elif direction == BREAK_UP:
        return "UP"
    return "UNKNOWN"


def format_channel_direction(direction: int) -> str:
    """Format channel direction as human-readable string."""
    if direction == DIRECTION_BEAR:
        return "BEAR"
    elif direction == DIRECTION_SIDEWAYS:
        return "SIDEWAYS"
    elif direction == DIRECTION_BULL:
        return "BULL"
    return "UNKNOWN"


def draw_price_position(position_pct: float, width: int = 40) -> str:
    """
    Draw ASCII representation of price position in channel.

    Args:
        position_pct: Position as percentage (0.0 = lower, 1.0 = upper)
        width: Width of the bar in characters

    Returns:
        ASCII string showing position marker
    """
    position_pct = max(0.0, min(1.0, position_pct))
    marker_pos = int(position_pct * (width - 1))
    bar = '=' * marker_pos + '|' + '=' * (width - marker_pos - 1)
    return bar


def display_sample(
    sample: ChannelSample,
    sample_idx: int,
    total_samples: int,
    current_window: int,
    current_tf: str,
) -> None:
    """
    Display a single sample's information.

    Args:
        sample: The ChannelSample to display
        sample_idx: Current sample index
        total_samples: Total number of samples
        current_window: Currently selected window size
        current_tf: Currently selected timeframe
    """
    clear_screen()

    print("=" * 60)
    print(f"  LABEL INSPECTOR - Sample {sample_idx + 1}/{total_samples}")
    print("=" * 60)
    print()

    # Basic sample info
    print(f"Timestamp:       {sample.timestamp}")
    print(f"Channel End Idx: {sample.channel_end_idx}")
    print(f"Best Window:     {sample.best_window}")
    print()

    # Current view selection
    print(f"Current Window:  {current_window}")
    print(f"Current TF:      {current_tf}")
    print("-" * 40)
    print()

    # Get channel for current window
    channel = sample.channels.get(current_window)
    if channel is not None:
        print("CHANNEL INFO:")
        # Try to access common channel attributes
        direction = getattr(channel, 'direction', None)
        bounce_count = getattr(channel, 'bounce_count', None)
        r_squared = getattr(channel, 'r_squared', None)
        width_pct = getattr(channel, 'width_pct', None)

        if direction is not None:
            print(f"  Direction:     {format_channel_direction(direction) if isinstance(direction, int) else direction}")
        if bounce_count is not None:
            print(f"  Bounce Count:  {bounce_count}")
        if r_squared is not None:
            print(f"  R-Squared:     {r_squared:.4f}" if isinstance(r_squared, float) else f"  R-Squared:     {r_squared}")
        if width_pct is not None:
            print(f"  Width %:       {width_pct:.4f}" if isinstance(width_pct, float) else f"  Width %:       {width_pct}")
        print()
    else:
        print("CHANNEL INFO: Not available for this window")
        print()

    # Get labels for current window and timeframe
    labels_for_window = sample.labels_per_window.get(current_window, {})
    labels: Optional[ChannelLabels] = labels_for_window.get(current_tf)

    print("LABELS:")
    if labels is not None:
        # Handle enum types (BreakDirection, NewChannelDirection)
        break_dir = labels.break_direction
        if hasattr(break_dir, 'name'):
            break_dir_str = break_dir.name
        else:
            break_dir_str = format_break_direction(break_dir)

        new_dir = labels.new_channel_direction
        if hasattr(new_dir, 'name'):
            new_dir_str = new_dir.name
        else:
            new_dir_str = format_channel_direction(new_dir)

        print(f"  Duration Bars:      {labels.duration_bars} {'(valid)' if labels.duration_valid else '(invalid)'}")
        print(f"  Break Direction:    {break_dir_str} {'(valid)' if labels.direction_valid else '(invalid)'}")
        print(f"  Permanent Break:    {labels.permanent_break}")
        print(f"  New Channel Dir:    {new_dir_str}")
    else:
        print("  No labels available for this window/timeframe combination")
    print()

    # Get features for current window
    features = sample.features_per_window.get(current_window, {})
    print("FEATURES:")
    if features:
        print(f"  Feature Count: {len(features)}")
        # Show a few sample features
        feature_items = list(features.items())[:5]
        for name, value in feature_items:
            if isinstance(value, float):
                print(f"    {name}: {value:.6f}")
            else:
                print(f"    {name}: {value}")
        if len(features) > 5:
            print(f"    ... and {len(features) - 5} more features")
    else:
        print("  No features available for this window")
    print()

    # ASCII visualization of price position
    print("PRICE POSITION IN CHANNEL:")
    # Try to get price position from features
    position_pct = features.get('price_position', features.get('position_pct', 0.5))

    print(f"  UPPER: {draw_price_position(1.0)}")
    print(f"  PRICE: {draw_price_position(position_pct)} ({position_pct:.2%})")
    print(f"  LOWER: {draw_price_position(0.0)}")
    print()

    # Show available windows and timeframes
    available_windows = sorted(sample.labels_per_window.keys())
    print(f"Available Windows: {available_windows}")

    if current_window in sample.labels_per_window:
        available_tfs = list(sample.labels_per_window[current_window].keys())
        print(f"Available TFs for window {current_window}: {available_tfs}")
    print()

    # Navigation help
    print("-" * 60)
    print("Navigation:")
    print("  Enter: Next sample    p: Previous sample")
    print("  w: Change window      t: Change timeframe")
    print("  q: Quit")
    print("-" * 60)


def inspect_cache(cache_path: str) -> None:
    """
    Main inspection loop for cache samples.

    Args:
        cache_path: Path to the pickle cache file containing List[ChannelSample]
    """
    # Load cache file
    print(f"Loading cache from: {cache_path}")

    if not os.path.exists(cache_path):
        print(f"Error: Cache file not found: {cache_path}")
        sys.exit(1)

    try:
        with open(cache_path, 'rb') as f:
            samples: List[ChannelSample] = pickle.load(f)
    except Exception as e:
        print(f"Error loading cache: {e}")
        sys.exit(1)

    if not samples:
        print("Error: Cache file is empty")
        sys.exit(1)

    print(f"Loaded {len(samples)} samples")

    # Initialize navigation state
    sample_idx = 0

    # Get available windows from first sample
    available_windows = sorted(samples[0].labels_per_window.keys()) if samples[0].labels_per_window else STANDARD_WINDOWS
    window_idx = 0
    current_window = available_windows[window_idx] if available_windows else 50

    # Get available timeframes
    available_tfs = TIMEFRAMES.copy()
    tf_idx = 0
    current_tf = available_tfs[tf_idx]

    # Main loop
    while True:
        sample = samples[sample_idx]

        # Update available windows/tfs for current sample
        if sample.labels_per_window:
            available_windows = sorted(sample.labels_per_window.keys())
            if current_window not in available_windows and available_windows:
                current_window = available_windows[0]
                window_idx = 0

            if current_window in sample.labels_per_window:
                sample_tfs = list(sample.labels_per_window[current_window].keys())
                if sample_tfs and current_tf not in sample_tfs:
                    current_tf = sample_tfs[0]
                    tf_idx = available_tfs.index(current_tf) if current_tf in available_tfs else 0

        # Display current sample
        display_sample(sample, sample_idx, len(samples), current_window, current_tf)

        # Get user input
        try:
            user_input = input("\nCommand: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        # Process input
        if user_input == 'q':
            print("Goodbye!")
            break
        elif user_input == '' or user_input == 'n':
            # Next sample
            sample_idx = (sample_idx + 1) % len(samples)
        elif user_input == 'p':
            # Previous sample
            sample_idx = (sample_idx - 1) % len(samples)
        elif user_input == 'w':
            # Cycle through windows
            if available_windows:
                window_idx = (window_idx + 1) % len(available_windows)
                current_window = available_windows[window_idx]
                print(f"Switched to window: {current_window}")
        elif user_input == 't':
            # Cycle through timeframes
            tf_idx = (tf_idx + 1) % len(available_tfs)
            current_tf = available_tfs[tf_idx]
            print(f"Switched to timeframe: {current_tf}")
        elif user_input.isdigit():
            # Jump to sample number
            target = int(user_input) - 1
            if 0 <= target < len(samples):
                sample_idx = target
            else:
                print(f"Invalid sample number. Must be 1-{len(samples)}")
        else:
            print(f"Unknown command: {user_input}")


def main() -> None:
    """Entry point for the inspector CLI."""
    parser = argparse.ArgumentParser(
        description="Terminal-based label inspector for cache samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m v15.inspector --cache v7/cache_v14/samples.pkl
    python -m v15.inspector --cache /path/to/cache.pkl
        """
    )
    parser.add_argument(
        '--cache',
        type=str,
        required=True,
        help='Path to the pickle cache file containing List[ChannelSample]'
    )

    args = parser.parse_args()
    inspect_cache(args.cache)


if __name__ == '__main__':
    main()
