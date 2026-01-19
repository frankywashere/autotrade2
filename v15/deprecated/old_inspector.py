"""
DEPRECATED: This inspector does not support the new dual-asset label structure.
Use v15/dual_inspector.py instead:
    python -m v15.dual_inspector --cache samples.pkl

This file is kept for reference only and will be removed in a future version.
"""
import warnings
warnings.warn(
    "v15.deprecated_inspector is deprecated. Use v15.dual_inspector instead.",
    DeprecationWarning,
    stacklevel=2
)

"""
v15/inspector.py - Terminal-based label inspector for cache samples.

A simple CLI tool to visualize and validate cache samples interactively.

Usage:
    python -m v15.inspector samples.pkl
    python -m v15.inspector --samples path/to/cache.pkl
    python -m v15.inspector -s cache.pkl --start 10
    python -m v15.inspector --cache path/to/cache.pkl  # legacy alias
"""

import argparse
import os
import pickle
import sys
from typing import List, Optional

from v15.dtypes import (
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

    # Display channel-related features from tf_features
    print("CHANNEL INFO (from features):")
    # Extract channel features for current TF from the flat tf_features dict
    tf_prefix = f"{current_tf}_"
    channel_features = {
        k.replace(tf_prefix, ''): v
        for k, v in sample.tf_features.items()
        if k.startswith(tf_prefix)
    }

    # Show key channel metrics if available
    direction = channel_features.get('direction')
    if direction is not None:
        print(f"  Direction:     {format_channel_direction(int(direction)) if isinstance(direction, (int, float)) else direction}")

    bounce_count = channel_features.get('bounce_count')
    if bounce_count is not None:
        print(f"  Bounce Count:  {int(bounce_count)}")

    r_squared = channel_features.get('r_squared')
    if r_squared is not None:
        print(f"  R-Squared:     {r_squared:.4f}")

    width_pct = channel_features.get('width_pct')
    if width_pct is not None:
        print(f"  Width %:       {width_pct:.4f}")

    price_position = channel_features.get('price_position')
    if price_position is not None:
        print(f"  Price Pos:     {price_position:.4f}")

    if not channel_features:
        print("  No features available for this timeframe")
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

        new_dir = labels.next_channel_direction
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

    # Display features for current TF from flat tf_features dict
    tf_prefix = f"{current_tf}_"
    tf_specific_features = {
        k: v for k, v in sample.tf_features.items()
        if k.startswith(tf_prefix)
    }
    print(f"FEATURES (for {current_tf}):")
    if tf_specific_features:
        print(f"  Feature Count: {len(tf_specific_features)}")
        # Show a few sample features (remove TF prefix for display)
        feature_items = list(tf_specific_features.items())[:5]
        for name, value in feature_items:
            display_name = name.replace(tf_prefix, '')
            if isinstance(value, float):
                print(f"    {display_name}: {value:.6f}")
            else:
                print(f"    {display_name}: {value}")
        if len(tf_specific_features) > 5:
            print(f"    ... and {len(tf_specific_features) - 5} more features")
    else:
        print(f"  No features available for {current_tf}")

    # Show total feature count
    print(f"\n  Total Features (all TFs): {len(sample.tf_features)}")
    print()

    # ASCII visualization of price position
    print("PRICE POSITION IN CHANNEL:")
    # Try to get price position from tf_features
    position_pct = sample.tf_features.get(f'{current_tf}_price_position',
                                          sample.tf_features.get(f'{current_tf}_position_pct', 0.5))

    print(f"  UPPER: {draw_price_position(1.0)}")
    print(f"  PRICE: {draw_price_position(position_pct)} ({position_pct:.2%})")
    print(f"  LOWER: {draw_price_position(0.0)}")
    print()

    # Show bar metadata if available
    if sample.bar_metadata:
        print("BAR METADATA:")
        tf_metadata = sample.bar_metadata.get(current_tf, {})
        if tf_metadata:
            for key, value in tf_metadata.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  No metadata for {current_tf}")
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


def inspect_cache(cache_path: str, start_idx: int = 0) -> None:
    """
    Main inspection loop for cache samples.

    Args:
        cache_path: Path to the pickle cache file containing List[ChannelSample]
        start_idx: Starting sample index (default: 0)
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
    sample_idx = max(0, min(start_idx, len(samples) - 1))

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
    python -m v15.inspector samples.pkl
    python -m v15.inspector --samples v7/cache_v14/samples.pkl
    python -m v15.inspector -s /path/to/cache.pkl --start 10
        """
    )
    # Positional argument (optional) - samples pickle file
    parser.add_argument(
        'samples_file',
        type=str,
        nargs='?',
        default=None,
        help='Path to the pickle cache file (positional argument)'
    )
    # Named argument (optional) - samples pickle file
    parser.add_argument(
        '--samples', '-s',
        type=str,
        default=None,
        help='Path to the pickle cache file containing List[ChannelSample]'
    )
    # Legacy alias for backwards compatibility
    parser.add_argument(
        '--cache', '-c',
        type=str,
        default=None,
        help='Alias for --samples (backwards compatibility)'
    )
    # Starting sample index
    parser.add_argument(
        '--start', '-i',
        type=int,
        default=0,
        help='Starting sample index (default: 0)'
    )

    args = parser.parse_args()

    # Resolve cache path: positional > --samples > --cache
    cache_path = args.samples_file or args.samples or args.cache

    if not cache_path:
        parser.error("Please provide a samples pickle file as a positional argument or via --samples/-s")

    inspect_cache(cache_path, start_idx=args.start)


if __name__ == '__main__':
    main()
