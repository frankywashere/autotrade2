"""
Walk-Forward Validation for Time Series Training

This module provides walk-forward validation utilities for training models on time series data.
Walk-forward validation is critical for financial time series to prevent look-ahead bias and
ensure models are evaluated on truly out-of-sample data.

Key Features:
1. Expanding window approach - training set grows over time (not sliding)
2. Fixed validation period - consistent validation window size
3. Proper date handling with pandas Timestamps
4. Validation of date ranges and edge cases

Example usage:
    >>> from v7.training.walk_forward import generate_walk_forward_windows, split_samples_by_window
    >>>
    >>> # Generate 5 windows with 3-month validation periods
    >>> windows = generate_walk_forward_windows(
    ...     data_start='2020-01-01',
    ...     data_end='2024-12-31',
    ...     num_windows=5,
    ...     validation_period_months=3
    ... )
    >>>
    >>> # Split samples for first window
    >>> train, val = split_samples_by_window(
    ...     samples=all_samples,
    ...     train_start=windows[0][0],
    ...     train_end=windows[0][1],
    ...     val_start=windows[0][2],
    ...     val_end=windows[0][3]
    ... )
"""

import pandas as pd
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime

if TYPE_CHECKING:
    from .dataset import ChannelSample


@dataclass
class WalkForwardWindow:
    """
    A single walk-forward validation window.

    Attributes:
        train_start: Start date of training period
        train_end: End date of training period (inclusive)
        val_start: Start date of validation period
        val_end: End date of validation period (inclusive)
        window_id: Index of this window (0-based)
    """
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    window_id: int

    def __repr__(self) -> str:
        return (
            f"WalkForwardWindow(id={self.window_id}, "
            f"train={self.train_start.date()} to {self.train_end.date()}, "
            f"val={self.val_start.date()} to {self.val_end.date()})"
        )


def generate_walk_forward_windows(
    data_start: str,
    data_end: str,
    num_windows: int,
    validation_period_months: int = 3
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate walk-forward validation windows using an expanding window approach.

    The expanding window approach means:
    - Training set starts at data_start and grows with each window
    - Each validation period is a fixed duration (validation_period_months)
    - Windows are contiguous - next validation starts where previous ended

    Example with 3 windows and 3-month validation:
        Window 0: Train [2020-01 to 2023-01], Val [2023-02 to 2023-04]
        Window 1: Train [2020-01 to 2023-05], Val [2023-05 to 2023-07]
        Window 2: Train [2020-01 to 2023-08], Val [2023-08 to 2023-10]

    Args:
        data_start: Start date of entire dataset (YYYY-MM-DD)
        data_end: End date of entire dataset (YYYY-MM-DD)
        num_windows: Number of walk-forward windows to generate
        validation_period_months: Size of each validation period in months

    Returns:
        List of tuples (train_start, train_end, val_start, val_end)
        Each tuple defines one complete walk-forward window

    Raises:
        ValueError: If parameters are invalid or windows cannot fit in date range

    Example:
        >>> windows = generate_walk_forward_windows(
        ...     data_start='2020-01-01',
        ...     data_end='2024-12-31',
        ...     num_windows=4,
        ...     validation_period_months=3
        ... )
        >>> len(windows)
        4
        >>> # First window
        >>> windows[0]  # (train_start, train_end, val_start, val_end)
    """
    # Input validation
    if num_windows < 1:
        raise ValueError(f"num_windows must be >= 1, got {num_windows}")

    if validation_period_months < 1:
        raise ValueError(
            f"validation_period_months must be >= 1, got {validation_period_months}"
        )

    # Parse dates
    try:
        start_date = pd.Timestamp(data_start)
        end_date = pd.Timestamp(data_end)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")

    if start_date >= end_date:
        raise ValueError(
            f"data_start ({data_start}) must be before data_end ({data_end})"
        )

    # Calculate total months available and required
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    validation_months_needed = num_windows * validation_period_months

    # Need at least some training data before first validation
    min_training_months = 6  # Minimum 6 months of initial training data

    if total_months < min_training_months + validation_months_needed:
        raise ValueError(
            f"Insufficient data range. Need at least {min_training_months + validation_months_needed} "
            f"months ({min_training_months} for initial training + "
            f"{validation_months_needed} for {num_windows} validation windows), "
            f"but only have {total_months} months available"
        )

    # Calculate validation start point
    # Leave minimum training period at start, distribute rest for validation windows
    training_buffer_months = total_months - validation_months_needed

    # First validation starts after initial training buffer
    first_val_start = start_date + pd.DateOffset(months=training_buffer_months)

    # Generate windows
    windows = []
    current_val_start = first_val_start

    for i in range(num_windows):
        # Calculate validation period for this window
        val_start = current_val_start
        val_end = val_start + pd.DateOffset(months=validation_period_months) - pd.DateOffset(days=1)

        # Ensure we don't exceed data_end
        if val_end > end_date:
            val_end = end_date

        # Training period: from start to just before validation
        train_start = start_date
        train_end = val_start - pd.DateOffset(days=1)

        # Ensure training end is after training start
        if train_end <= train_start:
            raise ValueError(
                f"Window {i}: Invalid training period. "
                f"train_end ({train_end.date()}) <= train_start ({train_start.date()})"
            )

        # Add window
        windows.append((train_start, train_end, val_start, val_end))

        # Move to next validation period
        current_val_start = val_end + pd.DateOffset(days=1)

        # Check if we have room for another window
        if i < num_windows - 1 and current_val_start > end_date:
            raise ValueError(
                f"Cannot fit {num_windows} windows. Only able to create {i + 1} window(s)"
            )

    return windows


def split_samples_by_window(
    samples: List,  # List[ChannelSample] but using List for runtime compatibility
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp
) -> Tuple[List, List]:  # Returns (List[ChannelSample], List[ChannelSample])
    """
    Split samples into training and validation sets for a specific walk-forward window.

    Samples are assigned based on their timestamp:
    - Training: samples where train_start <= timestamp <= train_end
    - Validation: samples where val_start <= timestamp <= val_end

    Args:
        samples: List of all ChannelSample objects
        train_start: Start of training period
        train_end: End of training period (inclusive)
        val_start: Start of validation period
        val_end: End of validation period (inclusive)

    Returns:
        Tuple of (train_samples, val_samples)

    Raises:
        ValueError: If date parameters are invalid

    Example:
        >>> windows = generate_walk_forward_windows('2020-01-01', '2024-12-31', 3)
        >>> train, val = split_samples_by_window(
        ...     samples=all_samples,
        ...     train_start=windows[0][0],
        ...     train_end=windows[0][1],
        ...     val_start=windows[0][2],
        ...     val_end=windows[0][3]
        ... )
        >>> print(f"Train: {len(train)} samples, Val: {len(val)} samples")
    """
    # Validate inputs
    if train_start >= train_end:
        raise ValueError(
            f"train_start ({train_start.date()}) must be before "
            f"train_end ({train_end.date()})"
        )

    if val_start >= val_end:
        raise ValueError(
            f"val_start ({val_start.date()}) must be before "
            f"val_end ({val_end.date()})"
        )

    if train_end >= val_start:
        raise ValueError(
            f"train_end ({train_end.date()}) must be before "
            f"val_start ({val_start.date()}) to prevent data leakage"
        )

    # Split samples based on timestamps
    train_samples = []
    val_samples = []

    for sample in samples:
        # Training period (inclusive)
        if train_start <= sample.timestamp <= train_end:
            train_samples.append(sample)
        # Validation period (inclusive)
        elif val_start <= sample.timestamp <= val_end:
            val_samples.append(sample)
        # Sample outside window - skip it

    return train_samples, val_samples


def validate_windows(
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    verbose: bool = True
) -> bool:
    """
    Validate walk-forward windows for correctness.

    Checks:
    1. Training period ends before validation starts (no data leakage)
    2. Windows are in chronological order
    3. Training sets are expanding (each training period includes all previous)
    4. No gaps or overlaps in validation periods

    Args:
        windows: List of (train_start, train_end, val_start, val_end) tuples
        verbose: Print validation details

    Returns:
        True if all windows are valid

    Raises:
        ValueError: If any validation check fails
    """
    if not windows:
        raise ValueError("Windows list is empty")

    for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
        # Check 1: Training ends before validation starts
        if train_end >= val_start:
            raise ValueError(
                f"Window {i}: Training period overlaps validation period. "
                f"train_end ({train_end.date()}) >= val_start ({val_start.date()})"
            )

        # Check 2: Validation period is valid
        if val_start >= val_end:
            raise ValueError(
                f"Window {i}: Invalid validation period. "
                f"val_start ({val_start.date()}) >= val_end ({val_end.date()})"
            )

        # Check 3: Training period is valid
        if train_start >= train_end:
            raise ValueError(
                f"Window {i}: Invalid training period. "
                f"train_start ({train_start.date()}) >= train_end ({train_end.date()})"
            )

        # Check 4: Windows are in chronological order
        if i > 0:
            prev_train_start, prev_train_end, prev_val_start, prev_val_end = windows[i - 1]

            # Training should expand (start stays the same or earlier)
            if train_start != prev_train_start:
                raise ValueError(
                    f"Window {i}: Training start changed (not expanding window). "
                    f"Expected {prev_train_start.date()}, got {train_start.date()}"
                )

            # Training end should grow
            if train_end <= prev_train_end:
                raise ValueError(
                    f"Window {i}: Training period not expanding. "
                    f"train_end ({train_end.date()}) <= previous ({prev_train_end.date()})"
                )

            # Validation should be contiguous (current starts after previous ends)
            expected_val_start = prev_val_end + pd.DateOffset(days=1)
            if val_start != expected_val_start:
                raise ValueError(
                    f"Window {i}: Validation periods not contiguous. "
                    f"Expected val_start {expected_val_start.date()}, got {val_start.date()}"
                )

    if verbose:
        print(f"All {len(windows)} windows validated successfully")
        print("\nWindow breakdown:")
        for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
            train_days = (train_end - train_start).days
            val_days = (val_end - val_start).days + 1
            print(f"  Window {i}: Train={train_days} days, Val={val_days} days")

    return True


if __name__ == '__main__':
    """
    Example usage and testing of walk-forward validation.
    """
    print("Walk-Forward Validation Example\n" + "=" * 50)

    # Example 1: Generate windows
    print("\nExample 1: Generate 5 windows with 3-month validation periods")
    windows = generate_walk_forward_windows(
        data_start='2020-01-01',
        data_end='2024-12-31',
        num_windows=5,
        validation_period_months=3
    )

    print(f"\nGenerated {len(windows)} windows:")
    for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
        print(f"\nWindow {i}:")
        print(f"  Training:   {train_start.date()} to {train_end.date()}")
        print(f"  Validation: {val_start.date()} to {val_end.date()}")
        train_months = (train_end.year - train_start.year) * 12 + (train_end.month - train_start.month)
        val_months = (val_end.year - val_start.year) * 12 + (val_end.month - val_start.month) + 1
        print(f"  Train duration: ~{train_months} months")
        print(f"  Val duration: ~{val_months} months")

    # Example 2: Validate windows
    print("\n" + "=" * 50)
    print("Example 2: Validate windows")
    try:
        validate_windows(windows, verbose=True)
        print("Validation passed!")
    except ValueError as e:
        print(f"Validation failed: {e}")

    # Example 3: Edge case testing
    print("\n" + "=" * 50)
    print("Example 3: Edge case - insufficient data")
    try:
        # Try to fit too many windows in a short period
        windows_invalid = generate_walk_forward_windows(
            data_start='2023-01-01',
            data_end='2023-06-30',  # Only 6 months
            num_windows=10,  # Too many windows
            validation_period_months=3
        )
    except ValueError as e:
        print(f"Expected error caught: {e}")

    print("\n" + "=" * 50)
    print("Walk-forward validation module ready for use!")
