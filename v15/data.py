"""
Data loading utilities for v15.

Provides clean market data loading and OHLC resampling functions.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd


# All 11 timeframes in hierarchical order
TIMEFRAMES = [
    '5min', '15min', '30min', '1h', '2h', '3h', '4h',
    'daily', 'weekly', 'monthly', '3month'
]

# Pandas resample rules for each timeframe
RESAMPLE_RULES = {
    '5min': '5min',
    '15min': '15min',
    '30min': '30min',
    '1h': '1h',
    '2h': '2h',
    '3h': '3h',
    '4h': '4h',
    'daily': '1D',
    'weekly': '1W',
    'monthly': '1ME',
    '3month': '3ME',
}


def load_market_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load TSLA, SPY, and VIX market data with proper alignment.

    Loads 1-minute CSV files, resamples to 5-minute bars, and aligns
    SPY and VIX to TSLA's index using forward-fill.

    Args:
        data_dir: Directory containing TSLA_1min.csv, SPY_1min.csv, VIX_History.csv

    Returns:
        Tuple of (tsla_df, spy_df, vix_df) - all with aligned DatetimeIndex
        and columns [open, high, low, close, volume] (VIX has no volume)

    Raises:
        FileNotFoundError: If any required CSV file is missing
        ValueError: If data is empty or no overlapping dates exist
    """
    data_path = Path(data_dir)

    # Load and resample TSLA
    tsla_file = data_path / "TSLA_1min.csv"
    if not tsla_file.exists():
        raise FileNotFoundError(f"TSLA data not found: {tsla_file}")

    tsla_df = pd.read_csv(tsla_file)
    if tsla_df.empty:
        raise ValueError(f"TSLA data file is empty: {tsla_file}")

    tsla_df['timestamp'] = pd.to_datetime(tsla_df['timestamp'])
    tsla_df.set_index('timestamp', inplace=True)
    tsla_df.sort_index(inplace=True)
    tsla_df = _resample_to_5min(tsla_df)

    # Load and resample SPY
    spy_file = data_path / "SPY_1min.csv"
    if not spy_file.exists():
        raise FileNotFoundError(f"SPY data not found: {spy_file}")

    spy_df = pd.read_csv(spy_file)
    if spy_df.empty:
        raise ValueError(f"SPY data file is empty: {spy_file}")

    spy_df['timestamp'] = pd.to_datetime(spy_df['timestamp'])
    spy_df.set_index('timestamp', inplace=True)
    spy_df.sort_index(inplace=True)
    spy_df = _resample_to_5min(spy_df)

    # Load VIX daily data
    vix_file = data_path / "VIX_History.csv"
    if not vix_file.exists():
        raise FileNotFoundError(f"VIX data not found: {vix_file}")

    vix_df = pd.read_csv(vix_file)
    if vix_df.empty:
        raise ValueError(f"VIX data file is empty: {vix_file}")

    # VIX uses DATE column with MM/DD/YYYY format
    vix_df.columns = [c.lower() for c in vix_df.columns]
    vix_df['date'] = pd.to_datetime(vix_df['date'], format='%m/%d/%Y')
    vix_df.set_index('date', inplace=True)
    vix_df.sort_index(inplace=True)

    # Align SPY and VIX to TSLA's index
    tsla_aligned, spy_aligned, vix_aligned = _align_to_tsla(tsla_df, spy_df, vix_df)

    return tsla_aligned, spy_aligned, vix_aligned


def _resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute OHLCV data to 5-minute bars."""
    return df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


def _align_to_tsla(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align SPY and VIX to TSLA's index using forward-fill.

    Finds overlapping date range, then reindexes SPY and VIX to match
    TSLA's timestamps.
    """
    # Find common date range
    tsla_dates = set(tsla_df.index.date)
    spy_dates = set(spy_df.index.date)
    vix_dates = set(vix_df.index.date)

    common_dates = tsla_dates & spy_dates & vix_dates

    if not common_dates:
        raise ValueError(
            f"No overlapping dates. "
            f"TSLA: {min(tsla_dates)} to {max(tsla_dates)}, "
            f"SPY: {min(spy_dates)} to {max(spy_dates)}, "
            f"VIX: {min(vix_dates)} to {max(vix_dates)}"
        )

    start_date = min(common_dates)
    end_date = max(common_dates)

    # Filter to common date range
    tsla_df = tsla_df[
        (tsla_df.index.date >= start_date) &
        (tsla_df.index.date <= end_date)
    ]
    spy_df = spy_df[
        (spy_df.index.date >= start_date) &
        (spy_df.index.date <= end_date)
    ]
    vix_df = vix_df[
        (vix_df.index.date >= start_date) &
        (vix_df.index.date <= end_date)
    ]

    # Reindex SPY and VIX to TSLA's index with forward-fill
    spy_aligned = spy_df.reindex(tsla_df.index, method='ffill')
    vix_aligned = vix_df.reindex(tsla_df.index, method='ffill')

    # Remove rows with NaN (at start before ffill has data)
    valid_mask = (
        ~tsla_df.isna().any(axis=1) &
        ~spy_aligned.isna().any(axis=1) &
        ~vix_aligned.isna().any(axis=1)
    )

    return (
        tsla_df[valid_mask].copy(),
        spy_aligned[valid_mask].copy(),
        vix_aligned[valid_mask].copy()
    )


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample 5-minute OHLCV data to a target timeframe.

    Args:
        df: DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
            (volume is optional for VIX data)
        timeframe: Target timeframe - one of:
            '5min', '15min', '30min', '1h', '2h', '3h', '4h',
            'daily', 'weekly', 'monthly', '3month'

    Returns:
        Resampled DataFrame with same columns

    Raises:
        ValueError: If timeframe is not recognized
    """
    if timeframe not in RESAMPLE_RULES:
        raise ValueError(
            f"Unknown timeframe: '{timeframe}'. "
            f"Valid timeframes: {TIMEFRAMES}"
        )

    rule = RESAMPLE_RULES[timeframe]

    # Build aggregation dict based on available columns
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }

    # Add volume if present
    if 'volume' in df.columns:
        agg_dict['volume'] = 'sum'

    resampled = df.resample(rule).agg(agg_dict).dropna()

    return resampled
