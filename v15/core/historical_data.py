"""
Historical minute-bar data loader for Channel Surfer backtesting.

Parses semicolon-delimited minute OHLCV files (TSLAMin.txt, SPYMin.txt)
and resamples to higher timeframes for multi-TF analysis.

Format: YYYYMMDD HHMMSS;open;high;low;close;volume
No header row. US/Eastern timezone assumed.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_minute_data(filepath: str) -> pd.DataFrame:
    """
    Load semicolon-delimited 1-minute OHLCV data.

    Args:
        filepath: Path to minute data file (e.g., data/TSLAMin.txt)

    Returns:
        DataFrame with DatetimeIndex (US/Eastern tz-aware),
        columns: [open, high, low, close, volume]
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Read raw data -- format: YYYYMMDD HHMMSS;open;high;low;close;volume
    df = pd.read_csv(
        filepath,
        sep=';',
        header=None,
        names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
        dtype={'datetime': str, 'open': float, 'high': float,
               'low': float, 'close': float, 'volume': float},
    )

    # Parse datetime: "20150102 114000" -> datetime
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')
    df = df.set_index('datetime')
    df.index = df.index.tz_localize('US/Eastern')

    # Drop any fully-NaN rows
    df = df.dropna(how='all')

    return df


def resample_to_tf(df_1min: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Resample 1-minute data to a higher timeframe.

    Args:
        df_1min: 1-minute OHLCV DataFrame
        interval: Target interval ('5min', '15min', '1h', '4h', '1D', '1W')

    Returns:
        Resampled OHLCV DataFrame
    """
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    resampled = df_1min.resample(interval).agg(agg)
    # Drop bars where close is NaN (gaps -- weekends, holidays, overnight)
    resampled = resampled.dropna(subset=['close'])
    return resampled


def prepare_backtest_data(tsla_path: str, spy_path: str = None) -> dict:
    """
    Load minute data and prepare all timeframes needed for Channel Surfer.

    Args:
        tsla_path: Path to TSLAMin.txt
        spy_path: Optional path to SPYMin.txt

    Returns:
        dict with keys:
            'tsla_5min': 5-min TSLA DataFrame
            'higher_tf_data': {'1h': df, 'daily': df, '4h': df}
            'spy_5min': 5-min SPY DataFrame or None
            'tsla_1min': raw 1-min data (for reference)
    """
    print(f"Loading TSLA minute data from {tsla_path}...")
    tsla_1min = load_minute_data(tsla_path)
    print(f"  Loaded {len(tsla_1min):,} 1-min bars: {tsla_1min.index[0]} to {tsla_1min.index[-1]}")

    # Resample to 5-min (primary trading timeframe)
    print("  Resampling to 5-min...")
    tsla_5min = resample_to_tf(tsla_1min, '5min')
    print(f"  5-min: {len(tsla_5min):,} bars")

    # Higher timeframes for channel context
    higher_tf_data = {}

    print("  Resampling to 1h...")
    higher_tf_data['1h'] = resample_to_tf(tsla_1min, '1h')
    print(f"  1h: {len(higher_tf_data['1h']):,} bars")

    print("  Resampling to 4h...")
    higher_tf_data['4h'] = resample_to_tf(tsla_1min, '4h')
    print(f"  4h: {len(higher_tf_data['4h']):,} bars")

    print("  Resampling to daily...")
    higher_tf_data['daily'] = resample_to_tf(tsla_1min, '1D')
    print(f"  daily: {len(higher_tf_data['daily']):,} bars")

    # SPY (optional -- physics mode doesn't need it)
    spy_5min = None
    if spy_path:
        print(f"Loading SPY minute data from {spy_path}...")
        spy_1min = load_minute_data(spy_path)
        print(f"  Loaded {len(spy_1min):,} 1-min bars")
        spy_5min = resample_to_tf(spy_1min, '5min')
        print(f"  SPY 5-min: {len(spy_5min):,} bars")

    return {
        'tsla_5min': tsla_5min,
        'higher_tf_data': higher_tf_data,
        'spy_5min': spy_5min,
        'tsla_1min': tsla_1min,
    }


def slice_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Extract a single calendar year from a DataFrame with DatetimeIndex."""
    return df[df.index.year == year]


def prepare_year_data(full_data: dict, year: int) -> dict:
    """
    Slice all dataframes to a single calendar year.

    Higher TF data gets a 60-day lookback buffer before the year starts,
    so channel detection has enough history at year boundaries.
    """
    tsla_5min = slice_year(full_data['tsla_5min'], year)
    if len(tsla_5min) == 0:
        return None

    # For higher TFs, include lookback buffer (60 days before year start)
    year_start = pd.Timestamp(f'{year}-01-01', tz='US/Eastern')
    buffer_start = year_start - pd.Timedelta(days=60)
    year_end = pd.Timestamp(f'{year}-12-31 23:59:59', tz='US/Eastern')

    higher_tf_data = {}
    for tf_name, tf_df in full_data['higher_tf_data'].items():
        higher_tf_data[tf_name] = tf_df[(tf_df.index >= buffer_start) & (tf_df.index <= year_end)]

    spy_5min = None
    if full_data.get('spy_5min') is not None:
        spy_5min = slice_year(full_data['spy_5min'], year)

    return {
        'tsla_5min': tsla_5min,
        'higher_tf_data': higher_tf_data,
        'spy_5min': spy_5min,
    }
