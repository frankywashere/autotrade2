"""
Data loader with validation - no silent failures.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple

from ..exceptions import DataLoadError


def load_market_data(
    data_dir: str,
    validate: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load TSLA, SPY, VIX market data.

    Args:
        data_dir: Directory containing CSV files
        validate: If True, validate data integrity

    Returns:
        Tuple of (tsla_df, spy_df, vix_df) all at 5min resolution

    Raises:
        DataLoadError: If any data loading or validation fails
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise DataLoadError(f"Data directory does not exist: {data_path}")

    if not data_path.is_dir():
        raise DataLoadError(f"Path is not a directory: {data_path}")

    # Load TSLA data
    tsla_df = _load_csv(data_path, "TSLA_1min.csv", "TSLA")
    tsla_df = _prepare_ohlcv(tsla_df, "TSLA")
    tsla_df = _resample_to_5min(tsla_df, "TSLA")

    if validate:
        validate_ohlcv(tsla_df, "TSLA")

    # Load SPY data
    spy_df = _load_csv(data_path, "SPY_1min.csv", "SPY")
    spy_df = _prepare_ohlcv(spy_df, "SPY")
    spy_df = _resample_to_5min(spy_df, "SPY")

    if validate:
        validate_ohlcv(spy_df, "SPY")

    # Load VIX data (different format - daily data)
    vix_df = _load_vix(data_path)

    if validate:
        # VIX is a calculated volatility index, not a traded instrument.
        # Its OHLC data can legitimately have high < open or high < close.
        validate_ohlcv(vix_df, "VIX", require_volume=False, strict_ohlc=False)

    # Align SPY and VIX to TSLA index
    tsla_aligned, spy_aligned, vix_aligned = _align_to_tsla(tsla_df, spy_df, vix_df)

    # Final validation after alignment
    if validate:
        _validate_alignment(tsla_aligned, spy_aligned, vix_aligned)

    return tsla_aligned, spy_aligned, vix_aligned


def validate_ohlcv(df: pd.DataFrame, name: str, require_volume: bool = True, strict_ohlc: bool = True) -> None:
    """
    Validate OHLCV DataFrame has required columns and no NaN.

    Args:
        df: DataFrame to validate
        name: Name for error messages
        require_volume: If True, require volume column
        strict_ohlc: If True, enforce strict OHLC relationships (high >= open/close, low <= open/close).
                     Set to False for indices like VIX where these relationships may not hold.

    Raises:
        DataLoadError: If validation fails
    """
    required_cols = ['open', 'high', 'low', 'close']
    if require_volume:
        required_cols.append('volume')

    # Check columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise DataLoadError(
            f"{name}: Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Check for empty DataFrame
    if df.empty:
        raise DataLoadError(f"{name}: DataFrame is empty after loading")

    # Check no NaN in OHLC columns
    ohlc_cols = ['open', 'high', 'low', 'close']
    for col in ohlc_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            nan_pct = (nan_count / len(df)) * 100
            first_nan_idx = df[df[col].isna()].index[0]
            raise DataLoadError(
                f"{name}: Column '{col}' has {nan_count} NaN values ({nan_pct:.2f}%). "
                f"First NaN at index: {first_nan_idx}"
            )

    # Fill volume NaN with 0 if present
    if 'volume' in df.columns:
        nan_volume = df['volume'].isna().sum()
        if nan_volume > 0:
            df['volume'] = df['volume'].fillna(0)

    # Check high >= low
    invalid_hl = df[df['high'] < df['low']]
    if len(invalid_hl) > 0:
        first_bad = invalid_hl.iloc[0]
        raise DataLoadError(
            f"{name}: Found {len(invalid_hl)} rows where high < low. "
            f"First occurrence at {invalid_hl.index[0]}: "
            f"high={first_bad['high']}, low={first_bad['low']}"
        )

    # Strict OHLC checks - only for traded instruments, not calculated indices like VIX
    if strict_ohlc:
        # Check high >= open and high >= close
        invalid_high_open = df[df['high'] < df['open']]
        if len(invalid_high_open) > 0:
            first_bad = invalid_high_open.iloc[0]
            raise DataLoadError(
                f"{name}: Found {len(invalid_high_open)} rows where high < open. "
                f"First occurrence at {invalid_high_open.index[0]}: "
                f"high={first_bad['high']}, open={first_bad['open']}"
            )

        invalid_high_close = df[df['high'] < df['close']]
        if len(invalid_high_close) > 0:
            first_bad = invalid_high_close.iloc[0]
            raise DataLoadError(
                f"{name}: Found {len(invalid_high_close)} rows where high < close. "
                f"First occurrence at {invalid_high_close.index[0]}: "
                f"high={first_bad['high']}, close={first_bad['close']}"
            )

        # Check low <= open and low <= close
        invalid_low_open = df[df['low'] > df['open']]
        if len(invalid_low_open) > 0:
            first_bad = invalid_low_open.iloc[0]
            raise DataLoadError(
                f"{name}: Found {len(invalid_low_open)} rows where low > open. "
                f"First occurrence at {invalid_low_open.index[0]}: "
                f"low={first_bad['low']}, open={first_bad['open']}"
            )

        invalid_low_close = df[df['low'] > df['close']]
        if len(invalid_low_close) > 0:
            first_bad = invalid_low_close.iloc[0]
            raise DataLoadError(
                f"{name}: Found {len(invalid_low_close)} rows where low > close. "
                f"First occurrence at {invalid_low_close.index[0]}: "
                f"low={first_bad['low']}, close={first_bad['close']}"
            )

    # Check for non-positive prices
    for col in ohlc_cols:
        non_positive = df[df[col] <= 0]
        if len(non_positive) > 0:
            first_bad = non_positive.iloc[0]
            raise DataLoadError(
                f"{name}: Found {len(non_positive)} rows with non-positive {col}. "
                f"First occurrence at {non_positive.index[0]}: {col}={first_bad[col]}"
            )

    # Check for infinite values
    for col in ohlc_cols:
        inf_count = df[col].apply(lambda x: pd.isna(x) or (isinstance(x, float) and (x == float('inf') or x == float('-inf')))).sum()
        actual_inf = df[~df[col].isna() & df[col].apply(lambda x: isinstance(x, float) and (x == float('inf') or x == float('-inf')))]
        if len(actual_inf) > 0:
            raise DataLoadError(
                f"{name}: Found {len(actual_inf)} infinite values in column '{col}'. "
                f"First occurrence at {actual_inf.index[0]}"
            )


def _load_csv(data_path: Path, filename: str, name: str) -> pd.DataFrame:
    """Load a CSV file with error handling."""
    filepath = data_path / filename

    if not filepath.exists():
        raise DataLoadError(
            f"{name}: File not found: {filepath}. "
            f"Expected file: {filename}"
        )

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise DataLoadError(
            f"{name}: Failed to read CSV file {filepath}: {e}"
        )

    if df.empty:
        raise DataLoadError(f"{name}: CSV file is empty: {filepath}")

    return df


def _prepare_ohlcv(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Prepare OHLCV DataFrame with proper index and columns."""
    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Check for timestamp column
    timestamp_col = None
    for col in ['timestamp', 'date', 'datetime', 'time']:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        raise DataLoadError(
            f"{name}: No timestamp column found. "
            f"Expected one of: timestamp, date, datetime, time. "
            f"Available columns: {list(df.columns)}"
        )

    # Parse timestamp
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    except Exception as e:
        raise DataLoadError(
            f"{name}: Failed to parse timestamp column '{timestamp_col}': {e}"
        )

    # Set index
    df.set_index(timestamp_col, inplace=True)
    df.sort_index(inplace=True)

    # Check for duplicate indices
    duplicates = df.index.duplicated()
    if duplicates.any():
        dup_count = duplicates.sum()
        first_dup = df.index[duplicates][0]
        raise DataLoadError(
            f"{name}: Found {dup_count} duplicate timestamps. "
            f"First duplicate: {first_dup}"
        )

    return df


def _resample_to_5min(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Resample 1-minute OHLCV data to 5-minute bars."""
    try:
        resampled = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    except KeyError as e:
        raise DataLoadError(
            f"{name}: Missing required column for resampling: {e}. "
            f"Available columns: {list(df.columns)}"
        )
    except Exception as e:
        raise DataLoadError(
            f"{name}: Failed to resample to 5-minute bars: {e}"
        )

    if resampled.empty:
        raise DataLoadError(
            f"{name}: Resampling resulted in empty DataFrame. "
            f"Original shape: {df.shape}"
        )

    return resampled


def _load_vix(data_path: Path) -> pd.DataFrame:
    """Load VIX data with special handling for its format."""
    filepath = data_path / "VIX_History.csv"

    if not filepath.exists():
        raise DataLoadError(
            f"VIX: File not found: {filepath}. "
            f"Expected file: VIX_History.csv"
        )

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise DataLoadError(f"VIX: Failed to read CSV file {filepath}: {e}")

    if df.empty:
        raise DataLoadError(f"VIX: CSV file is empty: {filepath}")

    # Standardize column names
    df.columns = [col.lower() for col in df.columns]

    # VIX uses DATE column with MM/DD/YYYY format
    if 'date' not in df.columns:
        raise DataLoadError(
            f"VIX: Missing 'date' column. "
            f"Available columns: {list(df.columns)}"
        )

    try:
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    except Exception as e:
        # Try alternative format
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e2:
            raise DataLoadError(
                f"VIX: Failed to parse date column. "
                f"Tried MM/DD/YYYY format: {e}. "
                f"Also tried auto-detect: {e2}"
            )

    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # Check for duplicate indices
    duplicates = df.index.duplicated()
    if duplicates.any():
        dup_count = duplicates.sum()
        first_dup = df.index[duplicates][0]
        raise DataLoadError(
            f"VIX: Found {dup_count} duplicate dates. "
            f"First duplicate: {first_dup}"
        )

    return df


def _align_to_tsla(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align SPY and VIX to TSLA's index using forward-fill.

    Raises:
        DataLoadError: If alignment fails or no overlapping dates
    """
    # Find common date range
    tsla_dates = set(tsla_df.index.date)
    spy_dates = set(spy_df.index.date)
    vix_dates = set(vix_df.index.date)

    common_dates = tsla_dates & spy_dates & vix_dates

    if not common_dates:
        raise DataLoadError(
            f"No overlapping dates found between TSLA, SPY, and VIX. "
            f"TSLA: {min(tsla_dates)} to {max(tsla_dates)} ({len(tsla_dates)} dates). "
            f"SPY: {min(spy_dates)} to {max(spy_dates)} ({len(spy_dates)} dates). "
            f"VIX: {min(vix_dates)} to {max(vix_dates)} ({len(vix_dates)} dates)."
        )

    start_date = min(common_dates)
    end_date = max(common_dates)

    # Filter to common date range
    tsla_filtered = tsla_df[
        (tsla_df.index.date >= start_date) &
        (tsla_df.index.date <= end_date)
    ]
    spy_filtered = spy_df[
        (spy_df.index.date >= start_date) &
        (spy_df.index.date <= end_date)
    ]
    vix_filtered = vix_df[
        (vix_df.index.date >= start_date) &
        (vix_df.index.date <= end_date)
    ]

    if tsla_filtered.empty:
        raise DataLoadError(
            f"TSLA has no data in common date range: {start_date} to {end_date}"
        )

    # Reindex SPY and VIX to TSLA's index with forward-fill
    spy_aligned = spy_filtered.reindex(tsla_filtered.index, method='ffill')
    vix_aligned = vix_filtered.reindex(tsla_filtered.index, method='ffill')

    # Remove rows with NaN (at start before ffill has data)
    valid_mask = (
        ~tsla_filtered.isna().any(axis=1) &
        ~spy_aligned.isna().any(axis=1) &
        ~vix_aligned.isna().any(axis=1)
    )

    tsla_result = tsla_filtered[valid_mask].copy()
    spy_result = spy_aligned[valid_mask].copy()
    vix_result = vix_aligned[valid_mask].copy()

    if tsla_result.empty:
        nan_tsla = tsla_filtered.isna().any(axis=1).sum()
        nan_spy = spy_aligned.isna().any(axis=1).sum()
        nan_vix = vix_aligned.isna().any(axis=1).sum()
        raise DataLoadError(
            f"No valid rows after alignment. "
            f"NaN rows - TSLA: {nan_tsla}, SPY: {nan_spy}, VIX: {nan_vix}"
        )

    return tsla_result, spy_result, vix_result


def _validate_alignment(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame
) -> None:
    """Validate that all DataFrames are properly aligned."""
    # Check same length
    if len(tsla_df) != len(spy_df):
        raise DataLoadError(
            f"Length mismatch after alignment: TSLA={len(tsla_df)}, SPY={len(spy_df)}"
        )

    if len(tsla_df) != len(vix_df):
        raise DataLoadError(
            f"Length mismatch after alignment: TSLA={len(tsla_df)}, VIX={len(vix_df)}"
        )

    # Check same index
    if not tsla_df.index.equals(spy_df.index):
        diff_count = (tsla_df.index != spy_df.index).sum()
        raise DataLoadError(
            f"Index mismatch between TSLA and SPY: {diff_count} indices differ"
        )

    if not tsla_df.index.equals(vix_df.index):
        diff_count = (tsla_df.index != vix_df.index).sum()
        raise DataLoadError(
            f"Index mismatch between TSLA and VIX: {diff_count} indices differ"
        )

    # Check no remaining NaN in critical columns
    for name, df in [('TSLA', tsla_df), ('SPY', spy_df), ('VIX', vix_df)]:
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    raise DataLoadError(
                        f"{name}: {nan_count} NaN values remain in '{col}' after alignment"
                    )
