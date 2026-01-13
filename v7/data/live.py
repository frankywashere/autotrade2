"""
Live Data Module for v7 Dashboard Integration

Provides fetch_live_data() function to replace load_data() in dashboard.py.
Integrates yfinance live data with historical CSV files.

Supports two modes:
1. CSV + yfinance: Full historical data from CSVs merged with recent live data
2. yfinance-only: Fetches data using native intervals when CSVs are unavailable
   (e.g., on Streamlit Cloud)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


# yfinance native intervals and their data availability limits
# (mirrored from live_data_fetcher.py for convenience)
NATIVE_INTERVALS = {
    '1m': {'max_days': 7, 'period_str': '7d'},
    '5m': {'max_days': 60, 'period_str': '60d'},
    '15m': {'max_days': 60, 'period_str': '60d'},
    '30m': {'max_days': 60, 'period_str': '60d'},
    '1h': {'max_days': 730, 'period_str': '730d'},
    '1d': {'max_days': None, 'period_str': 'max'},
    '1wk': {'max_days': None, 'period_str': 'max'},
    '1mo': {'max_days': None, 'period_str': 'max'},
}


@dataclass
class LiveDataResult:
    """Container for live data fetch results."""
    tsla_df: pd.DataFrame
    spy_df: pd.DataFrame
    vix_df: pd.DataFrame
    status: str  # 'LIVE', 'RECENT', 'STALE', 'HISTORICAL', 'YFINANCE_ONLY'
    timestamp: pd.Timestamp
    data_age_minutes: float


def fetch_live_data(
    lookback_days: int = 500,
    data_dir: Optional[Path] = None,
    force_historical: bool = False
) -> LiveDataResult:
    """
    Fetch live market data, merging yfinance with historical CSV files.

    This is a drop-in replacement for dashboard.py's load_data() function.

    Supports two modes:
    1. CSV mode: If CSV files exist, loads historical data and merges with
       7-day live data from yfinance for freshness.
    2. yfinance-only mode: If CSVs are missing (e.g., Streamlit Cloud),
       fetches data using native yfinance intervals:
       - 5m interval: 60 days (for 5min through 4h timeframes via resampling)
       - 1h interval: 730 days (for hourly data with more history)
       - 1d interval: unlimited (for daily/weekly/monthly)

    Note:
        The default lookback of 500 days ensures training compatibility.
        A minimum of 420 days is required for proper model training
        (365 days for training window + 55 days for walk-forward validation).
        When running in yfinance-only mode, only ~60 days of intraday data
        is available, so training features may be limited.

    Args:
        lookback_days: Days of historical data to load (minimum 420 for training)
        data_dir: Directory containing CSV files (default: ./data)
        force_historical: If True, skip yfinance and use only CSV data

    Returns:
        LiveDataResult containing tsla_df, spy_df, vix_df and metadata
        Status will be one of:
        - 'LIVE': CSV data + fresh yfinance data (< 15 min old)
        - 'RECENT': CSV data + yfinance data (< 60 min old)
        - 'STALE': CSV data + older yfinance data
        - 'HISTORICAL': CSV data only (no yfinance fetch)
        - 'YFINANCE_ONLY': No CSVs, using yfinance native intervals

    Example:
        >>> result = fetch_live_data(lookback_days=500)
        >>> tsla_df = result.tsla_df
        >>> spy_df = result.spy_df
        >>> vix_df = result.vix_df
        >>> print(f"Data status: {result.status}")
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / 'data'

    # CSV file paths
    tsla_csv = data_dir / 'TSLA_1min.csv'
    spy_csv = data_dir / 'SPY_1min.csv'
    vix_csv = data_dir / 'VIX_History.csv'

    cutoff = datetime.now() - timedelta(days=lookback_days)

    # Load historical data from CSV (gracefully handles missing files)
    tsla_hist = _load_csv(tsla_csv, cutoff)
    spy_hist = _load_csv(spy_csv, cutoff)
    vix_hist = _load_vix_csv(vix_csv, cutoff)

    # Check if CSVs are available
    csvs_available = len(tsla_hist) > 0 and len(spy_hist) > 0

    # Try to fetch live data unless force_historical is set
    data_status = 'HISTORICAL'
    data_age_minutes = float('inf')

    if not force_historical:
        if csvs_available:
            # CSV mode: merge with 7-day live data for freshness
            try:
                tsla_live, spy_live = _fetch_yfinance_data()

                if tsla_live is not None and spy_live is not None:
                    # Merge live with historical
                    tsla_hist = _merge_historical_live(tsla_hist, tsla_live)
                    spy_hist = _merge_historical_live(spy_hist, spy_live)

                    # Check freshness
                    latest_timestamp = tsla_hist.index[-1]
                    data_age_minutes = (datetime.now() - latest_timestamp).total_seconds() / 60

                    if data_age_minutes < 15:
                        data_status = 'LIVE'
                    elif data_age_minutes < 60:
                        data_status = 'RECENT'
                    else:
                        data_status = 'STALE'
            except Exception as e:
                print(f"Warning: Could not fetch live data from yfinance: {e}")
                print("Falling back to historical CSV data only")
        else:
            # yfinance-only mode: fetch using native intervals
            print("CSVs not found, fetching data from yfinance native intervals...")
            try:
                tsla_hist, spy_hist, vix_hist = _fetch_native_interval_data(lookback_days)

                if len(tsla_hist) > 0:
                    # Try to get fresh 1-minute data for the last 7 days
                    try:
                        tsla_live, spy_live = _fetch_yfinance_data()
                        if tsla_live is not None and spy_live is not None:
                            tsla_hist = _merge_historical_live(tsla_hist, tsla_live)
                            spy_hist = _merge_historical_live(spy_hist, spy_live)
                    except Exception as e:
                        print(f"Warning: Could not fetch 1-minute live data: {e}")

                    # Check freshness
                    latest_timestamp = tsla_hist.index[-1]
                    data_age_minutes = (datetime.now() - latest_timestamp).total_seconds() / 60
                    data_status = 'YFINANCE_ONLY'
                else:
                    print("Error: Could not fetch any data from yfinance")
                    data_status = 'NO_DATA'

            except Exception as e:
                print(f"Error fetching native interval data: {e}")
                data_status = 'NO_DATA'

    # Handle case where we have no data at all
    if len(tsla_hist) == 0 or len(spy_hist) == 0:
        # Return empty result
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        return LiveDataResult(
            tsla_df=empty_df,
            spy_df=empty_df,
            vix_df=vix_hist if len(vix_hist) > 0 else empty_df,
            status='NO_DATA',
            timestamp=pd.Timestamp.now(),
            data_age_minutes=float('inf')
        )

    # Resample to 5min (dashboard expects 5min data)
    tsla_5min = _resample_to_5min(tsla_hist)
    spy_5min = _resample_to_5min(spy_hist)

    # Get timestamp
    timestamp = tsla_5min.index[-1]

    return LiveDataResult(
        tsla_df=tsla_5min,
        spy_df=spy_5min,
        vix_df=vix_hist,
        status=data_status,
        timestamp=timestamp,
        data_age_minutes=data_age_minutes
    )


def _load_csv(csv_path: Path, cutoff: datetime) -> pd.DataFrame:
    """
    Load CSV and filter by cutoff date.

    Returns empty DataFrame if file doesn't exist or can't be read.
    This allows graceful fallback to yfinance-only mode.
    """
    try:
        if not csv_path.exists():
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.columns = df.columns.str.lower()
        df = df[df.index >= cutoff]
        return df
    except Exception as e:
        print(f"Warning: Could not load CSV {csv_path}: {e}")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])


def _load_vix_csv(csv_path: Path, cutoff: datetime) -> pd.DataFrame:
    """
    Load VIX CSV (different format).

    Returns empty DataFrame if file doesn't exist or can't be read.
    This allows graceful fallback to yfinance-only mode.
    """
    try:
        if not csv_path.exists():
            return pd.DataFrame(columns=['open', 'high', 'low', 'close'])

        df = pd.read_csv(csv_path, parse_dates=['DATE'])
        df.set_index('DATE', inplace=True)
        df.columns = df.columns.str.lower()
        df = df[df.index >= cutoff]
        return df
    except Exception as e:
        print(f"Warning: Could not load VIX CSV {csv_path}: {e}")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close'])


def _fetch_yfinance_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fetch latest data from yfinance (1min resolution, last 7 days).

    Returns:
        (tsla_df, spy_df) or (None, None) if failed
    """
    try:
        # Fetch TSLA 1min data (yfinance limit: 7 days)
        tsla = yf.Ticker('TSLA')
        tsla_df = tsla.history(period='7d', interval='1m')

        if len(tsla_df) == 0:
            return None, None

        # Fetch SPY 1min data
        spy = yf.Ticker('SPY')
        spy_df = spy.history(period='7d', interval='1m')

        if len(spy_df) == 0:
            return None, None

        # Format to match CSV structure
        tsla_df = _format_yfinance_df(tsla_df)
        spy_df = _format_yfinance_df(spy_df)

        return tsla_df, spy_df

    except Exception as e:
        print(f"Error fetching from yfinance: {e}")
        return None, None


def _fetch_native_interval_data(
    lookback_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch data using yfinance native intervals when CSVs are unavailable.

    This is used for Streamlit Cloud deployment where CSVs don't exist.
    Uses native intervals to get maximum history:
    - 5m: 60 days (resampled for 5min through 4h timeframes)
    - 1h: 730 days (2 years of hourly data)
    - 1d: unlimited (for daily/weekly/monthly)

    The data from different intervals is merged together, with finer
    granularity data taking precedence for overlapping periods.

    Args:
        lookback_days: Requested lookback days (may not be fully satisfied)

    Returns:
        Tuple of (tsla_df, spy_df, vix_df) DataFrames
        All at 1-minute equivalent resolution (for 5m data)
    """
    tsla_dfs = []
    spy_dfs = []

    print(f"Fetching TSLA and SPY data from yfinance native intervals...")

    # Fetch daily data (unlimited history) - for longer lookbacks
    if lookback_days > 60:
        try:
            print("  Fetching 1d interval data...")
            tsla = yf.Ticker('TSLA')
            spy = yf.Ticker('SPY')

            # Calculate how many years we need
            years_needed = min((lookback_days // 365) + 1, 10)  # Cap at 10 years
            period = f'{years_needed}y' if years_needed > 2 else '730d'

            tsla_1d = tsla.history(period=period, interval='1d')
            spy_1d = spy.history(period=period, interval='1d')

            if len(tsla_1d) > 0:
                tsla_1d = _format_yfinance_df(tsla_1d)
                tsla_dfs.append(('1d', tsla_1d))
                print(f"    TSLA 1d: {len(tsla_1d)} rows")

            if len(spy_1d) > 0:
                spy_1d = _format_yfinance_df(spy_1d)
                spy_dfs.append(('1d', spy_1d))
                print(f"    SPY 1d: {len(spy_1d)} rows")

        except Exception as e:
            print(f"  Warning: Could not fetch 1d data: {e}")

    # Fetch hourly data (730 days) - for medium-term history
    try:
        print("  Fetching 1h interval data...")
        tsla = yf.Ticker('TSLA')
        spy = yf.Ticker('SPY')

        tsla_1h = tsla.history(period='730d', interval='1h')
        spy_1h = spy.history(period='730d', interval='1h')

        if len(tsla_1h) > 0:
            tsla_1h = _format_yfinance_df(tsla_1h)
            tsla_dfs.append(('1h', tsla_1h))
            print(f"    TSLA 1h: {len(tsla_1h)} rows")

        if len(spy_1h) > 0:
            spy_1h = _format_yfinance_df(spy_1h)
            spy_dfs.append(('1h', spy_1h))
            print(f"    SPY 1h: {len(spy_1h)} rows")

    except Exception as e:
        print(f"  Warning: Could not fetch 1h data: {e}")

    # Fetch 5-minute data (60 days) - finest granularity for recent history
    try:
        print("  Fetching 5m interval data...")
        tsla = yf.Ticker('TSLA')
        spy = yf.Ticker('SPY')

        tsla_5m = tsla.history(period='60d', interval='5m')
        spy_5m = spy.history(period='60d', interval='5m')

        if len(tsla_5m) > 0:
            tsla_5m = _format_yfinance_df(tsla_5m)
            tsla_dfs.append(('5m', tsla_5m))
            print(f"    TSLA 5m: {len(tsla_5m)} rows")

        if len(spy_5m) > 0:
            spy_5m = _format_yfinance_df(spy_5m)
            spy_dfs.append(('5m', spy_5m))
            print(f"    SPY 5m: {len(spy_5m)} rows")

    except Exception as e:
        print(f"  Warning: Could not fetch 5m data: {e}")

    # Merge data from different intervals
    # Finer granularity takes precedence for overlapping periods
    tsla_merged = _merge_multi_interval_data(tsla_dfs)
    spy_merged = _merge_multi_interval_data(spy_dfs)

    # Fetch VIX data (daily only for VIX)
    vix_df = pd.DataFrame()
    try:
        print("  Fetching VIX data...")
        vix = yf.Ticker('^VIX')
        years_needed = min((lookback_days // 365) + 1, 10)
        period = f'{years_needed}y' if years_needed > 2 else 'max'
        vix_df = vix.history(period=period, interval='1d')

        if len(vix_df) > 0:
            vix_df = _format_yfinance_df(vix_df)
            print(f"    VIX: {len(vix_df)} rows")

    except Exception as e:
        print(f"  Warning: Could not fetch VIX data: {e}")

    print(f"  Final merged data:")
    print(f"    TSLA: {len(tsla_merged)} rows")
    print(f"    SPY: {len(spy_merged)} rows")
    print(f"    VIX: {len(vix_df)} rows")

    return tsla_merged, spy_merged, vix_df


def _merge_multi_interval_data(
    interval_dfs: list
) -> pd.DataFrame:
    """
    Merge DataFrames from multiple intervals into a single DataFrame.

    Finer granularity data takes precedence for overlapping time periods.
    Intervals are processed in order from coarsest to finest (1d -> 1h -> 5m).

    Args:
        interval_dfs: List of (interval_name, DataFrame) tuples

    Returns:
        Merged DataFrame with finest available granularity
    """
    if not interval_dfs:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    # Sort by interval granularity (coarsest first)
    interval_order = {'1mo': 0, '1wk': 1, '1d': 2, '1h': 3, '30m': 4, '15m': 5, '5m': 6, '1m': 7}
    interval_dfs_sorted = sorted(
        interval_dfs,
        key=lambda x: interval_order.get(x[0], 99)
    )

    # Start with coarsest data
    merged = pd.DataFrame()

    for interval_name, df in interval_dfs_sorted:
        if df.empty:
            continue

        if merged.empty:
            merged = df.copy()
        else:
            # For overlapping periods, finer granularity data takes precedence
            # Keep coarse data only where we don't have finer data
            df_start = df.index.min()
            merged_before = merged[merged.index < df_start]

            # Concatenate: older coarse data + newer fine data
            merged = pd.concat([merged_before, df])
            merged = merged.sort_index()

            # Remove any duplicates (keep last = finer data)
            merged = merged[~merged.index.duplicated(keep='last')]

    return merged


def _format_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format yfinance DataFrame to match CSV format."""
    df = df.copy()

    # Remove timezone to match CSV format
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Lowercase columns
    df.columns = [c.lower() for c in df.columns]

    # Select only OHLCV columns
    return df[['open', 'high', 'low', 'close', 'volume']]


def _merge_historical_live(historical: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """
    Merge historical and live data, removing duplicates.
    Live data takes precedence for overlapping timestamps.
    """
    if len(historical) == 0:
        return live

    if len(live) == 0:
        return historical

    # Remove historical data that overlaps with live
    live_start = live.index[0]
    historical_clean = historical[historical.index < live_start]

    # Concatenate
    merged = pd.concat([historical_clean, live])
    merged = merged.sort_index()

    # Remove duplicates (keep last)
    merged = merged[~merged.index.duplicated(keep='last')]

    return merged


def _resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1min data to 5min."""
    resampled = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return resampled


# Backward compatibility: provide separate function that returns tuple
def load_live_data_tuple(lookback_days: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible version that returns tuple like old load_data().

    Note:
        Default of 500 days ensures training compatibility (420-day minimum required).

    Returns:
        (tsla_df, spy_df, vix_df) tuple
    """
    result = fetch_live_data(lookback_days=lookback_days)
    return result.tsla_df, result.spy_df, result.vix_df


def is_market_open() -> bool:
    """Check if US stock market is currently open."""
    now = datetime.now()

    # Weekend check
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # Market hours: 9:30 AM - 4:00 PM ET
    # (Simplified - doesn't account for holidays)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now <= market_close
