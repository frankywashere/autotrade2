"""
Live Data Module for v7 Dashboard Integration

Provides fetch_live_data() function to replace load_data() in dashboard.py.
Integrates yfinance live data with historical CSV files.

Uses LiveDataFetcher for optimal interval selection to maximize API data:
- 1m: up to 7 days
- 5m: up to 60 days
- 1h: up to 730 days (2 years)
- 1d: unlimited
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

# Import LiveDataFetcher for native interval support
from .live_data_fetcher import LiveDataFetcher, MultiSymbolFetcher


# yfinance API limits by interval (mirrored from live_data_fetcher for reference)
YFINANCE_API_LIMITS = {
    '1m': 7,      # 1-minute: last 7 days
    '5m': 60,     # 5-minute: last 60 days
    '15m': 60,    # 15-minute: last 60 days
    '30m': 60,    # 30-minute: last 60 days
    '1h': 730,    # 1-hour: last 730 days (2 years)
    '1d': None,   # Daily: all available (unlimited)
}


@dataclass
class LiveDataResult:
    """Container for live data fetch results."""
    tsla_df: pd.DataFrame
    spy_df: pd.DataFrame
    vix_df: pd.DataFrame
    status: str  # 'LIVE', 'RECENT', 'STALE', 'HISTORICAL'
    timestamp: pd.Timestamp
    data_age_minutes: float
    live_data_days: int = 0  # How many days of live data were fetched


def fetch_live_data(
    lookback_days: int = 500,
    data_dir: Optional[Path] = None,
    force_historical: bool = False
) -> LiveDataResult:
    """
    Fetch live market data, merging yfinance with historical CSV files.

    This is a drop-in replacement for dashboard.py's load_data() function.

    Note:
        Default lookback of 500 days ensures compatibility with training pipelines
        that require a minimum of 420 days for proper walk-forward validation
        with sufficient history for feature computation and model training.

    Args:
        lookback_days: Days of historical data to load (minimum 420 for training)
        data_dir: Directory containing CSV files (default: ./data)
        force_historical: If True, skip yfinance and use only CSV data

    Returns:
        LiveDataResult containing tsla_df, spy_df, vix_df and metadata

    Example:
        >>> result = fetch_live_data(lookback_days=90)
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

    # Load historical data from CSV
    tsla_hist = _load_csv(tsla_csv, cutoff)
    spy_hist = _load_csv(spy_csv, cutoff)
    vix_hist = _load_vix_csv(vix_csv, cutoff)

    # Try to fetch live data unless force_historical is set
    data_status = 'HISTORICAL'
    data_age_minutes = float('inf')
    live_data_days = 0

    if not force_historical:
        try:
            # Pass lookback_days to fetch optimal interval (up to 730 days of 1h data)
            tsla_live, spy_live, live_data_days = _fetch_yfinance_data(lookback_days=lookback_days)

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
        data_age_minutes=data_age_minutes,
        live_data_days=live_data_days
    )


def _load_csv(csv_path: Path, cutoff: datetime) -> pd.DataFrame:
    """Load CSV and filter by cutoff date."""
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.columns = df.columns.str.lower()
    df = df[df.index >= cutoff]
    return df


def _load_vix_csv(csv_path: Path, cutoff: datetime) -> pd.DataFrame:
    """Load VIX CSV (different format)."""
    df = pd.read_csv(csv_path, parse_dates=['DATE'])
    df.set_index('DATE', inplace=True)
    df.columns = df.columns.str.lower()
    df = df[df.index >= cutoff]
    return df


def _fetch_yfinance_data(lookback_days: int = 7) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], int]:
    """
    Fetch latest data from yfinance using optimal interval for requested lookback.

    Uses LiveDataFetcher to select the best interval based on API limits:
    - <= 7 days: 1m interval (highest resolution)
    - <= 60 days: 5m interval
    - <= 730 days: 1h interval
    - > 730 days: 1d interval (no limit)

    Args:
        lookback_days: Number of days of live data to fetch

    Returns:
        (tsla_df, spy_df, actual_days) or (None, None, 0) if failed
        actual_days indicates how many days of live data were actually fetched
    """
    try:
        # Select optimal interval based on lookback_days
        if lookback_days <= 7:
            interval = '1m'
            fetch_days = min(lookback_days, 7)
        elif lookback_days <= 60:
            interval = '5m'
            fetch_days = min(lookback_days, 60)
        elif lookback_days <= 730:
            interval = '1h'
            fetch_days = min(lookback_days, 730)
        else:
            interval = '1d'
            fetch_days = lookback_days  # No limit for daily

        # Use LiveDataFetcher for robust fetching
        tsla_fetcher = LiveDataFetcher('TSLA')
        spy_fetcher = LiveDataFetcher('SPY')

        tsla_df = tsla_fetcher.fetch(interval=interval, days_back=fetch_days)

        if tsla_df.empty:
            return None, None, 0

        spy_df = spy_fetcher.fetch(interval=interval, days_back=fetch_days)

        if spy_df.empty:
            return None, None, 0

        # Note: LiveDataFetcher already normalizes the DataFrame
        # (lowercase columns, timezone removed, OHLCV only)

        return tsla_df, spy_df, fetch_days

    except Exception as e:
        print(f"Error fetching from yfinance: {e}")
        return None, None, 0


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
