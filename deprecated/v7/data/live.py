"""
Live Data Module for v7 Dashboard Integration

Provides fetch_live_data() function to replace load_data() in dashboard.py.
Integrates yfinance live data with historical CSV files.

Unified behavior (both local and Streamlit deployments):
1. ALWAYS calls _fetch_all_native_timeframes() to get per-TF data
2. Uses the 5min native data as the base tsla_df/spy_df (backward compatible)
3. If CSVs exist and are fresh (<= 7 days old), merges 1-min CSV data for finer resolution
4. Stores all native TF data in result.native_tfs for direct access

The only difference between local and Streamlit is that local deployments
can merge fresher 1-minute CSV data when available.
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
    tsla_df: pd.DataFrame  # Base 5min DataFrame (backward compatible)
    spy_df: pd.DataFrame
    vix_df: pd.DataFrame
    status: str  # 'LIVE', 'RECENT', 'STALE', 'NATIVE_ONLY'
    timestamp: pd.Timestamp
    data_age_minutes: float
    native_tfs: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None  # Per-TF data from _fetch_all_native_timeframes()


def fetch_live_data(
    lookback_days: int = 500,
    data_dir: Optional[Path] = None,
    force_historical: bool = False
) -> LiveDataResult:
    """
    Fetch live market data using native yfinance intervals for all timeframes.

    This is a drop-in replacement for dashboard.py's load_data() function.

    Unified behavior (both local and Streamlit):
    1. ALWAYS call _fetch_all_native_timeframes() to get per-TF data
    2. Use the 5min native data as the base tsla_df/spy_df (backward compatible)
    3. If CSVs exist and are fresh, merge 1-min CSV data for finer resolution
    4. Store all native TF data in result.native_tfs

    Note:
        The default lookback of 500 days ensures training compatibility.
        A minimum of 420 days is required for proper model training
        (365 days for training window + 55 days for walk-forward validation).
        yfinance native intervals provide ~60 days of intraday data and
        unlimited daily/weekly/monthly data.

    Args:
        lookback_days: Days of historical data to load (minimum 420 for training)
        data_dir: Directory containing CSV files (default: ./data)
        force_historical: If True, skip yfinance fetch (only uses CSVs if available)

    Returns:
        LiveDataResult containing:
        - tsla_df, spy_df, vix_df: Base 5min DataFrames (backward compatible)
        - native_tfs: Dict with per-TF data for all 11 timeframes
        - status: 'LIVE', 'RECENT', 'STALE', or 'NATIVE_ONLY'

    Example:
        >>> result = fetch_live_data(lookback_days=500)
        >>> tsla_df = result.tsla_df  # Base 5min DataFrame
        >>> tsla_daily = result.native_tfs['tsla']['daily']  # Native daily data
        >>> print(f"Data status: {result.status}")
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / 'data'

    # CSV file paths
    tsla_csv = data_dir / 'TSLA_1min.csv'
    spy_csv = data_dir / 'SPY_1min.csv'
    vix_csv = data_dir / 'VIX_History.csv'

    cutoff = datetime.now() - timedelta(days=lookback_days)

    # Initialize status and age
    data_status = 'NATIVE_ONLY'
    data_age_minutes = float('inf')
    native_tfs = None

    if force_historical:
        # Force historical mode: only use CSVs, no yfinance
        tsla_hist = _load_csv(tsla_csv, cutoff)
        spy_hist = _load_csv(spy_csv, cutoff)
        vix_hist = _load_vix_csv(vix_csv, cutoff)

        if len(tsla_hist) == 0 or len(spy_hist) == 0:
            empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            return LiveDataResult(
                tsla_df=empty_df,
                spy_df=empty_df,
                vix_df=vix_hist if len(vix_hist) > 0 else empty_df,
                status='NO_DATA',
                timestamp=pd.Timestamp.now(),
                data_age_minutes=float('inf'),
                native_tfs=None
            )

        # Resample to 5min
        tsla_5min = _resample_to_5min(tsla_hist)
        spy_5min = _resample_to_5min(spy_hist)
        timestamp = tsla_5min.index[-1]

        return LiveDataResult(
            tsla_df=tsla_5min,
            spy_df=spy_5min,
            vix_df=vix_hist,
            status='HISTORICAL',
            timestamp=timestamp,
            data_age_minutes=float('inf'),
            native_tfs=None
        )

    # === UNIFIED APPROACH: Always fetch native TF data ===
    try:
        native_tfs = _fetch_all_native_timeframes()
    except Exception as e:
        print(f"Error fetching native timeframes: {e}")
        native_tfs = None

    # Get base 5min data from native TFs
    if native_tfs and '5min' in native_tfs.get('tsla', {}) and '5min' in native_tfs.get('spy', {}):
        tsla_5min = native_tfs['tsla']['5min'].copy()
        spy_5min = native_tfs['spy']['5min'].copy()
        vix_df = native_tfs.get('vix', pd.DataFrame())
        if isinstance(vix_df, dict):
            vix_df = pd.DataFrame()
    else:
        # Fallback: fetch 5min data directly
        print("Warning: Could not get 5min data from native TFs, fetching directly...")
        try:
            tsla = yf.Ticker('TSLA')
            spy = yf.Ticker('SPY')
            tsla_5min = _format_yfinance_df(tsla.history(period='60d', interval='5m'))
            spy_5min = _format_yfinance_df(spy.history(period='60d', interval='5m'))
            vix = yf.Ticker('^VIX')
            vix_df = _format_yfinance_df(vix.history(period='max', interval='1d'))
        except Exception as e:
            print(f"Error fetching fallback data: {e}")
            empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            return LiveDataResult(
                tsla_df=empty_df,
                spy_df=empty_df,
                vix_df=empty_df,
                status='NO_DATA',
                timestamp=pd.Timestamp.now(),
                data_age_minutes=float('inf'),
                native_tfs=native_tfs
            )

    # Check if CSVs exist and are fresh for merging 1-min data
    tsla_csv_data = _load_csv(tsla_csv, cutoff)
    spy_csv_data = _load_csv(spy_csv, cutoff)
    vix_csv_data = _load_vix_csv(vix_csv, cutoff)

    csvs_available = len(tsla_csv_data) > 0 and len(spy_csv_data) > 0

    if csvs_available:
        csv_end_date = tsla_csv_data.index[-1]
        days_since_csv_end = (datetime.now() - csv_end_date).days

        # Only merge if CSV data is reasonably fresh (within 7 days for 1-min resolution benefit)
        if days_since_csv_end <= 7:
            print(f"CSV data is fresh ({days_since_csv_end} days old), merging 1-min data for finer resolution...")

            # Fetch 7-day 1-minute data for maximum freshness
            try:
                tsla_1min, spy_1min = _fetch_yfinance_data()

                if tsla_1min is not None and spy_1min is not None:
                    # Merge CSV + 1-min yfinance data
                    tsla_merged = _merge_historical_live(tsla_csv_data, tsla_1min)
                    spy_merged = _merge_historical_live(spy_csv_data, spy_1min)

                    # Resample merged data to 5min
                    tsla_5min = _resample_to_5min(tsla_merged)
                    spy_5min = _resample_to_5min(spy_merged)

                    # Update native_tfs 5min with the merged data
                    if native_tfs:
                        native_tfs['tsla']['5min'] = tsla_5min
                        native_tfs['spy']['5min'] = spy_5min

                    print(f"  Merged CSV + 1-min yfinance data")
            except Exception as e:
                print(f"Warning: Could not fetch 1-minute live data: {e}")
                print("  Using native 5-min data without CSV merge")
        else:
            print(f"CSV data is stale ({days_since_csv_end} days old), using native TF data only")

        # Use VIX CSV if fresher than yfinance
        if len(vix_csv_data) > 0:
            if len(vix_df) == 0 or vix_csv_data.index[-1] > vix_df.index[-1]:
                vix_df = vix_csv_data
    else:
        print("CSVs not available, using native TF data only")

    # Handle empty data
    if len(tsla_5min) == 0 or len(spy_5min) == 0:
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        return LiveDataResult(
            tsla_df=empty_df,
            spy_df=empty_df,
            vix_df=vix_df if len(vix_df) > 0 else empty_df,
            status='NO_DATA',
            timestamp=pd.Timestamp.now(),
            data_age_minutes=float('inf'),
            native_tfs=native_tfs
        )

    # Determine data freshness and status
    timestamp = tsla_5min.index[-1]
    data_age_minutes = (datetime.now() - timestamp).total_seconds() / 60

    if csvs_available and (datetime.now() - tsla_csv_data.index[-1]).days <= 7:
        # We merged fresh CSV data
        if data_age_minutes < 15:
            data_status = 'LIVE'
        elif data_age_minutes < 60:
            data_status = 'RECENT'
        else:
            data_status = 'STALE'
    else:
        # Native TF data only
        data_status = 'NATIVE_ONLY'

    return LiveDataResult(
        tsla_df=tsla_5min,
        spy_df=spy_5min,
        vix_df=vix_df,
        status=data_status,
        timestamp=timestamp,
        data_age_minutes=data_age_minutes,
        native_tfs=native_tfs
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


def _fetch_all_native_timeframes() -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch data for ALL 11 system timeframes using yfinance native intervals.

    This function fetches data for TSLA, SPY, and VIX across all timeframes
    defined in v7.core.timeframe.TIMEFRAMES using optimal yfinance native
    intervals and resampling where necessary.

    yfinance native intervals and their limits:
        - 5m: 60 days
        - 15m: 60 days
        - 30m: 60 days
        - 1h: 730 days (2 years)
        - 1d: unlimited
        - 1wk: unlimited
        - 1mo: unlimited

    Mapping strategy:
        - 5min -> fetch 5m native (60 days)
        - 15min -> fetch 15m native (60 days)
        - 30min -> fetch 30m native (60 days)
        - 1h -> fetch 1h native (730 days)
        - 2h -> fetch 1h, then resample to 2h
        - 3h -> fetch 1h, then resample to 3h
        - 4h -> fetch 1h, then resample to 4h
        - daily -> fetch 1d native (unlimited)
        - weekly -> fetch 1wk native (unlimited)
        - monthly -> fetch 1mo native (unlimited)
        - 3month -> fetch 1mo, then resample to 3month

    Returns:
        Dict with structure:
        {
            'tsla': {'5min': DataFrame, '15min': DataFrame, ..., '3month': DataFrame},
            'spy': {'5min': DataFrame, '15min': DataFrame, ..., '3month': DataFrame},
            'vix': DataFrame  # VIX is daily only, single DataFrame
        }

    Example:
        >>> data = _fetch_all_native_timeframes()
        >>> tsla_daily = data['tsla']['daily']
        >>> spy_1h = data['spy']['1h']
        >>> vix = data['vix']
    """
    from v7.core.timeframe import TIMEFRAMES

    # Initialize result structure
    result = {
        'tsla': {},
        'spy': {},
        'vix': pd.DataFrame()
    }

    # Define mapping from system timeframe to yfinance interval
    # Format: system_tf -> (yf_interval, period_str, needs_resample, resample_rule)
    tf_mapping = {
        '5min': ('5m', '60d', False, None),
        '15min': ('15m', '60d', False, None),
        '30min': ('30m', '60d', False, None),
        '1h': ('1h', '730d', False, None),
        '2h': ('1h', '730d', True, '2h'),
        '3h': ('1h', '730d', True, '3h'),
        '4h': ('1h', '730d', True, '4h'),
        'daily': ('1d', 'max', False, None),
        'weekly': ('1wk', 'max', False, None),
        'monthly': ('1mo', 'max', False, None),
        '3month': ('1mo', 'max', True, '3ME'),
    }

    # Cache for fetched raw data to avoid duplicate API calls
    # Key: (symbol, yf_interval), Value: DataFrame
    raw_data_cache: Dict[tuple, pd.DataFrame] = {}

    def fetch_raw_data(symbol: str, yf_interval: str, period: str) -> pd.DataFrame:
        """Fetch raw data from yfinance with caching."""
        cache_key = (symbol, yf_interval)
        if cache_key in raw_data_cache:
            return raw_data_cache[cache_key]

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=yf_interval)
            if len(df) > 0:
                df = _format_yfinance_df(df)
            raw_data_cache[cache_key] = df
            return df
        except Exception as e:
            print(f"    Error fetching {symbol} {yf_interval}: {e}")
            raw_data_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

    def resample_to_timeframe(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample OHLCV data to a higher timeframe."""
        if df.empty:
            return df
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return resampled

    print("=" * 60)
    print("Fetching ALL native timeframes for TSLA, SPY, VIX")
    print("=" * 60)

    # Process each timeframe
    for tf in TIMEFRAMES:
        if tf not in tf_mapping:
            print(f"  Warning: No mapping for timeframe {tf}, skipping")
            continue

        yf_interval, period, needs_resample, resample_rule = tf_mapping[tf]

        print(f"\n[{tf}] Fetching via {yf_interval} interval (period={period})...")

        # Fetch TSLA
        print(f"  TSLA: ", end="")
        tsla_df = fetch_raw_data('TSLA', yf_interval, period)
        if not tsla_df.empty:
            if needs_resample:
                tsla_df = resample_to_timeframe(tsla_df, resample_rule)
                print(f"{len(tsla_df)} bars (resampled to {resample_rule})")
            else:
                print(f"{len(tsla_df)} bars")
            result['tsla'][tf] = tsla_df
        else:
            print("FAILED")
            result['tsla'][tf] = pd.DataFrame()

        # Fetch SPY
        print(f"  SPY:  ", end="")
        spy_df = fetch_raw_data('SPY', yf_interval, period)
        if not spy_df.empty:
            if needs_resample:
                spy_df = resample_to_timeframe(spy_df, resample_rule)
                print(f"{len(spy_df)} bars (resampled to {resample_rule})")
            else:
                print(f"{len(spy_df)} bars")
            result['spy'][tf] = spy_df
        else:
            print("FAILED")
            result['spy'][tf] = pd.DataFrame()

    # Fetch VIX separately (daily data only)
    print(f"\n[VIX] Fetching daily data...")
    print(f"  VIX:  ", end="")
    try:
        vix = yf.Ticker('^VIX')
        vix_df = vix.history(period='max', interval='1d')
        if len(vix_df) > 0:
            vix_df = _format_yfinance_df(vix_df)
            print(f"{len(vix_df)} bars")
            result['vix'] = vix_df
        else:
            print("FAILED (no data)")
    except Exception as e:
        print(f"FAILED ({e})")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for symbol in ['tsla', 'spy']:
        print(f"\n{symbol.upper()}:")
        for tf in TIMEFRAMES:
            if tf in result[symbol] and not result[symbol][tf].empty:
                df = result[symbol][tf]
                date_range = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
                print(f"  {tf:10s}: {len(df):6d} bars | {date_range}")
            else:
                print(f"  {tf:10s}: MISSING")

    if not result['vix'].empty:
        df = result['vix']
        date_range = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
        print(f"\nVIX: {len(df)} bars | {date_range}")
    else:
        print("\nVIX: MISSING")

    print("=" * 60)

    return result
