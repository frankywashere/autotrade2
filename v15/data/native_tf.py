"""
Native Timeframe Data Loader

Fetches OHLCV data at native timeframes from yfinance instead of resampling.
This provides more accurate OHLC values for higher timeframes.

Native TF data may differ from resampled data due to:
- Pre/post market data handling
- Corporate actions adjustments
- Exchange-specific calculations

Supported timeframes and their yfinance mappings (10 timeframes):
- 5min -> 5m
- 15min -> 15m
- 30min -> 30m
- 1h -> 1h (or 60m)
- 2h -> aggregated from 1h
- 3h -> aggregated from 1h
- 4h -> aggregated from 1h
- daily -> 1d
- weekly -> 1wk
- monthly -> 1mo
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import pickle
from datetime import datetime, timedelta
import time
import hashlib
import logging

from ..exceptions import DataLoadError


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Timeframe Mappings
# =============================================================================

# Map our TFs to yfinance intervals
TF_TO_YF_INTERVAL = {
    '5min': '5m',
    '15min': '15m',
    '30min': '30m',
    '1h': '1h',
    '2h': None,     # Aggregate from 1h
    '3h': None,     # Aggregate from 1h
    '4h': None,     # Aggregate from 1h
    'daily': '1d',
    'weekly': '1wk',
    'monthly': '1mo',
}

# yfinance has data retention limits based on interval
# These are approximate - actual limits may vary
YF_INTERVAL_MAX_PERIOD = {
    '5m': 60,      # ~60 days
    '15m': 60,     # ~60 days
    '30m': 60,     # ~60 days
    '1h': 729,     # ~2 years (730 rejected by Yahoo's boundary check)
    '60m': 729,    # ~2 years
    '1d': 10000,   # Essentially unlimited
    '1wk': 10000,
    '1mo': 10000,
}

# Default symbols to fetch
DEFAULT_SYMBOLS = ['TSLA', 'SPY', '^VIX']

# All supported timeframes (10 total)
# Ordered: least rate-limit-prone first (daily/weekly/monthly have no
# retention limits), then 1h-based (2h/3h/4h reuse cached 1h), then
# intraday last (most likely to hit Yahoo rate limits).
ALL_TIMEFRAMES = [
    'daily', 'weekly', 'monthly',
    '1h', '2h', '3h', '4h',
    '5min', '15min', '30min',
]


# =============================================================================
# Cache Utilities
# =============================================================================

def _get_cache_key(symbol: str, tf: str, start_date: str, end_date: str) -> str:
    """Generate a unique cache key for the data request."""
    key_str = f"{symbol}_{tf}_{start_date}_{end_date}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def _get_cache_path(
    cache_dir: Path,
    symbol: str,
    tf: str,
    start_date: str,
    end_date: str
) -> Path:
    """Get the cache file path for a specific data request."""
    # Clean symbol for filename (^VIX -> VIX)
    clean_symbol = symbol.replace('^', '').replace('/', '_')
    cache_key = _get_cache_key(symbol, tf, start_date, end_date)
    return cache_dir / f"native_{clean_symbol}_{tf}_{cache_key}.pkl"


def _load_from_cache(cache_path: Path, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
    """
    Load data from cache if it exists and is fresh.

    Args:
        cache_path: Path to cache file
        max_age_hours: Maximum age of cache in hours before it's considered stale

    Returns:
        DataFrame if cache hit, None if cache miss or stale
    """
    if not cache_path.exists():
        return None

    # Check cache age
    mtime = cache_path.stat().st_mtime
    age_hours = (time.time() - mtime) / 3600

    if age_hours > max_age_hours:
        logger.debug(f"Cache stale: {cache_path.name} is {age_hours:.1f}h old")
        return None

    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        logger.debug(f"Cache hit: {cache_path.name}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None


def _save_to_cache(cache_path: Path, data: pd.DataFrame) -> None:
    """Save DataFrame to cache."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.debug(f"Cached: {cache_path.name}")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_path}: {e}")


# =============================================================================
# Data Fetching
# =============================================================================

def _fetch_yfinance_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    request_timeout: float = 10.0
) -> pd.DataFrame:
    """
    Fetch data from yfinance with retry logic.

    Args:
        symbol: Ticker symbol (e.g., 'TSLA', '^VIX')
        interval: yfinance interval (e.g., '5m', '1d')
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        request_timeout: Timeout (seconds) for each Yahoo request

    Returns:
        DataFrame with OHLCV data

    Raises:
        DataLoadError: If fetching fails after all retries
    """
    ticker = yf.Ticker(symbol)
    last_error = None

    for attempt in range(max_retries):
        try:
            # yfinance 'end' is exclusive, so +1 day to include today's bars
            end_inclusive = (
                pd.to_datetime(end_date) + timedelta(days=1)
            ).strftime('%Y-%m-%d')
            df = ticker.history(
                start=start_date,
                end=end_inclusive,
                interval=interval,
                auto_adjust=True,  # Adjust for splits/dividends
                prepost=False,     # Exclude pre/post market
                actions=False,     # Exclude dividends/splits columns
                timeout=request_timeout,
                raise_errors=True,
            )

            if df.empty:
                raise DataLoadError(
                    f"No data returned for {symbol} at interval {interval} "
                    f"from {start_date} to {end_date}"
                )

            # Standardize column names to lowercase
            df.columns = [col.lower() for col in df.columns]

            # Ensure we have required OHLCV columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise DataLoadError(
                    f"Missing required columns for {symbol}: {missing}. "
                    f"Available: {list(df.columns)}"
                )

            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Sort by index
            df = df.sort_index()

            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]

            return df

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for "
                    f"{symbol} {interval}: {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    raise DataLoadError(
        f"Failed to fetch {symbol} after {max_retries} attempts: {last_error}"
    )


def _aggregate_from_hourly(
    hourly_df: pd.DataFrame,
    target_hours: int
) -> pd.DataFrame:
    """
    Aggregate hourly data to multi-hour timeframes (2h, 3h, 4h).

    Args:
        hourly_df: DataFrame with 1h OHLCV data
        target_hours: Number of hours to aggregate (2, 3, or 4)

    Returns:
        DataFrame resampled to target timeframe
    """
    if hourly_df.empty:
        return hourly_df

    results = []
    # Group by trading day to avoid cross-day aggregation
    for date, day_bars in hourly_df.groupby(hourly_df.index.date):
        # Sequential aggregation within the day (matches C++ scanner)
        for i in range(0, len(day_bars), target_hours):
            chunk = day_bars.iloc[i:i + target_hours]
            if len(chunk) == 0:
                continue
            results.append({
                'timestamp': chunk.index[0],
                'open': chunk['open'].iloc[0],
                'high': chunk['high'].max(),
                'low': chunk['low'].min(),
                'close': chunk['close'].iloc[-1],
                'volume': chunk['volume'].sum(),
            })

    if not results:
        return pd.DataFrame(columns=hourly_df.columns)

    result_df = pd.DataFrame(results).set_index('timestamp')
    result_df.index.name = hourly_df.index.name
    return result_df


# =============================================================================
# Main API Functions
# =============================================================================

def fetch_native_tf(
    symbol: str,
    tf: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    cache_max_age_hours: int = 24,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    yf_request_timeout: float = 10.0
) -> pd.DataFrame:
    """
    Fetch native TF data from yfinance with caching.

    Args:
        symbol: Ticker symbol (e.g., 'TSLA', 'SPY', '^VIX')
        tf: Timeframe string (e.g., '5min', '1h', 'daily')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        cache_dir: Directory for caching data. If None, uses default.
        use_cache: Whether to use caching (default True)
        cache_max_age_hours: Maximum age of cached data before refetch
        max_retries: Maximum retries for yfinance fetches
        retry_delay: Initial retry delay in seconds (exponential backoff)
        yf_request_timeout: Timeout (seconds) for each yfinance request

    Returns:
        DataFrame with OHLCV data at the requested timeframe

    Raises:
        DataLoadError: If fetching fails
        ValueError: If timeframe is not supported

    Example:
        >>> df = fetch_native_tf('TSLA', 'daily', '2024-01-01', '2024-12-31')
        >>> print(df.shape)
        (252, 5)  # ~252 trading days
    """
    # Validate timeframe
    if tf not in TF_TO_YF_INTERVAL:
        raise ValueError(
            f"Unsupported timeframe: {tf}. "
            f"Supported: {list(TF_TO_YF_INTERVAL.keys())}"
        )

    # Set default cache directory
    if cache_dir is None:
        cache_dir = Path.home() / '.x14' / 'native_tf_cache'
    cache_dir = Path(cache_dir)

    # Get yfinance interval
    yf_interval = TF_TO_YF_INTERVAL[tf]

    # Adjust start_date for intraday retention limits BEFORE cache lookup
    # so the cache key matches on both read and write
    if yf_interval is not None and yf_interval in YF_INTERVAL_MAX_PERIOD:
        max_days = YF_INTERVAL_MAX_PERIOD[yf_interval]
        requested_days = (
            pd.to_datetime(end_date) - pd.to_datetime(start_date)
        ).days

        if requested_days > max_days:
            logger.warning(
                f"Requested {requested_days} days but {yf_interval} interval "
                f"only supports ~{max_days} days. Clamping start date."
            )
            # Subtract extra day to account for +1 day added in _fetch_yfinance_data
            adjusted_start = (
                pd.to_datetime(end_date) - timedelta(days=max_days - 1)
            ).strftime('%Y-%m-%d')
            start_date = adjusted_start

    # Check cache (now using the adjusted start_date)
    if use_cache:
        cache_path = _get_cache_path(cache_dir, symbol, tf, start_date, end_date)
        cached_data = _load_from_cache(cache_path, cache_max_age_hours)
        if cached_data is not None:
            return cached_data

    # Handle timeframes that need aggregation from hourly
    if yf_interval is None:
        # Need to aggregate from 1h data
        if tf == '2h':
            target_hours = 2
        elif tf == '3h':
            target_hours = 3
        elif tf == '4h':
            target_hours = 4
        else:
            raise ValueError(f"Cannot aggregate for timeframe: {tf}")

        # Fetch hourly data first
        hourly_df = fetch_native_tf(
            symbol=symbol,
            tf='1h',
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
            use_cache=use_cache,
            cache_max_age_hours=cache_max_age_hours,
            max_retries=max_retries,
            retry_delay=retry_delay,
            yf_request_timeout=yf_request_timeout,
        )

        # Aggregate to target timeframe
        df = _aggregate_from_hourly(hourly_df, target_hours)

    else:
        # Fetch from yfinance
        df = _fetch_yfinance_data(
            symbol=symbol,
            interval=yf_interval,
            start_date=start_date,
            end_date=end_date,
            max_retries=max_retries,
            retry_delay=retry_delay,
            request_timeout=yf_request_timeout,
        )

    # Cache the result
    if use_cache and not df.empty:
        cache_path = _get_cache_path(cache_dir, symbol, tf, start_date, end_date)
        _save_to_cache(cache_path, df)

    return df


def load_native_tf_data(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    start_date: str = '2015-01-01',
    end_date: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    cache_max_age_hours: int = 24,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    yf_request_timeout: float = 10.0,
    inter_request_delay: float = 0.5,
    verbose: bool = True
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load native TF data for multiple symbols and timeframes.

    Args:
        symbols: List of ticker symbols. Default: ['TSLA', 'SPY', '^VIX']
        timeframes: List of timeframes. Default: all supported TFs
        start_date: Start date (YYYY-MM-DD). Default: '2015-01-01'
        end_date: End date (YYYY-MM-DD). Default: today
        cache_dir: Directory for caching. If None, uses default.
        use_cache: Whether to use caching (default True)
        cache_max_age_hours: Maximum age of cached data
        max_retries: Maximum retries for each yfinance fetch
        retry_delay: Initial retry delay in seconds (exponential backoff)
        yf_request_timeout: Timeout (seconds) for each yfinance request
        inter_request_delay: Delay between network requests (seconds)
        verbose: Print progress information

    Returns:
        Dict[symbol][tf] -> DataFrame with OHLCV data

    Example:
        >>> data = load_native_tf_data(
        ...     symbols=['TSLA', 'SPY'],
        ...     timeframes=['daily', 'weekly'],
        ...     start_date='2024-01-01'
        ... )
        >>> tsla_daily = data['TSLA']['daily']
        >>> print(tsla_daily.shape)
    """
    # Set defaults
    if symbols is None:
        symbols = DEFAULT_SYMBOLS.copy()

    if timeframes is None:
        timeframes = ALL_TIMEFRAMES.copy()

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if cache_dir is None:
        cache_dir = Path.home() / '.x14' / 'native_tf_cache'
    cache_dir = Path(cache_dir)

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize result structure
    result: Dict[str, Dict[str, pd.DataFrame]] = {}

    total_requests = len(symbols) * len(timeframes)
    completed = 0
    failed = []
    request_idx = 0

    rate_limited = False

    for symbol in symbols:
        result[symbol] = {}

        for tf_idx, tf in enumerate(timeframes):
            request_idx += 1
            needs_network = TF_TO_YF_INTERVAL.get(tf) is not None

            # Throttle network calls only; aggregated TFs (2h/3h/4h) reuse 1h data.
            if tf_idx > 0 and needs_network:
                delay = 5.0 if rate_limited else inter_request_delay
                time.sleep(delay)

            request_start = time.perf_counter()
            try:
                if verbose:
                    print(
                        f"  [{request_idx:02d}/{total_requests:02d}] "
                        f"Fetching {symbol} {tf}...",
                        flush=True
                    )

                df = fetch_native_tf(
                    symbol=symbol,
                    tf=tf,
                    start_date=start_date,
                    end_date=end_date,
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    cache_max_age_hours=cache_max_age_hours,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    yf_request_timeout=yf_request_timeout,
                )

                result[symbol][tf] = df
                completed += 1
                rate_limited = False  # Reset on success

                if verbose:
                    elapsed = time.perf_counter() - request_start
                    print(
                        f"  [{request_idx:02d}/{total_requests:02d}] "
                        f"OK {symbol} {tf}: {len(df)} bars ({elapsed:.2f}s)",
                        flush=True
                    )

            except Exception as e:
                failed.append((symbol, tf, str(e)))
                if 'rate limit' in str(e).lower() or 'too many requests' in str(e).lower():
                    rate_limited = True
                if verbose:
                    elapsed = time.perf_counter() - request_start
                    print(
                        f"  [{request_idx:02d}/{total_requests:02d}] "
                        f"FAILED {symbol} {tf}: {e} ({elapsed:.2f}s)",
                        flush=True
                    )
                logger.error(f"Failed to fetch {symbol} {tf}: {e}")

    if verbose:
        print(f"\nCompleted: {completed}/{total_requests}")
        if failed:
            print(f"Failed: {len(failed)}")
            for sym, tf, err in failed[:5]:  # Show first 5 failures
                print(f"  - {sym} {tf}: {err}")

    return result


def get_native_tf_slice(
    native_data: Dict[str, Dict[str, pd.DataFrame]],
    symbol: str,
    tf: str,
    end_timestamp: pd.Timestamp,
    lookback_bars: int
) -> pd.DataFrame:
    """
    Get a slice of native TF data ending at timestamp with N bars lookback.

    This is useful for backtesting where you need to get the last N bars
    up to a specific point in time.

    Args:
        native_data: Dict from load_native_tf_data()
        symbol: Ticker symbol
        tf: Timeframe
        end_timestamp: End timestamp (inclusive)
        lookback_bars: Number of bars to return

    Returns:
        DataFrame with the last N bars up to end_timestamp

    Raises:
        KeyError: If symbol or timeframe not found
        ValueError: If not enough data for requested lookback

    Example:
        >>> data = load_native_tf_data(['TSLA'], ['daily'])
        >>> slice_df = get_native_tf_slice(
        ...     data, 'TSLA', 'daily',
        ...     pd.Timestamp('2024-06-30'),
        ...     lookback_bars=20
        ... )
        >>> print(len(slice_df))  # 20 bars
    """
    # Validate inputs
    if symbol not in native_data:
        raise KeyError(f"Symbol not found: {symbol}. Available: {list(native_data.keys())}")

    if tf not in native_data[symbol]:
        raise KeyError(
            f"Timeframe not found for {symbol}: {tf}. "
            f"Available: {list(native_data[symbol].keys())}"
        )

    df = native_data[symbol][tf]

    if df.empty:
        raise ValueError(f"No data available for {symbol} {tf}")

    # Ensure end_timestamp is timezone-aware if df index is
    if df.index.tz is not None and end_timestamp.tz is None:
        end_timestamp = end_timestamp.tz_localize(df.index.tz)
    elif df.index.tz is None and end_timestamp.tz is not None:
        end_timestamp = end_timestamp.tz_localize(None)

    # Filter to data up to and including end_timestamp
    mask = df.index <= end_timestamp
    available_df = df[mask]

    if len(available_df) < lookback_bars:
        raise ValueError(
            f"Not enough data for {symbol} {tf}: requested {lookback_bars} bars "
            f"but only {len(available_df)} available up to {end_timestamp}"
        )

    # Return last N bars
    return available_df.tail(lookback_bars).copy()


def align_native_tf_timestamps(
    native_data: Dict[str, Dict[str, pd.DataFrame]],
    reference_symbol: str = 'TSLA',
    method: str = 'inner'
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Align all symbols to the same timestamps for each timeframe.

    This is useful when you need all symbols to have data at the same
    points in time for analysis.

    Args:
        native_data: Dict from load_native_tf_data()
        reference_symbol: Symbol to use as reference for alignment
        method: Alignment method - 'inner' (only common timestamps) or
                'outer' (all timestamps with NaN fill)

    Returns:
        New dict with aligned DataFrames

    Example:
        >>> data = load_native_tf_data(['TSLA', 'SPY', '^VIX'], ['daily'])
        >>> aligned = align_native_tf_timestamps(data)
        >>> # Now all symbols have same index for each TF
    """
    if reference_symbol not in native_data:
        raise KeyError(f"Reference symbol not found: {reference_symbol}")

    aligned_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    symbols = list(native_data.keys())

    # Get all timeframes from reference symbol
    timeframes = list(native_data[reference_symbol].keys())

    for tf in timeframes:
        # Get reference index
        ref_df = native_data[reference_symbol].get(tf)
        if ref_df is None or ref_df.empty:
            continue

        if method == 'inner':
            # Find common timestamps across all symbols
            common_idx = ref_df.index
            for symbol in symbols:
                if symbol == reference_symbol:
                    continue
                sym_df = native_data[symbol].get(tf)
                if sym_df is not None and not sym_df.empty:
                    common_idx = common_idx.intersection(sym_df.index)

            # Align all symbols to common index
            for symbol in symbols:
                if symbol not in aligned_data:
                    aligned_data[symbol] = {}

                sym_df = native_data[symbol].get(tf)
                if sym_df is not None and not sym_df.empty:
                    aligned_data[symbol][tf] = sym_df.loc[common_idx].copy()
                else:
                    aligned_data[symbol][tf] = pd.DataFrame()

        elif method == 'outer':
            # Union of all timestamps
            all_idx = ref_df.index
            for symbol in symbols:
                if symbol == reference_symbol:
                    continue
                sym_df = native_data[symbol].get(tf)
                if sym_df is not None and not sym_df.empty:
                    all_idx = all_idx.union(sym_df.index)

            all_idx = all_idx.sort_values()

            # Reindex all symbols to union index with forward fill
            for symbol in symbols:
                if symbol not in aligned_data:
                    aligned_data[symbol] = {}

                sym_df = native_data[symbol].get(tf)
                if sym_df is not None and not sym_df.empty:
                    aligned_data[symbol][tf] = sym_df.reindex(all_idx, method='ffill')
                else:
                    aligned_data[symbol][tf] = pd.DataFrame(index=all_idx)
        else:
            raise ValueError(f"Unknown alignment method: {method}")

    return aligned_data


def clear_cache(cache_dir: Optional[Path] = None, older_than_hours: int = 0) -> int:
    """
    Clear cached native TF data.

    Args:
        cache_dir: Cache directory. If None, uses default.
        older_than_hours: Only clear files older than this many hours.
                         0 means clear all.

    Returns:
        Number of files deleted
    """
    if cache_dir is None:
        cache_dir = Path.home() / '.x14' / 'native_tf_cache'
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return 0

    deleted = 0
    current_time = time.time()

    for cache_file in cache_dir.glob('native_*.pkl'):
        if older_than_hours > 0:
            mtime = cache_file.stat().st_mtime
            age_hours = (current_time - mtime) / 3600
            if age_hours < older_than_hours:
                continue

        try:
            cache_file.unlink()
            deleted += 1
        except Exception as e:
            logger.warning(f"Failed to delete {cache_file}: {e}")

    return deleted


# =============================================================================
# Utility Functions
# =============================================================================

def get_available_data_range(
    symbol: str,
    tf: str
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Get the available date range for a symbol/timeframe from yfinance.

    This makes a test request to determine what data is available.

    Args:
        symbol: Ticker symbol
        tf: Timeframe

    Returns:
        Tuple of (earliest_date, latest_date) or (None, None) if unavailable
    """
    yf_interval = TF_TO_YF_INTERVAL.get(tf)
    if yf_interval is None:
        # For aggregated TFs, check the hourly data
        yf_interval = '1h'

    try:
        ticker = yf.Ticker(symbol)
        # Request a large range and see what we get
        df = ticker.history(period='max', interval=yf_interval)

        if df.empty:
            return None, None

        return df.index.min(), df.index.max()

    except Exception as e:
        logger.error(f"Failed to get data range for {symbol}: {e}")
        return None, None


def validate_native_data(
    df: pd.DataFrame,
    symbol: str,
    tf: str,
    strict_ohlc: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate native TF data quality.

    Args:
        df: DataFrame to validate
        symbol: Symbol name for error messages
        tf: Timeframe for error messages
        strict_ohlc: If True, enforce strict OHLC relationships
                    (set False for VIX which may violate these)

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check empty
    if df.empty:
        issues.append(f"{symbol} {tf}: DataFrame is empty")
        return False, issues

    # Check required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        issues.append(f"{symbol} {tf}: Missing columns: {missing}")

    # Check for NaN in OHLC
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_pct = (nan_count / len(df)) * 100
                issues.append(
                    f"{symbol} {tf}: {col} has {nan_count} NaN values ({nan_pct:.1f}%)"
                )

    # Check high >= low
    if 'high' in df.columns and 'low' in df.columns:
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            issues.append(f"{symbol} {tf}: {invalid_hl} rows where high < low")

    # Strict OHLC checks (not for VIX)
    if strict_ohlc:
        if 'high' in df.columns and 'open' in df.columns:
            invalid = (df['high'] < df['open']).sum()
            if invalid > 0:
                issues.append(f"{symbol} {tf}: {invalid} rows where high < open")

        if 'high' in df.columns and 'close' in df.columns:
            invalid = (df['high'] < df['close']).sum()
            if invalid > 0:
                issues.append(f"{symbol} {tf}: {invalid} rows where high < close")

        if 'low' in df.columns and 'open' in df.columns:
            invalid = (df['low'] > df['open']).sum()
            if invalid > 0:
                issues.append(f"{symbol} {tf}: {invalid} rows where low > open")

        if 'low' in df.columns and 'close' in df.columns:
            invalid = (df['low'] > df['close']).sum()
            if invalid > 0:
                issues.append(f"{symbol} {tf}: {invalid} rows where low > close")

    # Check for non-positive prices
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            non_positive = (df[col] <= 0).sum()
            if non_positive > 0:
                issues.append(
                    f"{symbol} {tf}: {non_positive} non-positive values in {col}"
                )

    is_valid = len(issues) == 0
    return is_valid, issues


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == '__main__':
    """Example usage demonstrating native TF data loading."""

    print("=" * 70)
    print("NATIVE TIMEFRAME DATA LOADER - DEMONSTRATION")
    print("=" * 70)

    # Test single symbol/TF fetch
    print("\n1. Fetching single symbol/timeframe...")
    try:
        tsla_daily = fetch_native_tf(
            symbol='TSLA',
            tf='daily',
            start_date='2024-01-01',
            end_date='2024-12-31'
        )
        print(f"   TSLA daily: {len(tsla_daily)} bars")
        print(f"   Date range: {tsla_daily.index[0]} to {tsla_daily.index[-1]}")
        print(f"   Columns: {list(tsla_daily.columns)}")
    except Exception as e:
        print(f"   Failed: {e}")

    # Test multi-hour aggregation
    print("\n2. Testing multi-hour aggregation (4h from 1h)...")
    try:
        tsla_4h = fetch_native_tf(
            symbol='TSLA',
            tf='4h',
            start_date='2024-10-01',
            end_date='2024-12-31'
        )
        print(f"   TSLA 4h: {len(tsla_4h)} bars")
    except Exception as e:
        print(f"   Failed: {e}")

    # Test VIX
    print("\n3. Testing VIX fetching...")
    try:
        vix_daily = fetch_native_tf(
            symbol='^VIX',
            tf='daily',
            start_date='2024-01-01',
            end_date='2024-12-31'
        )
        print(f"   ^VIX daily: {len(vix_daily)} bars")

        # Validate with relaxed OHLC checks
        is_valid, issues = validate_native_data(
            vix_daily, '^VIX', 'daily', strict_ohlc=False
        )
        print(f"   Validation: {'PASSED' if is_valid else 'FAILED'}")
        if issues:
            for issue in issues[:3]:
                print(f"     - {issue}")
    except Exception as e:
        print(f"   Failed: {e}")

    # Test multi-symbol, multi-TF load
    print("\n4. Loading multiple symbols and timeframes...")
    try:
        data = load_native_tf_data(
            symbols=['TSLA', 'SPY'],
            timeframes=['daily', 'weekly'],
            start_date='2024-01-01',
            verbose=True
        )

        print("\n   Summary:")
        for symbol in data:
            for tf in data[symbol]:
                df = data[symbol][tf]
                print(f"     {symbol} {tf}: {len(df)} bars")
    except Exception as e:
        print(f"   Failed: {e}")

    # Test slicing
    print("\n5. Testing data slicing...")
    try:
        data = load_native_tf_data(
            symbols=['TSLA'],
            timeframes=['daily'],
            start_date='2024-01-01',
            verbose=False
        )

        slice_df = get_native_tf_slice(
            data, 'TSLA', 'daily',
            pd.Timestamp('2024-06-30'),
            lookback_bars=20
        )
        print(f"   Slice: {len(slice_df)} bars ending at {slice_df.index[-1]}")
    except Exception as e:
        print(f"   Failed: {e}")

    # Test alignment
    print("\n6. Testing timestamp alignment...")
    try:
        data = load_native_tf_data(
            symbols=['TSLA', 'SPY', '^VIX'],
            timeframes=['daily'],
            start_date='2024-01-01',
            verbose=False
        )

        print(f"   Before alignment:")
        for symbol in data:
            print(f"     {symbol}: {len(data[symbol]['daily'])} bars")

        aligned = align_native_tf_timestamps(data, method='inner')

        print(f"   After alignment (inner join):")
        for symbol in aligned:
            print(f"     {symbol}: {len(aligned[symbol]['daily'])} bars")
    except Exception as e:
        print(f"   Failed: {e}")

    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print("=" * 70)
