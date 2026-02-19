"""
V15 Live Data Module - Real-time market data using Finnhub + yfinance.

Provides live and historical data for TSLA, SPY, and VIX for the dashboard
and live prediction systems.

Data sources:
- Finnhub (free tier): Real-time quotes for TSLA and SPY (sub-second latency)
- yfinance: Historical OHLCV candles for all symbols (15-min delayed but full history)
- VIX always via yfinance (Finnhub free tier doesn't support indices)

Usage:
    from v15.live_data import YFinanceLiveData, should_refresh

    # Initialize data feed
    data_feed = YFinanceLiveData()

    # Get historical data for warm-up
    tsla, spy, vix = data_feed.get_historical(period='60d', interval='5m')

    # Get real-time prices (Finnhub for TSLA/SPY, yfinance for VIX)
    prices = data_feed.get_realtime_prices()
    # prices = {'TSLA': 409.59, 'SPY': 598.12, '^VIX': 15.3}

    # Check if market is open
    if data_feed.is_market_open():
        # Make predictions...
        pass
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any, List
import logging
import time
from functools import lru_cache
import hashlib

from .exceptions import DataLoadError

logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Install with: pip install yfinance")

# Try to import Finnhub client
try:
    from .data.finnhub_client import FinnhubClient, FinnhubQuote
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False
    logger.info("Finnhub client not available, using yfinance only")


class YFinanceLiveData:
    """
    Live data feed using yfinance for TSLA, SPY, VIX.

    Handles yfinance quirks:
    - VIX symbol is '^VIX'
    - 5-min data limited to 60 days
    - Column names normalized to lowercase
    - Rate limiting with exponential backoff
    - Data caching to reduce API calls

    Attributes:
        symbols: List of symbols to fetch ['TSLA', 'SPY', '^VIX']
        cache_ttl: Cache time-to-live in seconds (default 60)
        max_retries: Maximum retry attempts for failed requests
    """

    # Default symbols
    DEFAULT_SYMBOLS = ['TSLA', 'SPY', '^VIX']

    # Symbol display names (for logging)
    SYMBOL_NAMES = {
        'TSLA': 'TSLA',
        'SPY': 'SPY',
        '^VIX': 'VIX'
    }

    def __init__(
        self,
        symbols: List[str] = None,
        cache_ttl: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the live data feed.

        Args:
            symbols: List of symbols ['TSLA', 'SPY', '^VIX']. Defaults to standard symbols.
            cache_ttl: Cache time-to-live in seconds (default 60)
            max_retries: Maximum retry attempts for failed requests (default 3)
            retry_delay: Base delay between retries in seconds (default 1.0)

        Raises:
            ImportError: If yfinance is not installed
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance is required for live data. Install with: pip install yfinance"
            )

        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Internal cache: {cache_key: (timestamp, data)}
        self._cache: Dict[str, Tuple[datetime, Any]] = {}

        # Track last update time
        self._last_update_time: Optional[datetime] = None

        # Finnhub client for real-time quotes (TSLA, SPY only)
        self._finnhub: Optional['FinnhubClient'] = None
        if FINNHUB_AVAILABLE:
            try:
                self._finnhub = FinnhubClient(cache_ttl=5.0)
                logger.info("Finnhub real-time quotes enabled for TSLA/SPY")
            except Exception as e:
                logger.warning(f"Failed to initialize Finnhub client: {e}")

        # Validate symbols
        self._validate_symbols()

        logger.info(
            f"YFinanceLiveData initialized for symbols: {self.symbols}, "
            f"cache_ttl={cache_ttl}s, max_retries={max_retries}, "
            f"finnhub={'enabled' if self._finnhub else 'disabled'}"
        )

    def _validate_symbols(self) -> None:
        """Validate that required symbols are present."""
        required = {'TSLA', 'SPY', '^VIX'}
        provided = set(self.symbols)

        # Check VIX symbol format
        if 'VIX' in provided and '^VIX' not in provided:
            logger.warning(
                "Found 'VIX' in symbols, but yfinance requires '^VIX'. "
                "Replacing 'VIX' with '^VIX'."
            )
            self.symbols = [s if s != 'VIX' else '^VIX' for s in self.symbols]

        missing = required - set(self.symbols)
        if missing:
            logger.warning(f"Missing recommended symbols: {missing}")

    def _get_cache_key(self, symbol: str, period: str, interval: str) -> str:
        """Generate a cache key for the given parameters."""
        return f"{symbol}_{period}_{interval}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False

        timestamp, _ = self._cache[cache_key]
        age = (datetime.now() - timestamp).total_seconds()
        return age < self.cache_ttl

    def _get_cached(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from cache if valid."""
        if self._is_cache_valid(cache_key):
            _, data = self._cache[cache_key]
            logger.debug(f"Cache hit for {cache_key}")
            return data
        return None

    def _set_cached(self, cache_key: str, data: pd.DataFrame) -> None:
        """Store data in cache."""
        self._cache[cache_key] = (datetime.now(), data)
        logger.debug(f"Cached {cache_key} with {len(data)} rows")

    def _fetch_with_retry(
        self,
        symbol: str,
        period: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Fetch data with retry logic for rate limiting.

        Args:
            symbol: Stock symbol (e.g., 'TSLA', '^VIX')
            period: Time period (e.g., '60d', '1mo')
            interval: Bar interval (e.g., '5m', '1d')

        Returns:
            DataFrame with OHLCV data

        Raises:
            DataLoadError: If all retries fail
        """
        display_name = self.SYMBOL_NAMES.get(symbol, symbol)
        last_error = None

        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)

                if df.empty:
                    raise DataLoadError(
                        f"{display_name}: No data returned from yfinance. "
                        f"Symbol may be invalid or market data unavailable."
                    )

                return df

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"{display_name}: Fetch attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"{display_name}: All {self.max_retries} fetch attempts failed"
                    )

        raise DataLoadError(
            f"{display_name}: Failed to fetch data after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame columns to lowercase.

        yfinance returns columns like 'Open', 'High', 'Low', 'Close', 'Volume'.
        We need lowercase: 'open', 'high', 'low', 'close', 'volume'.

        Args:
            df: Raw DataFrame from yfinance

        Returns:
            DataFrame with lowercase column names
        """
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        # Remove unnecessary columns (dividends, stock splits)
        keep_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in keep_cols if col in df.columns]
        df = df[available_cols]

        return df

    def _validate_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        require_volume: bool = True
    ) -> None:
        """
        Validate the fetched data.

        Args:
            df: DataFrame to validate
            symbol: Symbol for error messages
            require_volume: Whether volume column is required

        Raises:
            DataLoadError: If validation fails
        """
        display_name = self.SYMBOL_NAMES.get(symbol, symbol)
        required_cols = ['open', 'high', 'low', 'close']
        if require_volume:
            required_cols.append('volume')

        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataLoadError(
                f"{display_name}: Missing required columns: {missing_cols}. "
                f"Available: {list(df.columns)}"
            )

        # Check for empty data
        if df.empty:
            raise DataLoadError(f"{display_name}: DataFrame is empty")

        # Check for NaN in OHLC
        ohlc_cols = ['open', 'high', 'low', 'close']
        for col in ohlc_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_pct = (nan_count / len(df)) * 100
                logger.warning(
                    f"{display_name}: Column '{col}' has {nan_count} NaN values ({nan_pct:.1f}%). "
                    f"Forward-filling..."
                )
                df[col] = df[col].ffill()

                # Check if still NaN at start
                if df[col].isna().any():
                    df[col] = df[col].bfill()

    def _fetch_symbol(
        self,
        symbol: str,
        period: str,
        interval: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch and process data for a single symbol.

        Args:
            symbol: Stock symbol
            period: Time period
            interval: Bar interval
            use_cache: Whether to use caching

        Returns:
            Processed DataFrame with lowercase columns
        """
        cache_key = self._get_cache_key(symbol, period, interval)

        # Check cache
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        # Fetch fresh data
        display_name = self.SYMBOL_NAMES.get(symbol, symbol)
        logger.info(f"Fetching {display_name} data: period={period}, interval={interval}")

        df = self._fetch_with_retry(symbol, period, interval)
        df = self._normalize_columns(df)

        # VIX may not have volume
        require_volume = (symbol != '^VIX')
        self._validate_data(df, symbol, require_volume=require_volume)

        # For VIX, add dummy volume if missing
        if '^VIX' in symbol and 'volume' not in df.columns:
            df['volume'] = 0

        # Cache the result
        if use_cache:
            self._set_cached(cache_key, df)

        return df

    def get_historical(
        self,
        period: str = '60d',
        interval: str = '5m'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get historical 5-min data for all symbols.

        Note: yfinance only provides 5-minute data for the last 60 days.
        For older data, use daily interval.

        Args:
            period: Time period ('60d', '30d', '7d', etc.)
                   For 5m interval, max is '60d'
            interval: Bar interval ('5m', '15m', '1h', '1d')

        Returns:
            Tuple of (tsla_df, spy_df, vix_df) with columns:
            - open, high, low, close, volume
            - DatetimeIndex

        Raises:
            DataLoadError: If data fetching or validation fails
        """
        # Validate interval/period combination
        if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
            # Intraday data limited to 60 days
            period_days = self._parse_period_days(period)
            if period_days > 60:
                logger.warning(
                    f"Requested {period} but {interval} data limited to 60 days. "
                    f"Using period='60d'."
                )
                period = '60d'

        # Fetch all symbols
        tsla_df = self._fetch_symbol('TSLA', period, interval)
        spy_df = self._fetch_symbol('SPY', period, interval)
        vix_df = self._fetch_symbol('^VIX', period, interval)

        # Align indices
        tsla_df, spy_df, vix_df = self._align_dataframes(tsla_df, spy_df, vix_df)

        # Update last update time
        self._last_update_time = datetime.now()

        logger.info(
            f"Fetched historical data: {len(tsla_df)} bars from "
            f"{tsla_df.index[0]} to {tsla_df.index[-1]}"
        )

        return tsla_df, spy_df, vix_df

    def _parse_period_days(self, period: str) -> int:
        """Parse period string to number of days."""
        if period.endswith('d'):
            return int(period[:-1])
        elif period.endswith('mo'):
            return int(period[:-2]) * 30
        elif period.endswith('y'):
            return int(period[:-1]) * 365
        elif period.endswith('w'):
            return int(period[:-1]) * 7
        else:
            return 60  # Default

    def _align_dataframes(
        self,
        tsla_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        vix_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Align all DataFrames to a common index.

        Uses TSLA as the reference index (primary instrument).
        SPY and VIX are reindexed with forward-fill.

        Args:
            tsla_df: TSLA DataFrame
            spy_df: SPY DataFrame
            vix_df: VIX DataFrame

        Returns:
            Tuple of aligned DataFrames
        """
        # Use TSLA index as reference
        reference_index = tsla_df.index

        # Find common time range
        common_start = max(tsla_df.index[0], spy_df.index[0], vix_df.index[0])
        common_end = min(tsla_df.index[-1], spy_df.index[-1], vix_df.index[-1])

        # Filter to common range
        tsla_df = tsla_df[(tsla_df.index >= common_start) & (tsla_df.index <= common_end)]
        reference_index = tsla_df.index

        # Reindex SPY and VIX to TSLA's index
        spy_aligned = spy_df.reindex(reference_index, method='ffill')
        vix_aligned = vix_df.reindex(reference_index, method='ffill')

        # Remove any rows with NaN (at the start before ffill has data)
        valid_mask = (
            ~tsla_df.isna().any(axis=1) &
            ~spy_aligned.isna().any(axis=1) &
            ~vix_aligned.isna().any(axis=1)
        )

        tsla_result = tsla_df[valid_mask].copy()
        spy_result = spy_aligned[valid_mask].copy()
        vix_result = vix_aligned[valid_mask].copy()

        if tsla_result.empty:
            raise DataLoadError(
                "No valid data after alignment. Check if all symbols have overlapping data."
            )

        return tsla_result, spy_result, vix_result

    def get_latest_bars(
        self,
        lookback_bars: int = 100
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get most recent bars for live prediction.

        This fetches fresh data (bypassing cache) to ensure we have
        the most recent bars for live trading.

        Args:
            lookback_bars: Number of bars to return (default 100)

        Returns:
            Tuple of (tsla_df, spy_df, vix_df) with the most recent bars

        Raises:
            DataLoadError: If data fetching fails
        """
        # Fetch with short period to get most recent data
        # For 100 bars at 5min, we need about 1 day
        period = '5d'  # Fetch 5 days to ensure we have enough after market hours

        # Bypass cache for fresh data
        tsla_df = self._fetch_symbol('TSLA', period, '5m', use_cache=False)
        spy_df = self._fetch_symbol('SPY', period, '5m', use_cache=False)
        vix_df = self._fetch_symbol('^VIX', period, '5m', use_cache=False)

        # Align
        tsla_df, spy_df, vix_df = self._align_dataframes(tsla_df, spy_df, vix_df)

        # Take last N bars
        tsla_df = tsla_df.iloc[-lookback_bars:] if len(tsla_df) > lookback_bars else tsla_df
        spy_df = spy_df.iloc[-lookback_bars:] if len(spy_df) > lookback_bars else spy_df
        vix_df = vix_df.iloc[-lookback_bars:] if len(vix_df) > lookback_bars else vix_df

        # Update last update time
        self._last_update_time = datetime.now()

        logger.debug(f"Fetched {len(tsla_df)} latest bars")

        return tsla_df, spy_df, vix_df

    def get_realtime_prices(self) -> Dict[str, Optional[float]]:
        """
        Get real-time prices for TSLA, SPY, VIX.

        Uses Finnhub for TSLA and SPY (sub-second latency).
        Falls back to yfinance for VIX and when Finnhub is unavailable.

        Returns:
            Dict mapping symbol to current price (or None if unavailable).
            Keys: 'TSLA', 'SPY', '^VIX'
        """
        prices: Dict[str, Optional[float]] = {}

        # Finnhub for TSLA and SPY
        if self._finnhub:
            for symbol in ['TSLA', 'SPY']:
                quote = self._finnhub.get_quote(symbol)
                if quote and quote.current_price > 0:
                    prices[symbol] = quote.current_price

        # Fallback to yfinance for symbols not yet fetched
        for symbol in self.symbols:
            if symbol in prices:
                continue
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                price = getattr(info, 'last_price', None)
                if price and price > 0:
                    prices[symbol] = float(price)
            except Exception as e:
                logger.debug(f"yfinance fast_info failed for {symbol}: {e}")
                prices[symbol] = None

        return prices

    def get_realtime_quote(self, symbol: str) -> Optional['FinnhubQuote']:
        """
        Get full Finnhub quote for a symbol (includes OHLC, change, etc.).

        Args:
            symbol: Stock symbol (e.g., 'TSLA', 'SPY').

        Returns:
            FinnhubQuote or None if Finnhub unavailable or symbol unsupported.
        """
        if self._finnhub:
            return self._finnhub.get_quote(symbol)
        return None

    def is_market_open(self) -> bool:
        """
        Check if US market is currently open.

        US market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        Does not account for holidays.

        Returns:
            True if market is currently open
        """
        try:
            import pytz
        except ImportError:
            logger.warning("pytz not available, using naive datetime check")
            # Fallback: assume ET is UTC-5
            now = datetime.utcnow() - timedelta(hours=5)
            return self._is_market_open_naive(now)

        # Get current time in ET
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)

        return self._is_market_open_naive(now_et)

    def _is_market_open_naive(self, now: datetime) -> bool:
        """Check market hours with naive datetime."""
        # Check weekday (Monday = 0, Sunday = 6)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check time (9:30 AM - 4:00 PM)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    def get_last_update_time(self) -> Optional[datetime]:
        """
        Get timestamp of most recent data fetch.

        Returns:
            datetime of last successful data fetch, or None if no data fetched yet
        """
        return self._last_update_time

    def get_data_age_seconds(self) -> Optional[float]:
        """
        Get age of cached data in seconds.

        Returns:
            Age in seconds, or None if no data fetched yet
        """
        if self._last_update_time is None:
            return None
        return (datetime.now() - self._last_update_time).total_seconds()

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Data cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state.

        Returns:
            Dict with cache statistics
        """
        now = datetime.now()
        cache_info = {
            'num_entries': len(self._cache),
            'entries': {}
        }

        for key, (timestamp, data) in self._cache.items():
            age_seconds = (now - timestamp).total_seconds()
            cache_info['entries'][key] = {
                'rows': len(data) if isinstance(data, pd.DataFrame) else 0,
                'age_seconds': age_seconds,
                'valid': age_seconds < self.cache_ttl
            }

        return cache_info


def should_refresh(
    last_refresh: Optional[datetime],
    interval_seconds: int = 300
) -> bool:
    """
    Check if enough time has passed for refresh.

    This is a helper for Streamlit auto-refresh functionality.
    Use with st.rerun() to periodically refresh data.

    Example:
        if should_refresh(st.session_state.get('last_refresh'), 300):
            st.session_state['last_refresh'] = datetime.now()
            st.rerun()

    Args:
        last_refresh: Timestamp of last refresh, or None for first refresh
        interval_seconds: Minimum seconds between refreshes (default 300 = 5 min)

    Returns:
        True if refresh is needed
    """
    if last_refresh is None:
        return True

    elapsed = (datetime.now() - last_refresh).total_seconds()
    return elapsed >= interval_seconds


def get_market_status() -> Dict[str, Any]:
    """
    Get current market status information.

    Returns:
        Dict with:
        - is_open: bool
        - current_time_et: str
        - next_open: str (if closed)
        - next_close: str (if open)
    """
    try:
        import pytz
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
    except ImportError:
        now_et = datetime.utcnow() - timedelta(hours=5)
        et_tz = None

    # Check if market is open
    is_open = False
    if now_et.weekday() < 5:  # Weekday
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        is_open = market_open <= now_et <= market_close

    status = {
        'is_open': is_open,
        'current_time_et': now_et.strftime('%Y-%m-%d %H:%M:%S ET'),
        'weekday': now_et.strftime('%A'),
    }

    if is_open:
        close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        time_to_close = close_time - now_et
        status['next_close'] = '4:00 PM ET'
        status['time_to_close'] = str(time_to_close).split('.')[0]
    else:
        # Calculate next open
        if now_et.weekday() >= 5:  # Weekend
            days_until_monday = (7 - now_et.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            next_open = now_et + timedelta(days=days_until_monday)
        elif now_et.hour >= 16:  # After close
            next_open = now_et + timedelta(days=1)
        else:  # Before open
            next_open = now_et

        next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)

        # Skip weekends
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)

        status['next_open'] = next_open.strftime('%Y-%m-%d 9:30 AM ET')

    return status


# =============================================================================
# Convenience function for quick data fetching
# =============================================================================

def fetch_live_data(
    period: str = '60d',
    interval: str = '5m'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Quick function to fetch live market data.

    Creates a temporary YFinanceLiveData instance and fetches data.
    For repeated fetches, prefer creating a single YFinanceLiveData instance.

    Args:
        period: Time period (default '60d')
        interval: Bar interval (default '5m')

    Returns:
        Tuple of (tsla_df, spy_df, vix_df)

    Example:
        tsla, spy, vix = fetch_live_data()
        print(f"Fetched {len(tsla)} bars")
    """
    data_feed = YFinanceLiveData()
    return data_feed.get_historical(period=period, interval=interval)


# =============================================================================
# Testing / Verification
# =============================================================================

def verify_data_structure(
    df: pd.DataFrame,
    name: str = "data"
) -> Dict[str, Any]:
    """
    Verify that a DataFrame has the expected structure for the predictor.

    Args:
        df: DataFrame to verify
        name: Name for reporting

    Returns:
        Dict with verification results
    """
    results = {
        'name': name,
        'valid': True,
        'issues': [],
        'info': {}
    }

    # Check columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        results['valid'] = False
        results['issues'].append(f"Missing columns: {missing}")

    # Check index
    if not isinstance(df.index, pd.DatetimeIndex):
        results['valid'] = False
        results['issues'].append("Index is not DatetimeIndex")

    # Check data types
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            results['valid'] = False
            results['issues'].append(f"Column '{col}' is not numeric")

    # Check for NaN
    nan_counts = df[['open', 'high', 'low', 'close']].isna().sum()
    if nan_counts.any():
        results['issues'].append(f"NaN counts: {nan_counts.to_dict()}")

    # Info
    results['info'] = {
        'rows': len(df),
        'columns': list(df.columns),
        'start': str(df.index[0]) if len(df) > 0 else None,
        'end': str(df.index[-1]) if len(df) > 0 else None,
        'index_type': str(type(df.index).__name__),
    }

    return results


def test_live_data():
    """
    Test the live data module.

    Run this to verify yfinance integration is working.
    """
    print("=" * 60)
    print("Testing YFinanceLiveData module")
    print("=" * 60)

    if not YFINANCE_AVAILABLE:
        print("ERROR: yfinance not installed")
        print("Install with: pip install yfinance")
        return False

    try:
        # Initialize
        print("\n1. Initializing YFinanceLiveData...")
        data_feed = YFinanceLiveData(cache_ttl=30)
        print("   OK")

        # Check market status
        print("\n2. Checking market status...")
        status = get_market_status()
        print(f"   Market is {'OPEN' if status['is_open'] else 'CLOSED'}")
        print(f"   Current time: {status['current_time_et']}")

        # Fetch historical data
        print("\n3. Fetching historical data (period=7d, interval=5m)...")
        start_time = time.time()
        tsla, spy, vix = data_feed.get_historical(period='7d', interval='5m')
        elapsed = time.time() - start_time
        print(f"   Fetched {len(tsla)} bars in {elapsed:.2f}s")

        # Verify data structure
        print("\n4. Verifying data structure...")
        for name, df in [('TSLA', tsla), ('SPY', spy), ('VIX', vix)]:
            result = verify_data_structure(df, name)
            status_str = "OK" if result['valid'] else "FAILED"
            print(f"   {name}: {status_str}")
            if result['issues']:
                for issue in result['issues']:
                    print(f"      - {issue}")
            print(f"      Rows: {result['info']['rows']}")
            print(f"      Range: {result['info']['start']} to {result['info']['end']}")

        # Test cache
        print("\n5. Testing cache...")
        start_time = time.time()
        tsla2, _, _ = data_feed.get_historical(period='7d', interval='5m')
        elapsed = time.time() - start_time
        print(f"   Second fetch: {elapsed:.4f}s (should be fast - cache hit)")
        cache_info = data_feed.get_cache_info()
        print(f"   Cache entries: {cache_info['num_entries']}")

        # Test latest bars
        print("\n6. Fetching latest bars (lookback=50)...")
        start_time = time.time()
        tsla_latest, spy_latest, vix_latest = data_feed.get_latest_bars(lookback_bars=50)
        elapsed = time.time() - start_time
        print(f"   Fetched {len(tsla_latest)} bars in {elapsed:.2f}s")

        # Check data recency
        print("\n7. Checking data recency...")
        last_bar_time = tsla_latest.index[-1]
        age = datetime.now() - last_bar_time.to_pydatetime().replace(tzinfo=None)
        print(f"   Last bar: {last_bar_time}")
        print(f"   Age: {age}")

        # Check if within last trading day
        age_hours = age.total_seconds() / 3600
        if age_hours < 24:
            print("   Data is recent (within 24 hours)")
        elif age_hours < 72:
            print("   Data is from last few days (likely weekend)")
        else:
            print("   WARNING: Data may be stale")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_live_data()
