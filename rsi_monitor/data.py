"""
Data fetching module for RSI Monitor.

Provides DataFetcher class for retrieving OHLCV data from yfinance
with caching and error handling.
"""

import logging
import time
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches price data from yfinance with caching support.

    Attributes:
        cache_ttl: Cache time-to-live in seconds (default 60 for live data)
    """

    # Default periods for each interval
    DEFAULT_PERIODS = {
        '1m': '1d',
        '2m': '1d',
        '5m': '1d',
        '15m': '5d',
        '30m': '5d',
        '60m': '1mo',
        '90m': '1mo',
        '1h': '1mo',
        '4h': '1mo',
        '1d': '1y',
        '5d': '5y',
        '1wk': '5y',
        '1mo': '5y',
    }

    # Valid intervals supported by yfinance
    VALID_INTERVALS = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '4h', '1d', '5d', '1wk', '1mo'}

    def __init__(self, cache_ttl: int = 60):
        """
        Initialize the DataFetcher.

        Args:
            cache_ttl: Cache time-to-live in seconds. Default is 60 seconds.
        """
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}

    def _get_cache_key(self, symbol: str, interval: str, period: str) -> str:
        """Generate a cache key for the given parameters."""
        return f"{symbol}:{interval}:{period}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data exists and is not expired."""
        if cache_key not in self._cache:
            return False
        timestamp, _ = self._cache[cache_key]
        return (time.time() - timestamp) < self.cache_ttl

    def _get_default_period(self, interval: str) -> str:
        """Get the default period for a given interval."""
        return self.DEFAULT_PERIODS.get(interval, '1mo')

    def fetch(
        self,
        symbol: str,
        interval: str = '1h',
        period: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single symbol.

        Args:
            symbol: Stock/crypto ticker symbol (e.g., 'AAPL', 'BTC-USD')
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
                   If None, uses sensible default based on interval.

        Returns:
            DataFrame with OHLCV columns (Open, High, Low, Close, Volume),
            or None if fetch fails.
        """
        # Validate interval
        if interval not in self.VALID_INTERVALS:
            logger.warning(f"Invalid interval '{interval}' for {symbol}. Valid intervals: {self.VALID_INTERVALS}")
            return None

        # Use default period if not specified
        if period is None:
            period = self._get_default_period(interval)

        # Check cache
        cache_key = self._get_cache_key(symbol, interval, period)
        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {symbol} {interval} {period}")
            _, data = self._cache[cache_key]
            return data.copy()

        # Fetch from yfinance
        try:
            logger.debug(f"Fetching {symbol} with interval={interval}, period={period}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {symbol} ({interval}, {period})")
                return None

            # Keep only OHLCV columns (drop Dividends, Stock Splits if present)
            ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in ohlcv_columns if col in df.columns]]

            # Cache the result
            self._cache[cache_key] = (time.time(), df)
            logger.debug(f"Cached {len(df)} rows for {symbol} {interval}")

            return df.copy()

        except Exception as e:
            logger.warning(f"Failed to fetch data for {symbol}: {e}")
            return None

    def fetch_all(
        self,
        symbols: list[str],
        intervals: list[str]
    ) -> dict[str, dict[str, Optional[pd.DataFrame]]]:
        """
        Fetch data for all symbol/interval combinations.

        Args:
            symbols: List of ticker symbols
            intervals: List of intervals to fetch

        Returns:
            Nested dict: {symbol: {interval: DataFrame or None}}
        """
        results: dict[str, dict[str, Optional[pd.DataFrame]]] = {}

        for symbol in symbols:
            results[symbol] = {}
            for interval in intervals:
                results[symbol][interval] = self.fetch(symbol, interval)

        return results

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        logger.debug("Cache cleared")

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (size, valid entries, expired entries)
        """
        now = time.time()
        valid = sum(1 for ts, _ in self._cache.values() if (now - ts) < self.cache_ttl)
        expired = len(self._cache) - valid

        return {
            'total_entries': len(self._cache),
            'valid_entries': valid,
            'expired_entries': expired,
            'cache_ttl': self.cache_ttl,
        }
