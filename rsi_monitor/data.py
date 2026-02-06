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
        '1m': '7d',
        '2m': '60d',
        '5m': '60d',
        '15m': '60d',
        '30m': '60d',
        '60m': '2y',
        '90m': '60d',
        '1h': '2y',
        '4h': '2y',
        '1d': 'max',
        '5d': 'max',
        '1wk': 'max',
        '1mo': 'max',
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

    def _get_cache_key(self, symbol: str, interval: str, period: str, prepost: bool = False) -> str:
        """Generate a cache key for the given parameters."""
        return f"{symbol}:{interval}:{period}:prepost={prepost}"

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
        period: Optional[str] = None,
        prepost: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single symbol.

        Args:
            symbol: Stock/crypto ticker symbol (e.g., 'AAPL', 'BTC-USD')
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
                   If None, uses sensible default based on interval.
            prepost: Include pre-market and after-hours data (intraday only).

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
        cache_key = self._get_cache_key(symbol, interval, period, prepost)
        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {symbol} {interval} {period}")
            _, data = self._cache[cache_key]
            return data.copy()

        # Fetch from yfinance
        try:
            logger.debug(f"Fetching {symbol} with interval={interval}, period={period}, prepost={prepost}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, prepost=prepost)

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

    def fetch_batch(
        self,
        symbols: list[str],
        interval: str = '1h',
        period: Optional[str] = None,
        prepost: bool = False
    ) -> dict[str, Optional[pd.DataFrame]]:
        """
        Fetch data for multiple symbols in a single batched download.

        Uses yf.download() to fetch all uncached symbols at once per interval,
        reducing HTTP requests via yfinance's built-in threading.

        Args:
            symbols: List of ticker symbols
            interval: Data interval
            period: Data period (None = use default for interval)
            prepost: Include pre/post market data

        Returns:
            Dict: {symbol: DataFrame or None}
        """
        if interval not in self.VALID_INTERVALS:
            return {s: None for s in symbols}

        if period is None:
            period = self._get_default_period(interval)

        results: dict[str, Optional[pd.DataFrame]] = {}
        uncached = []

        # Check cache first
        for symbol in symbols:
            cache_key = self._get_cache_key(symbol, interval, period, prepost)
            if self._is_cache_valid(cache_key):
                _, data = self._cache[cache_key]
                results[symbol] = data.copy()
            else:
                uncached.append(symbol)

        if not uncached:
            return results

        # Batch download uncached symbols
        try:
            logger.debug(f"Batch downloading {uncached} with interval={interval}, period={period}")
            df = yf.download(
                uncached,
                interval=interval,
                period=period,
                prepost=prepost,
                group_by='ticker',
                threads=True,
                progress=False,
            )

            if df.empty:
                for symbol in uncached:
                    results[symbol] = None
                return results

            ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            if len(uncached) == 1:
                # Single ticker: flat columns
                symbol = uncached[0]
                symbol_df = df[[col for col in ohlcv_columns if col in df.columns]]
                if not symbol_df.empty:
                    cache_key = self._get_cache_key(symbol, interval, period, prepost)
                    self._cache[cache_key] = (time.time(), symbol_df)
                    results[symbol] = symbol_df.copy()
                else:
                    results[symbol] = None
            else:
                # Multiple tickers: multi-level columns grouped by ticker
                for symbol in uncached:
                    try:
                        if symbol in df.columns.get_level_values(0):
                            symbol_df = df[symbol]
                            symbol_df = symbol_df[[col for col in ohlcv_columns if col in symbol_df.columns]]
                            symbol_df = symbol_df.dropna(how='all')
                            if not symbol_df.empty:
                                cache_key = self._get_cache_key(symbol, interval, period, prepost)
                                self._cache[cache_key] = (time.time(), symbol_df)
                                results[symbol] = symbol_df.copy()
                            else:
                                results[symbol] = None
                        else:
                            results[symbol] = None
                    except Exception:
                        results[symbol] = None

        except Exception as e:
            logger.warning(f"Batch download failed for {uncached}: {e}")
            # Fall back to individual fetches
            for symbol in uncached:
                results[symbol] = self.fetch(symbol, interval, period, prepost)

        return results

    def fetch_all(
        self,
        symbols: list[str],
        intervals: list[str],
        prepost: bool = False
    ) -> dict[str, dict[str, Optional[pd.DataFrame]]]:
        """
        Fetch data for all symbol/interval combinations using batched downloads.

        Groups by interval and uses fetch_batch() to download all symbols at once
        per interval, reducing HTTP requests.

        Args:
            symbols: List of ticker symbols
            intervals: List of intervals to fetch
            prepost: Include pre-market and after-hours data (intraday only).

        Returns:
            Nested dict: {symbol: {interval: DataFrame or None}}
        """
        results: dict[str, dict[str, Optional[pd.DataFrame]]] = {}
        for symbol in symbols:
            results[symbol] = {}

        for interval in intervals:
            batch_results = self.fetch_batch(symbols, interval, prepost=prepost)
            for symbol in symbols:
                results[symbol][interval] = batch_results.get(symbol)

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
