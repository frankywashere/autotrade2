"""
Twelve Data API Client

Provides real-time quotes and historical OHLCV time series via Twelve Data's
REST API. Replaces yfinance for TSLA/SPY candles and Finnhub for real-time
quotes. VIX is NOT available on Twelve Data — use yfinance for VIX.

Basic plan limits:
- 8 API credits/minute (sliding window)
- 800 API credits/day
- 5min intraday with 66 days of history
- Native 2h/4h timeframes (no aggregation needed)

Usage:
    from v15.data.twelvedata_client import TwelveDataClient

    client = TwelveDataClient()
    df = client.get_time_series('TSLA', '5min', outputsize=100)
    quote = client.get_quote('TSLA')
"""

import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# API key: env var takes precedence, hardcoded fallback for convenience
TWELVEDATA_API_KEY = os.environ.get(
    'TWELVEDATA_API_KEY',
    'c37aef32727548bba7e1ff39feb73970',
)
TWELVEDATA_BASE_URL = 'https://api.twelvedata.com'

# Map internal timeframe names to Twelve Data interval strings.
# None means "not supported natively" (must aggregate from a smaller TF).
TF_TO_TD_INTERVAL = {
    '5min': '5min',
    '15min': '15min',
    '30min': '30min',
    '1h': '1h',
    '2h': '2h',       # Native on TD (yfinance must aggregate from 1h)
    '3h': None,        # TD doesn't support 3h — aggregate from 1h
    '4h': '4h',        # Native on TD
    'daily': '1day',
    'weekly': '1week',
    'monthly': '1month',
}

# Symbols that are NOT available on Twelve Data basic plan (indices)
UNSUPPORTED_SYMBOLS = {'^VIX', 'VIX', '^GSPC', '^DJI', '^IXIC'}


@dataclass
class TwelveDataQuote:
    """Real-time quote from Twelve Data — field-compatible with FinnhubQuote."""
    symbol: str
    current_price: float
    change: float
    change_pct: float
    high: float
    low: float
    open: float
    prev_close: float
    timestamp: float


class _RateLimiter:
    """Sliding-window rate limiter for Twelve Data's 8 credits/minute cap."""

    def __init__(self, max_credits: int = 8, window_seconds: float = 60.0):
        self.max_credits = max_credits
        self.window_seconds = window_seconds
        self._timestamps: deque = deque()

    def acquire(self, credits: int = 1) -> None:
        """Block until enough capacity is available, then record usage."""
        now = time.monotonic()
        # Purge expired entries
        while self._timestamps and (now - self._timestamps[0]) >= self.window_seconds:
            self._timestamps.popleft()

        # If adding this request would exceed the limit, sleep
        while len(self._timestamps) + credits > self.max_credits:
            oldest = self._timestamps[0]
            sleep_time = self.window_seconds - (now - oldest) + 0.1
            if sleep_time > 0:
                logger.info(f"TwelveData rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            now = time.monotonic()
            while self._timestamps and (now - self._timestamps[0]) >= self.window_seconds:
                self._timestamps.popleft()

        # Record this request
        now = time.monotonic()
        for _ in range(credits):
            self._timestamps.append(now)

    def backoff(self, seconds: float = 60.0) -> None:
        """Force a backoff by filling the window — called when the API
        reports rate-limit exhaustion despite our local tracking."""
        now = time.monotonic()
        self._timestamps.clear()
        for i in range(self.max_credits):
            self._timestamps.append(now - i * 0.01)


# ======================================================================
# Module-level singleton rate limiter — shared by ALL TwelveDataClient
# instances so that live_data.py, native_tf.py, and run_backtest.py all
# coordinate against the same 8 credits/minute budget.
# ======================================================================
_GLOBAL_RATE_LIMITER = _RateLimiter(max_credits=8, window_seconds=60.0)


class TwelveDataClient:
    """
    Twelve Data REST client for quotes and OHLCV time series.

    Rate-limited to 8 credits/minute via a process-wide sliding window.
    In-memory quote cache with configurable TTL.
    """

    def __init__(
        self,
        api_key: str = TWELVEDATA_API_KEY,
        cache_ttl: float = 5.0,
        request_timeout: float = 10.0,
    ):
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self.request_timeout = request_timeout
        self._rate_limiter = _GLOBAL_RATE_LIMITER  # shared singleton
        self._quote_cache: Dict[str, tuple] = {}  # symbol -> (monotonic_ts, TwelveDataQuote)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> Optional[TwelveDataQuote]:
        """
        Get real-time quote for a symbol. 1 credit per call, 5s cache.

        Returns None for unsupported symbols (VIX, indices) or on error.
        """
        if not self.is_supported(symbol):
            return None

        # Check cache
        cached = self._get_cached_quote(symbol)
        if cached is not None:
            return cached

        self._rate_limiter.acquire(1)

        try:
            resp = requests.get(
                f"{TWELVEDATA_BASE_URL}/quote",
                params={'symbol': symbol, 'apikey': self.api_key},
                timeout=self.request_timeout,
            )

            if resp.status_code == 429:
                logger.warning("TwelveData rate limit hit (HTTP 429), backing off")
                self._rate_limiter.backoff()
                return None

            if resp.status_code != 200:
                logger.warning(f"TwelveData quote failed for {symbol}: HTTP {resp.status_code}")
                return None

            data = resp.json()

            # Check for API-level rate limit error (returns 200 with error body)
            if 'code' in data and data['code'] != 200:
                msg = data.get('message', '')
                if 'API credits' in msg or 'rate' in msg.lower():
                    logger.info(f"TwelveData credit limit for {symbol}, waiting for next window")
                    self._rate_limiter.backoff()
                    return None
                logger.warning(f"TwelveData quote error for {symbol}: {msg}")
                return None

            # Parse — TD returns string values
            price = float(data.get('close', 0))
            if price <= 0:
                logger.warning(f"TwelveData returned zero/negative price for {symbol}")
                return None

            prev_close = float(data.get('previous_close', 0))
            change = float(data.get('change', 0))
            change_pct = float(data.get('percent_change', 0))

            quote = TwelveDataQuote(
                symbol=symbol,
                current_price=price,
                change=change,
                change_pct=change_pct,
                high=float(data.get('high', 0)),
                low=float(data.get('low', 0)),
                open=float(data.get('open', 0)),
                prev_close=prev_close,
                timestamp=float(data.get('timestamp', 0)),
            )

            self._quote_cache[symbol] = (time.monotonic(), quote)
            return quote

        except requests.RequestException as e:
            logger.warning(f"TwelveData request failed for {symbol}: {e}")
            return None

    def get_quotes(self, symbols: list) -> Dict[str, Optional[TwelveDataQuote]]:
        """Get quotes for multiple symbols. Skips unsupported ones."""
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_quote(symbol)
        return results

    def get_time_series(
        self,
        symbol: str,
        interval: str,
        outputsize: int = 5000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV time series. 1 credit per call.

        On rate-limit errors, waits for the next minute window and retries once.

        Args:
            symbol: Ticker symbol (e.g. 'TSLA', 'SPY')
            interval: TD interval string ('5min', '1h', '1day', etc.)
            outputsize: Max number of bars (up to 5000)
            start_date: Optional start date 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
            end_date: Optional end date

        Returns:
            DataFrame with lowercase OHLCV columns and DatetimeIndex,
            sorted oldest-first. Empty DataFrame on error.
        """
        if not self.is_supported(symbol):
            logger.warning(f"TwelveData: {symbol} not supported")
            return pd.DataFrame()

        for attempt in range(2):  # retry once on rate limit
            self._rate_limiter.acquire(1)

            try:
                df = self._do_time_series_request(symbol, interval, outputsize, start_date, end_date)
                if df is not None:
                    return df
                # df is None means rate-limit hit — retry after backoff
                if attempt == 0:
                    continue
                return pd.DataFrame()

            except requests.RequestException as e:
                logger.warning(f"TwelveData time_series request failed for {symbol}: {e}")
                return pd.DataFrame()

        return pd.DataFrame()

    def is_supported(self, symbol: str) -> bool:
        """Return False for VIX, indices, and other unsupported symbols."""
        return symbol not in UNSUPPORTED_SYMBOLS

    def clear_cache(self):
        """Clear the quote cache."""
        self._quote_cache.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_time_series_request(
        self,
        symbol: str,
        interval: str,
        outputsize: int,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Optional[pd.DataFrame]:
        """Execute one time_series request. Returns DataFrame on success,
        None on rate-limit (caller should retry), empty DataFrame on other errors."""
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': min(outputsize, 5000),
            'apikey': self.api_key,
        }
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date

        resp = requests.get(
            f"{TWELVEDATA_BASE_URL}/time_series",
            params=params,
            timeout=self.request_timeout,
        )

        if resp.status_code == 429:
            logger.warning("TwelveData rate limit hit (HTTP 429), waiting for next window")
            self._rate_limiter.backoff()
            self._rate_limiter.acquire(1)  # blocks until window clears
            return None  # signal retry

        if resp.status_code != 200:
            logger.warning(
                f"TwelveData time_series failed for {symbol}: HTTP {resp.status_code}"
            )
            return pd.DataFrame()

        data = resp.json()

        if 'code' in data and data['code'] != 200:
            msg = data.get('message', '')
            if 'API credits' in msg or 'rate' in msg.lower():
                logger.info(
                    f"TwelveData credit limit for {symbol} {interval}, "
                    f"waiting for next window"
                )
                self._rate_limiter.backoff()
                self._rate_limiter.acquire(1)  # blocks until window clears
                return None  # signal retry
            logger.warning(
                f"TwelveData time_series error for {symbol}: {msg}"
            )
            return pd.DataFrame()

        values = data.get('values', [])
        if not values:
            logger.warning(f"TwelveData returned no values for {symbol} {interval}")
            return pd.DataFrame()

        # Parse values — TD returns newest-first, values are strings
        rows = []
        for v in values:
            rows.append({
                'datetime': v['datetime'],
                'open': float(v['open']),
                'high': float(v['high']),
                'low': float(v['low']),
                'close': float(v['close']),
                'volume': float(v.get('volume', 0)),
            })

        df = pd.DataFrame(rows)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        df = df.sort_index()  # oldest-first

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        return df

    def _get_cached_quote(self, symbol: str) -> Optional[TwelveDataQuote]:
        """Return cached quote if fresh enough."""
        if symbol not in self._quote_cache:
            return None
        ts, quote = self._quote_cache[symbol]
        if time.monotonic() - ts > self.cache_ttl:
            return None
        return quote
