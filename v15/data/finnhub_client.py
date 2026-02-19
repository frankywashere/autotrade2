"""
Finnhub Real-Time Quote Client

Provides real-time stock quotes via Finnhub's free-tier REST API.
Used for live price display in the dashboard — NOT for historical candles
(those stay on yfinance since Finnhub paywalled /stock/candle).

Free-tier limitations:
- 30 API calls/second
- Quote endpoint only (no candles, no indices like ^VIX)
- US stocks only

Usage:
    from v15.data.finnhub_client import FinnhubClient

    client = FinnhubClient()
    quote = client.get_quote('TSLA')
    print(f"TSLA: ${quote['c']:.2f}")
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Hardcoded free-tier API key
FINNHUB_API_KEY = 'd4qh0u9r01quli1cimbgd4qh0u9r01quli1cimc0'
FINNHUB_BASE_URL = 'https://finnhub.io/api/v1'


@dataclass
class FinnhubQuote:
    """Real-time quote from Finnhub."""
    symbol: str
    current_price: float   # c - current price
    change: float          # d - change
    change_pct: float      # dp - percent change
    high: float            # h - day high
    low: float             # l - day low
    open: float            # o - day open
    prev_close: float      # pc - previous close
    timestamp: float       # t - unix timestamp


class FinnhubClient:
    """
    Minimal Finnhub REST client for real-time quotes.

    Only supports the /quote endpoint (free tier).
    Rate-limited to 30 calls/sec with in-memory caching.
    """

    # Symbols that work on Finnhub free tier
    SUPPORTED_SYMBOLS = {'TSLA', 'SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'}

    # Symbols that DON'T work (indices require paid subscription)
    UNSUPPORTED_SYMBOLS = {'^VIX', 'VIX', '^GSPC', '^DJI', '^IXIC'}

    def __init__(
        self,
        api_key: str = FINNHUB_API_KEY,
        cache_ttl: float = 5.0,
        request_timeout: float = 5.0,
    ):
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self.request_timeout = request_timeout
        self._cache: Dict[str, tuple] = {}  # symbol -> (timestamp, FinnhubQuote)
        self._last_request_time: float = 0.0

    def get_quote(self, symbol: str) -> Optional[FinnhubQuote]:
        """
        Get real-time quote for a US stock symbol.

        Args:
            symbol: Stock symbol (e.g., 'TSLA', 'SPY').
                    Indices like '^VIX' are NOT supported on free tier.

        Returns:
            FinnhubQuote or None if the symbol is unsupported or request fails.
        """
        if symbol in self.UNSUPPORTED_SYMBOLS:
            logger.debug(f"Finnhub: {symbol} not supported on free tier")
            return None

        # Check cache
        cached = self._get_cached(symbol)
        if cached is not None:
            return cached

        # Rate limiting: minimum 35ms between requests (< 30/sec)
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < 0.035:
            time.sleep(0.035 - elapsed)

        try:
            url = f"{FINNHUB_BASE_URL}/quote"
            params = {'symbol': symbol, 'token': self.api_key}
            resp = requests.get(url, params=params, timeout=self.request_timeout)
            self._last_request_time = time.monotonic()

            if resp.status_code == 429:
                logger.warning("Finnhub rate limit hit, backing off")
                time.sleep(1.0)
                return None

            if resp.status_code != 200:
                logger.warning(f"Finnhub quote failed for {symbol}: HTTP {resp.status_code}")
                return None

            data = resp.json()

            if 'error' in data:
                logger.warning(f"Finnhub quote error for {symbol}: {data['error']}")
                return None

            # Validate response has actual data (c=0 means no data)
            if data.get('c', 0) == 0 and data.get('t', 0) == 0:
                logger.warning(f"Finnhub returned empty quote for {symbol}")
                return None

            quote = FinnhubQuote(
                symbol=symbol,
                current_price=float(data.get('c', 0)),
                change=float(data.get('d', 0) or 0),
                change_pct=float(data.get('dp', 0) or 0),
                high=float(data.get('h', 0)),
                low=float(data.get('l', 0)),
                open=float(data.get('o', 0)),
                prev_close=float(data.get('pc', 0)),
                timestamp=float(data.get('t', 0)),
            )

            # Cache it
            self._cache[symbol] = (time.monotonic(), quote)
            return quote

        except requests.RequestException as e:
            logger.warning(f"Finnhub request failed for {symbol}: {e}")
            return None

    def get_quotes(self, symbols: list) -> Dict[str, Optional[FinnhubQuote]]:
        """
        Get real-time quotes for multiple symbols.

        Skips unsupported symbols (like ^VIX).

        Returns:
            Dict mapping symbol to FinnhubQuote (or None if unavailable).
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_quote(symbol)
        return results

    def is_supported(self, symbol: str) -> bool:
        """Check if a symbol is supported on Finnhub free tier."""
        return symbol not in self.UNSUPPORTED_SYMBOLS

    def _get_cached(self, symbol: str) -> Optional[FinnhubQuote]:
        """Return cached quote if fresh enough."""
        if symbol not in self._cache:
            return None
        ts, quote = self._cache[symbol]
        if time.monotonic() - ts > self.cache_ttl:
            return None
        return quote

    def clear_cache(self):
        """Clear the quote cache."""
        self._cache.clear()
