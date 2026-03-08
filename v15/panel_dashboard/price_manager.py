"""
PriceManager — Single source of truth for all price data.

Tracks IB tick-driven prices. Thread-safe for concurrent access from
price loops and UI callbacks.
"""

import threading
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PriceData:
    """Snapshot of price data for a single (symbol, source) pair."""
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    updated_at: float = 0.0  # time.time()
    stale: bool = True        # True if >30s since last update

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.price

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0


class PriceManager:
    """Single source of truth for all price data.

    Tracks IB tick-driven prices. Thread-safe — all updates and reads go
    through a lock.
    """

    STALE_THRESHOLD = 30.0  # seconds before marking a price stale

    def __init__(self, ib_client=None):
        self._ib = ib_client
        self._prices: dict[tuple[str, str], PriceData] = {}
        self._lock = threading.Lock()
        self._err_counts: dict[str, int] = {}  # {symbol: consecutive_error_count}

    def update_ib(self, symbol: str, price: float,
                  bid: float = 0.0, ask: float = 0.0):
        """Called from IB tick callback or price loop."""
        if price <= 0:
            return
        now = time.time()
        with self._lock:
            key = (symbol, 'ib')
            pd = self._prices.get(key)
            if pd is None:
                pd = PriceData()
                self._prices[key] = pd
            pd.price = price
            if bid > 0:
                pd.bid = bid
            if ask > 0:
                pd.ask = ask
            pd.updated_at = now
            pd.stale = False
            self._err_counts[symbol] = 0

    def get(self, symbol: str, source: str = 'ib') -> PriceData:
        """Get latest price for (symbol, source). Returns PriceData (never None)."""
        now = time.time()
        with self._lock:
            key = (symbol, source)
            pd = self._prices.get(key)
            if pd is None:
                return PriceData()  # No data yet
            # Mark stale if too old
            if now - pd.updated_at > self.STALE_THRESHOLD:
                pd.stale = True
            return PriceData(
                price=pd.price, bid=pd.bid, ask=pd.ask,
                updated_at=pd.updated_at, stale=pd.stale,
            )

    def get_price(self, symbol: str, source: str = 'ib') -> float:
        """Convenience: get just the price value."""
        return self.get(symbol, source).price

    def record_error(self, symbol: str):
        """Record a price fetch error (for consecutive error tracking)."""
        with self._lock:
            self._err_counts[symbol] = self._err_counts.get(symbol, 0) + 1

    def get_error_count(self, symbol: str) -> int:
        """Get consecutive error count for a symbol."""
        with self._lock:
            return self._err_counts.get(symbol, 0)

    @property
    def ib_connected(self) -> bool:
        """True if IB client is connected."""
        return self._ib is not None and self._ib.is_connected()

    @property
    def ib_stale(self) -> bool:
        """True if IB prices haven't updated in >30s."""
        tsla = self.get('TSLA', 'ib')
        return tsla.stale or tsla.price <= 0

    def refresh_from_ib(self):
        """Pull latest prices from IB client (called by price loop)."""
        if not self._ib or not self._ib.is_connected():
            return False

        updated = False
        for symbol in ['TSLA', 'SPY', 'VIX']:
            price = self._ib.get_last_price(symbol)
            if price > 0:
                bid = getattr(self._ib, 'get_bid', lambda s: 0.0)(symbol)
                ask = getattr(self._ib, 'get_ask', lambda s: 0.0)(symbol)
                self.update_ib(symbol, price, bid, ask)
                updated = True
        return updated

    def get_price_data(self) -> dict:
        """Get all IB prices as a dict (for UI compatibility)."""
        return {
            'tsla': self.get_price('TSLA', 'ib'),
            'spy': self.get_price('SPY', 'ib'),
            'vix': self.get_price('VIX', 'ib'),
            'bid': self.get('TSLA', 'ib').bid,
            'ask': self.get('TSLA', 'ib').ask,
            'mid': self.get('TSLA', 'ib').mid,
        }
