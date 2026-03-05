"""IB connection manager — singleton wrapper around ib_async.

Runs the ib_async event loop in a dedicated daemon thread so it
doesn't block the Panel/Tornado event loop.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class IBClient:
    """Thread-safe IB Gateway client for price streaming + historical data."""

    def __init__(self, host='127.0.0.1', port=4002, client_id=1):
        from ib_async import IB
        self.ib = IB()
        self._host = host
        self._port = port
        self._client_id = client_id
        self._connected = False
        self._prices = {}          # {symbol: {last, bid, ask, time}}
        self._lock = threading.Lock()
        self._loop = None
        self._thread = None
        self._contracts = {}       # {symbol: Contract}
        self._reconnect_delay = 2  # seconds, with exponential backoff

    # ── Connection ───────────────────────────────────────────────────

    def connect(self):
        """Connect to IB Gateway. Starts event loop in a daemon thread."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name='ib-event-loop')
        self._thread.start()

        # Wait for connection (up to 10s)
        deadline = time.time() + 10
        while not self._connected and time.time() < deadline:
            time.sleep(0.1)

        if not self._connected:
            raise ConnectionError(
                f"Could not connect to IB Gateway at {self._host}:{self._port}")
        logger.info("IB connected: %s:%d (client_id=%d)",
                     self._host, self._port, self._client_id)

    def _run_loop(self):
        """Event loop thread — connects and runs forever with auto-reconnect."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_run())

    async def _connect_and_run(self):
        """Connect and keep running. Auto-reconnect on disconnect."""
        delay = self._reconnect_delay
        while True:
            try:
                await self.ib.connectAsync(
                    self._host, self._port, clientId=self._client_id)
                self._connected = True
                delay = self._reconnect_delay  # reset backoff

                # Wire up disconnect handler
                self.ib.disconnectedEvent += self._on_disconnect
                # Wire up tick handler
                self.ib.pendingTickersEvent += self._on_pending_tickers

                logger.info("IB event loop running")
                # Re-subscribe any previously subscribed symbols
                for symbol in list(self._contracts.keys()):
                    await self._subscribe_async(symbol)

                # Run until disconnected
                while self.ib.isConnected():
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.warning("IB connection error: %s (retry in %ds)", e, delay)

            self._connected = False
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)  # exponential backoff, max 60s
            logger.info("IB reconnecting...")

    def _on_disconnect(self):
        """Called when IB Gateway disconnects."""
        self._connected = False
        logger.warning("IB disconnected — will auto-reconnect")

    def disconnect(self):
        """Disconnect from IB Gateway."""
        if self.ib.isConnected():
            self.ib.disconnect()
        self._connected = False
        logger.info("IB disconnected (manual)")

    def reconnect(self):
        """Force reconnect with a new random client ID (clears stale sessions)."""
        import random
        old_id = self._client_id
        self._client_id = random.randint(10, 99)
        logger.info("IB reconnect: client_id %d -> %d", old_id, self._client_id)
        if self.ib.isConnected():
            self.ib.disconnect()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self.ib.isConnected()

    # ── Price Streaming ──────────────────────────────────────────────

    def _make_contract(self, symbol: str):
        """Create IB contract for a symbol."""
        from ib_async import Stock
        return Stock(symbol, 'SMART', 'USD')

    def subscribe(self, symbol: str):
        """Start streaming market data for a symbol."""
        if self._loop and self._connected:
            future = asyncio.run_coroutine_threadsafe(
                self._subscribe_async(symbol), self._loop)
            future.result(timeout=5)

    async def _subscribe_async(self, symbol: str):
        """Subscribe to streaming ticks for symbol (runs in IB event loop)."""
        contract = self._make_contract(symbol)
        # Qualify to resolve conId (required before hashing/subscribing)
        qualified = await self.ib.qualifyContractsAsync(contract)
        if qualified:
            contract = qualified[0]
        self._contracts[symbol] = contract
        self.ib.reqMktData(contract, '', False, False)
        logger.info("Subscribed to %s market data (conId=%s)", symbol, contract.conId)

    def _on_pending_tickers(self, tickers):
        """Callback: update price cache from streaming ticks."""
        with self._lock:
            for ticker in tickers:
                symbol = ticker.contract.symbol
                self._prices[symbol] = {
                    'last': ticker.last if ticker.last == ticker.last else 0.0,
                    'bid': ticker.bid if ticker.bid == ticker.bid else 0.0,
                    'ask': ticker.ask if ticker.ask == ticker.ask else 0.0,
                    'time': datetime.now(),
                }

    def get_last_price(self, symbol: str, max_age_s: float = 30.0) -> float:
        """Read last price from cache. Returns 0.0 if unavailable or stale.

        Args:
            max_age_s: Max seconds since last tick before price is considered stale.
                       Returns 0.0 if stale to force caller to handle missing data.
        """
        with self._lock:
            data = self._prices.get(symbol)
            if not data:
                return 0.0
            # Staleness check: reject cached prices older than max_age_s
            age = (datetime.now() - data['time']).total_seconds()
            if age > max_age_s:
                logger.warning("IB price STALE for %s: %.1fs old (limit %.0fs)", symbol, age, max_age_s)
                return 0.0
            # Prefer last, fall back to mid of bid/ask
            if data['last'] > 0:
                return data['last']
            if data['bid'] > 0 and data['ask'] > 0:
                return (data['bid'] + data['ask']) / 2
            return 0.0

    def get_price_data(self, symbol: str) -> dict:
        """Read full price data (last/bid/ask/time) from cache."""
        with self._lock:
            return self._prices.get(symbol, {}).copy()

    # ── Historical Data ──────────────────────────────────────────────

    def fetch_historical(self, symbol: str, duration: str, bar_size: str,
                         use_rth: bool = False) -> pd.DataFrame:
        """Fetch historical bars from IB.

        Args:
            symbol: e.g. 'TSLA'
            duration: e.g. '1 D', '5 D', '1 M'
            bar_size: e.g. '1 min', '5 mins', '1 hour', '1 day'
            use_rth: True for regular trading hours only

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        contract = self._make_contract(symbol)
        # Qualify contract first
        qual_future = asyncio.run_coroutine_threadsafe(
            self.ib.qualifyContractsAsync(contract), self._loop)
        qualified = qual_future.result(timeout=10)
        if qualified:
            contract = qualified[0]
        future = asyncio.run_coroutine_threadsafe(
            self._fetch_historical_async(contract, duration, bar_size, use_rth),
            self._loop)
        bars = future.result(timeout=30)

        if not bars:
            return pd.DataFrame()

        records = []
        for bar in bars:
            records.append({
                'date': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': int(bar.volume),
            })
        return pd.DataFrame(records)

    async def _fetch_historical_async(self, contract, duration, bar_size, use_rth):
        """Async wrapper for reqHistoricalData."""
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=use_rth,
            formatDate=1,
        )
        return bars
