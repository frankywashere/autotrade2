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
        self._bar_aggregators = {} # {symbol: LiveBarAggregator}
        self._reconnect_delay = 2  # seconds, with exponential backoff
        self.tick_event = threading.Event()  # Set on every tick for instant wakeup
        # Order tracking
        self._trades = {}          # {order_id: ib_async.Trade}
        self._order_log = []       # List of dicts for blotter (max 50)
        self._order_lock = threading.Lock()
        # Account data (cached from IB event loop, read from Panel thread)
        self._account = {}         # {tag: value}
        self._account_lock = threading.Lock()

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

                # Subscribe to account summary updates
                await self._subscribe_account_async()

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
        from ib_async import Stock, Index
        if symbol in ('^VIX', 'VIX'):
            return Index('VIX', 'CBOE', 'USD')
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
                price = ticker.last if ticker.last == ticker.last else 0.0
                bid = ticker.bid if ticker.bid == ticker.bid else 0.0
                ask = ticker.ask if ticker.ask == ticker.ask else 0.0
                self._prices[symbol] = {
                    'last': price,
                    'bid': bid,
                    'ask': ask,
                    'time': datetime.now(),
                }
                # Feed tick into bar aggregator if present
                effective_price = price if price > 0 else (
                    (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0)
                if effective_price > 0 and symbol in self._bar_aggregators:
                    vol = ticker.volume if (hasattr(ticker, 'volume')
                           and ticker.volume == ticker.volume) else 0
                    self._bar_aggregators[symbol].on_tick(effective_price, vol)
        self.tick_event.set()  # Wake up price loop instantly

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
        # VIX is an index — use TRADES for stocks, TRADES for indices too
        # (IB accepts TRADES for most; could also use MIDPOINT for indices)
        what_to_show = 'TRADES'
        if hasattr(contract, 'secType') and contract.secType == 'IND':
            what_to_show = 'TRADES'
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
        )
        return bars

    def fetch_all_tf_history(self, symbol: str, use_rth: bool = True) -> dict:
        """Fetch historical bars at multiple timeframes for channel analysis.

        Returns dict matching native_tf_data format: {tf_name: DataFrame}
        Each DataFrame has columns: [open, high, low, close, volume] with DatetimeIndex.
        """
        tf_configs = {
            '1h':      ('250 D', '1 hour'),
            'daily':   ('1 Y',   '1 day'),
            'weekly':  ('2 Y',   '1 W'),
            'monthly': ('3 Y',   '1 M'),
        }
        result = {}
        for tf_name, (duration, bar_size) in tf_configs.items():
            try:
                df = self.fetch_historical(symbol, duration, bar_size, use_rth=use_rth)
                if df is not None and len(df) > 0:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    result[tf_name] = df
                    logger.info("IB historical %s %s: %d bars", symbol, tf_name, len(df))
            except Exception as e:
                logger.warning("Failed to fetch %s %s: %s", symbol, tf_name, e)
        return result

    def create_bar_aggregator(self, symbol: str, bar_size_minutes: int = 5):
        """Create and register a LiveBarAggregator for a symbol."""
        agg = LiveBarAggregator(bar_size_minutes)
        self._bar_aggregators[symbol] = agg
        logger.info("Bar aggregator created for %s (%d-min bars)", symbol, bar_size_minutes)
        return agg

    def get_bar_aggregator(self, symbol: str):
        """Get the bar aggregator for a symbol, or None."""
        return self._bar_aggregators.get(symbol)

    # ── Account Data ──────────────────────────────────────────────────

    async def _subscribe_account_async(self):
        """Subscribe to account summary (runs in IB event loop)."""
        try:
            await asyncio.wait_for(self.ib.reqAccountSummaryAsync(), timeout=10)
            # Snapshot initial data from wrapper cache into thread-safe dict
            self._snapshot_account()
            # Wire up event for live updates
            self.ib.accountSummaryEvent += self._on_account_summary
            logger.info("Subscribed to account summary (%d tags cached)",
                        len(self._account))
        except Exception as e:
            logger.warning("Account summary subscription failed: %s", e)

    def _snapshot_account(self):
        """Read wrapper.acctSummary directly (no event loop needed)."""
        try:
            items = self.ib.wrapper.acctSummary.values()
            with self._account_lock:
                for item in items:
                    if item.currency in ('USD', ''):
                        self._account[item.tag] = item.value
        except Exception as e:
            logger.error("_snapshot_account failed: %s", e)

    def _on_account_summary(self, item):
        """Callback fired by ib_async on account summary update."""
        if item.currency in ('USD', ''):
            with self._account_lock:
                self._account[item.tag] = item.value

    def get_account_summary(self) -> dict:
        """Return account summary as {tag: value} dict (thread-safe)."""
        with self._account_lock:
            return dict(self._account)

    # ── Order Placement ──────────────────────────────────────────────

    def place_order(self, symbol: str, action: str, qty: int,
                    order_type: str = 'MKT', price: float = 0.0,
                    tif: str = 'DAY', outside_rth: bool = False) -> dict:
        """Place an order via IB. Thread-safe (called from Panel UI thread)."""
        if not self.is_connected():
            return {'error': 'Not connected to IB'}
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._place_order_async(symbol, action, qty, order_type,
                                        price, tif, outside_rth),
                self._loop)
            return future.result(timeout=10)
        except Exception as e:
            logger.error("place_order failed: %s", e)
            return {'error': str(e)}

    async def _place_order_async(self, symbol, action, qty, order_type,
                                  price, tif, outside_rth):
        """Async order placement (runs in IB event loop)."""
        from ib_async import MarketOrder, LimitOrder, StopOrder

        contract = self._make_contract(symbol)
        qualified = await self.ib.qualifyContractsAsync(contract)
        if qualified:
            contract = qualified[0]

        if order_type == 'MKT':
            order = MarketOrder(action, qty)
        elif order_type == 'LMT':
            order = LimitOrder(action, qty, price)
        elif order_type == 'STP':
            order = StopOrder(action, qty, price)
        else:
            return {'error': f'Unknown order type: {order_type}'}

        order.tif = tif
        order.outsideRth = outside_rth

        trade = self.ib.placeOrder(contract, order)
        order_id = trade.order.orderId

        with self._order_lock:
            self._trades[order_id] = trade
            entry = {
                'order_id': order_id,
                'time': datetime.now().strftime('%H:%M:%S'),
                'symbol': symbol,
                'action': action,
                'qty': qty,
                'order_type': order_type,
                'price': price,
                'status': trade.orderStatus.status or 'Submitted',
                'fill_price': 0.0,
                'fill_time': '',
            }
            self._order_log.append(entry)
            if len(self._order_log) > 50:
                self._order_log = self._order_log[-50:]

        trade.statusEvent += self._on_order_status
        logger.info("Order placed: %s %d %s %s @ %.2f (id=%d)",
                     action, qty, symbol, order_type, price, order_id)
        return {'order_id': order_id, 'status': entry['status'], 'message': 'OK'}

    def _on_order_status(self, trade):
        """Callback fired by ib_async on order status change."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        with self._order_lock:
            for entry in self._order_log:
                if entry['order_id'] == order_id:
                    entry['status'] = status
                    if status == 'Filled':
                        entry['fill_price'] = trade.orderStatus.avgFillPrice
                        entry['fill_time'] = datetime.now().strftime('%H:%M:%S')
                    break
        logger.info("Order %d status: %s", order_id, status)

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending order. Returns True if cancel request sent."""
        with self._order_lock:
            trade = self._trades.get(order_id)
        if not trade:
            logger.warning("cancel_order: order_id %d not found", order_id)
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._cancel_order_async(trade), self._loop)
            future.result(timeout=5)
            return True
        except Exception as e:
            logger.error("cancel_order failed: %s", e)
            return False

    async def _cancel_order_async(self, trade):
        """Async cancel (runs in IB event loop)."""
        self.ib.cancelOrder(trade.order)

    def get_order_log(self) -> list:
        """Return a copy of the order log."""
        with self._order_lock:
            return list(self._order_log)


class LiveBarAggregator:
    """Aggregate streaming ticks into OHLCV bars in real-time."""

    def __init__(self, bar_size_minutes: int = 5):
        self.bar_size = bar_size_minutes
        self._current_bar = None       # {open, high, low, close, volume, time}
        self._completed_bars = []      # List of completed bar dicts
        self._lock = threading.Lock()
        self.bar_close_event = threading.Event()  # Set when a bar completes

    def _bar_start_time(self, now: datetime) -> datetime:
        """Round down to nearest bar boundary."""
        minute = (now.minute // self.bar_size) * self.bar_size
        return now.replace(minute=minute, second=0, microsecond=0)

    def on_tick(self, price: float, volume: float = 0):
        """Process a tick. If it crosses a bar boundary, finalize the bar."""
        now = datetime.now()
        bar_start = self._bar_start_time(now)

        with self._lock:
            if self._current_bar is None or self._current_bar['time'] != bar_start:
                # New bar boundary — finalize previous, start new
                if self._current_bar is not None:
                    self._completed_bars.append(self._current_bar)
                    self.bar_close_event.set()
                self._current_bar = {
                    'time': bar_start,
                    'open': price, 'high': price, 'low': price, 'close': price,
                    'volume': volume,
                }
            else:
                # Update current bar
                self._current_bar['high'] = max(self._current_bar['high'], price)
                self._current_bar['low'] = min(self._current_bar['low'], price)
                self._current_bar['close'] = price
                self._current_bar['volume'] += volume

    def get_bars_df(self, include_current: bool = False) -> pd.DataFrame:
        """Return completed bars as DataFrame. Optionally include current partial bar."""
        with self._lock:
            bars = list(self._completed_bars)
            if include_current and self._current_bar:
                bars.append(self._current_bar)
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df['date'] = pd.to_datetime(df['time'])
        df = df.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
        return df

    def bar_count(self) -> int:
        """Number of completed bars."""
        with self._lock:
            return len(self._completed_bars)
