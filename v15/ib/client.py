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
        self._bar_aggregators = {} # {symbol: LiveBarAggregator} (legacy, single per symbol)
        self._extra_aggregators = {}  # {symbol: [LiveBarAggregator, ...]} (additional)
        self._reconnect_delay = 2  # seconds, with exponential backoff
        self.tick_event = threading.Event()  # Set on every tick for instant wakeup
        # Order tracking — IB is source of truth
        self._trades = {}          # {order_id: ib_async.Trade} (this session only)
        self._order_log = []       # Unified blotter: open + completed from IB
        self._order_lock = threading.Lock()
        self._order_time_cache = {}  # {perm_id: (order_time, sort_time)} for stable timestamps
        self._order_log_version = 0  # Bumped on any change
        self._completed_cache = {}   # {perm_id: blotter_entry} — seeded once, updated reactively
        # External callbacks
        self._order_status_external = None   # IBOrderHandler.on_order_status
        self._degraded_callback = None       # called when critical callback fails
        self._reconnect_callbacks = []       # fired after IB reconnect

        # Account data (cached from IB event loop, read from Panel thread)
        self._account = {}         # {tag: value}
        self._account_lock = threading.Lock()
        # Portfolio positions (cached from IB event loop)
        self._positions = {}       # {symbol: {position, avgCost, marketPrice, marketValue, unrealizedPNL, realizedPNL}}
        self._positions_lock = threading.Lock()

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
        first_connect = True
        # Monkey-patch ib_async wrapper to preserve completedTime on Trade objects.
        # ib_async discards OrderState (which has completedTime) when building Trade
        # from completed orders — we patch to save it on OrderStatus as a custom attr.
        _orig_completed = self.ib.wrapper.completedOrder

        def _patched_completed(contract, order, orderState):
            _orig_completed(contract, order, orderState)
            # The Trade was just appended to results — attach completedTime
            results = self.ib.wrapper._results.get("completedOrders", [])
            if results:
                trade = results[-1]
                ct = getattr(orderState, 'completedTime', '')
                if ct:
                    trade.orderStatus.completedTime = ct

        self.ib.wrapper.completedOrder = _patched_completed

        # Wire up events ONCE (before reconnect loop to avoid duplicates)
        self.ib.disconnectedEvent += self._on_disconnect
        self.ib.pendingTickersEvent += self._on_pending_tickers
        self.ib.errorEvent += self._on_error
        self.ib.updatePortfolioEvent += self._on_portfolio_update
        self.ib.accountSummaryEvent += self._on_account_summary

        while True:
            try:
                await self.ib.connectAsync(
                    self._host, self._port, clientId=self._client_id)
                self._connected = True
                delay = self._reconnect_delay  # reset backoff

                logger.info("IB event loop running (server v%d)",
                            self.ib.client.serverVersion())
                # Re-subscribe any previously subscribed symbols
                for symbol in list(self._contracts.keys()):
                    await self._subscribe_async(symbol)

                # Subscribe to account summary + snapshot positions
                await self._subscribe_account_async()
                self._snapshot_positions()

                # Seed completed orders (one-time) then sync open orders
                await self._seed_completed_orders()
                await self._sync_open_orders()

                # On reconnect (not first connect): re-wire statusEvent on new
                # Trade objects and fire reconnect callbacks in a separate thread
                # to avoid deadlocking the IB event loop (callbacks may call
                # fetch_historical which uses run_coroutine_threadsafe on this loop).
                if not first_connect:
                    for trade in self.ib.openTrades():
                        trade.statusEvent += self._on_order_status
                    if self._reconnect_callbacks:
                        import threading as _th
                        def _fire_reconnect_cbs(cbs=list(self._reconnect_callbacks)):
                            for cb in cbs:
                                try:
                                    cb()
                                except Exception as e:
                                    logger.error("Reconnect callback failed: %s", e)
                        _th.Thread(target=_fire_reconnect_cbs, daemon=True,
                                   name='ib-reconnect-cb').start()
                        logger.info("IB reconnected — fired %d callbacks in background",
                                    len(self._reconnect_callbacks))
                first_connect = False

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
        self._disconnect_time = time.time()
        logger.warning("IB disconnected — will auto-reconnect")
        for cb in getattr(self, '_disconnect_callbacks', []):
            try:
                cb()
            except Exception as e:
                logger.error("Disconnect callback failed: %s", e)

    def disconnect(self):
        """Disconnect from IB Gateway."""
        if self.ib.isConnected():
            self.ib.disconnect()
        self._connected = False
        logger.info("IB disconnected (manual)")

    def reconnect(self):
        """Force reconnect (reuses same client ID so orders stay cancellable)."""
        logger.info("IB reconnect: reusing client_id %d", self._client_id)
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
                if effective_price > 0:
                    vol = ticker.volume if (hasattr(ticker, 'volume')
                           and ticker.volume == ticker.volume) else 0
                    if symbol in self._bar_aggregators:
                        self._bar_aggregators[symbol].on_tick(effective_price, vol)
                    for agg in self._extra_aggregators.get(symbol, []):
                        agg.on_tick(effective_price, vol)
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
                # Rate-limit stale warnings to once per 60s per symbol
                now = datetime.now()
                last_warn = getattr(self, '_stale_warn_times', {}).get(symbol)
                if not last_warn or (now - last_warn).total_seconds() > 60:
                    if not hasattr(self, '_stale_warn_times'):
                        self._stale_warn_times = {}
                    self._stale_warn_times[symbol] = now
                    logger.warning("IB price STALE for %s: %.1fs old (limit %.0fs)",
                                   symbol, age, max_age_s)
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
        df = pd.DataFrame(records)
        # Normalize all timestamps to tz-naive (project convention).
        # IB returns tz-aware datetimes for intraday bars, tz-naive date
        # objects for daily+.  Strip tz here so callers never see tz-aware.
        df['date'] = pd.to_datetime(df['date'])
        if getattr(df['date'].dt, 'tz', None) is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        return df

    async def _fetch_historical_async(self, contract, duration, bar_size, use_rth):
        """Async wrapper for reqHistoricalData."""
        # IB requires TRADES for index contracts (VIX) — MIDPOINT returns 0 bars
        # Index contracts have no RTH distinction — always use useRTH=False
        if contract.secType == 'IND':
            what_to_show = 'TRADES'
            actual_use_rth = False
        else:
            what_to_show = 'TRADES'
            actual_use_rth = use_rth
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=actual_use_rth,
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
        """Create and register a LiveBarAggregator for a symbol.

        Note: only one primary aggregator per symbol (overwrites previous).
        Use add_bar_aggregator() for additional aggregators.
        """
        agg = LiveBarAggregator(bar_size_minutes)
        self._bar_aggregators[symbol] = agg
        logger.info("Bar aggregator created for %s (%d-min bars)", symbol, bar_size_minutes)
        return agg

    def add_bar_aggregator(self, symbol: str, bar_size_minutes: int = 1):
        """Add an additional bar aggregator for a symbol (doesn't overwrite primary)."""
        agg = LiveBarAggregator(bar_size_minutes)
        self._extra_aggregators.setdefault(symbol, []).append(agg)
        logger.info("Extra bar aggregator added for %s (%d-min bars)", symbol, bar_size_minutes)
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
                    if item.account and 'AccountCode' not in self._account:
                        self._account['AccountCode'] = item.account
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

    # ── Portfolio Positions ───────────────────────────────────────────

    def _snapshot_positions(self):
        """Read initial positions from wrapper (called after connect)."""
        try:
            for item in self.ib.portfolio():
                symbol = item.contract.symbol
                with self._positions_lock:
                    self._positions[symbol] = {
                        'position': float(item.position),
                        'avgCost': float(item.averageCost),
                        'marketPrice': float(item.marketPrice),
                        'marketValue': float(item.marketValue),
                        'unrealizedPNL': float(item.unrealizedPNL),
                        'realizedPNL': float(item.realizedPNL),
                    }
            logger.info("Portfolio snapshot: %d positions", len(self._positions))
        except Exception as e:
            logger.error("_snapshot_positions failed: %s", e)

    def _on_portfolio_update(self, item):
        """Callback fired by ib_async on portfolio update."""
        symbol = item.contract.symbol
        with self._positions_lock:
            if item.position == 0:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = {
                    'position': float(item.position),
                    'avgCost': float(item.averageCost),
                    'marketPrice': float(item.marketPrice),
                    'marketValue': float(item.marketValue),
                    'unrealizedPNL': float(item.unrealizedPNL),
                    'realizedPNL': float(item.realizedPNL),
                }

    def get_positions(self) -> dict:
        """Return open positions as {symbol: {...}} dict (thread-safe)."""
        with self._positions_lock:
            return {k: dict(v) for k, v in self._positions.items()}

    # ── Order Placement ──────────────────────────────────────────────

    def place_order(self, symbol: str, action: str, qty: int,
                    order_type: str = 'MKT', price: float = 0.0,
                    tif: str = 'DAY', outside_rth: bool = False,
                    overnight: bool = False,
                    order_ref: str = '', model_code: str = None,
                    good_after_time: str = '') -> dict:
        """Place an order via IB. Thread-safe (called from Panel UI thread).

        Args:
            overnight: Enable Blue Ocean ATS routing (8PM-3:50AM ET) via includeOvernight.
            order_ref: Durable orderRef for crash recovery matching (max 80 chars).
            model_code: IB modelCode for FA accounts (algo_id). None = don't set.
            good_after_time: IB goodAfterTime string (yyyymmdd hh:mm:ss {tz}).
                             Order stays inactive until this time.
        """
        if not self.is_connected():
            return {'error': 'Not connected to IB'}
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._place_order_async(symbol, action, qty, order_type,
                                        price, tif, outside_rth,
                                        overnight, order_ref, model_code,
                                        good_after_time),
                self._loop)
            return future.result(timeout=10)
        except TimeoutError:
            # Order may be in-flight at IB but we didn't get the response.
            # The order_ref is set, so recovery can find it on next restart.
            logger.error("place_order TIMED OUT: %s %d %s %s @ %.2f (ref=%s) — "
                         "order may be in-flight at IB",
                         action, qty, symbol, order_type, price,
                         order_ref[:30] if order_ref else '')
            return {'error': f'Timeout — order may be in-flight (ref={order_ref[:30]})'}
        except Exception as e:
            logger.error("place_order failed: %s", e)
            return {'error': str(e)}

    async def _place_order_async(self, symbol, action, qty, order_type,
                                  price, tif, outside_rth,
                                  overnight=False, order_ref='',
                                  model_code=None,
                                  good_after_time=''):
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

        if overnight:
            # Blue Ocean ATS: exchange='OVERNIGHT', LMT only, DAY TIF only, 8PM-3:50AM ET
            # IBC config BypassOrderPrecautions=yes required (clears Error 10329)
            # outsideRth is ignored for OVERNIGHT exchange (IB warning 2109)
            contract.exchange = 'OVERNIGHT'
            order.tif = 'DAY'
            order.outsideRth = False

        # Durable orderRef for crash recovery matching
        if order_ref:
            order.orderRef = order_ref[:80]  # IB limit

        # FA modelCode for per-algo position tracking
        if model_code:
            order.modelCode = model_code

        # Delay activation (stop stays inactive until this time)
        if good_after_time:
            order.goodAfterTime = good_after_time

        trade = self.ib.placeOrder(contract, order)
        order_id = trade.order.orderId

        # Wire status callback IMMEDIATELY so rejections during the permId
        # poll are not lost (IB rejects asynchronously via statusEvent).
        with self._order_lock:
            self._trades[order_id] = trade
        trade.statusEvent += self._on_order_status

        # Wait for IB to assign permId (usually arrives in <100ms)
        if trade.order.permId == 0:
            for _ in range(10):  # 10 × 50ms = 500ms max
                await asyncio.sleep(0.05)
                if trade.order.permId != 0:
                    break

        perm_id = trade.order.permId
        if perm_id == 0:
            logger.warning("permId still 0 after 500ms for order %d (ref=%s) — "
                           "will update on first fill",
                           order_id, order_ref[:30] if order_ref else '')
        logger.info("Order placed: %s %d %s %s @ %.2f (id=%d, perm=%d, ref=%s)",
                     action, qty, symbol, order_type, price, order_id,
                     perm_id, order_ref[:30] if order_ref else '')

        return {'order_id': order_id, 'perm_id': perm_id,
                'status': trade.orderStatus.status or 'Submitted', 'message': 'OK'}

    def _on_order_status(self, trade):
        """Callback fired by ib_async on order status change."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        logger.info("Order %d status: %s", order_id, status)
        # Forward to external handler (IBOrderHandler) for terminal status tracking
        if self._order_status_external:
            try:
                self._order_status_external(trade)
            except Exception as e:
                logger.error("CRITICAL: Order status callback failed for "
                             "order %d (%s): %s", order_id, status, e, exc_info=True)
                if self._degraded_callback:
                    try:
                        self._degraded_callback(
                            f"Order status callback failed: order {order_id}, "
                            f"status {status}")
                    except Exception:
                        pass
        # Update completed cache reactively when order reaches terminal state
        if status in ('Filled', 'Cancelled', 'Inactive'):
            entry = self._trade_to_entry(trade)
            if entry:
                with self._order_lock:
                    self._completed_cache[entry['perm_id']] = entry
                    self._order_log_version += 1

    # IB warning codes that do NOT indicate order rejection
    _IB_WARNING_CODES = {
        105, 110, 161, 165, 321, 329, 399, 404, 434, 492,
        2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109,
        2110, 2137, 2158, 2169, 10167, 10197,
        10329,  # OVERNIGHT exchange precautionary (bypassed via IBC BypassOrderPrecautions)
    }

    def _on_error(self, reqId, errorCode, errorString, contract):
        """Callback for IB errors — log warnings vs rejections."""
        if reqId > 0 and errorCode not in self._IB_WARNING_CODES:
            logger.warning("Order %d rejected: [%d] %s", reqId, errorCode, errorString)
            # Bump version so blotter refreshes on next periodic tick
            with self._order_lock:
                self._order_log_version += 1
        elif reqId > 0:
            logger.info("Order %d warning: [%d] %s", reqId, errorCode, errorString)

    def modify_stop_price(self, order_id: int, new_price: float) -> bool:
        """Modify stop price on an existing order in place (no cancel+replace)."""
        with self._order_lock:
            trade = self._trades.get(order_id)
        if not trade:
            logger.error("modify_stop_price: order %d not found", order_id)
            return False
        try:
            trade.order.goodAfterTime = ''  # Clear grace — stop should be active
            trade.order.auxPrice = round(new_price, 2)
            future = asyncio.run_coroutine_threadsafe(
                self._modify_order_async(trade), self._loop)
            future.result(timeout=5)
            return True
        except Exception as e:
            logger.error("modify_stop_price failed for order %d: %s", order_id, e)
            return False

    async def _modify_order_async(self, trade):
        """Modify an existing order (IB treats placeOrder with existing orderId as modify)."""
        self.ib.placeOrder(trade.contract, trade.order)

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending order by orderId or permId. Returns True if cancel request sent."""
        with self._order_lock:
            trade = self._trades.get(order_id)
            # Also check by permId (cancel buttons use permId)
            if not trade:
                for t in self._trades.values():
                    if t.order.permId == order_id:
                        trade = t
                        break
        if trade:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._cancel_order_async(trade), self._loop)
                future.result(timeout=5)
                return True
            except Exception as e:
                logger.error("cancel_order failed: %s", e)
                return False
        # Not in our session trades — try to find it in IB's open trades
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._cancel_order_by_id_async(order_id), self._loop)
            return future.result(timeout=5)
        except Exception as e:
            logger.error("cancel_order (by id) failed: %s", e)
            return False

    async def _cancel_order_async(self, trade):
        """Async cancel (runs in IB event loop)."""
        self.ib.cancelOrder(trade.order)

    async def _cancel_order_by_id_async(self, order_id):
        """Cancel an order by ID — searches IB's open trades."""
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id or trade.order.permId == order_id:
                self.ib.cancelOrder(trade.order)
                logger.info("Cancel sent for order %d (found in IB open trades)", order_id)
                return True
        logger.warning("cancel_order: order_id %d not found in IB open trades", order_id)
        return False

    # ── IB Order Sync (source of truth) ──────────────────────────────

    async def _seed_completed_orders(self):
        """One-time startup: fetch completed orders from IB for blotter history."""
        try:
            completed = await asyncio.wait_for(
                self.ib.reqCompletedOrdersAsync(apiOnly=False), timeout=10.0)
            with self._order_lock:
                for trade in completed:
                    entry = self._trade_to_entry(trade)
                    if entry:
                        self._completed_cache[entry['perm_id']] = entry
            logger.info("Seeded %d completed orders from IB", len(self._completed_cache))
        except asyncio.TimeoutError:
            logger.warning("reqCompletedOrdersAsync timed out at startup — "
                           "blotter may miss older filled orders")
        except Exception as e:
            logger.error("_seed_completed_orders failed: %s", e)

    async def _sync_open_orders(self):
        """Lightweight periodic sync — only fetches open orders (cached by ib_async).

        Completed orders are seeded once at startup and updated reactively
        via _on_order_status when orders reach terminal state.
        """
        try:
            await asyncio.wait_for(self.ib.reqAllOpenOrdersAsync(), timeout=5.0)
            open_trades = self.ib.openTrades()

            orders = {}  # keyed by permId to deduplicate

            # Open orders from IB
            for trade in open_trades:
                entry = self._trade_to_entry(trade)
                if entry:
                    orders[entry['perm_id']] = entry
                    with self._order_lock:
                        self._trades[trade.order.orderId] = trade

            # Merge in completed order cache (don't overwrite open with completed)
            with self._order_lock:
                for perm_id, entry in self._completed_cache.items():
                    if perm_id not in orders:
                        orders[perm_id] = entry

            # Sort by time descending (newest first), limit to 50
            sorted_orders = sorted(orders.values(),
                                   key=lambda e: e.get('sort_time', ''),
                                   reverse=True)[:50]

            with self._order_lock:
                old_snapshot = [(e.get('perm_id'), e.get('status')) for e in self._order_log]
                new_snapshot = [(e.get('perm_id'), e.get('status')) for e in sorted_orders]
                self._order_log = sorted_orders
                if new_snapshot != old_snapshot:
                    self._order_log_version += 1

        except asyncio.TimeoutError:
            logger.debug("_sync_open_orders timed out")
        except Exception as e:
            logger.error("_sync_open_orders failed: %s", e)

    def _trade_to_entry(self, trade) -> dict:
        """Convert an ib_async Trade or CompletedOrder to a blotter entry."""
        try:
            order = trade.order
            status_obj = trade.orderStatus
            contract = trade.contract

            # Determine order type string
            otype = getattr(order, 'orderType', 'UNK')
            if otype == 'MKT':
                price = 0.0
            elif otype == 'LMT':
                price = getattr(order, 'lmtPrice', 0.0)
            elif otype == 'STP':
                price = getattr(order, 'auxPrice', 0.0)
            else:
                price = getattr(order, 'lmtPrice', 0.0) or getattr(order, 'auxPrice', 0.0)

            status = getattr(status_obj, 'status', 'Unknown')
            fill_price = getattr(status_obj, 'avgFillPrice', 0.0)
            order_id = getattr(order, 'orderId', 0)
            perm_id = getattr(order, 'permId', order_id)

            # Extract time from trade log entries, fills, or order attributes
            order_time = ''
            sort_time = ''
            fill_time = ''

            # 1. Try trade.log (populated for live/open trades)
            if hasattr(trade, 'log') and trade.log:
                first_log = trade.log[0]
                t = getattr(first_log, 'time', None)
                if t:
                    order_time = t.strftime('%H:%M:%S')
                    sort_time = t.isoformat()
                # Fill time from last Filled log entry
                if status == 'Filled':
                    for log_entry in reversed(trade.log):
                        if getattr(log_entry, 'status', '') == 'Filled':
                            t2 = getattr(log_entry, 'time', None)
                            if t2:
                                fill_time = t2.strftime('%H:%M:%S')
                            break

            # 2. Try fills list (has execution times)
            if not fill_time and hasattr(trade, 'fills') and trade.fills:
                last_fill = trade.fills[-1]
                exec_obj = getattr(last_fill, 'execution', None)
                if exec_obj:
                    ft = getattr(exec_obj, 'time', None)
                    if ft:
                        fill_time = ft.strftime('%H:%M:%S') if hasattr(ft, 'strftime') else str(ft)[:8]
                        if not sort_time:
                            sort_time = ft.isoformat() if hasattr(ft, 'isoformat') else str(ft)
                            order_time = fill_time

            # 3. Fallback: try completedTime on orderStatus (set by our monkey-patch)
            if not sort_time:
                completed_time = getattr(status_obj, 'completedTime', '')
                if completed_time:
                    sort_time = completed_time
                    try:
                        from datetime import datetime as dt
                        # IB format: "YYYYMMDD-HH:MM:SS" or "YYYYMMDD HH:MM:SS"
                        ct_clean = completed_time.replace('Z', '').strip()
                        for fmt in ('%Y%m%d-%H:%M:%S', '%Y%m%d %H:%M:%S',
                                    '%Y-%m-%dT%H:%M:%S', '%Y%m%d  %H:%M:%S'):
                            try:
                                ct = dt.strptime(ct_clean[:17], fmt)
                                order_time = ct.strftime('%H:%M:%S')
                                sort_time = ct.isoformat()
                                if not fill_time and status == 'Filled':
                                    fill_time = order_time
                                break
                            except ValueError:
                                continue
                        else:
                            # Couldn't parse — use raw string
                            order_time = completed_time[-8:] if len(completed_time) >= 8 else completed_time
                    except Exception as e:
                        logger.warning("Failed to parse order time '%s': %s", completed_time, e)
                        order_time = completed_time[-8:] if len(completed_time) >= 8 else completed_time

            # 4. Last resort: reuse previously cached time, or fall back to now
            if not sort_time:
                cached = self._order_time_cache.get(perm_id)
                if cached:
                    order_time, sort_time = cached
                else:
                    order_time = datetime.now().strftime('%H:%M:%S')
                    sort_time = datetime.now().isoformat()
                    self._order_time_cache[perm_id] = (order_time, sort_time)

            return {
                'order_id': order_id,
                'perm_id': perm_id,
                'time': order_time,
                'sort_time': sort_time,
                'symbol': getattr(contract, 'symbol', '?'),
                'action': getattr(order, 'action', '?'),
                'qty': int(getattr(order, 'totalQuantity', 0)),
                'order_type': otype,
                'price': price,
                'status': status,
                'fill_price': fill_price,
                'fill_time': fill_time,
                'exchange': getattr(contract, 'exchange', ''),
                'order_ref': getattr(order, 'orderRef', ''),
                'client_id': getattr(order, 'clientId', 0),
            }
        except Exception as e:
            logger.error("_trade_to_entry failed: %s", e)
            return None

    def get_order_log(self) -> list:
        """Return a copy of the order log (sourced from IB)."""
        with self._order_lock:
            return list(self._order_log)

    def get_order_log_version(self) -> int:
        """Return the current order log version (for change detection)."""
        with self._order_lock:
            return self._order_log_version

    def register_order_status_callback(self, callback):
        """Register external callback for order status events (e.g., IBOrderHandler).

        Called from _on_order_status for every status change on every order.
        """
        self._order_status_external = callback

    def register_degraded_callback(self, callback):
        """Register callback to set degraded when a critical callback fails."""
        self._degraded_callback = callback

    def register_reconnect_callback(self, callback):
        """Register callback fired after IB auto-reconnect."""
        self._reconnect_callbacks.append(callback)

    def register_disconnect_callback(self, callback):
        """Register callback fired when IB disconnects."""
        if not hasattr(self, '_disconnect_callbacks'):
            self._disconnect_callbacks = []
        self._disconnect_callbacks.append(callback)

    def sync_orders(self):
        """Thread-safe trigger to re-sync open orders from IB.

        Only fetches open orders (lightweight). Completed orders are tracked
        reactively via _on_order_status callbacks.
        """
        if self._loop and self._connected:
            # Guard: skip if a previous sync is still running
            if hasattr(self, '_sync_task') and self._sync_task and not self._sync_task.done():
                return
            self._sync_task = asyncio.run_coroutine_threadsafe(
                self._sync_open_orders(), self._loop)


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
