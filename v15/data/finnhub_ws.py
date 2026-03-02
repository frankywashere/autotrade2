"""
Finnhub WebSocket Service — real-time TSLA/SPY trade ticks.

Background daemon thread maintaining a persistent WebSocket connection
to Finnhub for sub-second price updates. Used by the dashboard for
live price display and tight exit monitoring.

Usage:
    from v15.data.finnhub_ws import get_ws_client

    ws = get_ws_client()
    tick = ws.get_price('TSLA')
    if tick:
        print(f"TSLA: ${tick.price:.2f} ({tick.age_seconds:.1f}s ago)")
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PriceTick:
    """Latest trade tick for a symbol."""
    price: float
    timestamp: float    # Unix epoch seconds
    volume: float
    symbol: str

    @property
    def age_seconds(self) -> float:
        """How stale this tick is."""
        return time.time() - self.timestamp

    @property
    def is_fresh(self) -> bool:
        """True if tick is less than 5 seconds old."""
        return self.age_seconds < 5.0


class FinnhubWebSocket:
    """
    Background WebSocket client for Finnhub real-time trades.

    Runs as a daemon thread. Stores only the latest tick per symbol.
    Auto-reconnects with exponential backoff on disconnect.
    """

    FINNHUB_WS_URL = 'wss://ws.finnhub.io?token={api_key}'
    MAX_BACKOFF = 30.0
    INITIAL_BACKOFF = 1.0

    def __init__(self, api_key: str, symbols: Optional[List[str]] = None):
        self._api_key = api_key
        self._symbols = symbols or ['TSLA', 'SPY']
        self._lock = threading.Lock()
        self._latest: Dict[str, PriceTick] = {}
        self._connected = False
        self._should_run = True
        self._backoff = self.INITIAL_BACKOFF
        self._ws = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background WebSocket thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._should_run = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name='finnhub-ws')
        self._thread.start()
        logger.info("Finnhub WebSocket thread started")

    def stop(self):
        """Stop the background thread."""
        self._should_run = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def get_price(self, symbol: str) -> Optional[PriceTick]:
        """Thread-safe read of latest price tick for a symbol."""
        with self._lock:
            return self._latest.get(symbol)

    @property
    def connected(self) -> bool:
        return self._connected

    def _run_loop(self):
        """Main loop: connect, subscribe, receive. Reconnect on failure."""
        while self._should_run:
            try:
                self._connect_and_listen()
            except Exception as e:
                logger.warning(f"Finnhub WS error: {e}")
            finally:
                self._connected = False

            if not self._should_run:
                break

            # Exponential backoff
            logger.info(f"Finnhub WS reconnecting in {self._backoff:.1f}s...")
            time.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, self.MAX_BACKOFF)

    def _connect_and_listen(self):
        """Open WebSocket, subscribe to symbols, and process messages."""
        try:
            import websocket
        except ImportError:
            logger.error("websocket-client not installed. Run: pip install websocket-client")
            self._should_run = False
            return

        url = self.FINNHUB_WS_URL.format(api_key=self._api_key)

        ws = websocket.WebSocket()
        ws.settimeout(30)  # Ping/pong timeout
        ws.connect(url)
        self._ws = ws
        self._connected = True
        self._backoff = self.INITIAL_BACKOFF  # Reset backoff on successful connect
        logger.info("Finnhub WS connected")

        # Subscribe to symbols
        for symbol in self._symbols:
            msg = json.dumps({"type": "subscribe", "symbol": symbol})
            ws.send(msg)
            logger.info(f"Finnhub WS subscribed to {symbol}")

        # Receive loop
        while self._should_run:
            try:
                raw = ws.recv()
                if not raw:
                    continue
                self._on_message(raw)
            except websocket.WebSocketTimeoutException:
                # Send ping to keep alive
                try:
                    ws.ping()
                except Exception:
                    break
            except websocket.WebSocketConnectionClosedException:
                logger.info("Finnhub WS connection closed")
                break
            except Exception as e:
                logger.warning(f"Finnhub WS recv error: {e}")
                break

        try:
            ws.close()
        except Exception:
            pass
        self._ws = None

    def _on_message(self, raw: str):
        """Parse Finnhub trade message and update latest prices."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        if data.get('type') != 'trade':
            return

        trades = data.get('data', [])
        if not trades:
            return

        # Process all trades, keep only latest per symbol
        updates: Dict[str, PriceTick] = {}
        for trade in trades:
            symbol = trade.get('s', '')
            price = trade.get('p', 0)
            volume = trade.get('v', 0)
            ts = trade.get('t', 0) / 1000.0  # Finnhub sends ms

            if symbol and price > 0:
                updates[symbol] = PriceTick(
                    price=float(price),
                    timestamp=ts,
                    volume=float(volume),
                    symbol=symbol,
                )

        if updates:
            with self._lock:
                self._latest.update(updates)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: Optional[FinnhubWebSocket] = None
_instance_lock = threading.Lock()


def get_ws_client() -> Optional[FinnhubWebSocket]:
    """Get or create the singleton FinnhubWebSocket instance.

    Returns None if the API key is not available.
    """
    global _instance
    if _instance is not None:
        return _instance

    with _instance_lock:
        if _instance is not None:
            return _instance

        try:
            from v15.data.finnhub_client import FINNHUB_API_KEY
        except ImportError:
            logger.warning("Cannot import FINNHUB_API_KEY — WebSocket disabled")
            return None

        _instance = FinnhubWebSocket(api_key=FINNHUB_API_KEY)
        _instance.start()
        return _instance
