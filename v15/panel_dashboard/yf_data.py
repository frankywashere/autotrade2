"""
YfinanceDataProvider — yfinance-based data provider for A/B comparison.

Same interface as LiveDataProvider. Seeds daily/weekly/monthly from yf.download()
at startup. Polls yfinance lastPrice to construct synthetic 5-min bars for exit
checking. Fires daily bar-close at market close for signal generation.

Bar indexing: all bars are end-indexed (timestamp = bar close time).
"""

import logging
import queue
import threading
import time as _time
from datetime import datetime, time, timedelta
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _is_market_open() -> bool:
    """Check if US market is currently open (9:30-16:00 ET, weekdays)."""
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('US/Eastern'))
    if now.weekday() >= 5:  # Weekend
        return False
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= now.time() <= market_close


def _now_et() -> datetime:
    """Current time in US/Eastern."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo('US/Eastern'))


class YfinanceDataProvider:
    """yfinance data provider — same interface as LiveDataProvider.

    Seeds daily/weekly/monthly from yf.download() at startup.
    Polls yfinance lastPrice to construct synthetic 5-min bars for exit checking.
    Fires daily bar-close at market close for signal generation.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Per-symbol bar storage: {symbol: {tf: pd.DataFrame}}
        self._bars: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Bar-close queue for LiveEngine dispatch
        self._bar_queue: queue.Queue = queue.Queue()

        # Synthetic 5-min bar accumulator: {symbol: {open, high, low, close, start_time}}
        self._current_5min: Dict[str, dict] = {}

        # Track whether daily bar has been emitted today
        self._last_daily_date: dict = {}  # {symbol: date}

        # Seed historical data
        self._seed_historical()

    @property
    def is_live(self) -> bool:
        return True

    @property
    def trading_days(self) -> list:
        with self._lock:
            df = self._bars.get('TSLA', {}).get('daily', pd.DataFrame())
            if len(df) == 0:
                return []
            return sorted(set(df.index.date))

    @property
    def start_time(self) -> pd.Timestamp:
        with self._lock:
            df = self._bars.get('TSLA', {}).get('daily', pd.DataFrame())
            if len(df) == 0:
                return pd.Timestamp.now()
            return df.index[0]

    @property
    def end_time(self) -> pd.Timestamp:
        return pd.Timestamp.now()

    def get_bars(self, tf: str, up_to: pd.Timestamp = None,
                 symbol: str = 'TSLA') -> pd.DataFrame:
        """Returns OHLCV bars up to (not beyond) given time. Thread-safe."""
        with self._lock:
            df = self._bars.get(symbol, {}).get(tf, pd.DataFrame())
            if len(df) == 0:
                return pd.DataFrame()
            if up_to is not None:
                return df[df.index <= up_to].copy()
            return df.copy()

    def get_bars_symbol(self, symbol: str, tf: str,
                        up_to: pd.Timestamp = None) -> pd.DataFrame:
        """Multi-symbol variant (for OE-Sig5 SPY/VIX access)."""
        return self.get_bars(tf, up_to, symbol=symbol)

    def on_price_update(self, symbol: str, price: float):
        """Called from yf_price_loop with latest price.

        Accumulates into synthetic 5-min bars and checks for daily close.
        """
        if not _is_market_open():
            # At market close, check if we need to emit today's daily bar
            self._check_daily_close(symbol)
            return

        now = _now_et()

        # Determine current 5-min boundary (end-indexed)
        minute = now.minute
        boundary_min = (minute // 5 + 1) * 5
        if boundary_min >= 60:
            bar_end = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            bar_end = now.replace(minute=boundary_min, second=0, microsecond=0)

        bar_end_ts = pd.Timestamp(bar_end)

        current = self._current_5min.get(symbol)

        if current is None or current['end_time'] != bar_end_ts:
            # New 5-min window — finalize previous bar if exists
            if current is not None:
                self._finalize_5min_bar(symbol, current)

            # Start new accumulator
            self._current_5min[symbol] = {
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'end_time': bar_end_ts,
            }
        else:
            # Update current bar
            current['high'] = max(current['high'], price)
            current['low'] = min(current['low'], price)
            current['close'] = price

    def _finalize_5min_bar(self, symbol: str, bar_acc: dict):
        """Finalize a synthetic 5-min bar, append to storage, emit event."""
        bar_dict = {
            'open': bar_acc['open'],
            'high': bar_acc['high'],
            'low': bar_acc['low'],
            'close': bar_acc['close'],
            'volume': 0,  # No volume from price polling
        }
        bar_time = bar_acc['end_time']

        with self._lock:
            storage = self._bars.setdefault(symbol, {})
            tf_df = storage.get('5min', pd.DataFrame())
            new_row = pd.DataFrame(
                {k: [v] for k, v in bar_dict.items()},
                index=[bar_time])
            storage['5min'] = pd.concat([tf_df, new_row])

        # Emit bar event for LiveEngine (TSLA only — algos evaluate TSLA)
        if symbol == 'TSLA':
            self._bar_queue.put({
                'tf': '5min', 'time': bar_time,
                'bar': bar_dict, 'symbol': symbol,
            })
            logger.debug("yf 5-min bar close: %s %s @ %.2f",
                         symbol, bar_time, bar_dict['close'])

    def _check_daily_close(self, symbol: str):
        """At market close, fetch official daily bar and emit event."""
        now = _now_et()
        today = now.date()

        # Only emit once per day, and only after 16:00
        if now.time() < time(16, 1):
            return
        if self._last_daily_date.get(symbol) == today:
            return

        self._last_daily_date[symbol] = today

        # Flush any remaining 5-min bar
        current = self._current_5min.pop(symbol, None)
        if current is not None:
            self._finalize_5min_bar(symbol, current)

        # Fetch today's official daily bar from yfinance
        try:
            import yfinance as yf
            yf_sym = '^VIX' if symbol == 'VIX' else symbol
            df = yf.download(yf_sym, period='5d', progress=False)
            if df is not None and len(df) > 0:
                # Normalize columns (yfinance sometimes returns MultiIndex)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]

                last_bar = df.iloc[-1]
                bar_dict = {
                    'open': float(last_bar['open']),
                    'high': float(last_bar['high']),
                    'low': float(last_bar['low']),
                    'close': float(last_bar['close']),
                    'volume': float(last_bar.get('volume', 0)),
                }
                bar_time = pd.Timestamp(df.index[-1])

                with self._lock:
                    storage = self._bars.setdefault(symbol, {})
                    daily_df = storage.get('daily', pd.DataFrame())
                    # Don't duplicate if already present
                    if len(daily_df) == 0 or daily_df.index[-1] < bar_time:
                        new_row = pd.DataFrame(
                            {k: [v] for k, v in bar_dict.items()},
                            index=[bar_time])
                        storage['daily'] = pd.concat([daily_df, new_row])

                        # Resample weekly/monthly from daily
                        self._resample_higher_tfs(storage)

                # Emit daily bar event for signal generation
                if symbol == 'TSLA':
                    self._bar_queue.put({
                        'tf': 'daily', 'time': bar_time,
                        'bar': bar_dict, 'symbol': symbol,
                    })
                    logger.info("yf daily bar close: %s @ %.2f", symbol,
                                bar_dict['close'])
        except Exception as e:
            logger.error("yf daily bar fetch failed for %s: %s", symbol, e)

    def _resample_higher_tfs(self, storage: dict):
        """Resample weekly/monthly from daily bars (called under lock)."""
        daily = storage.get('daily', pd.DataFrame())
        if len(daily) < 5:
            return
        for tf, rule in [('weekly', 'W-FRI'), ('monthly', 'ME')]:
            try:
                resampled = daily.resample(rule).agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum',
                }).dropna()
                if len(resampled) > 0:
                    storage[tf] = resampled
            except Exception as e:
                logger.warning("yf resample %s failed: %s", tf, e)

    def _seed_historical(self):
        """Seed daily/weekly/monthly bars from yf.download() at startup."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed — YfinanceDataProvider disabled")
            return

        for symbol, yf_sym in [('TSLA', 'TSLA'), ('SPY', 'SPY'), ('VIX', '^VIX')]:
            try:
                df = yf.download(yf_sym, period='2y', progress=False)
                if df is None or len(df) == 0:
                    logger.warning("yf seed: no data for %s", symbol)
                    continue

                # Normalize columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]

                # Keep last 500 daily bars
                daily = df.tail(500)

                with self._lock:
                    storage = self._bars.setdefault(symbol, {})
                    storage['daily'] = daily
                    self._resample_higher_tfs(storage)

                logger.info("Seeded %s daily: %d bars, weekly: %d, monthly: %d",
                            symbol, len(daily),
                            len(self._bars[symbol].get('weekly', [])),
                            len(self._bars[symbol].get('monthly', [])))
            except Exception as e:
                logger.error("yf seed failed for %s: %s", symbol, e)
