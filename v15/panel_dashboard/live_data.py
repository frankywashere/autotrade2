"""
LiveDataProvider — Live IB data provider implementing the same interface as
the backtester's DataProvider.

Accumulates 1-min bars from IB ticks, resamples to higher TFs, and provides
thread-safe bar access for algo evaluation.

Bar indexing: all bars are end-indexed (timestamp = bar close time).
Only completed bars are visible — no incomplete/in-progress bars.
"""

import logging
import queue
import threading
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LiveDataProvider:
    """Live data provider for algo evaluation.

    Implements the same get_bars() interface as the backtester's DataProvider
    so algo classes work identically in both contexts.

    Thread safety: all bar access goes through self._lock.
    Deadlock prevention: bar-close events are emitted AFTER releasing the lock.
    """

    def __init__(self, ib_client=None):
        self._ib = ib_client
        self._lock = threading.Lock()

        # Per-symbol bar storage: {symbol: {tf: pd.DataFrame}}
        self._bars: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Bar-close queue for LiveEngine dispatch (replaces Event + dict
        # to prevent event drops when multiple TFs close simultaneously)
        self._bar_queue: queue.Queue = queue.Queue()

        # Seed historical data at startup
        self._seed_historical()

    @property
    def is_live(self) -> bool:
        return True

    @property
    def trading_days(self) -> list:
        """List of trading days in the data."""
        with self._lock:
            df = self._bars.get('TSLA', {}).get('daily', pd.DataFrame())
            if len(df) == 0:
                return []
            return sorted(set(df.index.date))

    @property
    def start_time(self) -> pd.Timestamp:
        with self._lock:
            df = self._bars.get('TSLA', {}).get('1min', pd.DataFrame())
            if len(df) == 0:
                return pd.Timestamp.now()
            return df.index[0]

    @property
    def end_time(self) -> pd.Timestamp:
        return pd.Timestamp.now()

    def get_bars(self, tf: str, up_to: pd.Timestamp = None,
                 symbol: str = 'TSLA') -> pd.DataFrame:
        """Returns OHLCV bars up to (not beyond) given time.

        Same interface as DataProvider.get_bars(). Thread-safe.
        Only returns completed bars (end-indexed).
        """
        with self._lock:
            df = self._bars.get(symbol, {}).get(tf, pd.DataFrame())
            if len(df) == 0:
                return pd.DataFrame()
            if up_to is not None:
                # Normalize tz: strip timezone from up_to if index is naive,
                # or localize index if up_to is aware (IB bar timestamps are
                # tz-aware ET, historical seeded bars may be naive)
                idx_tz = getattr(df.index, 'tz', None)
                up_tz = getattr(up_to, 'tz', None) or getattr(up_to, 'tzinfo', None)
                if idx_tz is None and up_tz is not None:
                    up_to = up_to.tz_localize(None) if hasattr(up_to, 'tz_localize') else up_to.replace(tzinfo=None)
                elif idx_tz is not None and up_tz is None:
                    up_to = up_to.tz_localize(idx_tz)
                return df[df.index <= up_to].copy()
            return df.copy()

    def get_bars_symbol(self, symbol: str, tf: str,
                        up_to: pd.Timestamp = None) -> pd.DataFrame:
        """Multi-symbol variant (for OE-Sig5 SPY/VIX access)."""
        return self.get_bars(tf, up_to, symbol=symbol)

    def on_1min_close(self, symbol: str, bar_time: pd.Timestamp,
                      bar: dict):
        """Called when a 1-min bar closes. Appends and resamples.

        Thread-safe: acquires lock for data mutation, then emits event
        AFTER releasing lock to prevent deadlock with get_bars().
        """
        emit_events = []

        with self._lock:
            storage = self._bars.setdefault(symbol, {})

            # Append to 1-min storage
            df_1m = storage.get('1min', pd.DataFrame())
            new_row = pd.DataFrame({
                'open': [bar['open']],
                'high': [bar['high']],
                'low': [bar['low']],
                'close': [bar['close']],
                'volume': [bar.get('volume', 0)],
            }, index=[bar_time])
            storage['1min'] = pd.concat([df_1m, new_row])

            # Always emit 1min event
            emit_events.append(('1min', bar_time, bar))

            # Resample to higher TFs
            for tf, period_mins in [('5min', 5), ('15min', 15), ('30min', 30),
                                     ('1h', 60)]:
                completed = self._check_and_resample(storage, tf, period_mins,
                                                      bar_time)
                if completed is not None:
                    emit_events.append((tf, completed['time'],
                                        completed['bar']))

            # 4h: sequential hourly aggregation (matching backtester)
            completed_4h = self._check_4h_resample(storage, bar_time)
            if completed_4h is not None:
                emit_events.append(('4h', completed_4h['time'],
                                    completed_4h['bar']))

            # Daily: emit at RTH close (last bar of session)
            completed_daily = self._check_daily_resample(storage, bar_time)
            if completed_daily is not None:
                emit_events.append(('daily', completed_daily['time'],
                                    completed_daily['bar']))

                # Weekly/Monthly: resample from daily
                for tf in ('weekly', 'monthly'):
                    completed_tf = self._resample_from_daily(storage, tf)
                    if completed_tf is not None:
                        emit_events.append((tf, completed_tf['time'],
                                            completed_tf['bar']))

        # Emit events AFTER releasing lock (deadlock prevention)
        # Order: 1min first, then higher TFs ascending
        if symbol == 'TSLA':
            for tf, time, bar_data in emit_events:
                logger.info("Queue bar event: %s %s @ %s", symbol, tf, time)
                self._bar_queue.put({'tf': tf, 'time': time, 'bar': bar_data,
                                      'symbol': symbol})

    def _check_and_resample(self, storage, tf, period_mins, bar_time):
        """Check if a higher-TF bar just completed, resample if so."""
        minute = bar_time.minute
        hour = bar_time.hour
        total_min = hour * 60 + minute

        # Bar boundary: total_min is a multiple of period_mins
        if total_min % period_mins != 0:
            return None

        df_1m = storage.get('1min', pd.DataFrame())
        if len(df_1m) < 2:
            return None

        # Get the last period_mins 1-min bars
        bar_end = bar_time
        bar_start = bar_end - pd.Timedelta(minutes=period_mins)
        mask = (df_1m.index > bar_start) & (df_1m.index <= bar_end)
        period_bars = df_1m[mask]

        if len(period_bars) == 0:
            return None

        bar_dict = {
            'open': float(period_bars['open'].iloc[0]),
            'high': float(period_bars['high'].max()),
            'low': float(period_bars['low'].min()),
            'close': float(period_bars['close'].iloc[-1]),
            'volume': float(period_bars['volume'].sum()),
        }

        # Append to TF storage (end-indexed)
        tf_df = storage.get(tf, pd.DataFrame())
        new_row = pd.DataFrame({k: [v] for k, v in bar_dict.items()},
                                index=[bar_end])
        storage[tf] = pd.concat([tf_df, new_row])

        return {'time': bar_end, 'bar': bar_dict}

    def _check_4h_resample(self, storage, bar_time):
        """Sequential hourly aggregation for 4h bars (matching backtester)."""
        # 4h bars: group hours sequentially within each day
        # RTH: 9:30-16:00 → hours 10,11,12,13,14,15,16
        # 4h boundaries at: 13:00 (hours 10-13), 16:00 (hours 14-16+)
        hour = bar_time.hour
        minute = bar_time.minute

        # Only check at hour boundaries
        if minute != 0:
            return None

        # Check if this is a 4h boundary
        if hour not in (13, 16):
            return None

        df_1h = storage.get('1h', pd.DataFrame())
        if len(df_1h) < 2:
            return None

        today = bar_time.date()
        today_bars = df_1h[df_1h.index.date == today]

        if hour == 13:
            chunk = today_bars[today_bars.index.hour <= 13]
        else:  # 16
            chunk = today_bars[today_bars.index.hour > 13]

        if len(chunk) == 0:
            return None

        bar_dict = {
            'open': float(chunk['open'].iloc[0]),
            'high': float(chunk['high'].max()),
            'low': float(chunk['low'].min()),
            'close': float(chunk['close'].iloc[-1]),
            'volume': float(chunk['volume'].sum()),
        }

        tf_df = storage.get('4h', pd.DataFrame())
        new_row = pd.DataFrame({k: [v] for k, v in bar_dict.items()},
                                index=[bar_time])
        storage['4h'] = pd.concat([tf_df, new_row])

        return {'time': bar_time, 'bar': bar_dict}

    def _check_daily_resample(self, storage, bar_time):
        """Daily bar: emit at RTH close (15:59 or 16:00 bar)."""
        # RTH close: last bar at 16:00 (or 13:00 on early close days)
        hour = bar_time.hour
        minute = bar_time.minute

        if not (hour == 16 and minute == 0):
            return None

        df_1m = storage.get('1min', pd.DataFrame())
        if len(df_1m) == 0:
            return None

        today = bar_time.date()
        today_bars = df_1m[df_1m.index.date == today]
        # Filter to RTH only (9:30-16:00)
        rth = today_bars[(today_bars.index.hour > 9) |
                          ((today_bars.index.hour == 9) &
                           (today_bars.index.minute >= 30))]
        rth = rth[rth.index.hour <= 16]

        if len(rth) == 0:
            return None

        bar_dict = {
            'open': float(rth['open'].iloc[0]),
            'high': float(rth['high'].max()),
            'low': float(rth['low'].min()),
            'close': float(rth['close'].iloc[-1]),
            'volume': float(rth['volume'].sum()),
        }

        tf_df = storage.get('daily', pd.DataFrame())
        new_row = pd.DataFrame({k: [v] for k, v in bar_dict.items()},
                                index=[bar_time])
        storage['daily'] = pd.concat([tf_df, new_row])

        return {'time': bar_time, 'bar': bar_dict}

    def _resample_from_daily(self, storage, tf):
        """Resample weekly/monthly from daily bars."""
        daily = storage.get('daily', pd.DataFrame())
        if len(daily) < 2:
            return None

        if tf == 'weekly':
            rule = 'W-FRI'
        elif tf == 'monthly':
            rule = 'ME'
        else:
            return None

        resampled = daily.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()

        if len(resampled) == 0:
            return None

        storage[tf] = resampled

        # Return the last bar if it's new
        last_bar = resampled.iloc[-1]
        return {
            'time': resampled.index[-1],
            'bar': {
                'open': float(last_bar['open']),
                'high': float(last_bar['high']),
                'low': float(last_bar['low']),
                'close': float(last_bar['close']),
                'volume': float(last_bar['volume']),
            },
        }

    def _seed_historical(self):
        """Seed with IB historical bars at startup.

        Provides lookback for channel/feature computation:
        - TSLA: 15 days of 1-min bars (RTH only, for 5min/1h/4h resample — 30 4h bars)
        - TSLA/SPY/VIX: 500 daily bars (for channels + OE-Sig5)
        - TSLA/SPY: 104 weekly bars (for weekly channels)
        - VIX: daily only (no algo uses VIX weekly)
        - TSLA monthly: derived from daily resample
        """
        if not self._ib or not self._ib.is_connected():
            logger.warning("IB not connected, skipping historical seeding")
            return

        try:
            self._seed_1min_bars('TSLA', days=15)
        except Exception as e:
            logger.error("Failed to seed 1-min bars: %s", e)

        for symbol in ('TSLA', 'SPY', 'VIX'):
            try:
                self._seed_daily_bars(symbol, bars=500)
            except Exception as e:
                logger.error("Failed to seed %s daily bars: %s", symbol, e)
            # VIX weekly not used by any algo — skip to save IB pacing budget
            if symbol != 'VIX':
                try:
                    self._seed_weekly_bars(symbol)
                except Exception as e:
                    logger.error("Failed to seed %s weekly bars: %s", symbol, e)

        # Monthly: resample from daily (IB monthly bars limited anyway)
        with self._lock:
            storage = self._bars.get('TSLA', {})
            if 'daily' in storage:
                self._resample_from_daily(storage, 'monthly')

        # Validate seeded data meets algo minimum requirements
        self._validate_seeded_bars()

        logger.info("Historical seeding complete: %s",
                     {s: list(tfs.keys()) for s, tfs in self._bars.items()})

    def _validate_seeded_bars(self):
        """Check seeded bar counts against algo minimum requirements."""
        # Minimum bars needed per symbol/TF across all algos:
        # CS-Combo: daily=60, weekly=50, monthly=24, 5min=78, 1h=60, 4h=30
        # OE-Sig5: daily=36 (TSLA/SPY/VIX), weekly=51 (TSLA)
        # Surfer-ML: daily=20 (SPY/VIX for correlation)
        # Intraday: daily=40, 1h=24, 4h=20
        minimums = {
            'TSLA': {'1min': 390, '5min': 78, '1h': 60, '4h': 30,
                     'daily': 60, 'weekly': 51, 'monthly': 24},
            'SPY':  {'daily': 36},
            'VIX':  {'daily': 36},
        }
        with self._lock:
            for symbol, tf_mins in minimums.items():
                storage = self._bars.get(symbol, {})
                for tf, min_bars in tf_mins.items():
                    actual = len(storage.get(tf, pd.DataFrame()))
                    if actual < min_bars:
                        logger.warning(
                            "INSUFFICIENT DATA: %s %s has %d bars, need %d",
                            symbol, tf, actual, min_bars)

    def _seed_1min_bars(self, symbol, days=5):
        """Seed 1-min bars from IB historical API."""
        bars_df = self._ib.fetch_historical(symbol, f'{days} D', '1 min',
                                             use_rth=True)
        if bars_df is None or len(bars_df) == 0:
            logger.warning("No 1-min bars returned for %s", symbol)
            return
        bars_df['date'] = pd.to_datetime(bars_df['date'])
        bars_df = bars_df.set_index('date')
        with self._lock:
            self._bars.setdefault(symbol, {})['1min'] = bars_df
            for tf, rule in [('5min', '5min'), ('15min', '15min'),
                              ('1h', '1h')]:
                resampled = bars_df.resample(rule).agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum',
                }).dropna()
                self._bars[symbol][tf] = resampled
            # Derive 4h bars from 1h using same logic as _check_4h_resample
            df_1h = self._bars[symbol].get('1h', pd.DataFrame())
            if len(df_1h) > 0:
                bars_4h = []
                for day in sorted(set(df_1h.index.date)):
                    day_bars = df_1h[df_1h.index.date == day]
                    # First 4h: hours <= 13
                    chunk1 = day_bars[day_bars.index.hour <= 13]
                    if len(chunk1) > 0:
                        bars_4h.append((chunk1.index[-1], {
                            'open': float(chunk1['open'].iloc[0]),
                            'high': float(chunk1['high'].max()),
                            'low': float(chunk1['low'].min()),
                            'close': float(chunk1['close'].iloc[-1]),
                            'volume': float(chunk1['volume'].sum()),
                        }))
                    # Second 4h: hours > 13
                    chunk2 = day_bars[day_bars.index.hour > 13]
                    if len(chunk2) > 0:
                        bars_4h.append((chunk2.index[-1], {
                            'open': float(chunk2['open'].iloc[0]),
                            'high': float(chunk2['high'].max()),
                            'low': float(chunk2['low'].min()),
                            'close': float(chunk2['close'].iloc[-1]),
                            'volume': float(chunk2['volume'].sum()),
                        }))
                if bars_4h:
                    self._bars[symbol]['4h'] = pd.DataFrame(
                        [b for _, b in bars_4h],
                        index=[t for t, _ in bars_4h],
                    )
        logger.info("Seeded %s 1-min bars: %d", symbol, len(bars_df))

    def _seed_daily_bars(self, symbol, bars=500):
        """Seed daily bars from IB historical API."""
        daily_df = self._ib.fetch_historical(symbol, '2 Y', '1 day',
                                              use_rth=True)
        if daily_df is None or len(daily_df) == 0:
            logger.warning("No daily bars returned for %s", symbol)
            return
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.set_index('date')
        with self._lock:
            self._bars.setdefault(symbol, {})['daily'] = daily_df.tail(bars)
        logger.info("Seeded %s daily bars: %d", symbol,
                     min(len(daily_df), bars))

    def _seed_weekly_bars(self, symbol):
        """Seed weekly bars from IB historical API (native, not resampled)."""
        weekly_df = self._ib.fetch_historical(symbol, '2 Y', '1 W',
                                               use_rth=True)
        if weekly_df is None or len(weekly_df) == 0:
            logger.warning("No weekly bars returned for %s", symbol)
            return
        weekly_df['date'] = pd.to_datetime(weekly_df['date'])
        weekly_df = weekly_df.set_index('date')
        with self._lock:
            self._bars.setdefault(symbol, {})['weekly'] = weekly_df
        logger.info("Seeded %s weekly bars: %d", symbol, len(weekly_df))

    def backfill_gap(self, symbol: str, since: pd.Timestamp):
        """Backfill bars after IB disconnect/reconnect."""
        if not self._ib or not self._ib.is_connected():
            return
        try:
            now = pd.Timestamp.now()
            gap_mins = int((now - since).total_seconds() / 60)
            if gap_mins < 1:
                return
            duration_secs = gap_mins * 60
            bars_df = self._ib.fetch_historical(
                symbol, f'{duration_secs} S', '1 min', use_rth=False)
            if bars_df is not None and len(bars_df) > 0:
                bars_df['date'] = pd.to_datetime(bars_df['date'])
                bars_df = bars_df.set_index('date')
                with self._lock:
                    existing = self._bars.get(symbol, {}).get('1min',
                                                               pd.DataFrame())
                    combined = pd.concat([existing, bars_df])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    self._bars.setdefault(symbol, {})['1min'] = combined.sort_index()
                logger.info("Backfilled %d 1-min bars for %s", len(bars_df),
                             symbol)
        except Exception as e:
            logger.error("Backfill failed for %s: %s", symbol, e)
