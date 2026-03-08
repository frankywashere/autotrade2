"""
DataProvider — Loads 1-min data, resamples to any TF, enforces no-lookahead.

The ONLY way algorithms access market data. All access is gated by a time
parameter that prevents any algorithm from seeing future bars.
"""

import datetime as dt
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd


# RTH hours
MKT_OPEN = dt.time(9, 30)
MKT_CLOSE = dt.time(16, 0)

# Extended hours
EXT_OPEN = dt.time(4, 0)
EXT_CLOSE = dt.time(20, 0)

# Standard resampling rules
_RESAMPLE_RULES = {
    '1min': None,  # raw data
    '5min': '5min',
    '15min': '15min',
    '30min': '30min',
    '1h': '1h',
    'daily': '1D',
    'weekly': 'W-FRI',
    'monthly': 'ME',
}

# TFs that need sequential in-day aggregation from 1h (matching native_tf.py)
_HOURLY_AGGREGATE_TFS = {'2h': 2, '3h': 3, '4h': 4}
_START_INDEXED_INTRADAY_TFS = {'5min', '15min', '30min', '1h'}


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Standard OHLCV resampling."""
    return df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open'])


def _aggregate_from_hourly(hourly_df: pd.DataFrame, target_hours: int) -> pd.DataFrame:
    """Sequential in-day aggregation from 1h bars (matching native_tf.py).

    Groups by trading day, then chunks hours sequentially:
    chunk[0:N], chunk[N:2N], chunk[2N:3N], etc.
    """
    rows = []
    for day, group in hourly_df.groupby(hourly_df.index.date):
        n = len(group)
        for start in range(0, n, target_hours):
            end = min(start + target_hours, n)
            chunk = group.iloc[start:end]
            rows.append({
                'datetime': chunk.index[-1],  # Use last bar's timestamp
                'open': chunk['open'].iloc[0],
                'high': chunk['high'].max(),
                'low': chunk['low'].min(),
                'close': chunk['close'].iloc[-1],
                'volume': chunk['volume'].sum(),
            })
    if not rows:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    result = pd.DataFrame(rows).set_index('datetime')
    return result


def load_1min(path: str, start: str = None, end: str = None,
              rth_only: bool = True) -> pd.DataFrame:
    """Load 1-min OHLCV data from semicolon-delimited file.

    Format: YYYYMMDD HHMMSS;open;high;low;close;volume
    """
    df = pd.read_csv(
        path, sep=';',
        names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
        parse_dates=['datetime'], date_format='%Y%m%d %H%M%S',
    )
    df = df.set_index('datetime').sort_index()

    # Date range filter
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    # RTH filter
    if rth_only:
        times = df.index.time
        df = df[(times >= MKT_OPEN) & (times < MKT_CLOSE)]
    else:
        times = df.index.time
        df = df[(times >= EXT_OPEN) & (times < EXT_CLOSE)]

    return df


class DataProvider:
    """Provides market data to algorithms with no-lookahead guarantee.

    All data access is gated by a `up_to` timestamp. No bar with a close
    time after `up_to` will ever be returned.
    """

    def __init__(self, tsla_1min_path: str, start: str, end: str,
                 spy_path: str = None, rth_only: bool = True):
        """Load raw 1-min data and pre-compute all resampled TFs."""
        self._rth_only = rth_only

        # Load TSLA 1-min
        self._df1m = load_1min(tsla_1min_path, start, end, rth_only)
        if len(self._df1m) == 0:
            raise ValueError(f"No data loaded from {tsla_1min_path} for {start} to {end}")

        self._init_from_df1m(spy_path, start, end, rth_only)

    @classmethod
    def from_ticks(cls, tick_dir: str, symbol: str, start: str, end: str,
                   spy_path: str = None, rth_only: bool = True) -> 'DataProvider':
        """Build DataProvider with 1-min bars aggregated from tick data."""
        from .tick_provider import TickDataProvider

        tick_prov = TickDataProvider(tick_dir, symbol, start, end, rth_only)
        df1m = tick_prov.aggregate_to_1min()

        instance = cls.__new__(cls)
        instance._rth_only = rth_only
        instance._df1m = df1m
        instance._tick_count = tick_prov.tick_count
        instance._init_from_df1m(spy_path, start, end, rth_only)
        return instance

    def _init_from_df1m(self, spy_path: str = None, start: str = None,
                        end: str = None, rth_only: bool = True):
        """Shared init: resample all TFs, build _tf_bar_end, load SPY/VIX."""
        # Load SPY 1-min if provided (for intraday SPY features, optional)
        self._spy1m = None
        if spy_path and Path(spy_path).exists():
            self._spy1m = load_1min(spy_path, start, end, rth_only)

        # Pre-compute all resampled TFs
        self._tf_data: Dict[str, pd.DataFrame] = {'1min': self._df1m}
        for tf, rule in _RESAMPLE_RULES.items():
            if rule is not None and tf not in self._tf_data:
                self._tf_data[tf] = _resample_ohlcv(self._df1m, rule)

        # Hourly aggregates (2h, 3h, 4h)
        hourly = self._tf_data.get('1h')
        if hourly is not None:
            for tf, hours in _HOURLY_AGGREGATE_TFS.items():
                self._tf_data[tf] = _aggregate_from_hourly(hourly, hours)

        # SPY resampled TFs (from 1-min if available)
        self._spy_tf_data: Dict[str, pd.DataFrame] = {}
        if self._spy1m is not None:
            self._spy_tf_data['1min'] = self._spy1m
            for tf, rule in _RESAMPLE_RULES.items():
                if rule is not None:
                    self._spy_tf_data[tf] = _resample_ohlcv(self._spy1m, rule)

        # Auxiliary daily data: SPY + VIX from native_tf (yfinance cache).
        # This is the authoritative daily source — always loaded, overrides
        # SPY daily from 1-min resampling (which may end earlier).
        self._aux_daily: Dict[str, pd.DataFrame] = {}  # {symbol: daily DataFrame}
        self._load_aux_daily(start, end)

        # Build daily completion timestamps from TSLA 1-min bars.
        # A daily bar is only "complete" after the last 1-min bar of that day.
        # This prevents lookahead: at 10:30 AM you can't see today's daily bar.
        self._daily_complete_ts: Dict = {}  # {date -> last_1min_timestamp}
        for day, group in self._df1m.groupby(self._df1m.index.date):
            self._daily_complete_ts[day] = group.index[-1]

        # Precompute bar timestamps per TF for fast lookup
        self._tf_times: Dict[str, np.ndarray] = {}
        for tf, df in self._tf_data.items():
            self._tf_times[tf] = df.index.values

        # Precompute bar-end (completion) timestamps for start-indexed intraday TFs
        # A 5-min bar starting at 09:30 completes at 09:34 (last 1-min bar before 09:35)
        self._tf_bar_end: Dict[str, np.ndarray] = {}
        idx_1m = self._df1m.index
        for tf in _START_INDEXED_INTRADAY_TFS:
            if tf not in self._tf_data:
                continue
            df = self._tf_data[tf]
            ends = np.empty(len(df), dtype='datetime64[ns]')
            for i in range(len(df)):
                if i + 1 < len(df):
                    next_start = df.index[i + 1]
                    end_pos = idx_1m.searchsorted(next_start, side='left') - 1
                    ends[i] = idx_1m[end_pos] if end_pos >= 0 else df.index[i]
                else:
                    day_mask = idx_1m.date == df.index[i].date()
                    if day_mask.any():
                        ends[i] = idx_1m[np.flatnonzero(day_mask)[-1]]
                    else:
                        ends[i] = df.index[i]
            self._tf_bar_end[tf] = ends

    def _load_aux_daily(self, start: str = None, end: str = None):
        """Load SPY and VIX daily bars from native_tf (yfinance cache).

        These are the authoritative daily sources for ML features and OE-Sig5.
        Always loaded — no CLI flag needed. Overrides SPY daily from 1-min
        resampling since SPYMin.txt may end months before TSLA data.
        """
        try:
            from v15.data.native_tf import fetch_native_tf
        except ImportError:
            return

        # Use a wide start for multi-year lookback (channels, RSI, etc.)
        native_start = '2015-01-01'
        native_end = end or str(self._df1m.index[-1].date())

        for symbol, yf_symbol in [('SPY', 'SPY'), ('VIX', '^VIX')]:
            try:
                df = fetch_native_tf(yf_symbol, 'daily', native_start, native_end)
                df.columns = [c.lower() for c in df.columns]
                # Strip timezone if present (match TSLA resampled daily which is tz-naive)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                self._aux_daily[symbol] = df
            except Exception:
                pass  # Graceful degradation — algo will get None

        # Override SPY daily in _spy_tf_data so get_bars('daily', t, 'SPY')
        # uses the full-range native data instead of truncated 1-min resample
        if 'SPY' in self._aux_daily:
            self._spy_tf_data['daily'] = self._aux_daily['SPY']

    @property
    def is_live(self) -> bool:
        return False

    @property
    def start_time(self) -> pd.Timestamp:
        return self._df1m.index[0]

    @property
    def end_time(self) -> pd.Timestamp:
        return self._df1m.index[-1]

    @property
    def trading_days(self) -> list:
        """List of unique trading days in the data."""
        return sorted(set(self._df1m.index.date))

    def get_bars(self, tf: str, up_to: 'pd.Timestamp | dt.datetime',
                 symbol: str = 'TSLA') -> pd.DataFrame:
        """Return OHLCV bars for timeframe, up to and including given time.

        This is the primary data access method. The `up_to` parameter
        enforces no-lookahead — bars after this time are never returned.

        For daily bars, completion gating is enforced: a daily bar is only
        visible after the last 1-min bar of that trading day (matching live).

        Args:
            tf: Timeframe string ('1min', '5min', '1h', '4h', 'daily', etc.)
            up_to: Maximum timestamp (inclusive)
            symbol: 'TSLA', 'SPY', or 'VIX'

        Returns:
            DataFrame with columns [open, high, low, close, volume]
        """
        up_to_ts = pd.Timestamp(up_to)

        # VIX: daily-only from auxiliary store
        if symbol == 'VIX':
            if 'VIX' not in self._aux_daily:
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            df = self._aux_daily['VIX']
            return self._gate_daily(df, up_to_ts)

        # SPY: route to spy_tf_data (daily overridden by aux_daily in init)
        if symbol == 'SPY':
            source = self._spy_tf_data
            if tf == 'daily' and 'SPY' in self._aux_daily:
                # Use authoritative native daily (full date range)
                df = self._aux_daily['SPY']
                return self._gate_daily(df, up_to_ts)
            if tf not in source:
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            return source[tf][source[tf].index <= up_to_ts]

        # TSLA
        source = self._tf_data
        if tf not in source:
            raise ValueError(f"Timeframe '{tf}' not available for {symbol}. "
                             f"Available: {list(source.keys())}")

        df = source[tf]

        # For start-indexed intraday TFs, only return bars that have completed
        # (a 5-min bar starting at 09:30 is only available after 09:34)
        if tf in self._tf_bar_end and symbol == 'TSLA':
            ends = self._tf_bar_end[tf]
            mask = ends <= np.datetime64(up_to_ts)
            return df[mask]

        # Daily completion gating for TSLA too
        if tf == 'daily':
            return self._gate_daily(df, up_to_ts)

        return df[df.index <= up_to_ts]

    def _gate_daily(self, df: pd.DataFrame, up_to_ts: pd.Timestamp) -> pd.DataFrame:
        """Return only completed daily bars — no lookahead into today's bar.

        A daily bar is only visible after the last 1-min bar of that trading
        day has been processed. This matches live behavior where the daily bar
        materializes at RTH close.
        """
        completed_dates = set()
        for day, complete_ts in self._daily_complete_ts.items():
            if complete_ts <= up_to_ts:
                completed_dates.add(day)

        # Filter: only return bars whose date is in completed set
        mask = pd.Series(df.index.date, index=df.index).isin(completed_dates)
        return df[mask.values]

    def get_current_bar(self, tf: str, bar_time: 'pd.Timestamp | dt.datetime',
                        symbol: str = 'TSLA') -> Optional[dict]:
        """Return the single bar for the requested TF at the given time.

        For daily/weekly/monthly TFs where bar timestamps are at midnight,
        looks up by date instead of exact timestamp.

        For left-labeled intraday TFs (5min/15min/30min/1h), also accepts
        the completed bar's last 1-minute timestamp and resolves it back to
        the matching TF row.

        Returns None if no bar exists at that time.
        """
        # VIX: daily-only from auxiliary store
        if symbol == 'VIX':
            if 'VIX' not in self._aux_daily or tf != 'daily':
                return None
            df = self._aux_daily['VIX']
            bt = pd.Timestamp(bar_time)
            target_date = bt.date()
            # Completion gate: only return if that day is complete
            if target_date not in self._daily_complete_ts:
                return None
            if self._daily_complete_ts[target_date] > bt:
                return None
            date_matches = df[df.index.date == target_date]
            if len(date_matches) == 0:
                return None
            row = date_matches.iloc[0]
            return {
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'time': date_matches.index[0],
            }

        source = self._spy_tf_data if symbol == 'SPY' else self._tf_data
        if tf not in source:
            return None
        df = source[tf]
        bt = pd.Timestamp(bar_time)

        # Exact match first (works for intraday TFs)
        if bt in df.index:
            row = df.loc[bt]
            return {
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'time': bt,
            }

        # For daily+ TFs, try matching by date
        if tf in ('daily', 'weekly', 'monthly'):
            target_date = bt.date()
            date_matches = df[df.index.date == target_date]
            if len(date_matches) > 0:
                row = date_matches.iloc[0]
                return {
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'time': date_matches.index[0],
                }

        if tf in _START_INDEXED_INTRADAY_TFS and len(df) > 0:
            pos = df.index.searchsorted(bt, side='right') - 1
            if pos >= 0:
                resolved_ts = df.index[pos]

                if pos + 1 < len(df.index):
                    next_tf_ts = df.index[pos + 1]
                    end_pos = self._df1m.index.searchsorted(next_tf_ts, side='left') - 1
                else:
                    day_mask = self._df1m.index.date == resolved_ts.date()
                    end_pos = np.flatnonzero(day_mask)[-1] if day_mask.any() else -1

                if end_pos >= 0:
                    completed_at = self._df1m.index[end_pos]
                    if bt >= completed_at:
                        row = df.iloc[pos]
                        return {
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume']),
                            'time': resolved_ts,
                        }

        return None

    def get_price_at(self, time: 'pd.Timestamp | dt.datetime') -> float:
        """Return TSLA close price at or before the given time."""
        ts = pd.Timestamp(time)
        mask = self._df1m.index <= ts
        if not mask.any():
            return 0.0
        return float(self._df1m.loc[mask, 'close'].iloc[-1])

    def iter_bars(self, primary_tf: str) -> Iterator[Tuple[pd.Timestamp, dict]]:
        """Yield (timestamp, bar_dict) for each completed bar in the primary TF.

        This is what the engine uses to walk forward through time.
        """
        df = self._tf_data.get(primary_tf)
        if df is None:
            raise ValueError(f"Timeframe '{primary_tf}' not available")

        for ts, row in df.iterrows():
            yield ts, {
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'time': ts,
            }

    def get_1min_bars_for_day(self, day: dt.date) -> pd.DataFrame:
        """Return all 1-min bars for a specific trading day."""
        mask = self._df1m.index.date == day
        return self._df1m[mask]

    def get_1min_bars_between(self, start: 'pd.Timestamp', end: 'pd.Timestamp') -> pd.DataFrame:
        """Return 1-min bars between start and end (inclusive)."""
        return self._df1m[(self._df1m.index >= start) & (self._df1m.index <= end)]

    def build_native_data_dict(self, up_to: 'pd.Timestamp | dt.datetime',
                               tfs: list = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Build native_data dict for channel analysis: {symbol: {tf: DataFrame}}.

        This is the format expected by prepare_multi_tf_analysis().
        """
        if tfs is None:
            tfs = ['5min', '1h', '4h', 'daily', 'weekly']

        result = {'TSLA': {}}
        for tf in tfs:
            bars = self.get_bars(tf, up_to)
            if len(bars) > 0:
                result['TSLA'][tf] = bars

        return result
