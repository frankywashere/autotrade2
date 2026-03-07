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

        # Load SPY if provided
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

        # SPY resampled TFs
        self._spy_tf_data: Dict[str, pd.DataFrame] = {}
        if self._spy1m is not None:
            self._spy_tf_data['1min'] = self._spy1m
            for tf, rule in _RESAMPLE_RULES.items():
                if rule is not None:
                    self._spy_tf_data[tf] = _resample_ohlcv(self._spy1m, rule)

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

        Args:
            tf: Timeframe string ('1min', '5min', '1h', '4h', 'daily', etc.)
            up_to: Maximum timestamp (inclusive)
            symbol: 'TSLA' or 'SPY'

        Returns:
            DataFrame with columns [open, high, low, close, volume]
        """
        source = self._spy_tf_data if symbol == 'SPY' else self._tf_data
        if tf not in source:
            raise ValueError(f"Timeframe '{tf}' not available for {symbol}. "
                             f"Available: {list(source.keys())}")

        df = source[tf]
        up_to_ts = pd.Timestamp(up_to)

        # For start-indexed intraday TFs, only return bars that have completed
        # (a 5-min bar starting at 09:30 is only available after 09:34)
        if tf in self._tf_bar_end and symbol == 'TSLA':
            ends = self._tf_bar_end[tf]
            mask = ends <= np.datetime64(up_to_ts)
            return df[mask]

        return df[df.index <= up_to_ts]

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
