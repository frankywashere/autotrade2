"""TickDataProvider — Loads tick-level Parquet files and aggregates to 1-min bars.

Produces a df1m identical in format to load_1min() output, so DataProvider
can use it interchangeably via _init_from_df1m().
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Required Parquet columns — files missing any are REJECTED
REQUIRED_COLUMNS = {'time', 'price', 'size', 'past_limit', 'unreported', 'seq'}

# Session boundaries (ET, tz-naive)
_SESSION_START = '04:00'
_SESSION_END_NORMAL = '19:59'
_SESSION_END_EARLY = '16:59'

# Early close dates (month, day) — matching tick_downloader.py
_EARLY_CLOSE_MONTH_DAYS = {(7, 3), (11, 28), (11, 29), (12, 24)}

# Known US holidays (matching tick_downloader.py)
_US_HOLIDAYS = set()
for _y in range(2024, 2028):
    _US_HOLIDAYS.update([
        date(_y, 1, 1), date(_y, 7, 4), date(_y, 12, 25),
    ])
_US_HOLIDAYS.update([
    date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 9, 1),
    date(2025, 11, 27),
    date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 9, 7),
    date(2026, 11, 26),
])


def _is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in _US_HOLIDAYS


def _is_early_close(d: date) -> bool:
    return (d.month, d.day) in _EARLY_CLOSE_MONTH_DAYS


def _session_end_time(d: date) -> str:
    """Return session end time string for a given date."""
    return _SESSION_END_EARLY if _is_early_close(d) else _SESSION_END_NORMAL


class TickDataProvider:
    """Loads tick-level Parquet files and aggregates to 1-min bars."""

    def __init__(self, tick_dir: str, symbol: str, start: str, end: str,
                 rth_only: bool = True):
        self._tick_dir = Path(tick_dir)
        self._symbol = symbol
        self._start = pd.Timestamp(start).date()
        self._end = pd.Timestamp(end).date()
        self._rth_only = rth_only

        self._ticks: Optional[pd.DataFrame] = None
        self._tick_count = 0
        self._trading_days: List[date] = []

        self._load_and_validate()

    def _load_and_validate(self):
        """Load all per-day Parquet files in date range, validate, concatenate."""
        all_dfs = []
        expected_days = []
        missing_days = []

        d = self._start
        while d <= self._end:
            if _is_trading_day(d):
                expected_days.append(d)
            d += timedelta(days=1)

        for day in expected_days:
            path = self._tick_dir / f'{day.isoformat()}.parquet'
            if not path.exists():
                missing_days.append(day)
                continue

            try:
                df = pd.read_parquet(path)
            except Exception as e:
                raise ValueError(f"Corrupt Parquet file for {day}: {e}") from e

            # Schema check
            missing_cols = REQUIRED_COLUMNS - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"{day}: Missing required columns {missing_cols}. "
                    f"Re-download this file.")

            # Date ownership
            tick_dates = df['time'].dt.date
            wrong = tick_dates != day
            if wrong.any():
                raise ValueError(
                    f"{day}: {wrong.sum()} ticks belong to wrong date")

            # Seq monotonicity
            if not df['seq'].is_monotonic_increasing:
                raise ValueError(f"{day}: seq not monotonically increasing")

            # Session coverage — last tick must be in final minute
            session_end = pd.Timestamp(f'{day.isoformat()} {_session_end_time(day)}')
            if df['time'].iloc[-1] < session_end:
                raise ValueError(
                    f"{day}: Truncated — last tick at {df['time'].iloc[-1]}, "
                    f"expected >= {session_end}")

            self._trading_days.append(day)
            all_dfs.append(df)

        if missing_days:
            raise ValueError(
                f"Missing tick files for {len(missing_days)} expected trading days: "
                f"{missing_days[:5]}{'...' if len(missing_days) > 5 else ''}")

        if not all_dfs:
            raise ValueError(
                f"No tick data found in {self._tick_dir} for "
                f"{self._start} to {self._end}")

        # Concatenate all days
        ticks = pd.concat(all_dfs, ignore_index=True)

        # Cross-day monotonicity
        if not ticks['time'].is_monotonic_increasing:
            raise ValueError("Cross-day tick timestamps not monotonically increasing")

        # Price continuity check (warn only — earnings gaps are normal)
        prices = ticks.loc[ticks['price'] > 0, 'price'].values
        if len(prices) > 1:
            day_boundaries = ticks.loc[ticks['price'] > 0, 'time'].dt.date.values
            for i in range(1, len(prices)):
                if day_boundaries[i] != day_boundaries[i - 1]:
                    ratio = prices[i] / prices[i - 1]
                    if ratio > 1.2 or ratio < 0.8:
                        logger.warning(
                            "Overnight gap > 20%%: %.2f → %.2f",
                            prices[i - 1], prices[i])

        self._ticks = ticks
        self._tick_count = len(ticks)
        logger.info("Loaded %d ticks across %d trading days",
                     self._tick_count, len(self._trading_days))

    def aggregate_to_1min(self) -> pd.DataFrame:
        """Build 1-min OHLCV bars from ticks with session-aware minute grid."""
        df = self._ticks.copy()

        # 1. Filter: exclude halt markers and unreported prints
        mask = ~df['past_limit'] & ~df['unreported'] & (df['price'] > 0)
        filtered = df[mask].copy()

        # Also exclude zero-price without flags (already filtered by price > 0)

        # 2. Sort by (time, seq) for deterministic ordering
        filtered = filtered.sort_values(['time', 'seq'])

        # 3. Group by minute floor → raw OHLCV
        filtered['minute'] = filtered['time'].dt.floor('1min')
        raw_bars = filtered.groupby('minute').agg(
            open=('price', 'first'),
            high=('price', 'max'),
            low=('price', 'min'),
            close=('price', 'last'),
            volume=('size', 'sum'),
        )

        # 4. Build session-aware minute grid per day
        all_minutes = []
        for day in self._trading_days:
            day_str = day.isoformat()
            if self._rth_only:
                # Fixed clock window 09:30-15:59 (390 bars) — matches load_1min()
                grid = pd.date_range(
                    start=f'{day_str} 09:30',
                    end=f'{day_str} 15:59',
                    freq='1min',
                )
            else:
                # Extended hours: 04:00 to session end
                end_time = _session_end_time(day)
                grid = pd.date_range(
                    start=f'{day_str} {_SESSION_START}',
                    end=f'{day_str} {end_time}',
                    freq='1min',
                )
            all_minutes.append(grid)

        full_grid = pd.DatetimeIndex(
            np.concatenate([g.values for g in all_minutes]))

        # 5. Reindex onto grid with forward-fill (within day only)
        result = raw_bars.reindex(full_grid)

        # Forward-fill within each day (not across overnight boundaries)
        day_groups = pd.Series(result.index.date, index=result.index)
        for day_val in day_groups.unique():
            day_mask = day_groups == day_val
            day_slice = result.loc[day_mask]

            # Find first non-NaN bar in this day
            first_valid = day_slice['open'].first_valid_index()
            if first_valid is None:
                # No ticks this day in filtered data — should not happen
                # but drop these minutes rather than fabricate
                continue

            # Forward-fill only after first valid bar
            after_first = day_slice.loc[first_valid:]
            filled = after_first.ffill()
            # For ffilled bars: open=high=low=close=prev_close, volume=0
            nan_mask = after_first['open'].isna()
            if nan_mask.any():
                filled.loc[nan_mask, 'volume'] = 0
                # open/high/low already set to prev close by ffill
            result.loc[filled.index] = filled

            # Drop minutes before first tick (no forward-fill from yesterday)
            before_first = day_slice.loc[:first_valid].index[:-1]  # exclude first_valid
            result = result.drop(before_first, errors='ignore')

        # Drop any remaining NaN rows (minutes before first tick of each day)
        result = result.dropna(subset=['open'])

        # Ensure integer volume
        result['volume'] = result['volume'].astype(int)

        # 6. Post-aggregation validation
        # OHLC invariants
        assert (result['low'] <= result['open']).all(), "low > open found"
        assert (result['low'] <= result['close']).all(), "low > close found"
        assert (result['high'] >= result['open']).all(), "high < open found"
        assert (result['high'] >= result['close']).all(), "high < close found"
        assert not result[['open', 'high', 'low', 'close']].isna().any().any(), \
            "NaN in price columns"
        # No duplicate indices
        assert not result.index.duplicated().any(), "Duplicate minute indices"

        # Minute count per day
        for day in self._trading_days:
            day_count = (result.index.date == day).sum()
            if self._rth_only:
                expected = 390
            elif _is_early_close(day):
                expected = 780
            else:
                expected = 960
            if day_count != expected:
                logger.warning(
                    "%s: %d bars (expected %d)", day, day_count, expected)

        result.index.name = 'datetime'
        logger.info("Aggregated to %d 1-min bars", len(result))
        return result

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def trading_days_loaded(self) -> list:
        return list(self._trading_days)
