"""
Event Features Module

Implements all 46 event features for market-moving events:
- TSLA-specific: earnings, delivery reports
- Macro: FOMC, CPI, NFP
- Market structure: quad witching

Feature Categories (46 total):
1. Generic timing (2): days_until_event, days_since_event
2. Event-specific timing - forward (6): days_until_* for each event type
3. Event-specific timing - backward (6): days_since_* for each event type
4. Intraday event timing (6): hours_until_* for same-day granularity
5. Binary flags (2): is_high_impact_event, is_earnings_week
6. Multi-hot 3-day flags (6): event_is_*_3d for each event type
7. Earnings context - backward (4): last earnings surprise, EPS, beat/miss
8. Earnings context - forward (2): upcoming estimate, trajectory
9. Pre-event drift (6): price drift INTO each event type
10. Post-event drift (6): price drift AFTER each event type

Data Source: /Volumes/NVME2/x6/data/events.csv
API Sources: Alpha Vantage (earnings), FRED (macro), programmatic (quad witching)
"""

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, time, timedelta
from pathlib import Path


# Event type constants (CSV values)
EVENT_TYPES = ['earnings', 'delivery', 'fomc', 'cpi', 'nfp', 'quad_witching']

# RTH hours for intraday normalization (09:30-16:00 = 6.5 hours)
RTH_HOURS = 6.5

# Trading day calendar (NYSE)
NYSE = mcal.get_calendar('NYSE')


@dataclass
class EventFeatures:
    """
    Complete set of 46 event features for a single timestamp.

    All features are normalized to facilitate model training.
    """
    # Generic timing (2)
    days_until_event: float         # [0, 1] - nearest future event (any type)
    days_since_event: float         # [0, 1] - most recent past event (any type)

    # Event-specific timing - Forward (6)
    days_until_tsla_earnings: float  # [0, 1] - normalized by 14 trading days
    days_until_tsla_delivery: float  # [0, 1]
    days_until_fomc: float           # [0, 1]
    days_until_cpi: float            # [0, 1]
    days_until_nfp: float            # [0, 1]
    days_until_quad_witching: float  # [0, 1]

    # Event-specific timing - Backward (6)
    days_since_tsla_earnings: float  # [0, 1]
    days_since_tsla_delivery: float  # [0, 1]
    days_since_fomc: float           # [0, 1]
    days_since_cpi: float            # [0, 1]
    days_since_nfp: float            # [0, 1]
    days_since_quad_witching: float  # [0, 1]

    # Intraday event timing (6) - hour-level granularity for same-day events
    hours_until_tsla_earnings: float  # [0, 1] - normalized by RTH_HOURS
    hours_until_tsla_delivery: float  # [0, 1]
    hours_until_fomc: float           # [0, 1]
    hours_until_cpi: float            # [0, 1]
    hours_until_nfp: float            # [0, 1]
    hours_until_quad_witching: float  # [0, 1]

    # Binary flags (2)
    is_high_impact_event: int        # {0, 1} - any event within ±3 trading days
    is_earnings_week: int            # {0, 1} - TSLA earnings within ±14 trading days

    # Multi-hot 3-day flags (6) - one per event type
    event_is_tsla_earnings_3d: int   # {0, 1}
    event_is_tsla_delivery_3d: int   # {0, 1}
    event_is_fomc_3d: int            # {0, 1}
    event_is_cpi_3d: int             # {0, 1}
    event_is_nfp_3d: int             # {0, 1}
    event_is_quad_witching_3d: int   # {0, 1}

    # Backward-looking earnings context (4) - TSLA specific
    last_earnings_surprise_pct: float    # (-1, 1) - tanh compressed
    last_earnings_surprise_abs: float    # [-2, 2] - clipped
    last_earnings_actual_eps_norm: float # [-1, 1] - tanh normalized
    last_earnings_beat_miss: int         # {-1, 0, 1} - from API data

    # Forward-looking earnings context (2) - TSLA specific
    upcoming_earnings_estimate_norm: float  # [-1, 1] - tanh, only within 14 days
    estimate_trajectory: float              # (-1, 1) - this Q vs last Q estimate

    # Pre-event drift (6) - price drift INTO each event (E-14 to sample)
    pre_tsla_earnings_drift: float   # [-0.5, 0.5] - clipped
    pre_tsla_delivery_drift: float   # [-0.5, 0.5]
    pre_fomc_drift: float            # [-0.5, 0.5]
    pre_cpi_drift: float             # [-0.5, 0.5]
    pre_nfp_drift: float             # [-0.5, 0.5]
    pre_quad_witching_drift: float   # [-0.5, 0.5]

    # Post-event drift (6) - price drift AFTER each event (event to sample)
    post_tsla_earnings_drift: float  # [-0.5, 0.5]
    post_tsla_delivery_drift: float  # [-0.5, 0.5]
    post_fomc_drift: float           # [-0.5, 0.5]
    post_cpi_drift: float            # [-0.5, 0.5]
    post_nfp_drift: float            # [-0.5, 0.5]
    post_quad_witching_drift: float  # [-0.5, 0.5]


class EventsHandler:
    """
    Manages event loading and visibility gating.

    Handles:
    - Loading events from CSV
    - Release time parsing (HH:MM, ALL_DAY, UNKNOWN)
    - Timezone normalization (ET-naive)
    - Intraday visibility gating (can we see this event's result yet?)
    """

    def __init__(self, events_csv_path: str):
        """
        Load events from CSV.

        Args:
            events_csv_path: Path to events.csv

        Expected CSV format:
            date,event_type,expected,actual,surprise_pct,beat_miss,source[,release_time]

        Note: release_time is optional. If missing, conservative defaults are used:
            - earnings/delivery: 20:00 (after extended hours)
            - fomc: 14:00
            - cpi/nfp: 08:30
            - quad_witching: ALL_DAY (09:30)
        """
        self.csv_path = events_csv_path
        self.events_df = self._load_events()

    def _load_events(self) -> pd.DataFrame:
        """Load and validate events CSV."""
        df = pd.read_csv(self.csv_path)

        # Validate required columns
        required = ['date', 'event_type', 'expected', 'actual', 'surprise_pct', 'beat_miss', 'source']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"events.csv missing required columns: {missing}")

        # Parse date column
        df['date'] = pd.to_datetime(df['date']).dt.date

        # Add release_time if missing (backward compatibility)
        if 'release_time' not in df.columns:
            df['release_time'] = df['event_type'].apply(self._get_default_release_time)

        # Validate release_time format
        self._validate_release_times(df)

        # Sort by date + release_time for proper ordering
        df['_parsed_time'] = df['release_time'].apply(_parse_release_time)
        df = df.sort_values(by=['date', '_parsed_time']).reset_index(drop=True)

        return df

    @staticmethod
    def _get_default_release_time(event_type: str) -> str:
        """Conservative default release times (prevent leakage)."""
        defaults = {
            'earnings': '20:00',       # Conservative: after extended hours
            'delivery': '20:00',       # Conservative: after extended hours
            'fomc': '14:00',          # Statement release time
            'cpi': '08:30',           # Pre-market
            'nfp': '08:30',           # Pre-market
            'quad_witching': 'ALL_DAY'  # True all-day event
        }
        return defaults.get(event_type, 'UNKNOWN')

    @staticmethod
    def _validate_release_times(df: pd.DataFrame):
        """Validate release_time column format."""
        import re
        valid_pattern = r'^(\d{2}:\d{2}|ALL_DAY|UNKNOWN)$'

        invalid = df[~df['release_time'].astype(str).str.match(valid_pattern, na=False)]
        if len(invalid) > 0:
            raise ValueError(
                f"Invalid release_time format in rows {invalid.index.tolist()}. "
                f"Expected 'HH:MM', 'ALL_DAY', or 'UNKNOWN'."
            )

    def get_visible_events(self, sample_timestamp: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """
        Split events into past/future based on intraday visibility.

        Args:
            sample_timestamp: Current timestamp (ET-naive)

        Returns:
            Dict with keys 'past' and 'future', each containing a DataFrame

        Visibility rules:
        - Past: event release time < sample timestamp (intraday aware)
        - Future: event release time >= sample timestamp
        - Uses strict timestamp comparison (not just date)
        """
        sample_date = sample_timestamp.date()
        sample_time = sample_timestamp.time()

        df = self.events_df.copy()

        # Partition by visibility
        past_mask = (
            (df['date'] < sample_date) |
            ((df['date'] == sample_date) & (df['_parsed_time'] < sample_time))
        )

        future_mask = ~past_mask

        past_events = df[past_mask].sort_values(by=['date', '_parsed_time'])
        future_events = df[future_mask].sort_values(by=['date', '_parsed_time'])

        return {
            'past': past_events,
            'future': future_events
        }


def _parse_release_time(release_time_str: str) -> time:
    """
    Parse release_time string into time object.

    Args:
        release_time_str: "HH:MM", "ALL_DAY", or "UNKNOWN"

    Returns:
        time object

    Mappings:
        - ALL_DAY -> 09:30 (observable from market open)
        - UNKNOWN -> 20:00 (conservative, after extended hours)
        - HH:MM -> parsed time
    """
    if release_time_str == "ALL_DAY":
        return time(9, 30)
    elif release_time_str == "UNKNOWN":
        return time(20, 0)
    elif release_time_str is None or release_time_str == "":
        raise ValueError("release_time cannot be None or empty")
    else:
        try:
            return datetime.strptime(release_time_str, '%H:%M').time()
        except ValueError:
            raise ValueError(
                f"Invalid release_time format: '{release_time_str}'. "
                f"Expected 'HH:MM', 'ALL_DAY', or 'UNKNOWN'."
            )


def _get_feature_prefix(event_type: str) -> str:
    """
    Map CSV event_type to feature name prefix.

    Args:
        event_type: CSV event_type value

    Returns:
        Feature name prefix

    Mappings:
        - earnings -> tsla_earnings
        - delivery -> tsla_delivery
        - fomc/cpi/nfp/quad_witching -> unchanged
    """
    if event_type in ('earnings', 'delivery'):
        return f'tsla_{event_type}'
    return event_type


def get_trading_days_until(sample_date: date, event_date: date) -> int:
    """
    Count trading days between dates (exclusive start, inclusive end).

    Args:
        sample_date: Starting date (excluded)
        event_date: Ending date (included)

    Returns:
        Number of trading days

    Example:
        sample_date = 2024-01-15 (Monday)
        event_date = 2024-01-17 (Wednesday)
        Returns: 2 (Tue, Wed)
    """
    if event_date <= sample_date:
        return 0

    schedule = NYSE.schedule(start_date=sample_date, end_date=event_date)
    return max(0, len(schedule) - 1)  # Exclude start date


def get_trading_days_since(past_date: date, sample_date: date) -> int:
    """
    Count trading days since past event (exclusive past, inclusive sample).

    Args:
        past_date: Past event date (excluded)
        sample_date: Current date (included)

    Returns:
        Number of trading days
    """
    if sample_date <= past_date:
        return 0

    schedule = NYSE.schedule(start_date=past_date, end_date=sample_date)
    return max(0, len(schedule) - 1)  # Exclude past date


def _build_event_timestamp(event_date: date, release_time: str) -> pd.Timestamp:
    """
    Build full timestamp from date + release_time.

    Args:
        event_date: Event date
        release_time: Release time string (HH:MM, ALL_DAY, UNKNOWN)

    Returns:
        ET-naive timestamp
    """
    time_obj = _parse_release_time(release_time)
    return pd.Timestamp.combine(event_date, time_obj)


def _compute_timing_features(
    sample_timestamp: pd.Timestamp,
    visible_events: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    """
    Compute all timing features (14 total: 2 generic + 6 forward + 6 backward).

    Args:
        sample_timestamp: Current timestamp (ET-naive)
        visible_events: Dict with 'past' and 'future' DataFrames

    Returns:
        Dict of feature_name -> value
    """
    features = {}
    past_events = visible_events['past']
    future_events = visible_events['future']

    # Generic timing (2)
    if len(future_events) > 0:
        nearest_future = future_events.iloc[0]
        event_ts = _build_event_timestamp(nearest_future['date'], nearest_future['release_time'])
        days_until = get_trading_days_until(sample_timestamp.date(), event_ts.date())
        features['days_until_event'] = min(days_until / 14.0, 1.0)
    else:
        features['days_until_event'] = 1.0

    if len(past_events) > 0:
        nearest_past = past_events.iloc[-1]
        event_ts = _build_event_timestamp(nearest_past['date'], nearest_past['release_time'])
        days_since = get_trading_days_since(event_ts.date(), sample_timestamp.date())
        features['days_since_event'] = min(days_since / 14.0, 1.0)
    else:
        features['days_since_event'] = 1.0

    # Event-specific timing (12: 6 forward + 6 backward)
    for event_type in EVENT_TYPES:
        prefix = _get_feature_prefix(event_type)

        # Forward: days until next event of this type
        future_of_type = future_events[future_events['event_type'] == event_type]
        if len(future_of_type) > 0:
            next_event = future_of_type.iloc[0]
            event_ts = _build_event_timestamp(next_event['date'], next_event['release_time'])
            days_until = get_trading_days_until(sample_timestamp.date(), event_ts.date())
            features[f'days_until_{prefix}'] = min(days_until / 14.0, 1.0)
        else:
            features[f'days_until_{prefix}'] = 1.0

        # Backward: days since last event of this type
        past_of_type = past_events[past_events['event_type'] == event_type]
        if len(past_of_type) > 0:
            last_event = past_of_type.iloc[-1]
            event_ts = _build_event_timestamp(last_event['date'], last_event['release_time'])
            days_since = get_trading_days_since(event_ts.date(), sample_timestamp.date())
            features[f'days_since_{prefix}'] = min(days_since / 14.0, 1.0)
        else:
            features[f'days_since_{prefix}'] = 1.0

    return features


def _compute_intraday_timing_features(
    sample_timestamp: pd.Timestamp,
    visible_events: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    """
    Compute intraday timing features (6 total: hours_until_* for each event type).

    These provide hour-level granularity for same-day events, enabling the model
    to learn patterns like "30 minutes before FOMC".

    Args:
        sample_timestamp: Current timestamp (ET-naive)
        visible_events: Dict with 'past' and 'future' DataFrames

    Returns:
        Dict of feature_name -> value (normalized to [0, 1])

    Normalization:
        - 0.0: event already passed (not in future_events)
        - 0.0-1.0: hours until event / RTH_HOURS (same day)
        - 1.0: event on future day
    """
    features = {}
    future_events = visible_events['future']

    for event_type in EVENT_TYPES:
        prefix = _get_feature_prefix(event_type)

        type_events = future_events[future_events['event_type'] == event_type]
        if len(type_events) == 0:
            features[f'hours_until_{prefix}'] = 0.0
            continue

        # Get next event of this type
        next_event = type_events.iloc[0]
        event_date = next_event['date']
        sample_date = sample_timestamp.date()

        if event_date > sample_date:
            # Future day - max value
            features[f'hours_until_{prefix}'] = 1.0
        elif event_date < sample_date:
            # Should never happen (would be in past_events), but handle defensively
            features[f'hours_until_{prefix}'] = 0.0
        else:
            # Same day - compute hours until release
            event_ts = _build_event_timestamp(event_date, next_event['release_time'])
            hours_diff = (event_ts - sample_timestamp).total_seconds() / 3600.0

            if hours_diff <= 0:
                features[f'hours_until_{prefix}'] = 0.0
            else:
                # Normalize by RTH hours, cap at 1.0
                features[f'hours_until_{prefix}'] = min(hours_diff / RTH_HOURS, 1.0)

    return features


def _compute_binary_flags(
    sample_timestamp: pd.Timestamp,
    visible_events: Dict[str, pd.DataFrame]
) -> Dict[str, int]:
    """
    Compute binary flag features (2 total: is_high_impact_event, is_earnings_week).

    Args:
        sample_timestamp: Current timestamp (ET-naive)
        visible_events: Dict with 'past' and 'future' DataFrames

    Returns:
        Dict of feature_name -> {0, 1}
    """
    features = {}
    past_events = visible_events['past']
    future_events = visible_events['future']

    sample_date = sample_timestamp.date()

    # is_high_impact_event: any event within ±3 trading days
    has_future = False
    if len(future_events) > 0:
        schedule_future = NYSE.schedule(
            start_date=sample_timestamp,
            end_date=sample_timestamp + pd.Timedelta(days=10)
        )
        if len(schedule_future) >= 4:
            future_cutoff = schedule_future.index[3].date()
            has_future = len(future_events[future_events['date'] <= future_cutoff]) > 0

    has_past = False
    if len(past_events) > 0:
        schedule_past = NYSE.schedule(
            start_date=sample_timestamp - pd.Timedelta(days=10),
            end_date=sample_timestamp
        )
        if len(schedule_past) >= 4:
            past_cutoff = schedule_past.index[-4].date()
            has_past = len(past_events[past_events['date'] >= past_cutoff]) > 0

    features['is_high_impact_event'] = 1 if (has_future or has_past) else 0

    # is_earnings_week: TSLA earnings within ±14 trading days
    has_future_earnings = False
    if len(future_events) > 0:
        schedule_future = NYSE.schedule(
            start_date=sample_timestamp,
            end_date=sample_timestamp + pd.Timedelta(days=25)
        )
        if len(schedule_future) >= 15:
            future_cutoff = schedule_future.index[14].date()
            earnings_future = future_events[
                (future_events['event_type'] == 'earnings') &
                (future_events['date'] <= future_cutoff)
            ]
            has_future_earnings = len(earnings_future) > 0

    has_past_earnings = False
    if len(past_events) > 0:
        schedule_past = NYSE.schedule(
            start_date=sample_timestamp - pd.Timedelta(days=25),
            end_date=sample_timestamp
        )
        if len(schedule_past) >= 15:
            past_cutoff = schedule_past.index[-15].date()
            earnings_past = past_events[
                (past_events['event_type'] == 'earnings') &
                (past_events['date'] >= past_cutoff)
            ]
            has_past_earnings = len(earnings_past) > 0

    features['is_earnings_week'] = 1 if (has_future_earnings or has_past_earnings) else 0

    return features


def _compute_multi_hot_flags(
    sample_timestamp: pd.Timestamp,
    visible_events: Dict[str, pd.DataFrame]
) -> Dict[str, int]:
    """
    Compute multi-hot 3-day flags (6 total: event_is_*_3d for each event type).

    Multi-hot encoding sets ALL flags for events within 3 trading days,
    allowing the model to see multiple simultaneous events.

    Args:
        sample_timestamp: Current timestamp (ET-naive)
        visible_events: Dict with 'past' and 'future' DataFrames

    Returns:
        Dict of feature_name -> {0, 1}
    """
    features = {}
    future_events = visible_events['future']

    # Get date 3 trading days from sample
    schedule = NYSE.schedule(
        start_date=sample_timestamp,
        end_date=sample_timestamp + pd.Timedelta(days=10)
    )

    if len(schedule) < 4:
        cutoff_date = (sample_timestamp + pd.Timedelta(days=10)).date()
    else:
        cutoff_date = schedule.index[min(3, len(schedule) - 1)].date()

    # Filter to within 3 trading days
    upcoming = future_events[future_events['date'] <= cutoff_date]

    # Set flag for each event type
    for event_type in EVENT_TYPES:
        prefix = _get_feature_prefix(event_type)
        features[f'event_is_{prefix}_3d'] = (
            1 if event_type in upcoming['event_type'].values else 0
        )

    return features


def _compute_earnings_features(
    sample_timestamp: pd.Timestamp,
    visible_events: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    """
    Compute earnings-specific features (6 total: 4 backward + 2 forward).

    Backward-looking (4):
        - last_earnings_surprise_pct: Surprise % with tanh compression
        - last_earnings_surprise_abs: Absolute EPS difference, clipped
        - last_earnings_actual_eps_norm: Actual EPS normalized by tanh
        - last_earnings_beat_miss: -1=miss, 0=meet, 1=beat (from API)

    Forward-looking (2):
        - upcoming_earnings_estimate_norm: Consensus EPS (tanh), within 14 days
        - estimate_trajectory: This quarter vs last quarter estimate

    Args:
        sample_timestamp: Current timestamp (ET-naive)
        visible_events: Dict with 'past' and 'future' DataFrames

    Returns:
        Dict of feature_name -> value
    """
    features = {
        'last_earnings_surprise_pct': 0.0,
        'last_earnings_surprise_abs': 0.0,
        'last_earnings_actual_eps_norm': 0.0,
        'last_earnings_beat_miss': 0,
        'upcoming_earnings_estimate_norm': 0.0,
        'estimate_trajectory': 0.0,
    }

    past_events = visible_events['past']
    future_events = visible_events['future']

    # Backward-looking: last earnings
    past_earnings = past_events[past_events['event_type'] == 'earnings']
    if len(past_earnings) > 0:
        last = past_earnings.iloc[-1]

        actual = last.get('actual', 0.0)
        expected = last.get('expected', 0.0)

        # Surprise percentage (tanh compressed)
        if expected != 0:
            surprise_fraction = (actual - expected) / abs(expected)
            features['last_earnings_surprise_pct'] = float(np.tanh(surprise_fraction))

        # Surprise absolute (clipped)
        surprise_abs = actual - expected
        features['last_earnings_surprise_abs'] = float(np.clip(surprise_abs, -2, 2))

        # Actual EPS normalized
        features['last_earnings_actual_eps_norm'] = float(np.tanh(actual))

        # Beat/miss (from API data, not computed)
        features['last_earnings_beat_miss'] = int(last.get('beat_miss', 0))

    # Forward-looking: next earnings (only within 14 trading days)
    future_earnings = future_events[future_events['event_type'] == 'earnings']
    if len(future_earnings) > 0:
        next_earnings = future_earnings.iloc[0]

        # Check if within 14 trading days
        trading_days_until = get_trading_days_until(
            sample_timestamp.date(),
            next_earnings['date']
        )

        if trading_days_until <= 14:
            expected = next_earnings.get('expected', 0.0)
            features['upcoming_earnings_estimate_norm'] = float(np.tanh(expected))

            # Trajectory: compare to last quarter's estimate
            if len(past_earnings) > 0:
                last_expected = past_earnings.iloc[-1].get('expected', 0.0)

                # Safe division (prevent near-zero explosion)
                safe_denom = max(abs(last_expected), 0.10)
                trajectory = (expected - last_expected) / safe_denom
                features['estimate_trajectory'] = float(np.tanh(trajectory))

    return features


def _get_price_n_trading_days_ago(
    price_df: pd.DataFrame,
    event_timestamp: pd.Timestamp,
    n_days: int
) -> Optional[float]:
    """
    Get price N trading days before event.

    Args:
        price_df: OHLCV DataFrame with DatetimeIndex
        event_timestamp: Event timestamp
        n_days: Number of trading days to go back

    Returns:
        Close price or None if not available
    """
    event_date = event_timestamp.date()

    # Get schedule for N+1 days before event
    schedule = NYSE.schedule(
        start_date=event_date - pd.Timedelta(days=n_days * 2),  # Buffer for weekends
        end_date=event_date
    )

    if len(schedule) < n_days + 1:
        return None

    # Get the date N trading days ago
    target_date = schedule.index[-(n_days + 1)].date()

    # Find price on that date (any bar from that day)
    mask = price_df.index.date == target_date
    if mask.sum() == 0:
        return None

    return float(price_df[mask].iloc[-1]['close'])


def _get_post_event_anchor_price(
    event_timestamp: pd.Timestamp,
    price_df: pd.DataFrame
) -> Optional[float]:
    """
    Get first available price AFTER event for post-event drift anchor.

    Three cases based on event timing:
    1. Pre-market (00:00-09:30): Same-day 09:30 open (e.g., CPI/NFP at 08:30)
    2. During RTH (09:30-16:00): First bar after event (e.g., FOMC at 14:00)
    3. After-hours (16:00-23:59): Next trading day's 09:30 open (e.g., earnings at 16:05)

    Args:
        event_timestamp: Event timestamp (ET-naive)
        price_df: OHLCV DataFrame with DatetimeIndex

    Returns:
        Open price or None if not available
    """
    RTH_START = time(9, 30)
    RTH_END = time(16, 0)

    event_time = event_timestamp.time()
    event_date = event_timestamp.date()

    # Three-way classification
    is_pre_market = event_time < RTH_START
    is_during_rth = RTH_START <= event_time <= RTH_END

    if is_during_rth:
        # Use first bar after event_timestamp
        mask = price_df.index > event_timestamp
    elif is_pre_market:
        # Pre-market event: use same-day 09:30 open
        same_day_open = pd.Timestamp.combine(event_date, RTH_START)
        mask = price_df.index >= same_day_open
    else:
        # After-hours event: use next trading day's 09:30 open
        schedule = NYSE.schedule(
            start_date=event_date + timedelta(days=1),
            end_date=event_date + timedelta(days=7)
        )
        if len(schedule) == 0:
            return None

        next_trading_day = schedule.index[0].date()
        next_open = pd.Timestamp.combine(next_trading_day, RTH_START)
        mask = price_df.index >= next_open

    if mask.sum() == 0:
        return None

    # Return open price (captures gap reaction)
    return float(price_df[mask].iloc[0]['open'])


def _compute_drift_features(
    sample_timestamp: pd.Timestamp,
    visible_events: Dict[str, pd.DataFrame],
    price_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute all drift features (12 total: 6 pre-event + 6 post-event).

    Pre-event drift: Price movement from E-14 to sample (before event)
    Post-event drift: Price movement from event to sample (after event)

    Each event type has separate drift features, allowing the model to learn
    event-specific patterns.

    Args:
        sample_timestamp: Current timestamp (ET-naive)
        visible_events: Dict with 'past' and 'future' DataFrames
        price_df: OHLCV DataFrame for drift calculations (ET-naive index)

    Returns:
        Dict of feature_name -> value (clipped to [-0.5, 0.5])
    """
    features = {}
    past_events = visible_events['past']
    future_events = visible_events['future']

    for event_type in EVENT_TYPES:
        prefix = _get_feature_prefix(event_type)

        # Pre-event drift (E-14 to sample)
        future_of_type = future_events[future_events['event_type'] == event_type]
        if len(future_of_type) > 0:
            next_event = future_of_type.iloc[0]
            event_ts = _build_event_timestamp(next_event['date'], next_event['release_time'])

            # Check if within 14 days
            days_until = get_trading_days_until(sample_timestamp.date(), event_ts.date())

            if days_until <= 14 and sample_timestamp < event_ts:
                # Anchor price: 14 trading days before event
                anchor_price = _get_price_n_trading_days_ago(price_df, event_ts, 14)

                # Current price: last bar before sample (leak-safe)
                current_mask = price_df.index < sample_timestamp
                if current_mask.sum() > 0 and anchor_price is not None and anchor_price > 0:
                    current_price = price_df[current_mask].iloc[-1]['close']
                    drift = (current_price - anchor_price) / anchor_price
                    features[f'pre_{prefix}_drift'] = float(np.clip(drift, -0.5, 0.5))
                else:
                    features[f'pre_{prefix}_drift'] = 0.0
            else:
                features[f'pre_{prefix}_drift'] = 0.0
        else:
            features[f'pre_{prefix}_drift'] = 0.0

        # Post-event drift (event to sample)
        past_of_type = past_events[past_events['event_type'] == event_type]
        if len(past_of_type) > 0:
            last_event = past_of_type.iloc[-1]
            event_ts = _build_event_timestamp(last_event['date'], last_event['release_time'])

            # Check if within 14 days
            days_since = get_trading_days_since(event_ts.date(), sample_timestamp.date())

            if days_since <= 14 and sample_timestamp > event_ts:
                # Anchor price: first price after event
                anchor_price = _get_post_event_anchor_price(event_ts, price_df)

                # Current price: last bar before sample (leak-safe)
                current_mask = price_df.index < sample_timestamp
                if current_mask.sum() > 0 and anchor_price is not None and anchor_price > 0:
                    current_price = price_df[current_mask].iloc[-1]['close']
                    drift = (current_price - anchor_price) / anchor_price
                    features[f'post_{prefix}_drift'] = float(np.clip(drift, -0.5, 0.5))
                else:
                    features[f'post_{prefix}_drift'] = 0.0
            else:
                features[f'post_{prefix}_drift'] = 0.0
        else:
            features[f'post_{prefix}_drift'] = 0.0

    return features


def extract_event_features(
    sample_timestamp: pd.Timestamp,
    events_handler: EventsHandler,
    price_df: pd.DataFrame
) -> EventFeatures:
    """
    Extract all 46 event features for a single timestamp.

    This is the main entry point for event feature extraction.

    Args:
        sample_timestamp: Current timestamp (ET-naive)
        events_handler: EventsHandler instance with loaded events
        price_df: OHLCV DataFrame for drift calculations (ET-naive index)

    Returns:
        EventFeatures dataclass with all 46 features

    Feature breakdown:
        - Timing features: 14 (2 generic + 12 event-specific)
        - Intraday timing: 6
        - Binary flags: 2
        - Multi-hot flags: 6
        - Earnings context: 6
        - Drift features: 12
        Total: 46
    """
    # Get visible events (handles intraday gating)
    visible = events_handler.get_visible_events(sample_timestamp)

    # Compute feature groups
    timing = _compute_timing_features(sample_timestamp, visible)
    intraday = _compute_intraday_timing_features(sample_timestamp, visible)
    binary_flags = _compute_binary_flags(sample_timestamp, visible)
    multi_hot = _compute_multi_hot_flags(sample_timestamp, visible)
    earnings = _compute_earnings_features(sample_timestamp, visible)
    drift = _compute_drift_features(sample_timestamp, visible, price_df)

    # Combine all features into dataclass
    return EventFeatures(
        # Generic timing (2)
        days_until_event=timing['days_until_event'],
        days_since_event=timing['days_since_event'],

        # Event-specific timing - forward (6)
        days_until_tsla_earnings=timing['days_until_tsla_earnings'],
        days_until_tsla_delivery=timing['days_until_tsla_delivery'],
        days_until_fomc=timing['days_until_fomc'],
        days_until_cpi=timing['days_until_cpi'],
        days_until_nfp=timing['days_until_nfp'],
        days_until_quad_witching=timing['days_until_quad_witching'],

        # Event-specific timing - backward (6)
        days_since_tsla_earnings=timing['days_since_tsla_earnings'],
        days_since_tsla_delivery=timing['days_since_tsla_delivery'],
        days_since_fomc=timing['days_since_fomc'],
        days_since_cpi=timing['days_since_cpi'],
        days_since_nfp=timing['days_since_nfp'],
        days_since_quad_witching=timing['days_since_quad_witching'],

        # Intraday timing (6)
        hours_until_tsla_earnings=intraday['hours_until_tsla_earnings'],
        hours_until_tsla_delivery=intraday['hours_until_tsla_delivery'],
        hours_until_fomc=intraday['hours_until_fomc'],
        hours_until_cpi=intraday['hours_until_cpi'],
        hours_until_nfp=intraday['hours_until_nfp'],
        hours_until_quad_witching=intraday['hours_until_quad_witching'],

        # Binary flags (2)
        is_high_impact_event=binary_flags['is_high_impact_event'],
        is_earnings_week=binary_flags['is_earnings_week'],

        # Multi-hot flags (6)
        event_is_tsla_earnings_3d=multi_hot['event_is_tsla_earnings_3d'],
        event_is_tsla_delivery_3d=multi_hot['event_is_tsla_delivery_3d'],
        event_is_fomc_3d=multi_hot['event_is_fomc_3d'],
        event_is_cpi_3d=multi_hot['event_is_cpi_3d'],
        event_is_nfp_3d=multi_hot['event_is_nfp_3d'],
        event_is_quad_witching_3d=multi_hot['event_is_quad_witching_3d'],

        # Earnings context - backward (4)
        last_earnings_surprise_pct=earnings['last_earnings_surprise_pct'],
        last_earnings_surprise_abs=earnings['last_earnings_surprise_abs'],
        last_earnings_actual_eps_norm=earnings['last_earnings_actual_eps_norm'],
        last_earnings_beat_miss=earnings['last_earnings_beat_miss'],

        # Earnings context - forward (2)
        upcoming_earnings_estimate_norm=earnings['upcoming_earnings_estimate_norm'],
        estimate_trajectory=earnings['estimate_trajectory'],

        # Pre-event drift (6)
        pre_tsla_earnings_drift=drift['pre_tsla_earnings_drift'],
        pre_tsla_delivery_drift=drift['pre_tsla_delivery_drift'],
        pre_fomc_drift=drift['pre_fomc_drift'],
        pre_cpi_drift=drift['pre_cpi_drift'],
        pre_nfp_drift=drift['pre_nfp_drift'],
        pre_quad_witching_drift=drift['pre_quad_witching_drift'],

        # Post-event drift (6)
        post_tsla_earnings_drift=drift['post_tsla_earnings_drift'],
        post_tsla_delivery_drift=drift['post_tsla_delivery_drift'],
        post_fomc_drift=drift['post_fomc_drift'],
        post_cpi_drift=drift['post_cpi_drift'],
        post_nfp_drift=drift['post_nfp_drift'],
        post_quad_witching_drift=drift['post_quad_witching_drift'],
    )


def event_features_to_dict(features: EventFeatures) -> Dict[str, float]:
    """
    Convert EventFeatures dataclass to dictionary.

    Useful for serialization and integration with other feature extractors.

    Args:
        features: EventFeatures instance

    Returns:
        Dict mapping feature_name -> value (46 entries)
    """
    from dataclasses import asdict
    return asdict(features)


# Feature names in order (for array-based operations)
EVENT_FEATURE_NAMES = [
    # Generic timing (2)
    'days_until_event', 'days_since_event',

    # Event-specific timing - forward (6)
    'days_until_tsla_earnings', 'days_until_tsla_delivery', 'days_until_fomc',
    'days_until_cpi', 'days_until_nfp', 'days_until_quad_witching',

    # Event-specific timing - backward (6)
    'days_since_tsla_earnings', 'days_since_tsla_delivery', 'days_since_fomc',
    'days_since_cpi', 'days_since_nfp', 'days_since_quad_witching',

    # Intraday timing (6)
    'hours_until_tsla_earnings', 'hours_until_tsla_delivery', 'hours_until_fomc',
    'hours_until_cpi', 'hours_until_nfp', 'hours_until_quad_witching',

    # Binary flags (2)
    'is_high_impact_event', 'is_earnings_week',

    # Multi-hot flags (6)
    'event_is_tsla_earnings_3d', 'event_is_tsla_delivery_3d', 'event_is_fomc_3d',
    'event_is_cpi_3d', 'event_is_nfp_3d', 'event_is_quad_witching_3d',

    # Earnings context - backward (4)
    'last_earnings_surprise_pct', 'last_earnings_surprise_abs',
    'last_earnings_actual_eps_norm', 'last_earnings_beat_miss',

    # Earnings context - forward (2)
    'upcoming_earnings_estimate_norm', 'estimate_trajectory',

    # Pre-event drift (6)
    'pre_tsla_earnings_drift', 'pre_tsla_delivery_drift', 'pre_fomc_drift',
    'pre_cpi_drift', 'pre_nfp_drift', 'pre_quad_witching_drift',

    # Post-event drift (6)
    'post_tsla_earnings_drift', 'post_tsla_delivery_drift', 'post_fomc_drift',
    'post_cpi_drift', 'post_nfp_drift', 'post_quad_witching_drift',
]

assert len(EVENT_FEATURE_NAMES) == 46, f"Expected 46 features, got {len(EVENT_FEATURE_NAMES)}"
