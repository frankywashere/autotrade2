"""
Event and Calendar Features for V15

Implements 30 event/calendar features that capture temporal and event-based patterns
affecting trading behavior.

Categories:
- Time-Based Features (12): hour_of_day, minute_of_hour, day_of_week, is_monday, is_friday,
                            week_of_month, month_of_year, is_quarter_end, is_month_end,
                            is_market_open_hour, is_market_close_hour, is_lunch_hour
- Trading Session Features (8): session_progress, bars_since_open, bars_until_close,
                                is_first_30min, is_last_30min, is_overnight_gap,
                                overnight_gap_size, volume_vs_session_avg
- Calendar Event Proximity (10): days_to_next_friday, is_opex_week, is_triple_witching,
                                 days_to_month_end, is_first_trading_day_of_month,
                                 is_last_trading_day_of_month, is_first_trading_day_of_week,
                                 is_last_trading_day_of_week, days_since_last_fed_day,
                                 is_earnings_season

All functions return Dict[str, float] with valid float values (no NaN/inf).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, time, timedelta

from .utils import safe_float, safe_divide, safe_mean


# =============================================================================
# Market Hours Constants (Eastern Time)
# =============================================================================

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Session periods in ET
OPEN_HOUR_START = time(9, 30)
OPEN_HOUR_END = time(10, 30)
CLOSE_HOUR_START = time(15, 0)
CLOSE_HOUR_END = time(16, 0)
LUNCH_HOUR_START = time(12, 0)
LUNCH_HOUR_END = time(13, 0)

# Trading session duration in minutes
TRADING_SESSION_MINUTES = 390  # 9:30 AM to 4:00 PM = 6.5 hours


def extract_event_features(
    timestamp: pd.Timestamp,
    df: pd.DataFrame
) -> Dict[str, float]:
    """
    Extract 30 event/calendar features from timestamp and DataFrame context.

    Args:
        timestamp: The current bar's timestamp (should be timezone-aware or naive in ET)
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            and a DatetimeIndex. Used for session-relative calculations.

    Returns:
        Dict[str, float] with 30 features, all guaranteed to be valid floats
    """
    features: Dict[str, float] = {}

    # Time-Based Features (12)
    time_features = _calculate_time_features(timestamp)
    features.update(time_features)

    # Trading Session Features (8)
    session_features = _calculate_session_features(timestamp, df)
    features.update(session_features)

    # Calendar Event Proximity Features (10)
    calendar_features = _calculate_calendar_features(timestamp, df)
    features.update(calendar_features)

    # Final safety check
    for key, value in features.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            features[key] = 0.0

    return features


# =============================================================================
# Time-Based Features (12)
# =============================================================================

def _calculate_time_features(timestamp: pd.Timestamp) -> Dict[str, float]:
    """
    Calculate time-based features from the timestamp.

    Features (12):
    - hour_of_day: 0-23 normalized to 0-1
    - minute_of_hour: 0-59 normalized to 0-1
    - day_of_week: 0=Monday to 4=Friday normalized to 0-1
    - is_monday: Binary (1.0 if Monday)
    - is_friday: Binary (1.0 if Friday)
    - week_of_month: 1-5 normalized to 0-1
    - month_of_year: 1-12 normalized to 0-1
    - is_quarter_end: Binary (1.0 if quarter end month: Mar, Jun, Sep, Dec)
    - is_month_end: Binary (1.0 if last 3 trading days of month)
    - is_market_open_hour: Binary (1.0 if 9:30-10:30 ET)
    - is_market_close_hour: Binary (1.0 if 3:00-4:00 ET)
    - is_lunch_hour: Binary (1.0 if 12:00-1:00 ET)
    """
    features: Dict[str, float] = {}

    try:
        # Convert to datetime for easier manipulation
        dt = pd.to_datetime(timestamp)

        # Hour of day (0-23 normalized to 0-1)
        hour = dt.hour
        features['hour_of_day'] = safe_float(hour / 23.0, 0.5)

        # Minute of hour (0-59 normalized to 0-1)
        minute = dt.minute
        features['minute_of_hour'] = safe_float(minute / 59.0, 0.5)

        # Day of week (0=Monday, 4=Friday, normalized to 0-1)
        # weekday() returns 0-6 (Mon-Sun), but for trading we only care about 0-4
        day_of_week = dt.weekday()
        day_of_week = min(day_of_week, 4)  # Cap at Friday
        features['day_of_week'] = safe_float(day_of_week / 4.0, 0.5)

        # Is Monday (binary)
        features['is_monday'] = 1.0 if day_of_week == 0 else 0.0

        # Is Friday (binary)
        features['is_friday'] = 1.0 if day_of_week == 4 else 0.0

        # Week of month (1-5 normalized to 0-1)
        day_of_month = dt.day
        week_of_month = (day_of_month - 1) // 7 + 1  # 1-5
        week_of_month = min(week_of_month, 5)
        features['week_of_month'] = safe_float((week_of_month - 1) / 4.0, 0.5)

        # Month of year (1-12 normalized to 0-1)
        month = dt.month
        features['month_of_year'] = safe_float((month - 1) / 11.0, 0.5)

        # Is quarter end (Mar=3, Jun=6, Sep=9, Dec=12)
        is_quarter_end = month in [3, 6, 9, 12]
        features['is_quarter_end'] = 1.0 if is_quarter_end else 0.0

        # Is month end (approximate: last 3 trading days - day >= 26)
        features['is_month_end'] = 1.0 if day_of_month >= 26 else 0.0

        # Market session time features
        current_time = dt.time()

        # Is market open hour (9:30-10:30 ET)
        is_open_hour = OPEN_HOUR_START <= current_time < OPEN_HOUR_END
        features['is_market_open_hour'] = 1.0 if is_open_hour else 0.0

        # Is market close hour (3:00-4:00 ET)
        is_close_hour = CLOSE_HOUR_START <= current_time <= CLOSE_HOUR_END
        features['is_market_close_hour'] = 1.0 if is_close_hour else 0.0

        # Is lunch hour (12:00-1:00 ET - typically lower volume)
        is_lunch = LUNCH_HOUR_START <= current_time < LUNCH_HOUR_END
        features['is_lunch_hour'] = 1.0 if is_lunch else 0.0

    except Exception:
        # Return default values on any error
        features = {
            'hour_of_day': 0.5,
            'minute_of_hour': 0.5,
            'day_of_week': 0.5,
            'is_monday': 0.0,
            'is_friday': 0.0,
            'week_of_month': 0.5,
            'month_of_year': 0.5,
            'is_quarter_end': 0.0,
            'is_month_end': 0.0,
            'is_market_open_hour': 0.0,
            'is_market_close_hour': 0.0,
            'is_lunch_hour': 0.0,
        }

    return features


# =============================================================================
# Trading Session Features (8)
# =============================================================================

def _calculate_session_features(
    timestamp: pd.Timestamp,
    df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate trading session features.

    Features (8):
    - session_progress: 0-1, where in the trading day (0=open, 1=close)
    - bars_since_open: Number of bars since market open (normalized)
    - bars_until_close: Number of bars until market close (normalized)
    - is_first_30min: Binary (1.0 if within first 30 minutes)
    - is_last_30min: Binary (1.0 if within last 30 minutes)
    - is_overnight_gap: Binary (1.0 if significant gap from previous close)
    - overnight_gap_size: Size of overnight gap (normalized)
    - volume_vs_session_avg: Current volume vs session average
    """
    features: Dict[str, float] = {}

    try:
        dt = pd.to_datetime(timestamp)
        current_time = dt.time()

        # Calculate minutes since market open
        market_open = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
        market_close = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)

        # Minutes from market open
        open_minutes = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE
        current_minutes = current_time.hour * 60 + current_time.minute
        close_minutes = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE

        minutes_since_open = max(0, current_minutes - open_minutes)
        minutes_until_close = max(0, close_minutes - current_minutes)

        # Session progress (0=open, 1=close)
        session_progress = safe_divide(minutes_since_open, TRADING_SESSION_MINUTES, 0.5)
        session_progress = float(np.clip(session_progress, 0.0, 1.0))
        features['session_progress'] = session_progress

        # Bars since open (normalized by typical bar count)
        # Assuming we might have various timeframes, normalize by 390 (1-min bars in day)
        features['bars_since_open'] = safe_float(minutes_since_open / TRADING_SESSION_MINUTES, 0.5)

        # Bars until close (normalized)
        features['bars_until_close'] = safe_float(minutes_until_close / TRADING_SESSION_MINUTES, 0.5)

        # Is first 30 minutes (9:30-10:00 ET)
        is_first_30min = current_time >= time(9, 30) and current_time < time(10, 0)
        features['is_first_30min'] = 1.0 if is_first_30min else 0.0

        # Is last 30 minutes (3:30-4:00 ET)
        is_last_30min = current_time >= time(15, 30) and current_time <= time(16, 0)
        features['is_last_30min'] = 1.0 if is_last_30min else 0.0

        # Overnight gap features - need DataFrame for this
        overnight_gap = 0.0
        overnight_gap_size = 0.0

        if df is not None and len(df) >= 2 and 'close' in df.columns and 'open' in df.columns:
            try:
                # Get today's open and yesterday's close
                current_open = safe_float(df['open'].iloc[-1], 0.0)
                prev_close = safe_float(df['close'].iloc[-2], 0.0)

                if prev_close > 0 and current_open > 0:
                    # Calculate gap percentage
                    gap_pct = (current_open - prev_close) / prev_close

                    # Significant gap threshold: 0.5% (0.005)
                    if abs(gap_pct) >= 0.005:
                        overnight_gap = 1.0

                    # Normalize gap size: clip to +/- 10% range and scale to -1 to 1
                    overnight_gap_size = float(np.clip(gap_pct * 10, -1.0, 1.0))

            except Exception:
                pass

        features['is_overnight_gap'] = safe_float(overnight_gap, 0.0)
        features['overnight_gap_size'] = safe_float(overnight_gap_size, 0.0)

        # Volume vs session average
        volume_vs_avg = 1.0  # Default to 1.0 (average)

        if df is not None and 'volume' in df.columns and len(df) >= 1:
            try:
                current_volume = safe_float(df['volume'].iloc[-1], 0.0)
                avg_volume = safe_mean(df['volume'].values, default=1.0)

                if avg_volume > 0:
                    volume_vs_avg = safe_divide(current_volume, avg_volume, 1.0)
                    # Clip to reasonable range (0.1 to 10x average)
                    volume_vs_avg = float(np.clip(volume_vs_avg, 0.1, 10.0))

            except Exception:
                pass

        features['volume_vs_session_avg'] = safe_float(volume_vs_avg, 1.0)

    except Exception:
        # Return default values on any error
        features = {
            'session_progress': 0.5,
            'bars_since_open': 0.5,
            'bars_until_close': 0.5,
            'is_first_30min': 0.0,
            'is_last_30min': 0.0,
            'is_overnight_gap': 0.0,
            'overnight_gap_size': 0.0,
            'volume_vs_session_avg': 1.0,
        }

    return features


# =============================================================================
# Calendar Event Proximity Features (10)
# =============================================================================

def _calculate_calendar_features(
    timestamp: pd.Timestamp,
    df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate calendar event proximity features.

    Features (10):
    - days_to_next_friday: Days until next Friday (options expiration), normalized
    - is_opex_week: Binary (1.0 if options expiration week - 3rd Friday of month)
    - is_triple_witching: Binary (1.0 if quarterly options expiration - 3rd Friday of Mar/Jun/Sep/Dec)
    - days_to_month_end: Days until end of month, normalized
    - is_first_trading_day_of_month: Binary
    - is_last_trading_day_of_month: Binary (approximate)
    - is_first_trading_day_of_week: Binary (Monday or first after holiday)
    - is_last_trading_day_of_week: Binary (Friday or last before holiday)
    - days_since_last_fed_day: Days since last Fed meeting (6-week cycle approximation)
    - is_earnings_season: Binary (mid-month of Jan, Apr, Jul, Oct)
    """
    features: Dict[str, float] = {}

    try:
        dt = pd.to_datetime(timestamp)

        # Days to next Friday (normalized by 7 - max days to Friday)
        day_of_week = dt.weekday()  # 0=Monday, 4=Friday
        if day_of_week <= 4:  # Mon-Fri
            days_to_friday = 4 - day_of_week
        else:  # Sat-Sun
            days_to_friday = (7 - day_of_week) + 4
        features['days_to_next_friday'] = safe_float(days_to_friday / 7.0, 0.5)

        # Determine the 3rd Friday of the current month (OPEX day)
        first_day = dt.replace(day=1)
        first_friday_offset = (4 - first_day.weekday()) % 7
        third_friday = first_day + timedelta(days=first_friday_offset + 14)

        # Is OPEX week (week containing 3rd Friday)
        # Check if current date is in the same week as 3rd Friday
        current_week_start = dt - timedelta(days=dt.weekday())
        opex_week_start = third_friday - timedelta(days=third_friday.weekday())
        is_opex_week = current_week_start.date() == opex_week_start.date()
        features['is_opex_week'] = 1.0 if is_opex_week else 0.0

        # Is triple witching (3rd Friday of Mar, Jun, Sep, Dec)
        is_triple_witching = False
        if dt.month in [3, 6, 9, 12]:
            if dt.date() == third_friday.date():
                is_triple_witching = True
        features['is_triple_witching'] = 1.0 if is_triple_witching else 0.0

        # Days to month end (normalized by 31)
        days_in_month = _get_days_in_month(dt.year, dt.month)
        days_to_month_end = days_in_month - dt.day
        features['days_to_month_end'] = safe_float(days_to_month_end / 31.0, 0.5)

        # Is first trading day of month (day 1-3 and Monday-Friday)
        is_first_day_of_month = dt.day <= 3 and day_of_week <= 4
        features['is_first_trading_day_of_month'] = 1.0 if is_first_day_of_month else 0.0

        # Is last trading day of month (approximate: last 2 weekdays)
        is_last_day_of_month = (days_in_month - dt.day) <= 2 and day_of_week <= 4
        features['is_last_trading_day_of_month'] = 1.0 if is_last_day_of_month else 0.0

        # Is first trading day of week (Monday)
        features['is_first_trading_day_of_week'] = 1.0 if day_of_week == 0 else 0.0

        # Is last trading day of week (Friday)
        features['is_last_trading_day_of_week'] = 1.0 if day_of_week == 4 else 0.0

        # Days since last Fed day (approximate: Fed meets roughly every 6 weeks)
        # This is a rough approximation using a 42-day (6-week) cycle
        # Starting from a reference Fed meeting date
        fed_reference = datetime(2024, 1, 31)  # Known Fed meeting date
        days_since_reference = (dt - pd.Timestamp(fed_reference)).days
        days_in_cycle = days_since_reference % 42
        # Normalize: 0 means just had Fed meeting, 1 means about to have one
        features['days_since_last_fed_day'] = safe_float(days_in_cycle / 42.0, 0.5)

        # Is earnings season (mid-month of Jan, Apr, Jul, Oct - roughly days 10-25)
        is_earnings_month = dt.month in [1, 4, 7, 10]
        is_earnings_period = 10 <= dt.day <= 25
        is_earnings_season = is_earnings_month and is_earnings_period
        features['is_earnings_season'] = 1.0 if is_earnings_season else 0.0

    except Exception:
        # Return default values on any error
        features = {
            'days_to_next_friday': 0.5,
            'is_opex_week': 0.0,
            'is_triple_witching': 0.0,
            'days_to_month_end': 0.5,
            'is_first_trading_day_of_month': 0.0,
            'is_last_trading_day_of_month': 0.0,
            'is_first_trading_day_of_week': 0.0,
            'is_last_trading_day_of_week': 0.0,
            'days_since_last_fed_day': 0.5,
            'is_earnings_season': 0.0,
        }

    return features


# =============================================================================
# Helper Functions
# =============================================================================

def _get_days_in_month(year: int, month: int) -> int:
    """Get the number of days in a given month."""
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    return last_day.day


def _get_third_friday(year: int, month: int) -> datetime:
    """Get the third Friday of a given month (options expiration day)."""
    first_day = datetime(year, month, 1)
    # Find the first Friday
    first_friday_offset = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=first_friday_offset)
    # Third Friday is 14 days after first Friday
    return first_friday + timedelta(days=14)


def _is_trading_day(dt: datetime) -> bool:
    """
    Check if a date is a trading day (weekday and not a major holiday).
    This is a simplified check - only checks for weekdays.
    """
    return dt.weekday() <= 4  # Monday=0 to Friday=4


# =============================================================================
# Feature Name Getters
# =============================================================================

def get_event_feature_names() -> list:
    """Get the list of all 30 event/calendar feature names."""
    return [
        # Time-Based Features (12)
        'hour_of_day',
        'minute_of_hour',
        'day_of_week',
        'is_monday',
        'is_friday',
        'week_of_month',
        'month_of_year',
        'is_quarter_end',
        'is_month_end',
        'is_market_open_hour',
        'is_market_close_hour',
        'is_lunch_hour',
        # Trading Session Features (8)
        'session_progress',
        'bars_since_open',
        'bars_until_close',
        'is_first_30min',
        'is_last_30min',
        'is_overnight_gap',
        'overnight_gap_size',
        'volume_vs_session_avg',
        # Calendar Event Proximity (10)
        'days_to_next_friday',
        'is_opex_week',
        'is_triple_witching',
        'days_to_month_end',
        'is_first_trading_day_of_month',
        'is_last_trading_day_of_month',
        'is_first_trading_day_of_week',
        'is_last_trading_day_of_week',
        'days_since_last_fed_day',
        'is_earnings_season',
    ]


def get_event_feature_count() -> int:
    """Get the total number of event/calendar features (30)."""
    return len(get_event_feature_names())


# =============================================================================
# Batch Processing
# =============================================================================

def extract_event_features_batch(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract event features for all rows in a DataFrame.

    Args:
        df: DataFrame with OHLCV data and DatetimeIndex (or timestamp column)
        timestamp_col: Name of timestamp column if not using index

    Returns:
        DataFrame with event features for each row
    """
    feature_records = []

    # Get timestamps
    if timestamp_col is not None and timestamp_col in df.columns:
        timestamps = df[timestamp_col]
    elif isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index
    else:
        # Try to convert index to datetime
        try:
            timestamps = pd.to_datetime(df.index)
        except Exception:
            # Return empty features if no valid timestamps
            return pd.DataFrame(index=df.index)

    for i, ts in enumerate(timestamps):
        # Create a slice of data up to current row for session calculations
        df_slice = df.iloc[:i + 1] if i > 0 else df.iloc[:1]
        features = extract_event_features(ts, df_slice)
        feature_records.append(features)

    return pd.DataFrame(feature_records, index=df.index)


# =============================================================================
# Feature Metadata
# =============================================================================

def get_event_feature_descriptions() -> Dict[str, str]:
    """Get descriptions for all event/calendar features."""
    return {
        # Time-Based Features
        'hour_of_day': 'Hour of day (0-23) normalized to 0-1',
        'minute_of_hour': 'Minute of hour (0-59) normalized to 0-1',
        'day_of_week': 'Day of week (0=Mon, 4=Fri) normalized to 0-1',
        'is_monday': 'Binary: 1.0 if Monday',
        'is_friday': 'Binary: 1.0 if Friday',
        'week_of_month': 'Week of month (1-5) normalized to 0-1',
        'month_of_year': 'Month of year (1-12) normalized to 0-1',
        'is_quarter_end': 'Binary: 1.0 if quarter-end month (Mar/Jun/Sep/Dec)',
        'is_month_end': 'Binary: 1.0 if last ~3 trading days of month',
        'is_market_open_hour': 'Binary: 1.0 if 9:30-10:30 ET',
        'is_market_close_hour': 'Binary: 1.0 if 3:00-4:00 ET',
        'is_lunch_hour': 'Binary: 1.0 if 12:00-1:00 ET (lower volume period)',
        # Trading Session Features
        'session_progress': 'Progress through trading day (0=open, 1=close)',
        'bars_since_open': 'Minutes since market open, normalized by session length',
        'bars_until_close': 'Minutes until market close, normalized by session length',
        'is_first_30min': 'Binary: 1.0 if within first 30 minutes of trading',
        'is_last_30min': 'Binary: 1.0 if within last 30 minutes of trading',
        'is_overnight_gap': 'Binary: 1.0 if gap >= 0.5% from previous close',
        'overnight_gap_size': 'Size of overnight gap, scaled to -1 to 1',
        'volume_vs_session_avg': 'Current volume / session average volume',
        # Calendar Event Proximity
        'days_to_next_friday': 'Days until next Friday, normalized by 7',
        'is_opex_week': 'Binary: 1.0 if options expiration week (3rd Friday week)',
        'is_triple_witching': 'Binary: 1.0 if quarterly options expiration day',
        'days_to_month_end': 'Days until month end, normalized by 31',
        'is_first_trading_day_of_month': 'Binary: 1.0 if first trading day of month',
        'is_last_trading_day_of_month': 'Binary: 1.0 if last trading day of month',
        'is_first_trading_day_of_week': 'Binary: 1.0 if Monday',
        'is_last_trading_day_of_week': 'Binary: 1.0 if Friday',
        'days_since_last_fed_day': 'Days since last Fed meeting (6-week cycle)',
        'is_earnings_season': 'Binary: 1.0 if mid-month of Jan/Apr/Jul/Oct',
    }


def get_event_feature_ranges() -> Dict[str, tuple]:
    """Get expected value ranges for all event/calendar features."""
    return {
        # Time-Based Features - normalized values
        'hour_of_day': (0.0, 1.0),
        'minute_of_hour': (0.0, 1.0),
        'day_of_week': (0.0, 1.0),
        'is_monday': (0.0, 1.0),
        'is_friday': (0.0, 1.0),
        'week_of_month': (0.0, 1.0),
        'month_of_year': (0.0, 1.0),
        'is_quarter_end': (0.0, 1.0),
        'is_month_end': (0.0, 1.0),
        'is_market_open_hour': (0.0, 1.0),
        'is_market_close_hour': (0.0, 1.0),
        'is_lunch_hour': (0.0, 1.0),
        # Trading Session Features
        'session_progress': (0.0, 1.0),
        'bars_since_open': (0.0, 1.0),
        'bars_until_close': (0.0, 1.0),
        'is_first_30min': (0.0, 1.0),
        'is_last_30min': (0.0, 1.0),
        'is_overnight_gap': (0.0, 1.0),
        'overnight_gap_size': (-1.0, 1.0),
        'volume_vs_session_avg': (0.1, 10.0),
        # Calendar Event Proximity
        'days_to_next_friday': (0.0, 1.0),
        'is_opex_week': (0.0, 1.0),
        'is_triple_witching': (0.0, 1.0),
        'days_to_month_end': (0.0, 1.0),
        'is_first_trading_day_of_month': (0.0, 1.0),
        'is_last_trading_day_of_month': (0.0, 1.0),
        'is_first_trading_day_of_week': (0.0, 1.0),
        'is_last_trading_day_of_week': (0.0, 1.0),
        'days_since_last_fed_day': (0.0, 1.0),
        'is_earnings_season': (0.0, 1.0),
    }
