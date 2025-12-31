"""
Event Feature Extractor for AutoTrade v7.0

Extracts calendar event features (earnings, FOMC meetings).

Features (4 total):
  - days_to_next_earnings (TSLA)
  - days_since_last_earnings (TSLA)
  - days_to_next_fomc
  - is_fomc_week

Graceful Degradation:
  - If event data unavailable, returns neutral values (far from events)
  - Logs warning but does not fail extraction
  - Uses GracefulDegradation.get_default_events() as fallback
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import logging

from src.errors import EventFeaturesError
from src.monitoring import MetricsTracker
from src.errors import GracefulDegradation
from config import FeatureConfig

logger = logging.getLogger(__name__)


class EventFeatureExtractor:
    """
    Extract calendar event features.

    Tracks proximity to significant market events:
      - Earnings announcements (TSLA)
      - FOMC meetings (Federal Reserve interest rate decisions)

    Example:
        extractor = EventFeatureExtractor(config)
        features = extractor.extract(df, earnings_dates, fomc_dates)
    """

    def __init__(self, config: FeatureConfig, metrics: Optional[MetricsTracker] = None):
        """
        Initialize event feature extractor.

        Args:
            config: Feature configuration
            metrics: Optional metrics tracker
        """
        self.config = config
        self.metrics = metrics or MetricsTracker()
        self.recovery = GracefulDegradation()

        logger.info("EventFeatureExtractor initialized")

    def extract(
        self,
        df: pd.DataFrame,
        earnings_dates: Optional[List[datetime]] = None,
        fomc_dates: Optional[List[datetime]] = None,
        mode: str = 'batch'
    ) -> pd.DataFrame:
        """
        Extract event features.

        Args:
            df: Main DataFrame (for index alignment)
            earnings_dates: List of TSLA earnings dates (optional)
            fomc_dates: List of FOMC meeting dates (optional)
            mode: 'batch' or 'streaming'

        Returns:
            DataFrame with event features (same index as df)
        """
        with self.metrics.timer('event_features'):
            try:
                # If no dates provided, try to fetch
                if earnings_dates is None:
                    earnings_dates = self._fetch_earnings_dates(df.index[0], df.index[-1])

                if fomc_dates is None:
                    fomc_dates = self._fetch_fomc_dates(df.index[0], df.index[-1])

                # Calculate features
                features = self._calculate_event_features(df, earnings_dates, fomc_dates)

                logger.info(f"Event features extracted: {features.shape[1]} features")
                return features

            except Exception as e:
                logger.error(f"Event feature extraction failed: {e}")
                logger.warning("Using fallback event features")
                return self._get_fallback_features(df)

    def _fetch_earnings_dates(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[datetime]:
        """
        Fetch TSLA earnings dates.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of earnings dates
        """
        try:
            import yfinance as yf

            logger.info("Fetching TSLA earnings calendar")

            tsla = yf.Ticker("TSLA")
            calendar = tsla.calendar

            if calendar is None or calendar.empty:
                logger.warning("TSLA earnings calendar not available")
                return []

            # Extract earnings dates from calendar
            # yfinance calendar structure varies, handle gracefully
            earnings_dates = []

            if hasattr(calendar, 'index'):
                # Calendar is a DataFrame
                earnings_dates = list(pd.to_datetime(calendar.index))
            elif isinstance(calendar, dict) and 'Earnings Date' in calendar:
                # Calendar is a dict with earnings dates
                dates = calendar['Earnings Date']
                if isinstance(dates, list):
                    earnings_dates = [pd.to_datetime(d) for d in dates]
                else:
                    earnings_dates = [pd.to_datetime(dates)]

            # Filter to date range
            earnings_dates = [
                d for d in earnings_dates
                if start_date - timedelta(days=90) <= d <= end_date + timedelta(days=90)
            ]

            logger.info(f"Found {len(earnings_dates)} TSLA earnings dates")
            return earnings_dates

        except ImportError:
            logger.warning("yfinance not installed, cannot fetch earnings dates")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch earnings dates: {e}")
            return []

    def _fetch_fomc_dates(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[datetime]:
        """
        Fetch FOMC meeting dates.

        FOMC (Federal Open Market Committee) meets ~8 times per year.
        These are scheduled in advance and publicly available.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of FOMC meeting dates
        """
        # FOMC dates are scheduled in advance
        # In production, these would be fetched from an API or maintained manually
        # For now, return hardcoded 2024-2025 dates

        fomc_dates_2024_2025 = [
            # 2024
            datetime(2024, 1, 31),
            datetime(2024, 3, 20),
            datetime(2024, 5, 1),
            datetime(2024, 6, 12),
            datetime(2024, 7, 31),
            datetime(2024, 9, 18),
            datetime(2024, 11, 7),
            datetime(2024, 12, 18),
            # 2025
            datetime(2025, 1, 29),
            datetime(2025, 3, 19),
            datetime(2025, 5, 7),
            datetime(2025, 6, 18),
            datetime(2025, 7, 30),
            datetime(2025, 9, 17),
            datetime(2025, 10, 29),
            datetime(2025, 12, 17),
        ]

        # Filter to date range (with buffer)
        filtered = [
            d for d in fomc_dates_2024_2025
            if start_date - timedelta(days=30) <= d <= end_date + timedelta(days=30)
        ]

        logger.info(f"Using {len(filtered)} FOMC dates")
        return filtered

    def _calculate_event_features(
        self,
        df: pd.DataFrame,
        earnings_dates: Optional[List[datetime]],
        fomc_dates: Optional[List[datetime]]
    ) -> pd.DataFrame:
        """
        Calculate all event features.

        Returns DataFrame with event proximity features.
        """
        features = {}

        # Convert index to datetime if needed
        timestamps = pd.to_datetime(df.index)

        # === Earnings Features (2) ===
        if earnings_dates and len(earnings_dates) > 0:
            earnings_dates_sorted = sorted(earnings_dates)

            days_to_next = []
            days_since_last = []

            for ts in timestamps:
                # Find next earnings date
                future_dates = [d for d in earnings_dates_sorted if d >= ts]
                if future_dates:
                    next_date = min(future_dates)
                    days_to = (next_date - ts).days
                else:
                    days_to = 999  # Far future

                # Find last earnings date
                past_dates = [d for d in earnings_dates_sorted if d < ts]
                if past_dates:
                    last_date = max(past_dates)
                    days_since = (ts - last_date).days
                else:
                    days_since = 999  # Far past

                days_to_next.append(days_to)
                days_since_last.append(days_since)

            features['days_to_next_earnings'] = np.array(days_to_next, dtype=float)
            features['days_since_last_earnings'] = np.array(days_since_last, dtype=float)

        else:
            # No earnings data - use neutral values
            features['days_to_next_earnings'] = np.full(len(df), 999.0)
            features['days_since_last_earnings'] = np.full(len(df), 999.0)

        # === FOMC Features (2) ===
        if fomc_dates and len(fomc_dates) > 0:
            fomc_dates_sorted = sorted(fomc_dates)

            days_to_fomc = []
            is_fomc_week_flag = []

            for ts in timestamps:
                # Find next FOMC date
                future_dates = [d for d in fomc_dates_sorted if d >= ts]
                if future_dates:
                    next_date = min(future_dates)
                    days_to = (next_date - ts).days
                else:
                    days_to = 999

                # Find closest FOMC date (past or future)
                all_diffs = [abs((d - ts).days) for d in fomc_dates_sorted]
                min_diff = min(all_diffs) if all_diffs else 999

                # Within 7 days of FOMC?
                is_fomc_week = 1.0 if min_diff <= 7 else 0.0

                days_to_fomc.append(days_to)
                is_fomc_week_flag.append(is_fomc_week)

            features['days_to_next_fomc'] = np.array(days_to_fomc, dtype=float)
            features['is_fomc_week'] = np.array(is_fomc_week_flag, dtype=float)

        else:
            # No FOMC data - use neutral values
            features['days_to_next_fomc'] = np.full(len(df), 999.0)
            features['is_fomc_week'] = np.full(len(df), 0.0)

        result = pd.DataFrame(features, index=df.index)
        return result

    def _get_fallback_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get fallback event features when data unavailable.

        Returns neutral values (far from all events).
        """
        fallback = self.recovery.get_default_events()

        features = {
            'days_to_next_earnings': fallback['days_to_earnings'],
            'days_since_last_earnings': fallback['days_since_earnings'],
            'days_to_next_fomc': fallback['days_to_fomc'],
            'is_fomc_week': fallback['is_fomc_week'],
        }

        result = pd.DataFrame(features, index=df.index)

        logger.info("Using fallback event features (neutral)")
        return result


def extract_event_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    earnings_dates: Optional[List[datetime]] = None,
    fomc_dates: Optional[List[datetime]] = None,
    mode: str = 'batch',
    metrics: Optional[MetricsTracker] = None
) -> pd.DataFrame:
    """
    Convenience function to extract event features.

    Args:
        df: Main DataFrame (for index alignment)
        config: Feature configuration
        earnings_dates: Optional list of earnings dates
        fomc_dates: Optional list of FOMC dates
        mode: 'batch' or 'streaming'
        metrics: Optional metrics tracker

    Returns:
        DataFrame with event features

    Example:
        >>> config = get_feature_config()
        >>> df = load_5min_data()
        >>> event_features = extract_event_features(df, config)
    """
    extractor = EventFeatureExtractor(config, metrics)
    return extractor.extract(df, earnings_dates, fomc_dates, mode)
