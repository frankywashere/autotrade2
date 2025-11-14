"""
Prediction Cache - Prevents Redundant Predictions

Caches predictions per timeframe and only allows updates when:
1. New bar has closed for that timeframe
2. Manual refresh is requested
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
import threading


class PredictionCache:
    """
    Thread-safe prediction cache with timeframe-aware expiration.

    Predictions are cached until the next bar closes for that timeframe.
    """

    def __init__(self):
        self.cache = {}  # {timeframe: {'prediction': {...}, 'timestamp': datetime, 'expires_at': datetime}}
        self.lock = threading.Lock()

    def get(self, timeframe: str) -> Optional[Dict]:
        """
        Get cached prediction if still valid.

        Returns:
            Prediction dict or None if expired/missing
        """
        with self.lock:
            if timeframe not in self.cache:
                return None

            cached = self.cache[timeframe]
            now = datetime.now()

            # Check if expired
            if now >= cached['expires_at']:
                return None

            return cached['prediction']

    def set(self, timeframe: str, prediction: Dict):
        """
        Cache a prediction with automatic expiration.

        Prediction expires at the next bar close time for this timeframe.
        """
        with self.lock:
            now = datetime.now()
            expires_at = self._calculate_next_bar_close(timeframe, now)

            self.cache[timeframe] = {
                'prediction': prediction,
                'timestamp': now,
                'expires_at': expires_at
            }

    def invalidate(self, timeframe: str):
        """Force invalidate a cached prediction."""
        with self.lock:
            if timeframe in self.cache:
                del self.cache[timeframe]

    def invalidate_all(self):
        """Clear all cached predictions."""
        with self.lock:
            self.cache = {}

    def get_time_until_update(self, timeframe: str) -> Optional[float]:
        """
        Get seconds until next update for this timeframe.

        Returns:
            Seconds until next bar close, or None if no cache
        """
        with self.lock:
            if timeframe not in self.cache:
                return None

            cached = self.cache[timeframe]
            now = datetime.now()
            remaining = (cached['expires_at'] - now).total_seconds()

            return max(0, remaining)

    def _calculate_next_bar_close(self, timeframe: str, current_time: datetime) -> datetime:
        """
        Calculate when the next bar closes for this timeframe.

        Examples:
        - 15min: If now is 13:37, next close is 13:45
        - 1hour: If now is 13:37, next close is 14:00
        - 4hour: If now is 13:37, next close is 16:00
        - daily: Next market close (16:00)
        """
        now = current_time

        if timeframe == '15min':
            # Next 15-minute mark (:00, :15, :30, :45)
            minutes_past = now.minute % 15
            if minutes_past == 0 and now.second == 0:
                # Exactly at bar close, next is 15 min away
                return now + timedelta(minutes=15)
            else:
                # Round up to next 15-min mark
                minutes_to_add = 15 - minutes_past
                next_close = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
                return next_close

        elif timeframe == '1hour':
            # Next top of hour
            if now.minute == 0 and now.second == 0:
                return now + timedelta(hours=1)
            else:
                next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                return next_close

        elif timeframe == '4hour':
            # Next 4-hour mark (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
            current_hour = now.hour
            next_hour = ((current_hour // 4) + 1) * 4

            if next_hour >= 24:
                # Next day
                next_close = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_close = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)

            return next_close

        elif timeframe == 'daily':
            # Next market close (4:00 PM ET = 16:00)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

            if now >= market_close:
                # Already past today's close, next is tomorrow
                market_close = market_close + timedelta(days=1)

            # Skip weekends
            while market_close.weekday() >= 5:  # Saturday/Sunday
                market_close = market_close + timedelta(days=1)

            return market_close

        else:
            # Default: 1 hour expiration
            return now + timedelta(hours=1)

    def get_all_cached(self) -> Dict[str, Dict]:
        """Get all cached predictions (for dashboard display)."""
        with self.lock:
            return {
                tf: {
                    'prediction': data['prediction'],
                    'timestamp': data['timestamp'],
                    'expires_at': data['expires_at'],
                    'seconds_until_update': self.get_time_until_update(tf)
                }
                for tf, data in self.cache.items()
            }


if __name__ == '__main__':
    # Test prediction cache
    cache = PredictionCache()

    print("Testing prediction cache...")

    # Test 15min caching
    test_pred = {
        'predicted_high': 1.5,
        'predicted_low': -0.8,
        'confidence': 0.85
    }

    cache.set('15min', test_pred)
    print(f"✓ Cached 15min prediction")

    retrieved = cache.get('15min')
    print(f"✓ Retrieved: {retrieved}")

    time_until = cache.get_time_until_update('15min')
    print(f"✓ Time until update: {time_until:.0f} seconds")

    all_cached = cache.get_all_cached()
    print(f"✓ All cached: {list(all_cached.keys())}")

    print("\n✅ Prediction cache test complete")
