"""
Prediction Scheduler for Hierarchical LNN

Determines when each layer should make predictions based on:
- Time intervals (fast: 30min, medium: 2 hours, slow: daily)
- Events (channel breaks, high volatility)
- Errors (previous prediction was wrong)

This prevents excessive predictions while ensuring timely updates.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, List
import numpy as np


class PredictionScheduler:
    """
    Manages prediction scheduling for different layers.

    Each layer predicts on different schedules:
    - Fast (15min): Every 30 mins OR channel break OR high error
    - Medium (1hour): 3-4x per day OR when fast error >3%
    - Slow (daily): 1x per day OR regime change
    """

    def __init__(
        self,
        fast_interval_minutes: int = 30,
        medium_interval_hours: int = 2,
        slow_interval_days: int = 1,
        error_trigger_threshold: float = 2.0,  # >2% error triggers prediction
    ):
        """
        Initialize prediction scheduler.

        Args:
            fast_interval_minutes: Time between fast layer predictions
            medium_interval_hours: Time between medium layer predictions
            slow_interval_days: Time between slow layer predictions
            error_trigger_threshold: Error threshold to trigger early prediction
        """
        self.fast_interval = timedelta(minutes=fast_interval_minutes)
        self.medium_interval = timedelta(hours=medium_interval_hours)
        self.slow_interval = timedelta(days=slow_interval_days)
        self.error_trigger_threshold = error_trigger_threshold

        # Track last prediction times
        self.last_fast_prediction = None
        self.last_medium_prediction = None
        self.last_slow_prediction = None

        # Track last errors
        self.last_fast_error = 0.0
        self.last_medium_error = 0.0
        self.last_slow_error = 0.0

    def should_predict(
        self,
        layer: str,
        current_time: datetime,
        market_state: Optional[Dict] = None,
        last_error: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Determine if layer should make a prediction.

        Args:
            layer: 'fast', 'medium', or 'slow'
            current_time: Current timestamp
            market_state: Optional market state dict with:
                - channel_broken: bool
                - regime_changed: bool
                - volatility: float
            last_error: Optional last prediction error (%)

        Returns:
            decision: Dict with:
                - should_predict: bool
                - reason: str (why prediction is/isn't needed)
                - priority: int (0=skip, 1=low, 2=medium, 3=high)
        """
        if market_state is None:
            market_state = {}

        if layer == 'fast':
            return self._should_predict_fast(current_time, market_state, last_error)
        elif layer == 'medium':
            return self._should_predict_medium(current_time, market_state, last_error)
        elif layer == 'slow':
            return self._should_predict_slow(current_time, market_state, last_error)
        else:
            return {'should_predict': False, 'reason': 'Unknown layer', 'priority': 0}

    def _should_predict_fast(
        self,
        current_time: datetime,
        market_state: Dict,
        last_error: Optional[float]
    ) -> Dict[str, any]:
        """Determine if fast layer (15min) should predict."""

        # Priority 3 (HIGH): Channel break detected
        if market_state.get('channel_broken', False):
            self.last_fast_prediction = current_time
            return {
                'should_predict': True,
                'reason': 'Channel break detected',
                'priority': 3
            }

        # Priority 3 (HIGH): Previous prediction had high error
        if last_error is not None and last_error > self.error_trigger_threshold * 1.5:
            self.last_fast_prediction = current_time
            self.last_fast_error = last_error
            return {
                'should_predict': True,
                'reason': f'High error in last prediction ({last_error:.2f}%)',
                'priority': 3
            }

        # Priority 2 (MEDIUM): Time interval reached
        if self.last_fast_prediction is None:
            self.last_fast_prediction = current_time
            return {
                'should_predict': True,
                'reason': 'First prediction',
                'priority': 2
            }

        time_since_last = current_time - self.last_fast_prediction
        if time_since_last >= self.fast_interval:
            self.last_fast_prediction = current_time
            return {
                'should_predict': True,
                'reason': f'Time interval reached ({time_since_last})',
                'priority': 2
            }

        # Priority 2 (MEDIUM): High volatility
        if market_state.get('volatility', 0) > 0.05:  # >5% volatility
            self.last_fast_prediction = current_time
            return {
                'should_predict': True,
                'reason': f'High volatility ({market_state["volatility"]:.2f})',
                'priority': 2
            }

        # Don't predict
        return {
            'should_predict': False,
            'reason': f'Waiting for interval (last: {time_since_last} ago)',
            'priority': 0
        }

    def _should_predict_medium(
        self,
        current_time: datetime,
        market_state: Dict,
        last_error: Optional[float]
    ) -> Dict[str, any]:
        """Determine if medium layer (1hour) should predict."""

        # Priority 3 (HIGH): Fast layer had very high error
        if last_error is not None and last_error > 3.0:
            self.last_medium_prediction = current_time
            self.last_medium_error = last_error
            return {
                'should_predict': True,
                'reason': f'Fast layer high error ({last_error:.2f}%)',
                'priority': 3
            }

        # Priority 3 (HIGH): Regime change
        if market_state.get('regime_changed', False):
            self.last_medium_prediction = current_time
            return {
                'should_predict': True,
                'reason': 'Market regime changed',
                'priority': 3
            }

        # Priority 2 (MEDIUM): Time interval reached (2 hours)
        if self.last_medium_prediction is None:
            self.last_medium_prediction = current_time
            return {
                'should_predict': True,
                'reason': 'First prediction',
                'priority': 2
            }

        time_since_last = current_time - self.last_medium_prediction
        if time_since_last >= self.medium_interval:
            self.last_medium_prediction = current_time
            return {
                'should_predict': True,
                'reason': f'Time interval reached ({time_since_last})',
                'priority': 2
            }

        # Priority 1 (LOW): Significant price movement
        if market_state.get('price_change_1h', 0) > 0.02:  # >2% move in last hour
            return {
                'should_predict': True,
                'reason': f'Significant 1h move ({market_state["price_change_1h"]:.2f}%)',
                'priority': 1
            }

        # Don't predict
        return {
            'should_predict': False,
            'reason': f'Waiting for interval (last: {time_since_last} ago)',
            'priority': 0
        }

    def _should_predict_slow(
        self,
        current_time: datetime,
        market_state: Dict,
        last_error: Optional[float]
    ) -> Dict[str, any]:
        """Determine if slow layer (daily) should predict."""

        # Priority 3 (HIGH): Major regime change
        if market_state.get('regime_changed', False):
            self.last_slow_prediction = current_time
            return {
                'should_predict': True,
                'reason': 'Major regime change',
                'priority': 3
            }

        # Priority 2 (MEDIUM): Time interval reached (1 day)
        if self.last_slow_prediction is None:
            self.last_slow_prediction = current_time
            return {
                'should_predict': True,
                'reason': 'First prediction',
                'priority': 2
            }

        time_since_last = current_time - self.last_slow_prediction
        if time_since_last >= self.slow_interval:
            self.last_slow_prediction = current_time
            return {
                'should_predict': True,
                'reason': f'Time interval reached ({time_since_last})',
                'priority': 2
            }

        # Priority 2 (MEDIUM): Earnings/major event
        if market_state.get('major_event', False):
            self.last_slow_prediction = current_time
            return {
                'should_predict': True,
                'reason': 'Major event (earnings, FOMC, etc.)',
                'priority': 2
            }

        # Don't predict
        return {
            'should_predict': False,
            'reason': f'Waiting for daily interval (last: {time_since_last} ago)',
            'priority': 0
        }

    def update_error(self, layer: str, error: float):
        """
        Update last error for a layer.

        Args:
            layer: 'fast', 'medium', or 'slow'
            error: Average prediction error (%)
        """
        if layer == 'fast':
            self.last_fast_error = error
        elif layer == 'medium':
            self.last_medium_error = error
        elif layer == 'slow':
            self.last_slow_error = error

    def get_next_prediction_times(self, current_time: datetime) -> Dict[str, datetime]:
        """
        Get next scheduled prediction times for all layers.

        Args:
            current_time: Current timestamp

        Returns:
            next_times: Dict with 'fast', 'medium', 'slow' next prediction times
        """
        next_times = {}

        if self.last_fast_prediction:
            next_times['fast'] = self.last_fast_prediction + self.fast_interval
        else:
            next_times['fast'] = current_time

        if self.last_medium_prediction:
            next_times['medium'] = self.last_medium_prediction + self.medium_interval
        else:
            next_times['medium'] = current_time

        if self.last_slow_prediction:
            next_times['slow'] = self.last_slow_prediction + self.slow_interval
        else:
            next_times['slow'] = current_time

        return next_times

    def reset(self):
        """Reset scheduler (for new trading session)."""
        self.last_fast_prediction = None
        self.last_medium_prediction = None
        self.last_slow_prediction = None
        self.last_fast_error = 0.0
        self.last_medium_error = 0.0
        self.last_slow_error = 0.0

    def get_stats(self) -> Dict[str, any]:
        """Get scheduler statistics."""
        return {
            'last_fast_prediction': self.last_fast_prediction.isoformat() if self.last_fast_prediction else None,
            'last_medium_prediction': self.last_medium_prediction.isoformat() if self.last_medium_prediction else None,
            'last_slow_prediction': self.last_slow_prediction.isoformat() if self.last_slow_prediction else None,
            'last_fast_error': self.last_fast_error,
            'last_medium_error': self.last_medium_error,
            'last_slow_error': self.last_slow_error,
            'fast_interval_minutes': self.fast_interval.total_seconds() / 60,
            'medium_interval_hours': self.medium_interval.total_seconds() / 3600,
            'slow_interval_days': self.slow_interval.total_seconds() / 86400
        }


def detect_channel_break(
    current_features: Dict[str, float],
    previous_features: Dict[str, float],
    timeframe: str = '1h'
) -> bool:
    """
    Detect if a channel has broken.

    Args:
        current_features: Current feature dict
        previous_features: Previous feature dict (from last prediction)
        timeframe: Which timeframe to check ('1h', '4h', 'daily')

    Returns:
        broken: True if channel break detected
    """
    # Check if channel position moved from inside to outside
    current_pos = current_features.get(f'tsla_channel_position_norm_{timeframe}', 0)
    prev_pos = previous_features.get(f'tsla_channel_position_norm_{timeframe}', 0)

    # Break if position went from < 0.8 to > 1.0 (or < -1.0)
    if abs(prev_pos) < 0.8 and abs(current_pos) > 1.0:
        return True

    # Check volume surge
    volume_surge = current_features.get('tsla_volume_surge', 0)
    if volume_surge > 2.0:  # 2x average volume
        return True

    return False


def detect_regime_change(
    current_features: Dict[str, float],
    previous_features: Dict[str, float]
) -> bool:
    """
    Detect if market regime has changed significantly.

    Args:
        current_features: Current feature dict
        previous_features: Previous feature dict

    Returns:
        changed: True if regime change detected
    """
    # Check volatility spike
    current_vol = current_features.get('tsla_volatility_10', 0)
    prev_vol = previous_features.get('tsla_volatility_10', 0)

    if current_vol > prev_vol * 2.0:  # 2x volatility increase
        return True

    # Check SPY correlation flip
    current_corr = current_features.get('correlation_10', 0)
    prev_corr = previous_features.get('correlation_10', 0)

    if current_corr * prev_corr < 0 and abs(current_corr - prev_corr) > 0.5:
        # Correlation flipped sign significantly
        return True

    return False
