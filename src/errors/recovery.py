"""
Graceful degradation strategies for AutoTrade v7.0

Provides fallback behaviors when components fail.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from .exceptions import VIXFeaturesError, EventFeaturesError


logger = logging.getLogger(__name__)


class GracefulDegradation:
    """
    Implements graceful degradation strategies.

    When non-critical components fail, use fallbacks instead of crashing.

    Usage:
        recovery = GracefulDegradation()

        try:
            vix_features = extract_vix_features(data)
        except VIXFeaturesError:
            vix_features = recovery.get_zero_vix_features()

        try:
            events = fetch_events()
        except EventFeaturesError:
            events = recovery.get_default_events()
    """

    def __init__(self):
        self._vix_cache: Optional[Dict[str, Any]] = None
        self._events_cache: Optional[Dict[str, Any]] = None

    # ========================================================================
    # VIX FEATURES
    # ========================================================================
    def get_zero_vix_features(self) -> Dict[str, float]:
        """
        Return zero VIX features when VIX extraction fails.

        This is safe because VIX features are external regime indicators,
        not core to channel detection.

        Returns:
            Dict of VIX features with zero/neutral values
        """
        logger.warning("Using zero VIX features (VIX extraction failed)")

        return {
            'vix_level': 20.0,  # Historical median
            'vix_percentile_20d': 0.5,
            'vix_percentile_252d': 0.5,
            'vix_change_1d': 0.0,
            'vix_change_5d': 0.0,
            'vix_regime': 0,  # Neutral regime
            'vix_tsla_corr_20d': 0.0,
            'vix_spy_corr_20d': 0.0,
            'vix_momentum_10d': 0.0,
            'vix_ma_ratio': 1.0,
            'vix_high_low_range': 0.0,
            'vix_trend_20d': 0.0,
            'vix_above_20': 0.0,
            'vix_above_30': 0.0,
            'vix_spike': 0.0,
        }

    def get_cached_vix_features(self) -> Optional[Dict[str, float]]:
        """
        Return last known good VIX features.

        Returns:
            Last cached VIX features or None if no cache
        """
        if self._vix_cache is not None:
            logger.warning("Using cached VIX features (VIX extraction failed)")
            return self._vix_cache
        return None

    def cache_vix_features(self, features: Dict[str, float]):
        """Cache VIX features for fallback use"""
        self._vix_cache = features.copy()

    # ========================================================================
    # EVENT FEATURES
    # ========================================================================
    def get_default_events(self) -> Dict[str, Any]:
        """
        Return default event features when event extraction fails.

        Returns:
            Dict of event features with no events
        """
        logger.warning("Using default events (event extraction failed)")

        return {
            'is_earnings_week': False,
            'days_until_earnings': 999,  # Far in future
            'days_until_fomc': 999,
            'is_high_impact_event': False,
        }

    def get_cached_events(self) -> Optional[Dict[str, Any]]:
        """
        Return last known good events.

        Returns:
            Last cached events or None if no cache
        """
        if self._events_cache is not None:
            logger.warning("Using cached events (event extraction failed)")
            return self._events_cache
        return None

    def cache_events(self, events: Dict[str, Any]):
        """Cache events for fallback use"""
        self._events_cache = events.copy()

    # ========================================================================
    # PREDICTION FALLBACKS
    # ========================================================================
    def get_fallback_prediction(self, symbol: str = "TSLA") -> Dict[str, Any]:
        """
        Return fallback prediction when model fails.

        This is a conservative "no prediction" result that tells
        the trading system to stay neutral.

        Args:
            symbol: Stock symbol

        Returns:
            Fallback prediction dict
        """
        logger.error("Returning fallback prediction (model inference failed)")

        return {
            'timestamp': None,
            'symbol': symbol,
            'selected_timeframe': 'daily',
            'duration_bars': 10.0,  # Conservative
            'duration_hours': 24.0 * 10 / 6.5,  # ~15 trading days
            'confidence': 0.0,  # Zero confidence = don't trade on this
            'predicted_high_pct': 0.0,
            'predicted_low_pct': 0.0,
            'direction': 'sideways',
            'transition_type': 'continue',
            'risk_level': 'high',
            'containment_violations': [],
            'inference_time_ms': 0.0,
            'fallback': True,  # Flag to indicate fallback prediction
        }

    def get_cached_prediction(self, symbol: str = "TSLA") -> Optional[Dict[str, Any]]:
        """
        Return last successful prediction.

        TODO: Implement prediction caching in inference service.

        Args:
            symbol: Stock symbol

        Returns:
            Last cached prediction or None
        """
        # TODO: Implement in Week 9-10
        return None
