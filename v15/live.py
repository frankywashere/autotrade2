"""
V15 Live Trading Integration

Provides real-time prediction updates for live trading systems.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import time
from pathlib import Path

from .inference import Predictor, Prediction
from .config import TIMEFRAMES, STANDARD_WINDOWS
from .exceptions import V15Error

logger = logging.getLogger(__name__)


@dataclass
class LivePrediction:
    """Live prediction with additional metadata."""
    prediction: Prediction
    data_timestamp: pd.Timestamp
    prediction_time: datetime
    latency_ms: float
    channel_valid: bool


class LivePredictor:
    """
    Real-time predictor for live trading.

    Features:
    - Maintains rolling data window
    - Caches channel detection
    - Provides prediction latency metrics
    - Supports callbacks for new predictions
    """

    def __init__(
        self,
        checkpoint_path: str,
        min_bars: int = 35000,
        on_prediction: Optional[Callable[[LivePrediction], None]] = None
    ):
        self.predictor = Predictor.load(checkpoint_path)
        self.min_bars = min_bars
        self.on_prediction = on_prediction

        # Rolling data storage
        self.tsla_data: Optional[pd.DataFrame] = None
        self.spy_data: Optional[pd.DataFrame] = None
        self.vix_data: Optional[pd.DataFrame] = None

        # Metrics
        self.prediction_count = 0
        self.total_latency_ms = 0.0

        logger.info(f"LivePredictor initialized with {self.min_bars} min bars")

    def update_data(
        self,
        tsla_bar: Dict[str, float],
        spy_bar: Dict[str, float],
        vix_bar: Optional[Dict[str, float]] = None,
        timestamp: Optional[pd.Timestamp] = None
    ):
        """
        Update with new bar data.

        Args:
            tsla_bar: Dict with open, high, low, close, volume
            spy_bar: Dict with open, high, low, close, volume
            vix_bar: Optional VIX data
            timestamp: Bar timestamp
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        # Create single-row DataFrames
        tsla_row = pd.DataFrame([tsla_bar], index=[timestamp])
        spy_row = pd.DataFrame([spy_bar], index=[timestamp])

        # Append to rolling window
        if self.tsla_data is None:
            self.tsla_data = tsla_row
            self.spy_data = spy_row
        else:
            self.tsla_data = pd.concat([self.tsla_data, tsla_row])
            self.spy_data = pd.concat([self.spy_data, spy_row])

            # Trim to max size
            max_bars = self.min_bars + 1000
            if len(self.tsla_data) > max_bars:
                self.tsla_data = self.tsla_data.iloc[-self.min_bars:]
                self.spy_data = self.spy_data.iloc[-self.min_bars:]

        # Handle VIX
        if vix_bar:
            vix_row = pd.DataFrame([vix_bar], index=[timestamp])
            if self.vix_data is None:
                self.vix_data = vix_row
            else:
                self.vix_data = pd.concat([self.vix_data, vix_row])
                if len(self.vix_data) > max_bars:
                    self.vix_data = self.vix_data.iloc[-self.min_bars:]

    def can_predict(self) -> bool:
        """Check if we have enough data for prediction."""
        if self.tsla_data is None:
            return False
        return len(self.tsla_data) >= self.min_bars

    def predict(self) -> Optional[LivePrediction]:
        """
        Make prediction with current data.

        Returns:
            LivePrediction or None if not enough data
        """
        if not self.can_predict():
            logger.warning(f"Not enough data: {len(self.tsla_data) if self.tsla_data is not None else 0}/{self.min_bars}")
            return None

        start_time = time.perf_counter()

        try:
            # Use VIX or create dummy
            vix = self.vix_data if self.vix_data is not None else pd.DataFrame({
                'open': [20.0], 'high': [20.0], 'low': [20.0], 'close': [20.0]
            }, index=[self.tsla_data.index[-1]])

            prediction = self.predictor.predict(
                self.tsla_data,
                self.spy_data,
                vix
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            live_pred = LivePrediction(
                prediction=prediction,
                data_timestamp=self.tsla_data.index[-1],
                prediction_time=datetime.now(),
                latency_ms=latency_ms,
                channel_valid=True,
            )

            # Update metrics
            self.prediction_count += 1
            self.total_latency_ms += latency_ms

            # Callback
            if self.on_prediction:
                self.on_prediction(live_pred)

            return live_pred

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get predictor statistics."""
        return {
            'prediction_count': self.prediction_count,
            'avg_latency_ms': self.total_latency_ms / max(1, self.prediction_count),
            'data_bars': len(self.tsla_data) if self.tsla_data is not None else 0,
            'can_predict': self.can_predict(),
        }


def create_live_predictor(
    checkpoint_path: str,
    min_bars: int = 35000
) -> LivePredictor:
    """Factory function for LivePredictor."""
    return LivePredictor(checkpoint_path, min_bars)
