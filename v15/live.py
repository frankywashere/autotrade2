"""
V15 Live Trading Integration

Provides real-time prediction updates for live trading systems.

Key features:
- Partial bar support: The current bar is always incomplete during live trading.
  This module maintains proper bar_completion_pct tracking.
- Rolling data window: Maintains a sliding window of historical data
- Channel history tracking: Tracks recent channel transitions for history features
- Lazy label lookups: Uses pre-computed label maps when available for efficiency
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
from pathlib import Path

from .inference import Predictor, Prediction
from .config import TIMEFRAMES, STANDARD_WINDOWS, BARS_PER_TF
from .exceptions import V15Error

logger = logging.getLogger(__name__)


@dataclass
class LivePrediction:
    """
    Live prediction with additional metadata.

    Includes partial bar information critical for live trading:
    - bar_completion_pct shows how complete each TF's current bar is
    - This helps traders understand data freshness at each timeframe
    """
    prediction: Prediction
    data_timestamp: pd.Timestamp
    prediction_time: datetime
    latency_ms: float
    channel_valid: bool
    source_bar_count: int = 0  # Number of 5-min bars used
    bar_completion_by_tf: Dict[str, float] = field(default_factory=dict)  # TF -> completion %


class LivePredictor:
    """
    Real-time predictor for live trading with partial bar support.

    Features:
    - Maintains rolling data window with proper partial bar tracking
    - Tracks channel history per timeframe for history features
    - Computes accurate bar_completion_pct based on position within TF bars
    - Provides prediction latency metrics
    - Supports callbacks for new predictions

    The key improvement is tracking source_bar_count accurately, which determines
    bar_completion_pct for each timeframe. During live trading, this ensures the
    model knows how "complete" each TF bar is.
    """

    def __init__(
        self,
        checkpoint_path: str,
        min_bars: int = 35000,
        on_prediction: Optional[Callable[[LivePrediction], None]] = None,
        track_channel_history: bool = True,
    ):
        self.predictor = Predictor.load(checkpoint_path)
        self.min_bars = min_bars
        self.on_prediction = on_prediction
        self.track_channel_history = track_channel_history

        # Rolling data storage
        self.tsla_data: Optional[pd.DataFrame] = None
        self.spy_data: Optional[pd.DataFrame] = None
        self.vix_data: Optional[pd.DataFrame] = None

        # Track total bars received (not just stored) for accurate completion calc
        self.total_bars_received: int = 0

        # Channel history tracking per TF
        # Format: {tf: {'tsla': [last 5 channel dicts], 'spy': [last 5 channel dicts]}}
        self.channel_history_by_tf: Dict[str, Dict[str, List[Dict]]] = {
            tf: {'tsla': [], 'spy': []} for tf in TIMEFRAMES
        }

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

        IMPORTANT: This increments total_bars_received which is used to calculate
        bar_completion_pct for each timeframe. The position within a TF bar
        (e.g., 22 bars into a daily bar of 78) determines the completion percentage.

        Args:
            tsla_bar: Dict with open, high, low, close, volume
            spy_bar: Dict with open, high, low, close, volume
            vix_bar: Optional VIX data
            timestamp: Bar timestamp
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        # Increment total bars received for accurate partial bar calculation
        self.total_bars_received += 1

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

            # Trim to max size (keep more than min_bars for buffer)
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

    def _compute_bar_completion_by_tf(self) -> Dict[str, float]:
        """
        Compute bar completion percentage for each timeframe.

        This shows how "complete" each TF's current bar is based on
        the position within the bar period.

        Returns:
            Dict mapping TF -> completion percentage (0.0 to 1.0)
        """
        completion = {}
        for tf in TIMEFRAMES:
            bars_per_tf = BARS_PER_TF.get(tf, 1)
            bars_into_current = self.total_bars_received % bars_per_tf
            if bars_into_current == 0 and self.total_bars_received > 0:
                # Exactly at a TF boundary - bar is complete
                completion[tf] = 1.0
            elif bars_per_tf > 0:
                completion[tf] = bars_into_current / bars_per_tf
            else:
                completion[tf] = 1.0
        return completion

    def predict(self) -> Optional[LivePrediction]:
        """
        Make prediction with current data using partial bar support.

        Uses extract_all_tf_features() which:
        - Resamples to all 10 TFs keeping partial bars
        - Includes bar_completion_pct features
        - Detects channels at all 8 windows per TF

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

            # Make prediction with partial bar support
            # source_bar_count is critical for accurate bar_completion_pct
            prediction = self.predictor.predict(
                self.tsla_data,
                self.spy_data,
                vix,
                source_bar_count=self.total_bars_received,
                channel_history_by_tf=self.channel_history_by_tf if self.track_channel_history else None,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Compute bar completion for each TF
            bar_completion = self._compute_bar_completion_by_tf()

            live_pred = LivePrediction(
                prediction=prediction,
                data_timestamp=self.tsla_data.index[-1],
                prediction_time=datetime.now(),
                latency_ms=latency_ms,
                channel_valid=True,
                source_bar_count=self.total_bars_received,
                bar_completion_by_tf=bar_completion,
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
        """Get predictor statistics including partial bar info."""
        bar_completion = self._compute_bar_completion_by_tf() if self.total_bars_received > 0 else {}

        return {
            'prediction_count': self.prediction_count,
            'avg_latency_ms': self.total_latency_ms / max(1, self.prediction_count),
            'data_bars': len(self.tsla_data) if self.tsla_data is not None else 0,
            'total_bars_received': self.total_bars_received,
            'can_predict': self.can_predict(),
            'bar_completion_by_tf': bar_completion,
            'tracking_channel_history': self.track_channel_history,
        }

    def reset_bar_count(self, new_count: int = 0):
        """
        Reset the total_bars_received counter.

        Useful when initializing with historical data that has a known position.

        Args:
            new_count: The new bar count to set (default 0)
        """
        self.total_bars_received = new_count
        logger.info(f"Reset total_bars_received to {new_count}")

    def load_historical_data(
        self,
        tsla_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
        bar_count_offset: Optional[int] = None,
    ):
        """
        Load historical data to initialize the predictor.

        This is useful for starting live trading with a warm cache
        rather than waiting to accumulate min_bars of live data.

        Args:
            tsla_df: Historical TSLA 5-min OHLCV data
            spy_df: Historical SPY 5-min OHLCV data
            vix_df: Optional historical VIX data
            bar_count_offset: If provided, sets total_bars_received to this value.
                             This is important for accurate bar_completion_pct.
                             If None, uses len(tsla_df).
        """
        # Take last min_bars if more data provided
        if len(tsla_df) > self.min_bars:
            self.tsla_data = tsla_df.iloc[-self.min_bars:].copy()
            self.spy_data = spy_df.iloc[-self.min_bars:].copy()
            if vix_df is not None:
                self.vix_data = vix_df.iloc[-self.min_bars:].copy()
        else:
            self.tsla_data = tsla_df.copy()
            self.spy_data = spy_df.copy()
            if vix_df is not None:
                self.vix_data = vix_df.copy()

        # Set bar count for accurate partial bar tracking
        if bar_count_offset is not None:
            self.total_bars_received = bar_count_offset
        else:
            self.total_bars_received = len(tsla_df)

        logger.info(
            f"Loaded {len(self.tsla_data)} bars of historical data. "
            f"total_bars_received={self.total_bars_received}"
        )


def create_live_predictor(
    checkpoint_path: str,
    min_bars: int = 35000,
    track_channel_history: bool = True,
) -> LivePredictor:
    """
    Factory function for LivePredictor.

    Args:
        checkpoint_path: Path to model checkpoint
        min_bars: Minimum 5-min bars required for prediction
        track_channel_history: If True, track channel history for history features

    Returns:
        Configured LivePredictor instance
    """
    return LivePredictor(
        checkpoint_path,
        min_bars,
        track_channel_history=track_channel_history
    )


# =============================================================================
# Two-Pass Labeling Support for Live Inference
# =============================================================================
#
# For LIVE inference, we typically don't need labels (we're predicting the future).
# However, for backtesting or research scenarios where we want to compare predictions
# against actual outcomes, we can use the two-pass labeling system.
#
# The key functions from labels.py are:
# - get_labels_for_position(): O(log N) lazy lookup of pre-computed labels
# - LabeledChannelMap: Pre-computed labels for all channels
#
# Usage pattern for backtesting:
#   1. Run detect_all_channels() once on historical data (PASS 1)
#   2. Run generate_all_labels() once (PASS 2)
#   3. During backtest, use get_labels_for_position() for O(log N) lookups
#
# This is much more efficient than computing labels on-the-fly for each sample.
# =============================================================================

def create_backtest_predictor(
    checkpoint_path: str,
    labeled_map: Optional[Any] = None,
    min_bars: int = 35000,
) -> LivePredictor:
    """
    Create a LivePredictor configured for backtesting with pre-computed labels.

    This is useful for research scenarios where you want to compare model
    predictions against actual channel break outcomes.

    Args:
        checkpoint_path: Path to model checkpoint
        labeled_map: Pre-computed LabeledChannelMap from two-pass labeling
        min_bars: Minimum 5-min bars required for prediction

    Returns:
        LivePredictor configured for backtesting

    Example:
        from v15.labels import detect_all_channels, generate_all_labels

        # PASS 1: Detect all channels (do once)
        channel_map = detect_all_channels(df, verbose=True)

        # PASS 2: Generate labels (do once)
        labeled_map = generate_all_labels(channel_map, verbose=True)

        # Create predictor for backtesting
        predictor = create_backtest_predictor('model.pt', labeled_map)

        # Run backtest with O(log N) label lookups
        for position in backtest_positions:
            prediction = predictor.predict()
            actual_label = get_labels_for_position(labeled_map, df, position, tf, window)
            # Compare prediction vs actual_label
    """
    predictor = LivePredictor(
        checkpoint_path,
        min_bars,
        track_channel_history=True,
    )

    # Store labeled_map for backtesting (optional)
    # This allows the backtest loop to access it
    predictor._labeled_map = labeled_map

    return predictor
