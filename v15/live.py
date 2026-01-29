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
from .config import TIMEFRAMES, STANDARD_WINDOWS, BARS_PER_TF, BREAK_DETECTION
from .exceptions import V15Error
from .core.channel import detect_channel, Channel
from .core.break_scanner import scan_for_break, BreakResult, compute_durability_from_result
from .data.resampler import resample_with_partial
from .features.channel_history import channel_to_history_dict

logger = logging.getLogger(__name__)


@dataclass
class LivePrediction:
    """
    Live prediction with additional metadata.

    Includes partial bar information critical for live trading:
    - bar_completion_pct shows how complete each TF's current bar is
    - This helps traders understand data freshness at each timeframe

    Also includes learned window selection info when model supports it:
    - learned_window: Model's predicted optimal window
    - learned_window_probs: Probabilities for each window
    - used_learned_selection: Whether learned selection was used
    """
    prediction: Prediction
    data_timestamp: pd.Timestamp
    prediction_time: datetime
    latency_ms: float
    channel_valid: bool
    source_bar_count: int = 0  # Number of 5-min bars used
    bar_completion_by_tf: Dict[str, float] = field(default_factory=dict)  # TF -> completion %
    # Learned window selection fields
    used_learned_selection: bool = False
    learned_window: Optional[int] = None
    learned_window_probs: Optional[Dict[int, float]] = None


class LivePredictor:
    """
    Real-time predictor for live trading with partial bar support.

    Features:
    - Maintains rolling data window with proper partial bar tracking
    - Tracks channel history per timeframe for history features
    - Computes accurate bar_completion_pct based on position within TF bars
    - Provides prediction latency metrics
    - Supports callbacks for new predictions
    - Learned window selection (when model supports it)

    The key improvement is tracking source_bar_count accurately, which determines
    bar_completion_pct for each timeframe. During live trading, this ensures the
    model knows how "complete" each TF bar is.

    Learned Window Selection:
    When the model was trained with use_window_selector=True, the model predicts
    which of the 8 windows is optimal. During live inference:
    - The model's predicted window is used instead of heuristic selection
    - The LivePrediction includes learned_window, learned_window_probs, and
      used_learned_selection fields
    - Logging shows which window was selected and the selection confidence
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

        # Log learned window selection status
        if self.predictor.has_learned_window_selection:
            logger.info(
                f"LivePredictor initialized with {self.min_bars} min bars "
                "(model has learned window selection)"
            )
        else:
            logger.info(f"LivePredictor initialized with {self.min_bars} min bars")

    @property
    def has_learned_window_selection(self) -> bool:
        """Whether this predictor uses learned window selection."""
        return self.predictor.has_learned_window_selection

    @property
    def model(self):
        """Access the underlying model (delegates to Predictor)."""
        return self.predictor.model

    @property
    def feature_names(self) -> list:
        """Access feature names (delegates to Predictor)."""
        return self.predictor.feature_names

    @property
    def device(self):
        """Access device (delegates to Predictor)."""
        return self.predictor.device

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

        If model has learned window selection:
        - Uses model's predicted window instead of heuristic
        - Includes learned_window, learned_window_probs in output
        - Logs which window was selected

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

            # Create LivePrediction with learned window selection info
            live_pred = LivePrediction(
                prediction=prediction,
                data_timestamp=self.tsla_data.index[-1],
                prediction_time=datetime.now(),
                latency_ms=latency_ms,
                channel_valid=True,
                source_bar_count=self.total_bars_received,
                bar_completion_by_tf=bar_completion,
                # Learned window selection fields from prediction
                used_learned_selection=prediction.used_learned_selection,
                learned_window=prediction.learned_window,
                learned_window_probs=prediction.learned_window_probs,
            )

            # Log learned window selection if used
            if prediction.used_learned_selection:
                logger.debug(
                    f"Live prediction using learned window={prediction.learned_window} "
                    f"(best_window={prediction.best_window})"
                )

            # Check for channel transitions and update history (NEW)
            if self.track_channel_history:
                self._check_channel_transitions(self.tsla_data.index[-1])

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

    def predict_with_per_tf(self) -> Optional[LivePrediction]:
        """
        Make prediction with per-timeframe breakdown using channel history.

        Same as predict() but uses predictor.predict_with_per_tf() for
        per-TF duration and confidence breakdown. This is the method
        the dashboard should use.

        Returns:
            LivePrediction with prediction.per_tf_predictions populated,
            or None if not enough data.
        """
        if not self.can_predict():
            logger.warning(
                f"Not enough data: "
                f"{len(self.tsla_data) if self.tsla_data is not None else 0}/{self.min_bars}"
            )
            return None

        start_time = time.perf_counter()

        try:
            # Use VIX or create dummy
            vix = self.vix_data if self.vix_data is not None else pd.DataFrame({
                'open': [20.0], 'high': [20.0], 'low': [20.0], 'close': [20.0]
            }, index=[self.tsla_data.index[-1]])

            # Make prediction with per-TF breakdown + channel history
            prediction = self.predictor.predict_with_per_tf(
                self.tsla_data,
                self.spy_data,
                vix,
                source_bar_count=self.total_bars_received,
                channel_history_by_tf=self.channel_history_by_tf if self.track_channel_history else None,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Compute bar completion for each TF
            bar_completion = self._compute_bar_completion_by_tf()

            # Create LivePrediction with learned window selection info
            live_pred = LivePrediction(
                prediction=prediction,
                data_timestamp=self.tsla_data.index[-1],
                prediction_time=datetime.now(),
                latency_ms=latency_ms,
                channel_valid=True,
                source_bar_count=self.total_bars_received,
                bar_completion_by_tf=bar_completion,
                used_learned_selection=prediction.used_learned_selection,
                learned_window=prediction.learned_window,
                learned_window_probs=prediction.learned_window_probs,
            )

            # Log learned window selection if used
            if prediction.used_learned_selection:
                logger.debug(
                    f"Live prediction (per-TF) using learned window={prediction.learned_window} "
                    f"(best_window={prediction.best_window})"
                )

            # Check for channel transitions and update history
            if self.track_channel_history:
                self._check_channel_transitions(self.tsla_data.index[-1])

            # Update metrics
            self.prediction_count += 1
            self.total_latency_ms += latency_ms

            # Callback
            if self.on_prediction:
                self.on_prediction(live_pred)

            return live_pred

        except Exception as e:
            logger.error(f"Prediction (per-TF) failed: {e}")
            return None

    def _check_channel_transitions(self, current_ts: pd.Timestamp) -> None:
        """
        Check all timeframes for channel breaks and update history.

        Called after each prediction to maintain real-time channel tracking.
        This enables the 670 channel history features to be populated with real data.
        """
        # Only check transitions periodically (not every single prediction)
        # to reduce computational overhead
        if self.prediction_count % 10 != 0:
            return

        for tf_name in TIMEFRAMES:
            for asset in ['tsla', 'spy']:
                # Check if current channel has broken permanently
                break_info = self._detect_channel_break(tf_name, asset, current_ts)

                if break_info is not None:
                    # Channel broke - extract metrics and update history
                    self._update_channel_history(tf_name, asset, break_info, current_ts)

    def _detect_channel_break(
        self,
        tf_name: str,
        asset: str,
        current_ts: pd.Timestamp
    ) -> Optional[Dict]:
        """
        Detect if current channel has broken permanently using real break_scanner.

        Uses actual channel detection and break scanning from v15.core modules
        to detect permanent breaks with full metrics.

        Args:
            tf_name: Timeframe name (e.g., '5min', '1h')
            asset: 'tsla' or 'spy'
            current_ts: Current timestamp

        Returns:
            Dict with channel metrics and break info if permanent break detected, None otherwise
        """
        try:
            # Get current data for the asset
            data = self.tsla_data if asset == 'tsla' else self.spy_data
            if data is None or len(data) < 100:
                return None

            # Get bars_per_tf for this timeframe
            bars_per_tf = BARS_PER_TF.get(tf_name, 1)

            # Need enough data to resample and detect channels
            # Minimum: window * bars_per_tf + forward scan buffer
            min_window = STANDARD_WINDOWS[0]  # 10
            max_window = STANDARD_WINDOWS[-1]  # 80
            min_required_5min_bars = max_window * bars_per_tf + 100  # Buffer for forward scan

            if len(data) < min_required_5min_bars:
                return None

            # Resample to target timeframe
            # Use resample_with_partial to handle live partial bars
            try:
                resampled_df, _ = resample_with_partial(data, tf_name)
            except Exception as e:
                logger.debug(f"Resampling failed for {asset}/{tf_name}: {e}")
                return None

            if resampled_df is None or len(resampled_df) < max_window + 20:
                return None

            # Track the previous channel state for this asset/tf combo
            state_key = f"{asset}_{tf_name}"
            if not hasattr(self, '_channel_states'):
                self._channel_states: Dict[str, Dict] = {}

            prev_state = self._channel_states.get(state_key)

            # Detect current channel using the default window (50)
            # This gives us a channel at the current position
            default_window = 50
            if len(resampled_df) < default_window + 2:
                return None

            current_channel = detect_channel(
                resampled_df,
                window=default_window,
                std_multiplier=2.0,
                touch_threshold=0.10,
                min_cycles=1
            )

            if not current_channel.valid:
                # No valid channel - clear state and return
                self._channel_states[state_key] = None
                return None

            # If we have a previous channel, check for permanent break
            if prev_state is not None and prev_state.get('channel') is not None:
                prev_channel: Channel = prev_state['channel']
                prev_channel_end_idx = prev_state['end_idx']

                # Get forward data from where previous channel ended
                # We need high, low, close for the break scanner
                forward_start = prev_channel_end_idx
                if forward_start < 0 or forward_start >= len(resampled_df) - 1:
                    # Reset state - previous position is no longer valid
                    self._channel_states[state_key] = {
                        'channel': current_channel,
                        'end_idx': len(resampled_df) - 1,
                        'end_ts': current_ts
                    }
                    return None

                # Get forward OHLC arrays
                forward_high = resampled_df['high'].values[forward_start:]
                forward_low = resampled_df['low'].values[forward_start:]
                forward_close = resampled_df['close'].values[forward_start:]

                if len(forward_high) < 5:  # Need at least some forward data
                    # Update state with current channel
                    self._channel_states[state_key] = {
                        'channel': current_channel,
                        'end_idx': len(resampled_df) - 1,
                        'end_ts': current_ts
                    }
                    return None

                # Run break scanner on the previous channel
                try:
                    break_result: BreakResult = scan_for_break(
                        channel=prev_channel,
                        forward_high=forward_high,
                        forward_low=forward_low,
                        forward_close=forward_close,
                        max_scan_bars=min(100, len(forward_high)),  # Limit scan for live
                        return_threshold_bars=BREAK_DETECTION.get('return_threshold_bars', 10),
                        min_break_magnitude=BREAK_DETECTION.get('min_break_magnitude', 0.5)
                    )
                except Exception as e:
                    logger.debug(f"Break scan failed for {asset}/{tf_name}: {e}")
                    # Update state with current channel
                    self._channel_states[state_key] = {
                        'channel': current_channel,
                        'end_idx': len(resampled_df) - 1,
                        'end_ts': current_ts
                    }
                    return None

                # Check if we have a permanent break
                if break_result.permanent_break_direction >= 0:
                    # Permanent break detected!
                    # Compute durability metrics
                    false_break_count, false_break_rate, durability_score = compute_durability_from_result(break_result)

                    # Calculate exit metrics from the break result
                    exit_events = break_result.all_exit_events or []
                    exit_count = len(exit_events)
                    exit_magnitudes = [e.magnitude for e in exit_events]
                    avg_exit_magnitude = np.mean(exit_magnitudes) if exit_magnitudes else 0.0
                    avg_bars_outside = (
                        np.mean([e.bars_outside for e in exit_events if e.returned])
                        if any(e.returned for e in exit_events)
                        else 0.0
                    )

                    # Build the break info dict matching ChannelHistoryEntry format
                    break_info = {
                        'end_timestamp': int(current_ts.value / 1e6),  # Milliseconds
                        'duration': prev_channel.window,  # Channel duration in TF bars
                        'slope': prev_channel.slope,
                        'direction': int(prev_channel.direction),  # 0=BEAR, 1=SIDEWAYS, 2=BULL
                        'break_direction': break_result.permanent_break_direction,  # 0=DOWN, 1=UP
                        'r_squared': prev_channel.r_squared,
                        'bounce_count': prev_channel.bounce_count,
                        # Exit metrics
                        'exit_count': exit_count,
                        'avg_exit_magnitude': float(avg_exit_magnitude),
                        'avg_bars_outside': float(avg_bars_outside),
                        'exit_return_rate': break_result.exit_return_rate,
                        'durability_score': durability_score,
                        'false_break_count': false_break_count,
                        # Store the channel object for channel_to_history_dict
                        '_channel': prev_channel,
                        '_break_result': break_result,
                    }

                    # Update state with current channel (the new channel after break)
                    self._channel_states[state_key] = {
                        'channel': current_channel,
                        'end_idx': len(resampled_df) - 1,
                        'end_ts': current_ts
                    }

                    return break_info

            # No permanent break yet - update state with current channel
            self._channel_states[state_key] = {
                'channel': current_channel,
                'end_idx': len(resampled_df) - 1,
                'end_ts': current_ts
            }
            return None

        except Exception as e:
            logger.debug(f"Channel break detection failed for {asset}/{tf_name}: {e}")
            return None

    def _update_channel_history(
        self,
        tf_name: str,
        asset: str,
        break_info: Dict,
        current_ts: pd.Timestamp
    ) -> None:
        """
        Update channel history with a new broken channel using proper history dict format.

        Uses channel_to_history_dict() from v15.features.channel_history to ensure
        the history entry has all required fields for the 67 channel history features.

        Maintains a rolling window of the last 5 channels per timeframe per asset.

        Args:
            tf_name: Timeframe name (e.g., '5min')
            asset: 'tsla' or 'spy'
            break_info: Dict from _detect_channel_break containing channel and break metrics
            current_ts: Current timestamp
        """
        # Initialize if needed
        if tf_name not in self.channel_history_by_tf:
            self.channel_history_by_tf[tf_name] = {'tsla': [], 'spy': []}

        # Extract the channel object and break result if available
        channel = break_info.get('_channel')
        break_result = break_info.get('_break_result')

        # Build a ChannelLabels-like object for channel_to_history_dict
        # This extracts proper exit metrics from the break result
        class LabelsProxy:
            """Proxy object with ChannelLabels-compatible attributes for exit metrics."""
            def __init__(self, break_result: Optional[BreakResult]):
                if break_result is not None and break_result.all_exit_events:
                    exit_events = break_result.all_exit_events
                    self.exit_bars = [e.bar_index for e in exit_events]
                    self.exit_magnitudes = [e.magnitude for e in exit_events]
                    self.avg_bars_outside = (
                        np.mean([e.bars_outside for e in exit_events if e.returned])
                        if any(e.returned for e in exit_events)
                        else 0.0
                    )
                    self.exit_return_rate = break_result.exit_return_rate
                    self.durability_score = break_info.get('durability_score', 0.0)
                    self.bounces_after_return = break_result.false_break_count
                else:
                    self.exit_bars = []
                    self.exit_magnitudes = []
                    self.avg_bars_outside = 0.0
                    self.exit_return_rate = 0.0
                    self.durability_score = 0.0
                    self.bounces_after_return = 0

        labels_proxy = LabelsProxy(break_result)

        # Use channel_to_history_dict if we have a channel object
        if channel is not None:
            history_entry = channel_to_history_dict(
                channel=channel,
                duration=break_info.get('duration', channel.window),
                break_direction=break_info.get('break_direction', 0),
                labels=labels_proxy
            )
        else:
            # Fallback: build history entry directly from break_info
            # This ensures we always have a valid history entry with all required fields
            history_entry = {
                'duration': break_info.get('duration', 50),
                'slope': break_info.get('slope', 0.0),
                'direction': break_info.get('direction', 1),  # 0=BEAR, 1=SIDEWAYS, 2=BULL
                'break_direction': break_info.get('break_direction', 0),
                'r_squared': break_info.get('r_squared', 0.0),
                'bounce_count': break_info.get('bounce_count', 0),
                # Exit metrics
                'exit_count': break_info.get('exit_count', 0),
                'avg_exit_magnitude': break_info.get('avg_exit_magnitude', 0.0),
                'avg_bars_outside': break_info.get('avg_bars_outside', 0.0),
                'exit_return_rate': break_info.get('exit_return_rate', 0.0),
                'durability_score': break_info.get('durability_score', 0.0),
                'false_break_count': break_info.get('false_break_count', 0),
            }

        # Append to history
        history_list = self.channel_history_by_tf[tf_name][asset]
        history_list.append(history_entry)

        # Keep last 5
        if len(history_list) > 5:
            self.channel_history_by_tf[tf_name][asset] = history_list[-5:]

        logger.debug(
            f"[{current_ts}] Channel transition detected in {tf_name}/{asset}: "
            f"direction={history_entry['direction']}, duration={history_entry['duration']}, "
            f"break_dir={history_entry['break_direction']}, bounce_count={history_entry['bounce_count']}, "
            f"r_squared={history_entry['r_squared']:.3f}, durability={history_entry['durability_score']:.3f}"
        )

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
            'has_learned_window_selection': self.has_learned_window_selection,
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
