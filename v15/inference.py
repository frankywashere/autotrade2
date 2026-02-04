"""
V15 Inference Module - Make predictions with trained models.

Supports:
- Partial bar feature extraction (critical for live trading)
- Multi-timeframe feature extraction via extract_all_tf_features()
- TF-prefixed feature structure (see config.py TOTAL_FEATURES for current count)

Usage:
    from v15.inference import Predictor

    predictor = Predictor.load('checkpoints/best.pt')
    predictions = predictor.predict(tsla_df, spy_df, vix_df)
"""
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import logging

from .models import V15Model, create_model
from .features.tf_extractor import (
    extract_all_tf_features,
    get_tf_feature_names,
    get_tf_feature_count,
)
from .config import TIMEFRAMES, STANDARD_WINDOWS, TOTAL_FEATURES
from .exceptions import ModelError, FeatureExtractionError

logger = logging.getLogger(__name__)


@dataclass
class PerTFPrediction:
    """Per-timeframe prediction breakdown."""
    duration_mean: float
    duration_std: float
    confidence: float
    best_window: int  # Heuristic window for this TF


@dataclass
class Prediction:
    """Single prediction result."""
    timestamp: pd.Timestamp
    duration_mean: float
    duration_std: float
    direction: str  # 'up' or 'down'
    direction_prob: float
    new_channel: str  # 'bear', 'sideways', 'bull'
    new_channel_probs: Dict[str, float]
    confidence: float
    best_window: int
    # Learned window selection fields (optional)
    learned_window: Optional[int] = None  # Window selected by model
    learned_window_probs: Optional[Dict[int, float]] = None  # Probabilities for each window
    used_learned_selection: bool = False  # Whether learned selection was used
    # Per-timeframe predictions (optional)
    per_tf_predictions: Optional[Dict[str, PerTFPrediction]] = None
    # Structure: {'5min': PerTFPrediction(...), '15min': PerTFPrediction(...), ...}


class Predictor:
    """
    Make predictions using a trained V15 model.

    Handles:
    - Model loading
    - Feature extraction
    - Batch prediction
    - Result formatting
    - Learned window selection (when model has window_selector head)

    When the model was trained with learned window selection
    (use_window_selector=True), the model predicts which of the 8 windows
    is optimal. During inference, this learned selection is used instead
    of the heuristic best_window from channel detection.
    """

    def __init__(
        self,
        model: V15Model,
        feature_names: List[str],
        device: str = 'auto'
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.feature_names = feature_names

        # Check if model has learned window selection
        self._has_learned_window_selection = self._detect_window_selector()
        if self._has_learned_window_selection:
            logger.info("Model has learned window selection - will use model predictions for window choice")
        else:
            logger.debug("Model does not have learned window selection - using heuristic best_window")

    def _detect_window_selector(self) -> bool:
        """
        Detect if the model has a learned window selector.

        Checks for window_selector head in the model's state_dict or
        via the has_window_selector() method.

        Returns:
            True if model has learned window selection capability
        """
        # Method 1: Check via model method
        if hasattr(self.model, 'has_window_selector'):
            return self.model.has_window_selector()

        # Method 2: Check state_dict for window_selector parameters
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            if 'window_selector' in key:
                return True

        return False

    @property
    def has_learned_window_selection(self) -> bool:
        """Whether this predictor uses learned window selection."""
        return self._has_learned_window_selection

    @classmethod
    def load(cls, checkpoint_path: str, device: str = 'auto') -> 'Predictor':
        """
        Load predictor from checkpoint.

        Automatically detects if the model was trained with learned window
        selection by checking for window_selector keys in the state_dict.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use ('auto', 'cuda', 'cpu', etc.)

        Returns:
            Predictor instance with model loaded
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise ModelError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # Check if model was trained with window selector
        state_dict = checkpoint['model_state_dict']
        has_window_selector = any('window_selector' in key for key in state_dict.keys())

        # Get config from checkpoint or use defaults
        model_config = checkpoint.get('config', {})

        # Detect actual input_dim from state_dict (ground truth) instead of stored config
        # The config may have stale values (e.g., 14190) while weights are 14840
        if 'feature_weights.weights' in state_dict:
            input_dim = state_dict['feature_weights.weights'].shape[0]
            logger.info(f"Detected input_dim={input_dim} from checkpoint weights")
        else:
            # Fallback to config if no feature_weights in checkpoint
            input_dim = model_config.get('total_features', TOTAL_FEATURES)
            logger.info(f"Using input_dim={input_dim} from checkpoint config")

        # Create model with appropriate configuration
        config = {
            'input_dim': input_dim,
            'use_window_selector': has_window_selector,
            'num_windows': model_config.get('num_windows', 8),
        }

        model = create_model(config)

        # Load state dict with graceful handling of missing per_tf_heads weights
        # Older checkpoints may not have per_tf_heads - use strict=False and warn
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Check if per_tf_heads weights are missing (expected for older checkpoints)
        per_tf_missing = [k for k in missing_keys if 'per_tf_heads' in k]
        other_missing = [k for k in missing_keys if 'per_tf_heads' not in k]

        if per_tf_missing:
            logger.warning(
                f"Checkpoint missing per_tf_heads weights ({len(per_tf_missing)} keys) - "
                "per-TF predictions will be untrained"
            )

        # Warn about any other missing keys (these might be actual problems)
        if other_missing:
            logger.warning(f"Checkpoint missing unexpected keys: {other_missing}")

        if unexpected_keys:
            logger.warning(f"Checkpoint has unexpected keys: {unexpected_keys}")

        # Get feature names from checkpoint (must match training data exactly)
        feature_names = checkpoint.get('feature_names')
        if feature_names is None:
            logger.warning("Checkpoint has no feature_names — feature importance "
                           "and name-based lookups will be unavailable. "
                           "Patch the checkpoint or retrain with updated code.")

        logger.info(f"Loaded model from {path}")
        if has_window_selector:
            logger.info("Model includes learned window selection")

        return cls(model, feature_names, device)

    @torch.no_grad()
    def predict_features(self, features: Dict[str, float]) -> Prediction:
        """
        Make prediction from pre-extracted features.

        If the model has learned window selection, this will also output
        the model's predicted optimal window.

        Args:
            features: Dict of feature name -> value

        Returns:
            Prediction object with all outputs
        """
        # Convert to tensor
        feature_array = np.array([
            features.get(name, 0.0) for name in self.feature_names
        ], dtype=np.float32)

        x = torch.from_numpy(feature_array).unsqueeze(0).to(self.device)

        # Forward pass with hard window selection for inference
        outputs = self.model(
            x,
            validate=True,
            window_selector_hard=True,  # Use argmax for inference
        )

        # Parse outputs
        duration_mean = outputs['duration_mean'].item()
        duration_std = torch.exp(outputs['duration_log_std']).item()

        direction_prob = torch.sigmoid(outputs['direction_logits']).item()
        direction = 'up' if direction_prob > 0.5 else 'down'

        new_channel_probs = torch.softmax(outputs['new_channel_logits'], dim=-1).squeeze()
        new_channel_idx = new_channel_probs.argmax().item()
        new_channel_names = ['bear', 'sideways', 'bull']
        new_channel = new_channel_names[new_channel_idx]

        confidence = outputs['confidence'].item()

        # Handle learned window selection
        learned_window = None
        learned_window_probs = None
        used_learned_selection = False

        if self._has_learned_window_selection and 'window_selection' in outputs:
            window_sel = outputs['window_selection']
            selected_idx = window_sel['selected_idx'].item()
            learned_window = STANDARD_WINDOWS[selected_idx]
            used_learned_selection = True

            # Get probabilities for all windows
            probs = window_sel['probs'].squeeze()
            learned_window_probs = {
                STANDARD_WINDOWS[i]: probs[i].item()
                for i in range(len(STANDARD_WINDOWS))
            }

            logger.info(
                f"Learned window selection: window={learned_window} "
                f"(idx={selected_idx}, prob={probs[selected_idx].item():.3f})"
            )

        return Prediction(
            timestamp=pd.Timestamp.now(),
            duration_mean=duration_mean,
            duration_std=duration_std,
            direction=direction,
            direction_prob=direction_prob,
            new_channel=new_channel,
            new_channel_probs={
                name: new_channel_probs[i].item()
                for i, name in enumerate(new_channel_names)
            },
            confidence=confidence,
            best_window=50,  # Will be updated by caller
            learned_window=learned_window,
            learned_window_probs=learned_window_probs,
            used_learned_selection=used_learned_selection,
        )

    @torch.no_grad()
    def predict_features_with_per_tf(
        self,
        features: Dict[str, float],
        heuristic_windows_by_tf: Optional[Dict[str, int]] = None,
    ) -> Prediction:
        """
        Make prediction with per-timeframe breakdown.

        This method returns both the aggregated prediction and per-TF
        predictions for duration and confidence.

        Args:
            features: Dict of feature name -> value
            heuristic_windows_by_tf: Optional dict mapping TF name -> best window
                                     for that TF from channel detection

        Returns:
            Prediction object with per_tf_predictions populated
        """
        # Convert to tensor
        feature_array = np.array([
            features.get(name, 0.0) for name in self.feature_names
        ], dtype=np.float32)

        x = torch.from_numpy(feature_array).unsqueeze(0).to(self.device)

        # Forward pass with per-TF outputs
        outputs, per_tf_outputs = self.model.forward_with_per_tf(
            x,
            validate=True,
            window_selector_hard=True,
        )

        # Parse aggregated outputs (same as predict_features)
        duration_mean = outputs['duration_mean'].item()
        duration_std = torch.exp(outputs['duration_log_std']).item()

        direction_prob = torch.sigmoid(outputs['direction_logits']).item()
        direction = 'up' if direction_prob > 0.5 else 'down'

        new_channel_probs = torch.softmax(outputs['new_channel_logits'], dim=-1).squeeze()
        new_channel_idx = new_channel_probs.argmax().item()
        new_channel_names = ['bear', 'sideways', 'bull']
        new_channel = new_channel_names[new_channel_idx]

        confidence = outputs['confidence'].item()

        # Handle learned window selection
        learned_window = None
        learned_window_probs = None
        used_learned_selection = False

        if self._has_learned_window_selection and 'window_selection' in outputs:
            window_sel = outputs['window_selection']
            selected_idx = window_sel['selected_idx'].item()
            learned_window = STANDARD_WINDOWS[selected_idx]
            used_learned_selection = True

            probs = window_sel['probs'].squeeze()
            learned_window_probs = {
                STANDARD_WINDOWS[i]: probs[i].item()
                for i in range(len(STANDARD_WINDOWS))
            }

        # Parse per-TF outputs
        per_tf_predictions = {}
        for i, tf_name in enumerate(TIMEFRAMES):
            tf_duration_mean = per_tf_outputs['duration_mean'][0, i].item()
            tf_duration_std = torch.exp(per_tf_outputs['duration_log_std'][0, i]).item()
            tf_confidence = per_tf_outputs['confidence'][0, i].item()

            # Get heuristic window for this TF (default to 50 if not provided)
            tf_best_window = 50
            if heuristic_windows_by_tf and tf_name in heuristic_windows_by_tf:
                tf_best_window = heuristic_windows_by_tf[tf_name]

            per_tf_predictions[tf_name] = PerTFPrediction(
                duration_mean=tf_duration_mean,
                duration_std=tf_duration_std,
                confidence=tf_confidence,
                best_window=tf_best_window,
            )

        return Prediction(
            timestamp=pd.Timestamp.now(),
            duration_mean=duration_mean,
            duration_std=duration_std,
            direction=direction,
            direction_prob=direction_prob,
            new_channel=new_channel,
            new_channel_probs={
                name: new_channel_probs[i].item()
                for i, name in enumerate(new_channel_names)
            },
            confidence=confidence,
            best_window=50,  # Will be updated by caller
            learned_window=learned_window,
            learned_window_probs=learned_window_probs,
            used_learned_selection=used_learned_selection,
            per_tf_predictions=per_tf_predictions,
        )

    def predict(
        self,
        tsla_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None,
        source_bar_count: Optional[int] = None,
        channel_history_by_tf: Optional[Dict[str, Dict]] = None,
        native_bars_by_tf: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    ) -> Prediction:
        """
        Make prediction from raw market data with partial bar support.

        This method uses the new TF-aware feature extraction that:
        - Resamples to all 10 timeframes keeping partial bars
        - Detects channels at all 8 windows per timeframe
        - Includes bar_completion_pct features for partial bar awareness

        Window Selection:
        - If model has learned window selection, uses model's predicted window
        - Otherwise, falls back to heuristic best_window from channel detection
        - Both values are returned in the Prediction object for comparison

        Args:
            tsla_df: TSLA OHLCV DataFrame (5-min base data)
            spy_df: SPY OHLCV DataFrame (5-min base data)
            vix_df: VIX OHLCV DataFrame (5-min base data)
            timestamp: Timestamp for prediction (default: last bar)
            source_bar_count: Number of 5min bars for partial bar calculation.
                             If None, uses len(tsla_df). This is critical for
                             accurate bar_completion_pct during live trading.
            channel_history_by_tf: Optional dict mapping TF -> {'tsla': [...], 'spy': [...]}
                                  for channel history features.
            native_bars_by_tf: Optional pre-fetched native TF bars from yfinance.
                              Format: {'tsla': {'daily': df, ...}, 'spy': {...}, 'vix': {...}}

        Returns:
            Prediction object with all outputs including:
            - best_window: Heuristic window (from channel detection) or learned window
            - learned_window: Model's predicted window (if learned selection enabled)
            - learned_window_probs: Probabilities for each window
            - used_learned_selection: Whether learned selection was used
        """
        from v15.core.channel import detect_channels_multi_window, select_best_channel

        if timestamp is None:
            timestamp = tsla_df.index[-1]

        # Use source_bar_count for accurate partial bar calculation
        if source_bar_count is None:
            source_bar_count = len(tsla_df)

        # Detect channels on 5-min data for heuristic best_window selection
        channels = detect_channels_multi_window(tsla_df, windows=STANDARD_WINDOWS)
        heuristic_channel, heuristic_window = select_best_channel(channels)
        heuristic_window = heuristic_window if heuristic_window is not None else 50

        # Extract all TF features with partial bar support
        # This resamples to all 10 TFs and extracts ~7,880 features
        features = extract_all_tf_features(
            tsla_df=tsla_df,
            spy_df=spy_df,
            vix_df=vix_df,
            timestamp=timestamp,
            channel_history_by_tf=channel_history_by_tf,
            source_bar_count=source_bar_count,
            include_bar_metadata=True,  # Include bar_completion_pct features
            native_bars_by_tf=native_bars_by_tf,
        )

        # Make prediction
        prediction = self.predict_features(features)
        prediction.timestamp = timestamp

        # Determine which window to use
        if prediction.used_learned_selection and prediction.learned_window is not None:
            # Use learned window selection
            prediction.best_window = prediction.learned_window

            # Log comparison between heuristic and learned selection
            if prediction.learned_window != heuristic_window:
                logger.info(
                    f"Window selection: learned={prediction.learned_window} vs "
                    f"heuristic={heuristic_window} (using learned)"
                )
            else:
                logger.debug(
                    f"Window selection: learned and heuristic agree on window={prediction.learned_window}"
                )
        else:
            # Fall back to heuristic
            prediction.best_window = heuristic_window
            logger.debug(f"Window selection: using heuristic window={heuristic_window}")

        return prediction

    def predict_with_per_tf(
        self,
        tsla_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None,
        source_bar_count: Optional[int] = None,
        channel_history_by_tf: Optional[Dict[str, Dict]] = None,
        native_bars_by_tf: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    ) -> Prediction:
        """
        Make prediction with per-timeframe breakdown.

        Same as predict() but also populates per_tf_predictions with
        duration and confidence for each of the 10 timeframes.

        This enables dashboard to show:
        - Which timeframes are most confident
        - How duration estimates vary across TFs
        - Per-TF best windows from channel detection

        Args:
            tsla_df: TSLA OHLCV DataFrame (5-min base data)
            spy_df: SPY OHLCV DataFrame (5-min base data)
            vix_df: VIX OHLCV DataFrame (5-min base data)
            timestamp: Timestamp for prediction (default: last bar)
            source_bar_count: Number of 5min bars for partial bar calculation
            channel_history_by_tf: Optional dict mapping TF -> {'tsla': [...], 'spy': [...]}
            native_bars_by_tf: Optional pre-fetched native TF bars from yfinance.
                              Format: {'tsla': {'daily': df, ...}, 'spy': {...}, 'vix': {...}}

        Returns:
            Prediction object with per_tf_predictions populated:
            - per_tf_predictions: Dict[str, PerTFPrediction] mapping TF name to predictions
              Each PerTFPrediction has: duration_mean, duration_std, confidence, best_window
        """
        from v15.core.channel import detect_channels_multi_window, select_best_channel
        from v15.features.tf_extractor import resample_to_timeframe

        if timestamp is None:
            timestamp = tsla_df.index[-1]

        if source_bar_count is None:
            source_bar_count = len(tsla_df)

        # Detect channels on 5-min data for overall heuristic best_window
        channels = detect_channels_multi_window(tsla_df, windows=STANDARD_WINDOWS)
        heuristic_channel, heuristic_window = select_best_channel(channels)
        heuristic_window = heuristic_window if heuristic_window is not None else 50

        # Detect best window per timeframe for per-TF breakdown
        heuristic_windows_by_tf = {}
        for tf_name in TIMEFRAMES:
            try:
                # Use native bars for channel detection if available
                if (native_bars_by_tf
                        and native_bars_by_tf.get('tsla', {}).get(tf_name) is not None
                        and len(native_bars_by_tf['tsla'][tf_name]) >= 20):
                    tf_df = native_bars_by_tf['tsla'][tf_name]
                else:
                    tf_df = resample_to_timeframe(tsla_df, tf_name)
                if len(tf_df) >= 20:  # Need enough bars for channel detection
                    tf_channels = detect_channels_multi_window(tf_df, windows=STANDARD_WINDOWS)
                    _, tf_window = select_best_channel(tf_channels)
                    heuristic_windows_by_tf[tf_name] = tf_window if tf_window else 50
                else:
                    heuristic_windows_by_tf[tf_name] = 50
            except Exception as e:
                logger.debug(f"Could not detect channel for {tf_name}: {e}")
                heuristic_windows_by_tf[tf_name] = 50

        # Extract all TF features
        features = extract_all_tf_features(
            tsla_df=tsla_df,
            spy_df=spy_df,
            vix_df=vix_df,
            timestamp=timestamp,
            channel_history_by_tf=channel_history_by_tf,
            source_bar_count=source_bar_count,
            include_bar_metadata=True,
            native_bars_by_tf=native_bars_by_tf,
        )

        # Make prediction with per-TF breakdown
        prediction = self.predict_features_with_per_tf(
            features,
            heuristic_windows_by_tf=heuristic_windows_by_tf,
        )
        prediction.timestamp = timestamp

        # Determine which window to use for aggregated prediction
        if prediction.used_learned_selection and prediction.learned_window is not None:
            prediction.best_window = prediction.learned_window
            if prediction.learned_window != heuristic_window:
                logger.info(
                    f"Window selection: learned={prediction.learned_window} vs "
                    f"heuristic={heuristic_window} (using learned)"
                )
        else:
            prediction.best_window = heuristic_window

        return prediction

    def predict_batch(
        self,
        features_list: List[Dict[str, float]]
    ) -> List[Prediction]:
        """Make predictions on a batch of feature dicts."""
        predictions = []
        for features in features_list:
            pred = self.predict_features(features)
            predictions.append(pred)
        return predictions


def quick_predict(
    checkpoint_path: str,
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    source_bar_count: Optional[int] = None,
) -> Prediction:
    """
    Quick prediction without keeping predictor in memory.

    For one-off predictions. Use Predictor class for repeated predictions.

    Args:
        checkpoint_path: Path to model checkpoint
        tsla_df: TSLA OHLCV DataFrame (5-min base data)
        spy_df: SPY OHLCV DataFrame (5-min base data)
        vix_df: VIX OHLCV DataFrame (5-min base data)
        source_bar_count: Number of 5min bars for partial bar calculation.
                         If None, uses len(tsla_df).

    Returns:
        Prediction object
    """
    predictor = Predictor.load(checkpoint_path)
    return predictor.predict(
        tsla_df, spy_df, vix_df,
        source_bar_count=source_bar_count
    )
