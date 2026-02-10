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
from .config import TIMEFRAMES, STANDARD_WINDOWS, TOTAL_FEATURES, HORIZON_GROUPS, TF_TO_HORIZON
from .exceptions import ModelError, FeatureExtractionError
from .signals.bounce_signal import BounceSignalEngine, BounceSignal, SignalStrategy

logger = logging.getLogger(__name__)


@dataclass
class PerTFPrediction:
    """Per-timeframe prediction breakdown."""
    duration_mean: float
    duration_std: float
    direction: str          # 'up' or 'down'
    direction_prob: float   # P(break_up), calibrated if temp scaler available
    confidence: float       # max(direction_prob, 1-direction_prob)
    best_window: int        # Heuristic window for this TF
    next_channel: str       # 'bear', 'sideways', 'bull'
    next_channel_probs: Dict[str, float]  # probabilities for each class


@dataclass
class HorizonSummary:
    """Summary of predictions for a horizon group (short/medium/long)."""
    horizon: str           # 'short', 'medium', 'long'
    direction: str         # majority direction (weighted by confidence)
    avg_confidence: float
    best_tf: str           # highest confidence TF in this horizon
    best_tf_confidence: float


@dataclass
class TradeRecommendation:
    """Trade recommendation for a horizon."""
    horizon: str
    timeframe: str
    direction: str
    confidence: float
    duration_mean: float
    duration_std: float
    score: float           # confidence - uncertainty_penalty


@dataclass
class HorizonConflict:
    """Conflict between two horizon groups."""
    horizon_a: str
    horizon_b: str
    direction_a: str
    direction_b: str


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
    # Horizon-based analysis
    horizon_summaries: Optional[Dict[str, HorizonSummary]] = None
    trade_recommendations: Optional[Dict[str, TradeRecommendation]] = None
    conflicts: Optional[List[HorizonConflict]] = None
    # Bounce signal (buy/sell from channel boundaries)
    bounce_signal: Optional['BounceSignal'] = None


class TemperatureScaler:
    """Post-training temperature scaler for calibrating direction probabilities."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def calibrate(self, logit: float) -> float:
        """Apply temperature scaling to a raw logit and return calibrated probability."""
        import math
        return 1.0 / (1.0 + math.exp(-logit / self.temperature))

    @classmethod
    def load(cls, path: str) -> 'TemperatureScaler':
        """Load temperature scaler from JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(temperature=data['temperature'])


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
        device: str = 'auto',
        temperature_scaler: Optional[TemperatureScaler] = None,
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.feature_names = feature_names
        self._temperature_scaler = temperature_scaler

        # Check if model has learned window selection
        self._has_learned_window_selection = self._detect_window_selector()
        if self._has_learned_window_selection:
            logger.info("Model has learned window selection - will use model predictions for window choice")
        else:
            logger.debug("Model does not have learned window selection - using heuristic best_window")

        # Initialize bounce signal engine
        self._signal_engine = BounceSignalEngine()

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

        # Detect per-TF head version from state dict
        # V2 has 'per_tf_heads.tf_embedding.weight', V1 does not
        has_tf_embedding = any('per_tf_heads.tf_embedding' in k for k in state_dict)
        per_tf_head_version = 2 if has_tf_embedding else 1

        # Detect horizon attention from state dict
        has_horizon_attention = any('horizon_attention' in k for k in state_dict)

        # Create model with appropriate configuration
        config = {
            'input_dim': input_dim,
            'use_window_selector': has_window_selector,
            'num_windows': model_config.get('num_windows', 8),
            'per_tf_head_version': per_tf_head_version,
            'use_horizon_attention': has_horizon_attention,
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

        # Load temperature scaler if available
        temperature_scaler = None
        temp_path = path.parent / 'temperature_calibration.json'
        if temp_path.exists():
            try:
                temperature_scaler = TemperatureScaler.load(str(temp_path))
                logger.info(f"Loaded temperature scaler (T={temperature_scaler.temperature:.4f})")
            except Exception as e:
                logger.warning(f"Failed to load temperature scaler: {e}")

        logger.info(f"Loaded model from {path}")
        if has_window_selector:
            logger.info("Model includes learned window selection")

        return cls(model, feature_names, device, temperature_scaler=temperature_scaler)

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

        confidence = max(direction_prob, 1.0 - direction_prob)

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
            bounce_signal=None,  # No per-TF, so no bounce signal
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

        # Parse per-TF outputs with direction
        per_tf_predictions = {}
        for i, tf_name in enumerate(TIMEFRAMES):
            tf_dur_mean = per_tf_outputs['duration_mean'][0, i].item()
            tf_dur_std = torch.exp(per_tf_outputs['duration_log_std'][0, i]).item()
            tf_dir_logit = per_tf_outputs['direction_logits'][0, i].item()

            if self._temperature_scaler:
                tf_dir_prob = self._temperature_scaler.calibrate(tf_dir_logit)
            else:
                tf_dir_prob = torch.sigmoid(per_tf_outputs['direction_logits'][0, i]).item()

            tf_direction = 'up' if tf_dir_prob > 0.5 else 'down'
            tf_confidence = max(tf_dir_prob, 1.0 - tf_dir_prob)

            # Extract next_channel predictions (3-class: bear/sideways/bull)
            tf_nc_logits = per_tf_outputs.get('new_channel_logits')
            if tf_nc_logits is not None:
                nc_probs = torch.softmax(tf_nc_logits[0, i], dim=0)  # [3]
                nc_probs_dict = {
                    'bear': nc_probs[0].item(),
                    'sideways': nc_probs[1].item(),
                    'bull': nc_probs[2].item(),
                }
                tf_nc_idx = nc_probs.argmax().item()
                tf_nc_name = ['bear', 'sideways', 'bull'][tf_nc_idx]
            else:
                # Fallback if model doesn't have per-TF new_channel head
                nc_probs_dict = {'bear': 0.33, 'sideways': 0.34, 'bull': 0.33}
                tf_nc_name = 'sideways'

            # Get heuristic window for this TF (default to 50 if not provided)
            tf_best_window = 50
            if heuristic_windows_by_tf and tf_name in heuristic_windows_by_tf:
                tf_best_window = heuristic_windows_by_tf[tf_name]

            per_tf_predictions[tf_name] = PerTFPrediction(
                duration_mean=tf_dur_mean,
                duration_std=tf_dur_std,
                direction=tf_direction,
                direction_prob=tf_dir_prob,
                confidence=tf_confidence,
                best_window=tf_best_window,
                next_channel=tf_nc_name,
                next_channel_probs=nc_probs_dict,
            )

        # Aggregated confidence from per-TF
        confidence = max(p.confidence for p in per_tf_predictions.values()) if per_tf_predictions else max(direction_prob, 1 - direction_prob)

        # Compute horizon analysis
        horizon_summaries = self._compute_horizon_summaries(per_tf_predictions)
        trade_recommendations = self._compute_trade_recommendations(per_tf_predictions)
        conflicts = self._detect_conflicts(horizon_summaries)

        # Compute bounce signal (default to most_confident strategy)
        bounce_signal = self._signal_engine.generate_signal(
            per_tf_predictions=per_tf_predictions,
            strategy=SignalStrategy.MOST_CONFIDENT,
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
            horizon_summaries=horizon_summaries,
            trade_recommendations=trade_recommendations,
            conflicts=conflicts,
            bounce_signal=bounce_signal,
        )

    @staticmethod
    def _compute_horizon_summaries(
        per_tf_predictions: Dict[str, PerTFPrediction],
    ) -> Dict[str, HorizonSummary]:
        """Compute horizon summaries from per-TF predictions."""
        summaries = {}
        for horizon, tf_list in HORIZON_GROUPS.items():
            preds = [(tf, per_tf_predictions[tf]) for tf in tf_list if tf in per_tf_predictions]
            if not preds:
                continue

            # Weighted direction vote: sum confidence-weighted votes
            up_weight = sum(p.confidence for _, p in preds if p.direction == 'up')
            down_weight = sum(p.confidence for _, p in preds if p.direction == 'down')
            majority_dir = 'up' if up_weight >= down_weight else 'down'

            avg_conf = sum(p.confidence for _, p in preds) / len(preds)

            # Best TF = highest confidence
            best_tf, best_pred = max(preds, key=lambda x: x[1].confidence)

            summaries[horizon] = HorizonSummary(
                horizon=horizon,
                direction=majority_dir,
                avg_confidence=avg_conf,
                best_tf=best_tf,
                best_tf_confidence=best_pred.confidence,
            )
        return summaries

    @staticmethod
    def _compute_trade_recommendations(
        per_tf_predictions: Dict[str, PerTFPrediction],
    ) -> Dict[str, TradeRecommendation]:
        """Compute best trade recommendation per horizon."""
        recommendations = {}
        for horizon, tf_list in HORIZON_GROUPS.items():
            preds = [(tf, per_tf_predictions[tf]) for tf in tf_list if tf in per_tf_predictions]
            if not preds:
                continue

            # Score = confidence - uncertainty_penalty
            def score(p: PerTFPrediction) -> float:
                uncertainty_penalty = 0.3 * min(p.duration_std / max(p.duration_mean, 1.0), 0.5)
                return p.confidence - uncertainty_penalty

            best_tf, best_pred = max(preds, key=lambda x: score(x[1]))

            recommendations[horizon] = TradeRecommendation(
                horizon=horizon,
                timeframe=best_tf,
                direction=best_pred.direction,
                confidence=best_pred.confidence,
                duration_mean=best_pred.duration_mean,
                duration_std=best_pred.duration_std,
                score=score(best_pred),
            )
        return recommendations

    @staticmethod
    def _detect_conflicts(
        horizon_summaries: Dict[str, HorizonSummary],
    ) -> List[HorizonConflict]:
        """Detect direction conflicts between horizon groups."""
        conflicts = []
        horizons = list(horizon_summaries.keys())
        for i in range(len(horizons)):
            for j in range(i + 1, len(horizons)):
                h_a = horizon_summaries[horizons[i]]
                h_b = horizon_summaries[horizons[j]]
                if h_a.direction != h_b.direction:
                    conflicts.append(HorizonConflict(
                        horizon_a=horizons[i],
                        horizon_b=horizons[j],
                        direction_a=h_a.direction,
                        direction_b=h_b.direction,
                    ))
        return conflicts

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
              Each PerTFPrediction has: duration_mean, duration_std, direction, direction_prob, confidence, best_window
            - horizon_summaries, trade_recommendations, conflicts
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
