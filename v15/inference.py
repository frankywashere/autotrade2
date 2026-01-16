"""
V15 Inference Module - Make predictions with trained models.

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
from .features.extractor import extract_all_features, get_feature_names
from .data.resampler import resample_with_partial
from .config import TIMEFRAMES, STANDARD_WINDOWS, TOTAL_FEATURES
from .exceptions import ModelError, FeatureExtractionError

logger = logging.getLogger(__name__)


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


class Predictor:
    """
    Make predictions using a trained V15 model.

    Handles:
    - Model loading
    - Feature extraction
    - Batch prediction
    - Result formatting
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

    @classmethod
    def load(cls, checkpoint_path: str, device: str = 'auto') -> 'Predictor':
        """Load predictor from checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise ModelError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location='cpu')

        # Create model
        model = create_model()
        model.load_state_dict(checkpoint['model_state_dict'])

        # Get feature names
        feature_names = checkpoint.get('feature_names', get_feature_names())

        return cls(model, feature_names, device)

    @torch.no_grad()
    def predict_features(self, features: Dict[str, float]) -> Prediction:
        """Make prediction from pre-extracted features."""
        # Convert to tensor
        feature_array = np.array([
            features.get(name, 0.0) for name in self.feature_names
        ], dtype=np.float32)

        x = torch.from_numpy(feature_array).unsqueeze(0).to(self.device)

        # Forward pass
        outputs = self.model(x, validate=True)

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
            best_window=50,  # Will be updated
        )

    def predict(
        self,
        tsla_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Prediction:
        """
        Make prediction from raw market data.

        Args:
            tsla_df: TSLA OHLCV DataFrame
            spy_df: SPY OHLCV DataFrame
            vix_df: VIX OHLCV DataFrame
            timestamp: Timestamp for prediction (default: last bar)

        Returns:
            Prediction object with all outputs
        """
        from v7.core.channel import detect_channels_multi_window, select_best_channel

        if timestamp is None:
            timestamp = tsla_df.index[-1]

        # Detect channels
        channels = detect_channels_multi_window(tsla_df, windows=STANDARD_WINDOWS)
        best_channel, best_window = select_best_channel(channels)

        # Extract features
        features = extract_all_features(
            tsla_df=tsla_df,
            spy_df=spy_df,
            vix_df=vix_df,
            timestamp=timestamp,
            channels_by_window=channels,
            validate=True
        )

        # Make prediction
        prediction = self.predict_features(features)
        prediction.timestamp = timestamp
        prediction.best_window = best_window

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
    vix_df: pd.DataFrame
) -> Prediction:
    """
    Quick prediction without keeping predictor in memory.

    For one-off predictions. Use Predictor class for repeated predictions.
    """
    predictor = Predictor.load(checkpoint_path)
    return predictor.predict(tsla_df, spy_df, vix_df)
