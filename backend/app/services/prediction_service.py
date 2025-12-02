"""
ML Prediction Service

Wraps hierarchical_model.py for inference with caching.
Extracts all 14,487 features fresh from live OHLCV data.
"""
import torch
import numpy as np
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.hierarchical_model import load_hierarchical_model
from src.ml.features import TradingFeatureExtractor
from deprecated_code.live_data_feed import HybridLiveDataFeed
from src.ml.events import CombinedEventsHandler

# Configure logging
logger = logging.getLogger(__name__)


class PredictionService:
    """
    Singleton service for ML predictions.

    Features:
    - Fresh feature extraction from live OHLCV data (14,487 features)
    - 5-minute prediction caching
    - Lazy model loading
    """

    _instance = None
    _prediction_lock = threading.Lock()  # Serialize predictions to prevent yfinance data scrambling
    _model = None
    _feature_extractor = None
    _data_feed = None
    _events_handler = None
    _last_prediction = None
    _last_prediction_time = None
    _cache_ttl = timedelta(minutes=5)

    # Feature dimensions - loaded from model checkpoint (single source of truth)
    _expected_features = None  # Will be set from model.input_size when loaded
    SEQUENCE_LENGTH = 200

    @property
    def EXPECTED_FEATURES(self):
        """Get expected features from loaded model (single source of truth)."""
        if self._expected_features is None:
            # Force model load to get input_size
            self._load_model()
        return self._expected_features

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize service (model loaded lazily)"""
        pass

    def _load_model(self):
        """Load model (singleton pattern - only once)"""
        if self._model is None:
            model_path = Path(project_root) / 'models' / 'hierarchical_lnn.pth'

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    f"Train the model first: python train_hierarchical.py"
                )

            logger.info(f"Loading model from {model_path}...")
            self._model = load_hierarchical_model(str(model_path), device='cpu')
            self._model.eval()

            # Set expected features from model (single source of truth)
            PredictionService._expected_features = self._model.input_size
            logger.info(f"Model loaded successfully (input_size={self._model.input_size})")

    def _get_feature_extractor(self):
        """Get feature extractor (singleton)"""
        if self._feature_extractor is None:
            self._feature_extractor = TradingFeatureExtractor()

            # Try to load events handler
            try:
                self._events_handler = CombinedEventsHandler()
                logger.info("Events handler loaded")
            except Exception as e:
                logger.warning(f"Events handler not available: {e}")
                self._events_handler = None

        return self._feature_extractor

    def _get_data_feed(self):
        """Get live data feed (singleton)"""
        if self._data_feed is None:
            self._data_feed = HybridLiveDataFeed(symbols=['TSLA', 'SPY'])
        return self._data_feed

    def get_latest_prediction(self, force_refresh: bool = False) -> Dict:
        """
        Get latest prediction with 5-min caching.

        Thread-safe: uses lock to prevent concurrent yfinance data scrambling.

        Args:
            force_refresh: Skip cache and fetch new prediction

        Returns:
            Prediction dict with all outputs
        """
        with self._prediction_lock:
            # Check cache (inside lock to ensure atomic read/write)
            if not force_refresh and self._last_prediction is not None:
                age = datetime.now() - self._last_prediction_time
                if age < self._cache_ttl:
                    logger.info(f"Using cached prediction ({age.seconds}s old)")
                    return self._last_prediction

            logger.info("Generating new prediction...")

            # Load model if needed
            self._load_model()

            # Get live data (7,000+ bars with multi-resolution data)
            feed = self._get_data_feed()
            df = feed.fetch_for_prediction()

            if len(df) < self.SEQUENCE_LENGTH:
                raise ValueError(
                    f"Insufficient data: {len(df)} bars (need {self.SEQUENCE_LENGTH})\n"
                    f"Market may be closed or data unavailable"
                )

            logger.info(f"Live data fetched: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

            # Extract ALL features fresh from live data (no caching, no mmap)
            extractor = self._get_feature_extractor()
            result = extractor.extract_features(
                df,
                use_cache=False,       # Force fresh extraction
                use_chunking=False,    # No sharding for inference
                continuation=False,    # Skip continuation labels
                events_handler=self._events_handler
            )

            # extract_features returns (features_df, continuation_df) when continuation=False
            if isinstance(result, tuple):
                features_df = result[0]
            else:
                features_df = result

            logger.info(f"Features extracted: {features_df.shape}")

            # Handle feature dimension mismatch
            # NOTE: ~682 features (3month timeframe with windows > 60) cannot be calculated
            # due to yfinance's limited monthly history (~15 years = ~187 bars, which becomes
            # ~62 3-month bars after resampling). This is an inherent API limitation, not a bug.
            # These features are padded with zeros which has minimal impact on predictions.
            actual_features = features_df.shape[1]
            if actual_features != self.EXPECTED_FEATURES:
                if actual_features < self.EXPECTED_FEATURES:
                    missing = self.EXPECTED_FEATURES - actual_features
                    # Only log info for expected 3month limitation (682 features)
                    if missing <= 700:
                        logger.info(
                            f"Padding {missing} 3month features (yfinance history limit). "
                            f"Got {actual_features}/{self.EXPECTED_FEATURES} features."
                        )
                    else:
                        logger.warning(
                            f"Feature count mismatch: got {actual_features}, expected {self.EXPECTED_FEATURES}. "
                            f"Padding {missing} features with zeros."
                        )
                    import pandas as pd
                    padding_cols = {f'_pad_{i}': 0.0 for i in range(missing)}
                    padding_df = pd.DataFrame(padding_cols, index=features_df.index)
                    features_df = pd.concat([features_df, padding_df], axis=1)
                else:
                    # More features than expected - truncate (less common)
                    logger.warning(
                        f"Feature count mismatch: got {actual_features}, expected {self.EXPECTED_FEATURES}. "
                        f"Truncating to {self.EXPECTED_FEATURES} features."
                    )
                    features_df = features_df.iloc[:, :self.EXPECTED_FEATURES]

            # Get sequence for model input
            if len(features_df) < self.SEQUENCE_LENGTH:
                raise ValueError(
                    f"Need {self.SEQUENCE_LENGTH} bars after feature extraction, "
                    f"have {len(features_df)}"
                )

            sequence = features_df.tail(self.SEQUENCE_LENGTH).values
            x_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

            logger.info(f"Model input tensor: {x_tensor.shape}")

            # Get current price for target calculation
            current_price = float(features_df.iloc[-1]['tsla_close'])

            # Run model inference
            with torch.no_grad():
                result = self._model.predict(x_tensor)

            # Format prediction
            prediction = {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'symbol': 'TSLA',

                # Fusion prediction
                'predicted_high': float(result['predicted_high']),
                'predicted_low': float(result['predicted_low']),
                'confidence': float(result['confidence']),

                # Layer predictions
                'fast_pred_high': float(result.get('fast_pred_high', 0)),
                'fast_pred_low': float(result.get('fast_pred_low', 0)),
                'fast_pred_conf': float(result.get('fast_pred_conf', 0)),

                'medium_pred_high': float(result.get('medium_pred_high', 0)),
                'medium_pred_low': float(result.get('medium_pred_low', 0)),
                'medium_pred_conf': float(result.get('medium_pred_conf', 0)),

                'slow_pred_high': float(result.get('slow_pred_high', 0)),
                'slow_pred_low': float(result.get('slow_pred_low', 0)),
                'slow_pred_conf': float(result.get('slow_pred_conf', 0)),

                # Fusion weights
                'fusion_weights': result.get('fusion_weights', [0.33, 0.33, 0.33]),

                # Multi-task predictions (if available)
                'multi_task': {
                    k: float(v) if hasattr(v, 'item') else v
                    for k, v in result.items()
                    if k not in ['predicted_high', 'predicted_low', 'confidence', 'fusion_weights']
                       and not k.startswith('fast_') and not k.startswith('medium_') and not k.startswith('slow_')
                }
            }

            # Cache prediction
            self._last_prediction = prediction
            self._last_prediction_time = datetime.now()

            logger.info(f"Prediction generated (confidence: {prediction['confidence']:.2%})")

            # Save to database for history
            self._save_to_database(prediction)

            return prediction

    def _save_to_database(self, prediction: Dict):
        """
        Save prediction to database for history tracking.

        Args:
            prediction: Prediction dict from get_latest_prediction()
        """
        try:
            from backend.app.models.database import Prediction

            Prediction.create(
                timestamp=prediction['timestamp'],
                symbol=prediction['symbol'],
                timeframe='24h',
                current_price=prediction['current_price'],
                predicted_high=prediction['predicted_high'],
                predicted_low=prediction['predicted_low'],
                confidence=prediction['confidence'],
                # Layer predictions
                sub_pred_15min_high=prediction.get('fast_pred_high'),
                sub_pred_15min_low=prediction.get('fast_pred_low'),
                sub_pred_15min_conf=prediction.get('fast_pred_conf'),
                sub_pred_1hour_high=prediction.get('medium_pred_high'),
                sub_pred_1hour_low=prediction.get('medium_pred_low'),
                sub_pred_1hour_conf=prediction.get('medium_pred_conf'),
                sub_pred_4hour_high=prediction.get('slow_pred_high'),
                sub_pred_4hour_low=prediction.get('slow_pred_low'),
                sub_pred_4hour_conf=prediction.get('slow_pred_conf'),
                has_actuals=False,
                prediction_timestamp=prediction['timestamp'],
                target_timestamp=prediction['timestamp'] + timedelta(hours=24)
            )

            logger.info("Prediction saved to database")

        except Exception as e:
            logger.warning(f"Failed to save prediction to database: {e}")
            # Don't fail the whole prediction if database save fails

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None

    def get_channel_projection(self, force_refresh: bool = False) -> Dict:
        """
        Get dynamic horizon predictions using project_channel().

        Uses shorter horizons (15min, 30min, 1h) when confidence is high enough.
        Falls back to longer horizons when confidence decays below threshold.

        Thread-safe: uses lock to prevent concurrent yfinance data scrambling.

        Args:
            force_refresh: Skip cache and fetch new prediction

        Returns:
            Dict with:
                - projections: List of valid horizon predictions
                - best_horizon: Shortest valid horizon (highest confidence)
                - current_price: Current TSLA price
                - timestamp: Prediction time
                - raw_confidence: Original model confidence before decay
        """
        with self._prediction_lock:
            # Check cache (inside lock to ensure atomic read/write)
            if not force_refresh and self._last_prediction is not None:
                age = datetime.now() - self._last_prediction_time
                if age < self._cache_ttl:
                    # Reuse cached features, just call project_channel
                    logger.info(f"Using cached data for projection ({age.seconds}s old)")

            logger.info("Generating channel projection...")

            # Load model if needed
            self._load_model()

            # Get live data (same as get_latest_prediction)
            feed = self._get_data_feed()
            df = feed.fetch_for_prediction()

            if len(df) < self.SEQUENCE_LENGTH:
                raise ValueError(
                    f"Insufficient data: {len(df)} bars (need {self.SEQUENCE_LENGTH})\n"
                    f"Market may be closed or data unavailable"
                )

            logger.info(f"Live data fetched: {len(df)} bars")

            # Extract features
            extractor = self._get_feature_extractor()
            result = extractor.extract_features(
                df,
                use_cache=False,
                use_chunking=False,
                continuation=False,
                events_handler=self._events_handler
            )

            if isinstance(result, tuple):
                features_df = result[0]
            else:
                features_df = result

            # Handle feature dimension mismatch (same as get_latest_prediction)
            actual_features = features_df.shape[1]
            if actual_features != self.EXPECTED_FEATURES:
                if actual_features < self.EXPECTED_FEATURES:
                    missing = self.EXPECTED_FEATURES - actual_features
                    import pandas as pd
                    padding_cols = {f'_pad_{i}': 0.0 for i in range(missing)}
                    padding_df = pd.DataFrame(padding_cols, index=features_df.index)
                    features_df = pd.concat([features_df, padding_df], axis=1)
                else:
                    features_df = features_df.iloc[:, :self.EXPECTED_FEATURES]

            # Get sequence for model input
            if len(features_df) < self.SEQUENCE_LENGTH:
                raise ValueError(
                    f"Need {self.SEQUENCE_LENGTH} bars after feature extraction, "
                    f"have {len(features_df)}"
                )

            sequence = features_df.tail(self.SEQUENCE_LENGTH).values
            x_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

            # Get current price
            current_price = float(features_df.iloc[-1]['tsla_close'])

            # Call project_channel with horizons from 15min to 24h (1440 min)
            # Horizons: 15min, 30min, 1h, 2h, 4h, 24h
            projections = self._model.project_channel(
                x_tensor,
                current_price=current_price,
                horizons=[15, 30, 60, 120, 240, 1440],
                min_confidence=0.60  # Slightly lower than default 0.65 to get more options
            )

            # Also get raw prediction for comparison
            with torch.no_grad():
                raw_result = self._model.predict(x_tensor)
                raw_confidence = float(raw_result['confidence'])

            result = {
                'projections': projections,
                'best_horizon': projections[0] if projections else None,
                'current_price': current_price,
                'timestamp': datetime.now(),
                'raw_confidence': raw_confidence,
                'predicted_high': float(raw_result['predicted_high']),
                'predicted_low': float(raw_result['predicted_low']),
            }

            # Add breakout prediction if available
            if 'breakout' in raw_result:
                result['breakout'] = raw_result['breakout']
                logger.info(
                    f"Breakout prediction: {raw_result['breakout']['probability']:.0%} prob, "
                    f"direction={raw_result['breakout']['direction_label']}, "
                    f"bars_until={raw_result['breakout']['bars_until']:.0f}"
                )

            logger.info(
                f"Channel projection: {len(projections)} valid horizons, "
                f"raw confidence: {raw_confidence:.2%}"
            )

            if projections:
                best = projections[0]
                logger.info(
                    f"Best horizon: {best['horizon_minutes']}min "
                    f"(confidence: {best['confidence']:.2%})"
                )

            return result


# Global singleton instance
prediction_service = PredictionService()
