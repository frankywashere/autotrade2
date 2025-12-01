"""
ML Prediction Service

Wraps hierarchical_model.py for inference with caching
"""
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.hierarchical_model import load_hierarchical_model
from src.ml.features import TradingFeatureExtractor
from deprecated.live_data_feed import HybridLiveDataFeed
from src.ml.events import CombinedEventsHandler


class PredictionService:
    """
    Singleton service for ML predictions

    Features:
    - Lazy model loading
    - 5-minute prediction caching
    - Async-ready feature extraction
    """

    _instance = None
    _model = None
    _feature_extractor = None
    _data_feed = None
    _events_handler = None
    _last_prediction = None
    _last_prediction_time = None
    _cache_ttl = timedelta(minutes=5)

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

            print(f"Loading model from {model_path}...")
            self._model = load_hierarchical_model(str(model_path), device='cpu')
            self._model.eval()
            print("✓ Model loaded successfully")

    def _get_feature_extractor(self):
        """Get feature extractor (singleton)"""
        if self._feature_extractor is None:
            self._feature_extractor = TradingFeatureExtractor()

            # Try to load events handler
            try:
                self._events_handler = CombinedEventsHandler()
            except:
                print("⚠️  Events handler not available")
                self._events_handler = None

        return self._feature_extractor

    def _get_data_feed(self):
        """Get live data feed (singleton)"""
        if self._data_feed is None:
            self._data_feed = HybridLiveDataFeed(symbols=['TSLA', 'SPY'])
        return self._data_feed

    def get_latest_prediction(self, force_refresh: bool = False) -> Dict:
        """
        Get latest prediction with 5-min caching

        Args:
            force_refresh: Skip cache and fetch new prediction

        Returns:
            Prediction dict with all outputs
        """
        # Check cache
        if not force_refresh and self._last_prediction is not None:
            age = datetime.now() - self._last_prediction_time
            if age < self._cache_ttl:
                print(f"✓ Using cached prediction ({age.seconds}s old)")
                return self._last_prediction

        # Fetch new prediction
        print("Fetching new prediction...")

        # Load model if needed
        self._load_model()

        # Get live data
        feed = self._get_data_feed()
        df = feed.fetch_for_prediction()

        if len(df) < 200:
            raise ValueError(
                f"Insufficient data: {len(df)} bars (need 200)\n"
                f"Market may be closed or data unavailable"
            )

        # Extract features
        extractor = self._get_feature_extractor()
        result = extractor.extract_features(
            df,
            use_cache=True,
            events_handler=self._events_handler
        )

        # extract_features returns tuple: (features_df, continuation_df) or (features_df, continuation_df, mmap_path)
        if isinstance(result, tuple):
            features_df = result[0]  # Just get the features DataFrame
        else:
            features_df = result

        print(f"  ✓ Features extracted: {features_df.shape}")
        print(f"    Index type: {type(features_df.index)}")
        print(f"    First timestamp: {features_df.index[0] if len(features_df) > 0 else 'EMPTY'}")
        print(f"    Last timestamp: {features_df.index[-1] if len(features_df) > 0 else 'EMPTY'}")

        # Get sequence
        sequence_length = 200
        if len(features_df) < sequence_length:
            raise ValueError(f"Need {sequence_length} bars, have {len(features_df)}")

        sequence = features_df.tail(sequence_length).values
        x_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        # Current price
        current_price = float(features_df.iloc[-1]['tsla_close'])

        # Predict
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

        print(f"✓ New prediction generated (confidence: {prediction['confidence']:.2%})")

        # Save to database for history and online learning
        self._save_to_database(prediction)

        return prediction

    def _save_to_database(self, prediction: Dict):
        """
        Save prediction to database for history tracking and online learning

        Args:
            prediction: Prediction dict from get_latest_prediction()
        """
        try:
            from backend.app.models.database import Prediction

            Prediction.create(
                timestamp=prediction['timestamp'],
                symbol=prediction['symbol'],
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
                # Future: Add multi-task predictions to database schema
                has_actuals=False,
                prediction_timestamp=prediction['timestamp'],
                target_timestamp=prediction['timestamp'] + timedelta(hours=24)
            )

            print("  ✓ Prediction saved to database")

        except Exception as e:
            print(f"  ⚠️ Failed to save prediction: {e}")
            # Don't fail the whole prediction if database save fails

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None


# Global singleton instance
prediction_service = PredictionService()
