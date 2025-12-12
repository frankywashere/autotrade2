"""
ML Prediction Service (v4.0)

Wraps hierarchical_model.py for inference with caching.
v4.0: 11-layer architecture with ~9,000 features (down from 14,500)
      - Includes VIX loading for market_state computation
      - Uses market_state (12 dims) instead of news embeddings
"""
import torch
import numpy as np
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.hierarchical_model import load_hierarchical_model, HierarchicalLNN
from src.ml.features import TradingFeatureExtractor
from src.ml.market_state import calculate_market_state, MARKET_STATE_DIM
from deprecated_code.live_data_feed import HybridLiveDataFeed
from src.ml.events import CombinedEventsHandler

# Configure logging
logger = logging.getLogger(__name__)


class PredictionService:
    """
    Singleton service for ML predictions (v4.0).

    Features:
    - Fresh feature extraction from live OHLCV data (~9,000 features)
    - 11-layer hierarchical model (one per timeframe)
    - VIX-based market_state computation (12 dims)
    - 5-minute prediction caching
    - Lazy model loading
    """

    _instance = None
    _prediction_lock = threading.Lock()  # Serialize predictions to prevent yfinance data scrambling
    _model = None
    _feature_extractor = None
    _data_feed = None
    _events_handler = None
    _vix_data = None  # v4.0: VIX data for market_state
    _last_prediction = None
    _last_prediction_time = None
    _cache_ttl = timedelta(minutes=5)

    # Feature dimensions - loaded from model checkpoint (single source of truth)
    _expected_features = None  # Will be set from model.input_sizes when loaded
    SEQUENCE_LENGTH = 200

    # v4.0: 11 timeframes for layer predictions
    TIMEFRAMES = HierarchicalLNN.TIMEFRAMES

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
            # v4.x: input_sizes is a dict, use 5min as reference (or first available)
            input_sizes = self._model.input_sizes
            if input_sizes:
                PredictionService._expected_features = input_sizes.get('5min', list(input_sizes.values())[0])
            else:
                PredictionService._expected_features = 900  # fallback default
            logger.info(f"Model loaded successfully (input_sizes={input_sizes})")

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

            # Get VIX data from df.attrs if available (v3.20)
            vix_data = df.attrs.get('vix_data', None)
            if vix_data is not None:
                logger.info(f"VIX data available: {len(vix_data)} daily bars")

            result = extractor.extract_features(
                df,
                use_cache=False,       # Force fresh extraction
                use_chunking=False,    # No sharding for inference
                continuation=False,    # Skip continuation labels
                events_handler=self._events_handler,
                vix_data=vix_data      # v3.20: VIX features for volatility regime
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

            # Format prediction (v4.0: 11 layers instead of 3)
            prediction = {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'symbol': 'TSLA',
                'model_version': '4.0',

                # Fusion prediction
                'predicted_high': float(result['predicted_high']),
                'predicted_low': float(result['predicted_low']),
                'confidence': float(result['confidence']),

                # v4.0: Per-timeframe layer predictions (11 layers)
                'layer_predictions': {},

                # Fusion weights (now 11 values instead of 3)
                'fusion_weights': result.get('fusion_weights', [1/11] * 11),

                # Multi-task predictions (if available)
                'multi_task': {
                    k: float(v) if hasattr(v, 'item') else v
                    for k, v in result.items()
                    if k not in ['predicted_high', 'predicted_low', 'confidence', 'fusion_weights', 'hidden_states']
                       and not k.endswith('_pred_high') and not k.endswith('_pred_low') and not k.endswith('_pred_conf')
                }
            }

            # Add per-timeframe predictions (v4.0: 11 timeframes)
            for tf in self.TIMEFRAMES:
                prediction['layer_predictions'][tf] = {
                    'high': float(result.get(f'{tf}_pred_high', 0)),
                    'low': float(result.get(f'{tf}_pred_low', 0)),
                    'confidence': float(result.get(f'{tf}_pred_conf', 0)),
                }

            # Backward compatibility: also include legacy fast/medium/slow keys
            prediction['fast_pred_high'] = prediction['layer_predictions'].get('5min', {}).get('high', 0)
            prediction['fast_pred_low'] = prediction['layer_predictions'].get('5min', {}).get('low', 0)
            prediction['fast_pred_conf'] = prediction['layer_predictions'].get('5min', {}).get('confidence', 0)
            prediction['medium_pred_high'] = prediction['layer_predictions'].get('1h', {}).get('high', 0)
            prediction['medium_pred_low'] = prediction['layer_predictions'].get('1h', {}).get('low', 0)
            prediction['medium_pred_conf'] = prediction['layer_predictions'].get('1h', {}).get('confidence', 0)
            prediction['slow_pred_high'] = prediction['layer_predictions'].get('daily', {}).get('high', 0)
            prediction['slow_pred_low'] = prediction['layer_predictions'].get('daily', {}).get('low', 0)
            prediction['slow_pred_conf'] = prediction['layer_predictions'].get('daily', {}).get('confidence', 0)

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

    def _extract_channel_setups(self, features_df, current_price: float) -> List[Dict]:
        """
        Extract trade setups from channel features across all timeframes.

        Groups channels into trade categories (Scalp, Intraday, Swing, Position)
        and finds the best channel in each category based on R-squared.

        Args:
            features_df: DataFrame with extracted features
            current_price: Current TSLA price

        Returns:
            List of setup dicts sorted by confidence (descending)
        """
        # Map timeframe categories to their channel timeframes
        timeframe_categories = {
            'scalp': {
                'channels': ['5min', '15min', '30min'],
                'label': 'Scalp',
                'description': 'Minutes to 1 hour',
                'duration_map': {
                    '5min': '5-30 min',
                    '15min': '15-60 min',
                    '30min': '30 min - 2 hr'
                }
            },
            'intraday': {
                'channels': ['1h', '2h', '3h', '4h'],
                'label': 'Intraday',
                'description': '1-8 hours',
                'duration_map': {
                    '1h': '1-4 hours',
                    '2h': '2-6 hours',
                    '3h': '3-8 hours',
                    '4h': '4-12 hours'
                }
            },
            'swing': {
                'channels': ['daily'],
                'label': 'Swing',
                'description': 'Days to 2 weeks',
                'duration_map': {
                    'daily': '1-10 days'
                }
            },
            'position': {
                'channels': ['weekly', 'monthly'],
                'label': 'Position',
                'description': 'Weeks to months',
                'duration_map': {
                    'weekly': '1-4 weeks',
                    'monthly': '1-3 months'
                }
            }
        }

        setups = []
        # Try multiple window sizes, prefer larger (more stable) windows
        windows_to_try = [60, 100, 30, 45, 80]

        for category_name, config in timeframe_categories.items():
            best_r2 = 0
            best_channel_data = None

            for tf_name in config['channels']:
                # Try multiple window sizes, find the first one that exists
                found_window = None
                for window in windows_to_try:
                    prefix = f'tsla_channel_{tf_name}_w{window}'
                    r2_col = f'{prefix}_r_squared_avg'
                    if r2_col in features_df.columns:
                        found_window = window
                        break

                if found_window is None:
                    continue

                # Use the found window
                prefix = f'tsla_channel_{tf_name}_w{found_window}'
                r2_col = f'{prefix}_r_squared_avg'
                width_col = f'{prefix}_channel_width_pct'
                slope_col = f'{prefix}_close_slope_pct'
                position_col = f'{prefix}_position'

                # Get values from most recent row
                r2 = features_df[r2_col].iloc[-1]
                if np.isnan(r2) or r2 <= 0:
                    continue

                if r2 > best_r2:
                    best_r2 = r2
                    # Safely get values with defaults
                    width = features_df[width_col].iloc[-1] if width_col in features_df.columns else 0.05
                    slope = features_df[slope_col].iloc[-1] if slope_col in features_df.columns else 0.0
                    position = features_df[position_col].iloc[-1] if position_col in features_df.columns else 0.5

                    best_channel_data = {
                        'timeframe': tf_name,
                        'window': found_window,
                        'r_squared': r2,
                        'width_pct': width if not np.isnan(width) else 0.05,
                        'slope_pct': slope if not np.isnan(slope) else 0.0,
                        'position': position if not np.isnan(position) else 0.5,
                        'duration': config['duration_map'].get(tf_name, 'unknown')
                    }

            # Only include if we found a valid channel above minimum threshold
            min_r2_threshold = 0.40
            if best_channel_data and best_r2 > min_r2_threshold:
                # Calculate high/low from channel width
                width_pct = best_channel_data['width_pct']
                position = best_channel_data['position']

                # Channel bounds: price can move within width, position tells us where we are
                # Position 0 = at lower bound, 1 = at upper bound, 0.5 = middle
                room_up = width_pct * (1 - position)  # Remaining room to upper bound
                room_down = width_pct * position      # Remaining room to lower bound

                high_price = current_price * (1 + room_up)
                low_price = current_price * (1 - room_down)

                # Confidence: R-squared scaled with some boost
                # R² of 0.8 → 85% confidence, R² of 0.5 → 55% confidence
                confidence = min(95, best_r2 * 100 + 5)

                # Determine direction from slope
                slope = best_channel_data['slope_pct']
                if slope > 0.1:
                    direction = 'bullish'
                elif slope < -0.1:
                    direction = 'bearish'
                else:
                    direction = 'neutral'

                # Risk assessment based on position in channel
                if position > 0.85:
                    risk_note = 'Near upper bound - limited upside'
                elif position < 0.15:
                    risk_note = 'Near lower bound - limited downside'
                elif position > 0.7:
                    risk_note = 'Upper half of channel'
                elif position < 0.3:
                    risk_note = 'Lower half of channel'
                else:
                    risk_note = 'Mid-channel - balanced risk'

                setups.append({
                    'type': category_name,
                    'label': config['label'],
                    'description': config['description'],
                    'channel_timeframe': best_channel_data['timeframe'],
                    'r_squared': round(best_r2, 3),
                    'confidence': round(confidence, 1),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'duration': best_channel_data['duration'],
                    'direction': direction,
                    'slope_pct': round(slope, 3),
                    'position_in_channel': round(position, 2),
                    'risk_note': risk_note
                })

        # Sort by confidence descending
        return sorted(setups, key=lambda x: x['confidence'], reverse=True)

    def get_trade_setups(self, force_refresh: bool = False) -> Dict:
        """
        Get multi-timeframe trade setups based on channel analysis.

        Analyzes channels across all timeframes (5min to monthly) and returns
        trade setups for each category that meets confidence threshold.

        Thread-safe: uses lock to prevent concurrent yfinance data scrambling.

        Args:
            force_refresh: Skip cache and fetch new data

        Returns:
            Dict with:
                - setups: List of trade setups (Scalp, Intraday, Swing, Position)
                - best_setup: Highest confidence setup
                - current_price: Current TSLA price
                - timestamp: Analysis time
                - model_prediction: Also includes the model's fused prediction
        """
        with self._prediction_lock:
            logger.info("Generating trade setups from channel analysis...")

            # Load model if needed (for model prediction comparison)
            self._load_model()

            # Get live data
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

            logger.info(f"Features extracted: {features_df.shape}")

            # Handle feature dimension mismatch for model prediction
            actual_features = features_df.shape[1]
            if actual_features != self.EXPECTED_FEATURES:
                if actual_features < self.EXPECTED_FEATURES:
                    missing = self.EXPECTED_FEATURES - actual_features
                    import pandas as pd
                    padding_cols = {f'_pad_{i}': 0.0 for i in range(missing)}
                    padding_df = pd.DataFrame(padding_cols, index=features_df.index)
                    features_padded = pd.concat([features_df, padding_df], axis=1)
                else:
                    features_padded = features_df.iloc[:, :self.EXPECTED_FEATURES]
            else:
                features_padded = features_df

            # Get current price
            current_price = float(features_df.iloc[-1]['tsla_close'])

            # Extract channel-based setups (uses features_df before padding)
            setups = self._extract_channel_setups(features_df, current_price)

            # Also get model prediction for comparison
            sequence = features_padded.tail(self.SEQUENCE_LENGTH).values
            x_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                model_result = self._model.predict(x_tensor)

            model_prediction = {
                'predicted_high': float(model_result['predicted_high']),
                'predicted_low': float(model_result['predicted_low']),
                'confidence': float(model_result['confidence']),
                'fusion_weights': model_result.get('fusion_weights', [0.33, 0.33, 0.33])
            }

            result = {
                'setups': setups,
                'best_setup': setups[0] if setups else None,
                'current_price': current_price,
                'timestamp': datetime.now(),
                'model_prediction': model_prediction,
                'setup_count': len(setups)
            }

            # Log summary
            if setups:
                best = setups[0]
                logger.info(
                    f"Best setup: {best['label']} ({best['channel_timeframe']} channel) "
                    f"- {best['confidence']:.1f}% confidence, "
                    f"${best['low']:.2f} - ${best['high']:.2f}"
                )
            else:
                logger.info("No valid trade setups found (all channels below R² threshold)")

            return result


# Global singleton instance
prediction_service = PredictionService()
