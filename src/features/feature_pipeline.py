"""
Feature Pipeline Orchestrator for AutoTrade v7.0

Coordinates all feature extractors with error handling, logging, and metrics.

Usage:
    from src.features import FeaturePipeline
    from config import get_feature_config

    config = get_feature_config()
    pipeline = FeaturePipeline(config)

    # Training mode (batch)
    features = pipeline.extract(df, mode='batch')

    # Inference mode (streaming)
    features = pipeline.extract(latest_bars, mode='streaming')
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import sys

# Add parent to path for accessing deprecated code during migration
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from config import FeatureConfig
from src.monitoring import get_logger, MetricsTracker
from src.errors import (
    FeatureExtractionError,
    InsufficientDataError,
    VIXFeaturesError,
    EventFeaturesError,
    GracefulDegradation,
)

# Import modular extractors (v7.0 - completed)
# from .channel_features import ChannelFeatureExtractor
# from .market_features import MarketFeatureExtractor
# from .vix_features import VIXFeatureExtractor
# from .event_features import EventFeatureExtractor
# from .channel_history import ChannelHistoryExtractor
# from .breakdown_features import BreakdownFeatureExtractor
# TODO: Wire up extractors in extract() method


logger = get_logger(__name__)


class FeaturePipeline:
    """
    Main feature extraction orchestrator.

    Delegates to old TradingFeatureExtractor for now but provides:
    - Config-driven feature selection
    - Error handling with graceful degradation
    - Structured logging
    - Performance metrics

    Migration Strategy (Week 3-4):
    - Phase 1: Use this wrapper (current)
    - Phase 2: Replace extractors one by one
    - Phase 3: Remove old TradingFeatureExtractor entirely
    """

    def __init__(self, config: FeatureConfig):
        """
        Initialize feature pipeline.

        Args:
            config: Feature configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.metrics = MetricsTracker()
        self.recovery = GracefulDegradation()

        # Initialize old extractor (delegated for now)
        self._old_extractor = None

        self.logger.info(
            "FeaturePipeline initialized",
            version=config.version,
            total_features=config.count_features()['total']
        )

    def extract(
        self,
        df: pd.DataFrame,
        mode: str = 'batch',
        vix_data: Optional[pd.DataFrame] = None,
        events_handler: Optional[any] = None,
    ) -> pd.DataFrame:
        """
        Extract features in batch or streaming mode.

        Args:
            df: OHLC DataFrame with aligned TSLA/SPY data
            mode: 'batch' (training, all history) or 'streaming' (inference, latest only)
            vix_data: Optional VIX data
            events_handler: Optional events handler

        Returns:
            DataFrame with all features

        Raises:
            FeatureExtractionError: If extraction fails
            InsufficientDataError: If not enough data
        """
        with self.metrics.timer('feature_extraction'):
            try:
                # Validate input
                self._validate_input(df)

                # Delegate to old extractor for now
                # TODO (Week 3-4): Replace with modular extractors
                features_df = self._extract_via_old_pipeline(
                    df, vix_data, events_handler
                )

                # Apply config-driven filtering
                features_df = self._filter_features_by_config(features_df)

                # Validate output
                self._validate_output(features_df)

                self.logger.info(
                    "Feature extraction complete",
                    mode=mode,
                    rows=len(features_df),
                    features=len(features_df.columns)
                )

                return features_df

            except InsufficientDataError:
                self.logger.info("Insufficient data for feature extraction")
                raise

            except Exception as e:
                self.logger.error(
                    "Feature extraction failed",
                    error=str(e),
                    exc_info=True
                )
                raise FeatureExtractionError(f"Feature extraction failed: {e}") from e

    def _extract_via_old_pipeline(
        self,
        df: pd.DataFrame,
        vix_data: Optional[pd.DataFrame],
        events_handler: Optional[any],
    ) -> pd.DataFrame:
        """
        Delegate to old TradingFeatureExtractor.

        TODO (Week 3-4): Replace this with modular extractors.
        """
        if self._old_extractor is None:
            # Lazy initialization
            self._old_extractor = TradingFeatureExtractor(
                tsla_df=None,  # Will be extracted from df
                spy_df=None,   # Will be extracted from df
                vix_df=vix_data,
                events_handler=events_handler,
            )

        # Extract features using old pipeline
        # This still works but uses the vibe-coded monolith
        features_df, _ = self._old_extractor.extract_features(
            df,
            vix_data=vix_data,
            use_cache=True,
            continuation=False,  # Skip labels for now
            use_chunking=False,
        )

        return features_df

    def _filter_features_by_config(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter features based on config (e.g., only keep configured windows).

        This is where the config-driven feature selection happens!

        Args:
            features_df: Full feature DataFrame

        Returns:
            Filtered DataFrame with only configured features
        """
        # Get configured windows
        configured_windows = self.config.channel_windows

        # TODO (Week 3-4): Implement proper filtering
        # For now, return all features (old extractor already uses config)
        # When we build modular extractors, they'll only compute configured features

        return features_df

    def _validate_input(self, df: pd.DataFrame):
        """
        Validate input DataFrame.

        Args:
            df: Input DataFrame

        Raises:
            InsufficientDataError: If not enough data
            FeatureExtractionError: If data is invalid
        """
        if df is None or len(df) == 0:
            raise InsufficientDataError("Empty DataFrame provided")

        # Check for minimum bars (need at least 200 for rolling features)
        min_bars = 200
        if len(df) < min_bars:
            raise InsufficientDataError(
                f"Need at least {min_bars} bars, got {len(df)}"
            )

        # Check for required columns
        required_cols = ['tsla_close', 'spy_close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise FeatureExtractionError(f"Missing required columns: {missing}")

    def _validate_output(self, features_df: pd.DataFrame):
        """
        Validate extracted features.

        Args:
            features_df: Feature DataFrame

        Raises:
            FeatureExtractionError: If features are invalid
        """
        # Check for NaN
        nan_count = features_df.isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(
                "Features contain NaN values",
                nan_count=nan_count
            )

        # Check feature count matches config
        expected_count = self.config.count_features()['total']
        actual_count = len(features_df.columns)

        # Allow some tolerance (old extractor may have extra features)
        if actual_count < expected_count * 0.8:
            raise FeatureExtractionError(
                f"Feature count mismatch: expected ~{expected_count}, got {actual_count}"
            )

        self.logger.debug(
            "Feature validation passed",
            expected=expected_count,
            actual=actual_count,
            nan_count=nan_count
        )
