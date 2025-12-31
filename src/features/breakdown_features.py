"""
Channel Breakdown Feature Extractor for AutoTrade v7.0

Detects channel breakdown and breakout patterns across timeframes.

Breakdown = Price breaking through channel bounds (upper or lower)
Breakout = Sustained move beyond channel after breakdown

Features per timeframe (3-4 features):
  - breakdown_detected (binary flag)
  - breakdown_direction (1=up, -1=down, 0=none)
  - breakdown_magnitude (% beyond channel)
  - is_sustained_breakout (breakdown lasting >N bars)

v7.0 Reduced: 4 timeframes instead of 11 (5min, 1h, 4h, daily)
Total: 4 TF × 4 features × 2 symbols + global flags = ~38 features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from src.errors import FeatureExtractionError
from src.monitoring import MetricsTracker
from config import FeatureConfig

logger = logging.getLogger(__name__)


class BreakdownFeatureExtractor:
    """
    Extract channel breakdown and breakout features.

    Detects when price breaks through channel bounds and whether
    the breakdown is sustained (true breakout) or false signal.

    Example:
        extractor = BreakdownFeatureExtractor(config)
        features = extractor.extract(df, channel_features)
    """

    def __init__(self, config: FeatureConfig, metrics: Optional[MetricsTracker] = None):
        """
        Initialize breakdown feature extractor.

        Args:
            config: Feature configuration
            metrics: Optional metrics tracker
        """
        self.config = config
        self.metrics = metrics or MetricsTracker()

        # Use reduced timeframes for breakdown detection (v7.0)
        self.timeframes = config.breakdown_timeframes  # [5min, 1h, 4h, daily]

        # Breakdown thresholds
        self.breakdown_threshold = 0.005  # 0.5% beyond channel
        self.sustained_bars = 5  # Must stay broken for 5 bars

        logger.info(f"BreakdownFeatureExtractor initialized: "
                   f"{len(self.timeframes)} timeframes")

    def extract(
        self,
        df: pd.DataFrame,
        channel_features: pd.DataFrame,
        symbols: List[str] = ['tsla', 'spy'],
        mode: str = 'batch'
    ) -> pd.DataFrame:
        """
        Extract breakdown features.

        Args:
            df: Main DataFrame with OHLCV data
            channel_features: Channel features DataFrame
            symbols: List of symbols to process
            mode: 'batch' or 'streaming'

        Returns:
            DataFrame with breakdown features
        """
        with self.metrics.timer('breakdown_features'):
            try:
                all_features = []

                # Extract for each symbol and timeframe
                for symbol in symbols:
                    for timeframe in self.timeframes:
                        tf_features = self._extract_for_symbol_timeframe(
                            df, channel_features, symbol, timeframe
                        )
                        all_features.append(tf_features)

                # Global breakdown flags (any breakdown across all TF/symbols)
                global_features = self._extract_global_flags(all_features, df)
                all_features.append(global_features)

                # Concatenate all features
                result = pd.concat(all_features, axis=1)

                logger.info(f"Breakdown features extracted: {result.shape[1]} features")
                return result

            except Exception as e:
                logger.error(f"Breakdown feature extraction failed: {e}")
                raise FeatureExtractionError(
                    "Failed to extract breakdown features"
                ) from e

    def _extract_for_symbol_timeframe(
        self,
        df: pd.DataFrame,
        channel_features: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Extract breakdown features for one symbol/timeframe combination.

        Uses w50 (reference window) channel features for breakdown detection.
        """
        prefix = f'{symbol}_breakdown_{timeframe}'
        features = {}

        # Find channel position features (using w50 reference window)
        position_col = f'{symbol}_channel_{timeframe}_w50_position'
        upper_dist_col = f'{symbol}_channel_{timeframe}_w50_upper_dist'
        lower_dist_col = f'{symbol}_channel_{timeframe}_w50_lower_dist'

        if position_col not in channel_features.columns:
            logger.warning(f"Channel features not found for {symbol}_{timeframe}")
            # Return zero features
            features[f'{prefix}_detected'] = np.zeros(len(df))
            features[f'{prefix}_direction'] = np.zeros(len(df))
            features[f'{prefix}_magnitude'] = np.zeros(len(df))
            features[f'{prefix}_is_sustained'] = np.zeros(len(df))
            return pd.DataFrame(features, index=df.index)

        position = channel_features[position_col]
        upper_dist = channel_features[upper_dist_col]
        lower_dist = channel_features[lower_dist_col]

        # === Breakdown Detection ===
        # Upper breakdown: position > 1.0 (price above upper line)
        # Lower breakdown: position < 0.0 (price below lower line)

        upper_breakdown = position > 1.0
        lower_breakdown = position < 0.0

        breakdown_detected = (upper_breakdown | lower_breakdown).astype(float)
        features[f'{prefix}_detected'] = breakdown_detected

        # === Breakdown Direction ===
        # 1 = upward (above channel), -1 = downward (below), 0 = none
        direction = np.zeros(len(df))
        direction[upper_breakdown] = 1.0
        direction[lower_breakdown] = -1.0
        features[f'{prefix}_direction'] = direction

        # === Breakdown Magnitude ===
        # How far beyond channel? (as percentage)
        # For upper: use upper_dist (already percentage above)
        # For lower: use lower_dist (already percentage below)

        magnitude = np.zeros(len(df))

        # Upper breakdown magnitude
        upper_mask = upper_breakdown.values
        if upper_mask.any():
            # Position > 1 means we're above channel
            # Magnitude = how much position exceeds 1.0, scaled by channel width
            magnitude[upper_mask] = (position.values[upper_mask] - 1.0) * 100

        # Lower breakdown magnitude
        lower_mask = lower_breakdown.values
        if lower_mask.any():
            # Position < 0 means we're below channel
            # Magnitude = abs(position), scaled by channel width
            magnitude[lower_mask] = abs(position.values[lower_mask]) * 100

        features[f'{prefix}_magnitude'] = magnitude

        # === Sustained Breakout ===
        # True breakout = breakdown sustained for N consecutive bars
        sustained = np.zeros(len(df))

        for i in range(self.sustained_bars, len(df)):
            # Check if breakdown in same direction for last N bars
            recent_dir = direction.values[i-self.sustained_bars:i+1]

            # All upper breakdowns
            if np.all(recent_dir == 1.0):
                sustained[i] = 1.0

            # All lower breakdowns
            elif np.all(recent_dir == -1.0):
                sustained[i] = 1.0

        features[f'{prefix}_is_sustained'] = sustained

        result = pd.DataFrame(features, index=df.index)
        return result

    def _extract_global_flags(
        self,
        symbol_tf_features: List[pd.DataFrame],
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract global breakdown flags aggregated across all symbols/timeframes.

        Global features (6):
          - any_breakdown_detected (any TF/symbol has breakdown)
          - multiple_breakdowns (2+ TF showing breakdown)
          - breakdown_confluence (same direction across TFs)
          - breakdown_divergence (opposite directions across TFs)
          - sustained_breakout_count (number of sustained breakouts)
          - max_breakdown_magnitude (largest magnitude across all)
        """
        features = {}

        # Concatenate all symbol/TF features
        all_tf = pd.concat(symbol_tf_features, axis=1)

        # Find all detected columns
        detected_cols = [c for c in all_tf.columns if c.endswith('_detected')]
        direction_cols = [c for c in all_tf.columns if c.endswith('_direction')]
        magnitude_cols = [c for c in all_tf.columns if c.endswith('_magnitude')]
        sustained_cols = [c for c in all_tf.columns if c.endswith('_is_sustained')]

        # === Any Breakdown ===
        if detected_cols:
            features['any_breakdown_detected'] = all_tf[detected_cols].max(axis=1)
        else:
            features['any_breakdown_detected'] = np.zeros(len(df))

        # === Multiple Breakdowns ===
        if detected_cols:
            features['multiple_breakdowns'] = (
                all_tf[detected_cols].sum(axis=1) >= 2
            ).astype(float)
        else:
            features['multiple_breakdowns'] = np.zeros(len(df))

        # === Breakdown Confluence ===
        # All non-zero directions are the same sign
        if direction_cols:
            directions = all_tf[direction_cols]

            # Count positive and negative
            positive_count = (directions > 0).sum(axis=1)
            negative_count = (directions < 0).sum(axis=1)

            # Confluence = all same direction (and at least 2)
            features['breakdown_confluence'] = (
                ((positive_count >= 2) & (negative_count == 0)) |
                ((negative_count >= 2) & (positive_count == 0))
            ).astype(float)
        else:
            features['breakdown_confluence'] = np.zeros(len(df))

        # === Breakdown Divergence ===
        # Opposite directions across timeframes
        if direction_cols:
            features['breakdown_divergence'] = (
                (positive_count >= 1) & (negative_count >= 1)
            ).astype(float)
        else:
            features['breakdown_divergence'] = np.zeros(len(df))

        # === Sustained Breakout Count ===
        if sustained_cols:
            features['sustained_breakout_count'] = all_tf[sustained_cols].sum(axis=1)
        else:
            features['sustained_breakout_count'] = np.zeros(len(df))

        # === Max Breakdown Magnitude ===
        if magnitude_cols:
            features['max_breakdown_magnitude'] = all_tf[magnitude_cols].max(axis=1)
        else:
            features['max_breakdown_magnitude'] = np.zeros(len(df))

        result = pd.DataFrame(features, index=df.index)
        return result


def extract_breakdown_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    channel_features: pd.DataFrame,
    symbols: List[str] = ['tsla', 'spy'],
    mode: str = 'batch',
    metrics: Optional[MetricsTracker] = None
) -> pd.DataFrame:
    """
    Convenience function to extract breakdown features.

    Args:
        df: Main DataFrame with OHLCV data
        config: Feature configuration
        channel_features: Channel features DataFrame
        symbols: List of symbols to process
        mode: 'batch' or 'streaming'
        metrics: Optional metrics tracker

    Returns:
        DataFrame with breakdown features

    Example:
        >>> config = get_feature_config()
        >>> df = load_5min_data()
        >>> channel_features = extract_channel_features(df, config)
        >>> breakdown = extract_breakdown_features(df, config, channel_features)
    """
    extractor = BreakdownFeatureExtractor(config, metrics)
    return extractor.extract(df, channel_features, symbols, mode)
