"""
Channel History Feature Extractor for AutoTrade v7.0

Extracts temporal context features about recent channel behavior.
Provides "memory" of what happened in previous channels.

v6.0 Innovation: These features give the model context about channel dynamics:
  - How long recent channels lasted
  - What direction they were
  - Trend in channel durations (getting shorter/longer?)
  - Frequency of channel transitions

Features per timeframe (9 × 11 = 99 total):
  - prev_channel_duration: Duration of last completed channel
  - prev_channel_direction: Direction (bull/bear/sideways)
  - channel_duration_trend: Are channels getting shorter/longer?
  - channels_count_recent: Number of recent transitions
  - consecutive_same_direction: Streak of same-direction channels
  - avg_recent_duration: Average recent channel duration
  - prev_channel_bounce_count: Bounces in previous channel
  - bounce_count_trend: Are bounces increasing/decreasing?
  - transition_frequency: Rate of channel changes

Note: In v7.0, this is simplified to compute from channel features directly.
      Full implementation requires transition labels from training pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import logging

from src.errors import ChannelFeaturesError
from src.monitoring import MetricsTracker
from config import FeatureConfig

logger = logging.getLogger(__name__)


class ChannelHistoryExtractor:
    """
    Extract channel history features.

    Provides temporal context about recent channel behavior across timeframes.

    v7.0 Simplified: Computes basic history features from channel state.
    For full v6.0 features, use transition labels from training pipeline.

    Example:
        extractor = ChannelHistoryExtractor(config)
        features = extractor.extract(df, channel_features)
    """

    def __init__(self, config: FeatureConfig, metrics: Optional[MetricsTracker] = None):
        """
        Initialize channel history extractor.

        Args:
            config: Feature configuration
            metrics: Optional metrics tracker
        """
        self.config = config
        self.metrics = metrics or MetricsTracker()
        self.timeframes = config.channel_timeframes

        logger.info(f"ChannelHistoryExtractor initialized: {len(self.timeframes)} timeframes")

    def extract(
        self,
        df: pd.DataFrame,
        channel_features: Optional[pd.DataFrame] = None,
        transition_labels_path: Optional[Path] = None,
        mode: str = 'batch'
    ) -> pd.DataFrame:
        """
        Extract channel history features.

        Args:
            df: Main DataFrame (for index alignment)
            channel_features: Channel features DataFrame (optional)
            transition_labels_path: Path to transition labels (if available)
            mode: 'batch' or 'streaming'

        Returns:
            DataFrame with channel history features

        Note:
            If transition_labels_path is provided, loads full v6.0 features.
            Otherwise, computes simplified features from channel_features.
        """
        with self.metrics.timer('channel_history_features'):
            try:
                # Try to load full v6.0 features from transition labels
                if transition_labels_path and Path(transition_labels_path).exists():
                    logger.info("Loading full channel history from transition labels")
                    return self._load_from_transition_labels(df, transition_labels_path)

                # Otherwise, compute simplified features
                if channel_features is None:
                    logger.warning("No channel features provided, returning zero features")
                    return self._get_zero_features(df)

                logger.info("Computing simplified channel history features")
                features = self._compute_simplified_history(df, channel_features)

                logger.info(f"Channel history features extracted: {features.shape[1]} features")
                return features

            except Exception as e:
                logger.error(f"Channel history extraction failed: {e}")
                return self._get_zero_features(df)

    def _load_from_transition_labels(
        self,
        df: pd.DataFrame,
        labels_path: Path
    ) -> pd.DataFrame:
        """
        Load full v6.0 channel history features from transition labels.

        This is the full implementation used during training.
        """
        labels_dir = Path(labels_path)
        all_features = []

        for tf in self.timeframes:
            # Look for transition labels file
            pattern = f"transition_labels_{tf}_*.pkl"
            files = list(labels_dir.glob(pattern))

            if not files:
                logger.warning(f"No transition labels found for {tf}")
                # Create zero features for this TF
                tf_features = self._get_zero_features_for_tf(df, tf)
                all_features.append(tf_features)
                continue

            # Load most recent file
            trans_df = pd.read_pickle(files[0]).sort_index()

            # Extract features for this TF
            tf_features = self._extract_from_transitions(df, trans_df, tf)
            all_features.append(tf_features)

        # Concatenate all timeframes
        result = pd.concat(all_features, axis=1)
        return result

    def _extract_from_transitions(
        self,
        df: pd.DataFrame,
        trans_df: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Extract history features from transition labels for one timeframe.

        Implements full v6.0 feature extraction.
        """
        n_samples = len(df)
        feature_timestamps = df.index.values
        trans_timestamps = trans_df.index.values

        # Initialize feature arrays
        features = {
            f'prev_channel_duration_{timeframe}': np.zeros(n_samples, dtype=np.float32),
            f'prev_channel_direction_{timeframe}': np.full(n_samples, 2.0, dtype=np.float32),
            f'prev_transition_type_{timeframe}': np.zeros(n_samples, dtype=np.float32),
            f'channel_duration_trend_{timeframe}': np.zeros(n_samples, dtype=np.float32),
            f'channels_count_recent_{timeframe}': np.zeros(n_samples, dtype=np.float32),
            f'consecutive_same_direction_{timeframe}': np.zeros(n_samples, dtype=np.float32),
            f'avg_recent_duration_{timeframe}': np.zeros(n_samples, dtype=np.float32),
            f'prev_channel_bounce_count_{timeframe}': np.zeros(n_samples, dtype=np.float32),
            f'bounce_count_trend_{timeframe}': np.zeros(n_samples, dtype=np.float32),
        }

        # Get transition data
        trans_durations = trans_df['duration_bars'].values.astype(np.float32)
        trans_directions = trans_df['current_direction'].values.astype(np.float32)
        trans_types = trans_df['transition_type'].values.astype(np.float32)

        # Bounce counts (with fallback for old labels)
        if 'current_cycles' in trans_df.columns:
            trans_cycles = trans_df['current_cycles'].fillna(0).values.astype(np.float32)
        else:
            trans_cycles = np.zeros(len(trans_df), dtype=np.float32)

        n_trans = len(trans_timestamps)

        # Find most recent transition for each feature timestamp
        insertion_indices = np.searchsorted(trans_timestamps, feature_timestamps, side='left')
        has_history = insertion_indices > 0
        last_trans_idx = np.clip(insertion_indices - 1, 0, n_trans - 1)

        # Fill most recent transition values
        features[f'prev_channel_duration_{timeframe}'][has_history] = trans_durations[last_trans_idx[has_history]]
        features[f'prev_channel_direction_{timeframe}'][has_history] = trans_directions[last_trans_idx[has_history]]
        features[f'prev_transition_type_{timeframe}'][has_history] = trans_types[last_trans_idx[has_history]]
        features[f'prev_channel_bounce_count_{timeframe}'][has_history] = trans_cycles[last_trans_idx[has_history]]

        # Channels count in recent window (500 bars)
        lookback_ns = np.timedelta64(500, 'm')
        cutoff_timestamps = feature_timestamps - lookback_ns
        cutoff_indices = np.searchsorted(trans_timestamps, cutoff_timestamps, side='left')
        counts = np.maximum(0, last_trans_idx - cutoff_indices + 1).astype(np.float32)
        counts[~has_history] = 0
        features[f'channels_count_recent_{timeframe}'] = counts

        # Average recent duration (last 5 channels)
        window_size = 5
        for i, idx in enumerate(last_trans_idx):
            if has_history[i]:
                start = max(0, idx - window_size + 1)
                recent_durs = trans_durations[start:idx+1]
                features[f'avg_recent_duration_{timeframe}'][i] = recent_durs.mean()

        # Duration trend (slope of last 5 durations)
        for i, idx in enumerate(last_trans_idx):
            if has_history[i]:
                start = max(0, idx - window_size + 1)
                recent_durs = trans_durations[start:idx+1]
                if len(recent_durs) >= 2:
                    x = np.arange(len(recent_durs))
                    slope = np.polyfit(x, recent_durs, 1)[0]
                    # Normalize by mean
                    if recent_durs.mean() > 0:
                        features[f'channel_duration_trend_{timeframe}'][i] = np.clip(
                            slope / recent_durs.mean(), -1.0, 1.0
                        )

        # Consecutive same-direction streak
        for i, idx in enumerate(last_trans_idx):
            if has_history[i]:
                streak = 1
                current_dir = trans_directions[idx]
                for j in range(idx - 1, -1, -1):
                    if trans_directions[j] == current_dir:
                        streak += 1
                    else:
                        break
                features[f'consecutive_same_direction_{timeframe}'][i] = float(streak)

        # Bounce count trend (slope of last 5 bounce counts)
        for i, idx in enumerate(last_trans_idx):
            if has_history[i]:
                start = max(0, idx - window_size + 1)
                recent_cycles = trans_cycles[start:idx+1]
                if len(recent_cycles) >= 2 and recent_cycles.mean() > 0:
                    x = np.arange(len(recent_cycles))
                    slope = np.polyfit(x, recent_cycles, 1)[0]
                    features[f'bounce_count_trend_{timeframe}'][i] = np.clip(
                        slope / recent_cycles.mean(), -1.0, 1.0
                    )

        result = pd.DataFrame(features, index=df.index)
        return result

    def _compute_simplified_history(
        self,
        df: pd.DataFrame,
        channel_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute simplified channel history features from channel state.

        v7.0 Simplified: Uses channel features to infer recent behavior.
        Not as accurate as full v6.0 with transition labels, but provides
        useful context for inference when labels unavailable.
        """
        all_features = []

        for tf in self.timeframes:
            # Find channel features for this timeframe
            tf_cols = [c for c in channel_features.columns if f'_channel_{tf}_' in c]

            if not tf_cols:
                logger.warning(f"No channel features found for {tf}")
                tf_features = self._get_zero_features_for_tf(df, tf)
                all_features.append(tf_features)
                continue

            # Extract simplified features from channel state
            features = {}

            # Use channel duration as proxy for previous channel duration
            # (This is a simplification - actual previous would come from transitions)
            duration_col = f'tsla_channel_{tf}_w50_duration'
            if duration_col in channel_features.columns:
                features[f'prev_channel_duration_{tf}'] = channel_features[duration_col].shift(1).fillna(0)
            else:
                features[f'prev_channel_duration_{tf}'] = np.zeros(len(df))

            # Direction from channel slopes
            slope_col = f'tsla_channel_{tf}_w50_close_slope_pct'
            if slope_col in channel_features.columns:
                slopes = channel_features[slope_col]
                # 0=bull, 1=bear, 2=sideways
                direction = np.full(len(df), 2.0)  # Default sideways
                direction[slopes > 0.1] = 0.0  # Bull
                direction[slopes < -0.1] = 1.0  # Bear
                features[f'prev_channel_direction_{tf}'] = pd.Series(direction, index=df.index).shift(1).fillna(2.0)
            else:
                features[f'prev_channel_direction_{tf}'] = np.full(len(df), 2.0)

            # Transition type - simplified (always 0 = continue)
            features[f'prev_transition_type_{tf}'] = np.zeros(len(df))

            # Duration trend - rolling slope of durations
            if duration_col in channel_features.columns:
                durations = channel_features[duration_col]
                trends = durations.rolling(5).apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / x.mean() if len(x) > 1 and x.mean() > 0 else 0,
                    raw=False
                )
                features[f'channel_duration_trend_{tf}'] = trends.fillna(0)
            else:
                features[f'channel_duration_trend_{tf}'] = np.zeros(len(df))

            # Channels count - estimate from duration changes
            # (Simplified - actual count would come from transitions)
            features[f'channels_count_recent_{tf}'] = np.zeros(len(df))

            # Consecutive same direction - simplified
            if f'prev_channel_direction_{tf}' in features:
                direction_series = features[f'prev_channel_direction_{tf}']
                streaks = (direction_series != direction_series.shift()).cumsum()
                features[f'consecutive_same_direction_{tf}'] = direction_series.groupby(streaks).cumcount() + 1
            else:
                features[f'consecutive_same_direction_{tf}'] = np.ones(len(df))

            # Average recent duration
            if duration_col in channel_features.columns:
                features[f'avg_recent_duration_{tf}'] = channel_features[duration_col].rolling(5).mean().fillna(0)
            else:
                features[f'avg_recent_duration_{tf}'] = np.zeros(len(df))

            # Bounce counts - use complete_cycles features
            cycles_col = f'tsla_channel_{tf}_w50_complete_cycles_2_0pct'
            if cycles_col in channel_features.columns:
                features[f'prev_channel_bounce_count_{tf}'] = channel_features[cycles_col].shift(1).fillna(0)

                # Bounce trend
                cycles = channel_features[cycles_col]
                bounce_trends = cycles.rolling(5).apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / x.mean() if len(x) > 1 and x.mean() > 0 else 0,
                    raw=False
                )
                features[f'bounce_count_trend_{tf}'] = bounce_trends.fillna(0)
            else:
                features[f'prev_channel_bounce_count_{tf}'] = np.zeros(len(df))
                features[f'bounce_count_trend_{tf}'] = np.zeros(len(df))

            tf_features = pd.DataFrame(features, index=df.index)
            all_features.append(tf_features)

        # Concatenate all timeframes
        result = pd.concat(all_features, axis=1)
        return result

    def _get_zero_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get zero-filled channel history features.

        Returns DataFrame with all 99 features set to neutral values.
        """
        all_features = []

        for tf in self.timeframes:
            tf_features = self._get_zero_features_for_tf(df, tf)
            all_features.append(tf_features)

        result = pd.concat(all_features, axis=1)
        return result

    def _get_zero_features_for_tf(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Get zero-filled features for a single timeframe.
        """
        features = {
            f'prev_channel_duration_{timeframe}': np.zeros(len(df)),
            f'prev_channel_direction_{timeframe}': np.full(len(df), 2.0),  # Sideways
            f'prev_transition_type_{timeframe}': np.zeros(len(df)),
            f'channel_duration_trend_{timeframe}': np.zeros(len(df)),
            f'channels_count_recent_{timeframe}': np.zeros(len(df)),
            f'consecutive_same_direction_{timeframe}': np.zeros(len(df)),
            f'avg_recent_duration_{timeframe}': np.zeros(len(df)),
            f'prev_channel_bounce_count_{timeframe}': np.zeros(len(df)),
            f'bounce_count_trend_{timeframe}': np.zeros(len(df)),
        }

        return pd.DataFrame(features, index=df.index)


def extract_channel_history(
    df: pd.DataFrame,
    config: FeatureConfig,
    channel_features: Optional[pd.DataFrame] = None,
    transition_labels_path: Optional[Path] = None,
    mode: str = 'batch',
    metrics: Optional[MetricsTracker] = None
) -> pd.DataFrame:
    """
    Convenience function to extract channel history features.

    Args:
        df: Main DataFrame (for index alignment)
        config: Feature configuration
        channel_features: Channel features (for simplified computation)
        transition_labels_path: Path to transition labels (for full v6.0)
        mode: 'batch' or 'streaming'
        metrics: Optional metrics tracker

    Returns:
        DataFrame with channel history features

    Example:
        >>> config = get_feature_config()
        >>> df = load_5min_data()
        >>> channel_features = extract_channel_features(df, config)
        >>> history = extract_channel_history(df, config, channel_features)
    """
    extractor = ChannelHistoryExtractor(config, metrics)
    return extractor.extract(df, channel_features, transition_labels_path, mode)
