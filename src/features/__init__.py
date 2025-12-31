"""
Feature Extraction Module for AutoTrade v7.0

Modular feature extractors following clean architecture principles.

Extractors:
  - ChannelFeatureExtractor: Channel features across windows/timeframes (3,410 features)
  - MarketFeatureExtractor: Price, RSI, volume, correlation (64 features)
  - VIXFeatureExtractor: Volatility regime features (15 features)
  - EventFeatureExtractor: Earnings, FOMC proximity (4 features)
  - ChannelHistoryExtractor: Temporal context from past channels (99 features)
  - BreakdownFeatureExtractor: Channel breakdowns/breakouts (38 features)
  - FeaturePipeline: Orchestrates all extractors

Total: ~3,630 features (v7.0 minimal config)

Usage:
    from src.features import FeaturePipeline
    from config import get_feature_config

    config = get_feature_config()
    pipeline = FeaturePipeline(config)

    # Extract all features
    features = pipeline.extract(df, mode='batch')
"""

from .feature_pipeline import FeaturePipeline
from .channel_features import (
    ChannelFeatureExtractor,
    extract_channel_features_multi_symbol,
)
from .market_features import (
    MarketFeatureExtractor,
    extract_market_features,
)
from .vix_features import (
    VIXFeatureExtractor,
    extract_vix_features,
)
from .event_features import (
    EventFeatureExtractor,
    extract_event_features,
)
from .channel_history import (
    ChannelHistoryExtractor,
    extract_channel_history,
)
from .breakdown_features import (
    BreakdownFeatureExtractor,
    extract_breakdown_features,
)

__all__ = [
    # Main pipeline
    'FeaturePipeline',

    # Extractors
    'ChannelFeatureExtractor',
    'MarketFeatureExtractor',
    'VIXFeatureExtractor',
    'EventFeatureExtractor',
    'ChannelHistoryExtractor',
    'BreakdownFeatureExtractor',

    # Convenience functions
    'extract_channel_features_multi_symbol',
    'extract_market_features',
    'extract_vix_features',
    'extract_event_features',
    'extract_channel_history',
    'extract_breakdown_features',
]
