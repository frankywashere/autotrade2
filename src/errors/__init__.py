"""
Error handling for AutoTrade v7.0

Custom exception hierarchy for proper error handling and graceful degradation.
"""

from .exceptions import (
    TradingMLError,
    FeatureExtractionError,
    InsufficientDataError,
    CacheInvalidError,
    PredictionError,
    ModelLoadError,
    DataValidationError,
    ConfigurationError,
    VIXFeaturesError,
    EventFeaturesError,
    ChannelFeaturesError,
)

from .handlers import (
    setup_error_handlers,
    handle_errors,
)

from .recovery import (
    GracefulDegradation,
)

__all__ = [
    # Exceptions
    'TradingMLError',
    'FeatureExtractionError',
    'InsufficientDataError',
    'CacheInvalidError',
    'PredictionError',
    'ModelLoadError',
    'DataValidationError',
    'ConfigurationError',
    'VIXFeaturesError',
    'EventFeaturesError',
    'ChannelFeaturesError',
    # Handlers
    'setup_error_handlers',
    'handle_errors',
    # Recovery
    'GracefulDegradation',
]
