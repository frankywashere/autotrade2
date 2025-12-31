"""
Custom exception hierarchy for AutoTrade v7.0

Provides specific exceptions for different failure modes to enable
proper error handling and graceful degradation.

Usage:
    from src.errors import FeatureExtractionError, InsufficientDataError

    try:
        features = extractor.extract(data)
    except InsufficientDataError:
        # Expected - just log and continue
        logger.info("Not enough data yet")
    except FeatureExtractionError:
        # Serious but recoverable
        logger.error("Feature extraction failed")
        alert_team(severity='high')
"""


class TradingMLError(Exception):
    """
    Base exception for all trading ML errors.

    All custom exceptions inherit from this for easy catching.
    """
    pass


class FeatureExtractionError(TradingMLError):
    """
    Feature extraction failed.

    Raised when feature computation fails for any reason.
    This is serious but may be recoverable (use cached features).

    Examples:
        - Channel calculation failed
        - RSI computation error
        - VIX data corrupt
    """
    pass


class InsufficientDataError(TradingMLError):
    """
    Not enough data for prediction.

    Raised when there aren't enough bars for feature extraction.
    This is expected during warmup and not an error condition.

    Examples:
        - Need 200 bars but only have 50
        - Waiting for higher timeframe bars to complete
        - First N minutes after market open

    Note: This is NOT an error - just means "try again later"
    """
    pass


class CacheInvalidError(TradingMLError):
    """
    Cache is invalid or corrupted.

    Raised when cached features don't match current config or are corrupted.

    Examples:
        - Version mismatch (v6.0 cache, v7.0 code)
        - Corrupted mmap file
        - Missing cache files

    Recovery: Regenerate cache
    """
    pass


class PredictionError(TradingMLError):
    """
    Prediction failed.

    Raised when model inference fails.
    Serious - may indicate model corruption or data issues.

    Examples:
        - Model forward pass failed
        - Input shape mismatch
        - NaN in predictions

    Recovery: Use cached prediction or fallback model
    """
    pass


class ModelLoadError(TradingMLError):
    """
    Model checkpoint could not be loaded.

    Raised when model initialization fails.
    Critical - prevents inference.

    Examples:
        - Checkpoint file missing
        - Checkpoint corrupted
        - Version mismatch
        - Feature config mismatch

    Recovery: Load previous checkpoint or alert team
    """
    pass


class DataValidationError(TradingMLError):
    """
    Data failed validation checks.

    Raised when input data is invalid (NaN, out of range, duplicates).

    Examples:
        - NaN in OHLC data
        - Negative volume
        - Duplicate timestamps
        - Price > 10x previous bar

    Recovery: Skip invalid bar or use interpolation
    """
    pass


class ConfigurationError(TradingMLError):
    """
    Invalid configuration.

    Raised when config validation fails.

    Examples:
        - Invalid window size (< 0)
        - Unknown timeframe
        - Missing required config field

    Recovery: Load default config
    """
    pass


class VIXFeaturesError(FeatureExtractionError):
    """VIX feature extraction failed (non-critical)"""
    pass


class EventFeaturesError(FeatureExtractionError):
    """Event feature extraction failed (non-critical)"""
    pass


class ChannelFeaturesError(FeatureExtractionError):
    """Channel feature extraction failed (CRITICAL)"""
    pass
