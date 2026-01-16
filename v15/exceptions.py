"""
V15 Custom Exceptions - All failures should be LOUD, not silent.
"""

class V15Error(Exception):
    """Base exception for all v15 errors."""
    pass

class FeatureExtractionError(V15Error):
    """Raised when feature extraction fails."""
    pass

class InvalidFeatureError(V15Error):
    """Raised when a feature value is NaN, Inf, or invalid type."""
    def __init__(self, feature_name: str, value, message: str = None):
        self.feature_name = feature_name
        self.value = value
        msg = message or f"Feature '{feature_name}' has invalid value: {value}"
        super().__init__(msg)

class DataLoadError(V15Error):
    """Raised when data loading fails."""
    pass

class ResamplingError(V15Error):
    """Raised when timeframe resampling fails."""
    pass

class ChannelDetectionError(V15Error):
    """Raised when channel detection fails unexpectedly."""
    pass

class LabelGenerationError(V15Error):
    """Raised when label generation fails."""
    pass

class ModelError(V15Error):
    """Raised when model forward pass fails."""
    pass

class ConfigurationError(V15Error):
    """Raised when configuration is invalid."""
    pass

class ValidationError(V15Error):
    """Raised when validation checks fail."""
    pass

class FeatureCorrelationWarning(UserWarning):
    """Warning when features are highly correlated."""
    pass
