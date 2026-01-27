"""
V15 Channel Prediction System

A complete rewrite with:
- 8,665 TF-aware features with explicit individual weights
- Partial bar support (no stale TF data)
- Loud failures (no silent exceptions)
- Feature correlation analysis
- Inference module for model loading and predictions
- Dashboard for visualization and monitoring

Usage:
    # Scan data for features
    python -m v15.pipeline scan --data-dir data --output samples.pkl

    # Train model
    python -m v15.pipeline train --samples samples.pkl --output model.pt

    # Analyze features
    python -m v15.pipeline analyze --samples samples.pkl

    # Load model for inference
    predictor = v15.load_predictor('model.pt')
"""

from .config import (
    TIMEFRAMES,
    STANDARD_WINDOWS,
    TOTAL_FEATURES,
    N_TIMEFRAMES,
    N_WINDOWS,
)

from .exceptions import (
    V15Error,
    FeatureExtractionError,
    InvalidFeatureError,
    DataLoadError,
    ResamplingError,
    ModelError,
)

from .dtypes import ChannelSample, ChannelLabels

# Lazy imports for optional components
def load_market_data(data_dir: str):
    """Load market data from directory."""
    from .data import load_market_data as _load
    return _load(data_dir)

def create_model(**kwargs):
    """Create V15 model."""
    from .models import create_model as _create
    return _create(**kwargs)

def scan_channels(tsla_df, spy_df, vix_df, **kwargs):
    """
    DEPRECATED: Python scanner has been removed.

    Use the C++ scanner instead for 10x faster performance:
        cd v15_cpp/build && ./v15_scanner --data-dir ../../data --output samples.bin

    Then load results with v15.binary_loader:
        from v15.binary_loader import load_samples
        version, num_samples, num_features, samples = load_samples('samples.bin')
    """
    raise NotImplementedError(
        "Python scanner has been removed. Use v15_cpp/build/v15_scanner instead. "
        "See v15.scan_channels.__doc__ for details."
    )

def load_predictor(checkpoint_path: str):
    """Load a trained model for inference."""
    from .inference import Predictor
    return Predictor.load(checkpoint_path)

__version__ = '15.0.0'

__all__ = [
    # Config
    'TIMEFRAMES',
    'STANDARD_WINDOWS',
    'TOTAL_FEATURES',
    'N_TIMEFRAMES',
    'N_WINDOWS',
    # Exceptions
    'V15Error',
    'FeatureExtractionError',
    'InvalidFeatureError',
    'DataLoadError',
    'ResamplingError',
    'ModelError',
    # Types
    'ChannelSample',
    'ChannelLabels',
    # Functions
    'load_market_data',
    'create_model',
    'scan_channels',
    'load_predictor',
]
