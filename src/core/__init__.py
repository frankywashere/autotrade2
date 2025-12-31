"""
Core domain logic for AutoTrade v7.0

Pure business logic with no dependencies on ML frameworks.
100% testable without loading models or data.

Modules:
- channel: LinearRegressionChannel calculation
- indicators: RSI, technical indicators
- market_data: OHLC data structures
"""

from .channel import LinearRegressionChannel, ChannelData
from .indicators import RSICalculator

__all__ = [
    'LinearRegressionChannel',
    'ChannelData',
    'RSICalculator',
]
