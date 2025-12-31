"""
Base classes for modular ML architecture
Defines abstract interfaces for plug-and-play components
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import pandas as pd
import torch


class DataFeed(ABC):
    """
    Abstract base class for data sources
    Allows swapping between CSV, IBKR, Alpha Vantage, etc.
    """

    @abstractmethod
    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data for a symbol"""
        pass

    @abstractmethod
    def get_latest_data(self, symbol: str, bars: int = 100) -> pd.DataFrame:
        """Get recent data for real-time predictions"""
        pass

    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data integrity (no nulls, zeros, gaps)"""
        pass


class FeatureExtractor(ABC):
    """
    Abstract base class for feature engineering
    Modular design allows adding new indicators/patterns
    """

    @abstractmethod
    def extract_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Extract features from raw OHLCV data"""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature column names"""
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return total number of features"""
        pass


class EventHandler(ABC):
    """
    Abstract base class for event data
    Pluggable system for earnings, macro events, etc.
    """

    @abstractmethod
    def load_events(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load events within date range"""
        pass

    @abstractmethod
    def get_events_for_date(self, date: str, lookback_days: int = 7) -> List[Dict]:
        """Get events around a specific date"""
        pass

    @abstractmethod
    def embed_events(self, events: List[Dict]) -> torch.Tensor:
        """Convert events to tensor embeddings"""
        pass


class ModelBase(ABC):
    """
    Abstract base class for ML models
    Allows swapping between LNN, LSTM, Transformer, etc.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through model"""
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, h: torch.Tensor = None) -> Dict[str, Any]:
        """Generate predictions with probabilities"""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str, metadata: Dict = None):
        """Save model weights and metadata"""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> Dict:
        """Load model weights and return metadata"""
        pass

    @abstractmethod
    def update_online(self, x: torch.Tensor, y: torch.Tensor, lr: float = 0.001):
        """Perform online learning update"""
        pass


class PredictionDatabase(ABC):
    """
    Abstract base class for prediction logging
    Allows swapping between SQLite, PostgreSQL, cloud DB
    """

    @abstractmethod
    def log_prediction(self, prediction: Dict[str, Any]):
        """Log a prediction to database"""
        pass

    @abstractmethod
    def get_prediction(self, prediction_id: int) -> Dict[str, Any]:
        """Retrieve a specific prediction"""
        pass

    @abstractmethod
    def update_actual(self, prediction_id: int, actual_high: float, actual_low: float):
        """Update prediction with actual values"""
        pass

    @abstractmethod
    def get_accuracy_metrics(self, timeframe: str = None) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        pass

    @abstractmethod
    def get_error_patterns(self, limit: int = 100) -> pd.DataFrame:
        """Get predictions with largest errors for analysis"""
        pass
