"""
VIX Feature Extractor for AutoTrade v7.0

Extracts market volatility regime features from VIX (CBOE Volatility Index).

Features (15 total):
  - vix_close (raw VIX level)
  - vix_percentile_252d (where VIX is in 1-year range)
  - vix_regime: low(<15), normal(15-25), elevated(25-35), high(>35)
  - vix_spike (rapid increase)
  - vix_declining (cooling off)
  - vix_roc_5d, vix_roc_20d (rate of change)
  - vix_z_score_60d (statistical deviation)
  - vix_above_20, vix_above_30, vix_above_40 (threshold flags)
  - vix_moving_avg_20d, vix_moving_avg_60d
  - vix_distance_from_ma (current vs moving average)

Graceful Degradation:
  - If VIX data unavailable, returns neutral values (vix=20, normal regime)
  - Logs warning but does not fail extraction
  - Uses GracefulDegradation.get_zero_vix_features() as fallback
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging
from datetime import datetime, timedelta

from src.errors import VIXFeaturesError
from src.monitoring import MetricsTracker
from src.errors import GracefulDegradation
from config import FeatureConfig

logger = logging.getLogger(__name__)


class VIXFeatureExtractor:
    """
    Extract VIX-based market regime features.

    VIX provides external volatility signal that helps model understand
    market conditions (calm vs volatile).

    Example:
        extractor = VIXFeatureExtractor(config)
        features = extractor.extract(df, vix_data)
    """

    def __init__(self, config: FeatureConfig, metrics: Optional[MetricsTracker] = None):
        """
        Initialize VIX feature extractor.

        Args:
            config: Feature configuration
            metrics: Optional metrics tracker
        """
        self.config = config
        self.metrics = metrics or MetricsTracker()
        self.recovery = GracefulDegradation()

        logger.info("VIXFeatureExtractor initialized")

    def extract(
        self,
        df: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None,
        mode: str = 'batch'
    ) -> pd.DataFrame:
        """
        Extract VIX features.

        Args:
            df: Main DataFrame (for index alignment)
            vix_data: DataFrame with VIX OHLC data (optional)
                      If None, will attempt to fetch from yfinance
            mode: 'batch' or 'streaming'

        Returns:
            DataFrame with VIX features (same index as df)

        Raises:
            VIXFeaturesError: Only if fatal error (otherwise returns fallback)
        """
        with self.metrics.timer('vix_features'):
            try:
                # If no VIX data provided, try to fetch
                if vix_data is None:
                    vix_data = self._fetch_vix_data(df.index[0], df.index[-1])

                if vix_data is None or len(vix_data) == 0:
                    logger.warning("VIX data unavailable, using fallback")
                    return self._get_fallback_features(df)

                # Extract features from VIX data
                features = self._calculate_vix_features(df, vix_data)

                logger.info(f"VIX features extracted: {features.shape[1]} features")
                return features

            except Exception as e:
                logger.error(f"VIX feature extraction failed: {e}")
                logger.warning("Using fallback VIX features")
                return self._get_fallback_features(df)

    def _fetch_vix_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch VIX data from yfinance.

        Args:
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with VIX OHLC data or None if failed
        """
        try:
            import yfinance as yf

            # Fetch VIX with some buffer (252 trading days = 1 year)
            buffer_start = start_date - timedelta(days=400)

            logger.info(f"Fetching VIX data from {buffer_start.date()} to {end_date.date()}")

            vix = yf.Ticker("^VIX")
            vix_data = vix.history(
                start=buffer_start,
                end=end_date + timedelta(days=1),
                interval='1d'
            )

            if vix_data.empty:
                logger.warning("VIX fetch returned empty data")
                return None

            # Rename columns to lowercase
            vix_data.columns = vix_data.columns.str.lower()

            logger.info(f"Fetched {len(vix_data)} days of VIX data")
            return vix_data

        except ImportError:
            logger.warning("yfinance not installed, cannot fetch VIX")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch VIX data: {e}")
            return None

    def _calculate_vix_features(
        self,
        df: pd.DataFrame,
        vix_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate all VIX features.

        Aligns VIX (daily data) to df index (5min data) using forward fill.
        """
        features = {}

        vix_close = vix_data['close']

        # === Raw VIX Level (1) ===
        features['vix_close'] = vix_close

        # === VIX Percentile (1) ===
        # Where is VIX in 252-day (1 year) range?
        window_252d = min(252, len(vix_close))
        vix_min_252d = vix_close.rolling(window_252d, min_periods=20).min()
        vix_max_252d = vix_close.rolling(window_252d, min_periods=20).max()

        vix_range = vix_max_252d - vix_min_252d
        vix_range = vix_range.replace(0, np.nan)

        features['vix_percentile_252d'] = (
            (vix_close - vix_min_252d) / vix_range * 100
        ).fillna(50.0)

        # === VIX Regime Flags (4) ===
        # Low: <15, Normal: 15-25, Elevated: 25-35, High: >35
        features['vix_regime_low'] = (vix_close < 15).astype(float)
        features['vix_regime_normal'] = ((vix_close >= 15) & (vix_close < 25)).astype(float)
        features['vix_regime_elevated'] = ((vix_close >= 25) & (vix_close < 35)).astype(float)
        features['vix_regime_high'] = (vix_close >= 35).astype(float)

        # === VIX Spike/Decline (2) ===
        # Spike: rapid increase (>10% in 5 days)
        vix_change_5d = vix_close.pct_change(5)
        features['vix_spike'] = (vix_change_5d > 0.10).astype(float)

        # Declining: cooling off (< -5% in 5 days)
        features['vix_declining'] = (vix_change_5d < -0.05).astype(float)

        # === Rate of Change (2) ===
        features['vix_roc_5d'] = vix_close.pct_change(5) * 100  # As percentage
        features['vix_roc_20d'] = vix_close.pct_change(20) * 100

        # === Z-Score (1) ===
        # How many standard deviations from 60-day mean?
        vix_mean_60d = vix_close.rolling(60, min_periods=20).mean()
        vix_std_60d = vix_close.rolling(60, min_periods=20).std()

        features['vix_z_score_60d'] = (
            (vix_close - vix_mean_60d) / vix_std_60d
        ).fillna(0)

        # === Threshold Flags (3) ===
        features['vix_above_20'] = (vix_close > 20).astype(float)
        features['vix_above_30'] = (vix_close > 30).astype(float)
        features['vix_above_40'] = (vix_close > 40).astype(float)

        # === Moving Averages (2) ===
        features['vix_ma_20d'] = vix_close.rolling(20, min_periods=5).mean()
        features['vix_ma_60d'] = vix_close.rolling(60, min_periods=20).mean()

        # === Distance from MA (1) ===
        features['vix_distance_from_ma'] = (
            (vix_close - features['vix_ma_20d']) / features['vix_ma_20d'] * 100
        ).fillna(0)

        # Convert to DataFrame
        vix_features = pd.DataFrame(features, index=vix_data.index)

        # Fill any remaining NaN
        vix_features = vix_features.fillna(method='ffill').fillna(method='bfill')

        # Align to main df index (forward fill daily VIX to 5min bars)
        aligned = vix_features.reindex(df.index, method='ffill')

        # Fill any gaps (weekends, holidays) with last known value
        aligned = aligned.fillna(method='ffill')

        # If still NaN (before first VIX date), use fallback
        fallback = self.recovery.get_zero_vix_features()
        for col in aligned.columns:
            if aligned[col].isna().any():
                # Use fallback value for this column
                fallback_key = col.replace('vix_', '')
                if fallback_key in fallback:
                    aligned[col].fillna(fallback[fallback_key], inplace=True)
                else:
                    aligned[col].fillna(0, inplace=True)

        return aligned

    def _get_fallback_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get fallback VIX features when data unavailable.

        Returns neutral values (VIX=20, normal regime).
        """
        fallback = self.recovery.get_zero_vix_features()

        # Create DataFrame with fallback values (using keys from GracefulDegradation)
        features = {
            'vix_close': fallback.get('vix_level', 20.0),
            'vix_percentile_252d': fallback.get('vix_percentile_252d', 50.0),
            'vix_regime_low': 0.0,  # VIX=20 is normal, not low
            'vix_regime_normal': 1.0,  # Default to normal
            'vix_regime_elevated': 0.0,
            'vix_regime_high': 0.0,
            'vix_spike': fallback.get('vix_spike', 0.0),
            'vix_declining': 0.0,
            'vix_roc_5d': fallback.get('vix_change_5d', 0.0),
            'vix_roc_20d': fallback.get('vix_trend_20d', 0.0),
            'vix_z_score_60d': 0.0,
            'vix_above_20': fallback.get('vix_above_20', 0.0),
            'vix_above_30': fallback.get('vix_above_30', 0.0),
            'vix_above_40': 0.0,
            'vix_ma_20d': fallback.get('vix_level', 20.0),
            'vix_ma_60d': fallback.get('vix_level', 20.0),
            'vix_distance_from_ma': 0.0,
        }

        result = pd.DataFrame(features, index=df.index)

        logger.info("Using fallback VIX features (neutral regime)")
        return result


def extract_vix_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    vix_data: Optional[pd.DataFrame] = None,
    mode: str = 'batch',
    metrics: Optional[MetricsTracker] = None
) -> pd.DataFrame:
    """
    Convenience function to extract VIX features.

    Args:
        df: Main DataFrame (for index alignment)
        config: Feature configuration
        vix_data: Optional VIX OHLC data (if None, will fetch)
        mode: 'batch' or 'streaming'
        metrics: Optional metrics tracker

    Returns:
        DataFrame with VIX features

    Example:
        >>> config = get_feature_config()
        >>> df = load_5min_data()
        >>> vix_features = extract_vix_features(df, config)
    """
    extractor = VIXFeatureExtractor(config, metrics)
    return extractor.extract(df, vix_data, mode)
