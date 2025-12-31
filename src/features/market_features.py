"""
Market Feature Extractor for AutoTrade v7.0

Extracts market-based features including:
  - Price features (12): close, returns, volatility, normalized
  - RSI features (24): 4 timeframes × 3 metrics × 2 symbols (reduced from 11 TF)
  - Correlation (5): SPY-TSLA correlation and divergence
  - Cycle features (4): 52-week high/low, mega channel
  - Volume (2): Volume ratios
  - Time features (4): Hour, day, month, day-of-week
  - Binary flags (13): Day-of-week one-hot, volatile regime flags

Total: ~64 features (reduced from 114 in original system)

v7.0 Changes:
  - RSI reduced from 11 timeframes to 4 (5min, 1h, 4h, daily)
  - Removed redundant price metrics
  - Config-driven timeframe selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

from src.core.indicators import RSICalculator
from src.errors import FeatureExtractionError, InsufficientDataError
from src.monitoring import MetricsTracker
from config import FeatureConfig

logger = logging.getLogger(__name__)


class MarketFeatureExtractor:
    """
    Extract market and technical indicator features.

    Features:
      - Price: Returns, volatility, normalized price
      - RSI: Multi-timeframe RSI (configurable)
      - Volume: Relative volume, volume ratios
      - Correlation: SPY-TSLA correlation
      - Time: Hour, day, month encoding
      - Cycles: 52-week high/low

    Example:
        extractor = MarketFeatureExtractor(config)
        features = extractor.extract(df, symbols=['tsla', 'spy'])
    """

    def __init__(self, config: FeatureConfig, metrics: Optional[MetricsTracker] = None):
        """
        Initialize market feature extractor.

        Args:
            config: Feature configuration
            metrics: Optional metrics tracker
        """
        self.config = config
        self.metrics = metrics or MetricsTracker()
        self.rsi_calc = RSICalculator(period=14, oversold=30, overbought=70)

        # Get RSI timeframes from config (reduced to 4 in v7.0 minimal)
        self.rsi_timeframes = config.rsi_timeframes

        logger.info(f"MarketFeatureExtractor initialized: "
                   f"{len(self.rsi_timeframes)} RSI timeframes")

    def extract(
        self,
        df: pd.DataFrame,
        symbols: List[str] = ['tsla', 'spy'],
        mode: str = 'batch'
    ) -> pd.DataFrame:
        """
        Extract all market features.

        Args:
            df: DataFrame with OHLCV data at 5min resolution
            symbols: List of symbols (default: ['tsla', 'spy'])
            mode: 'batch' (full history) or 'streaming' (latest only)

        Returns:
            DataFrame with market features

        Raises:
            InsufficientDataError: Not enough data
            FeatureExtractionError: Extraction failed
        """
        if len(df) < 50:
            raise InsufficientDataError(
                f"Need at least 50 bars for market features, got {len(df)}"
            )

        with self.metrics.timer('market_features'):
            try:
                features = []

                # Price features (per symbol)
                for symbol in symbols:
                    price_features = self._extract_price_features(df, symbol)
                    features.append(price_features)

                # RSI features (per symbol, per timeframe)
                rsi_features = self._extract_rsi_features(df, symbols)
                features.append(rsi_features)

                # Volume features (per symbol)
                for symbol in symbols:
                    volume_features = self._extract_volume_features(df, symbol)
                    features.append(volume_features)

                # Correlation features (multi-symbol)
                if 'tsla' in symbols and 'spy' in symbols:
                    corr_features = self._extract_correlation_features(df)
                    features.append(corr_features)

                # Cycle features (52-week high/low)
                for symbol in symbols:
                    cycle_features = self._extract_cycle_features(df, symbol)
                    features.append(cycle_features)

                # Time features (timestamp encoding)
                time_features = self._extract_time_features(df)
                features.append(time_features)

                # Concatenate all features
                result = pd.concat(features, axis=1)

                logger.info(f"Market features extracted: {result.shape[1]} features")
                return result

            except Exception as e:
                logger.error(f"Market feature extraction failed: {e}")
                raise FeatureExtractionError(
                    "Failed to extract market features"
                ) from e

    def _extract_price_features(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Extract price-based features for a symbol.

        Features (12 per symbol):
          - close_price (raw)
          - returns_1bar, returns_5bar, returns_20bar
          - volatility_20bar, volatility_60bar
          - price_normalized_52w (0-1 range)
          - distance_from_52w_high, distance_from_52w_low
          - is_near_52w_high, is_near_52w_low
          - price_momentum_20bar, price_acceleration_20bar
        """
        prefix = f'{symbol}'
        features = {}

        # Get price columns
        close = df[f'{symbol}_close']
        high = df[f'{symbol}_high']
        low = df[f'{symbol}_low']

        # Raw close price
        features[f'{prefix}_close_price'] = close

        # Returns (various windows)
        features[f'{prefix}_returns_1bar'] = close.pct_change(1)
        features[f'{prefix}_returns_5bar'] = close.pct_change(5)
        features[f'{prefix}_returns_20bar'] = close.pct_change(20)

        # Volatility (rolling std of returns)
        returns = close.pct_change()
        features[f'{prefix}_volatility_20bar'] = returns.rolling(20).std()
        features[f'{prefix}_volatility_60bar'] = returns.rolling(60).std()

        # 52-week (252 days * 78 bars/day = ~19,656 5min bars) cycle features
        window_52w = min(19656, len(df))
        high_52w = high.rolling(window_52w, min_periods=20).max()
        low_52w = low.rolling(window_52w, min_periods=20).min()

        # Normalized price (0 = 52w low, 1 = 52w high)
        price_range = high_52w - low_52w
        price_range = price_range.replace(0, np.nan)  # Avoid division by zero
        features[f'{prefix}_price_normalized_52w'] = (
            (close - low_52w) / price_range
        ).fillna(0.5)

        # Distance from 52w high/low (percentage)
        features[f'{prefix}_distance_from_52w_high'] = (
            (high_52w - close) / close * 100
        ).fillna(0)
        features[f'{prefix}_distance_from_52w_low'] = (
            (close - low_52w) / close * 100
        ).fillna(0)

        # Near 52w high/low flags (within 2%)
        features[f'{prefix}_is_near_52w_high'] = (
            features[f'{prefix}_distance_from_52w_high'] < 2.0
        ).astype(float)
        features[f'{prefix}_is_near_52w_low'] = (
            features[f'{prefix}_distance_from_52w_low'] < 2.0
        ).astype(float)

        # Momentum (rate of change)
        features[f'{prefix}_price_momentum_20bar'] = (
            close.rolling(20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
        )

        # Acceleration (2nd derivative)
        momentum = features[f'{prefix}_price_momentum_20bar']
        features[f'{prefix}_price_acceleration_20bar'] = momentum.diff(5)

        # Fill NaN values
        result = pd.DataFrame(features, index=df.index).fillna(method='ffill').fillna(0)
        return result

    def _extract_rsi_features(
        self,
        df: pd.DataFrame,
        symbols: List[str]
    ) -> pd.DataFrame:
        """
        Extract RSI features for configured timeframes.

        v7.0: Reduced from 11 timeframes to 4 (5min, 1h, 4h, daily)

        Features per symbol per timeframe:
          - rsi_value
          - rsi_oversold (binary)
          - rsi_overbought (binary)

        Total: 4 TF × 3 metrics × 2 symbols = 24 features
        """
        features = {}

        for symbol in symbols:
            symbol_df = df[[c for c in df.columns if c.startswith(f'{symbol}_')]].rename(
                columns=lambda c: c.replace(f'{symbol}_', '')
            )

            for timeframe in self.rsi_timeframes:
                prefix = f'{symbol}_rsi_{timeframe}'

                # Resample to target timeframe (simplified - assumes df is 5min)
                if timeframe == '5min':
                    resampled = symbol_df
                elif timeframe == '1h':
                    resampled = symbol_df.resample('1h').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'volume': 'sum'
                    }).dropna()
                elif timeframe == '4h':
                    resampled = symbol_df.resample('4h').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'volume': 'sum'
                    }).dropna()
                elif timeframe == 'daily':
                    resampled = symbol_df.resample('1D').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'volume': 'sum'
                    }).dropna()
                else:
                    logger.warning(f"Unknown timeframe {timeframe}, skipping")
                    continue

                if len(resampled) < 20:
                    # Insufficient data for RSI
                    features[f'{prefix}_value'] = pd.Series(50.0, index=df.index)
                    features[f'{prefix}_oversold'] = pd.Series(0.0, index=df.index)
                    features[f'{prefix}_overbought'] = pd.Series(0.0, index=df.index)
                    continue

                # Calculate RSI
                try:
                    rsi_series = self.rsi_calc.calculate_rsi(resampled, column='close')

                    # Resample back to 5min (forward fill)
                    rsi_5min = rsi_series.reindex(df.index, method='ffill').fillna(50.0)

                    features[f'{prefix}_value'] = rsi_5min
                    features[f'{prefix}_oversold'] = (rsi_5min < 30).astype(float)
                    features[f'{prefix}_overbought'] = (rsi_5min > 70).astype(float)

                except Exception as e:
                    logger.warning(f"RSI calculation failed for {symbol}_{timeframe}: {e}")
                    features[f'{prefix}_value'] = pd.Series(50.0, index=df.index)
                    features[f'{prefix}_oversold'] = pd.Series(0.0, index=df.index)
                    features[f'{prefix}_overbought'] = pd.Series(0.0, index=df.index)

        result = pd.DataFrame(features, index=df.index)
        return result

    def _extract_volume_features(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Extract volume-based features.

        Features (2 per symbol):
          - volume_ratio_20bar (current vol / avg vol)
          - volume_trend_20bar (volume momentum)
        """
        prefix = f'{symbol}'
        features = {}

        volume = df[f'{symbol}_volume']

        # Volume ratio (current / 20-bar average)
        avg_volume = volume.rolling(20, min_periods=1).mean()
        features[f'{prefix}_volume_ratio_20bar'] = (
            volume / avg_volume
        ).fillna(1.0)

        # Volume trend (is volume increasing?)
        features[f'{prefix}_volume_trend_20bar'] = (
            volume.rolling(20).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0,
                raw=True
            )
        ).fillna(0)

        result = pd.DataFrame(features, index=df.index).fillna(method='ffill').fillna(0)
        return result

    def _extract_correlation_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract correlation features between TSLA and SPY.

        Features (5):
          - tsla_spy_correlation_20bar
          - tsla_spy_correlation_60bar
          - tsla_spy_divergence (returns diff)
          - tsla_outperforming_spy
          - tsla_underperforming_spy
        """
        features = {}

        tsla_close = df['tsla_close']
        spy_close = df['spy_close']

        tsla_returns = tsla_close.pct_change()
        spy_returns = spy_close.pct_change()

        # Rolling correlation (20-bar and 60-bar)
        features['tsla_spy_correlation_20bar'] = (
            tsla_returns.rolling(20).corr(spy_returns).fillna(0)
        )
        features['tsla_spy_correlation_60bar'] = (
            tsla_returns.rolling(60).corr(spy_returns).fillna(0)
        )

        # Divergence (difference in returns)
        features['tsla_spy_divergence'] = tsla_returns - spy_returns

        # Outperformance flags
        features['tsla_outperforming_spy'] = (
            tsla_returns > spy_returns
        ).astype(float)
        features['tsla_underperforming_spy'] = (
            tsla_returns < spy_returns
        ).astype(float)

        result = pd.DataFrame(features, index=df.index).fillna(method='ffill').fillna(0)
        return result

    def _extract_cycle_features(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Extract long-term cycle features.

        Features (4 per symbol):
          - mega_channel_position (position in 252-day channel)
          - mega_channel_slope
          - days_since_52w_high
          - days_since_52w_low
        """
        prefix = f'{symbol}'
        features = {}

        close = df[f'{symbol}_close']
        high = df[f'{symbol}_high']
        low = df[f'{symbol}_low']

        # 252-day mega channel (at 5min resolution = ~19,656 bars)
        window_252d = min(19656, len(df))

        if window_252d >= 100:
            # Calculate linear regression for mega channel
            window_data = close.iloc[-window_252d:]
            x = np.arange(len(window_data))
            y = window_data.values

            # Fit linear regression
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]

            # Channel lines
            y_pred = slope * x + intercept
            residuals = y - y_pred
            std_dev = residuals.std()

            upper_line = y_pred + 2.0 * std_dev
            lower_line = y_pred - 2.0 * std_dev

            # Position in mega channel
            current_price = close.iloc[-1]
            current_pred = slope * (len(window_data) - 1) + intercept
            current_upper = current_pred + 2.0 * std_dev
            current_lower = current_pred - 2.0 * std_dev

            if current_upper > current_lower:
                position = (current_price - current_lower) / (current_upper - current_lower)
            else:
                position = 0.5

            features[f'{prefix}_mega_channel_position'] = pd.Series(
                np.clip(position, 0, 1), index=df.index
            )
            features[f'{prefix}_mega_channel_slope'] = pd.Series(
                slope / current_price * 100 if current_price > 0 else 0,
                index=df.index
            )
        else:
            features[f'{prefix}_mega_channel_position'] = pd.Series(0.5, index=df.index)
            features[f'{prefix}_mega_channel_slope'] = pd.Series(0.0, index=df.index)

        # Days since 52-week high/low
        window_52w = min(19656, len(df))
        high_52w = high.rolling(window_52w, min_periods=20).max()
        low_52w = low.rolling(window_52w, min_periods=20).min()

        # Find bars since high/low
        is_at_high = (high >= high_52w)
        is_at_low = (low <= low_52w)

        # Count bars since last occurrence (convert to days: bars / 78)
        bars_since_high = pd.Series(range(len(df)), index=df.index)
        bars_since_high[is_at_high] = 0
        bars_since_high = bars_since_high.groupby((is_at_high != is_at_high.shift()).cumsum()).cumcount()

        bars_since_low = pd.Series(range(len(df)), index=df.index)
        bars_since_low[is_at_low] = 0
        bars_since_low = bars_since_low.groupby((is_at_low != is_at_low.shift()).cumsum()).cumcount()

        features[f'{prefix}_days_since_52w_high'] = bars_since_high / 78.0  # 78 5min bars per day
        features[f'{prefix}_days_since_52w_low'] = bars_since_low / 78.0

        result = pd.DataFrame(features, index=df.index).fillna(method='ffill').fillna(0)
        return result

    def _extract_time_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract time-based features from index.

        Features (17):
          - hour (0-23)
          - day_of_month (1-31)
          - month (1-12)
          - day_of_week (0-6)
          - is_monday, is_tuesday, ..., is_friday (5 flags)
          - is_market_open (9:30-16:00 ET)
          - is_first_hour, is_last_hour
          - minutes_into_day
        """
        features = {}

        # Extract time components
        features['hour'] = df.index.hour.values.astype(float)
        features['day_of_month'] = df.index.day.values.astype(float)
        features['month'] = df.index.month.values.astype(float)
        features['day_of_week'] = df.index.dayofweek.values.astype(float)

        # Day-of-week one-hot encoding
        features['is_monday'] = (df.index.dayofweek == 0).astype(float)
        features['is_tuesday'] = (df.index.dayofweek == 1).astype(float)
        features['is_wednesday'] = (df.index.dayofweek == 2).astype(float)
        features['is_thursday'] = (df.index.dayofweek == 3).astype(float)
        features['is_friday'] = (df.index.dayofweek == 4).astype(float)

        # Market hours (9:30-16:00 ET)
        hour = df.index.hour.values
        minute = df.index.minute.values
        time_minutes = hour * 60 + minute

        features['is_market_open'] = (
            (time_minutes >= 9 * 60 + 30) & (time_minutes < 16 * 60)
        ).astype(float)

        features['is_first_hour'] = (
            (time_minutes >= 9 * 60 + 30) & (time_minutes < 10 * 60 + 30)
        ).astype(float)

        features['is_last_hour'] = (
            (time_minutes >= 15 * 60) & (time_minutes < 16 * 60)
        ).astype(float)

        # Minutes into trading day (0 = 9:30am)
        features['minutes_into_day'] = np.clip(time_minutes - (9 * 60 + 30), 0, 390)  # 6.5 hours

        result = pd.DataFrame(features, index=df.index)
        return result


def extract_market_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    symbols: List[str] = ['tsla', 'spy'],
    mode: str = 'batch',
    metrics: Optional[MetricsTracker] = None
) -> pd.DataFrame:
    """
    Convenience function to extract all market features.

    Args:
        df: DataFrame with OHLCV data at 5min resolution
        config: Feature configuration
        symbols: List of symbols to process
        mode: 'batch' or 'streaming'
        metrics: Optional metrics tracker

    Returns:
        DataFrame with all market features

    Example:
        >>> config = get_feature_config()
        >>> df = load_5min_data()
        >>> features = extract_market_features(df, config)
    """
    extractor = MarketFeatureExtractor(config, metrics)
    return extractor.extract(df, symbols, mode)
