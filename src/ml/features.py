"""
Feature extraction system for ML model
Leverages existing Stage 1 components (channels, RSI) plus new features
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config
from src.linear_regression import LinearRegressionChannel
from src.rsi_calculator import RSICalculator
from .base import FeatureExtractor


class TradingFeatureExtractor(FeatureExtractor):
    """
    Comprehensive feature extractor for stock trading
    Combines technical indicators, channel patterns, correlations, and cycles
    """

    def __init__(self):
        self.channel_calc = LinearRegressionChannel()
        self.rsi_calc = RSICalculator()
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        """Build list of all feature names"""
        features = []

        # Price features (both SPY and TSLA)
        for symbol in ['spy', 'tsla']:
            features.extend([
                f'{symbol}_close',
                f'{symbol}_returns',
                f'{symbol}_log_returns',
                f'{symbol}_volatility_10',
                f'{symbol}_volatility_50',
            ])

        # TSLA Channel features (multi-timeframe) - RENAMED for consistency
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.extend([
                f'tsla_channel_{tf}_position',  # 0-1 position in channel
                f'tsla_channel_{tf}_upper_dist',  # Distance to upper
                f'tsla_channel_{tf}_lower_dist',  # Distance to lower
                f'tsla_channel_{tf}_slope',
                f'tsla_channel_{tf}_stability',
                f'tsla_channel_{tf}_ping_pongs',
                f'tsla_channel_{tf}_r_squared',
            ])

        # SPY Channel features (multi-timeframe) - NEW in v3.4
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.extend([
                f'spy_channel_{tf}_position',  # 0-1 position in channel
                f'spy_channel_{tf}_upper_dist',  # Distance to upper
                f'spy_channel_{tf}_lower_dist',  # Distance to lower
                f'spy_channel_{tf}_slope',
                f'spy_channel_{tf}_stability',
                f'spy_channel_{tf}_ping_pongs',
                f'spy_channel_{tf}_r_squared',
            ])

        # TSLA RSI features (multi-timeframe) - RENAMED for consistency
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.extend([
                f'tsla_rsi_{tf}',
                f'tsla_rsi_{tf}_oversold',  # Binary
                f'tsla_rsi_{tf}_overbought',  # Binary
            ])

        # SPY RSI features (multi-timeframe) - NEW in v3.4
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.extend([
                f'spy_rsi_{tf}',
                f'spy_rsi_{tf}_oversold',  # Binary
                f'spy_rsi_{tf}_overbought',  # Binary
            ])

        # SPY-TSLA correlation features
        features.extend([
            'correlation_10',  # 10-bar rolling correlation
            'correlation_50',
            'correlation_200',
            'divergence',  # SPY up, TSLA down (or vice versa)
            'divergence_magnitude',
        ])

        # Larger cycle features
        features.extend([
            'distance_from_52w_high',
            'distance_from_52w_low',
            'within_mega_channel',  # Binary: in 3-4 year channel
            'mega_channel_position',
        ])

        # Volume features
        features.extend([
            'tsla_volume_ratio',  # Current / average
            'spy_volume_ratio',
        ])

        # Time features
        features.extend([
            'hour_of_day',
            'day_of_week',
            'day_of_month',
            'month_of_year',
        ])

        self.feature_names = features

    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return self.feature_names

    def get_feature_dim(self) -> int:
        """Return total number of features"""
        return len(self.feature_names)

    def extract_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Extract all 245 features from aligned SPY-TSLA data (OPTIMIZED v3.4).

        df should have columns: spy_open, spy_high, spy_low, spy_close, spy_volume,
                                tsla_open, tsla_high, tsla_low, tsla_close, tsla_volume

        Returns DataFrame with 245 columns:
        - 10 price features
        - 154 channel features (77 TSLA + 77 SPY)
        - 66 RSI features (33 TSLA + 33 SPY)
        - 5 correlation features
        - 4 cycle features
        - 2 volume features
        - 4 time features
        """
        # Extract each category (each returns its own DataFrame)
        price_df = self._extract_price_features(df)
        channel_df = self._extract_channel_features(df)
        rsi_df = self._extract_rsi_features(df)
        correlation_df = self._extract_correlation_features(df)
        cycle_df = self._extract_cycle_features(df)
        volume_df = self._extract_volume_features(df)
        time_df = self._extract_time_features(df)

        # Concat all at once (FAST! No DataFrame fragmentation)
        features_df = pd.concat([
            price_df,
            channel_df,
            rsi_df,
            correlation_df,
            cycle_df,
            volume_df,
            time_df
        ], axis=1)

        # Fill NaNs
        features_df = features_df.bfill().fillna(0)

        return features_df

    def _extract_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic price features. Returns DataFrame with 10 columns."""
        price_features = {}

        for symbol in ['spy', 'tsla']:
            close_col = f'{symbol}_close'
            price_features[close_col] = df[close_col]

            # Returns
            returns = df[close_col].pct_change()
            price_features[f'{symbol}_returns'] = returns
            price_features[f'{symbol}_log_returns'] = np.log(df[close_col] / df[close_col].shift(1))

            # Volatility (rolling std of returns)
            price_features[f'{symbol}_volatility_10'] = returns.rolling(10).std()
            price_features[f'{symbol}_volatility_50'] = returns.rolling(50).std()

        return pd.DataFrame(price_features, index=df.index)

    def _extract_channel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract linear regression channel features for multiple timeframes.
        NOW PROCESSES BOTH TSLA AND SPY (v3.4)
        Returns DataFrame with 154 columns (77 TSLA + 77 SPY).
        """
        channel_features = {}
        num_rows = len(df)

        # Resample to different timeframes
        timeframes = {
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': '1h',
            '2h': '2h',
            '3h': '3h',
            '4h': '4h',
            'daily': '1D',
            'weekly': '1W',
            'monthly': '1ME',
            '3month': '3ME'
        }

        # Process both TSLA and SPY
        for symbol in ['tsla', 'spy']:
            for tf_name, tf_rule in timeframes.items():
                # Resample symbol data
                symbol_df = df[[c for c in df.columns if c.startswith(f'{symbol}_')]].copy()
                symbol_df.columns = [c.replace(f'{symbol}_', '') for c in symbol_df.columns]

                resampled = symbol_df.resample(tf_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

                prefix = f'{symbol}_channel'

                if len(resampled) < 20:
                    # Not enough data - fill with zeros
                    for feat in ['position', 'upper_dist', 'lower_dist', 'slope', 'stability', 'ping_pongs', 'r_squared']:
                        channel_features[f'{prefix}_{tf_name}_{feat}'] = np.zeros(num_rows)
                    continue

                # Calculate channel
                lookback = min(168, len(resampled) - 1)
                try:
                    channel = self.channel_calc.calculate_channel(resampled, lookback, tf_name)
                    current_price = resampled['close'].iloc[-1]
                    position_data = self.channel_calc.get_channel_position(current_price, channel)

                    # Store features (broadcast scalar to all rows)
                    channel_features[f'{prefix}_{tf_name}_position'] = np.full(num_rows, position_data['position'])
                    channel_features[f'{prefix}_{tf_name}_upper_dist'] = np.full(num_rows, position_data['distance_to_upper_pct'])
                    channel_features[f'{prefix}_{tf_name}_lower_dist'] = np.full(num_rows, position_data['distance_to_lower_pct'])
                    channel_features[f'{prefix}_{tf_name}_slope'] = np.full(num_rows, channel.slope)
                    channel_features[f'{prefix}_{tf_name}_stability'] = np.full(num_rows, channel.stability_score)
                    channel_features[f'{prefix}_{tf_name}_ping_pongs'] = np.full(num_rows, channel.ping_pongs)
                    channel_features[f'{prefix}_{tf_name}_r_squared'] = np.full(num_rows, channel.r_squared)

                except Exception as e:
                    # Fill with zeros if calculation fails
                    for feat in ['position', 'upper_dist', 'lower_dist', 'slope', 'stability', 'ping_pongs', 'r_squared']:
                        channel_features[f'{prefix}_{tf_name}_{feat}'] = np.zeros(num_rows)

        return pd.DataFrame(channel_features, index=df.index)

    def _extract_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract RSI features for multiple timeframes.
        NOW PROCESSES BOTH TSLA AND SPY (v3.4)
        Returns DataFrame with 66 columns (33 TSLA + 33 SPY).
        """
        rsi_features = {}
        num_rows = len(df)

        timeframes = {
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': '1h',
            '2h': '2h',
            '3h': '3h',
            '4h': '4h',
            'daily': '1D',
            'weekly': '1W',
            'monthly': '1ME',
            '3month': '3ME'
        }

        # Process both TSLA and SPY
        for symbol in ['tsla', 'spy']:
            for tf_name, tf_rule in timeframes.items():
                # Resample symbol data
                symbol_df = df[[c for c in df.columns if c.startswith(f'{symbol}_')]].copy()
                symbol_df.columns = [c.replace(f'{symbol}_', '') for c in symbol_df.columns]

                resampled = symbol_df.resample(tf_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

                prefix = f'{symbol}_rsi'

                if len(resampled) < 20:
                    # Not enough data - fill with defaults
                    rsi_features[f'{prefix}_{tf_name}'] = np.full(num_rows, 50.0)
                    rsi_features[f'{prefix}_{tf_name}_oversold'] = np.zeros(num_rows)
                    rsi_features[f'{prefix}_{tf_name}_overbought'] = np.zeros(num_rows)
                    continue

                try:
                    rsi_data = self.rsi_calc.get_rsi_data(resampled)
                    rsi_value = rsi_data.value if rsi_data.value is not None else 50.0

                    # Store features (broadcast scalar to all rows)
                    rsi_features[f'{prefix}_{tf_name}'] = np.full(num_rows, rsi_value)
                    rsi_features[f'{prefix}_{tf_name}_oversold'] = np.full(num_rows, 1.0 if rsi_data.oversold else 0.0)
                    rsi_features[f'{prefix}_{tf_name}_overbought'] = np.full(num_rows, 1.0 if rsi_data.overbought else 0.0)

                except Exception:
                    # Fill with defaults
                    rsi_features[f'{prefix}_{tf_name}'] = np.full(num_rows, 50.0)
                    rsi_features[f'{prefix}_{tf_name}_oversold'] = np.zeros(num_rows)
                    rsi_features[f'{prefix}_{tf_name}_overbought'] = np.zeros(num_rows)

        return pd.DataFrame(rsi_features, index=df.index)

    def _extract_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract SPY-TSLA correlation and divergence features. Returns DataFrame with 5 columns."""
        spy_returns = df['spy_close'].pct_change()
        tsla_returns = df['tsla_close'].pct_change()

        correlation_features = {
            'correlation_10': spy_returns.rolling(10).corr(tsla_returns),
            'correlation_50': spy_returns.rolling(50).corr(tsla_returns),
            'correlation_200': spy_returns.rolling(200).corr(tsla_returns),
            'divergence': (((spy_returns > 0) & (tsla_returns < 0)) |
                          ((spy_returns < 0) & (tsla_returns > 0))).astype(float),
            'divergence_magnitude': abs(spy_returns - tsla_returns)
        }

        return pd.DataFrame(correlation_features, index=df.index)

    def _extract_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract larger cycle features (52-week highs/lows, mega channels). Returns DataFrame with 4 columns."""
        tsla_close = df['tsla_close']
        high_52w = tsla_close.rolling(252, min_periods=50).max()
        low_52w = tsla_close.rolling(252, min_periods=50).min()

        cycle_features = {
            'distance_from_52w_high': (high_52w - tsla_close) / high_52w,
            'distance_from_52w_low': (tsla_close - low_52w) / low_52w
        }

        # Mega channel: 3-4 year channels
        if len(df) > 756:  # 3 years of daily data (approx)
            lookback = min(1008, len(df))
            mega_high = tsla_close.rolling(lookback, min_periods=252).max()
            mega_low = tsla_close.rolling(lookback, min_periods=252).min()

            cycle_features['within_mega_channel'] = ((tsla_close >= mega_low * 0.9) &
                                                      (tsla_close <= mega_high * 1.1)).astype(float)
            cycle_features['mega_channel_position'] = (tsla_close - mega_low) / (mega_high - mega_low + 1e-8)
        else:
            cycle_features['within_mega_channel'] = np.zeros(len(df))
            cycle_features['mega_channel_position'] = np.full(len(df), 0.5)

        return pd.DataFrame(cycle_features, index=df.index)

    def _extract_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volume-based features. Returns DataFrame with 2 columns."""
        volume_features = {
            'tsla_volume_ratio': df['tsla_volume'] / df['tsla_volume'].rolling(20, min_periods=1).mean(),
            'spy_volume_ratio': df['spy_volume'] / df['spy_volume'].rolling(20, min_periods=1).mean()
        }

        return pd.DataFrame(volume_features, index=df.index)

    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features. Returns DataFrame with 4 columns."""
        time_features = {
            'hour_of_day': df.index.hour / 24.0,
            'day_of_week': df.index.dayofweek / 7.0,
            'day_of_month': df.index.day / 31.0,
            'month_of_year': df.index.month / 12.0
        }

        return pd.DataFrame(time_features, index=df.index)

    def create_sequences(self, features_df: pd.DataFrame, sequence_length: int = 168,
                        target_horizon: int = 24) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sequences for training
        Returns (X, y) where:
        - X: (num_sequences, sequence_length, num_features)
        - y: (num_sequences, 2) - [predicted_high, predicted_low] for next target_horizon bars
        """
        features = features_df.values
        num_samples = len(features) - sequence_length - target_horizon

        if num_samples <= 0:
            raise ValueError(f"Not enough data. Need at least {sequence_length + target_horizon} bars")

        X = []
        y = []

        for i in range(num_samples):
            # Input sequence
            seq = features[i:i + sequence_length]
            X.append(seq)

            # Target: high and low in next target_horizon bars
            future_window = features_df.iloc[i + sequence_length:i + sequence_length + target_horizon]
            target_high = future_window['tsla_close'].max()
            target_low = future_window['tsla_close'].min()
            y.append([target_high, target_low])

        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)

        return X, y
