"""
Channel Feature Extractor for AutoTrade v7.0

Extracts linear regression channel features across multiple windows and timeframes.
Implements config-driven feature selection with graceful degradation.

Features per window (31 total):
  - Position metrics (3): position, upper_dist, lower_dist
  - Raw slopes (3): close_slope, high_slope, low_slope
  - Normalized slopes (3): close_slope_pct, high_slope_pct, low_slope_pct
  - R-squared (4): close_r², high_r², low_r², avg_r²
  - Channel metrics (3): width_pct, slope_convergence, stability
  - Ping-pongs (4): 4 thresholds (0.5%, 1%, 2%, 3%)
  - Complete cycles (4): 4 thresholds (v6.0 bounce-based validity)
  - Direction flags (3): is_bull, is_bear, is_sideways
  - Quality (3): quality_score, is_valid, insufficient_data
  - Duration (1): bars in current channel

Architecture:
  - Uses LinearRegressionChannel from src/core/channel.py
  - Config-driven window selection (5 windows in v7.0 minimal)
  - Processes both TSLA and SPY
  - Parallel processing support for batch mode
  - Streaming mode for inference
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from src.core.channel import LinearRegressionChannel
from src.errors import ChannelFeaturesError, InsufficientDataError
from src.monitoring import MetricsTracker
from config import FeatureConfig

logger = logging.getLogger(__name__)


class ChannelFeatureExtractor:
    """
    Extract channel features for multiple windows and timeframes.

    Features are calculated using rolling windows of historical data.
    Each window size provides multi-scale channel information.

    Example:
        extractor = ChannelFeatureExtractor(config)
        features = extractor.extract_for_timeframe(
            df, symbol='tsla', timeframe='4h'
        )
    """

    def __init__(self, config: FeatureConfig, metrics: Optional[MetricsTracker] = None):
        """
        Initialize channel feature extractor.

        Args:
            config: Feature configuration (defines windows, validity criteria)
            metrics: Optional metrics tracker for monitoring
        """
        self.config = config
        self.metrics = metrics or MetricsTracker()
        self.channel_calc = LinearRegressionChannel()

        # Get windows from config
        self.windows = config.channel_windows
        self.min_cycles = config.min_cycles
        self.min_r_squared = config.min_r_squared

        logger.info(f"ChannelFeatureExtractor initialized: {len(self.windows)} windows, "
                   f"validity: cycles>={self.min_cycles}, r²>{self.min_r_squared}")

    def extract_for_timeframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        mode: str = 'batch'
    ) -> pd.DataFrame:
        """
        Extract channel features for a single timeframe.

        Args:
            df: OHLCV DataFrame (already resampled to target timeframe)
            symbol: Stock symbol ('tsla' or 'spy')
            timeframe: Timeframe name ('5min', '4h', 'daily', etc.)
            mode: 'batch' (full historical) or 'streaming' (latest only)

        Returns:
            DataFrame with channel features (same index as input)

        Raises:
            InsufficientDataError: Not enough data for channel calculation
            ChannelFeaturesError: Error during feature extraction
        """
        if len(df) < 20:
            raise InsufficientDataError(
                f"Need at least 20 bars for channel calculation, got {len(df)}"
            )

        with self.metrics.timer(f'channel_features_{timeframe}'):
            try:
                if mode == 'streaming':
                    # Extract only latest bar (for inference)
                    features = self._extract_streaming(df, symbol, timeframe)
                else:
                    # Extract all historical bars (for training)
                    features = self._extract_batch(df, symbol, timeframe)

                logger.info(f"Extracted {symbol}_{timeframe} channel features: "
                           f"{features.shape[1]} features × {len(features)} bars")

                return features

            except Exception as e:
                logger.error(f"Channel feature extraction failed for {symbol}_{timeframe}: {e}")
                raise ChannelFeaturesError(
                    f"Failed to extract channel features for {symbol}_{timeframe}"
                ) from e

    def _extract_batch(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Extract channel features for all bars (training mode).

        Uses rolling window approach - calculates channel at each bar
        using only data available up to that point (no lookahead bias).
        """
        n_bars = len(df)
        prefix = f'{symbol}_channel_{timeframe}'

        # Pre-allocate feature arrays
        feature_data = self._allocate_feature_arrays(n_bars, prefix)

        # Minimum lookback for channels
        min_lookback = min(max(self.windows), 50)

        # Process each bar sequentially
        for i in range(min_lookback, n_bars):
            # Data available up to this bar
            available_data = df.iloc[:i+1]
            current_price = df['close'].iloc[i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]

            # Extract features for all windows
            bar_features = self._extract_bar_features(
                available_data,
                current_price,
                current_high,
                current_low,
                symbol,
                timeframe
            )

            # Store features for this bar
            for feature_name, value in bar_features.items():
                col_name = f'{prefix}_{feature_name}'
                if col_name in feature_data:
                    feature_data[col_name][i] = value

        # Convert to DataFrame
        result = pd.DataFrame(feature_data, index=df.index)
        return result

    def _extract_streaming(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Extract channel features for latest bar only (inference mode).

        Fast path for live trading - only computes features for most recent bar.
        """
        n_bars = len(df)
        prefix = f'{symbol}_channel_{timeframe}'

        # Allocate for single bar
        feature_data = self._allocate_feature_arrays(1, prefix)

        # Extract features for latest bar
        current_price = df['close'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]

        bar_features = self._extract_bar_features(
            df,
            current_price,
            current_high,
            current_low,
            symbol,
            timeframe
        )

        # Store features
        for feature_name, value in bar_features.items():
            col_name = f'{prefix}_{feature_name}'
            if col_name in feature_data:
                feature_data[col_name][0] = value

        # Convert to DataFrame with single row
        result = pd.DataFrame(feature_data, index=[df.index[-1]])
        return result

    def _extract_bar_features(
        self,
        df: pd.DataFrame,
        current_price: float,
        current_high: float,
        current_low: float,
        symbol: str,
        timeframe: str
    ) -> Dict[str, float]:
        """
        Extract channel features for all windows at a single bar.

        Args:
            df: Historical data available up to current bar
            current_price: Close price at current bar
            current_high: High price at current bar
            current_low: Low price at current bar
            symbol: Stock symbol
            timeframe: Timeframe name

        Returns:
            Dictionary of feature_name -> value for all windows
        """
        features = {}

        # Process each window size
        for window in self.windows:
            window_prefix = f'w{window}'

            # Extract lookback window
            lookback_data = df.iloc[-min(window, len(df)):]

            if len(lookback_data) < 10:
                # Insufficient data for this window
                window_features = self._get_default_features(window_prefix)
                features.update(window_features)
                continue

            # Calculate channel for close, high, low
            close_channel = self._calculate_channel(lookback_data['close'].values)
            high_channel = self._calculate_channel(lookback_data['high'].values)
            low_channel = self._calculate_channel(lookback_data['low'].values)

            # Extract features for this window
            window_features = self._calculate_window_features(
                window_prefix,
                close_channel,
                high_channel,
                low_channel,
                lookback_data,
                current_price,
                current_high,
                current_low
            )

            features.update(window_features)

        return features

    def _calculate_channel(
        self,
        prices: np.ndarray
    ) -> Optional[Dict]:
        """
        Calculate linear regression channel for price series.

        Args:
            prices: Price array

        Returns:
            Dictionary with channel parameters or None if insufficient data
        """
        if len(prices) < 10:
            return None

        # Fit linear regression
        n = len(prices)
        x = np.arange(n)

        # Calculate coefficients using numpy
        x_mean = x.mean()
        y_mean = prices.mean()

        numerator = ((x - x_mean) * (prices - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()

        if denominator == 0:
            return None

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = ((prices - y_pred) ** 2).sum()
        ss_tot = ((prices - y_mean) ** 2).sum()

        if ss_tot == 0:
            r_squared = 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        # Calculate channel width (standard deviation of residuals)
        residuals = prices - y_pred
        std_dev = residuals.std()

        # Channel lines (±2 std dev)
        upper_line = y_pred + 2.0 * std_dev
        lower_line = y_pred - 2.0 * std_dev

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'upper_line': upper_line,
            'lower_line': lower_line,
            'center_line': y_pred,
            'width': std_dev,
            'duration': n
        }

    def _calculate_window_features(
        self,
        window_prefix: str,
        close_channel: Optional[Dict],
        high_channel: Optional[Dict],
        low_channel: Optional[Dict],
        lookback_data: pd.DataFrame,
        current_price: float,
        current_high: float,
        current_low: float
    ) -> Dict[str, float]:
        """
        Calculate all 31 features for a single window.

        Returns feature dictionary with window_prefix prepended to each feature name.
        """
        features = {}

        # Handle insufficient data case
        if close_channel is None or high_channel is None or low_channel is None:
            return self._get_default_features(window_prefix)

        # === Position Metrics (3) ===
        close_upper = close_channel['upper_line'][-1]
        close_lower = close_channel['lower_line'][-1]
        close_center = close_channel['center_line'][-1]

        channel_range = close_upper - close_lower
        if channel_range > 0:
            position = (current_price - close_lower) / channel_range
            upper_dist = (close_upper - current_price) / current_price * 100
            lower_dist = (current_price - close_lower) / current_price * 100
        else:
            position = 0.5
            upper_dist = 0.0
            lower_dist = 0.0

        features[f'{window_prefix}_position'] = np.clip(position, 0, 1)
        features[f'{window_prefix}_upper_dist'] = upper_dist
        features[f'{window_prefix}_lower_dist'] = lower_dist

        # === Raw Slopes (3) ===
        features[f'{window_prefix}_close_slope'] = close_channel['slope']
        features[f'{window_prefix}_high_slope'] = high_channel['slope']
        features[f'{window_prefix}_low_slope'] = low_channel['slope']

        # === Normalized Slopes (3) ===
        if current_price > 0:
            features[f'{window_prefix}_close_slope_pct'] = (close_channel['slope'] / current_price) * 100
            features[f'{window_prefix}_high_slope_pct'] = (high_channel['slope'] / current_price) * 100
            features[f'{window_prefix}_low_slope_pct'] = (low_channel['slope'] / current_price) * 100
        else:
            features[f'{window_prefix}_close_slope_pct'] = 0.0
            features[f'{window_prefix}_high_slope_pct'] = 0.0
            features[f'{window_prefix}_low_slope_pct'] = 0.0

        # === R-squared (4) ===
        features[f'{window_prefix}_close_r_squared'] = close_channel['r_squared']
        features[f'{window_prefix}_high_r_squared'] = high_channel['r_squared']
        features[f'{window_prefix}_low_r_squared'] = low_channel['r_squared']
        features[f'{window_prefix}_r_squared_avg'] = (
            close_channel['r_squared'] + high_channel['r_squared'] + low_channel['r_squared']
        ) / 3.0

        # === Channel Metrics (3) ===
        features[f'{window_prefix}_width_pct'] = (channel_range / current_price) * 100 if current_price > 0 else 0.0

        # Slope convergence (how aligned are close/high/low channels)
        slope_std = np.std([close_channel['slope'], high_channel['slope'], low_channel['slope']])
        features[f'{window_prefix}_slope_convergence'] = slope_std

        # Stability (inverse of width - narrow channels are more stable)
        features[f'{window_prefix}_stability'] = 1.0 / (1.0 + channel_range / current_price) if current_price > 0 else 0.0

        # === Ping-Pongs (4 thresholds) ===
        prices = lookback_data['close'].values
        upper_line = close_channel['upper_line']
        lower_line = close_channel['lower_line']

        features[f'{window_prefix}_ping_pongs_0_5pct'] = self._count_bounces(prices, upper_line, lower_line, 0.005)
        features[f'{window_prefix}_ping_pongs_1_0pct'] = self._count_bounces(prices, upper_line, lower_line, 0.01)
        features[f'{window_prefix}_ping_pongs_2_0pct'] = self._count_bounces(prices, upper_line, lower_line, 0.02)
        features[f'{window_prefix}_ping_pongs_3_0pct'] = self._count_bounces(prices, upper_line, lower_line, 0.03)

        # === Complete Cycles (4 thresholds) ===
        features[f'{window_prefix}_complete_cycles_0_5pct'] = self._count_cycles(prices, upper_line, lower_line, 0.005)
        features[f'{window_prefix}_complete_cycles_1_0pct'] = self._count_cycles(prices, upper_line, lower_line, 0.01)
        features[f'{window_prefix}_complete_cycles_2_0pct'] = self._count_cycles(prices, upper_line, lower_line, 0.02)
        features[f'{window_prefix}_complete_cycles_3_0pct'] = self._count_cycles(prices, upper_line, lower_line, 0.03)

        # === Direction Flags (3) ===
        slope_pct = features[f'{window_prefix}_close_slope_pct']
        features[f'{window_prefix}_is_bull'] = 1.0 if slope_pct > 0.1 else 0.0
        features[f'{window_prefix}_is_bear'] = 1.0 if slope_pct < -0.1 else 0.0
        features[f'{window_prefix}_is_sideways'] = 1.0 if abs(slope_pct) <= 0.1 else 0.0

        # === Quality (3) ===
        # Get max cycles across all thresholds
        max_cycles = max([
            features[f'{window_prefix}_complete_cycles_0_5pct'],
            features[f'{window_prefix}_complete_cycles_1_0pct'],
            features[f'{window_prefix}_complete_cycles_2_0pct'],
            features[f'{window_prefix}_complete_cycles_3_0pct']
        ])

        # v6.0 validity: cycles >= 1, r² > 0.1
        is_valid = (max_cycles >= self.min_cycles) and (close_channel['r_squared'] > self.min_r_squared)
        features[f'{window_prefix}_is_valid'] = 1.0 if is_valid else 0.0

        # Quality score (0-100)
        r_sq_score = min(close_channel['r_squared'] * 100, 100)
        cycle_score = min(max_cycles * 20, 100)
        features[f'{window_prefix}_quality_score'] = (r_sq_score + cycle_score) / 2.0

        # Insufficient data flag
        features[f'{window_prefix}_insufficient_data'] = 0.0

        # === Duration (1) ===
        features[f'{window_prefix}_duration'] = float(close_channel['duration'])

        return features

    def _count_bounces(
        self,
        prices: np.ndarray,
        upper_line: np.ndarray,
        lower_line: np.ndarray,
        threshold: float
    ) -> int:
        """
        Count ping-pongs (alternating touches of upper/lower bounds).

        Args:
            prices: Price array
            upper_line: Upper channel line
            lower_line: Lower channel line
            threshold: Distance threshold (e.g., 0.02 = 2%)

        Returns:
            Number of bounces detected
        """
        bounces = 0
        last_touch = 0  # 0=none, 1=upper, 2=lower

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper_line[i]
            lower_val = lower_line[i]

            # Check upper touch
            if upper_val > 0:
                upper_dist = abs(price - upper_val) / upper_val
                if upper_dist <= threshold:
                    if last_touch == 2:  # Was at lower
                        bounces += 1
                    last_touch = 1

            # Check lower touch
            if abs(lower_val) > 0:
                lower_dist = abs(price - lower_val) / abs(lower_val)
                if lower_dist <= threshold:
                    if last_touch == 1:  # Was at upper
                        bounces += 1
                    last_touch = 2

        return bounces

    def _count_cycles(
        self,
        prices: np.ndarray,
        upper_line: np.ndarray,
        lower_line: np.ndarray,
        threshold: float
    ) -> int:
        """
        Count complete cycles (full round-trips).

        Lower → Upper → Lower = 1 cycle
        Upper → Lower → Upper = 1 cycle

        Args:
            prices: Price array
            upper_line: Upper channel line
            lower_line: Lower channel line
            threshold: Distance threshold (e.g., 0.02 = 2%)

        Returns:
            Number of complete cycles
        """
        touches = []
        last_touch = 0

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper_line[i]
            lower_val = lower_line[i]

            # Check upper touch
            if upper_val > 0:
                upper_dist = abs(price - upper_val) / upper_val
                if upper_dist <= threshold and last_touch != 1:
                    touches.append(1)
                    last_touch = 1

            # Check lower touch
            if abs(lower_val) > 0:
                lower_dist = abs(price - lower_val) / abs(lower_val)
                if lower_dist <= threshold and last_touch != 2:
                    touches.append(2)
                    last_touch = 2

        # Count complete cycles
        cycles = 0
        i = 0
        while i < len(touches) - 2:
            # Look for pattern: A -> B -> A
            if touches[i] == touches[i+2] and touches[i] != touches[i+1]:
                cycles += 1
                i += 2  # Skip ahead
            else:
                i += 1

        return cycles

    def _allocate_feature_arrays(
        self,
        n_bars: int,
        prefix: str
    ) -> Dict[str, np.ndarray]:
        """
        Pre-allocate feature arrays for all windows.

        Returns dictionary of feature_name -> np.ndarray
        """
        feature_names = []

        for window in self.windows:
            wp = f'w{window}'

            # Position metrics (3)
            feature_names.extend([
                f'{prefix}_{wp}_position',
                f'{prefix}_{wp}_upper_dist',
                f'{prefix}_{wp}_lower_dist',
            ])

            # Raw slopes (3)
            feature_names.extend([
                f'{prefix}_{wp}_close_slope',
                f'{prefix}_{wp}_high_slope',
                f'{prefix}_{wp}_low_slope',
            ])

            # Normalized slopes (3)
            feature_names.extend([
                f'{prefix}_{wp}_close_slope_pct',
                f'{prefix}_{wp}_high_slope_pct',
                f'{prefix}_{wp}_low_slope_pct',
            ])

            # R-squared (4)
            feature_names.extend([
                f'{prefix}_{wp}_close_r_squared',
                f'{prefix}_{wp}_high_r_squared',
                f'{prefix}_{wp}_low_r_squared',
                f'{prefix}_{wp}_r_squared_avg',
            ])

            # Channel metrics (3)
            feature_names.extend([
                f'{prefix}_{wp}_width_pct',
                f'{prefix}_{wp}_slope_convergence',
                f'{prefix}_{wp}_stability',
            ])

            # Ping-pongs (4)
            feature_names.extend([
                f'{prefix}_{wp}_ping_pongs_0_5pct',
                f'{prefix}_{wp}_ping_pongs_1_0pct',
                f'{prefix}_{wp}_ping_pongs_2_0pct',
                f'{prefix}_{wp}_ping_pongs_3_0pct',
            ])

            # Complete cycles (4)
            feature_names.extend([
                f'{prefix}_{wp}_complete_cycles_0_5pct',
                f'{prefix}_{wp}_complete_cycles_1_0pct',
                f'{prefix}_{wp}_complete_cycles_2_0pct',
                f'{prefix}_{wp}_complete_cycles_3_0pct',
            ])

            # Direction flags (3)
            feature_names.extend([
                f'{prefix}_{wp}_is_bull',
                f'{prefix}_{wp}_is_bear',
                f'{prefix}_{wp}_is_sideways',
            ])

            # Quality (3)
            feature_names.extend([
                f'{prefix}_{wp}_quality_score',
                f'{prefix}_{wp}_is_valid',
                f'{prefix}_{wp}_insufficient_data',
            ])

            # Duration (1)
            feature_names.append(f'{prefix}_{wp}_duration')

        # Allocate arrays with default values
        arrays = {}
        for name in feature_names:
            arrays[name] = np.full(n_bars, 0.5 if 'position' in name else 0.0, dtype=np.float32)

        return arrays

    def _get_default_features(self, window_prefix: str) -> Dict[str, float]:
        """
        Get default feature values when data is insufficient.

        Returns dictionary with all 31 features set to defaults.
        """
        features = {
            # Position (3)
            f'{window_prefix}_position': 0.5,
            f'{window_prefix}_upper_dist': 0.0,
            f'{window_prefix}_lower_dist': 0.0,

            # Raw slopes (3)
            f'{window_prefix}_close_slope': 0.0,
            f'{window_prefix}_high_slope': 0.0,
            f'{window_prefix}_low_slope': 0.0,

            # Normalized slopes (3)
            f'{window_prefix}_close_slope_pct': 0.0,
            f'{window_prefix}_high_slope_pct': 0.0,
            f'{window_prefix}_low_slope_pct': 0.0,

            # R-squared (4)
            f'{window_prefix}_close_r_squared': 0.0,
            f'{window_prefix}_high_r_squared': 0.0,
            f'{window_prefix}_low_r_squared': 0.0,
            f'{window_prefix}_r_squared_avg': 0.0,

            # Channel metrics (3)
            f'{window_prefix}_width_pct': 0.0,
            f'{window_prefix}_slope_convergence': 0.0,
            f'{window_prefix}_stability': 0.0,

            # Ping-pongs (4)
            f'{window_prefix}_ping_pongs_0_5pct': 0,
            f'{window_prefix}_ping_pongs_1_0pct': 0,
            f'{window_prefix}_ping_pongs_2_0pct': 0,
            f'{window_prefix}_ping_pongs_3_0pct': 0,

            # Complete cycles (4)
            f'{window_prefix}_complete_cycles_0_5pct': 0,
            f'{window_prefix}_complete_cycles_1_0pct': 0,
            f'{window_prefix}_complete_cycles_2_0pct': 0,
            f'{window_prefix}_complete_cycles_3_0pct': 0,

            # Direction (3)
            f'{window_prefix}_is_bull': 0.0,
            f'{window_prefix}_is_bear': 0.0,
            f'{window_prefix}_is_sideways': 1.0,  # Default to sideways

            # Quality (3)
            f'{window_prefix}_quality_score': 0.0,
            f'{window_prefix}_is_valid': 0.0,
            f'{window_prefix}_insufficient_data': 1.0,  # Mark as insufficient

            # Duration (1)
            f'{window_prefix}_duration': 0.0,
        }

        return features


def extract_channel_features_multi_symbol(
    df: pd.DataFrame,
    config: FeatureConfig,
    symbols: List[str] = ['tsla', 'spy'],
    timeframes: Optional[List[str]] = None,
    mode: str = 'batch',
    metrics: Optional[MetricsTracker] = None
) -> pd.DataFrame:
    """
    Extract channel features for multiple symbols and timeframes.

    High-level convenience function for batch extraction.

    Args:
        df: DataFrame with columns like 'tsla_close', 'spy_high', etc.
        config: Feature configuration
        symbols: List of symbols to process
        timeframes: List of timeframes (if None, process all 11)
        mode: 'batch' or 'streaming'
        metrics: Optional metrics tracker

    Returns:
        DataFrame with all channel features concatenated

    Example:
        >>> config = get_feature_config()
        >>> df = load_5min_data()  # Has tsla_close, spy_close, etc.
        >>> features = extract_channel_features_multi_symbol(
        ...     df, config, symbols=['tsla', 'spy'], timeframes=['5min', '4h']
        ... )
    """
    if timeframes is None:
        timeframes = config.timeframes

    extractor = ChannelFeatureExtractor(config, metrics)
    all_features = []

    for symbol in symbols:
        for timeframe in timeframes:
            # Resample to target timeframe
            # (In real implementation, would need proper resampling logic)
            # For now, assume df is already at correct resolution

            # Extract symbol-specific columns
            symbol_cols = [c for c in df.columns if c.startswith(f'{symbol}_')]
            if not symbol_cols:
                logger.warning(f"No columns found for {symbol}, skipping")
                continue

            symbol_df = df[symbol_cols].rename(columns=lambda c: c.replace(f'{symbol}_', ''))

            # Extract features
            try:
                features = extractor.extract_for_timeframe(
                    symbol_df, symbol, timeframe, mode
                )
                all_features.append(features)

            except InsufficientDataError as e:
                logger.warning(f"Insufficient data for {symbol}_{timeframe}: {e}")
                continue
            except ChannelFeaturesError as e:
                logger.error(f"Failed to extract {symbol}_{timeframe}: {e}")
                continue

    if not all_features:
        raise ChannelFeaturesError("No channel features extracted for any symbol/timeframe")

    # Concatenate all features
    result = pd.concat(all_features, axis=1)

    logger.info(f"Channel feature extraction complete: {result.shape[1]} features")
    return result
