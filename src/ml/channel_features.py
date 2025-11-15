"""
Channel Feature Extraction for Hierarchical LNN

Calculates linear regression channels, ping-pongs, and breakdown indicators
across multiple timeframes for TSLA and SPY.

Features per stock per timeframe:
- Linear regression slope
- Linear regression intercept
- Channel width (std of residuals)
- Ping-pong count (touches of bounds)
- Time in channel (bars since break)

Timeframes: 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3month
Total: 11 timeframes × 5 features × 2 stocks = 110 features
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LinearRegression


class ChannelFeatureExtractor:
    """
    Extracts linear regression channel features across multiple timeframes.

    Key Concepts:
    - Channel: Linear regression line ± N std deviations
    - Ping-pong: Price touching upper/lower channel bounds
    - Break: Price moving beyond channel bounds significantly
    - Position: Where price sits in channel (-1=bottom, 0=middle, +1=top)
    """

    def __init__(self):
        """Initialize feature extractor."""
        # Timeframes to calculate channels for (in bars)
        self.timeframes = {
            '5min': 5,
            '15min': 15,
            '30min': 30,
            '1hour': 60,
            '2hour': 120,
            '3hour': 180,
            '4hour': 240,
            'daily': 390,  # 6.5 hours = 390 minutes
            'weekly': 1950,  # 5 trading days
            'monthly': 8580,  # ~22 trading days
            '3month': 25740  # ~66 trading days
        }

        # Channel width multiplier (std deviations)
        self.channel_std = 2.0

        # Breakout threshold (beyond channel bounds)
        self.breakout_threshold = 1.5  # 1.5x channel width

    def calculate_linear_regression(
        self,
        prices: np.ndarray,
        window: int
    ) -> Tuple[float, float, float, np.ndarray]:
        """
        Calculate linear regression for a price window.

        Args:
            prices: Array of prices
            window: Number of bars to look back

        Returns:
            slope: Regression slope
            intercept: Regression intercept
            channel_width: Std deviation of residuals
            residuals: Array of residuals (price - fitted)
        """
        if len(prices) < window or window < 2:
            return 0.0, 0.0, 0.0, np.array([0.0])

        # Take last 'window' prices
        y = prices[-window:]
        X = np.arange(window).reshape(-1, 1)

        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Calculate residuals
        fitted = model.predict(X)
        residuals = y - fitted

        # Channel width = std of residuals
        channel_width = np.std(residuals)

        return float(model.coef_[0]), float(model.intercept_), channel_width, residuals

    def count_ping_pongs(
        self,
        prices: np.ndarray,
        fitted_line: np.ndarray,
        channel_width: float,
        window: int
    ) -> int:
        """
        Count how many times price touched upper or lower channel bounds.

        A "touch" is when price comes within 0.2 * channel_width of bounds.

        Args:
            prices: Price array
            fitted_line: Linear regression fitted values
            channel_width: Channel width (std)
            window: Lookback window

        Returns:
            ping_pong_count: Number of touches
        """
        if len(prices) < window or channel_width == 0:
            return 0

        y = prices[-window:]
        residuals = y - fitted_line

        # Upper and lower bounds
        upper_bound = self.channel_std * channel_width
        lower_bound = -self.channel_std * channel_width

        # Touch threshold (20% of channel width from bounds)
        touch_threshold = 0.2 * channel_width

        # Count touches
        upper_touches = np.sum(residuals >= (upper_bound - touch_threshold))
        lower_touches = np.sum(residuals <= (lower_bound + touch_threshold))

        return int(upper_touches + lower_touches)

    def detect_channel_break(
        self,
        current_price: float,
        fitted_value: float,
        channel_width: float
    ) -> bool:
        """
        Detect if current price has broken out of channel.

        Args:
            current_price: Most recent price
            fitted_value: Expected price from regression
            channel_width: Channel width (std)

        Returns:
            True if broken, False otherwise
        """
        if channel_width == 0:
            return False

        residual = abs(current_price - fitted_value)
        breakout_distance = self.breakout_threshold * self.channel_std * channel_width

        return residual > breakout_distance

    def calculate_time_in_channel(
        self,
        prices: np.ndarray,
        window: int
    ) -> int:
        """
        Calculate how many bars price has stayed within channel.

        Counts backward from current bar until a break is found.

        Args:
            prices: Price array
            window: Lookback window

        Returns:
            bars_in_channel: Number of bars since last break
        """
        if len(prices) < window:
            return 0

        # Calculate channel for full window
        slope, intercept, channel_width, _ = self.calculate_linear_regression(
            prices, window
        )

        if channel_width == 0:
            return window

        # Check each bar backward
        for i in range(window - 1, -1, -1):
            idx = len(prices) - window + i
            current_price = prices[idx]

            # Fitted value at this point
            fitted_value = slope * i + intercept

            # Check if break
            if self.detect_channel_break(current_price, fitted_value, channel_width):
                # Found break - return bars since then
                return window - i - 1

        # No break found - entire window is in channel
        return window

    def get_channel_position(
        self,
        current_price: float,
        fitted_value: float,
        channel_width: float
    ) -> float:
        """
        Calculate where price sits in channel.

        Args:
            current_price: Most recent price
            fitted_value: Expected price from regression
            channel_width: Channel width (std)

        Returns:
            position: -1 (bottom) to +1 (top), 0 = middle
        """
        if channel_width == 0:
            return 0.0

        residual = current_price - fitted_value
        max_deviation = self.channel_std * channel_width

        # Normalize to [-1, 1]
        position = residual / max_deviation

        # Clamp to [-1, 1]
        return float(np.clip(position, -1.0, 1.0))

    def extract_channel_features_single_stock(
        self,
        prices: np.ndarray,
        prefix: str = 'tsla'
    ) -> Dict[str, float]:
        """
        Extract all channel features for a single stock.

        Args:
            prices: Array of close prices
            prefix: Feature name prefix ('tsla' or 'spy')

        Returns:
            features: Dictionary of features
        """
        features = {}

        for tf_name, window_bars in self.timeframes.items():
            # Calculate linear regression
            slope, intercept, channel_width, residuals = \
                self.calculate_linear_regression(prices, window_bars)

            # Fitted line for full window
            if len(prices) >= window_bars and window_bars >= 2:
                X = np.arange(window_bars).reshape(-1, 1)
                fitted_line = slope * X.flatten() + intercept

                # Ping-pong count
                ping_pongs = self.count_ping_pongs(
                    prices, fitted_line, channel_width, window_bars
                )

                # Time in channel
                time_in_channel = self.calculate_time_in_channel(prices, window_bars)

                # Current channel position
                current_price = prices[-1]
                current_fitted = slope * (window_bars - 1) + intercept
                channel_position = self.get_channel_position(
                    current_price, current_fitted, channel_width
                )
            else:
                ping_pongs = 0
                time_in_channel = 0
                channel_position = 0.0

            # Store features
            features[f'{prefix}_lr_slope_{tf_name}'] = slope
            features[f'{prefix}_lr_intercept_{tf_name}'] = intercept
            features[f'{prefix}_channel_width_{tf_name}'] = channel_width
            features[f'{prefix}_ping_pongs_{tf_name}'] = float(ping_pongs)
            features[f'{prefix}_time_in_channel_{tf_name}'] = float(time_in_channel)
            features[f'{prefix}_channel_position_{tf_name}'] = channel_position

        return features

    def extract_breakdown_indicators(
        self,
        tsla_prices: np.ndarray,
        spy_prices: np.ndarray,
        tsla_volume: np.ndarray,
        tsla_rsi: Dict[str, float],
        spy_rsi: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Extract channel breakdown prediction indicators.

        Args:
            tsla_prices: TSLA close prices
            spy_prices: SPY close prices
            tsla_volume: TSLA volume
            tsla_rsi: TSLA RSI values by timeframe
            spy_rsi: SPY RSI values by timeframe

        Returns:
            breakdown_features: Dictionary of features
        """
        features = {}

        # Volume surge detection (last 10 bars vs previous 50)
        if len(tsla_volume) >= 60:
            recent_vol = np.mean(tsla_volume[-10:])
            historical_vol = np.mean(tsla_volume[-60:-10])

            if historical_vol > 0:
                volume_surge = (recent_vol - historical_vol) / historical_vol
            else:
                volume_surge = 0.0

            features['tsla_volume_surge'] = volume_surge
        else:
            features['tsla_volume_surge'] = 0.0

        # RSI divergence (RSI overbought/oversold at channel extremes)
        # Check if RSI and channel position diverge
        for tf_name in ['15min', '1hour', '4hour', 'daily']:
            tsla_rsi_val = tsla_rsi.get(tf_name, 50.0)

            # Calculate channel position for this timeframe
            window = self.timeframes[tf_name]
            if len(tsla_prices) >= window:
                slope, intercept, channel_width, _ = \
                    self.calculate_linear_regression(tsla_prices, window)

                current_price = tsla_prices[-1]
                current_fitted = slope * (window - 1) + intercept
                position = self.get_channel_position(
                    current_price, current_fitted, channel_width
                )

                # Divergence: High RSI but low position = potential reversal down
                # Low RSI but high position = potential reversal up
                divergence = tsla_rsi_val / 100.0 - position  # [-2, 2] range

                features[f'tsla_rsi_divergence_{tf_name}'] = divergence
            else:
                features[f'tsla_rsi_divergence_{tf_name}'] = 0.0

        # Channel duration vs historical average
        # For key timeframes, see if current channel is unusually long
        for tf_name in ['1hour', '4hour', 'daily']:
            window = self.timeframes[tf_name]
            if len(tsla_prices) >= window * 3:
                # Current time in channel
                current_time = self.calculate_time_in_channel(tsla_prices, window)

                # Historical average (split into 3 segments)
                segment_size = window
                times = []
                for i in range(3):
                    start_idx = len(tsla_prices) - (i + 1) * segment_size
                    end_idx = len(tsla_prices) - i * segment_size
                    if start_idx >= 0:
                        segment_prices = tsla_prices[start_idx:end_idx]
                        if len(segment_prices) >= window:
                            time_val = self.calculate_time_in_channel(
                                segment_prices, min(window, len(segment_prices))
                            )
                            times.append(time_val)

                if times:
                    avg_time = np.mean(times)
                    if avg_time > 0:
                        duration_ratio = current_time / avg_time
                    else:
                        duration_ratio = 1.0
                else:
                    duration_ratio = 1.0

                features[f'tsla_channel_duration_ratio_{tf_name}'] = duration_ratio
            else:
                features[f'tsla_channel_duration_ratio_{tf_name}'] = 1.0

        # SPY-TSLA channel alignment (both at top/bottom = higher break probability)
        for tf_name in ['1hour', '4hour']:
            window = self.timeframes[tf_name]

            if len(tsla_prices) >= window and len(spy_prices) >= window:
                # TSLA position
                slope_t, intercept_t, width_t, _ = \
                    self.calculate_linear_regression(tsla_prices, window)
                pos_t = self.get_channel_position(
                    tsla_prices[-1],
                    slope_t * (window - 1) + intercept_t,
                    width_t
                )

                # SPY position
                slope_s, intercept_s, width_s, _ = \
                    self.calculate_linear_regression(spy_prices, window)
                pos_s = self.get_channel_position(
                    spy_prices[-1],
                    slope_s * (window - 1) + intercept_s,
                    width_s
                )

                # Alignment score: 1 if both same direction, -1 if opposite
                alignment = pos_t * pos_s  # [-1, 1]

                features[f'channel_alignment_spy_tsla_{tf_name}'] = alignment
            else:
                features[f'channel_alignment_spy_tsla_{tf_name}'] = 0.0

        return features

    def extract_all_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract all channel features from aligned TSLA-SPY dataframe.

        Args:
            df: DataFrame with columns: tsla_close, spy_close, tsla_volume, etc.

        Returns:
            features_df: Original df with new channel feature columns added
        """
        # Extract TSLA channel features
        tsla_prices = df['tsla_close'].values
        tsla_features = self.extract_channel_features_single_stock(
            tsla_prices, prefix='tsla'
        )

        # Extract SPY channel features
        spy_prices = df['spy_close'].values
        spy_features = self.extract_channel_features_single_stock(
            spy_prices, prefix='spy'
        )

        # Note: RSI features should already exist in df from features.py
        # Extract them for breakdown indicators
        tsla_rsi = {}
        spy_rsi = {}
        for tf in ['15min', '1hour', '4hour', 'daily']:
            tsla_rsi[tf] = df[f'tsla_rsi_{tf}'].iloc[-1] if f'tsla_rsi_{tf}' in df.columns else 50.0
            spy_rsi[tf] = df[f'spy_rsi_{tf}'].iloc[-1] if f'spy_rsi_{tf}' in df.columns else 50.0

        # Extract breakdown indicators
        tsla_volume = df['tsla_volume'].values
        breakdown_features = self.extract_breakdown_indicators(
            tsla_prices,
            spy_prices,
            tsla_volume,
            tsla_rsi,
            spy_rsi
        )

        # Combine all features
        all_features = {**tsla_features, **spy_features, **breakdown_features}

        # Add as new columns (broadcast to all rows - these are rolling features)
        # For simplicity, we'll calculate these on a rolling basis
        # In practice, this should be called row-by-row in features.py

        return all_features


def add_channel_features_to_dataframe(
    df: pd.DataFrame,
    extractor: Optional[ChannelFeatureExtractor] = None
) -> pd.DataFrame:
    """
    Add channel features to an existing features dataframe.

    This function processes the dataframe row-by-row to calculate rolling
    channel features, similar to how RSI and other indicators are calculated.

    Args:
        df: Features dataframe with tsla_close, spy_close, etc.
        extractor: ChannelFeatureExtractor instance (creates new if None)

    Returns:
        df_with_channels: DataFrame with channel feature columns added
    """
    if extractor is None:
        extractor = ChannelFeatureExtractor()

    # Initialize feature columns with zeros
    feature_names = []

    # TSLA channel features
    for tf_name in extractor.timeframes.keys():
        for suffix in ['lr_slope', 'lr_intercept', 'channel_width',
                       'ping_pongs', 'time_in_channel', 'channel_position']:
            feature_names.append(f'tsla_{suffix}_{tf_name}')

    # SPY channel features
    for tf_name in extractor.timeframes.keys():
        for suffix in ['lr_slope', 'lr_intercept', 'channel_width',
                       'ping_pongs', 'time_in_channel', 'channel_position']:
            feature_names.append(f'spy_{suffix}_{tf_name}')

    # Breakdown features
    feature_names.append('tsla_volume_surge')
    for tf in ['15min', '1hour', '4hour', 'daily']:
        feature_names.append(f'tsla_rsi_divergence_{tf}')
    for tf in ['1hour', '4hour', 'daily']:
        feature_names.append(f'tsla_channel_duration_ratio_{tf}')
    for tf in ['1hour', '4hour']:
        feature_names.append(f'channel_alignment_spy_tsla_{tf}')

    # Initialize columns
    for fname in feature_names:
        df[fname] = 0.0

    # Calculate features row-by-row (vectorized where possible)
    # For efficiency, we'll calculate on the full array and assign
    tsla_prices = df['tsla_close'].values
    spy_prices = df['spy_close'].values
    tsla_volume = df['tsla_volume'].values if 'tsla_volume' in df.columns else np.zeros(len(df))

    # Process each row
    for i in range(len(df)):
        # Need enough history for largest window
        max_window = max(extractor.timeframes.values())
        if i < max_window:
            continue  # Skip early rows without enough history

        # Get price history up to current row
        tsla_hist = tsla_prices[:i+1]
        spy_hist = spy_prices[:i+1]
        vol_hist = tsla_volume[:i+1]

        # Extract TSLA features
        tsla_feats = extractor.extract_channel_features_single_stock(tsla_hist, 'tsla')
        for key, val in tsla_feats.items():
            df.loc[i, key] = val

        # Extract SPY features
        spy_feats = extractor.extract_channel_features_single_stock(spy_hist, 'spy')
        for key, val in spy_feats.items():
            df.loc[i, key] = val

        # Extract RSI for breakdown indicators
        tsla_rsi = {}
        spy_rsi = {}
        for tf in ['15min', '1hour', '4hour', 'daily']:
            tsla_rsi[tf] = df.loc[i, f'tsla_rsi_{tf}'] if f'tsla_rsi_{tf}' in df.columns else 50.0
            spy_rsi[tf] = df.loc[i, f'spy_rsi_{tf}'] if f'spy_rsi_{tf}' in df.columns else 50.0

        # Extract breakdown features
        breakdown_feats = extractor.extract_breakdown_indicators(
            tsla_hist,
            spy_hist,
            vol_hist,
            tsla_rsi,
            spy_rsi
        )
        for key, val in breakdown_feats.items():
            df.loc[i, key] = val

    return df
