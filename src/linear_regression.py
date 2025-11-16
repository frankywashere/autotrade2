"""Linear regression channel calculator with ping-pong detection."""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import config


@dataclass
class ChannelData:
    """Data class for linear regression channel."""
    slope: float
    intercept: float
    upper_line: np.ndarray
    lower_line: np.ndarray
    center_line: np.ndarray
    std_dev: float
    r_squared: float
    ping_pongs: int
    stability_score: float
    predicted_high: float
    predicted_low: float
    predicted_center: float
    actual_duration: int = 0  # v3.11: Actual bars where channel holds (dynamic)


class LinearRegressionChannel:
    """Calculate and analyze linear regression channels."""

    def __init__(self, std_dev: float = config.CHANNEL_STD_DEV):
        """
        Initialize channel calculator.

        Args:
            std_dev: Number of standard deviations for channel width
        """
        self.std_dev = std_dev

    def calculate_channel(self, df: pd.DataFrame, lookback_bars: Optional[int] = None,
                         timeframe: str = "4hour") -> ChannelData:
        """
        Calculate linear regression channel for given data.
        Predicts high/low for next 24 hours.

        Args:
            df: DataFrame with OHLC data
            lookback_bars: Number of bars to look back (None = all data)
            timeframe: Timeframe for calculating 24-hour prediction

        Returns:
            ChannelData object with channel information
        """
        # Use lookback period if specified
        if lookback_bars and lookback_bars < len(df):
            df = df.iloc[-lookback_bars:]

        # Get close prices
        prices = df['close'].values
        n = len(prices)

        # Create x values (bar indices)
        x = np.arange(n)

        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        r_squared = r_value ** 2

        # Center line (regression line)
        center_line = slope * x + intercept

        # Calculate residuals (distances from regression line)
        residuals = prices - center_line

        # Standard deviation of residuals
        residual_std = np.std(residuals)

        # Upper and lower channel lines
        upper_line = center_line + (self.std_dev * residual_std)
        lower_line = center_line - (self.std_dev * residual_std)

        # Detect ping-pongs (bounces between upper and lower lines)
        ping_pongs = self._detect_ping_pongs(prices, upper_line, lower_line)

        # Calculate stability score (0-100)
        stability_score = self._calculate_stability(r_squared, ping_pongs, n)

        # Predict 24-hour range (not just next bar)
        # Calculate how many bars = 24 hours for this timeframe
        bars_per_24h = {
            '1hour': 24,
            '2hour': 12,
            '3hour': 8,
            '4hour': 6,
            'daily': 1,
            'weekly': 1  # Weekly can't predict 24h, use 1 bar
        }
        forecast_bars = bars_per_24h.get(timeframe, 1)

        # Project channel forward for next 24 hours
        future_x = np.arange(n, n + forecast_bars)
        future_center = slope * future_x + intercept
        future_upper = future_center + (self.std_dev * residual_std)
        future_lower = future_center - (self.std_dev * residual_std)

        # 24-hour predicted high/low is the range over this period
        predicted_high = np.max(future_upper)
        predicted_low = np.min(future_lower)
        predicted_center = future_center[-1]  # End of 24h period

        return ChannelData(
            slope=slope,
            intercept=intercept,
            upper_line=upper_line,
            lower_line=lower_line,
            center_line=center_line,
            std_dev=residual_std,
            r_squared=r_squared,
            ping_pongs=ping_pongs,
            stability_score=stability_score,
            predicted_high=predicted_high,
            predicted_low=predicted_low,
            predicted_center=predicted_center,
            actual_duration=lookback_bars if lookback_bars else n  # Record actual window used
        )

    def find_optimal_channel_window(
        self,
        df: pd.DataFrame,
        timeframe: str = "1h",
        max_lookback: int = 168,
        min_ping_pongs: int = 3
    ) -> Optional[ChannelData]:
        """
        Find the optimal lookback window that produces a valid channel.

        Tests multiple windows from longest to shortest, returns the best channel
        with at least min_ping_pongs bounces.

        Args:
            df: DataFrame with OHLC data
            timeframe: Timeframe name (for prediction)
            max_lookback: Maximum bars to consider
            min_ping_pongs: Minimum ping-pongs required (default: 3)

        Returns:
            ChannelData for best window, or None if no valid channel found
        """
        # Candidate windows to test (from longest to shortest for stability)
        candidates = [168, 120, 90, 60, 45, 30]
        candidates = [c for c in candidates if c <= max_lookback and c <= len(df)]

        if not candidates:
            return None

        best_channel = None
        best_score = 0

        for lookback in candidates:
            # Calculate channel for this window
            window = df.tail(lookback)
            channel = self.calculate_channel(window, lookback, timeframe)

            # Require minimum ping-pongs (real channels bounce!)
            if channel.ping_pongs < min_ping_pongs:
                continue

            # Score = R² (70%) + normalized ping-pongs (30%)
            # Higher R² = better fit, more ping-pongs = more confirmations
            ping_pong_score = min(channel.ping_pongs / 10.0, 1.0)  # Normalize to 0-1
            composite_score = (channel.r_squared * 0.7) + (ping_pong_score * 0.3)

            # Track best
            if composite_score > best_score:
                best_channel = channel
                best_score = composite_score

        return best_channel  # None if no valid channel found

    def _detect_ping_pongs(self, prices: np.ndarray, upper: np.ndarray, lower: np.ndarray,
                          threshold: float = 0.02) -> int:
        """
        Detect bounces between upper and lower channel lines.

        Args:
            prices: Price array
            upper: Upper channel line
            lower: Lower channel line
            threshold: Percentage threshold for detecting touch (2% default)

        Returns:
            Number of ping-pongs detected
        """
        bounces = 0
        last_touch = None  # 'upper' or 'lower'

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper[i]
            lower_val = lower[i]

            # Calculate distances as percentage
            upper_dist = abs(price - upper_val) / upper_val
            lower_dist = abs(price - lower_val) / lower_val

            # Check if price touches upper line
            if upper_dist <= threshold:
                if last_touch == 'lower':
                    bounces += 1
                last_touch = 'upper'

            # Check if price touches lower line
            elif lower_dist <= threshold:
                if last_touch == 'upper':
                    bounces += 1
                last_touch = 'lower'

        return bounces

    def _detect_ping_pongs_multi_threshold(
        self,
        prices: np.ndarray,
        upper: np.ndarray,
        lower: np.ndarray,
        thresholds: list = [0.005, 0.01, 0.02, 0.03]
    ) -> dict:
        """
        Detect bounces at multiple thresholds simultaneously (efficient single-pass).

        Args:
            prices: Price array
            upper: Upper channel line
            lower: Lower channel line
            thresholds: List of percentage thresholds to test

        Returns:
            Dict mapping threshold to ping-pong count
            Example: {0.005: 4, 0.01: 6, 0.02: 8, 0.03: 10}
        """
        results = {threshold: 0 for threshold in thresholds}
        last_touch = {threshold: None for threshold in thresholds}

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper[i]
            lower_val = lower[i]

            # Calculate distances as percentage
            upper_dist = abs(price - upper_val) / upper_val
            lower_dist = abs(price - lower_val) / lower_val

            # Check each threshold
            for threshold in thresholds:
                # Check if price touches upper line at this threshold
                if upper_dist <= threshold:
                    if last_touch[threshold] == 'lower':
                        results[threshold] += 1
                    last_touch[threshold] = 'upper'

                # Check if price touches lower line at this threshold
                elif lower_dist <= threshold:
                    if last_touch[threshold] == 'upper':
                        results[threshold] += 1
                    last_touch[threshold] = 'lower'

        return results

    def _calculate_stability(self, r_squared: float, ping_pongs: int, n_bars: int) -> float:
        """
        Calculate channel stability score (0-100).

        Args:
            r_squared: R-squared value of regression
            ping_pongs: Number of bounces
            n_bars: Number of bars in channel

        Returns:
            Stability score (0-100)
        """
        # R-squared component (0-40 points)
        r2_score = r_squared * 40

        # Ping-pong component (0-40 points)
        # More bounces = more stable, max at 5 bounces
        pp_score = min(ping_pongs / 5.0, 1.0) * 40

        # Length component (0-20 points)
        # Longer channels = more stable, max at 100 bars
        length_score = min(n_bars / 100.0, 1.0) * 20

        return r2_score + pp_score + length_score

    def get_channel_position(self, price: float, channel: ChannelData, bar_index: int = -1) -> Dict:
        """
        Get current price position within channel.

        Args:
            price: Current price
            channel: ChannelData object
            bar_index: Index of bar to check (-1 for latest)

        Returns:
            Dictionary with position info
        """
        upper = channel.upper_line[bar_index]
        lower = channel.lower_line[bar_index]
        center = channel.center_line[bar_index]

        # Calculate position (0 = lower line, 0.5 = center, 1 = upper line)
        channel_height = upper - lower
        position = (price - lower) / channel_height if channel_height > 0 else 0.5

        # Determine zone
        if position >= 0.9:
            zone = "upper_extreme"
        elif position >= 0.7:
            zone = "upper"
        elif position >= 0.3:
            zone = "middle"
        elif position >= 0.1:
            zone = "lower"
        else:
            zone = "lower_extreme"

        # Calculate distances
        dist_to_upper = ((upper - price) / price) * 100
        dist_to_lower = ((price - lower) / price) * 100

        return {
            "position": position,
            "zone": zone,
            "distance_to_upper_pct": dist_to_upper,
            "distance_to_lower_pct": dist_to_lower,
            "upper_value": upper,
            "lower_value": lower,
            "center_value": center
        }

    def analyze_multiple_timeframes(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, ChannelData]:
        """
        Analyze channels across multiple timeframes.

        Args:
            data_dict: Dictionary of timeframe -> DataFrame

        Returns:
            Dictionary of timeframe -> ChannelData
        """
        results = {}

        for timeframe, df in data_dict.items():
            if timeframe == "1min":
                continue  # Skip 1-minute for channel analysis

            # Determine lookback based on timeframe
            if timeframe in ["1hour", "2hour", "3hour"]:
                lookback = min(config.CHANNEL_LOOKBACK_HOURS, len(df))
            elif timeframe == "4hour":
                lookback = min(config.CHANNEL_LOOKBACK_HOURS // 4, len(df))
            else:
                lookback = None  # Use all data for daily/weekly

            try:
                channel = self.calculate_channel(df, lookback, timeframe)
                results[timeframe] = channel
            except Exception as e:
                print(f"Error calculating channel for {timeframe}: {e}")

        return results


if __name__ == "__main__":
    # Test the linear regression channel
    from data_handler import DataHandler

    handler = DataHandler("TSLA")
    handler.load_1min_data()

    calc = LinearRegressionChannel()

    # Test on 4-hour data
    df_4h = handler.get_data("4hour")
    channel = calc.calculate_channel(df_4h, lookback_bars=42)  # ~1 week

    print(f"\n4-Hour Channel Analysis:")
    print(f"Slope: {channel.slope:.4f}")
    print(f"R-squared: {channel.r_squared:.4f}")
    print(f"Ping-pongs: {channel.ping_pongs}")
    print(f"Stability Score: {channel.stability_score:.2f}/100")
    print(f"\nPredicted for next 4-hour bar:")
    print(f"  High: ${channel.predicted_high:.2f}")
    print(f"  Center: ${channel.predicted_center:.2f}")
    print(f"  Low: ${channel.predicted_low:.2f}")

    # Test position
    current_price = handler.get_latest_price()
    position = calc.get_channel_position(current_price, channel)
    print(f"\nCurrent Price Position:")
    print(f"  Price: ${current_price:.2f}")
    print(f"  Zone: {position['zone']}")
    print(f"  Distance to upper: {position['distance_to_upper_pct']:.2f}%")
    print(f"  Distance to lower: {position['distance_to_lower_pct']:.2f}%")
