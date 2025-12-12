"""Linear regression channel calculator with ping-pong detection."""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import config

# Numba JIT compilation for performance-critical loops
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@dataclass
class ChannelData:
    """Data class for OHLC linear regression channel."""
    # Close-based regression (primary trend)
    close_slope: float
    close_intercept: float
    close_r_squared: float

    # High-based regression (resistance)
    high_slope: float
    high_intercept: float
    high_r_squared: float

    # Low-based regression (support)
    low_slope: float
    low_intercept: float
    low_r_squared: float

    # Channel boundaries (composite of OHLC)
    upper_line: np.ndarray
    lower_line: np.ndarray
    center_line: np.ndarray  # Based on close

    # Channel metrics
    std_dev: float
    channel_width_pct: float  # (upper - lower) / close as percentage
    slope_convergence: float  # High/low slope divergence

    # Ping-pong detection at multiple thresholds (LEGACY: v3.16-)
    ping_pongs: int  # At 2% threshold (alternating transitions - deprecated)
    ping_pongs_0_5pct: int
    ping_pongs_1_0pct: int
    ping_pongs_3_0pct: int

    # Complete cycle detection (v3.17+: NEW - preferred metric)
    complete_cycles: int  # At 2% threshold (full round-trips) - REQUIRED like ping_pongs
    complete_cycles_0_5pct: int
    complete_cycles_1_0pct: int
    complete_cycles_3_0pct: int

    # Quality metrics (required fields)
    r_squared: float  # Average of close/high/low r²
    stability_score: float

    # Predictions (required fields)
    predicted_high: float
    predicted_low: float
    predicted_center: float

    # Optional fields with defaults (MUST come last in dataclass!)
    actual_duration: int = 0
    quality_score: float = 0.0  # Composite quality (0-1): v5.3.2: cycles × (0.5 + 0.5 × r²) - bounces primary
    is_valid: float = 0.0  # 1.0 if complete_cycles >= 2, else 0.0
    insufficient_data: float = 0.0  # 1.0 if window > available bars, else 0.0

    # Backward compatibility properties
    @property
    def slope(self) -> float:
        """Backward compatibility: return close_slope as 'slope'"""
        return self.close_slope

    @property
    def intercept(self) -> float:
        """Backward compatibility: return close_intercept as 'intercept'"""
        return self.close_intercept


# Numba JIT-compiled ping-pong detection functions for 10-15% speedup
if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, fastmath=True)
    def _detect_ping_pongs_jit(prices: np.ndarray, upper: np.ndarray, lower: np.ndarray,
                                threshold: float = 0.02) -> int:
        """
        JIT-compiled ping-pong detection - MUST use regular range (not prange)
        because last_touch is sequential state
        """
        bounces = 0
        last_touch = 0  # 0=none, 1=upper, 2=lower

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper[i]
            lower_val = lower[i]

            # Calculate distances as percentage
            upper_dist = abs(price - upper_val) / upper_val
            lower_dist = abs(price - lower_val) / lower_val

            # Check if price touches upper line
            if upper_dist <= threshold:
                if last_touch == 2:  # Was at lower
                    bounces += 1
                last_touch = 1

            # Check if price touches lower line
            elif lower_dist <= threshold:
                if last_touch == 1:  # Was at upper
                    bounces += 1
                last_touch = 2

        return bounces

    @numba.jit(nopython=True, fastmath=True)
    def _detect_complete_cycles_jit(prices: np.ndarray, upper: np.ndarray, lower: np.ndarray,
                                     threshold: float = 0.02) -> int:
        """
        JIT-compiled complete cycles detection - MUST use regular range
        Note: Returns touches as list, then counts cycles in Python
        """
        touches = np.empty(len(prices), dtype=np.int8)  # 0=none, 1=upper, 2=lower
        touch_count = 0
        last_touch = 0

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper[i]
            lower_val = lower[i]

            # Calculate distances with zero protection
            if upper_val > 0:
                upper_dist = abs(price - upper_val) / upper_val
            else:
                upper_dist = 1.0

            if lower_val != 0:
                lower_dist = abs(price - lower_val) / abs(lower_val)
            else:
                lower_dist = 1.0

            # Record touches (only when transitioning)
            if upper_dist <= threshold and last_touch != 1:
                touches[touch_count] = 1  # upper
                touch_count += 1
                last_touch = 1
            elif lower_dist <= threshold and last_touch != 2:
                touches[touch_count] = 2  # lower
                touch_count += 1
                last_touch = 2

        # Count complete cycles
        complete_cycles = 0
        i = 0
        while i < touch_count - 2:
            # Lower → Upper → Lower (2 → 1 → 2)
            if touches[i] == 2 and touches[i+1] == 1 and touches[i+2] == 2:
                complete_cycles += 1
                i += 2
            # Upper → Lower → Upper (1 → 2 → 1)
            elif touches[i] == 1 and touches[i+1] == 2 and touches[i+2] == 1:
                complete_cycles += 1
                i += 2
            else:
                i += 1

        return complete_cycles


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

        # Detect ping-pongs and complete cycles at multiple thresholds
        multi_pp = self._detect_ping_pongs_multi_threshold(
            prices, upper_line, lower_line,
            thresholds=[0.005, 0.01, 0.02, 0.03]
        )

        # v3.17: Calculate complete cycles (full round-trips)
        multi_cycles = self._detect_complete_cycles_multi_threshold(
            prices, upper_line, lower_line,
            thresholds=[0.005, 0.01, 0.02, 0.03]
        )

        # Calculate stability score (0-100) - use complete_cycles now
        stability_score = self._calculate_stability(r_squared, multi_cycles[0.02], n)

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
            # Close-based regression (primary)
            close_slope=slope,
            close_intercept=intercept,
            close_r_squared=r_squared,
            # For simple channel, high/low use same slope as close
            high_slope=slope,
            high_intercept=intercept + (self.std_dev * residual_std),
            high_r_squared=r_squared,
            low_slope=slope,
            low_intercept=intercept - (self.std_dev * residual_std),
            low_r_squared=r_squared,
            # Channel lines
            upper_line=upper_line,
            lower_line=lower_line,
            center_line=center_line,
            # Metrics
            std_dev=residual_std,
            channel_width_pct=((upper_line.mean() - lower_line.mean()) / center_line.mean() * 100) if center_line.mean() > 0 else 0,
            slope_convergence=0.0,  # No divergence in simple channel
            # Ping-pongs at multiple thresholds
            ping_pongs=multi_pp[0.02],
            ping_pongs_0_5pct=multi_pp[0.005],
            ping_pongs_1_0pct=multi_pp[0.01],
            ping_pongs_3_0pct=multi_pp[0.03],
            # v3.17: Complete cycles at multiple thresholds
            complete_cycles=multi_cycles[0.02],
            complete_cycles_0_5pct=multi_cycles[0.005],
            complete_cycles_1_0pct=multi_cycles[0.01],
            complete_cycles_3_0pct=multi_cycles[0.03],
            # Quality metrics
            r_squared=r_squared,
            stability_score=stability_score,
            # Predictions
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

            # Score = Ping-pongs primary, R² only matters if bounces exist
            # v5.3.2: Flipped weighting - actual price confirmations more important than statistical fit
            ping_pong_score = min(channel.ping_pongs / 10.0, 1.0)  # Normalize to 0-1
            composite_score = ping_pong_score * (0.5 + 0.5 * channel.r_squared)

            # Track best
            if composite_score > best_score:
                best_channel = channel
                best_score = composite_score

        return best_channel  # None if no valid channel found

    def find_best_channel_any_quality(
        self,
        df: pd.DataFrame,
        timeframe: str = "1h",
        max_lookback: int = 168
    ) -> Optional[ChannelData]:
        """
        Find best channel regardless of cycle quality (NO FILTERING).

        Returns channel with highest r² even if complete_cycles=0.
        This allows model to learn from "bad" channels as signals of timeframe unreliability.

        Used in continuation labels to avoid skipping timestamps - "bad" channels indicate
        that THIS timeframe is unreliable right now, model should learn to switch to another TF.

        Args:
            df: DataFrame with OHLC data
            timeframe: Timeframe name (for prediction)
            max_lookback: Maximum bars to consider

        Returns:
            ChannelData for best-fit window, or None if insufficient data
        """
        # Candidate windows to test (from longest to shortest)
        candidates = [168, 120, 90, 60, 45, 30, 20, 10]
        candidates = [c for c in candidates if c <= max_lookback and c <= len(df)]

        if not candidates:
            return None

        best_channel = None
        best_r_squared = 0

        for lookback in candidates:
            # Calculate channel for this window
            window = df.tail(lookback)
            channel = self.calculate_channel(window, lookback, timeframe)

            # NO FILTERING - just find best statistical fit
            # Even channels with 0 complete_cycles are useful signals!
            if channel.r_squared > best_r_squared:
                best_channel = channel
                best_r_squared = channel.r_squared

        return best_channel  # Always returns something (unless no data)

    def _detect_ping_pongs(self, prices: np.ndarray, upper: np.ndarray, lower: np.ndarray,
                          threshold: float = 0.02) -> int:
        """
        Detect bounces between upper and lower channel lines.
        Uses JIT-compiled version if Numba is available (10-15% faster).

        Args:
            prices: Price array
            upper: Upper channel line
            lower: Lower channel line
            threshold: Percentage threshold for detecting touch (2% default)

        Returns:
            Number of ping-pongs detected
        """
        # Use JIT version if available
        if NUMBA_AVAILABLE:
            return _detect_ping_pongs_jit(prices, upper, lower, threshold)

        # Fallback to Python implementation
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

    def _detect_complete_cycles(self, prices: np.ndarray, upper: np.ndarray, lower: np.ndarray,
                                threshold: float = 0.02) -> int:
        """
        Count complete oscillation cycles (lower→upper→lower or upper→lower→upper).
        Uses JIT-compiled version if Numba is available (10-15% faster).

        This is a stricter measure than transitions - requires full round-trips.
        Better indicator of channel stability and mean reversion behavior.

        Args:
            prices: Price array
            upper: Upper channel line
            lower: Lower channel line
            threshold: Percentage threshold for detecting touch (2% default)

        Returns:
            Number of complete oscillation cycles
        """
        # Use JIT version if available
        if NUMBA_AVAILABLE:
            return _detect_complete_cycles_jit(prices, upper, lower, threshold)

        # Fallback to Python implementation
        touches = []
        last_touch = None

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper[i]
            lower_val = lower[i]

            # Calculate distances as percentage
            if upper_val > 0:
                upper_dist = abs(price - upper_val) / upper_val
            else:
                upper_dist = 1.0

            if lower_val != 0:
                lower_dist = abs(price - lower_val) / abs(lower_val)
            else:
                lower_dist = 1.0

            # Record touches (only when transitioning to new boundary)
            if upper_dist <= threshold and last_touch != 'upper':
                touches.append('upper')
                last_touch = 'upper'
            elif lower_dist <= threshold and last_touch != 'lower':
                touches.append('lower')
                last_touch = 'lower'

        # Count complete cycles (need at least 3 touches for 1 cycle)
        complete_cycles = 0
        i = 0
        while i < len(touches) - 2:
            # Lower → Upper → Lower (one complete cycle)
            if (touches[i] == 'lower' and touches[i+1] == 'upper' and touches[i+2] == 'lower'):
                complete_cycles += 1
                i += 2  # Skip to next potential cycle start
            # Upper → Lower → Upper (one complete cycle)
            elif (touches[i] == 'upper' and touches[i+1] == 'lower' and touches[i+2] == 'upper'):
                complete_cycles += 1
                i += 2  # Skip to next potential cycle start
            else:
                i += 1  # Move forward one touch

        return complete_cycles

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

    def _detect_complete_cycles_multi_threshold(
        self,
        prices: np.ndarray,
        upper: np.ndarray,
        lower: np.ndarray,
        thresholds: list = [0.005, 0.01, 0.02, 0.03]
    ) -> dict:
        """
        Count complete oscillation cycles at multiple thresholds (efficient single-pass).

        Complete cycle = lower→upper→lower OR upper→lower→upper (full round-trip).
        More robust measure of channel stability than simple transitions.

        Args:
            prices: Price array
            upper: Upper channel line
            lower: Lower channel line
            thresholds: List of percentage thresholds to test

        Returns:
            Dict mapping threshold to complete cycle count
            Example: {0.005: 2, 0.01: 3, 0.02: 4, 0.03: 5}
        """
        results = {threshold: 0 for threshold in thresholds}
        touch_sequences = {threshold: [] for threshold in thresholds}
        last_touch = {threshold: None for threshold in thresholds}

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper[i]
            lower_val = lower[i]

            # Calculate distances as percentage
            if upper_val > 0:
                upper_dist = abs(price - upper_val) / upper_val
            else:
                upper_dist = 1.0

            if lower_val != 0:
                lower_dist = abs(price - lower_val) / abs(lower_val)
            else:
                lower_dist = 1.0

            # Check each threshold
            for threshold in thresholds:
                # Record touches (transitions only)
                if upper_dist <= threshold and last_touch[threshold] != 'upper':
                    touch_sequences[threshold].append('upper')
                    last_touch[threshold] = 'upper'
                elif lower_dist <= threshold and last_touch[threshold] != 'lower':
                    touch_sequences[threshold].append('lower')
                    last_touch[threshold] = 'lower'

        # Count complete cycles for each threshold
        for threshold in thresholds:
            touches = touch_sequences[threshold]
            cycles = 0
            i = 0

            while i < len(touches) - 2:
                # Check for complete round-trip patterns
                if (touches[i] == 'lower' and touches[i+1] == 'upper' and touches[i+2] == 'lower') or \
                   (touches[i] == 'upper' and touches[i+1] == 'lower' and touches[i+2] == 'upper'):
                    cycles += 1
                    i += 2  # Skip to next potential cycle
                else:
                    i += 1  # Move forward one touch

            results[threshold] = cycles

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

    def _calculate_r_squared(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate R-squared (coefficient of determination)."""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - (ss_res / ss_tot)

    def _detect_ping_pongs_vectorized(self, highs: np.ndarray, lows: np.ndarray,
                                      closes: np.ndarray, upper: np.ndarray,
                                      lower: np.ndarray) -> dict:
        """Vectorized ping-pong detection (10x faster than Python loops)."""
        thresholds = [0.005, 0.01, 0.02, 0.03]
        results = {}

        for threshold in thresholds:
            high_upper_dist = np.abs(highs - upper) / np.maximum(upper, 1e-10)
            low_lower_dist = np.abs(lows - lower) / np.maximum(np.abs(lower), 1e-10)
            close_upper_dist = np.abs(closes - upper) / np.maximum(upper, 1e-10)
            close_lower_dist = np.abs(closes - lower) / np.maximum(np.abs(lower), 1e-10)

            touches_upper = (high_upper_dist <= threshold) | (close_upper_dist <= threshold)
            touches_lower = (low_lower_dist <= threshold) | (close_lower_dist <= threshold)

            state = np.zeros(len(closes), dtype=int)
            state[touches_upper & ~touches_lower] = 1
            state[touches_lower & ~touches_upper] = 2

            bounces = 0
            last_state = 0
            for s in state:
                if s != 0 and s != last_state and last_state != 0:
                    bounces += 1
                if s != 0:
                    last_state = s

            results[threshold] = int(bounces)

        return results

    def _detect_complete_cycles_vectorized(self, highs: np.ndarray, lows: np.ndarray,
                                          closes: np.ndarray, upper: np.ndarray,
                                          lower: np.ndarray) -> dict:
        """
        Vectorized complete cycle detection (10x faster than Python loops).

        Uses numpy operations for touch detection, then Python loop for cycle counting
        (cycle counting is inherently sequential - difficult to fully vectorize).
        """
        thresholds = [0.005, 0.01, 0.02, 0.03]
        results = {}

        for threshold in thresholds:
            # Vectorized touch detection (same as ping_pongs)
            high_upper_dist = np.abs(highs - upper) / np.maximum(upper, 1e-10)
            low_lower_dist = np.abs(lows - lower) / np.maximum(np.abs(lower), 1e-10)
            close_upper_dist = np.abs(closes - upper) / np.maximum(upper, 1e-10)
            close_lower_dist = np.abs(closes - lower) / np.maximum(np.abs(lower), 1e-10)

            touches_upper = (high_upper_dist <= threshold) | (close_upper_dist <= threshold)
            touches_lower = (low_lower_dist <= threshold) | (close_lower_dist <= threshold)

            # Build state array: 0=none, 1=upper, 2=lower
            state = np.zeros(len(closes), dtype=int)
            state[touches_upper & ~touches_lower] = 1
            state[touches_lower & ~touches_upper] = 2

            # Build touch sequence (filter state transitions)
            touch_seq = []
            last_state = 0
            for s in state:
                if s != 0 and s != last_state:
                    touch_seq.append('upper' if s == 1 else 'lower')
                    last_state = s

            # Count complete cycles
            cycles = 0
            i = 0
            while i < len(touch_seq) - 2:
                if (touch_seq[i] == 'lower' and touch_seq[i+1] == 'upper' and touch_seq[i+2] == 'lower') or \
                   (touch_seq[i] == 'upper' and touch_seq[i+1] == 'lower' and touch_seq[i+2] == 'upper'):
                    cycles += 1
                    i += 2
                else:
                    i += 1

            results[threshold] = int(cycles)

        return results

    def calculate_multi_window_rolling(self, df: pd.DataFrame, timeframe: str = "1h") -> Dict[int, List[Optional[ChannelData]]]:
        """
        Calculate channels for MULTIPLE window sizes using rolling statistics.
        Returns results for ALL candidate windows, not just the best one.

        ~45x faster than old method per window, processes 6 windows simultaneously.

        Args:
            df: DataFrame with OHLC data
            timeframe: Timeframe name (for adaptive window selection)

        Returns:
            Dict mapping window_size -> List[ChannelData]
            Example: {168: [ch1, ch2, ...], 120: [ch1, ch2, ...], ...}
        """
        # Use centralized window sizes for ALL timeframes (no filtering!)
        candidates = config.CHANNEL_WINDOW_SIZES

        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        n = len(closes)

        all_results = {}

        # Calculate for each window size
        for window in candidates:
            results = []

            # Check if we have enough data for this window
            if window > n or window < 10:
                # Insufficient data - create bad-score channels for all bars
                for i in range(n):
                    if i < window:
                        results.append(None)  # Not enough history yet
                    else:
                        # Create channel with bad score
                        empty_array = np.zeros(window)
                        results.append(ChannelData(
                            close_slope=0.0, close_intercept=0.0, close_r_squared=0.0,
                            high_slope=0.0, high_intercept=0.0, high_r_squared=0.0,
                            low_slope=0.0, low_intercept=0.0, low_r_squared=0.0,
                            upper_line=empty_array, lower_line=empty_array, center_line=empty_array,
                            std_dev=0.0, channel_width_pct=0.0, slope_convergence=0.0,
                            ping_pongs=0, ping_pongs_0_5pct=0, ping_pongs_1_0pct=0, ping_pongs_3_0pct=0,
                            complete_cycles=0, complete_cycles_0_5pct=0, complete_cycles_1_0pct=0, complete_cycles_3_0pct=0,
                            r_squared=0.0, stability_score=0.0,
                            predicted_high=0.0, predicted_low=0.0, predicted_center=0.0,
                            actual_duration=0, quality_score=0.0, is_valid=0.0,
                            insufficient_data=1.0  # Flag: not enough data
                        ))
                all_results[window] = results
                continue  # Skip to next window

            # Pre-calculate constant sums for this window
            sum_x = np.arange(window).sum()
            sum_xx = (np.arange(window) ** 2).sum()

            # Initialize sums for first window
            sum_close = closes[0:window].sum()
            sum_high = highs[0:window].sum()
            sum_low = lows[0:window].sum()

            sum_x_close = (np.arange(window) * closes[0:window]).sum()
            sum_x_high = (np.arange(window) * highs[0:window]).sum()
            sum_x_low = (np.arange(window) * lows[0:window]).sum()

            # Fill early bars with None
            for i in range(window):
                results.append(None)

            # Rolling calculation for this window size
            for i in range(window, n):
                denom = (window * sum_xx - sum_x**2)
                if abs(denom) < 1e-10:
                    results.append(None)
                    continue

                close_slope = (window * sum_x_close - sum_x * sum_close) / denom
                high_slope = (window * sum_x_high - sum_x * sum_high) / denom
                low_slope = (window * sum_x_low - sum_x * sum_low) / denom

                close_intercept = (sum_close - close_slope * sum_x) / window
                high_intercept = (sum_high - high_slope * sum_x) / window
                low_intercept = (sum_low - low_slope * sum_x) / window

                x_window = np.arange(window)
                close_line = close_slope * x_window + close_intercept
                high_line = high_slope * x_window + high_intercept
                low_line = low_slope * x_window + low_intercept

                window_closes = closes[i-window:i]
                window_highs = highs[i-window:i]
                window_lows = lows[i-window:i]

                residuals = window_closes - close_line
                residual_std = np.std(residuals)

                upper_line = np.maximum(high_line, close_line + (self.std_dev * residual_std))
                lower_line = np.minimum(low_line, close_line - (self.std_dev * residual_std))

                channel_width_pct = ((upper_line - lower_line) / np.maximum(close_line, 1e-10)).mean() * 100
                slope_convergence = (high_slope - low_slope) / abs(close_slope) if abs(close_slope) > 1e-10 else 0.0

                multi_pp = self._detect_ping_pongs_vectorized(
                    window_highs, window_lows, window_closes, upper_line, lower_line
                )

                # NEW: Complete cycle detection (v3.17)
                multi_cycles = self._detect_complete_cycles_vectorized(
                    window_highs, window_lows, window_closes, upper_line, lower_line
                )

                close_r_squared = self._calculate_r_squared(window_closes, close_line)
                high_r_squared = self._calculate_r_squared(window_highs, high_line)
                low_r_squared = self._calculate_r_squared(window_lows, low_line)
                r_squared_avg = (close_r_squared + high_r_squared + low_r_squared) / 3

                stability_score = self._calculate_stability(r_squared_avg, multi_pp[0.02], window)

                # Calculate quality indicators (v3.17: Switched to complete_cycles)
                # v5.3.2: Ping-pongs/cycles primary - R² only matters if bounces exist
                cycle_score = min(multi_cycles[0.02] / 5.0, 1.0)  # Normalize: 5 cycles = max
                quality_score = cycle_score * (0.5 + 0.5 * r_squared_avg)
                is_valid = 1.0 if multi_cycles[0.02] >= 2 else 0.0  # Require 2 complete cycles

                channel = ChannelData(
                    close_slope=close_slope,
                    close_intercept=close_intercept,
                    close_r_squared=close_r_squared,
                    high_slope=high_slope,
                    high_intercept=high_intercept,
                    high_r_squared=high_r_squared,
                    low_slope=low_slope,
                    low_intercept=low_intercept,
                    low_r_squared=low_r_squared,
                    upper_line=upper_line,
                    lower_line=lower_line,
                    center_line=close_line,
                    std_dev=residual_std,
                    channel_width_pct=channel_width_pct,
                    slope_convergence=slope_convergence,
                    ping_pongs=multi_pp[0.02],
                    ping_pongs_0_5pct=multi_pp[0.005],
                    ping_pongs_1_0pct=multi_pp[0.01],
                    ping_pongs_3_0pct=multi_pp[0.03],
                    complete_cycles=multi_cycles[0.02],
                    complete_cycles_0_5pct=multi_cycles[0.005],
                    complete_cycles_1_0pct=multi_cycles[0.01],
                    complete_cycles_3_0pct=multi_cycles[0.03],
                    r_squared=r_squared_avg,
                    stability_score=stability_score,
                    predicted_high=upper_line[-1],
                    predicted_low=lower_line[-1],
                    predicted_center=close_line[-1],
                    actual_duration=window,
                    quality_score=quality_score,
                    is_valid=is_valid,
                    insufficient_data=0.0  # Sufficient data for this calculation
                )
                results.append(channel)

                # Update sums for next iteration (O(1) operation!)
                old_close = closes[i-window]
                old_high = highs[i-window]
                old_low = lows[i-window]
                new_close = closes[i]
                new_high = highs[i]
                new_low = lows[i]

                sum_close = sum_close - old_close + new_close
                sum_high = sum_high - old_high + new_high
                sum_low = sum_low - old_low + new_low

                sum_x_close = sum_x_close - sum_close + window * new_close
                sum_x_high = sum_x_high - sum_high + window * new_high
                sum_x_low = sum_x_low - sum_low + window * new_low

            all_results[window] = results

        return all_results


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
