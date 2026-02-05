"""
RSI Monitor Core Module

Provides the main RSIMonitor class for calculating and tracking RSI values
across multiple symbols and timeframes.
"""

import numpy as np
from typing import Dict, List, Optional

from .data import DataFetcher


class RSIMonitor:
    """
    Monitor RSI values across multiple symbols and timeframes.

    The RSI (Relative Strength Index) is a momentum indicator that measures
    the speed and magnitude of recent price changes to evaluate overbought
    or oversold conditions.

    Attributes:
        symbols: List of ticker symbols to monitor.
        timeframes: List of timeframes to analyze.
        data_fetcher: DataFetcher instance for retrieving price data.

    Example:
        >>> monitor = RSIMonitor()
        >>> rsi_data = monitor.get_all_rsi()
        >>> score = monitor.get_confluence_score('TSLA')
    """

    DEFAULT_SYMBOLS = ['TSLA', 'SPY', '^VIX']
    DEFAULT_TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d', '1wk']

    OVERSOLD_THRESHOLD = 30
    OVERBOUGHT_THRESHOLD = 70

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        data_fetcher: Optional[DataFetcher] = None
    ):
        """
        Initialize the RSI Monitor.

        Args:
            symbols: List of ticker symbols to monitor.
                     Defaults to ['TSLA', 'SPY', '^VIX'].
            timeframes: List of timeframes to analyze.
                        Defaults to ['5m', '15m', '1h', '4h', '1d', '1wk'].
            data_fetcher: Optional DataFetcher instance. If not provided,
                          a new instance will be created.
        """
        self.symbols = symbols if symbols is not None else self.DEFAULT_SYMBOLS.copy()
        self.timeframes = timeframes if timeframes is not None else self.DEFAULT_TIMEFRAMES.copy()
        self.data_fetcher = data_fetcher if data_fetcher is not None else DataFetcher()

    # Timeframes that support percentile calculation (daily and weekly)
    PERCENTILE_TIMEFRAMES = {'5m', '15m', '1h', '4h', '1d', '1wk', 'daily', 'weekly', 'D', 'W'}

    def _calculate_rsi_series(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate RSI values for all bars in a price series (rolling calculation).

        This method computes RSI for each bar in the series, allowing you to
        build a historical distribution of RSI values.

        Args:
            prices: Array of closing prices, ordered from oldest to newest.
            period: The lookback period for RSI calculation. Defaults to 14.

        Returns:
            A numpy array of RSI values, one for each bar starting from
            index `period`. Earlier indices will be np.nan due to insufficient
            data for calculation.

        Example:
            >>> prices = np.array([44, 44.5, 43.5, 44.5, 45, 45.5, 46, ...])
            >>> monitor = RSIMonitor()
            >>> rsi_series = monitor._calculate_rsi_series(prices)
            >>> # rsi_series[period] is the first valid RSI value
        """
        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)

        # Initialize RSI array with NaN
        rsi_values = np.full(n, np.nan)

        # Need at least period + 1 prices to calculate RSI
        if n < period + 1:
            return rsi_values

        # Calculate price changes
        deltas = np.diff(prices)

        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Calculate initial average gain and loss using SMA
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        # Calculate RSI for the first valid point (index = period)
        if avg_loss == 0:
            if avg_gain == 0:
                rsi_values[period] = 50.0
            else:
                rsi_values[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[period] = 100.0 - (100.0 / (1.0 + rs))

        # Apply Wilder's smoothing for remaining periods
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            # Calculate RSI for this bar (i+1 because deltas is offset by 1)
            if avg_loss == 0:
                if avg_gain == 0:
                    rsi_values[i + 1] = 50.0
                else:
                    rsi_values[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return rsi_values

    def _calculate_percentile(self, rsi_series: np.ndarray, current_rsi: float) -> Optional[float]:
        """
        Calculate the percentile rank of the current RSI within historical values.

        Args:
            rsi_series: Array of historical RSI values (may contain NaN).
            current_rsi: The current RSI value to rank.

        Returns:
            The percentile (0-100) indicating what percentage of historical
            RSI values are below the current RSI. Returns None if insufficient
            data or current_rsi is NaN.
        """
        if np.isnan(current_rsi):
            return None

        # Remove NaN values for percentile calculation
        valid_rsi = rsi_series[~np.isnan(rsi_series)]

        if len(valid_rsi) < 2:
            return None

        # Count values below current RSI
        values_below = np.sum(valid_rsi < current_rsi)
        total_values = len(valid_rsi)

        # Calculate percentile
        percentile = (values_below / total_values) * 100.0

        return round(percentile, 2)

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """
        Calculate the RSI (Relative Strength Index) for a price series.

        Uses the standard RSI calculation with exponential moving average
        (Wilder's smoothing method) for the average gains and losses.

        Args:
            prices: Array of closing prices, ordered from oldest to newest.
            period: The lookback period for RSI calculation. Defaults to 14.

        Returns:
            The RSI value as a float between 0 and 100.
            Returns np.nan if there is insufficient data.

        Raises:
            ValueError: If period is less than 1.

        Example:
            >>> prices = np.array([44, 44.5, 43.5, 44.5, 45, 45.5, 46, 45.5, 46, 46.5, 47, 46.5, 46, 46.5, 47])
            >>> monitor = RSIMonitor()
            >>> rsi = monitor.calculate_rsi(prices, period=14)
        """
        if period < 1:
            raise ValueError("Period must be at least 1")

        prices = np.asarray(prices, dtype=np.float64)

        # Need at least period + 1 prices to calculate RSI
        if len(prices) < period + 1:
            return np.nan

        # Calculate price changes
        deltas = np.diff(prices)

        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Calculate initial average gain and loss using SMA
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        # Apply Wilder's smoothing for remaining periods
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        # Calculate RS and RSI
        if avg_loss == 0:
            if avg_gain == 0:
                return 50.0  # No movement, neutral RSI
            return 100.0  # All gains, maximum RSI

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    def get_all_rsi(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate RSI values for all symbols across all timeframes.

        Fetches price data using the DataFetcher and calculates RSI
        for each symbol/timeframe combination.

        Returns:
            A nested dictionary with structure:
            {symbol: {timeframe: rsi_value}}

            RSI values will be np.nan if insufficient data is available.

        Example:
            >>> monitor = RSIMonitor(symbols=['TSLA', 'SPY'])
            >>> rsi_data = monitor.get_all_rsi()
            >>> print(rsi_data['TSLA']['1h'])
            65.43
        """
        results: Dict[str, Dict[str, float]] = {}

        for symbol in self.symbols:
            results[symbol] = {}
            for timeframe in self.timeframes:
                data = self.data_fetcher.fetch(symbol, timeframe)
                if data is not None and len(data) > 0:
                    prices = data['Close'].values
                    rsi = self.calculate_rsi(prices)
                else:
                    rsi = np.nan
                results[symbol][timeframe] = rsi

        return results

    def get_all_rsi_with_percentile(self) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
        """
        Calculate RSI values and percentiles for all symbols across all timeframes.

        For daily and weekly timeframes ('1d', '1wk', 'daily', 'weekly', 'D', 'W'),
        this method also calculates the historical percentile of the current RSI.
        The percentile indicates where the current RSI sits relative to all
        historical RSI values for that symbol/timeframe.

        Returns:
            A nested dictionary with structure:
            {
                symbol: {
                    timeframe: {
                        'rsi': float or np.nan,
                        'percentile': float or None
                    }
                }
            }

            - 'rsi': The current RSI value (0-100), or np.nan if insufficient data.
            - 'percentile': The percentile rank (0-100) of current RSI within
              historical values. Only calculated for daily/weekly timeframes;
              None for intraday timeframes or if insufficient data.

        Example:
            >>> monitor = RSIMonitor(symbols=['TSLA'])
            >>> data = monitor.get_all_rsi_with_percentile()
            >>> print(data['TSLA']['1d'])
            {'rsi': 28.5, 'percentile': 8.5}
            >>> print(data['TSLA']['5m'])
            {'rsi': 45.2, 'percentile': None}
        """
        results: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}

        for symbol in self.symbols:
            results[symbol] = {}
            for timeframe in self.timeframes:
                data = self.data_fetcher.fetch(symbol, timeframe)

                if data is not None and len(data) > 0:
                    prices = data['Close'].values
                    rsi = self.calculate_rsi(prices)

                    # Calculate percentile only for daily/weekly timeframes
                    if timeframe in self.PERCENTILE_TIMEFRAMES:
                        rsi_series = self._calculate_rsi_series(prices)
                        percentile = self._calculate_percentile(rsi_series, rsi)
                    else:
                        percentile = None
                else:
                    rsi = np.nan
                    percentile = None

                results[symbol][timeframe] = {
                    'rsi': rsi,
                    'percentile': percentile
                }

        return results

    def get_confluence_score(self, symbol: str) -> Dict[str, int]:
        """
        Calculate the confluence score for a symbol.

        Counts how many timeframes show oversold (<30) or overbought (>70)
        conditions. Higher confluence suggests stronger signals.

        Args:
            symbol: The ticker symbol to analyze.

        Returns:
            A dictionary with keys:
            - 'oversold': Count of timeframes with RSI < 30
            - 'overbought': Count of timeframes with RSI > 70
            - 'total_timeframes': Total number of timeframes analyzed
            - 'valid_readings': Number of timeframes with valid RSI data

        Example:
            >>> monitor = RSIMonitor()
            >>> score = monitor.get_confluence_score('TSLA')
            >>> if score['oversold'] >= 3:
            ...     print("Strong oversold confluence!")
        """
        oversold_count = 0
        overbought_count = 0
        valid_readings = 0

        for timeframe in self.timeframes:
            data = self.data_fetcher.fetch(symbol, timeframe)
            prices = data['Close'].values if data is not None and len(data) > 0 else None
            if prices is not None and len(prices) > 0:
                rsi = self.calculate_rsi(prices)
                if not np.isnan(rsi):
                    valid_readings += 1
                    if rsi < self.OVERSOLD_THRESHOLD:
                        oversold_count += 1
                    elif rsi > self.OVERBOUGHT_THRESHOLD:
                        overbought_count += 1

        return {
            'oversold': oversold_count,
            'overbought': overbought_count,
            'total_timeframes': len(self.timeframes),
            'valid_readings': valid_readings
        }
