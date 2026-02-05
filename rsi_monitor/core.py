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
            prices = self.data_fetcher.get_prices(symbol, timeframe)
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
