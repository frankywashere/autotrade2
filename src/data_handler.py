"""Data handler for loading and resampling stock data."""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import config


class DataHandler:
    """Handles loading and resampling of stock data."""

    def __init__(self, stock: str = config.DEFAULT_STOCK, use_live_data: Optional[bool] = None):
        """
        Initialize data handler.

        Args:
            stock: Stock symbol (e.g., 'TSLA', 'SPY')
            use_live_data: If True, merge CSV with live data from yfinance (defaults to config.USE_LIVE_DATA)
        """
        self.stock = stock
        self.use_live_data = use_live_data if use_live_data is not None else config.USE_LIVE_DATA
        self.data_1min: Optional[pd.DataFrame] = None
        self.resampled_data: Dict[str, pd.DataFrame] = {}
        self.data_freshness: Optional[dict] = None

    def load_1min_data(self) -> pd.DataFrame:
        """
        Load 1-minute data from CSV file, optionally merging with live data.

        Returns:
            DataFrame with 1-minute OHLCV data
        """
        file_path = config.DATA_DIR / f"{self.stock}_1min.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Load data with proper datetime parsing
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # Ensure we have OHLCV columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        print(f"Loaded {len(df)} rows of historical data for {self.stock}")
        print(f"Historical range: {df.index.min()} to {df.index.max()}")

        # Merge with live data if enabled
        if self.use_live_data:
            try:
                from live_data_fetcher import LiveDataFetcher
                fetcher = LiveDataFetcher(self.stock)

                # Merge historical + live
                df = fetcher.merge_with_historical(df, days_back=config.LIVE_DATA_DAYS_BACK)

                # Get data freshness
                self.data_freshness = fetcher.get_data_freshness(df.index.max())

                print(f"\n📊 Data Status: {self.data_freshness['status'].upper()}")
                print(f"   {self.data_freshness['message']}")

            except Exception as e:
                print(f"Warning: Could not fetch live data: {e}")
                print("Continuing with historical data only...")

        self.data_1min = df
        return df

    def resample_data(self, timeframe: str) -> pd.DataFrame:
        """
        Resample 1-minute data to specified timeframe.

        Args:
            timeframe: One of '1hour', '2hour', '3hour', '4hour', 'daily', 'weekly'

        Returns:
            Resampled DataFrame with OHLCV data
        """
        if self.data_1min is None:
            self.load_1min_data()

        if timeframe not in config.TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(config.TIMEFRAMES.keys())}")

        # Get pandas resample rule
        rule = config.TIMEFRAMES[timeframe]

        # Resample OHLCV data
        resampled = self.data_1min.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Drop rows with NaN (incomplete periods)
        resampled.dropna(inplace=True)

        self.resampled_data[timeframe] = resampled
        print(f"Resampled to {timeframe}: {len(resampled)} rows")

        return resampled

    def get_data(self, timeframe: str = "1min") -> pd.DataFrame:
        """
        Get data for specified timeframe.

        Args:
            timeframe: Timeframe to retrieve

        Returns:
            DataFrame with OHLCV data
        """
        if timeframe == "1min":
            if self.data_1min is None:
                self.load_1min_data()
            return self.data_1min

        if timeframe not in self.resampled_data:
            self.resample_data(timeframe)

        return self.resampled_data[timeframe]

    def get_all_timeframes(self) -> Dict[str, pd.DataFrame]:
        """
        Get data for all configured timeframes.

        Returns:
            Dictionary mapping timeframe names to DataFrames
        """
        # Load 1-minute data first
        if self.data_1min is None:
            self.load_1min_data()

        # Resample to all timeframes
        for timeframe in config.TIMEFRAMES.keys():
            if timeframe != "1min" and timeframe not in self.resampled_data:
                self.resample_data(timeframe)

        return {
            "1min": self.data_1min,
            **self.resampled_data
        }

    def get_latest_price(self) -> float:
        """
        Get the most recent close price.

        Returns:
            Latest close price
        """
        if self.data_1min is None:
            self.load_1min_data()

        return float(self.data_1min['close'].iloc[-1])

    def get_price_at_time(self, timestamp: pd.Timestamp, timeframe: str = "1min") -> Optional[float]:
        """
        Get close price at specific timestamp.

        Args:
            timestamp: Time to query
            timeframe: Timeframe to use

        Returns:
            Close price at timestamp, or None if not found
        """
        df = self.get_data(timeframe)

        if timestamp in df.index:
            return float(df.loc[timestamp, 'close'])

        # Find nearest timestamp
        idx = df.index.get_indexer([timestamp], method='nearest')[0]
        if idx >= 0 and idx < len(df):
            return float(df.iloc[idx]['close'])

        return None


if __name__ == "__main__":
    # Test the data handler
    handler = DataHandler("TSLA")
    handler.load_1min_data()

    print("\nResampling to different timeframes:")
    for tf in ["1hour", "4hour", "daily"]:
        data = handler.get_data(tf)
        print(f"\n{tf}: {len(data)} bars")
        print(data.tail())
