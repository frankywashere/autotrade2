"""
Concrete DataFeed implementations
Currently supports CSV, easily extensible to IBKR/Alpha Vantage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from .base import DataFeed
import sys
import os
from tqdm import tqdm

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config


class CSVDataFeed(DataFeed):
    """
    CSV-based data feed using existing historical data
    Aligns SPY and TSLA by timestamp using inner join
    """

    def __init__(self, data_dir: str = None, timeframe: str = '1min'):
        self.data_dir = data_dir or config.DATA_DIR
        self.timeframe = timeframe
        self.cache = {}

    def load_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load historical data from CSV for specified timeframe"""
        csv_path = Path(self.data_dir) / f"{symbol}_{self.timeframe}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        # Load CSV
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Filter by date range if specified
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        # Validate
        if not self.validate_data(df):
            raise ValueError(f"Data validation failed for {symbol}")

        return df

    def get_latest_data(self, symbol: str, bars: int = 100) -> pd.DataFrame:
        """Get most recent N bars"""
        df = self.load_data(symbol)
        return df.tail(bars)

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data integrity:
        - No nulls in OHLCV columns
        - No zeros in price columns
        - Timestamps are sorted
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']

        # Check for required columns
        if not all(col in df.columns for col in required_cols):
            return False

        # Check for nulls
        if df[required_cols].isnull().any().any():
            return False

        # Check for zeros in price columns
        price_cols = ['open', 'high', 'low', 'close']
        if (df[price_cols] == 0).any().any():
            return False

        # Check if timestamps are sorted
        if not df.index.is_monotonic_increasing:
            return False

        return True

    def align_symbols(self, spy_df: pd.DataFrame, tsla_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align SPY and TSLA data by timestamp using inner join
        Ensures no nulls or zeros in overlapping periods
        """
        # Inner join on timestamp index
        common_timestamps = spy_df.index.intersection(tsla_df.index)

        spy_aligned = spy_df.loc[common_timestamps].copy()
        tsla_aligned = tsla_df.loc[common_timestamps].copy()

        # Add suffix to distinguish columns
        spy_aligned.columns = [f'spy_{col}' for col in spy_aligned.columns]
        tsla_aligned.columns = [f'tsla_{col}' for col in tsla_aligned.columns]

        # Validate alignment
        assert len(spy_aligned) == len(tsla_aligned), "Alignment failed: length mismatch"
        assert (spy_aligned.index == tsla_aligned.index).all(), "Alignment failed: timestamp mismatch"

        # Final validation
        if not self.validate_data(spy_aligned.rename(columns=lambda x: x.replace('spy_', ''))) or \
           not self.validate_data(tsla_aligned.rename(columns=lambda x: x.replace('tsla_', ''))):
            raise ValueError("Aligned data validation failed")

        return spy_aligned, tsla_aligned

    def load_aligned_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load SPY and TSLA data aligned by timestamp
        Returns single DataFrame with both symbols
        """
        # Progress bar for data loading steps
        with tqdm(total=3, desc="   Loading data", ncols=80, leave=False, ascii=True) as pbar:
            # Load SPY
            pbar.set_description("   Loading SPY")
            spy_df = self.load_data('SPY', start_date, end_date)
            pbar.update(1)

            # Load TSLA
            pbar.set_description("   Loading TSLA")
            tsla_df = self.load_data('TSLA', start_date, end_date)
            pbar.update(1)

            # Align symbols
            pbar.set_description("   Aligning data")
            spy_aligned, tsla_aligned = self.align_symbols(spy_df, tsla_df)
            pbar.update(1)

        # Merge into single DataFrame
        aligned_df = pd.concat([spy_aligned, tsla_aligned], axis=1)

        return aligned_df


class YFinanceDataFeed(DataFeed):
    """
    Live data feed using yfinance
    Future extension point for real-time data
    """

    def __init__(self):
        import yfinance as yf
        self.yf = yf

    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data from yfinance"""
        ticker = self.yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1m')

        # Rename columns to match our format
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        df.index.name = 'timestamp'
        df = df[['open', 'high', 'low', 'close', 'volume']]

        return df

    def get_latest_data(self, symbol: str, bars: int = 100) -> pd.DataFrame:
        """Get recent data from yfinance"""
        ticker = self.yf.Ticker(symbol)
        df = ticker.history(period='7d', interval='1m')

        # Rename columns
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        return df.tail(bars)

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data from yfinance"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        return all(col in df.columns for col in required_cols)
