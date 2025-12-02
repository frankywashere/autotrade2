"""
Concrete DataFeed implementations
Currently supports CSV, easily extensible to IBKR/Alpha Vantage

v4.0: Added VIX loading and multi-timeframe native OHLC support
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
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

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate DataFrame has required columns and reasonable data.
        Provides detailed terminal output for debugging failures.
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']

        # Check for required columns
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"❌ Validation failed: Missing columns {missing}")
            print(f"   Available columns: {df.columns.tolist()}")
            return False

        # Check for nulls
        if df[required_cols].isnull().any().any():
            null_counts = df[required_cols].isnull().sum()
            print("❌ Validation failed: Null values in required columns")
            print(f"   Null counts: {null_counts.to_dict()}")
            return False

        # Check for zeros in price columns
        price_cols = ['open', 'high', 'low', 'close']
        zero_mask = (df[price_cols] == 0).any(axis=1)
        if zero_mask.any():
            zero_count = zero_mask.sum()
            print(f"❌ Validation failed: Zero values in price columns ({zero_count} rows)")
            print(f"   First few zero rows: {df[zero_mask].head(3).index.tolist()}")
            return False

        # Check if timestamps are sorted
        if not df.index.is_monotonic_increasing:
            print("❌ Validation failed: Timestamps not sorted")
            print("   Consider sorting data by timestamp before validation")
            return False

        # Check for reasonable data ranges
        if (df['high'] < df['low']).any():
            bad_count = (df['high'] < df['low']).sum()
            print(f"❌ Validation failed: High < Low in {bad_count} rows")
            return False

        if (df['close'] < df['low']).any() or (df['close'] > df['high']).any():
            bad_count = ((df['close'] < df['low']) | (df['close'] > df['high'])).sum()
            print(f"❌ Validation failed: Close outside High/Low range in {bad_count} rows")
            return False

        print("✅ Data validation passed")
        return True

    def load_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load historical data from CSV for specified timeframe"""
        csv_path = Path(self.data_dir) / f"{symbol}_{self.timeframe}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        # Load CSV with explicit dtype for precision control
        dtype_spec = {
            'open': config.NUMPY_DTYPE,
            'high': config.NUMPY_DTYPE,
            'low': config.NUMPY_DTYPE,
            'close': config.NUMPY_DTYPE,
            'volume': np.float64  # Volume can stay float64 for precision
        }
        df = pd.read_csv(csv_path, dtype=dtype_spec)
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

        # Optimize: avoid intermediate copy by chaining loc + rename
        # rename() creates a new DataFrame, so no need for explicit .copy()
        spy_aligned = spy_df.loc[common_timestamps].rename(columns=lambda x: f'spy_{x}')
        tsla_aligned = tsla_df.loc[common_timestamps].rename(columns=lambda x: f'tsla_{x}')

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

        print("  🔍 Validating merged data...")

        # Extract and validate SPY subset
        spy_cols = [c for c in aligned_df.columns if c.startswith('spy_')]
        spy_data = aligned_df[spy_cols].copy()
        spy_data.columns = [c.replace('spy_', '') for c in spy_data.columns]
        if not self.validate_data(spy_data):
            raise ValueError("SPY merged data validation failed - check terminal output above")

        # Extract and validate TSLA subset
        tsla_cols = [c for c in aligned_df.columns if c.startswith('tsla_')]
        tsla_data = aligned_df[tsla_cols].copy()
        tsla_data.columns = [c.replace('tsla_', '') for c in tsla_data.columns]
        if not self.validate_data(tsla_data):
            raise ValueError("TSLA merged data validation failed - check terminal output above")

        return aligned_df

    def load_vix_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load VIX data from CSV file.

        VIX is loaded as daily data and will be forward-filled to match 1-min data
        when merged with SPY/TSLA.

        Returns:
            DataFrame with columns: vix_open, vix_high, vix_low, vix_close
        """
        vix_path = config.VIX_DATA_FILE

        if not vix_path.exists():
            print(f"⚠️ VIX data file not found: {vix_path}")
            print("   VIX features will be zeros. Download from: https://www.cboe.com/tradable_products/vix/vix_historical_data/")
            return None

        # Load VIX CSV
        vix_df = pd.read_csv(vix_path)

        # Handle different column name formats
        date_col = None
        for col in ['Date', 'date', 'DATE', 'timestamp']:
            if col in vix_df.columns:
                date_col = col
                break

        if date_col is None:
            raise ValueError(f"VIX CSV must have a date column. Found: {vix_df.columns.tolist()}")

        vix_df[date_col] = pd.to_datetime(vix_df[date_col])
        vix_df.set_index(date_col, inplace=True)

        # Standardize column names (handle CBOE format: Open, High, Low, Close)
        col_mapping = {}
        for col in vix_df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                col_mapping[col] = 'vix_open'
            elif 'high' in col_lower:
                col_mapping[col] = 'vix_high'
            elif 'low' in col_lower:
                col_mapping[col] = 'vix_low'
            elif 'close' in col_lower:
                col_mapping[col] = 'vix_close'

        vix_df = vix_df.rename(columns=col_mapping)

        # Keep only OHLC columns
        ohlc_cols = ['vix_open', 'vix_high', 'vix_low', 'vix_close']
        available_cols = [c for c in ohlc_cols if c in vix_df.columns]

        if not available_cols:
            raise ValueError(f"VIX CSV must have OHLC columns. Found: {vix_df.columns.tolist()}")

        vix_df = vix_df[available_cols]

        # Convert to float and handle any formatting issues
        for col in vix_df.columns:
            vix_df[col] = pd.to_numeric(vix_df[col], errors='coerce')

        # Filter by date range
        if start_date:
            vix_df = vix_df[vix_df.index >= pd.to_datetime(start_date)]
        if end_date:
            vix_df = vix_df[vix_df.index <= pd.to_datetime(end_date)]

        # Sort by date
        vix_df = vix_df.sort_index()

        print(f"✅ Loaded VIX data: {len(vix_df)} daily bars ({vix_df.index.min()} to {vix_df.index.max()})")

        return vix_df

    def load_aligned_data_with_vix(self, start_date: str = None, end_date: str = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load SPY, TSLA, and VIX data.

        Returns:
            Tuple of (aligned_df, vix_df)
            - aligned_df: SPY/TSLA 1-min data aligned by timestamp
            - vix_df: VIX daily data (to be merged during feature extraction)
        """
        # Load and align SPY/TSLA
        aligned_df = self.load_aligned_data(start_date, end_date)

        # Load VIX
        vix_df = self.load_vix_data(start_date, end_date)

        return aligned_df, vix_df

    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str, symbol_prefix: str = '') -> pd.DataFrame:
        """
        Resample 1-min OHLCV data to a higher timeframe.

        Args:
            df: DataFrame with OHLCV columns (optionally prefixed)
            timeframe: Target timeframe ('5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month')
            symbol_prefix: Optional prefix for columns (e.g., 'spy_', 'tsla_')

        Returns:
            Resampled DataFrame with OHLCV at target timeframe
        """
        # Map timeframe to pandas resample rule
        tf_map = {
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1h': '1H',
            '2h': '2H',
            '3h': '3H',
            '4h': '4H',
            'daily': '1D',
            'weekly': '1W',
            'monthly': '1M',
            '3month': '3M'
        }

        if timeframe not in tf_map:
            raise ValueError(f"Unknown timeframe: {timeframe}. Valid: {list(tf_map.keys())}")

        rule = tf_map[timeframe]

        # Get column names
        open_col = f'{symbol_prefix}open' if symbol_prefix else 'open'
        high_col = f'{symbol_prefix}high' if symbol_prefix else 'high'
        low_col = f'{symbol_prefix}low' if symbol_prefix else 'low'
        close_col = f'{symbol_prefix}close' if symbol_prefix else 'close'
        volume_col = f'{symbol_prefix}volume' if symbol_prefix else 'volume'

        # Check which columns exist
        cols_to_resample = {}
        if open_col in df.columns:
            cols_to_resample[open_col] = 'first'
        if high_col in df.columns:
            cols_to_resample[high_col] = 'max'
        if low_col in df.columns:
            cols_to_resample[low_col] = 'min'
        if close_col in df.columns:
            cols_to_resample[close_col] = 'last'
        if volume_col in df.columns:
            cols_to_resample[volume_col] = 'sum'

        if not cols_to_resample:
            raise ValueError(f"No OHLCV columns found with prefix '{symbol_prefix}'")

        # Resample
        resampled = df[list(cols_to_resample.keys())].resample(rule).agg(cols_to_resample)

        # Drop any rows with NaN (incomplete bars)
        resampled = resampled.dropna()

        return resampled

    def get_native_timeframe_data(self, aligned_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Get native OHLCV data for a specific timeframe.

        Resamples both SPY and TSLA to the target timeframe.

        Args:
            aligned_df: Aligned SPY/TSLA 1-min data
            timeframe: Target timeframe

        Returns:
            DataFrame with resampled SPY and TSLA OHLCV
        """
        # Resample SPY
        spy_cols = [c for c in aligned_df.columns if c.startswith('spy_')]
        spy_resampled = self.resample_to_timeframe(aligned_df[spy_cols], timeframe, 'spy_')

        # Resample TSLA
        tsla_cols = [c for c in aligned_df.columns if c.startswith('tsla_')]
        tsla_resampled = self.resample_to_timeframe(aligned_df[tsla_cols], timeframe, 'tsla_')

        # Merge on index
        resampled_df = pd.concat([spy_resampled, tsla_resampled], axis=1)

        return resampled_df

    def get_all_timeframe_data(self, aligned_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Get native OHLCV data for all configured timeframes.

        Args:
            aligned_df: Aligned SPY/TSLA 1-min data

        Returns:
            Dict mapping timeframe name to resampled DataFrame
        """
        timeframe_data = {}

        for tf in config.MODEL_TIMEFRAMES:
            print(f"   Resampling to {tf}...", end=' ')
            tf_df = self.get_native_timeframe_data(aligned_df, tf)
            timeframe_data[tf] = tf_df
            print(f"✓ {len(tf_df)} bars")

        return timeframe_data


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
