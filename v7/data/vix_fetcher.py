"""
VIX Data Fetcher with Multiple Fallbacks

Fetches VIX data with the following priority:
1. FRED API (Federal Reserve Economic Data) - Primary source
2. yfinance (Yahoo Finance) - Secondary fallback
3. Local CSV file - Final fallback

Features:
- Automatic fallback chain on API failures
- Daily data forward-fill logic for missing dates
- Comprehensive error handling
- Data validation and normalization
- Caching support
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VIXDataSource:
    """Information about the data source used."""
    source: str  # 'fred', 'yfinance', or 'csv'
    date_range: Tuple[datetime, datetime]
    num_records: int
    has_gaps: bool


class FREDVixFetcher:
    """
    VIX data fetcher with FRED API primary source and fallbacks.

    This class implements a robust VIX data fetching strategy:
    1. Try FRED API first (most reliable, official source)
    2. Fall back to yfinance if FRED fails
    3. Fall back to local CSV as last resort
    4. Forward-fill missing dates to ensure complete daily coverage

    Usage:
        fetcher = FREDVixFetcher(fred_api_key="your_api_key")
        vix_df = fetcher.fetch(start_date="2020-01-01", end_date="2023-12-31")
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        csv_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the VIX fetcher.

        Args:
            fred_api_key: FRED API key (get free key from https://fred.stlouisfed.org/docs/api/api_key.html)
            csv_path: Path to local VIX_History.csv file
            cache_dir: Directory to cache fetched data (optional)
        """
        self.fred_api_key = fred_api_key
        self.csv_path = csv_path or self._find_default_csv()
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track which source was used
        self.last_source: Optional[VIXDataSource] = None

    def _find_default_csv(self) -> Optional[str]:
        """Find the default VIX_History.csv in the project."""
        # Try common locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "data" / "VIX_History.csv",
            Path.cwd() / "data" / "VIX_History.csv",
            Path.cwd() / "VIX_History.csv",
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"Found VIX CSV at: {path}")
                return str(path)

        logger.warning("No default VIX_History.csv found")
        return None

    def fetch(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        forward_fill: bool = True
    ) -> pd.DataFrame:
        """
        Fetch VIX data with automatic fallback chain.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format (default: 1990-01-01)
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            forward_fill: Whether to forward-fill missing dates (default: True)

        Returns:
            DataFrame with columns: ['open', 'high', 'low', 'close']
            Index: DatetimeIndex

        Raises:
            RuntimeError: If all data sources fail
        """
        # Set default dates
        if start_date is None:
            start_date = "1990-01-01"
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Try each source in order
        df = None
        errors = []

        # 1. Try FRED API
        if self.fred_api_key:
            try:
                logger.info("Attempting to fetch VIX from FRED API...")
                df = self._fetch_from_fred(start_date, end_date)
                if df is not None and len(df) > 0:
                    logger.info(f"Successfully fetched {len(df)} records from FRED")
                    self._set_source_info(df, 'fred')
                    return self._process_data(df, forward_fill)
            except Exception as e:
                error_msg = f"FRED API failed: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
        else:
            logger.info("No FRED API key provided, skipping FRED source")

        # 2. Try yfinance
        try:
            logger.info("Attempting to fetch VIX from yfinance...")
            df = self._fetch_from_yfinance(start_date, end_date)
            if df is not None and len(df) > 0:
                logger.info(f"Successfully fetched {len(df)} records from yfinance")
                self._set_source_info(df, 'yfinance')
                return self._process_data(df, forward_fill)
        except Exception as e:
            error_msg = f"yfinance failed: {str(e)}"
            logger.warning(error_msg)
            errors.append(error_msg)

        # 3. Try local CSV
        if self.csv_path and Path(self.csv_path).exists():
            try:
                logger.info(f"Attempting to load VIX from local CSV: {self.csv_path}")
                df = self._load_from_csv(start_date, end_date)
                if df is not None and len(df) > 0:
                    logger.info(f"Successfully loaded {len(df)} records from CSV")
                    self._set_source_info(df, 'csv')
                    return self._process_data(df, forward_fill)
            except Exception as e:
                error_msg = f"CSV loading failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        else:
            logger.warning("No CSV path configured or file doesn't exist")

        # All sources failed
        error_summary = "\n".join(errors)
        raise RuntimeError(
            f"All VIX data sources failed:\n{error_summary}\n"
            f"Please provide a FRED API key, ensure yfinance is working, "
            f"or provide a valid CSV file path."
        )

    def _fetch_from_fred(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch VIX from FRED API.

        FRED Series: VIXCLS (CBOE Volatility Index: VIX)
        """
        try:
            from fredapi import Fred
        except ImportError:
            raise ImportError(
                "fredapi package not installed. Install with: pip install fredapi"
            )

        fred = Fred(api_key=self.fred_api_key)

        # Fetch VIX close prices (FRED only has close, not OHLC)
        vix_series = fred.get_series(
            'VIXCLS',
            observation_start=start_date,
            observation_end=end_date
        )

        if vix_series is None or len(vix_series) == 0:
            return None

        # Convert to DataFrame
        df = pd.DataFrame({'close': vix_series})
        df.index.name = 'date'

        # FRED only provides close prices, so we'll use close for all OHLC
        df['open'] = df['close']
        df['high'] = df['close']
        df['low'] = df['close']

        return df[['open', 'high', 'low', 'close']]

    def _fetch_from_yfinance(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch VIX from Yahoo Finance using yfinance.

        Ticker: ^VIX
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance package not installed. Install with: pip install yfinance"
            )

        # Download VIX data
        vix = yf.Ticker("^VIX")
        df = vix.history(start=start_date, end=end_date, auto_adjust=False)

        if df is None or len(df) == 0:
            return None

        # Normalize column names to lowercase
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close'
        })

        # Keep only OHLC columns
        df = df[['open', 'high', 'low', 'close']]

        return df

    def _load_from_csv(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Load VIX data from local CSV file.

        Expected CSV format:
        DATE,OPEN,HIGH,LOW,CLOSE
        01/02/1990,17.240000,17.240000,17.240000,17.240000
        """
        df = pd.read_csv(self.csv_path)

        # Normalize column names
        df.columns = df.columns.str.upper()

        # Parse date column
        if 'DATE' in df.columns:
            df['date'] = pd.to_datetime(df['DATE'])
            df = df.drop('DATE', axis=1)
        else:
            raise ValueError("CSV must have a 'DATE' column")

        # Set date as index
        df = df.set_index('date')

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()

        # Filter by date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df = df[(df.index >= start) & (df.index <= end)]

        # Keep only OHLC columns
        return df[['open', 'high', 'low', 'close']]

    def _process_data(self, df: pd.DataFrame, forward_fill: bool) -> pd.DataFrame:
        """
        Process and clean VIX data.

        Args:
            df: Raw VIX DataFrame
            forward_fill: Whether to forward-fill missing dates

        Returns:
            Processed DataFrame
        """
        # Remove any NaN values
        df = df.dropna()

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Remove timezone info for consistency with rest of system
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Sort by date
        df = df.sort_index()

        # Forward-fill missing dates if requested
        if forward_fill:
            df = self._forward_fill_missing_dates(df)

        # Validate data
        self._validate_data(df)

        return df

    def _forward_fill_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill missing dates in VIX data.

        This ensures we have a complete daily time series, even for
        weekends, holidays, and other non-trading days.

        Args:
            df: VIX DataFrame with potential gaps

        Returns:
            DataFrame with all dates filled
        """
        if len(df) == 0:
            return df

        # Create complete date range
        start_date = df.index.min()
        end_date = df.index.max()
        complete_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Reindex with complete dates and forward-fill
        df_filled = df.reindex(complete_dates, method='ffill')

        # Count how many dates were filled
        original_count = len(df)
        filled_count = len(df_filled)
        if filled_count > original_count:
            logger.info(
                f"Forward-filled {filled_count - original_count} missing dates "
                f"({original_count} -> {filled_count} records)"
            )

        return df_filled

    def _validate_data(self, df: pd.DataFrame):
        """
        Validate VIX data quality.

        Raises:
            ValueError: If data is invalid
        """
        if len(df) == 0:
            raise ValueError("VIX data is empty")

        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for negative values (VIX should always be positive)
        if (df < 0).any().any():
            raise ValueError("VIX data contains negative values")

        # Check for unrealistic values (VIX rarely exceeds 100)
        if (df > 200).any().any():
            logger.warning("VIX data contains unusually high values (>200)")

        # Check high >= low
        if (df['high'] < df['low']).any():
            raise ValueError("VIX data has high < low")

        # Check close is within [low, high]
        if ((df['close'] < df['low']) | (df['close'] > df['high'])).any():
            raise ValueError("VIX close price outside [low, high] range")

    def _set_source_info(self, df: pd.DataFrame, source: str):
        """Record information about the data source used."""
        has_gaps = False
        if len(df) > 1:
            date_diffs = df.index.to_series().diff()
            has_gaps = (date_diffs > pd.Timedelta(days=1)).any()

        self.last_source = VIXDataSource(
            source=source,
            date_range=(df.index.min().to_pydatetime(), df.index.max().to_pydatetime()),
            num_records=len(df),
            has_gaps=has_gaps
        )

    def get_source_info(self) -> Optional[VIXDataSource]:
        """Get information about the last successful data source."""
        return self.last_source


def fetch_vix_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fred_api_key: Optional[str] = None,
    csv_path: Optional[str] = None,
    forward_fill: bool = True
) -> pd.DataFrame:
    """
    Convenience function to fetch VIX data.

    This is a simple wrapper around FREDVixFetcher for quick usage.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        fred_api_key: FRED API key (optional)
        csv_path: Path to local CSV file (optional)
        forward_fill: Whether to forward-fill missing dates

    Returns:
        DataFrame with VIX data

    Example:
        # Using FRED API
        vix_df = fetch_vix_data(
            start_date="2020-01-01",
            fred_api_key="your_api_key"
        )

        # Using local CSV fallback
        vix_df = fetch_vix_data(
            start_date="2020-01-01",
            csv_path="data/VIX_History.csv"
        )
    """
    fetcher = FREDVixFetcher(
        fred_api_key=fred_api_key,
        csv_path=csv_path
    )

    df = fetcher.fetch(
        start_date=start_date,
        end_date=end_date,
        forward_fill=forward_fill
    )

    # Log source info
    if fetcher.last_source:
        source = fetcher.last_source
        logger.info(
            f"VIX data fetched from {source.source}: "
            f"{source.num_records} records from {source.date_range[0].date()} "
            f"to {source.date_range[1].date()}"
        )

    return df


# Example usage and testing
if __name__ == "__main__":
    import os

    # Example 1: Using FRED API (requires API key)
    print("=" * 80)
    print("Example 1: Fetching VIX from FRED API")
    print("=" * 80)

    # Get API key from environment or use None to skip FRED
    fred_key = os.getenv('FRED_API_KEY')

    if fred_key:
        try:
            vix_df = fetch_vix_data(
                start_date="2023-01-01",
                end_date="2023-12-31",
                fred_api_key=fred_key,
                forward_fill=True
            )
            print(f"\nFetched {len(vix_df)} records")
            print("\nFirst 5 rows:")
            print(vix_df.head())
            print("\nLast 5 rows:")
            print(vix_df.tail())
            print("\nData summary:")
            print(vix_df.describe())
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No FRED_API_KEY found in environment, skipping FRED example")

    # Example 2: Using yfinance fallback
    print("\n" + "=" * 80)
    print("Example 2: Fetching VIX from yfinance (fallback)")
    print("=" * 80)

    try:
        vix_df = fetch_vix_data(
            start_date="2023-01-01",
            end_date="2023-12-31",
            forward_fill=True
        )
        print(f"\nFetched {len(vix_df)} records")
        print("\nFirst 5 rows:")
        print(vix_df.head())
        print("\nData info:")
        print(f"Date range: {vix_df.index.min().date()} to {vix_df.index.max().date()}")
        print(f"Missing dates: {pd.date_range(vix_df.index.min(), vix_df.index.max()).difference(vix_df.index).size}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Using local CSV
    print("\n" + "=" * 80)
    print("Example 3: Loading VIX from local CSV")
    print("=" * 80)

    # Try to find the CSV
    csv_path = Path(__file__).parent.parent.parent / "data" / "VIX_History.csv"
    if csv_path.exists():
        try:
            vix_df = fetch_vix_data(
                start_date="2023-01-01",
                end_date="2023-12-31",
                csv_path=str(csv_path),
                forward_fill=True
            )
            print(f"\nLoaded {len(vix_df)} records from CSV")
            print("\nFirst 5 rows:")
            print(vix_df.head())
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"CSV not found at: {csv_path}")

    # Example 4: Show source priority
    print("\n" + "=" * 80)
    print("Example 4: Source Priority Demonstration")
    print("=" * 80)

    fetcher = FREDVixFetcher(
        fred_api_key=fred_key,  # Will be None if not in env
        csv_path=str(csv_path) if csv_path.exists() else None
    )

    try:
        vix_df = fetcher.fetch(start_date="2023-01-01", end_date="2023-01-31")
        source_info = fetcher.get_source_info()

        if source_info:
            print(f"\nData source used: {source_info.source.upper()}")
            print(f"Records: {source_info.num_records}")
            print(f"Date range: {source_info.date_range[0].date()} to {source_info.date_range[1].date()}")
            print(f"Has gaps: {source_info.has_gaps}")

        print("\nSample data:")
        print(vix_df.head(10))
    except Exception as e:
        print(f"Error: {e}")
