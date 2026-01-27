"""
Live Data Fetcher for v7

Fetches real-time market data from yfinance with support for:
- Multiple symbols (TSLA, SPY, etc.)
- All 8 native intervals (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
- Robust error handling with empty DataFrame returns
- Column normalization (lowercase)
- Timezone removal to match CSV format
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import warnings


class LiveDataFetcher:
    """
    Fetch live stock data from yfinance.

    Supports multiple symbols and all yfinance native intervals.
    Handles errors gracefully by returning empty DataFrames.
    Normalizes output to match CSV format (lowercase columns, no timezone).

    Attributes:
        symbol: Stock symbol (e.g., 'TSLA', 'SPY')
        ticker: yfinance Ticker object
    """

    # yfinance native intervals and their data availability limits
    NATIVE_INTERVALS = {
        '1m': {'max_days': 7, 'period_str': '7d'},      # 1-minute: last 7 days
        '5m': {'max_days': 60, 'period_str': '60d'},    # 5-minute: last 60 days
        '15m': {'max_days': 60, 'period_str': '60d'},   # 15-minute: last 60 days
        '30m': {'max_days': 60, 'period_str': '60d'},   # 30-minute: last 60 days
        '1h': {'max_days': 730, 'period_str': '730d'},  # 1-hour: last 730 days (2 years)
        '1d': {'max_days': None, 'period_str': 'max'},  # Daily: all available
        '1wk': {'max_days': None, 'period_str': 'max'}, # Weekly: all available
        '1mo': {'max_days': None, 'period_str': 'max'}, # Monthly: all available
    }

    def __init__(self, symbol: str = "TSLA"):
        """
        Initialize live data fetcher.

        Args:
            symbol: Stock symbol (e.g., 'TSLA', 'SPY', 'VIX')
        """
        self.symbol = symbol.upper()
        self.ticker = yf.Ticker(self.symbol)

    def fetch(
        self,
        interval: str = '1m',
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        days_back: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch data from yfinance with comprehensive error handling.

        Args:
            interval: One of ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo']
            period: Period string (e.g., '7d', '1mo', '1y', 'max') - overrides days_back
            start: Start date string (YYYY-MM-DD) - use with end
            end: End date string (YYYY-MM-DD) - use with start
            days_back: Number of days to fetch (respects interval limits)

        Returns:
            DataFrame with columns [open, high, low, close, volume] and DatetimeIndex
            Returns empty DataFrame on error

        Examples:
            # Fetch last 7 days of 1-minute data
            df = fetcher.fetch(interval='1m', days_back=7)

            # Fetch all daily data
            df = fetcher.fetch(interval='1d', period='max')

            # Fetch specific date range
            df = fetcher.fetch(interval='1h', start='2024-01-01', end='2024-12-31')
        """
        # Validate interval
        if interval not in self.NATIVE_INTERVALS:
            print(f"Error: Invalid interval '{interval}'. Valid: {list(self.NATIVE_INTERVALS.keys())}")
            return pd.DataFrame()

        try:
            # Determine period/start/end
            if period is None and start is None and end is None:
                # Use days_back or default to max for interval
                if days_back is not None:
                    max_days = self.NATIVE_INTERVALS[interval]['max_days']
                    if max_days is not None:
                        days_back = min(days_back, max_days)
                    period = f"{days_back}d"
                else:
                    period = self.NATIVE_INTERVALS[interval]['period_str']

            # Fetch data
            if start and end:
                # Use date range
                df = self.ticker.history(
                    interval=interval,
                    start=start,
                    end=end
                )
            else:
                # Use period
                df = self.ticker.history(
                    interval=interval,
                    period=period
                )

            # Handle empty response
            if df.empty:
                warnings.warn(f"No data received for {self.symbol} (interval={interval})")
                return pd.DataFrame()

            # Normalize columns and format
            df = self._normalize_dataframe(df)

            return df

        except Exception as e:
            print(f"Error fetching {self.symbol} data (interval={interval}): {e}")
            return pd.DataFrame()

    def fetch_multiple(
        self,
        intervals: List[str],
        period: Optional[str] = None,
        days_back: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple intervals at once.

        Args:
            intervals: List of interval strings
            period: Period string (e.g., '7d', '1mo', 'max')
            days_back: Number of days to fetch

        Returns:
            Dictionary mapping interval -> DataFrame

        Example:
            dfs = fetcher.fetch_multiple(['1m', '5m', '15m'], days_back=7)
            df_1m = dfs['1m']
            df_5m = dfs['5m']
        """
        results = {}
        for interval in intervals:
            df = self.fetch(
                interval=interval,
                period=period,
                days_back=days_back
            )
            results[interval] = df
        return results

    def get_current_price(self) -> Optional[float]:
        """
        Get the current stock price.

        Returns:
            Current price or None if unavailable
        """
        try:
            df = self.ticker.history(period="1d", interval="1m")
            if not df.empty:
                return float(df['Close'].iloc[-1])
        except Exception as e:
            print(f"Error fetching current price for {self.symbol}: {e}")

        return None

    def get_info(self) -> Dict:
        """
        Get ticker info from yfinance.

        Returns:
            Dictionary with ticker information, or empty dict on error
        """
        try:
            return self.ticker.info
        except Exception as e:
            print(f"Error fetching info for {self.symbol}: {e}")
            return {}

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame to match CSV format.

        - Lowercase column names
        - Remove timezone from index
        - Keep only OHLCV columns
        - Set index name to 'timestamp'

        Args:
            df: Raw DataFrame from yfinance

        Returns:
            Normalized DataFrame
        """
        # Rename columns to lowercase
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Keep only OHLCV columns (drop Dividends, Stock Splits, etc.)
        valid_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        df = df[valid_cols]

        # Remove timezone info to match CSV format
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Set index name
        df.index.name = 'timestamp'

        return df

    def validate_data(self, df: pd.DataFrame, min_rows: int = 1) -> bool:
        """
        Validate that DataFrame meets minimum requirements.

        Args:
            df: DataFrame to validate
            min_rows: Minimum number of rows required

        Returns:
            True if valid, False otherwise
        """
        if df.empty:
            return False

        if len(df) < min_rows:
            return False

        # Check for required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return False

        # Check for excessive NaNs
        if df.isna().sum().sum() > len(df) * 0.1:  # More than 10% NaNs
            return False

        return True


class MultiSymbolFetcher:
    """
    Fetch data for multiple symbols simultaneously.

    Convenience class for fetching TSLA, SPY, VIX, etc. in one call.
    """

    def __init__(self, symbols: List[str] = None):
        """
        Initialize multi-symbol fetcher.

        Args:
            symbols: List of symbols (default: ['TSLA', 'SPY'])
        """
        if symbols is None:
            symbols = ['TSLA', 'SPY']

        self.symbols = [s.upper() for s in symbols]
        self.fetchers = {
            symbol: LiveDataFetcher(symbol)
            for symbol in self.symbols
        }

    def fetch_all(
        self,
        interval: str = '1m',
        period: Optional[str] = None,
        days_back: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all symbols.

        Args:
            interval: Interval string
            period: Period string
            days_back: Number of days to fetch

        Returns:
            Dictionary mapping symbol -> DataFrame

        Example:
            fetcher = MultiSymbolFetcher(['TSLA', 'SPY'])
            data = fetcher.fetch_all(interval='1m', days_back=7)
            tsla_df = data['TSLA']
            spy_df = data['SPY']
        """
        results = {}
        for symbol, fetcher in self.fetchers.items():
            df = fetcher.fetch(
                interval=interval,
                period=period,
                days_back=days_back
            )
            results[symbol] = df
        return results

    def get_current_prices(self) -> Dict[str, Optional[float]]:
        """
        Get current prices for all symbols.

        Returns:
            Dictionary mapping symbol -> price
        """
        return {
            symbol: fetcher.get_current_price()
            for symbol, fetcher in self.fetchers.items()
        }


def merge_with_historical(
    historical_df: pd.DataFrame,
    live_df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Merge historical CSV data with recent live data.

    Utility function to combine historical data with fresh live data.
    Handles overlaps and duplicates intelligently.

    Args:
        historical_df: Historical data from CSV
        live_df: Fresh data from yfinance
        verbose: Print merge information

    Returns:
        Combined DataFrame with historical + new live data
    """
    if live_df.empty:
        if verbose:
            print("No live data to merge")
        return historical_df

    if historical_df.empty:
        if verbose:
            print("No historical data, returning live data only")
        return live_df

    # Find where historical data ends
    historical_end = historical_df.index.max()
    live_start = live_df.index.min()
    live_end = live_df.index.max()

    if verbose:
        print(f"Merging data:")
        print(f"  Historical: {historical_df.index.min()} to {historical_end}")
        print(f"  Live: {live_start} to {live_end}")

    # Filter live data to only include new data after historical
    new_data = live_df[live_df.index > historical_end]

    if new_data.empty:
        if verbose:
            print("  No new data to add (historical is up-to-date)")
        return historical_df

    # Combine historical + new live data
    combined = pd.concat([historical_df, new_data])
    combined.sort_index(inplace=True)

    # Remove duplicates (keep last)
    combined = combined[~combined.index.duplicated(keep='last')]

    if verbose:
        print(f"  Added {len(new_data)} new rows")
        print(f"  Combined total: {len(combined)} rows")
        print(f"  Latest timestamp: {combined.index.max()}")

    return combined


def get_data_freshness(latest_timestamp: datetime) -> Dict:
    """
    Check how fresh the data is.

    Args:
        latest_timestamp: Most recent timestamp in data

    Returns:
        Dictionary with freshness information
    """
    now = datetime.now()
    age = now - latest_timestamp

    # Determine freshness status
    if age < timedelta(minutes=5):
        status = "live"
        color = "green"
    elif age < timedelta(hours=1):
        status = "recent"
        color = "yellow"
    elif age < timedelta(days=1):
        status = "stale"
        color = "orange"
    else:
        status = "outdated"
        color = "red"

    return {
        "latest_timestamp": latest_timestamp,
        "age_seconds": age.total_seconds(),
        "age_minutes": age.total_seconds() / 60,
        "age_hours": age.total_seconds() / 3600,
        "age_days": age.days,
        "status": status,
        "color": color,
        "is_live": status == "live",
        "message": f"Data is {age.days}d {age.seconds//3600}h {(age.seconds//60)%60}m old"
    }


if __name__ == "__main__":
    """
    Test the LiveDataFetcher with various scenarios.
    """
    print("=" * 70)
    print("LiveDataFetcher Test Suite")
    print("=" * 70)

    # Test 1: Single symbol, single interval
    print("\n1. Fetching TSLA 1-minute data (last 2 days)")
    print("-" * 70)
    fetcher = LiveDataFetcher("TSLA")
    df_1m = fetcher.fetch(interval='1m', days_back=2)

    if not df_1m.empty:
        print(f"Success! Fetched {len(df_1m)} rows")
        print(f"Date range: {df_1m.index.min()} to {df_1m.index.max()}")
        print(f"Columns: {list(df_1m.columns)}")
        print(f"\nFirst 3 rows:")
        print(df_1m.head(3))
        print(f"\nLast 3 rows:")
        print(df_1m.tail(3))
    else:
        print("Failed to fetch data")

    # Test 2: Multiple intervals
    print("\n\n2. Fetching TSLA multiple intervals")
    print("-" * 70)
    intervals = ['1m', '5m', '15m', '1h']
    multi_data = fetcher.fetch_multiple(intervals, days_back=2)

    for interval, df in multi_data.items():
        status = f"{len(df)} rows" if not df.empty else "EMPTY"
        print(f"  {interval:6s}: {status}")

    # Test 3: Multiple symbols
    print("\n\n3. Fetching multiple symbols (TSLA, SPY)")
    print("-" * 70)
    multi_fetcher = MultiSymbolFetcher(['TSLA', 'SPY'])
    symbol_data = multi_fetcher.fetch_all(interval='1h', days_back=7)

    for symbol, df in symbol_data.items():
        status = f"{len(df)} rows" if not df.empty else "EMPTY"
        print(f"  {symbol:6s}: {status}")

    # Test 4: Current prices
    print("\n\n4. Current prices")
    print("-" * 70)
    prices = multi_fetcher.get_current_prices()
    for symbol, price in prices.items():
        if price:
            print(f"  {symbol:6s}: ${price:.2f}")
        else:
            print(f"  {symbol:6s}: N/A")

    # Test 5: Data freshness
    print("\n\n5. Data freshness check")
    print("-" * 70)
    if not df_1m.empty:
        freshness = get_data_freshness(df_1m.index.max())
        print(f"  Status: {freshness['status']}")
        print(f"  Message: {freshness['message']}")
        print(f"  Is Live: {freshness['is_live']}")

    # Test 6: All 8 native intervals (quick check)
    print("\n\n6. Testing all 8 native intervals")
    print("-" * 70)
    all_intervals = ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo']
    for interval in all_intervals:
        df = fetcher.fetch(interval=interval, period='5d')
        status = f"{len(df)} rows" if not df.empty else "EMPTY"
        print(f"  {interval:6s}: {status}")

    # Test 7: Error handling (invalid interval)
    print("\n\n7. Error handling test (invalid interval)")
    print("-" * 70)
    df_invalid = fetcher.fetch(interval='invalid')
    print(f"  Result: {'EMPTY (as expected)' if df_invalid.empty else 'UNEXPECTED DATA'}")

    # Test 8: Data validation
    print("\n\n8. Data validation test")
    print("-" * 70)
    if not df_1m.empty:
        is_valid = fetcher.validate_data(df_1m, min_rows=10)
        print(f"  Data valid (min 10 rows): {is_valid}")
        print(f"  Has NaNs: {df_1m.isna().sum().sum() > 0}")
        print(f"  Timezone: {df_1m.index.tz}")

    print("\n" + "=" * 70)
    print("Test suite completed!")
    print("=" * 70)
