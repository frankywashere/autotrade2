"""Live data fetcher using yfinance for real-time updates."""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import sys
import os

# Add parent directory to path for config
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


class LiveDataFetcher:
    """Fetch live stock data from yfinance."""

    def __init__(self, stock: str = config.DEFAULT_STOCK):
        """
        Initialize live data fetcher.

        Args:
            stock: Stock symbol (e.g., 'TSLA', 'SPY')
        """
        self.stock = stock
        self.ticker = yf.Ticker(stock)

    def fetch_recent_data(self, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch recent 1-minute data from yfinance.

        Args:
            days_back: Number of days to fetch (max 7 for 1-min data)

        Returns:
            DataFrame with 1-minute OHLCV data
        """
        print(f"Fetching live data for {self.stock} (last {days_back} days)...")

        try:
            # yfinance limits: 1m data available for last 7 days only
            days_back = min(days_back, 7)

            # Fetch 1-minute data
            df = self.ticker.history(
                period=f"{days_back}d",
                interval="1m"
            )

            if df.empty:
                print(f"Warning: No live data received for {self.stock}")
                return pd.DataFrame()

            # Rename columns to match our format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Remove timezone info to match CSV format
            df.index = df.index.tz_localize(None)

            # Rename index to timestamp
            df.index.name = 'timestamp'

            print(f"Fetched {len(df)} rows of live data")
            print(f"Date range: {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            print(f"Error fetching live data: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> Optional[float]:
        """
        Get current stock price.

        Returns:
            Current price or None if unavailable
        """
        try:
            data = self.ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            print(f"Error fetching current price: {e}")

        return None

    def merge_with_historical(self, historical_df: pd.DataFrame,
                             days_back: int = 7) -> pd.DataFrame:
        """
        Merge historical CSV data with recent live data.

        Args:
            historical_df: Historical data from CSV
            days_back: Number of days of live data to fetch

        Returns:
            Combined DataFrame with historical + live data
        """
        # Fetch live data
        live_df = self.fetch_recent_data(days_back)

        if live_df.empty:
            print("No live data available, using historical only")
            return historical_df

        # Find where historical data ends
        historical_end = historical_df.index.max()
        live_start = live_df.index.min()

        print(f"\nMerging data:")
        print(f"  Historical ends: {historical_end}")
        print(f"  Live starts: {live_start}")

        # Filter live data to only include new data after historical
        new_data = live_df[live_df.index > historical_end]

        if new_data.empty:
            print("  No new data to add (historical is up-to-date)")
            return historical_df

        # Combine historical + new live data
        combined = pd.concat([historical_df, new_data])
        combined.sort_index(inplace=True)

        # Remove duplicates (keep last)
        combined = combined[~combined.index.duplicated(keep='last')]

        print(f"  Added {len(new_data)} new rows")
        print(f"  Combined total: {len(combined)} rows")
        print(f"  Latest timestamp: {combined.index.max()}")

        return combined

    def get_data_freshness(self, latest_timestamp: datetime) -> dict:
        """
        Check how fresh the data is.

        Args:
            latest_timestamp: Most recent timestamp in data

        Returns:
            Dictionary with freshness information
        """
        now = datetime.now()
        age = now - latest_timestamp

        # Determine freshness
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
    # Test live data fetcher
    fetcher = LiveDataFetcher("TSLA")

    # Get current price
    print("=" * 60)
    current = fetcher.get_current_price()
    if current:
        print(f"Current TSLA price: ${current:.2f}")

    # Fetch recent data
    print("\n" + "=" * 60)
    live_data = fetcher.fetch_recent_data(days_back=2)
    print(f"\nLive data sample (last 5 rows):")
    print(live_data.tail())

    # Test data freshness
    print("\n" + "=" * 60)
    if not live_data.empty:
        freshness = fetcher.get_data_freshness(live_data.index.max())
        print(f"\nData Freshness:")
        print(f"  Status: {freshness['status']}")
        print(f"  {freshness['message']}")
        print(f"  Is Live: {freshness['is_live']}")
