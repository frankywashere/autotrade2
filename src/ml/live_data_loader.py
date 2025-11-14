"""
Live Data Loader for ML Predictions

Fetches live data from yfinance and merges with historical CSV data
to create seamless dataset for feature extraction.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config


class LiveDataLoader:
    """
    Loads and merges historical CSV data with live yfinance data.

    Ensures seamless transition between historical and live data for
    feature extraction without gaps or duplicates.
    """

    def __init__(self, timeframe: str = '1min', data_dir: str = 'data'):
        """
        Args:
            timeframe: Data resolution ('1min', '15min', '1hour', etc.)
            data_dir: Directory containing historical CSV files
        """
        self.timeframe = timeframe
        self.data_dir = Path(data_dir)

        # CSV file paths
        self.tsla_csv = self.data_dir / f'TSLA_{timeframe}.csv'
        self.spy_csv = self.data_dir / f'SPY_{timeframe}.csv'

    def load_live_data(self,
                       lookback_days: int = 5,
                       end_date: Optional[datetime] = None) -> Tuple[pd.DataFrame, str]:
        """
        Load historical + live data merged seamlessly.

        Args:
            lookback_days: How many days to load (for context)
            end_date: End date (default: now)

        Returns:
            (aligned_df, data_status)

        data_status: 'LIVE' if fresh, 'STALE' if old, 'HISTORICAL' if no live data
        """
        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=lookback_days)

        print(f"\n📊 Loading {self.timeframe} data...")
        print(f"   Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Step 1: Load historical CSV data
        try:
            tsla_hist = self._load_csv(self.tsla_csv, start_date, end_date, 'tsla')
            spy_hist = self._load_csv(self.spy_csv, start_date, end_date, 'spy')

            print(f"   ✓ Historical: TSLA={len(tsla_hist)} bars, SPY={len(spy_hist)} bars")
        except Exception as e:
            print(f"   ✗ Error loading historical data: {e}")
            raise

        # Step 2: Try to fetch live data (last 7 days from yfinance)
        data_status = 'HISTORICAL'

        try:
            tsla_live, spy_live = self._fetch_yfinance_data()

            if tsla_live is not None and spy_live is not None:
                # Merge live data with historical
                tsla_merged = self._merge_historical_live(tsla_hist, tsla_live)
                spy_merged = self._merge_historical_live(spy_hist, spy_live)

                # Check how fresh the data is
                latest_timestamp = tsla_merged.index[-1]
                data_age_minutes = (datetime.now() - latest_timestamp).total_seconds() / 60

                if data_age_minutes < 15:
                    data_status = 'LIVE'
                elif data_age_minutes < 60:
                    data_status = 'RECENT'
                else:
                    data_status = 'STALE'

                print(f"   ✓ Live data: {len(tsla_live)} new TSLA bars, {len(spy_live)} new SPY bars")
                print(f"   ✓ Data age: {data_age_minutes:.1f} minutes ({data_status})")

                tsla_hist = tsla_merged
                spy_hist = spy_merged
            else:
                print(f"   ⚠ No live data fetched, using historical only")

        except Exception as e:
            print(f"   ⚠ Could not fetch live data: {e}")
            print(f"   ℹ Using historical data only")

        # Step 3: Align TSLA and SPY (inner join)
        aligned_df = self._align_data(tsla_hist, spy_hist)

        print(f"   ✓ Aligned: {len(aligned_df)} bars (TSLA+SPY)")
        print(f"   ✓ Date range: {aligned_df.index[0]} to {aligned_df.index[-1]}")

        return aligned_df, data_status

    def _load_csv(self, csv_path: Path, start_date: datetime, end_date: datetime, symbol: str) -> pd.DataFrame:
        """Load historical CSV and filter by date range."""
        df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)

        # Filter by date range
        mask = (df.index >= start_date) & (df.index <= end_date)
        df = df[mask]

        # Add symbol prefix to columns to match yfinance format
        df = df.rename(columns={
            'open': f'{symbol}_open',
            'high': f'{symbol}_high',
            'low': f'{symbol}_low',
            'close': f'{symbol}_close',
            'volume': f'{symbol}_volume'
        })

        return df

    def _fetch_yfinance_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Fetch data from yfinance (optimized by timeframe).

        yfinance limits:
        - 1min: 7 days
        - 15min: 60 days
        - 1hour: 730 days (2 years!)
        - daily: unlimited

        Returns:
            (tsla_df, spy_df) or (None, None) if failed
        """
        try:
            # Determine optimal period and interval based on our timeframe
            if self.timeframe == '1min':
                period = '7d'
                interval = '1m'
            elif self.timeframe in ['5min', '15min', '30min']:
                period = '60d'  # yfinance allows 60 days for these intervals
                interval = '15m'  # Fetch at 15min, resample if needed
            elif self.timeframe in ['1hour', '2hour', '3hour', '4hour']:
                period = '730d'  # 2 years available for hourly data!
                interval = '1h'  # Fetch at 1hour, resample if needed
            else:  # daily, weekly, etc
                period = 'max'
                interval = '1d'

            # Fetch TSLA
            tsla = yf.Ticker('TSLA')
            tsla_df = tsla.history(period=period, interval=interval)

            if len(tsla_df) == 0:
                return None, None

            # Fetch SPY
            spy = yf.Ticker('SPY')
            spy_df = spy.history(period=period, interval=interval)

            if len(spy_df) == 0:
                return None, None

            # Rename columns to match CSV format
            tsla_df = self._format_yfinance_df(tsla_df, 'tsla')
            spy_df = self._format_yfinance_df(spy_df, 'spy')

            # Resample if needed (e.g., if we fetched 1h but need 4h)
            if interval != self.timeframe:
                # Only resample if target timeframe is different from fetched interval
                needs_resample = (
                    (self.timeframe == '5min' and interval == '15m') or
                    (self.timeframe == '30min' and interval == '15m') or
                    (self.timeframe in ['2hour', '3hour', '4hour'] and interval == '1h')
                )

                if needs_resample:
                    tsla_df = self._resample_to_timeframe(tsla_df)
                    spy_df = self._resample_to_timeframe(spy_df)

            return tsla_df, spy_df

        except Exception as e:
            print(f"      Error fetching from yfinance: {e}")
            return None, None

    def _format_yfinance_df(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Format yfinance DataFrame to match CSV format."""
        df = df.copy()

        # Remove timezone to match CSV format (CSV is tz-naive)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={
            'open': f'{symbol}_open',
            'high': f'{symbol}_high',
            'low': f'{symbol}_low',
            'close': f'{symbol}_close',
            'volume': f'{symbol}_volume'
        })
        return df[[f'{symbol}_open', f'{symbol}_high', f'{symbol}_low',
                   f'{symbol}_close', f'{symbol}_volume']]

    def _resample_to_timeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1min data to target timeframe."""
        timeframe_rules = {
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1hour': '1h',
            '2hour': '2h',
            '3hour': '3h',
            '4hour': '4h',
            'daily': '1D'
        }

        rule = timeframe_rules.get(self.timeframe, '1h')

        # Get column prefix (tsla_ or spy_)
        prefix = df.columns[0].split('_')[0]

        resampled = df.resample(rule).agg({
            f'{prefix}_open': 'first',
            f'{prefix}_high': 'max',
            f'{prefix}_low': 'min',
            f'{prefix}_close': 'last',
            f'{prefix}_volume': 'sum'
        }).dropna()

        return resampled

    def _merge_historical_live(self, historical: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
        """
        Merge historical and live data, removing duplicates.
        Live data takes precedence for overlapping timestamps.
        """
        # Find where historical ends
        if len(historical) == 0:
            return live

        if len(live) == 0:
            return historical

        # Remove any historical data that overlaps with live
        live_start = live.index[0]
        historical_clean = historical[historical.index < live_start]

        # Concatenate
        merged = pd.concat([historical_clean, live])
        merged = merged.sort_index()

        # Remove any duplicate timestamps (shouldn't happen, but safety check)
        merged = merged[~merged.index.duplicated(keep='last')]

        return merged

    def _align_data(self, tsla_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align TSLA and SPY data via inner join on timestamps.
        Zero-tolerance: only exact timestamp matches.
        """
        # Inner join on index (timestamps)
        common_idx = tsla_df.index.intersection(spy_df.index)

        aligned = pd.concat([
            tsla_df.loc[common_idx],
            spy_df.loc[common_idx]
        ], axis=1)

        return aligned

    def is_market_open(self) -> bool:
        """Check if US stock market is currently open."""
        now = datetime.now()

        # Weekend check
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Market hours: 9:30 AM - 4:00 PM ET
        # (Simplified - doesn't account for holidays)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close


def test_live_data_loader():
    """Test live data loading."""
    print("="*70)
    print("TESTING LIVE DATA LOADER")
    print("="*70)

    loader = LiveDataLoader(timeframe='1hour')

    # Load last 5 days
    aligned_df, status = loader.load_live_data(lookback_days=5)

    print(f"\n✓ Loaded {len(aligned_df)} aligned bars")
    print(f"✓ Status: {status}")
    print(f"✓ Latest timestamp: {aligned_df.index[-1]}")
    print(f"✓ Latest TSLA close: ${aligned_df.iloc[-1]['tsla_close']:.2f}")
    print(f"✓ Market open: {loader.is_market_open()}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == '__main__':
    test_live_data_loader()
