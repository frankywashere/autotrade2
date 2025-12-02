"""
Hybrid Live Data Feed for Real-Time Predictions

Fetches ALL native yfinance intervals to avoid resampling issues:
- 1m:  7 days (base data for alignment)
- 5m:  60 days (native 5-minute data)
- 15m: 60 days (native 15-minute data)
- 30m: 60 days (native 30-minute data)
- 1h:  2 years (native hourly data)
- 1d:  max history (native daily data)
- 1wk: 10 years (native weekly data)
- 1mo: max history (native monthly data)

This ensures all 14,487 features can be extracted without skipping timeframes.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple
from pathlib import Path
import sys

# Add parent to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


class HybridLiveDataFeed:
    """
    Smart data fetcher that fetches all native yfinance intervals.

    Fetches 8 different resolutions natively to avoid resampling gaps.
    """

    # Native interval configurations: (period, yfinance_interval)
    INTERVALS = {
        '1m':  ('7d', '1m'),      # Base data - 7 days max from yfinance
        '5m':  ('60d', '5m'),     # 60 days of 5-min bars
        '15m': ('60d', '15m'),    # 60 days of 15-min bars
        '30m': ('60d', '30m'),    # 60 days of 30-min bars
        '1h':  ('2y', '1h'),      # 2 years of hourly bars
        '1d':  ('max', '1d'),     # Max history daily bars
        '1wk': ('10y', '1wk'),    # 10 years of weekly bars
        '1mo': ('max', '1mo'),    # Max history monthly bars
    }

    def __init__(self, symbols: list = ['TSLA', 'SPY']):
        """
        Initialize hybrid data feed.

        Args:
            symbols: List of symbols to fetch (default: ['TSLA', 'SPY'])
        """
        self.symbols = symbols

    def fetch_for_prediction(self, as_of_time: datetime = None) -> pd.DataFrame:
        """
        Fetch all necessary data for making a live prediction.

        Downloads all 8 native intervals from yfinance for each symbol,
        then aligns SPY and TSLA at each resolution.

        Args:
            as_of_time: Prediction timestamp (default: now)

        Returns:
            Aligned 1-min DataFrame with multi_resolution attr containing all intervals
        """
        if as_of_time is None:
            as_of_time = datetime.now()

        print(f"\n📡 Fetching live data for prediction at {as_of_time.strftime('%Y-%m-%d %H:%M')}...")

        # Fetch all intervals for all symbols
        all_data = {}
        for symbol in self.symbols:
            print(f"   Downloading {symbol}...")
            all_data[symbol] = self._fetch_all_intervals(symbol)

        # Align SPY and TSLA at each resolution
        print(f"\n   Aligning SPY and TSLA at all resolutions...")

        aligned_1min = self._align_at_resolution(all_data, '1m')
        aligned_5min = self._align_at_resolution(all_data, '5m')
        aligned_15min = self._align_at_resolution(all_data, '15m')
        aligned_30min = self._align_at_resolution(all_data, '30m')
        aligned_1hour = self._align_at_resolution(all_data, '1h')
        aligned_daily = self._align_at_resolution(all_data, '1d')
        aligned_weekly = self._align_at_resolution(all_data, '1wk')
        aligned_monthly = self._align_at_resolution(all_data, '1mo')

        # Print alignment stats
        print(f"   ✓ 1-min aligned:    {len(aligned_1min):,} bars")
        print(f"   ✓ 5-min aligned:    {len(aligned_5min):,} bars")
        print(f"   ✓ 15-min aligned:   {len(aligned_15min):,} bars")
        print(f"   ✓ 30-min aligned:   {len(aligned_30min):,} bars")
        print(f"   ✓ 1-hour aligned:   {len(aligned_1hour):,} bars")
        print(f"   ✓ Daily aligned:    {len(aligned_daily):,} bars")
        print(f"   ✓ Weekly aligned:   {len(aligned_weekly):,} bars")
        print(f"   ✓ Monthly aligned:  {len(aligned_monthly):,} bars")
        print(f"   ✓ Ready for feature extraction (all native intervals)")

        # Store multi-resolution data for feature extraction
        # Keys match what features.py expects in is_live_mode routing
        aligned_1min.attrs['multi_resolution'] = {
            '1min': aligned_1min,
            '5min': aligned_5min,
            '15min': aligned_15min,
            '30min': aligned_30min,
            '1hour': aligned_1hour,
            'daily': aligned_daily,
            'weekly': aligned_weekly,
            'monthly': aligned_monthly,
        }

        return aligned_1min

    def _fetch_all_intervals(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch all 8 native intervals for a symbol.

        Args:
            symbol: Ticker symbol (e.g., 'TSLA')

        Returns:
            Dict mapping interval name to DataFrame
        """
        data = {}

        for interval_name, (period, yf_interval) in self.INTERVALS.items():
            try:
                df = yf.download(symbol, period=period, interval=yf_interval, progress=False)

                if not df.empty:
                    # Flatten MultiIndex columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    # Normalize column names to lowercase
                    df.columns = [c.lower() for c in df.columns]

                    # Remove timezone info
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)

                    print(f"      ✓ {interval_name:5s}: {len(df):,} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
                    data[interval_name] = df
                else:
                    print(f"      ✗ {interval_name}: No data")
                    data[interval_name] = pd.DataFrame()

            except Exception as e:
                print(f"      ✗ {interval_name}: Error - {e}")
                data[interval_name] = pd.DataFrame()

        return data

    def _align_at_resolution(self, all_data: Dict, resolution: str) -> pd.DataFrame:
        """
        Align SPY and TSLA at specified resolution.

        Args:
            all_data: Dict with symbol → {resolution → DataFrame}
            resolution: '1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'

        Returns:
            Aligned DataFrame with spy_* and tsla_* columns
        """
        # Get data for both symbols at this resolution
        tsla_data = all_data.get('TSLA', {}).get(resolution, pd.DataFrame()).copy()
        spy_data = all_data.get('SPY', {}).get(resolution, pd.DataFrame()).copy()

        if tsla_data.empty or spy_data.empty:
            print(f"   ⚠️  Warning: Missing {resolution} data for alignment")
            return pd.DataFrame()

        # Add symbol prefix
        tsla_data = tsla_data.add_prefix('tsla_')
        spy_data = spy_data.add_prefix('spy_')

        # Inner join (zero-tolerance alignment)
        aligned = tsla_data.join(spy_data, how='inner')

        return aligned


def test_hybrid_feed():
    """Test the hybrid feed with all native intervals."""
    print("Testing HybridLiveDataFeed with ALL native intervals...")
    print("=" * 60)

    feed = HybridLiveDataFeed(symbols=['TSLA', 'SPY'])

    # Fetch live data
    df = feed.fetch_for_prediction()

    if not df.empty:
        print(f"\n✅ Successfully fetched {len(df)} aligned 1-min bars")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")

        # Verify multi-resolution data
        if 'multi_resolution' in df.attrs:
            print(f"\n   Multi-resolution data available:")
            for res_name, res_data in df.attrs['multi_resolution'].items():
                if isinstance(res_data, pd.DataFrame) and not res_data.empty:
                    print(f"      {res_name:10s}: {len(res_data):,} bars")
                else:
                    print(f"      {res_name:10s}: MISSING")

        # Test feature extraction
        print(f"\n   Testing feature extraction with native intervals...")
        from src.ml.features import TradingFeatureExtractor
        from src.ml.events import CombinedEventsHandler

        extractor = TradingFeatureExtractor()
        events = CombinedEventsHandler()

        result = extractor.extract_features(
            df,
            use_cache=False,
            use_chunking=False,
            continuation=False,
            events_handler=events
        )

        if isinstance(result, tuple):
            features_df = result[0]
        else:
            features_df = result

        print(f"   ✓ Extracted {features_df.shape[1]} features")

        if features_df.shape[1] == 14487:
            print(f"   ✅ SUCCESS: All 14,487 features extracted!")
        else:
            print(f"   ⚠️  Expected 14,487 features, got {features_df.shape[1]}")
            print(f"      Missing: {14487 - features_df.shape[1]} features")

    else:
        print(f"\n✗ Failed to fetch data")


if __name__ == '__main__':
    test_hybrid_feed()
