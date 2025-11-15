"""
Hybrid Live Data Feed for Real-Time Predictions

Handles yfinance limitations by intelligently merging different resolutions:
- 1-min data: 7 days max (for short timeframe channels)
- 1-hour data: 2 years (for medium timeframe channels)
- Daily data: Max history (for long timeframe channels)

This ensures we always have enough history to calculate all timeframe channels.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple
from pathlib import Path
import sys

# Add parent to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))


class HybridLiveDataFeed:
    """
    Smart data fetcher that works around yfinance limitations.

    Fetches multiple resolutions and aligns them for channel calculation.
    """

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

        Intelligently downloads:
        - 1-min: 7 days (for 5min, 15min, 30min channels)
        - 1-hour: 2 years (for 1h, 2h, 3h, 4h channels)
        - Daily: Max (for daily, weekly, monthly channels)

        Args:
            as_of_time: Prediction timestamp (default: now)

        Returns:
            Aligned 1-min DataFrame with SPY and TSLA
        """
        if as_of_time is None:
            as_of_time = datetime.now()

        print(f"\n📡 Fetching live data for prediction at {as_of_time.strftime('%Y-%m-%d %H:%M')}...")

        all_data = {}

        for symbol in self.symbols:
            print(f"   Downloading {symbol}...")

            # Download 1-min (7 days max from yfinance)
            try:
                data_1min = yf.download(symbol, period='7d', interval='1m', progress=False)
                print(f"      ✓ 1-min: {len(data_1min)} bars ({data_1min.index[0]} to {data_1min.index[-1]})")
            except Exception as e:
                print(f"      ✗ 1-min failed: {e}")
                data_1min = pd.DataFrame()

            # Download hourly (2 years)
            try:
                data_1h = yf.download(symbol, period='2y', interval='1h', progress=False)
                print(f"      ✓ 1-hour: {len(data_1h)} bars ({data_1h.index[0]} to {data_1h.index[-1]})")
            except Exception as e:
                print(f"      ✗ 1-hour failed: {e}")
                data_1h = pd.DataFrame()

            # Download daily (max history)
            try:
                data_daily = yf.download(symbol, period='max', interval='1d', progress=False)
                print(f"      ✓ Daily: {len(data_daily)} bars ({data_daily.index[0]} to {data_daily.index[-1]})")
            except Exception as e:
                print(f"      ✗ Daily failed: {e}")
                data_daily = pd.DataFrame()

            all_data[symbol] = {
                '1min': data_1min,
                '1h': data_1h,
                'daily': data_daily
            }

        # Align SPY and TSLA at all resolutions
        print(f"\n   Aligning SPY and TSLA at all resolutions...")
        aligned_1min = self._align_at_resolution(all_data, '1min')
        aligned_1hour = self._align_at_resolution(all_data, '1h')
        aligned_daily = self._align_at_resolution(all_data, 'daily')

        print(f"   ✓ 1-min aligned: {len(aligned_1min)} bars")
        print(f"   ✓ 1-hour aligned: {len(aligned_1hour)} bars")
        print(f"   ✓ Daily aligned: {len(aligned_daily)} bars")
        print(f"   ✓ Ready for feature extraction")

        # Store multi-resolution data for channel calculations
        aligned_1min.attrs['multi_resolution'] = {
            '1min': aligned_1min,
            '1hour': aligned_1hour,
            'daily': aligned_daily
        }

        return aligned_1min

    def _align_at_resolution(self, all_data: Dict, resolution: str) -> pd.DataFrame:
        """
        Align SPY and TSLA at specified resolution.

        Args:
            all_data: Dict with symbol → {resolution → DataFrame}
            resolution: '1min', '1h', or 'daily'

        Returns:
            Aligned DataFrame with spy_* and tsla_* columns
        """
        # Get data for both symbols at this resolution
        tsla_data = all_data['TSLA'][resolution].copy()
        spy_data = all_data['SPY'][resolution].copy()

        if tsla_data.empty or spy_data.empty:
            print(f"   ⚠️  Warning: Missing {resolution} data for alignment")
            return pd.DataFrame()

        # Flatten MultiIndex columns if present (yfinance sometimes returns MultiIndex)
        if isinstance(tsla_data.columns, pd.MultiIndex):
            tsla_data.columns = tsla_data.columns.get_level_values(0)
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = spy_data.columns.get_level_values(0)

        # Normalize column names to lowercase
        tsla_data.columns = [c.lower() for c in tsla_data.columns]
        spy_data.columns = [c.lower() for c in spy_data.columns]

        # Add symbol prefix
        tsla_data = tsla_data.add_prefix('tsla_')
        spy_data = spy_data.add_prefix('spy_')

        # Inner join (zero-tolerance alignment)
        aligned = tsla_data.join(spy_data, how='inner')

        return aligned

    def get_data_for_timeframe_channel(
        self,
        symbol: str,
        timeframe: str,
        multi_res_data: Dict,
        lookback_bars: int = 168
    ) -> pd.DataFrame:
        """
        Get appropriate data resolution for calculating a specific timeframe channel.

        This handles yfinance's 7-day 1-min limit by using pre-downloaded higher resolutions.

        Args:
            symbol: 'TSLA' or 'SPY'
            timeframe: '15min', '1h', '4h', 'daily', etc.
            multi_res_data: Multi-resolution data dict
            lookback_bars: How many bars needed

        Returns:
            DataFrame with enough history to calculate channel
        """
        symbol_data = multi_res_data.get(symbol, {})

        # Route to appropriate resolution
        if timeframe in ['5min', '15min', '30min']:
            # Use 1-min data, resample
            data_1min = symbol_data.get('1min', pd.DataFrame())

            if data_1min.empty:
                return pd.DataFrame()

            # Resample
            resampled = data_1min.resample(timeframe).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

            # Check if enough data
            if len(resampled) < lookback_bars:
                print(f"   ⚠️  Only {len(resampled)} bars for {timeframe} (need {lookback_bars})")
                print(f"       Using what's available (channel quality may be lower)")

            return resampled

        elif timeframe in ['1h', '2h', '3h', '4h']:
            # Use pre-downloaded hourly data
            data_1h = symbol_data.get('1h', pd.DataFrame())

            if data_1h.empty:
                return pd.DataFrame()

            # Resample to target if needed
            if timeframe != '1h':
                resampled = data_1h.resample(timeframe).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            else:
                resampled = data_1h

            # We have 2 years of hourly = ~3,120 bars (plenty for lookback=168)
            return resampled

        elif timeframe in ['daily', 'weekly', 'monthly', '3month']:
            # Use daily data
            data_daily = symbol_data.get('daily', pd.DataFrame())

            if data_daily.empty:
                return pd.DataFrame()

            # Resample if needed
            if timeframe != 'daily':
                resample_rule = {'weekly': '1W', 'monthly': '1ME', '3month': '3ME'}[timeframe]
                resampled = data_daily.resample(resample_rule).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            else:
                resampled = data_daily

            # We have many years of daily data (plenty for any lookback)
            return resampled

        else:
            print(f"   ⚠️  Unknown timeframe: {timeframe}")
            return pd.DataFrame()


def test_hybrid_feed():
    """Test the hybrid feed."""
    print("Testing HybridLiveDataFeed...")

    feed = HybridLiveDataFeed(symbols=['TSLA', 'SPY'])

    # Fetch live data
    df = feed.fetch_for_prediction()

    if not df.empty:
        print(f"\n✅ Successfully fetched {len(df)} aligned 1-min bars")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")

        # Check multi-resolution data
        if 'multi_resolution' in df.attrs:
            print(f"\n   Multi-resolution data available:")
            for symbol, resolutions in df.attrs['multi_resolution'].items():
                print(f"      {symbol}:")
                for res, data in resolutions.items():
                    if not data.empty:
                        print(f"         {res}: {len(data)} bars")

        # Test feature extraction (will use cached channels if available)
        from src.ml.features import TradingFeatureExtractor

        print(f"\n   Testing feature extraction...")
        extractor = TradingFeatureExtractor()
        features = extractor.extract_features(df)

        print(f"   ✓ Extracted {len(features.columns)} features")
        print(f"   ✓ Ready for prediction!")

    else:
        print(f"\n✗ Failed to fetch data")


if __name__ == '__main__':
    test_hybrid_feed()
