"""
Live Data Fetcher for Dashboard

Public API for fetching, merging, and validating live market data.
Orchestrates LiveDataFetcher, DataMerger, and VIX loading with comprehensive error handling.

Main Function:
    fetch_live_data() -> Returns (base_df, metadata) tuple for dashboard consumption

Design:
- Graceful degradation: Falls back to historical data if live fetch fails
- Multi-resolution support: Sets multi_resolution dict in base_df.attrs
- Comprehensive metadata: Returns data quality, freshness, source info
- Error resilience: Catches and logs all errors, never crashes the dashboard
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional, Any
import warnings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import yfinance for live data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not available - live data fetching disabled")


# ============================================================================
# Constants
# ============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
TSLA_CSV = DATA_DIR / 'TSLA_1min.csv'
SPY_CSV = DATA_DIR / 'SPY_1min.csv'
VIX_CSV = DATA_DIR / 'VIX_History.csv'

# Data quality thresholds
MIN_BARS_REQUIRED = 100  # Minimum bars needed for analysis
MAX_DATA_AGE_HOURS = 48  # Max hours before data is considered stale
LIVE_FETCH_DAYS = 7      # Days of live data to fetch (yfinance limit for 1min)


# ============================================================================
# LiveDataFetcher - Fetches live data from yfinance
# ============================================================================

class LiveDataFetcher:
    """
    Fetches live stock data from yfinance.

    Features:
    - Fetches 1-minute data for last 7 days
    - Returns standardized OHLCV format
    - Handles API errors gracefully
    """

    def __init__(self, symbol: str):
        """
        Initialize fetcher for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'TSLA', 'SPY', 'VIX')
        """
        self.symbol = symbol
        self.ticker = None

        if YFINANCE_AVAILABLE:
            try:
                self.ticker = yf.Ticker(symbol)
            except Exception as e:
                warnings.warn(f"Failed to initialize ticker for {symbol}: {e}")

    def fetch(self, days: int = LIVE_FETCH_DAYS) -> Optional[pd.DataFrame]:
        """
        Fetch recent 1-minute data.

        Args:
            days: Number of days to fetch (max 7 for 1min data)

        Returns:
            DataFrame with DatetimeIndex and [open, high, low, close, volume] columns
            None if fetch fails
        """
        if not YFINANCE_AVAILABLE or self.ticker is None:
            return None

        try:
            # Limit to yfinance constraints
            days = min(days, 7)

            # Fetch 1-minute data
            df = self.ticker.history(period=f"{days}d", interval="1m")

            if df.empty:
                return None

            # Standardize column names
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Remove timezone info
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Clean data: remove NaN rows
            df = df.dropna()

            return df

        except Exception as e:
            warnings.warn(f"Failed to fetch live data for {self.symbol}: {e}")
            return None

    def get_current_price(self) -> Optional[float]:
        """
        Get current price.

        Returns:
            Current close price or None
        """
        if not YFINANCE_AVAILABLE or self.ticker is None:
            return None

        try:
            df = self.ticker.history(period="1d", interval="1m")
            if not df.empty:
                return float(df['Close'].iloc[-1])
        except Exception:
            pass

        return None


# ============================================================================
# DataMerger - Merges historical CSV data with live data
# ============================================================================

class DataMerger:
    """
    Merges historical CSV data with live fetched data.

    Features:
    - Intelligent merging (no duplicates)
    - Data validation and cleaning
    - Gap detection and reporting
    """

    @staticmethod
    def load_csv(csv_path: Path) -> Optional[pd.DataFrame]:
        """
        Load historical data from CSV.

        Args:
            csv_path: Path to CSV file

        Returns:
            DataFrame with DatetimeIndex or None if failed
        """
        if not csv_path.exists():
            warnings.warn(f"CSV not found: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.columns = df.columns.str.lower()

            # Validate required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                warnings.warn(f"CSV missing required columns: {csv_path}")
                return None

            # Keep only OHLCV
            df = df[required]

            # Clean data
            df = df.dropna()
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            warnings.warn(f"Failed to load CSV {csv_path}: {e}")
            return None

    @staticmethod
    def merge(historical_df: pd.DataFrame, live_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Merge historical and live data.

        Args:
            historical_df: Historical data from CSV
            live_df: Live data from API (can be None)

        Returns:
            (merged_df, merge_info) tuple
            merge_info contains: new_bars, source, gap_detected, etc.
        """
        merge_info = {
            'new_bars': 0,
            'source': 'historical_only',
            'gap_detected': False,
            'gap_hours': 0.0,
            'total_bars': len(historical_df),
            'latest_timestamp': historical_df.index[-1] if len(historical_df) > 0 else None
        }

        # If no live data, return historical only
        if live_df is None or len(live_df) == 0:
            return historical_df, merge_info

        # Find where historical ends and live starts
        hist_end = historical_df.index.max()
        live_start = live_df.index.min()

        # Filter live data to only new bars after historical
        new_data = live_df[live_df.index > hist_end]

        if len(new_data) == 0:
            merge_info['source'] = 'historical_only'
            return historical_df, merge_info

        # Detect gap between historical and live
        gap = (new_data.index.min() - hist_end).total_seconds() / 3600
        if gap > 1.0:  # More than 1 hour gap
            merge_info['gap_detected'] = True
            merge_info['gap_hours'] = gap

        # Merge: concatenate and remove duplicates
        merged = pd.concat([historical_df, new_data])
        merged.sort_index(inplace=True)
        merged = merged[~merged.index.duplicated(keep='last')]

        # Update merge info
        merge_info['new_bars'] = len(new_data)
        merge_info['source'] = 'historical+live'
        merge_info['total_bars'] = len(merged)
        merge_info['latest_timestamp'] = merged.index[-1]

        return merged, merge_info


# ============================================================================
# VIX Loader - Loads VIX data with fallback
# ============================================================================

class VIXLoader:
    """
    Loads VIX data from CSV or live source.

    VIX is typically daily data, so we don't fetch 1-minute.
    """

    @staticmethod
    def load(csv_path: Path = VIX_CSV) -> Optional[pd.DataFrame]:
        """
        Load VIX data.

        Args:
            csv_path: Path to VIX CSV

        Returns:
            DataFrame with DatetimeIndex or None
        """
        if not csv_path.exists():
            warnings.warn(f"VIX CSV not found: {csv_path}")
            return VIXLoader._create_fallback_vix()

        try:
            df = pd.read_csv(csv_path, parse_dates=['DATE'])
            df.set_index('DATE', inplace=True)
            df.columns = df.columns.str.lower()

            # Validate
            if 'close' not in df.columns:
                return VIXLoader._create_fallback_vix()

            # Clean
            df = df.dropna(subset=['close'])
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            warnings.warn(f"Failed to load VIX data: {e}")
            return VIXLoader._create_fallback_vix()

    @staticmethod
    def _create_fallback_vix() -> pd.DataFrame:
        """
        Create fallback VIX data (constant 20.0).

        Returns:
            DataFrame with dummy VIX data
        """
        # Create 1 year of daily data at VIX=20
        dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
        df = pd.DataFrame({
            'open': 20.0,
            'high': 21.0,
            'low': 19.0,
            'close': 20.0,
            'volume': 0
        }, index=dates)
        df.index.name = 'DATE'
        return df


# ============================================================================
# Multi-Resolution Resampler
# ============================================================================

def create_multi_resolution(base_df: pd.DataFrame, base_resolution: str = '5min') -> Dict[str, pd.DataFrame]:
    """
    Create multi-resolution views of the base data.

    Args:
        base_df: Base OHLCV data (e.g., 5min bars)
        base_resolution: Base timeframe (default '5min')

    Returns:
        Dictionary mapping resolution -> DataFrame
        Includes: '1min', '5min', '15min', '1h', '4h', 'daily'
    """
    from v7.core.timeframe import resample_ohlc, TIMEFRAMES

    multi_res = {}

    # Base resolution
    multi_res[base_resolution] = base_df.copy()

    # Resample to other timeframes
    for tf in TIMEFRAMES:
        if tf == base_resolution:
            continue

        try:
            multi_res[tf] = resample_ohlc(base_df, tf)
        except Exception as e:
            warnings.warn(f"Failed to resample to {tf}: {e}")
            # Create empty placeholder
            multi_res[tf] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    return multi_res


# ============================================================================
# Data Freshness Checker
# ============================================================================

def check_data_freshness(timestamp: pd.Timestamp) -> Dict[str, Any]:
    """
    Check how fresh the data is.

    Args:
        timestamp: Latest data timestamp

    Returns:
        Dictionary with freshness info
    """
    now = pd.Timestamp.now()
    age = now - timestamp
    age_hours = age.total_seconds() / 3600

    # Determine status
    if age_hours < 0.5:
        status = 'live'
        quality = 'excellent'
    elif age_hours < 4:
        status = 'recent'
        quality = 'good'
    elif age_hours < 24:
        status = 'stale'
        quality = 'fair'
    elif age_hours < MAX_DATA_AGE_HOURS:
        status = 'old'
        quality = 'poor'
    else:
        status = 'outdated'
        quality = 'unusable'

    return {
        'timestamp': timestamp,
        'age_hours': age_hours,
        'age_minutes': age.total_seconds() / 60,
        'status': status,
        'quality': quality,
        'message': f"Data is {int(age_hours)}h {int((age.total_seconds() % 3600) / 60)}m old"
    }


# ============================================================================
# MAIN PUBLIC API
# ============================================================================

def fetch_live_data(
    lookback_days: int = 90,
    resample_to: str = '5min',
    enable_live_fetch: bool = True,
    include_multi_resolution: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetch and merge live market data for dashboard consumption.

    This is the main public API function that dashboard.py calls.

    Orchestration:
    1. Load historical data from CSV files (TSLA, SPY, VIX)
    2. Fetch live data from yfinance (if enabled and available)
    3. Merge historical + live with intelligent deduplication
    4. Validate data quality and freshness
    5. Create multi-resolution views
    6. Return data + comprehensive metadata

    Args:
        lookback_days: Days of historical data to load (default 90)
        resample_to: Target base resolution (default '5min')
        enable_live_fetch: Whether to attempt live data fetch (default True)
        include_multi_resolution: Whether to create multi-resolution views (default True)

    Returns:
        Tuple of (base_df, metadata):

        base_df: pd.DataFrame
            - TSLA OHLCV data at base resolution (default 5min)
            - DatetimeIndex
            - Columns: [open, high, low, close, volume]
            - attrs['multi_resolution']: Dict[str, pd.DataFrame] (if include_multi_resolution=True)
            - attrs['spy_df']: SPY data at same resolution
            - attrs['vix_df']: VIX daily data

        metadata: Dict containing:
            - 'tsla': Dict with TSLA data info (source, bars, freshness, etc.)
            - 'spy': Dict with SPY data info
            - 'vix': Dict with VIX data info
            - 'quality': Overall data quality ('excellent', 'good', 'fair', 'poor', 'unusable')
            - 'errors': List of errors encountered (empty if none)
            - 'warnings': List of warnings
            - 'timestamp': When this fetch was performed

    Example:
        >>> base_df, metadata = fetch_live_data(lookback_days=60, resample_to='5min')
        >>> print(f"Loaded {len(base_df)} bars, quality: {metadata['quality']}")
        >>> multi_res = base_df.attrs['multi_resolution']
        >>> hourly_df = multi_res['1h']

    Error Handling:
        - Never raises exceptions
        - Falls back to historical data if live fetch fails
        - Returns error information in metadata
        - Creates fallback data if CSV files missing
    """
    metadata = {
        'tsla': {},
        'spy': {},
        'vix': {},
        'quality': 'unknown',
        'errors': [],
        'warnings': [],
        'timestamp': datetime.now(),
        'live_fetch_enabled': enable_live_fetch and YFINANCE_AVAILABLE
    }

    # ========================================================================
    # Step 1: Load TSLA Data
    # ========================================================================

    tsla_df = None
    tsla_live_df = None

    try:
        # Load historical TSLA
        tsla_historical = DataMerger.load_csv(TSLA_CSV)

        if tsla_historical is not None and len(tsla_historical) > 0:
            # Filter to lookback period
            cutoff = datetime.now() - timedelta(days=lookback_days)
            tsla_historical = tsla_historical[tsla_historical.index >= cutoff]

            # Check if filtering left us with no data
            if len(tsla_historical) > 0:
                metadata['tsla']['historical_bars'] = len(tsla_historical)
                metadata['tsla']['historical_range'] = (
                    tsla_historical.index.min(),
                    tsla_historical.index.max()
                )
            else:
                # CSV exists but data is too old
                metadata['warnings'].append(f"TSLA CSV data is outdated (ends before {cutoff.date()}), using fallback")
                tsla_historical = _create_fallback_ohlcv('TSLA', days=lookback_days, resolution='1min')
                metadata['tsla']['historical_bars'] = len(tsla_historical)
                metadata['tsla']['historical_range'] = (
                    tsla_historical.index.min(),
                    tsla_historical.index.max()
                )
                metadata['tsla']['source'] = 'fallback'
        else:
            metadata['warnings'].append("TSLA CSV not found, using fallback data")
            # Create minimal fallback (1min resolution to match CSV)
            tsla_historical = _create_fallback_ohlcv('TSLA', days=lookback_days, resolution='1min')

            # Update metadata for fallback
            metadata['tsla']['historical_bars'] = len(tsla_historical)
            metadata['tsla']['historical_range'] = (
                tsla_historical.index.min(),
                tsla_historical.index.max()
            )
            metadata['tsla']['source'] = 'fallback'

        # Fetch live TSLA (if enabled)
        if enable_live_fetch and YFINANCE_AVAILABLE:
            try:
                fetcher = LiveDataFetcher('TSLA')
                tsla_live_df = fetcher.fetch(days=LIVE_FETCH_DAYS)

                if tsla_live_df is not None:
                    metadata['tsla']['live_bars'] = len(tsla_live_df)
                    metadata['tsla']['live_range'] = (
                        tsla_live_df.index.min(),
                        tsla_live_df.index.max()
                    )
            except Exception as e:
                metadata['warnings'].append(f"Live TSLA fetch failed: {e}")
                tsla_live_df = None

        # Merge historical + live
        tsla_df, merge_info = DataMerger.merge(tsla_historical, tsla_live_df)
        metadata['tsla'].update(merge_info)

        # Check freshness
        if len(tsla_df) > 0:
            freshness = check_data_freshness(tsla_df.index[-1])
            metadata['tsla']['freshness'] = freshness

    except Exception as e:
        metadata['errors'].append(f"TSLA data pipeline failed: {e}")
        tsla_df = _create_fallback_ohlcv('TSLA', days=lookback_days, resolution='1min')

    # ========================================================================
    # Step 2: Load SPY Data
    # ========================================================================

    spy_df = None
    spy_live_df = None

    try:
        # Load historical SPY
        spy_historical = DataMerger.load_csv(SPY_CSV)

        if spy_historical is not None and len(spy_historical) > 0:
            cutoff = datetime.now() - timedelta(days=lookback_days)
            spy_historical = spy_historical[spy_historical.index >= cutoff]

            # Check if filtering left us with no data
            if len(spy_historical) > 0:
                metadata['spy']['historical_bars'] = len(spy_historical)
                metadata['spy']['historical_range'] = (
                    spy_historical.index.min(),
                    spy_historical.index.max()
                )
            else:
                # CSV exists but data is too old
                metadata['warnings'].append(f"SPY CSV data is outdated (ends before {cutoff.date()}), using fallback")
                spy_historical = _create_fallback_ohlcv('SPY', days=lookback_days, resolution='1min')
                metadata['spy']['historical_bars'] = len(spy_historical)
                metadata['spy']['historical_range'] = (
                    spy_historical.index.min(),
                    spy_historical.index.max()
                )
                metadata['spy']['source'] = 'fallback'
        else:
            metadata['warnings'].append("SPY CSV not found, using fallback data")
            spy_historical = _create_fallback_ohlcv('SPY', days=lookback_days, resolution='1min')

            # Update metadata for fallback
            metadata['spy']['historical_bars'] = len(spy_historical)
            metadata['spy']['historical_range'] = (
                spy_historical.index.min(),
                spy_historical.index.max()
            )
            metadata['spy']['source'] = 'fallback'

        # Fetch live SPY (if enabled)
        if enable_live_fetch and YFINANCE_AVAILABLE:
            try:
                fetcher = LiveDataFetcher('SPY')
                spy_live_df = fetcher.fetch(days=LIVE_FETCH_DAYS)

                if spy_live_df is not None:
                    metadata['spy']['live_bars'] = len(spy_live_df)
                    metadata['spy']['live_range'] = (
                        spy_live_df.index.min(),
                        spy_live_df.index.max()
                    )
            except Exception as e:
                metadata['warnings'].append(f"Live SPY fetch failed: {e}")
                spy_live_df = None

        # Merge
        spy_df, merge_info = DataMerger.merge(spy_historical, spy_live_df)
        metadata['spy'].update(merge_info)

        # Freshness
        if len(spy_df) > 0:
            freshness = check_data_freshness(spy_df.index[-1])
            metadata['spy']['freshness'] = freshness

    except Exception as e:
        metadata['errors'].append(f"SPY data pipeline failed: {e}")
        spy_df = _create_fallback_ohlcv('SPY', days=lookback_days, resolution='1min')

    # ========================================================================
    # Step 3: Load VIX Data
    # ========================================================================

    vix_df = None

    try:
        vix_df = VIXLoader.load()

        if vix_df is not None and len(vix_df) > 0:
            # Filter to lookback period
            cutoff = datetime.now() - timedelta(days=lookback_days)
            vix_df = vix_df[vix_df.index >= cutoff]

            # Check if still has data after filtering
            if len(vix_df) > 0:
                metadata['vix']['bars'] = len(vix_df)
                metadata['vix']['range'] = (vix_df.index.min(), vix_df.index.max())
                metadata['vix']['current_level'] = float(vix_df['close'].iloc[-1])

                freshness = check_data_freshness(vix_df.index[-1])
                metadata['vix']['freshness'] = freshness
            else:
                metadata['warnings'].append("VIX data empty after filtering, using fallback")
                vix_df = VIXLoader._create_fallback_vix()
                # Filter fallback to lookback period
                vix_df = vix_df[vix_df.index >= cutoff]
                metadata['vix']['bars'] = len(vix_df)
                metadata['vix']['range'] = (vix_df.index.min(), vix_df.index.max())
                if len(vix_df) > 0:
                    metadata['vix']['current_level'] = float(vix_df['close'].iloc[-1])
        else:
            metadata['warnings'].append("Using fallback VIX data")
            vix_df = VIXLoader._create_fallback_vix()
            # Filter fallback to lookback period
            cutoff = datetime.now() - timedelta(days=lookback_days)
            vix_df = vix_df[vix_df.index >= cutoff]
            metadata['vix']['bars'] = len(vix_df)
            if len(vix_df) > 0:
                metadata['vix']['range'] = (vix_df.index.min(), vix_df.index.max())
                metadata['vix']['current_level'] = float(vix_df['close'].iloc[-1])

    except Exception as e:
        metadata['errors'].append(f"VIX data pipeline failed: {e}")
        try:
            vix_df = VIXLoader._create_fallback_vix()
            cutoff = datetime.now() - timedelta(days=lookback_days)
            vix_df = vix_df[vix_df.index >= cutoff]
        except:
            # Ultimate fallback: empty dataframe
            vix_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    # ========================================================================
    # Step 4: Resample to Target Resolution
    # ========================================================================

    try:
        if resample_to != '1min':
            from v7.core.timeframe import resample_ohlc

            # Resample TSLA
            if len(tsla_df) > 0:
                tsla_df = resample_ohlc(tsla_df, resample_to)

            # Resample SPY
            if len(spy_df) > 0:
                spy_df = resample_ohlc(spy_df, resample_to)

        metadata['base_resolution'] = resample_to

    except Exception as e:
        metadata['errors'].append(f"Resampling failed: {e}")

    # ========================================================================
    # Step 5: Create Multi-Resolution Views
    # ========================================================================

    multi_resolution = {}

    if include_multi_resolution:
        try:
            multi_resolution = create_multi_resolution(tsla_df, resample_to)
            metadata['multi_resolution_timeframes'] = list(multi_resolution.keys())
        except Exception as e:
            metadata['errors'].append(f"Multi-resolution creation failed: {e}")
            multi_resolution = {resample_to: tsla_df}

    # ========================================================================
    # Step 6: Validate Data Quality
    # ========================================================================

    try:
        # Check minimum bars
        if len(tsla_df) < MIN_BARS_REQUIRED:
            metadata['errors'].append(f"Insufficient TSLA data: {len(tsla_df)} < {MIN_BARS_REQUIRED}")
            metadata['quality'] = 'unusable'
        elif len(spy_df) < MIN_BARS_REQUIRED:
            metadata['errors'].append(f"Insufficient SPY data: {len(spy_df)} < {MIN_BARS_REQUIRED}")
            metadata['quality'] = 'unusable'
        else:
            # Determine overall quality based on freshness
            tsla_quality = metadata['tsla'].get('freshness', {}).get('quality', 'poor')
            spy_quality = metadata['spy'].get('freshness', {}).get('quality', 'poor')

            # Take worst quality
            qualities = ['excellent', 'good', 'fair', 'poor', 'unusable']
            tsla_idx = qualities.index(tsla_quality) if tsla_quality in qualities else 3
            spy_idx = qualities.index(spy_quality) if spy_quality in qualities else 3
            overall_idx = max(tsla_idx, spy_idx)

            metadata['quality'] = qualities[overall_idx]

        # Check for errors
        if len(metadata['errors']) > 0:
            metadata['quality'] = 'poor'  # Errors degrade quality

    except Exception as e:
        metadata['errors'].append(f"Quality validation failed: {e}")
        metadata['quality'] = 'unknown'

    # ========================================================================
    # Step 7: Attach Additional Data to base_df.attrs
    # ========================================================================

    try:
        tsla_df.attrs['multi_resolution'] = multi_resolution
        tsla_df.attrs['spy_df'] = spy_df
        tsla_df.attrs['vix_df'] = vix_df
        tsla_df.attrs['metadata'] = metadata

    except Exception as e:
        metadata['warnings'].append(f"Failed to attach attrs: {e}")

    # ========================================================================
    # Step 8: Final Summary
    # ========================================================================

    metadata['summary'] = {
        'tsla_bars': len(tsla_df),
        'spy_bars': len(spy_df),
        'vix_bars': len(vix_df),
        'resolution': resample_to,
        'quality': metadata['quality'],
        'has_live_data': metadata['tsla'].get('source') == 'historical+live',
        'error_count': len(metadata['errors']),
        'warning_count': len(metadata['warnings'])
    }

    return tsla_df, metadata


# ============================================================================
# Fallback Data Creator
# ============================================================================

def _create_fallback_ohlcv(symbol: str, days: int = 90, resolution: str = '5min') -> pd.DataFrame:
    """
    Create fallback OHLCV data with realistic prices.

    Args:
        symbol: Stock symbol (for price scaling)
        days: Number of days of data
        resolution: Data resolution (default '5min')

    Returns:
        DataFrame with dummy OHLCV data
    """
    # Realistic base prices
    base_prices = {
        'TSLA': 250.0,
        'SPY': 500.0,
        'default': 100.0
    }

    base = base_prices.get(symbol, base_prices['default'])

    # Determine bars per day based on resolution
    bars_per_day = {
        '1min': 390,   # Market hours
        '5min': 78,    # 390/5
        '15min': 26,   # 390/15
        '1h': 6,       # Approx
        '4h': 2,
        'daily': 1
    }
    bars = bars_per_day.get(resolution, 78)  # Default to 5min

    # Create dates
    dates = pd.date_range(end=datetime.now(), periods=days * bars, freq=resolution)

    # Create random walk around base price
    np.random.seed(42)
    returns = np.random.normal(0, 0.002, len(dates))  # 0.2% std
    prices = base * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)

    df.index.name = 'timestamp'

    return df


# ============================================================================
# CLI Test/Demo
# ============================================================================

if __name__ == '__main__':
    """
    Test the fetch_live_data() function.

    Usage:
        python fetch_live_data.py
    """
    print("=" * 80)
    print("Testing fetch_live_data()")
    print("=" * 80)

    # Fetch data
    base_df, metadata = fetch_live_data(
        lookback_days=30,
        resample_to='5min',
        enable_live_fetch=True,
        include_multi_resolution=True
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nBase DataFrame:")
    print(f"  Bars: {len(base_df)}")
    print(f"  Resolution: {metadata['base_resolution']}")
    if len(base_df) > 0:
        print(f"  Range: {base_df.index.min()} to {base_df.index.max()}")
        print(f"  Latest Close: ${base_df['close'].iloc[-1]:.2f}")
    else:
        print(f"  Range: EMPTY")
        print(f"  Latest Close: N/A")

    print(f"\nData Quality: {metadata['quality'].upper()}")

    print(f"\nTSLA Data:")
    for key, value in metadata['tsla'].items():
        print(f"  {key}: {value}")

    print(f"\nSPY Data:")
    for key, value in metadata['spy'].items():
        print(f"  {key}: {value}")

    print(f"\nVIX Data:")
    for key, value in metadata['vix'].items():
        print(f"  {key}: {value}")

    if metadata['errors']:
        print(f"\nErrors ({len(metadata['errors'])}):")
        for err in metadata['errors']:
            print(f"  - {err}")

    if metadata['warnings']:
        print(f"\nWarnings ({len(metadata['warnings'])}):")
        for warn in metadata['warnings']:
            print(f"  - {warn}")

    print(f"\nMulti-Resolution Views:")
    multi_res = base_df.attrs.get('multi_resolution', {})
    for tf, df in multi_res.items():
        print(f"  {tf:8s}: {len(df):5d} bars")

    print(f"\nSPY DataFrame:")
    spy_df = base_df.attrs.get('spy_df')
    if spy_df is not None and len(spy_df) > 0:
        print(f"  Bars: {len(spy_df)}")
        print(f"  Latest: ${spy_df['close'].iloc[-1]:.2f}")
    else:
        print(f"  EMPTY or None")

    print(f"\nVIX DataFrame:")
    vix_df = base_df.attrs.get('vix_df')
    if vix_df is not None and len(vix_df) > 0:
        print(f"  Bars: {len(vix_df)}")
        print(f"  Latest: {vix_df['close'].iloc[-1]:.2f}")
    else:
        print(f"  EMPTY or None")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
