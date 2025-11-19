"""
Feature extraction system for ML model
Leverages existing Stage 1 components (channels, RSI) plus new features
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple
from pathlib import Path
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config
from src.linear_regression import LinearRegressionChannel
from src.rsi_calculator import RSICalculator
from .base import FeatureExtractor

# Feature cache version - increment when calculation logic changes
FEATURE_VERSION = "v3.13_multiwindow_21"  # 21-window multi-OHLC system with no filtering


def _check_gpu_available() -> tuple:
    """
    Check if GPU is available for acceleration.

    Returns:
        (available, device_type): (True, 'cuda') or (True, 'mps') or (False, 'cpu')
    """
    if torch.cuda.is_available():
        return True, 'cuda'
    elif torch.backends.mps.is_available():
        return True, 'mps'
    else:
        return False, 'cpu'


class TradingFeatureExtractor(FeatureExtractor):
    """
    Comprehensive feature extractor for stock trading
    Combines technical indicators, channel patterns, correlations, and cycles
    """

    def __init__(self):
        self.channel_calc = LinearRegressionChannel()
        self.rsi_calc = RSICalculator()
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        """Build list of all feature names"""
        features = []

        # Price features (both SPY and TSLA) - v3.8: +2 normalized prices
        for symbol in ['spy', 'tsla']:
            features.extend([
                f'{symbol}_close',
                f'{symbol}_close_norm',  # v3.8: Position in 252-bar (yearly) range
                f'{symbol}_returns',
                f'{symbol}_log_returns',
                f'{symbol}_volatility_10',
                f'{symbol}_volatility_50',
            ])

        # TSLA Channel features (multi-timeframe) - v3.7 with normalized slope + direction flags
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.extend([
                f'tsla_channel_{tf}_position',  # 0-1 position in channel
                f'tsla_channel_{tf}_upper_dist',  # Distance to upper
                f'tsla_channel_{tf}_lower_dist',  # Distance to lower
                f'tsla_channel_{tf}_slope',  # Raw slope ($/bar)
                f'tsla_channel_{tf}_slope_pct',  # Normalized slope (% per bar)
                f'tsla_channel_{tf}_stability',
                f'tsla_channel_{tf}_ping_pongs',  # 2% threshold (default)
                f'tsla_channel_{tf}_ping_pongs_0_5pct',  # 0.5% threshold (strict)
                f'tsla_channel_{tf}_ping_pongs_1_0pct',  # 1.0% threshold
                f'tsla_channel_{tf}_ping_pongs_3_0pct',  # 3.0% threshold (loose)
                f'tsla_channel_{tf}_r_squared',
                f'tsla_channel_{tf}_is_bull',  # Bull channel (>0.1% per bar)
                f'tsla_channel_{tf}_is_bear',  # Bear channel (<-0.1% per bar)
                f'tsla_channel_{tf}_is_sideways',  # Sideways channel (±0.1% per bar)
                f'tsla_channel_{tf}_duration',  # v3.11: How many bars channel actually holds
            ])

        # SPY Channel features (multi-timeframe) - v3.7 with normalized slope + direction flags
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.extend([
                f'spy_channel_{tf}_position',  # 0-1 position in channel
                f'spy_channel_{tf}_upper_dist',  # Distance to upper
                f'spy_channel_{tf}_lower_dist',  # Distance to lower
                f'spy_channel_{tf}_slope',  # Raw slope ($/bar)
                f'spy_channel_{tf}_slope_pct',  # Normalized slope (% per bar)
                f'spy_channel_{tf}_stability',
                f'spy_channel_{tf}_ping_pongs',  # 2% threshold (default)
                f'spy_channel_{tf}_ping_pongs_0_5pct',  # 0.5% threshold (strict)
                f'spy_channel_{tf}_ping_pongs_1_0pct',  # 1.0% threshold
                f'spy_channel_{tf}_ping_pongs_3_0pct',  # 3.0% threshold (loose)
                f'spy_channel_{tf}_r_squared',
                f'spy_channel_{tf}_is_bull',  # Bull channel (>0.1% per bar)
                f'spy_channel_{tf}_is_bear',  # Bear channel (<-0.1% per bar)
                f'spy_channel_{tf}_is_sideways',  # Sideways channel (±0.1% per bar)
                f'spy_channel_{tf}_duration',  # v3.11: How many bars channel actually holds
            ])

        # TSLA RSI features (multi-timeframe) - RENAMED for consistency
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.extend([
                f'tsla_rsi_{tf}',
                f'tsla_rsi_{tf}_oversold',  # Binary
                f'tsla_rsi_{tf}_overbought',  # Binary
            ])

        # SPY RSI features (multi-timeframe) - NEW in v3.4
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.extend([
                f'spy_rsi_{tf}',
                f'spy_rsi_{tf}_oversold',  # Binary
                f'spy_rsi_{tf}_overbought',  # Binary
            ])

        # SPY-TSLA correlation features
        features.extend([
            'correlation_10',  # 10-bar rolling correlation
            'correlation_50',
            'correlation_200',
            'divergence',  # SPY up, TSLA down (or vice versa)
            'divergence_magnitude',
        ])

        # Larger cycle features
        features.extend([
            'distance_from_52w_high',
            'distance_from_52w_low',
            'within_mega_channel',  # Binary: in 3-4 year channel
            'mega_channel_position',
        ])

        # Volume features
        features.extend([
            'tsla_volume_ratio',  # Current / average
            'spy_volume_ratio',
        ])

        # Time features
        features.extend([
            'hour_of_day',
            'day_of_week',
            'day_of_month',
            'month_of_year',
        ])

        # Breakdown indicator features (NEW for hierarchical model)
        features.append('tsla_volume_surge')
        for tf in ['15min', '1h', '4h', 'daily']:
            features.append(f'tsla_rsi_divergence_{tf}')
        for tf in ['1h', '4h', 'daily']:
            features.append(f'tsla_channel_duration_ratio_{tf}')
        for tf in ['1h', '4h']:
            features.append(f'channel_alignment_spy_tsla_{tf}')

        # Time-in-channel features (additional breakdown indicators)
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.append(f'tsla_time_in_channel_{tf}')
            features.append(f'spy_time_in_channel_{tf}')

        # Enhanced channel position features (normalized -1 to +1)
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.append(f'tsla_channel_position_norm_{tf}')
            features.append(f'spy_channel_position_norm_{tf}')

        # Binary feature flags (Phase 4)
        # Binary flags and event features (v3.11: +5 missing flags)
        features.extend([
            'is_monday',
            'is_tuesday',                # v3.11: Added missing weekday
            'is_wednesday',              # v3.11: Added missing weekday
            'is_thursday',               # v3.11: Added missing weekday
            'is_friday',
            'is_first_hour',             # v3.11: Market open hour (9:30-10:30 ET)
            'is_last_hour',              # v3.11: Power hour (15:00-16:00 ET)
            'is_volatile_now',
            'is_earnings_week',          # v3.10: Within ±14 days of earnings/delivery
            'days_until_earnings',       # v3.10: -14 to +14 days (0 = day of)
            'days_until_fomc',           # v3.10: -14 to +14 days (0 = day of)
            'is_high_impact_event'       # v3.10: Earnings/FOMC/Delivery within 3 days
        ])

        # In-channel binary flags
        for tf in ['1h', '4h', 'daily']:
            features.append(f'tsla_in_channel_{tf}')
            features.append(f'spy_in_channel_{tf}')

        self.feature_names = features

    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return self.feature_names

    def get_feature_dim(self) -> int:
        """Return total number of features"""
        return len(self.feature_names)

    def extract_features(self, df: pd.DataFrame, use_cache: bool = True, use_gpu: str = 'auto', cache_suffix: str = None, events_handler=None, continuation: bool = False, use_chunking: bool = False, chunk_size_years: int = 1, **kwargs) -> tuple:
        """
        Extract all 495 features from aligned SPY-TSLA data (v3.11 - With Dynamic Channel Duration Detection).

        Args:
            df: DataFrame with SPY and TSLA OHLCV columns
            use_cache: Whether to use cached rolling channels (default: True). Set to False to force regeneration.
            use_gpu: GPU acceleration mode (default: 'auto')
                - 'auto': Use GPU only if data size > 50,000 bars (smart default)
                - True: Force GPU (if available, otherwise fallback to CPU)
            use_chunking: Use chunked processing to save memory (default: False)
            chunk_size_years: Size of each chunk in years when using chunked processing (default: 1)
                - False/'never': Always use CPU
            cache_suffix: Optional suffix for cache filename (for testing GPU/CPU separately)
                - None (default): Normal cache filename
                - 'GPU_TEST': Appends to cache name for GPU testing
                - 'CPU_TEST': Appends to cache name for CPU testing
            events_handler: Optional CombinedEventsHandler for event-driven features (v3.9)
                - If provided: Enables earnings/FOMC proximity features
                - If None: Event features will be zeros (backward compatible)
            continuation: Whether to generate continuation prediction labels (default: False)
            **kwargs: Additional arguments (reserved for future use)

        df should have columns: spy_open, spy_high, spy_low, spy_close, spy_volume,
                                tsla_open, tsla_high, tsla_low, tsla_close, tsla_volume

        Returns tuple of (features_df, continuation_df):
        - features_df: DataFrame with 495 columns
        - continuation_df: DataFrame with continuation labels (or None if continuation=False)
        - 12 price features (6 per stock: close, close_norm, returns, log_returns, volatility_10, volatility_50)
        - 330 channel features (165 TSLA + 165 SPY) - v3.11: +22 duration features
          - Per timeframe (11): position, upper_dist, lower_dist, slope, slope_pct, stability, r_squared, duration
          - Ping-pongs (4 thresholds): ping_pongs (2%), ping_pongs_0_5pct, ping_pongs_1_0pct, ping_pongs_3_0pct
          - Direction flags (3): is_bull, is_bear, is_sideways
          - Total: 15 metrics per timeframe (was 14)
        - 66 RSI features (33 TSLA + 33 SPY)
        - 5 correlation features
        - 4 cycle features
        - 2 volume features
        - 4 time features
        - 54 breakdown/channel enhancement features
        - 14 binary feature flags (is_monday, is_friday, is_volatile_now, in_channel flags)
        - 4 event features (v3.9): is_earnings_week, days_until_earnings, days_until_fomc, is_high_impact_event
        """
        # Extract multi-resolution data if present (for live mode) and remove from attrs to prevent deep copy recursion
        multi_res_data = df.attrs.pop('multi_resolution', None)

        # GPU auto-detection logic
        if use_gpu == 'auto':
            # Use GPU only for large datasets (>50K bars, well above 2.5K break-even)
            # This ensures GPU for training, CPU for live/backtest
            gpu_available, device_type = _check_gpu_available()
            use_gpu_resolved = len(df) > 50_000 and gpu_available
            if use_gpu_resolved:
                print(f"   🚀 Auto-detected: Using {device_type.upper()} for feature extraction (data size: {len(df):,} bars)")
            else:
                reason = "data too small" if len(df) <= 50_000 else "GPU not available"
                print(f"   💾 Auto-detected: Using CPU for feature extraction ({reason})")
        elif use_gpu is True:
            # User explicitly requested GPU
            gpu_available, device_type = _check_gpu_available()
            use_gpu_resolved = gpu_available
            if use_gpu_resolved:
                print(f"   🚀 Using {device_type.upper()} for feature extraction (user requested)")
            else:
                print(f"   ⚠️  GPU requested but not available, falling back to CPU")
                use_gpu_resolved = False
        else:
            # use_gpu is False or 'never'
            use_gpu_resolved = False
            print(f"   💾 Using CPU for feature extraction (user requested)")

        # PASS 1: Extract base features
        print("   Extracting base features...")
        with tqdm(total=7, desc="   Feature extraction", leave=True, ncols=100, ascii=True, mininterval=0.5) as pbar:
            price_df = self._extract_price_features(df)
            pbar.update(1)

            # Chunked extraction logic
            if use_chunking and not use_cache:
                print(f"      Using chunked extraction ({chunk_size_years}-year chunks)")
                channel_df = self._extract_channel_features_chunked(
                    df,
                    multi_res_data=multi_res_data,
                    use_gpu=use_gpu_resolved,
                    chunk_size_years=chunk_size_years
                )
                # Save the combined result to cache for next time
                if cache_suffix is None:
                    cache_suffix = self._generate_cache_suffix(df)
                cache_file = self._get_cache_path(cache_suffix)
                print(f"      Saving combined result to cache: {cache_file.name}")
                cache_file.parent.mkdir(exist_ok=True, parents=True)
                channel_df.to_pickle(cache_file)
            else:
                # Normal extraction (all at once) or load from cache
                channel_df = self._extract_channel_features(
                    df,
                    multi_res_data=multi_res_data,
                    use_cache=use_cache,
                    use_gpu=use_gpu_resolved,
                    cache_suffix=cache_suffix
                )
            pbar.update(1)

            rsi_df = self._extract_rsi_features(df, multi_res_data=multi_res_data)
            pbar.update(1)

            correlation_df = self._extract_correlation_features(df)
            pbar.update(1)

            cycle_df = self._extract_cycle_features(df)
            pbar.update(1)

            volume_df = self._extract_volume_features(df)
            pbar.update(1)

            time_df = self._extract_time_features(df)
            pbar.update(1)

        # Concat base features FIRST
        base_features_df = pd.concat([
            price_df,
            channel_df,
            rsi_df,
            correlation_df,
            cycle_df,
            volume_df,
            time_df
        ], axis=1)

        # PASS 2: Extract breakdown features (needs base features + optional events)
        with tqdm(total=1, desc="   Breakdown features", leave=False, ncols=100, ascii=True, mininterval=0.5) as pbar:
            breakdown_df = self._extract_breakdown_features(base_features_df, df, events_handler)
            pbar.update(1)

        # Final concat
        features_df = pd.concat([base_features_df, breakdown_df], axis=1)

        # Generate continuation labels if requested
        continuation_df = None
        if continuation:
            print("   Generating continuation labels...")
            timestamps = df.index.tolist()
            continuation_df = self.generate_continuation_labels(df, timestamps, prediction_horizon=24)
            print(f"   ✓ Generated {len(continuation_df)} continuation labels")

        # Fill NaNs
        features_df = features_df.bfill().fillna(0)

        print(f"   ✓ Extracted {len(features_df.columns)} features")
        return features_df, continuation_df

    def _extract_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic price features. Returns DataFrame with 12 columns (v3.8: +2 normalized prices)."""
        price_features = {}

        for symbol in ['spy', 'tsla']:
            close_col = f'{symbol}_close'
            price_features[close_col] = df[close_col]

            # Normalized price (position in 252-bar/1-year range) - v3.8
            # 0 = at yearly low, 1 = at yearly high, 0.5 = middle
            rolling_min = df[close_col].rolling(window=252, min_periods=20).min()
            rolling_max = df[close_col].rolling(window=252, min_periods=20).max()
            price_range = rolling_max - rolling_min
            price_features[f'{symbol}_close_norm'] = ((df[close_col] - rolling_min) / price_range).fillna(0.5)

            # Returns
            returns = df[close_col].pct_change()
            price_features[f'{symbol}_returns'] = returns
            price_features[f'{symbol}_log_returns'] = np.log(df[close_col] / df[close_col].shift(1))

            # Volatility (rolling std of returns)
            price_features[f'{symbol}_volatility_10'] = returns.rolling(10).std()
            price_features[f'{symbol}_volatility_50'] = returns.rolling(50).std()

        return pd.DataFrame(price_features, index=df.index)

    def _calculate_dynamic_window(self, prices: pd.Series, base_window: int = 168, min_window: int = 30, max_window: int = 300) -> int:
        """
        Calculate dynamic window size based on volatility.

        Args:
            prices: Price series
            base_window: Base window size (default 168 bars)
            min_window: Minimum window size
            max_window: Maximum window size

        Returns:
            Adjusted window size based on ATR volatility
        """
        if len(prices) < 20:
            return min_window

        # Calculate ATR (Average True Range) as volatility measure
        high = prices.rolling(2).max()
        low = prices.rolling(2).min()
        close_prev = prices.shift(1)

        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()

        # Get current ATR
        current_atr = atr.iloc[-1] if not atr.empty else prices.std()

        # Calculate average ATR over recent period
        avg_atr = atr.tail(50).mean() if len(atr) > 50 else atr.mean()

        if pd.isna(avg_atr) or avg_atr == 0:
            return base_window

        # Adjust window: higher volatility = shorter window, lower volatility = longer window
        volatility_ratio = current_atr / avg_atr

        # Scale window inversely with volatility (clamp between min/max)
        adjusted_window = int(base_window / volatility_ratio)
        adjusted_window = max(min_window, min(max_window, adjusted_window))

        return adjusted_window

    def _extract_channel_features(self, df: pd.DataFrame, use_cache: bool = True, multi_res_data: dict = None, use_gpu: bool = False, cache_suffix: str = None) -> pd.DataFrame:
        """
        Extract ROLLING linear regression channel features for multiple timeframes.

        CRITICAL: Channels are calculated at EACH timestamp using a rolling lookback window.
        This captures channel dynamics (formation, strength, breakdown) over time.

        NOW PROCESSES BOTH TSLA AND SPY (v3.4)
        Returns DataFrame with 154 columns (77 TSLA + 77 SPY).

        Args:
            df: OHLCV DataFrame
            use_cache: If True, load from cache or save to cache (recommended)
            use_gpu: If True, use GPU acceleration (10-20x faster for large datasets)
            cache_suffix: Optional suffix for cache filename (for testing, e.g., 'GPU_TEST')
        """
        import hashlib
        import pickle

        # Check cache first
        if use_cache:
            cache_dir = Path('data/feature_cache')
            cache_dir.mkdir(exist_ok=True)

            # Create cache key from version + data range (version ensures cache invalidation when logic changes)
            cache_key = f"{FEATURE_VERSION}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}_{len(df)}"

            # Add suffix if provided (for testing GPU/CPU separately)
            if cache_suffix:
                cache_file = cache_dir / f'rolling_channels_{cache_key}_{cache_suffix}.pkl'
            else:
                cache_file = cache_dir / f'rolling_channels_{cache_key}.pkl'

            if cache_file.exists():
                try:
                    # Validate cache file before loading
                    file_size = cache_file.stat().st_size

                    if file_size < 1000:  # Less than 1KB is too small
                        print(f"   ⚠️  Cache file too small ({file_size} bytes), regenerating...")
                        cache_file.unlink()
                    else:
                        # Load cache with progress bar
                        with tqdm(total=1, desc=f"   Loading cache", leave=False, ncols=100, ascii=True, mininterval=0.5) as pbar:
                            with open(cache_file, 'rb') as f:
                                result = pickle.load(f)
                            pbar.update(1)

                        # Validate loaded data
                        expected_cols = 154  # 77 features × 2 stocks (TSLA + SPY)
                        if len(result.columns) != expected_cols:
                            print(f"   ⚠️  Cache has {len(result.columns)} columns, expected {expected_cols}")
                            print(f"   ⚠️  Regenerating cache...")
                        elif len(result) != len(df):
                            print(f"   ⚠️  Cache has {len(result)} rows, expected {len(df)}")
                            print(f"   ⚠️  Regenerating cache...")
                        else:
                            # Cache is valid!
                            print(f"   ✓ Loaded channel features from cache: {cache_file.name}")
                            return result

                except Exception as e:
                    print(f"   ⚠️  Cache load failed ({type(e).__name__}: {e}), regenerating...")
                    if cache_file.exists():
                        cache_file.unlink()

        # No cache - calculate rolling channels
        # Check if multi-resolution data was provided (live mode)
        is_live_mode = multi_res_data is not None

        channel_features = {}
        num_rows = len(df)

        # Resample to different timeframes
        timeframes = {
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': '1h',
            '2h': '2h',
            '3h': '3h',
            '4h': '4h',
            'daily': '1D',
            'weekly': '1W',
            'monthly': '1ME',
            '3month': '3ME'
        }

        # Process both TSLA and SPY
        total_calcs = len(timeframes) * 2  # 11 timeframes × 2 stocks

        # Print status messages with detailed info
        if is_live_mode:
            print(f"   🔄 Extracting channel features in LIVE mode (using multi-resolution data)...")
            print(f"   📊 Processing {total_calcs} calculations (11 timeframes × 2 stocks: SPY + TSLA)")
        else:
            print(f"   🔄 Calculating ROLLING channels (this will take ~30-60 mins first time)...")
            print(f"   📊 Processing {total_calcs} calculations (11 timeframes × 2 stocks: SPY + TSLA)")
            print(f"   ⏱️  Estimated time: ~{total_calcs * 2.5:.0f} minutes")
            print(f"   💡 Results will be cached for instant loading next time")

        # Determine whether to use parallel processing
        n_cores = mp.cpu_count()
        is_live_mode = multi_res_data is not None  # Live mode uses multi-resolution data
        # Check config setting and other conditions
        parallel_enabled = config.PARALLEL_CHANNEL_CALC if hasattr(config, 'PARALLEL_CHANNEL_CALC') else True

        # Simplified: CPU always uses parallel (can be n_jobs=1), GPU/live use sequential
        use_parallel = parallel_enabled and not use_gpu and not is_live_mode

        # Notify user of processing mode and time estimate
        if not use_parallel:
            reasons = []
            if use_gpu:
                reasons.append("GPU mode (requires sequential)")
            if is_live_mode:
                reasons.append("live mode (stability)")
            if not parallel_enabled:
                reasons.append("disabled in config")

            print(f"   ℹ️  Sequential processing: {', '.join(reasons)}")
            print(f"   ⏱️  Using multi-window OHLC channels ({len(config.CHANNEL_WINDOW_SIZES)} windows per timeframe)")
        else:
            cores_to_use = config.MAX_PARALLEL_WORKERS if hasattr(config, 'MAX_PARALLEL_WORKERS') and config.MAX_PARALLEL_WORKERS > 0 else n_cores
            print(f"   🚀 Parallel processing: using {cores_to_use} of {n_cores} available cores")
            print(f"   ⏱️  Multi-window OHLC channels ({len(config.CHANNEL_WINDOW_SIZES)} windows per timeframe)")
            if cores_to_use == 1:
                print(f"   💡 Single-core parallel mode - good for debugging!")

        if use_parallel:
            # ─── MEMORY-EFFICIENT PARALLEL CHANNEL CALCULATION (16GB-SAFE + FAST) ───
            # Extract only price arrays (uses float64 for accuracy)
            tsla_close = df['tsla_close'].values
            spy_close = df['spy_close'].values
            tsla_ohlcv = {
                'open': df['tsla_open'].values,
                'high': df['tsla_high'].values,
                'low': df['tsla_low'].values,
                'close': tsla_close,
                'volume': df['tsla_volume'].values
            }
            spy_ohlcv = {
                'open': df['spy_open'].values,
                'high': df['spy_high'].values,
                'low': df['spy_low'].values,
                'close': spy_close,
                'volume': df['spy_volume'].values
            }

            # Store original timestamps for resampling
            timestamps = df.index.values

            # Build lightweight task list (no unpicklable objects!)
            tasks = []
            for symbol, ohlcv_data in [('tsla', tsla_ohlcv), ('spy', spy_ohlcv)]:
                for tf_name, tf_rule in timeframes.items():
                    tasks.append((
                        ohlcv_data,
                        timestamps,
                        tf_name,
                        tf_rule,
                        symbol
                        # Removed self.channel_calc and self.channel_features_calc
                        # These will be created fresh in each worker process
                    ))

            # Use configured number of cores or all available
            max_workers = config.MAX_PARALLEL_WORKERS if hasattr(config, 'MAX_PARALLEL_WORKERS') else 0
            n_jobs = max_workers if max_workers > 0 else -1

            # Check if we should use multi-progress bars
            use_multi_progress = False
            try:
                from rich.progress import Progress
                use_multi_progress = True
            except ImportError:
                pass

            if use_multi_progress:
                # Use custom parallel extraction with individual progress bars
                from .parallel_channel_extraction import parallel_channel_extraction_with_multi_progress
                print("   🎨 Using multi-progress display (one bar per timeframe)...")
                results = parallel_channel_extraction_with_multi_progress(tasks, n_jobs)
            else:
                # Fallback to standard tqdm with joblib
                results = list(
                    tqdm(
                        Parallel(
                            n_jobs=n_jobs,
                            backend='loky',
                            prefer="processes",
                            verbose=0,
                            return_as="generator"
                        )(delayed(self._compute_channel_memory_efficient)(task) for task in tasks),
                        total=len(tasks),
                        desc="   🔄 Channels",
                        unit="tf",
                        ncols=100,
                        mininterval=0.5,
                        bar_format="{l_bar}{bar:30}{r_bar}  {postfix}"
                    )
                )

            # Merge results back into DataFrame format – ZERO fragmentation
            all_channel_data = {}
            for result_dict in results:
                all_channel_data.update(result_dict)

            # Clear results to free memory (26 GB freed)
            del results
            import gc
            gc.collect()

            channel_features = pd.DataFrame(all_channel_data, index=df.index)

        else:
            # Sequential processing with multi-window (for GPU mode or live mode)
            calc_progress = tqdm(total=total_calcs, desc="   Sequential multi-window channels", ncols=100, leave=False, ascii=True, mininterval=0.5)

            channel_features = {}  # Initialize

            for symbol in ['tsla', 'spy']:
                for tf_name, tf_rule in timeframes.items():
                    # Get data
                    if is_live_mode:
                        if tf_name in ['5min', '15min', '30min']:
                            source_data = multi_res_data['1min']
                        elif tf_name in ['1h', '2h', '3h', '4h']:
                            source_data = multi_res_data['1hour']
                        else:
                            source_data = multi_res_data['daily']
                        symbol_df = source_data[[c for c in source_data.columns if c.startswith(f'{symbol}_')]].copy()
                    else:
                        symbol_df = df[[c for c in df.columns if c.startswith(f'{symbol}_')]].copy()

                    symbol_df.columns = [c.replace(f'{symbol}_', '') for c in symbol_df.columns]

                    # Resample
                    resampled = symbol_df.resample(tf_rule).agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'volume': 'sum'
                    }).dropna()

                    if len(resampled) < 20:
                        calc_progress.update(1)
                        continue

                    # Use multi-window rolling calculation
                    all_windows = self.channel_calc.calculate_multi_window_rolling(resampled, tf_name)

                    # Store features from ALL windows
                    for window, channels_list in all_windows.items():
                        for i, channel in enumerate(channels_list):
                            if channel is None:
                                continue

                            # Map to original timestamps
                            if i < len(resampled):
                                timestamp = resampled.index[i]
                                if i < len(resampled) - 1:
                                    next_timestamp = resampled.index[i + 1]
                                    mask = (df.index >= timestamp) & (df.index < next_timestamp)
                                else:
                                    mask = df.index >= timestamp

                                indices = np.where(mask)[0]
                                current_price = resampled['close'].iloc[i]
                                position_data = self.channel_calc.get_channel_position(current_price, channel)
                                slope_pct = (channel.close_slope / current_price) * 100 if current_price > 0 else 0.0
                                high_slope_pct = (channel.high_slope / current_price) * 100 if current_price > 0 else 0.0
                                low_slope_pct = (channel.low_slope / current_price) * 100 if current_price > 0 else 0.0

                                # Window-specific prefix
                                w_prefix = f'{symbol}_channel_{tf_name}_w{window}'

                                # Initialize arrays if first time
                                if f'{w_prefix}_close_slope' not in channel_features:
                                    for feat in ['position', 'upper_dist', 'lower_dist',
                                                'close_slope', 'close_slope_pct', 'high_slope', 'high_slope_pct',
                                                'low_slope', 'low_slope_pct', 'close_r_squared', 'high_r_squared',
                                                'low_r_squared', 'r_squared_avg', 'channel_width_pct',
                                                'slope_convergence', 'stability', 'ping_pongs',
                                                'ping_pongs_0_5pct', 'ping_pongs_1_0pct', 'ping_pongs_3_0pct',
                                                'is_bull', 'is_bear', 'is_sideways',
                                                'quality_score', 'is_valid', 'insufficient_data', 'duration']:
                                        channel_features[f'{w_prefix}_{feat}'] = np.zeros(num_rows, dtype=config.NUMPY_DTYPE)

                                # Store features (vectorized - no loop for memory efficiency)
                                channel_features[f'{w_prefix}_position'][indices] = position_data['position']
                                channel_features[f'{w_prefix}_upper_dist'][indices] = position_data['distance_to_upper_pct']
                                channel_features[f'{w_prefix}_lower_dist'][indices] = position_data['distance_to_lower_pct']
                                channel_features[f'{w_prefix}_close_slope'][indices] = channel.close_slope
                                channel_features[f'{w_prefix}_close_slope_pct'][indices] = slope_pct
                                channel_features[f'{w_prefix}_high_slope'][indices] = channel.high_slope
                                channel_features[f'{w_prefix}_high_slope_pct'][indices] = high_slope_pct
                                channel_features[f'{w_prefix}_low_slope'][indices] = channel.low_slope
                                channel_features[f'{w_prefix}_low_slope_pct'][indices] = low_slope_pct
                                channel_features[f'{w_prefix}_close_r_squared'][indices] = channel.close_r_squared
                                channel_features[f'{w_prefix}_high_r_squared'][indices] = channel.high_r_squared
                                channel_features[f'{w_prefix}_low_r_squared'][indices] = channel.low_r_squared
                                channel_features[f'{w_prefix}_r_squared_avg'][indices] = channel.r_squared
                                channel_features[f'{w_prefix}_channel_width_pct'][indices] = channel.channel_width_pct
                                channel_features[f'{w_prefix}_slope_convergence'][indices] = channel.slope_convergence
                                channel_features[f'{w_prefix}_stability'][indices] = channel.stability_score
                                channel_features[f'{w_prefix}_ping_pongs'][indices] = channel.ping_pongs
                                channel_features[f'{w_prefix}_ping_pongs_0_5pct'][indices] = channel.ping_pongs_0_5pct
                                channel_features[f'{w_prefix}_ping_pongs_1_0pct'][indices] = channel.ping_pongs_1_0pct
                                channel_features[f'{w_prefix}_ping_pongs_3_0pct'][indices] = channel.ping_pongs_3_0pct
                                channel_features[f'{w_prefix}_is_bull'][indices] = float(slope_pct > 0.1)
                                channel_features[f'{w_prefix}_is_bear'][indices] = float(slope_pct < -0.1)
                                channel_features[f'{w_prefix}_is_sideways'][indices] = float(abs(slope_pct) <= 0.1)
                                channel_features[f'{w_prefix}_quality_score'][indices] = channel.quality_score
                                channel_features[f'{w_prefix}_is_valid'][indices] = channel.is_valid
                                channel_features[f'{w_prefix}_insufficient_data'][indices] = channel.insufficient_data
                                channel_features[f'{w_prefix}_duration'][indices] = channel.actual_duration

                    calc_progress.update(1)

                    # CRITICAL: Clear memory after each timeframe
                    del all_windows
                    if 'resampled' in locals():
                        del resampled
                    if 'symbol_df' in locals():
                        del symbol_df
                    import gc
                    gc.collect()

            calc_progress.close()

        result_df = pd.DataFrame(channel_features, index=df.index)

        # Clear dict now that DataFrame created (frees 26 GB)
        del channel_features
        import gc
        gc.collect()

        # Save to cache (atomic write to prevent corruption)
        if use_cache:
            print(f"   💾 Saving to cache: {cache_file.name}")
            # Write to temp file first (atomic operation)
            temp_file = cache_file.with_suffix('.pkl.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(result_df, f)
            # Atomic rename (ensures no corruption even if interrupted)
            temp_file.rename(cache_file)
            print(f"   ✓ Cache saved successfully")

        return result_df

    def _extract_channel_features_chunked(
        self,
        df: pd.DataFrame,
        multi_res_data: dict = None,
        use_gpu: bool = False,
        chunk_size_years: int = 1
    ) -> pd.DataFrame:
        """
        Extract channel features using chunked processing to save memory.

        Processes data in time-based chunks (e.g., 1 year at a time) with overlap
        to ensure rolling features are calculated correctly at boundaries.

        Args:
            df: Full OHLC dataframe
            multi_res_data: Multi-resolution data dict
            use_gpu: Whether to use GPU acceleration
            chunk_size_years: Size of each chunk in years

        Returns:
            Combined features dataframe
        """
        import gc
        import config

        print(f"\n  📦 Chunked Feature Extraction")
        print(f"     Chunk size: {chunk_size_years} year(s)")
        print(f"     Overlap: {config.CHUNK_OVERLAP_MONTHS} months")

        # Calculate chunk boundaries
        start_date = df.index[0]
        end_date = df.index[-1]
        total_years = (end_date - start_date).days / 365.25

        print(f"     Total period: {start_date.date()} to {end_date.date()} ({total_years:.1f} years)")

        # Create chunk date ranges
        chunk_starts = pd.date_range(start=start_date, end=end_date, freq=f'{chunk_size_years}YS')
        if chunk_starts[-1] < end_date:
            chunk_starts = chunk_starts.append(pd.DatetimeIndex([end_date]))

        chunk_results = []
        overlap_months = config.CHUNK_OVERLAP_MONTHS

        for i in range(len(chunk_starts) - 1):
            chunk_start = chunk_starts[i]
            chunk_end = chunk_starts[i + 1]

            # Add overlap for lookback (except for first chunk)
            if i > 0:
                chunk_start_with_overlap = chunk_start - pd.DateOffset(months=overlap_months)
            else:
                chunk_start_with_overlap = chunk_start

            # Extract chunk
            chunk_df = df[(df.index >= chunk_start_with_overlap) & (df.index < chunk_end)].copy()
            chunk_multi_res = None
            if multi_res_data:
                chunk_multi_res = {
                    tf: mdf[(mdf.index >= chunk_start_with_overlap) & (mdf.index < chunk_end)].copy()
                    for tf, mdf in multi_res_data.items()
                }

            print(f"\n     Chunk {i+1}/{len(chunk_starts)-1}: {chunk_start.date()} to {chunk_end.date()}")
            print(f"       Bars: {len(chunk_df):,} (including {overlap_months}mo overlap)")

            # Process chunk (no cache for individual chunks)
            chunk_features = self._extract_channel_features(
                chunk_df,
                multi_res_data=chunk_multi_res,
                use_cache=False,  # Don't cache individual chunks
                use_gpu=use_gpu,
                cache_suffix=None
            )

            # Remove overlap from results (keep only the actual chunk period)
            chunk_features = chunk_features[chunk_features.index >= chunk_start]

            print(f"       Result: {len(chunk_features):,} bars after trimming overlap")
            print(f"       Memory: ~{chunk_features.memory_usage(deep=True).sum() / 1e6:.1f} MB")

            chunk_results.append(chunk_features)

            # Aggressive memory cleanup
            del chunk_df, chunk_features
            if chunk_multi_res:
                del chunk_multi_res
            gc.collect()

        # Combine all chunks
        print(f"\n     Combining {len(chunk_results)} chunks...")
        combined_df = pd.concat(chunk_results, axis=0)

        # Verify no gaps or overlaps
        assert len(combined_df) == len(combined_df.index.unique()), "Duplicate timestamps found!"
        combined_df = combined_df.sort_index()

        print(f"     ✓ Combined: {len(combined_df):,} bars")
        print(f"     ✓ Date range: {combined_df.index[0].date()} to {combined_df.index[-1].date()}")
        print(f"     ✓ Memory: ~{combined_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

        # Final cleanup
        del chunk_results
        gc.collect()

        return combined_df

    def _compute_channel_memory_efficient(self, args):
        """
        Memory-efficient channel computation using only numpy arrays.
        Processes ~3.7MB of data instead of 1GB+ DataFrames.

        Args:
            args: Tuple of (ohlcv_data, timestamps, tf_name, tf_rule, symbol)

        Returns:
            Dictionary with column names as keys and numpy arrays as values
        """
        import pandas as pd
        import numpy as np
        import sys
        from pathlib import Path

        # Debug: Track worker process
        import os
        pid = os.getpid()

        # Add parent directory to path for imports
        parent_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(parent_dir))

        # Import and create fresh instances in worker process
        from src.linear_regression import LinearRegressionChannel

        # Unpack args (no class instances now!)
        ohlcv_data, timestamps, tf_name, tf_rule, symbol = args

        # Create fresh instances in this worker process
        channel_calc = LinearRegressionChannel()

        try:
            # Create a minimal DataFrame just for resampling
            df_minimal = pd.DataFrame(ohlcv_data, index=pd.DatetimeIndex(timestamps))

            # Resample to target timeframe
            resampled = df_minimal.resample(tf_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            n = len(timestamps)
            prefix = f'{symbol}_channel_{tf_name}'

            # Pre-allocate result arrays
            results = {
                f'{prefix}_position': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_upper_dist': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_lower_dist': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_slope': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_slope_pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_stability': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_ping_pongs': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_ping_pongs_0_5pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_ping_pongs_1_0pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_ping_pongs_3_0pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_r_squared': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_is_bull': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_is_bear': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_is_sideways': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_duration': np.zeros(n, dtype=config.NUMPY_DTYPE)
            }

            # Handle insufficient data case
            if len(resampled) < 20:
                print(f"      [Worker {pid}] {symbol}_{tf_name} - Insufficient data, returning zeros")
                return results  # Return zeros

            # Dynamic lookback calculation
            base_lookback = min(168, len(resampled) // 2)
            total_bars = len(resampled) - base_lookback

            # Process each resampled bar
            for idx, i in enumerate(range(base_lookback, len(resampled))):
                try:
                    # Get available data up to this point
                    available_window = resampled.iloc[:i]

                    # Calculate dynamic lookback based on volatility
                    dynamic_lookback = self._calculate_dynamic_window(
                        available_window['close'],
                        base_window=min(168, len(available_window) // 2),
                        min_window=30,
                        max_window=min(300, len(available_window) // 2)
                    )

                    # Find optimal channel window
                    channel = channel_calc.find_optimal_channel_window(
                        available_window,
                        timeframe=tf_name,
                        max_lookback=dynamic_lookback,
                        min_ping_pongs=3
                    )

                    # Skip if no valid channel
                    if channel is None:
                        continue

                    current_price = resampled['close'].iloc[i]
                    position_data = channel_calc.get_channel_position(current_price, channel)

                    # Get the actual window used
                    actual_window = resampled.iloc[i-channel.actual_duration:i]

                    # Calculate multi-threshold ping-pongs
                    window_prices = actual_window['close'].values
                    multi_pp = channel_calc._detect_ping_pongs_multi_threshold(
                        window_prices,
                        channel.upper_line,
                        channel.lower_line,
                        thresholds=[0.005, 0.01, 0.02, 0.03]
                    )

                    # Map to original timestamps
                    timestamp = resampled.index[i]
                    original_timestamps = pd.DatetimeIndex(timestamps)

                    # Find all original bars that map to this resampled bar
                    if i < len(resampled) - 1:
                        next_timestamp = resampled.index[i + 1]
                        mask = (original_timestamps >= timestamp) & (original_timestamps < next_timestamp)
                    else:
                        mask = original_timestamps >= timestamp

                    # Assign values to all matching timestamps
                    indices = np.where(mask)[0]
                    for idx in indices:
                        results[f'{prefix}_position'][idx] = position_data['position']
                        results[f'{prefix}_upper_dist'][idx] = position_data['distance_to_upper_pct']
                        results[f'{prefix}_lower_dist'][idx] = position_data['distance_to_lower_pct']
                        results[f'{prefix}_slope'][idx] = channel.slope

                        # Normalized slope
                        slope_pct = (channel.slope / current_price) * 100 if current_price > 0 else 0.0
                        results[f'{prefix}_slope_pct'][idx] = slope_pct

                        # Direction flags
                        results[f'{prefix}_is_bull'][idx] = float(slope_pct > 0.1)
                        results[f'{prefix}_is_bear'][idx] = float(slope_pct < -0.1)
                        results[f'{prefix}_is_sideways'][idx] = float(abs(slope_pct) <= 0.1)

                        # Other metrics
                        results[f'{prefix}_stability'][idx] = channel.stability_score if hasattr(channel, 'stability_score') else 0.0
                        results[f'{prefix}_r_squared'][idx] = channel.r_squared
                        results[f'{prefix}_duration'][idx] = channel.actual_duration

                        # Multi-threshold ping-pongs
                        results[f'{prefix}_ping_pongs_0_5pct'][idx] = multi_pp[0.005]
                        results[f'{prefix}_ping_pongs_1_0pct'][idx] = multi_pp[0.01]
                        results[f'{prefix}_ping_pongs'][idx] = multi_pp[0.02]  # Default 2%
                        results[f'{prefix}_ping_pongs_3_0pct'][idx] = multi_pp[0.03]

                except Exception as e:
                    # Skip this timestamp on error
                    continue

            return results

        except Exception as e:
            import traceback
            print(f"      [Worker {pid}] ERROR in {symbol}_{tf_name}: {str(e)}")
            print(f"      [Worker {pid}] Traceback: {traceback.format_exc()}")
            # Return zeros on error
            n = len(timestamps)
            prefix = f'{symbol}_channel_{tf_name}'
            results = {
                f'{prefix}_position': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_upper_dist': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_lower_dist': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_slope': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_slope_pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_stability': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_ping_pongs': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_ping_pongs_0_5pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_ping_pongs_1_0pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_ping_pongs_3_0pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_r_squared': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_is_bull': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_is_bear': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_is_sideways': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_duration': np.zeros(n, dtype=config.NUMPY_DTYPE)
            }
            return results

    def _calculate_single_channel_task(self, args):
        """
        Calculate channels for a single (symbol, timeframe) pair.
        This is the parallelizable unit of work for joblib.

        Args:
            args: Tuple of (symbol, tf_name, tf_rule, symbol_df, lookback, original_index, is_live_mode)

        Returns:
            Tuple of (symbol, tf_name, results_dict)
        """
        symbol, tf_name, tf_rule, symbol_df, lookback, original_index, is_live_mode = args

        # Resample to target timeframe
        resampled = symbol_df.resample(tf_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        prefix = f'{symbol}_channel'
        num_rows = len(original_index)

        # Handle insufficient data
        if len(resampled) < 20:
            # Not enough data - fill with zeros for ALL channel features
            results = {}
            for feat in ['position', 'upper_dist', 'lower_dist', 'slope', 'slope_pct', 'stability',
                         'ping_pongs', 'ping_pongs_0_5pct', 'ping_pongs_1_0pct', 'ping_pongs_3_0pct',
                         'r_squared', 'is_bull', 'is_bear', 'is_sideways', 'duration']:
                results[f'{prefix}_{tf_name}_{feat}'] = np.zeros(num_rows)
            return (symbol, tf_name, results)

        # Calculate rolling channels (CPU only for parallel mode)
        rolling_results = self._calculate_rolling_channels(
            resampled, lookback, tf_name, symbol, original_index,
            show_progress=False  # Disable progress bar in parallel mode
        )

        # Format results
        results = {}
        for feat_name, values in rolling_results.items():
            results[f'{prefix}_{tf_name}_{feat_name}'] = values

        return (symbol, tf_name, results)

    def _calculate_rolling_channels(
        self,
        resampled_df: pd.DataFrame,
        lookback: int,
        tf_name: str,
        symbol: str,
        original_index: pd.DatetimeIndex,
        show_progress: bool = True
    ) -> dict:
        """
        Calculate channels at each timestamp using rolling window.

        Args:
            resampled_df: Resampled OHLCV data at target timeframe
            lookback: Number of bars to look back
            tf_name: Timeframe name
            symbol: 'tsla' or 'spy'
            original_index: Original 1-min index (for alignment)
            show_progress: Whether to show progress bar (False for parallel mode)

        Returns:
            Dictionary with arrays for each channel feature
        """
        num_original_rows = len(original_index)

        # Initialize result arrays (including multi-threshold ping-pongs, normalized slope, direction flags)
        results = {
            'position': np.zeros(num_original_rows),
            'upper_dist': np.zeros(num_original_rows),
            'lower_dist': np.zeros(num_original_rows),
            'slope': np.zeros(num_original_rows),  # Raw slope ($/bar)
            'slope_pct': np.zeros(num_original_rows),  # Normalized slope (% per bar)
            'stability': np.zeros(num_original_rows),
            'ping_pongs': np.zeros(num_original_rows),  # Default 2% threshold
            'ping_pongs_0_5pct': np.zeros(num_original_rows),  # 0.5% threshold
            'ping_pongs_1_0pct': np.zeros(num_original_rows),  # 1.0% threshold
            'ping_pongs_3_0pct': np.zeros(num_original_rows),  # 3.0% threshold
            'r_squared': np.zeros(num_original_rows),
            'is_bull': np.zeros(num_original_rows),  # Uptrending channel (>0.1% per bar)
            'is_bear': np.zeros(num_original_rows),  # Downtrending channel (<-0.1% per bar)
            'is_sideways': np.zeros(num_original_rows),  # Ranging channel (±0.1% per bar)
            'duration': np.zeros(num_original_rows)  # v3.11: Actual bars where channel holds
        }

        # Calculate channel at each timestamp
        bar_range = range(lookback, len(resampled_df))

        # Only show progress bar in sequential mode (removed nested position=2 to avoid terminal corruption)
        if show_progress:
            progress_desc = f"      {symbol} {tf_name}"
            bar_range_iter = tqdm(bar_range, desc=progress_desc, leave=False, ncols=100, ascii=True, mininterval=0.5)
        else:
            bar_range_iter = bar_range

        for i in bar_range_iter:
            try:
                # Get available data up to this point
                available_window = resampled_df.iloc[:i]

                # Calculate dynamic lookback based on volatility
                dynamic_lookback = self._calculate_dynamic_window(
                    available_window['close'],
                    base_window=min(168, len(available_window) // 2),
                    min_window=30,
                    max_window=min(300, len(available_window) // 2)
                )

                # Find optimal channel window (v3.11: dynamic duration detection)
                # Tests multiple lookbacks, requires minimum 3 ping-pongs
                channel = self.channel_calc.find_optimal_channel_window(
                    available_window,
                    timeframe=tf_name,
                    max_lookback=dynamic_lookback,
                    min_ping_pongs=3
                )

                # If no valid channel found, use zeros for all metrics
                if channel is None:
                    # No channel exists (all windows had <3 ping-pongs or poor R²)
                    # Skip this timestamp or use zeros
                    continue

                current_price = resampled_df['close'].iloc[i]
                position_data = self.channel_calc.get_channel_position(current_price, channel)

                # Get the actual window used for this channel
                actual_window = resampled_df.iloc[i-channel.actual_duration:i]

                # Calculate multi-threshold ping-pongs
                window_prices = actual_window['close'].values
                multi_pp = self.channel_calc._detect_ping_pongs_multi_threshold(
                    window_prices,
                    channel.upper_line,
                    channel.lower_line,
                    thresholds=[0.005, 0.01, 0.02, 0.03]
                )

                # Map resampled timestamp to original 1-min index
                timestamp = resampled_df.index[i]

                # Find all 1-min bars that map to this resampled bar
                # (All 1-min bars between this timestamp and next)
                if i < len(resampled_df) - 1:
                    next_timestamp = resampled_df.index[i + 1]
                    mask = (original_index >= timestamp) & (original_index < next_timestamp)
                else:
                    # Last bar
                    mask = original_index >= timestamp

                # Assign channel metrics to all 1-min bars in this window
                results['position'][mask] = position_data['position']
                results['upper_dist'][mask] = position_data['distance_to_upper_pct']
                results['lower_dist'][mask] = position_data['distance_to_lower_pct']
                results['slope'][mask] = channel.slope  # Raw slope ($/bar)

                # Normalized slope (percentage per bar - comparable across timeframes)
                slope_pct = (channel.slope / current_price) * 100 if current_price > 0 else 0.0
                results['slope_pct'][mask] = slope_pct

                # Direction flags (based on normalized slope)
                # Threshold: ±0.1% per bar to distinguish from noise
                results['is_bull'][mask] = float(slope_pct > 0.1)  # Bull: >0.1% per bar
                results['is_bear'][mask] = float(slope_pct < -0.1)  # Bear: <-0.1% per bar
                results['is_sideways'][mask] = float(abs(slope_pct) <= 0.1)  # Sideways: ±0.1% per bar

                results['stability'][mask] = channel.stability_score
                results['ping_pongs'][mask] = channel.ping_pongs  # 2% threshold (default)
                results['ping_pongs_0_5pct'][mask] = multi_pp[0.005]  # 0.5% threshold
                results['ping_pongs_1_0pct'][mask] = multi_pp[0.01]   # 1.0% threshold
                results['ping_pongs_3_0pct'][mask] = multi_pp[0.03]   # 3.0% threshold
                results['r_squared'][mask] = channel.r_squared
                results['duration'][mask] = channel.actual_duration  # v3.11: How many bars channel holds

            except Exception as e:
                # If calculation fails, leave as zeros
                continue

        return results

    def _calculate_ping_pongs_cpu_multi_threshold(
        self,
        prices: np.ndarray,
        pred_prices: np.ndarray,
        residual_std: float,
        thresholds: list = [0.005, 0.01, 0.02, 0.03]
    ) -> dict:
        """
        Ping-pong counting at multiple thresholds (efficient single-pass).

        Args:
            prices: Actual prices for one window
            pred_prices: Predicted prices (regression line) for one window
            residual_std: Standard deviation of residuals
            thresholds: List of percentage thresholds

        Returns:
            Dict mapping threshold to bounce count
        """
        # Calculate channel bounds
        upper = pred_prices + (2.0 * residual_std)
        lower = pred_prices - (2.0 * residual_std)

        results = {threshold: 0 for threshold in thresholds}
        last_touch = {threshold: None for threshold in thresholds}

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper[i]
            lower_val = lower[i]

            # Calculate distances as percentage
            upper_dist = abs(price - upper_val) / upper_val
            lower_dist = abs(price - lower_val) / lower_val

            # Check each threshold
            for threshold in thresholds:
                # Check if price touches upper line
                if upper_dist <= threshold:
                    if last_touch[threshold] == 'lower':
                        results[threshold] += 1
                    last_touch[threshold] = 'upper'

                # Check if price touches lower line
                elif lower_dist <= threshold:
                    if last_touch[threshold] == 'upper':
                        results[threshold] += 1
                    last_touch[threshold] = 'lower'

        return results

    def _linear_regression_gpu(
        self,
        windows: torch.Tensor,
        device: str
    ) -> dict:
        """
        Vectorized linear regression on GPU for all windows simultaneously.

        Args:
            windows: [num_windows, lookback] tensor of price windows
            device: 'cuda' or 'mps'

        Returns:
            Dict with slopes, intercepts, r_squared, predictions (all on GPU)
        """
        num_windows, lookback = windows.shape

        # X values (0, 1, 2, ..., lookback-1) for all windows
        X = torch.arange(lookback, dtype=config.TORCH_DTYPE, device=device).unsqueeze(0)  # [1, lookback]
        X_mean = X.mean()

        # Y values (prices) for each window
        Y_mean = windows.mean(dim=1, keepdim=True)  # [num_windows, 1]

        # Calculate slopes (vectorized for all windows simultaneously)
        numerator = ((X - X_mean) * (windows - Y_mean)).sum(dim=1)  # [num_windows]
        denominator = ((X - X_mean) ** 2).sum()  # scalar
        slopes = numerator / denominator  # [num_windows]

        # Calculate intercepts
        intercepts = Y_mean.squeeze() - slopes * X_mean  # [num_windows]

        # Predictions (regression line)
        y_pred = slopes.unsqueeze(1) * X + intercepts.unsqueeze(1)  # [num_windows, lookback]

        # Calculate R² (vectorized)
        ss_res = ((windows - y_pred) ** 2).sum(dim=1)  # [num_windows]
        ss_tot = ((windows - Y_mean) ** 2).sum(dim=1)  # [num_windows]
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))  # [num_windows]

        # Handle edge cases (constant prices, div by zero)
        r_squared = torch.clamp(r_squared, 0.0, 1.0)
        r_squared = torch.nan_to_num(r_squared, nan=0.0)

        return {
            'slopes': slopes,
            'intercepts': intercepts,
            'r_squared': r_squared,
            'predictions': y_pred
        }

    def _calculate_ping_pongs_cpu(
        self,
        prices: np.ndarray,
        pred_prices: np.ndarray,
        residual_std: float,
        threshold: float = 0.02
    ) -> int:
        """
        Ping-pong counting on CPU (exact match of LinearRegressionChannel._detect_ping_pongs).

        Args:
            prices: Actual prices for one window
            pred_prices: Predicted prices (regression line) for one window
            residual_std: Standard deviation of residuals
            threshold: Percentage threshold for detecting touch (2% default)

        Returns:
            Number of bounces detected
        """
        # Calculate channel bounds
        upper = pred_prices + (2.0 * residual_std)
        lower = pred_prices - (2.0 * residual_std)

        bounces = 0
        last_touch = None

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper[i]
            lower_val = lower[i]

            # Calculate distances as percentage
            upper_dist = abs(price - upper_val) / upper_val
            lower_dist = abs(price - lower_val) / lower_val

            # Check if price touches upper line
            if upper_dist <= threshold:
                if last_touch == 'lower':
                    bounces += 1
                last_touch = 'upper'

            # Check if price touches lower line
            elif lower_dist <= threshold:
                if last_touch == 'upper':
                    bounces += 1
                last_touch = 'lower'

        return bounces

    def _calculate_rolling_channels_gpu(
        self,
        resampled_df: pd.DataFrame,
        lookback: int,
        tf_name: str,
        symbol: str,
        original_index: pd.DatetimeIndex,
        device: str,
        batch_size: int = 10000
    ) -> dict:
        """
        Hybrid GPU+CPU rolling channel calculation for optimal performance.

        Strategy:
        - GPU: Linear regression (vectorized, 80% of computation time) → 15x speedup
        - CPU: Derived metrics (ping-pongs, position, stability) → Exact formula match

        Performance: ~5-6x total speedup (45 mins → 8-10 mins)

        Args:
            resampled_df: Resampled OHLCV data at target timeframe
            lookback: Number of bars to look back
            tf_name: Timeframe name
            symbol: 'tsla' or 'spy'
            original_index: Original 1-min index (for alignment)
            device: 'cuda' or 'mps'
            batch_size: Max windows per GPU batch (default: 10000)

        Returns:
            Dictionary with arrays for each channel feature
        """
        num_original_rows = len(original_index)
        prices = resampled_df['close'].values

        # Initialize result arrays (including multi-threshold ping-pongs, normalized slope, direction flags)
        results = {
            'position': np.zeros(num_original_rows),
            'upper_dist': np.zeros(num_original_rows),
            'lower_dist': np.zeros(num_original_rows),
            'slope': np.zeros(num_original_rows),  # Raw slope ($/bar)
            'slope_pct': np.zeros(num_original_rows),  # Normalized slope (% per bar)
            'stability': np.zeros(num_original_rows),
            'ping_pongs': np.zeros(num_original_rows),  # Default 2% threshold
            'ping_pongs_0_5pct': np.zeros(num_original_rows),  # 0.5% threshold
            'ping_pongs_1_0pct': np.zeros(num_original_rows),  # 1.0% threshold
            'ping_pongs_3_0pct': np.zeros(num_original_rows),  # 3.0% threshold
            'r_squared': np.zeros(num_original_rows),
            'is_bull': np.zeros(num_original_rows),  # Uptrending channel
            'is_bear': np.zeros(num_original_rows),  # Downtrending channel
            'is_sideways': np.zeros(num_original_rows),  # Ranging channel
            'duration': np.zeros(num_original_rows)  # v3.11: Channel duration (fixed lookback for GPU)
        }

        # Convert to PyTorch tensor (dtype from config)
        prices_tensor = torch.tensor(prices, dtype=config.TORCH_DTYPE)

        # Process in batches (to fit in GPU memory)
        num_windows = len(prices) - lookback
        num_batches = (num_windows + batch_size - 1) // batch_size

        # GPU processing
        progress_desc = f"      GPU: {symbol} {tf_name}"
        batch_range = tqdm(range(num_batches), desc=progress_desc, leave=False, ncols=100, ascii=True, mininterval=0.5)
        for batch_idx in batch_range:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_windows)

                # Create rolling windows for this batch
                batch_windows = []
                for i in range(start_idx, end_idx):
                    window = prices_tensor[i:i+lookback]
                    batch_windows.append(window)

                if not batch_windows:
                    continue

                # Stack windows and move to GPU
                windows_batch = torch.stack(batch_windows).to(device)  # [batch, lookback]

                try:
                    # HYBRID APPROACH: GPU for regression (80% of time), CPU for derived metrics (20% of time)

                    # Step 1: Calculate linear regression on GPU (FAST - vectorized)
                    regression_results = self._linear_regression_gpu(windows_batch, device)

                    # Step 2: Move regression results to CPU immediately
                    slopes_cpu = regression_results['slopes'].cpu().numpy()
                    intercepts_cpu = regression_results['intercepts'].cpu().numpy()
                    r_squared_cpu = regression_results['r_squared'].cpu().numpy()
                    predictions_cpu = regression_results['predictions'].cpu().numpy()
                    windows_cpu = windows_batch.cpu().numpy()

                    # Step 3: Calculate derived metrics on CPU (sequential but fast)
                    # This avoids slow GPU→CPU transfers in loops
                    for batch_i, global_i in enumerate(range(start_idx + lookback, end_idx + lookback)):
                        current_price = prices[global_i]
                        pred_price = predictions_cpu[batch_i, -1]

                        # Get actual and predicted prices for this window
                        actual_prices = prices[global_i-lookback:global_i]
                        pred_prices = predictions_cpu[batch_i]

                        # Calculate standard deviation of residuals
                        residuals = actual_prices - pred_prices
                        residual_std = np.std(residuals) if len(residuals) > 0 else 1.0

                        # Calculate ping-pongs at multiple thresholds
                        ping_pongs_multi = self._calculate_ping_pongs_cpu_multi_threshold(
                            actual_prices, pred_prices, residual_std,
                            thresholds=[0.005, 0.01, 0.02, 0.03]
                        )
                        ping_pongs = ping_pongs_multi[0.02]  # Default 2% for stability calc

                        # Calculate channel bounds
                        upper_line = pred_price + (2.0 * residual_std)
                        lower_line = pred_price - (2.0 * residual_std)

                        # Calculate position (exact get_channel_position formula)
                        channel_height = upper_line - lower_line
                        if channel_height > 0:
                            position = (current_price - lower_line) / channel_height
                        else:
                            position = 0.5

                        # Calculate stability (exact _calculate_stability formula)
                        r2_score = r_squared_cpu[batch_i] * 40
                        pp_score = min(ping_pongs / 5.0, 1.0) * 40
                        length_score = min(lookback / 100.0, 1.0) * 20
                        stability = r2_score + pp_score + length_score

                        # Calculate distances (exact get_channel_position formula)
                        upper_dist = ((upper_line - current_price) / current_price) * 100
                        lower_dist = ((current_price - lower_line) / current_price) * 100

                        # Map to original 1-min index
                        timestamp = resampled_df.index[global_i]
                        if global_i < len(resampled_df) - 1:
                            next_timestamp = resampled_df.index[global_i + 1]
                            mask = (original_index >= timestamp) & (original_index < next_timestamp)
                        else:
                            mask = original_index >= timestamp

                        # Store results
                        results['position'][mask] = position
                        results['slope'][mask] = slopes_cpu[batch_i]  # Raw slope ($/bar)

                        # Normalized slope (percentage per bar - comparable across timeframes)
                        slope_pct = (slopes_cpu[batch_i] / current_price) * 100 if current_price > 0 else 0.0
                        results['slope_pct'][mask] = slope_pct

                        # Direction flags (based on normalized slope)
                        results['is_bull'][mask] = float(slope_pct > 0.1)  # Bull: >0.1% per bar
                        results['is_bear'][mask] = float(slope_pct < -0.1)  # Bear: <-0.1% per bar
                        results['is_sideways'][mask] = float(abs(slope_pct) <= 0.1)  # Sideways: ±0.1%

                        results['r_squared'][mask] = r_squared_cpu[batch_i]
                        results['ping_pongs'][mask] = ping_pongs  # 2% threshold
                        results['ping_pongs_0_5pct'][mask] = ping_pongs_multi[0.005]
                        results['ping_pongs_1_0pct'][mask] = ping_pongs_multi[0.01]
                        results['ping_pongs_3_0pct'][mask] = ping_pongs_multi[0.03]
                        results['stability'][mask] = stability
                        results['upper_dist'][mask] = upper_dist
                        results['lower_dist'][mask] = lower_dist
                        results['duration'][mask] = lookback  # v3.11: GPU uses fixed lookback

                    # Clear GPU memory (synchronize first to ensure operations complete)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    elif device == 'mps':
                        # MPS doesn't have synchronize, but manual sync via event
                        torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None

                    del windows_batch, regression_results

                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    elif device == 'mps':
                        torch.mps.empty_cache()

                except Exception as e:
                    print(f"      ⚠️  GPU batch {batch_idx} failed ({e}), skipping...")
                    continue

        return results

    def _extract_rsi_features(self, df: pd.DataFrame, multi_res_data: dict = None) -> pd.DataFrame:
        """
        Extract RSI features for multiple timeframes.
        NOW PROCESSES BOTH TSLA AND SPY (v3.4)
        Returns DataFrame with 66 columns (33 TSLA + 33 SPY).
        Supports HYBRID mode for live predictions (uses multi-resolution data).
        """
        rsi_features = {}
        num_rows = len(df)

        # Check if multi-resolution data was provided (live mode)
        is_live_mode = multi_res_data is not None

        timeframes = {
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': '1h',
            '2h': '2h',
            '3h': '3h',
            '4h': '4h',
            'daily': '1D',
            'weekly': '1W',
            'monthly': '1ME',
            '3month': '3ME'
        }

        total_calcs = len(timeframes) * 2
        with tqdm(total=total_calcs, desc="   RSI features (SPY+TSLA)", ncols=100, leave=False, ascii=True, mininterval=0.5) as pbar:
            # Process both TSLA and SPY
            for symbol in ['tsla', 'spy']:
                for tf_name, tf_rule in timeframes.items():
                    # HYBRID DATA SELECTION: Use appropriate resolution for live mode
                    if is_live_mode:
                        # Live mode: Get data from appropriate resolution
                        if tf_name in ['5min', '15min', '30min']:
                            # Use 1-min data (sufficient history)
                            source_data = multi_res_data['1min']
                        elif tf_name in ['1h', '2h', '3h', '4h']:
                            # Use hourly data (2 years of history)
                            source_data = multi_res_data['1hour']
                        else:  # daily, weekly, monthly, 3month
                            # Use daily data (max history)
                            source_data = multi_res_data['daily']

                        # Extract symbol columns
                        symbol_df = source_data[[c for c in source_data.columns if c.startswith(f'{symbol}_')]].copy()
                    else:
                        # Training mode: Use input dataframe
                        symbol_df = df[[c for c in df.columns if c.startswith(f'{symbol}_')]].copy()

                    symbol_df.columns = [c.replace(f'{symbol}_', '') for c in symbol_df.columns]

                    # Resample to target timeframe
                    resampled = symbol_df.resample(tf_rule).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()

                    prefix = f'{symbol}_rsi'

                    if len(resampled) < 20:
                        # Not enough data - fill with defaults
                        rsi_features[f'{prefix}_{tf_name}'] = np.full(num_rows, 50.0)
                        rsi_features[f'{prefix}_{tf_name}_oversold'] = np.zeros(num_rows)
                        rsi_features[f'{prefix}_{tf_name}_overbought'] = np.zeros(num_rows)
                        pbar.update(1)
                        continue

                    try:
                        rsi_data = self.rsi_calc.get_rsi_data(resampled)
                        rsi_value = rsi_data.value if rsi_data.value is not None else 50.0

                        # Store features (broadcast scalar to all rows)
                        rsi_features[f'{prefix}_{tf_name}'] = np.full(num_rows, rsi_value)
                        rsi_features[f'{prefix}_{tf_name}_oversold'] = np.full(num_rows, 1.0 if rsi_data.oversold else 0.0)
                        rsi_features[f'{prefix}_{tf_name}_overbought'] = np.full(num_rows, 1.0 if rsi_data.overbought else 0.0)

                    except Exception:
                        # Fill with defaults
                        rsi_features[f'{prefix}_{tf_name}'] = np.full(num_rows, 50.0)
                        rsi_features[f'{prefix}_{tf_name}_oversold'] = np.zeros(num_rows)
                        rsi_features[f'{prefix}_{tf_name}_overbought'] = np.zeros(num_rows)
                    
                    pbar.update(1)

        return pd.DataFrame(rsi_features, index=df.index)

    def _extract_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract SPY-TSLA correlation and divergence features. Returns DataFrame with 5 columns."""
        spy_returns = df['spy_close'].pct_change()
        tsla_returns = df['tsla_close'].pct_change()

        correlation_features = {
            'correlation_10': spy_returns.rolling(10).corr(tsla_returns),
            'correlation_50': spy_returns.rolling(50).corr(tsla_returns),
            'correlation_200': spy_returns.rolling(200).corr(tsla_returns),
            'divergence': (((spy_returns > 0) & (tsla_returns < 0)) |
                          ((spy_returns < 0) & (tsla_returns > 0))).astype(float),
            'divergence_magnitude': abs(spy_returns - tsla_returns)
        }

        return pd.DataFrame(correlation_features, index=df.index)

    def _extract_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract larger cycle features (52-week highs/lows, mega channels). Returns DataFrame with 4 columns."""
        tsla_close = df['tsla_close']
        high_52w = tsla_close.rolling(252, min_periods=50).max()
        low_52w = tsla_close.rolling(252, min_periods=50).min()

        cycle_features = {
            'distance_from_52w_high': (high_52w - tsla_close) / high_52w,
            'distance_from_52w_low': (tsla_close - low_52w) / low_52w
        }

        # Mega channel: 3-4 year channels
        if len(df) > 756:  # 3 years of daily data (approx)
            lookback = min(1008, len(df))
            mega_high = tsla_close.rolling(lookback, min_periods=252).max()
            mega_low = tsla_close.rolling(lookback, min_periods=252).min()

            cycle_features['within_mega_channel'] = ((tsla_close >= mega_low * 0.9) &
                                                      (tsla_close <= mega_high * 1.1)).astype(float)
            cycle_features['mega_channel_position'] = (tsla_close - mega_low) / (mega_high - mega_low + 1e-8)
        else:
            cycle_features['within_mega_channel'] = np.zeros(len(df))
            cycle_features['mega_channel_position'] = np.full(len(df), 0.5)

        return pd.DataFrame(cycle_features, index=df.index)

    def _extract_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volume-based features. Returns DataFrame with 2 columns."""
        volume_features = {
            'tsla_volume_ratio': df['tsla_volume'] / df['tsla_volume'].rolling(20, min_periods=1).mean(),
            'spy_volume_ratio': df['spy_volume'] / df['spy_volume'].rolling(20, min_periods=1).mean()
        }

        return pd.DataFrame(volume_features, index=df.index)

    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features. Returns DataFrame with 4 columns."""
        time_features = {
            'hour_of_day': df.index.hour / 24.0,
            'day_of_week': df.index.dayofweek / 7.0,
            'day_of_month': df.index.day / 31.0,
            'month_of_year': df.index.month / 12.0
        }

        return pd.DataFrame(time_features, index=df.index)

    def _extract_breakdown_features(
        self,
        features_df: pd.DataFrame,  # Base features already extracted
        raw_df: pd.DataFrame,       # Original OHLCV data (for index.dayofweek)
        events_handler=None          # Optional event handler for earnings/FOMC features (v3.9)
    ) -> pd.DataFrame:
        """
        Extract channel breakdown and enhancement features. Returns DataFrame with 68 columns (v3.9: +4 event features = 72).

        Args:
            features_df: Base features DataFrame
            raw_df: Original OHLCV DataFrame (for timestamp index)
            events_handler: Optional CombinedEventsHandler for event-driven features

        Features:
        - Volume surge indicator
        - RSI divergence from channel position
        - Channel duration vs historical average
        - SPY-TSLA channel alignment
        - Time in channel (bars since last break)
        - Normalized channel position (-1 to +1)
        - Binary flags (day of week, volatility, in_channel)
        - Event proximity (v3.9): is_earnings_week, days_until_earnings, days_until_fomc, is_high_impact_event
        """
        breakdown_features = {}
        num_rows = len(features_df)

        # Extract necessary data from RAW df (OHLCV)
        tsla_prices = raw_df['tsla_close'].values
        spy_prices = raw_df['spy_close'].values
        tsla_volume = raw_df['tsla_volume'].values if 'tsla_volume' in raw_df.columns else np.zeros(num_rows)

        # 1. Volume surge (recent vs historical)
        if len(tsla_volume) >= 60:
            recent_vol = pd.Series(tsla_volume).rolling(10, min_periods=1).mean()
            historical_vol = pd.Series(tsla_volume).rolling(60, min_periods=10).mean().shift(10)
            volume_surge = ((recent_vol - historical_vol) / (historical_vol + 1e-8)).fillna(0)
            breakdown_features['tsla_volume_surge'] = volume_surge.values
        else:
            breakdown_features['tsla_volume_surge'] = np.zeros(num_rows)

        # 2. RSI divergence from channel position (4 timeframes)
        for tf_name in ['15min', '1h', '4h', 'daily']:
            # Get RSI value from already calculated base features
            rsi_col = f'tsla_rsi_{tf_name}'
            channel_pos_col = f'tsla_channel_{tf_name}_position'

            if rsi_col in features_df.columns and channel_pos_col in features_df.columns:
                rsi_normalized = features_df[rsi_col] / 100.0  # 0-1 range
                channel_pos = features_df[channel_pos_col]  # 0-1 range

                # Divergence: High RSI + low position = potential reversal
                divergence = rsi_normalized - channel_pos
                breakdown_features[f'tsla_rsi_divergence_{tf_name}'] = divergence.values
            else:
                breakdown_features[f'tsla_rsi_divergence_{tf_name}'] = np.zeros(num_rows)

        # 3. Channel duration vs historical average (3 timeframes)
        # Use stability score as proxy for channel duration
        for tf_name in ['1h', '4h', 'daily']:
            stability_col = f'tsla_channel_{tf_name}_stability'

            if stability_col in features_df.columns:
                stability = features_df[stability_col]
                # Duration ratio = current stability vs rolling average
                avg_stability = stability.rolling(50, min_periods=10).mean()
                duration_ratio = (stability / (avg_stability + 1e-8)).fillna(1.0)
                breakdown_features[f'tsla_channel_duration_ratio_{tf_name}'] = duration_ratio.values
            else:
                breakdown_features[f'tsla_channel_duration_ratio_{tf_name}'] = np.ones(num_rows)

        # 4. SPY-TSLA channel alignment (2 timeframes)
        for tf_name in ['1h', '4h']:
            tsla_pos_col = f'tsla_channel_{tf_name}_position'
            spy_pos_col = f'spy_channel_{tf_name}_position'

            if tsla_pos_col in features_df.columns and spy_pos_col in features_df.columns:
                tsla_pos = features_df[tsla_pos_col] * 2 - 1  # Convert 0-1 to -1 to +1
                spy_pos = features_df[spy_pos_col] * 2 - 1

                # Alignment: both at top (+1) or both at bottom (-1) = high alignment
                alignment = tsla_pos * spy_pos  # -1 to +1
                breakdown_features[f'channel_alignment_spy_tsla_{tf_name}'] = alignment.values
            else:
                breakdown_features[f'channel_alignment_spy_tsla_{tf_name}'] = np.zeros(num_rows)

        # 5. Time in channel features (11 timeframes × 2 stocks)
        # Use channel stability as proxy (higher stability = longer time in channel)
        for tf_name in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            for symbol in ['tsla', 'spy']:
                stability_col = f'{symbol}_channel_{tf_name}_stability'

                if stability_col in features_df.columns:
                    # Normalize stability to represent "time in channel" score
                    stability = features_df[stability_col]
                    time_in_channel = np.clip(stability * 100, 0, 100)  # 0-100 scale
                    breakdown_features[f'{symbol}_time_in_channel_{tf_name}'] = time_in_channel.values
                else:
                    breakdown_features[f'{symbol}_time_in_channel_{tf_name}'] = np.zeros(num_rows)

        # 6. Enhanced normalized channel position (11 timeframes × 2 stocks)
        for tf_name in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            for symbol in ['tsla', 'spy']:
                pos_col = f'{symbol}_channel_{tf_name}_position'

                if pos_col in features_df.columns:
                    # Convert 0-1 position to -1 to +1 (bottom to top)
                    position_norm = features_df[pos_col] * 2 - 1
                    breakdown_features[f'{symbol}_channel_position_norm_{tf_name}'] = position_norm.values
                else:
                    breakdown_features[f'{symbol}_channel_position_norm_{tf_name}'] = np.zeros(num_rows)

        # 7. Binary feature flags (NO LEAKAGE - past data only)

        # Day of week flags (known at prediction time - OK) - Use raw_df index
        breakdown_features['is_monday'] = (raw_df.index.dayofweek == 0).astype(float)
        breakdown_features['is_tuesday'] = (raw_df.index.dayofweek == 1).astype(float)  # v3.11
        breakdown_features['is_wednesday'] = (raw_df.index.dayofweek == 2).astype(float)  # v3.11
        breakdown_features['is_thursday'] = (raw_df.index.dayofweek == 3).astype(float)  # v3.11
        breakdown_features['is_friday'] = (raw_df.index.dayofweek == 4).astype(float)

        # Market timing flags (v3.11) - First/last hour effects
        hour = raw_df.index.hour
        breakdown_features['is_first_hour'] = ((hour >= 9) & (hour < 11)).astype(float)  # 9:30-10:30 ET open
        breakdown_features['is_last_hour'] = ((hour >= 15) & (hour < 16)).astype(float)  # 15:00-16:00 ET power hour

        # Volatility regime (uses PAST volatility only - NO LEAKAGE) - Use features_df
        current_vol_10 = features_df['tsla_volatility_10']  # From base features
        historical_avg_vol = current_vol_10.rolling(200, min_periods=20).mean()  # Past 200 bars

        breakdown_features['is_volatile_now'] = (
            current_vol_10 > historical_avg_vol * 1.5
        ).fillna(0).astype(float)

        # In channel binary flags (for key timeframes)
        # Note: time_in_channel features were just created above in section #5
        for tf_name in ['1h', '4h', 'daily']:
            for symbol in ['tsla', 'spy']:
                time_col = f'{symbol}_time_in_channel_{tf_name}'

                # Check if we just created this feature (it's in breakdown_features dict)
                if time_col in breakdown_features:
                    # Binary: in channel if time_in_channel > 5 bars
                    in_channel = (breakdown_features[time_col] > 5).astype(float)
                    breakdown_features[f'{symbol}_in_channel_{tf_name}'] = in_channel
                else:
                    # Fallback to zeros if not found
                    breakdown_features[f'{symbol}_in_channel_{tf_name}'] = np.zeros(num_rows)

        # Event proximity features (v3.9) - scheduled dates are public (NO LEAKAGE)
        if events_handler is not None:
            # Initialize event feature arrays
            is_earnings_week = np.zeros(num_rows)
            days_until_earnings = np.zeros(num_rows)
            days_until_fomc = np.zeros(num_rows)
            is_high_impact_event = np.zeros(num_rows)

            # Extract events for each timestamp
            event_pbar = tqdm(range(num_rows), desc="      Event features", leave=False, ncols=100, ascii=True, mininterval=0.5)
            for idx in event_pbar:
                timestamp = raw_df.index[idx]
                date_str = timestamp.strftime('%Y-%m-%d')

                try:
                    # Get events within configured window (±14 days by default)
                    import config as cfg
                    events = events_handler.get_events_for_date(date_str, lookback_days=cfg.EVENT_LOOKBACK_DAYS)

                    if events:
                        # Find closest earnings event
                        earnings_events = [e for e in events if e['event_type'] in ['earnings', 'delivery']]
                        if earnings_events:
                            # Get closest earnings
                            closest_earnings = min(earnings_events, key=lambda e: abs(e['days_until']))
                            days_until_earnings[idx] = closest_earnings['days_until']
                            is_earnings_week[idx] = float(abs(closest_earnings['days_until']) <= cfg.EVENT_LOOKBACK_DAYS)

                        # Find closest FOMC event
                        fomc_events = [e for e in events if e['event_type'] == 'fomc']
                        if fomc_events:
                            closest_fomc = min(fomc_events, key=lambda e: abs(e['days_until']))
                            days_until_fomc[idx] = closest_fomc['days_until']

                        # High impact = earnings/FOMC within 3 days
                        high_impact_events = [e for e in events
                                             if e['event_type'] in ['earnings', 'fomc', 'delivery']
                                             and abs(e['days_until']) <= 3]
                        is_high_impact_event[idx] = float(len(high_impact_events) > 0)

                except Exception:
                    # If event lookup fails, leave as zeros
                    continue

            # Store event features
            breakdown_features['is_earnings_week'] = is_earnings_week
            breakdown_features['days_until_earnings'] = days_until_earnings
            breakdown_features['days_until_fomc'] = days_until_fomc
            breakdown_features['is_high_impact_event'] = is_high_impact_event
        else:
            # Backward compatibility: If no events_handler, use zeros
            breakdown_features['is_earnings_week'] = np.zeros(num_rows)
            breakdown_features['days_until_earnings'] = np.zeros(num_rows)
            breakdown_features['days_until_fomc'] = np.zeros(num_rows)
            breakdown_features['is_high_impact_event'] = np.zeros(num_rows)

        # Debug: Check breakdown feature count
        num_breakdown = len(breakdown_features)
        expected_breakdown = 72  # 60 original + 12 total flags/events (v3.11)
        if num_breakdown != expected_breakdown:
            print(f"   ⚠️  Breakdown features: {num_breakdown} (expected {expected_breakdown})")
            print(f"   Missing/Extra: {num_breakdown - expected_breakdown} features")
        else:
            print(f"   ✓ Breakdown features: {num_breakdown} (correct)")

        return pd.DataFrame(breakdown_features, index=raw_df.index)

    def test_continuation_labels(self, df: pd.DataFrame) -> None:
        """Test continuation label generation on sample data."""
        print("Testing continuation label generation...")

        # Sample timestamps
        timestamps = df.index[:10]  # First 10 timestamps

        # Generate labels
        labels_df = self.generate_continuation_labels(df, timestamps, prediction_horizon=24)
        return labels_df

        print(f"Generated {len(labels_df)} continuation labels")
        if not labels_df.empty:
            print("Sample labels:")
            for _, row in labels_df.head(3).iterrows():
                print(f"  {row['timestamp']}: {row['label']}")

        print("✓ Continuation label test completed")

    def validate_continuation_data_availability(self, df: pd.DataFrame, timestamps: list) -> dict:
        """
        Validate how many timestamps have sufficient data for continuation analysis.

        Returns dict with validation results.
        """
        validation_results = {
            'total_timestamps': len(timestamps),
            'sufficient_raw_data': 0,
            'insufficient_raw_data': 0,
            'data_distribution': []
        }

        print("  🔍 Validating continuation data availability...")

        with tqdm(total=len(timestamps), desc="    Data validation",
                  unit="timestamps", ncols=100, leave=False, mininterval=0.5) as pbar:

            for i, ts in enumerate(timestamps):
                try:
                    current_idx = df.index.get_loc(ts)

                    # Check raw data availability with config values
                    one_h_available = min(config.CONTINUATION_LOOKBACK_1H, current_idx)  # How much 1h data we could get
                    four_h_available = min(config.CONTINUATION_LOOKBACK_4H, current_idx)  # How much 4h data we could get

                    # Require at least 120 bars for 1h and 480 for 4h (minimum for basic analysis)
                    has_sufficient = (one_h_available >= 120 and four_h_available >= 480)

                    if has_sufficient:
                        validation_results['sufficient_raw_data'] += 1
                    else:
                        validation_results['insufficient_raw_data'] += 1

                    # Track distribution for analysis (sample every 100th for memory efficiency)
                    if i % 100 == 0:
                        validation_results['data_distribution'].append({
                            'timestamp': ts,
                            'available_1h': one_h_available,
                            'available_4h': four_h_available,
                            'sufficient': has_sufficient
                        })

                except Exception as e:
                    validation_results['insufficient_raw_data'] += 1

                pbar.update(1)

        # Summary
        sufficient_pct = (validation_results['sufficient_raw_data'] / validation_results['total_timestamps']) * 100
        print(f"  ✅ Data validation complete:")
        print(f"     Sufficient data: {validation_results['sufficient_raw_data']}/{validation_results['total_timestamps']} ({sufficient_pct:.1f}%)")
        print(f"     Insufficient data: {validation_results['insufficient_raw_data']}/{validation_results['total_timestamps']}")

        return validation_results

    def generate_continuation_labels(self, df: pd.DataFrame, timestamps: list, prediction_horizon: int = 24, debug: bool = False) -> pd.DataFrame:
        """
        Generate continuation prediction labels using multi-timeframe analysis.

        Implements the pseudo-code logic:
        - Pull 1h and 4h OHLC chunks
        - Calculate RSI for both timeframes
        - Check slope alignment
        - Apply continuation scoring
        - Look ahead for actual duration/gain

        Args:
            df: Full OHLC DataFrame (5-min bars)
            timestamps: List of timestamps to process
            prediction_horizon: How many bars ahead to look for continuation
            debug: Enable debug logging for troubleshooting

        Returns:
            DataFrame with continuation labels
        """
        labels = []
        skip_reasons = {
            'insufficient_raw_data': 0,
            'insufficient_resampled_data': 0,
            'channel_fit_failed': 0,
            'scoring_failed': 0,
            'other_errors': 0
        }

        # Progress bar for continuation label generation
        with tqdm(total=len(timestamps), desc="   Continuation labels",
                  unit="timestamps", ncols=100, leave=False, ascii=True, mininterval=0.5) as pbar:

            for i, ts in enumerate(timestamps):
                try:
                    # Get current index and price
                    current_idx = df.index.get_loc(ts)
                    current_price = df.loc[ts, 'tsla_close']

                    # Step 1: Pull multi-timeframe OHLC chunks
                    # 1h chunk: Use config value (default 1512 bars = ~3 months)
                    one_h_lookback = min(current_idx, config.CONTINUATION_LOOKBACK_1H)
                    one_h_start = max(0, current_idx - one_h_lookback)
                    one_h_chunk = df.iloc[one_h_start:current_idx+1].copy()

                    # 4h chunk: Use config value (default 6048 bars = ~1 year)
                    four_h_lookback = min(current_idx, config.CONTINUATION_LOOKBACK_4H)
                    four_h_start = max(0, current_idx - four_h_lookback)
                    four_h_chunk = df.iloc[four_h_start:current_idx+1].copy()

                    # DEBUG: Log data availability for first few timestamps
                    if debug and i < 5:
                        print(f"  TS {ts}: 1h_chunk={len(one_h_chunk)}, 4h_chunk={len(four_h_chunk)}")

                    # Resample to 1h and 4h timeframes
                    one_h_ohlc = one_h_chunk.resample('1h').agg({
                        'tsla_open': 'first',
                        'tsla_high': 'max',
                        'tsla_low': 'min',
                        'tsla_close': 'last'
                    }).dropna()

                    four_h_ohlc = four_h_chunk.resample('4h').agg({
                        'tsla_open': 'first',
                        'tsla_high': 'max',
                        'tsla_low': 'min',
                        'tsla_close': 'last'
                    }).dropna()

                    # Rename columns to generic OHLC names IMMEDIATELY after resampling
                    # RSI and channel calculators expect 'close', not 'tsla_close'
                    one_h_ohlc.columns = [c.replace('tsla_', '') for c in one_h_ohlc.columns]
                    four_h_ohlc.columns = [c.replace('tsla_', '') for c in four_h_ohlc.columns]

                    # DEBUG: Verify column renaming worked
                    if debug and i < 3:
                        print(f"  DEBUG: 1h columns after rename: {one_h_ohlc.columns.tolist()}")
                        print(f"  DEBUG: 4h columns after rename: {four_h_ohlc.columns.tolist()}")

                    if len(one_h_ohlc) < 3 or len(four_h_ohlc) < 2:
                        skip_reasons['insufficient_resampled_data'] += 1
                        if debug and skip_reasons['insufficient_resampled_data'] < 3:
                            print(f"  SKIP {ts}: Insufficient resampled data ({len(one_h_ohlc)} 1h, {len(four_h_ohlc)} 4h)")
                        pbar.update(1)
                        continue

                    # Step 2: Compute RSI for both timeframes
                    rsi_1h = self.rsi_calc.get_rsi_data(one_h_ohlc).value or 50.0
                    rsi_4h = self.rsi_calc.get_rsi_data(four_h_ohlc).value or 50.0

                    # Step 3: Fit channels and get slopes (allow using more data)
                    channel_1h = self.channel_calc.find_optimal_channel_window(
                        one_h_ohlc, timeframe='1h', max_lookback=min(60, max(5, len(one_h_ohlc)-2)), min_ping_pongs=2
                    )

                    channel_4h = self.channel_calc.find_optimal_channel_window(
                        four_h_ohlc, timeframe='4h', max_lookback=min(120, max(10, len(four_h_ohlc)-2)), min_ping_pongs=2
                    )

                    # Get slopes (default to 0 if no channel found)
                    slope_1h = channel_1h.slope if channel_1h else 0.0
                    slope_4h = channel_4h.slope if channel_4h else 0.0

                    # Check if channel fitting failed
                    if channel_1h is None or channel_4h is None:
                        skip_reasons['channel_fit_failed'] += 1
                        if debug and skip_reasons['channel_fit_failed'] < 3:
                            print(f"  SKIP {ts}: Channel fitting failed")
                        pbar.update(1)
                        continue

                    # Step 4: Apply continuation scoring logic
                    score = 0

                    # +1 for low RSI on short frame (room to run upward)
                    if rsi_1h < 40:
                        score += 1

                    # +1 for low RSI on long frame (broader support)
                    if rsi_4h < 40:
                        score += 1

                    # +1 if slopes align (both bull or both bear)
                    slope_1h_direction = 1 if slope_1h > 0.0001 else (-1 if slope_1h < -0.0001 else 0)
                    slope_4h_direction = 1 if slope_4h > 0.0001 else (-1 if slope_4h < -0.0001 else 0)

                    if slope_1h_direction == slope_4h_direction and slope_1h_direction != 0:
                        score += 1
                    elif slope_1h_direction != slope_4h_direction and slope_1h_direction != 0 and slope_4h_direction != 0:
                        # Conflict: e.g., 1h bull vs 4h bear
                        score -= 1

                    # -1 if overbought on higher frame (break likely)
                    if rsi_4h > 70:
                        score -= 1

                    # Step 5: Look ahead for actual duration/gain
                    future_end = min(current_idx + prediction_horizon, len(df) - 1)
                    future_prices = df.iloc[current_idx:future_end+1]['tsla_close'].values

                    if len(future_prices) < 2:
                        pbar.update(1)
                        continue

                    # Calculate actual continuation metrics
                    future_high = np.max(future_prices)
                    future_low = np.min(future_prices)
                    max_gain = (future_high - current_price) / current_price * 100

                    # Find break point (first time price moves >2% from entry)
                    break_threshold = current_price * 1.02  # +2%
                    break_indices = np.where(future_prices >= break_threshold)[0]

                    if len(break_indices) > 0:
                        break_idx = break_indices[0]
                        actual_duration_hours = break_idx * 5 / 60  # 5-min bars to hours
                        continues = True
                        label = f"continues {actual_duration_hours:.1f}h, +{max_gain:.1f}%"
                    else:
                        # No break within horizon
                        actual_duration_hours = len(future_prices) * 5 / 60
                        continues = True
                        label = f"continues {actual_duration_hours:.1f}h, +{max_gain:.1f}%"

                    # If score <= 1, mark as breaking soon regardless
                    if score <= 1:
                        skip_reasons['scoring_failed'] += 1
                        continues = False
                        early_break_hours = min(actual_duration_hours * 0.5, 1.0)  # Assume breaks in half the time
                        label = f"breaks in {early_break_hours:.1f}h"

                        if debug and skip_reasons['scoring_failed'] < 3:
                            print(f"  SKIP {ts}: Scoring failed (score={score})")
                        pbar.update(1)
                        continue

                    # Calculate confidence based on score
                    confidence = min(max(abs(score) * 0.2, 0.1), 0.9)

                    labels.append({
                        'timestamp': ts,
                        'label': label,
                        'continues': float(continues),
                        'duration_hours': actual_duration_hours,
                        'projected_gain': max_gain if continues else 0.0,
                        'confidence': confidence,
                        'score': score,
                        'rsi_1h': rsi_1h,
                        'rsi_4h': rsi_4h,
                        'slope_1h': slope_1h,
                        'slope_4h': slope_4h
                    })

                    # Update progress bar with current stats
                    processed = len(labels)
                    success_rate = processed / (i + 1) * 100
                    pbar.set_postfix({
                        'processed': f'{processed}/{len(timestamps)}',
                        'success_rate': f'{success_rate:.1f}%'
                    })

                except Exception as e:
                    skip_reasons['other_errors'] += 1
                    if debug and skip_reasons['other_errors'] < 3:
                        print(f"  ERROR {ts}: {str(e)}")
                    # Still update progress on errors
                    processed = len(labels)
                    success_rate = processed / (i + 1) * 100 if i > 0 else 0
                    pbar.set_postfix({
                        'processed': f'{processed}/{len(timestamps)}',
                        'success_rate': f'{success_rate:.1f}%'
                    })

                pbar.update(1)

        # Final debug summary
        if debug:
            total_skipped = sum(skip_reasons.values())
            print(f"  DEBUG: Generated {len(labels)} labels, skipped {total_skipped}")
            for reason, count in skip_reasons.items():
                if count > 0:
                    print(f"    {reason}: {count}")

        return pd.DataFrame(labels)

        return pd.DataFrame(labels)

    def create_sequences(self, features_df: pd.DataFrame, sequence_length: int = 168,
                        target_horizon: int = 24) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sequences for training
        Returns (X, y) where:
        - X: (num_sequences, sequence_length, num_features)
        - y: (num_sequences, 2) - [predicted_high, predicted_low] for next target_horizon bars
        """
        features = features_df.values
        num_samples = len(features) - sequence_length - target_horizon

        if num_samples <= 0:
            raise ValueError(f"Not enough data. Need at least {sequence_length + target_horizon} bars")

        X = []
        y = []

        for i in range(num_samples):
            # Input sequence
            seq = features[i:i + sequence_length]
            X.append(seq)

            # Target: high and low in next target_horizon bars
            future_window = features_df.iloc[i + sequence_length:i + sequence_length + target_horizon]
            target_high = future_window['tsla_close'].max()
            target_low = future_window['tsla_close'].min()
            y.append([target_high, target_low])

        X = torch.tensor(np.array(X), dtype=config.TORCH_DTYPE)
        y = torch.tensor(np.array(y), dtype=config.TORCH_DTYPE)

        return X, y
