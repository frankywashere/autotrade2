"""
Feature extraction system for ML model
Leverages existing Stage 1 components (channels, RSI) plus new features
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import concurrent.futures
from collections import defaultdict
import gc

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config
from src.linear_regression import LinearRegressionChannel
from src.rsi_calculator import RSICalculator
from .base import FeatureExtractor

# GPU acceleration (CUDA-only, gated by device selection)
# Only enable if CUDA is actually available - MPS uses CPU path
try:
    from .gpu_rolling import CUDARollingStats
    GPU_ROLLING_AVAILABLE = torch.cuda.is_available()  # CUDA only, not MPS
except ImportError:
    GPU_ROLLING_AVAILABLE = False

# Feature cache version - increment when calculation logic changes
VIX_CALC_VERSION = "v1"  # v4.4: Track VIX feature calculation version (increment if VIX logic changes)
EVENTS_CALC_VERSION = "v1"  # v4.4: Track events calculation version
CHANNEL_PROJECTION_VERSION = "v1"  # v5.0: Track channel projection features
BREAKDOWN_CALC_VERSION = "v2"  # v5.3.3: Track breakdown calculation method (v1=1-min, v2=native TF)
FEATURE_VERSION = f"v5.3.3_vix{VIX_CALC_VERSION}_ev{EVENTS_CALC_VERSION}_proj{CHANNEL_PROJECTION_VERSION}_bd{BREAKDOWN_CALC_VERSION}"
# v5.3.3: Native TF breakdown (calculated AFTER resampling, not before) + adaptive windows corrected + yfinance limits

# v4.1: Native timeframe sequence lengths for hierarchical model
# IMPORTED FROM config.py (single source of truth)
# Each layer sees enough bars at its native resolution to learn channel patterns
TIMEFRAME_SEQUENCE_LENGTHS = config.TIMEFRAME_SEQUENCE_LENGTHS

# Timeframe resample rules for pandas
TIMEFRAME_RESAMPLE_RULES = {
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
    '3month': '3ME',
}

# All timeframes in order (matches hierarchical model layers)
HIERARCHICAL_TIMEFRAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']


def load_vix_data(csv_path: str = "data/VIX_History.csv") -> pd.DataFrame:
    """
    Load VIX historical data from CSV file.

    Args:
        csv_path: Path to VIX_History.csv file (default: data/VIX_History.csv)

    Returns:
        DataFrame with columns: vix_open, vix_high, vix_low, vix_close
        Index: DatetimeIndex (date only, no timezone)

    Expected CSV format:
        DATE,OPEN,HIGH,LOW,CLOSE
        01/02/1990,17.24,17.24,17.24,17.24
        ...

    Usage in training:
        vix_data = load_vix_data()
        features_df, _ = extractor.extract_features(df, vix_data=vix_data)
    """
    vix_df = pd.read_csv(csv_path)

    # Parse date (format: MM/DD/YYYY)
    vix_df['DATE'] = pd.to_datetime(vix_df['DATE'], format='%m/%d/%Y')
    vix_df = vix_df.set_index('DATE')

    # Rename columns to lowercase with vix_ prefix
    vix_df = vix_df.rename(columns={
        'OPEN': 'vix_open',
        'HIGH': 'vix_high',
        'LOW': 'vix_low',
        'CLOSE': 'vix_close'
    })

    # Sort by date
    vix_df = vix_df.sort_index()

    print(f"   📊 Loaded VIX data: {len(vix_df):,} rows ({vix_df.index[0].date()} to {vix_df.index[-1].date()})")
    return vix_df


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


def get_safe_worker_count(requested_workers: int = None, container_ram_gb: float = None) -> int:
    """
    Calculate safe number of parallel workers based on available RAM.
    Each worker uses ~15GB during feature extraction.

    Args:
        requested_workers: User-requested worker count (None = auto-detect)
        container_ram_gb: Override RAM detection (for containers where psutil misreads)

    Returns:
        Safe worker count with warning if requested exceeds recommendation
    """
    import os

    # Check for environment variable override (useful for containers)
    if container_ram_gb is None:
        container_ram_gb = float(os.environ.get('CONTAINER_RAM_GB', '0'))

    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        total_gb = psutil.virtual_memory().total / (1024**3)

        # Detect container: psutil sees host RAM (>200GB usually means container)
        if total_gb > 200 and container_ram_gb == 0:
            print(f"    ⚠️  Container detected: psutil sees {total_gb:.0f}GB (host RAM)")
            print(f"    ⚠️  Using conservative 2 workers. Set CONTAINER_RAM_GB env var to override.")
            return requested_workers if requested_workers and requested_workers > 0 else 2

        # Use container override if provided
        if container_ram_gb > 0:
            total_gb = container_ram_gb
            available_gb = container_ram_gb * 0.8  # Assume 80% available
            print(f"    ℹ️  Using container RAM override: {container_ram_gb}GB")
    except ImportError:
        # psutil not available, fall back to requested or default
        print("    WARNING: psutil not installed, cannot detect available RAM")
        return requested_workers if requested_workers and requested_workers > 0 else 4

    # Each worker uses ~15GB, leave 5GB headroom for system
    safe_workers = max(1, int((available_gb - 5) / 15))
    max_safe = max(1, int((total_gb - 10) / 15))

    if requested_workers is None or requested_workers == 0:
        # Auto-detect: use safe count
        print(f"    RAM: {total_gb:.1f}GB total, {available_gb:.1f}GB available")
        print(f"    Auto-selected {safe_workers} parallel workers (each uses ~15GB)")
        return safe_workers

    if requested_workers > max_safe:
        print(f"    WARNING: {requested_workers} workers requested but only {max_safe} recommended for {total_gb:.1f}GB RAM")
        print(f"    Each worker uses ~15GB. Risk of OOM with {requested_workers} workers.")
        print(f"    Recommended: {max_safe} workers or fewer. Proceeding with {requested_workers}...")

    return requested_workers


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
        # v3.19: Channel caching for continuation labels (10-20× speedup)
        self._channel_continuation_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _build_feature_names(self):
        """Build list matching multi-window extraction (v3.13+) - CRITICAL: Must match actual shard structure"""
        features = []

        # Price features (12 total: 6 per symbol × 2 symbols)
        for symbol in ['spy', 'tsla']:
            features.extend([
                f'{symbol}_close',
                f'{symbol}_close_norm',  # v3.8: Position in 252-bar (yearly) range
                f'{symbol}_returns',
                f'{symbol}_log_returns',
                f'{symbol}_volatility_10',
                f'{symbol}_volatility_50',
            ])

        # Multi-window channel features (v3.18: CRITICAL - Must match extraction code exactly!)
        # Extraction creates 31 features per channel window
        # When chunked: 21 windows × 9 TFs (5min-weekly) × 31 metrics × 2 = 11,718 in shards
        #               + 21 windows × 2 TFs (monthly/3month) × 31 × 2 = 2,604 in non-channel
        #               = 14,322 total channel features
        # When not chunked: 21 windows × 11 TFs × 31 × 2 = 14,322 (all in shards)
        windows = config.CHANNEL_WINDOW_SIZES  # [168, 160, 150, ..., 10] (21 values)
        timeframes = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']

        # ALL 31 metrics that extraction code actually creates (from lines 848-878):
        metrics = [
            # Position metrics (3)
            'position', 'upper_dist', 'lower_dist',
            # OHLC slopes - raw $/bar (3)
            'close_slope', 'high_slope', 'low_slope',
            # OHLC slopes - normalized % per bar (3)
            'close_slope_pct', 'high_slope_pct', 'low_slope_pct',
            # OHLC r-squared values (4)
            'close_r_squared', 'high_r_squared', 'low_r_squared', 'r_squared_avg',
            # Channel metrics (3)
            'channel_width_pct', 'slope_convergence', 'stability',
            # Ping-pongs - legacy transitions (4 thresholds)
            'ping_pongs', 'ping_pongs_0_5pct', 'ping_pongs_1_0pct', 'ping_pongs_3_0pct',
            # Complete cycles - v3.17 round-trips (4 thresholds)
            'complete_cycles', 'complete_cycles_0_5pct', 'complete_cycles_1_0pct', 'complete_cycles_3_0pct',
            # Direction flags (3)
            'is_bull', 'is_bear', 'is_sideways',
            # Quality indicators (3)
            'quality_score', 'is_valid', 'insufficient_data',
            # Duration (1)
            'duration'
        ]

        for symbol in ['tsla', 'spy']:
            for tf in timeframes:
                for w in windows:
                    for m in metrics:
                        features.append(f'{symbol}_channel_{tf}_{m}_w{w}')

        # RSI features (11 tfs × 3 metrics × 2 symbols = 66)
        for symbol in ['tsla', 'spy']:
            for tf in timeframes:
                features.extend([
                    f'{symbol}_rsi_{tf}',
                    f'{symbol}_rsi_{tf}_oversold',  # Binary
                    f'{symbol}_rsi_{tf}_overbought',  # Binary
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
        # v5.3.2: Expanded to ALL 11 TFs with adaptive rolling windows for break prediction
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.append(f'tsla_channel_duration_ratio_{tf}')
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
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

    def extract_features(self, df: pd.DataFrame, use_cache: bool = True, use_gpu: str = 'auto', cache_suffix: str = None, events_handler=None, continuation: bool = False, continuation_mode: str = 'simple', use_chunking: bool = False, chunk_size_years: int = 1, shard_storage_path: str = None, vix_data: pd.DataFrame = None, skip_native_tf_generation: bool = False, **kwargs) -> tuple:
        """
        Extract all features from aligned SPY-TSLA data (v3.20: 14,322 channel + 180 non-channel = 14,502 total).

        Args:
            df: DataFrame with SPY and TSLA OHLCV columns
            use_cache: Whether to use cached rolling channels (default: True). Set to False to force regeneration.
            use_gpu: GPU acceleration mode (default: 'auto')
                - 'auto': Use GPU only if data size > 50,000 bars (smart default)
                - True: Force GPU (if available, otherwise fallback to CPU)
            use_chunking: Use chunked processing to save memory (default: False)
            chunk_size_years: Size of each chunk in years when using chunked processing (default: 1)
            shard_storage_path: Custom path for shard storage (default: None = use data/feature_cache)
                - False/'never': Always use CPU
            cache_suffix: Optional suffix for cache filename (for testing GPU/CPU separately)
                - None (default): Normal cache filename
                - 'GPU_TEST': Appends to cache name for GPU testing
                - 'CPU_TEST': Appends to cache name for CPU testing
            events_handler: Optional CombinedEventsHandler for event-driven features (v3.9)
                - If provided: Enables earnings/FOMC proximity features
                - If None: Event features will be zeros (backward compatible)
            continuation: Whether to generate continuation prediction labels (default: False)
            vix_data: Optional VIX DataFrame for volatility regime features (v3.20)
                - If provided: Enables 15 VIX-based features (level, percentile, regime, correlations, etc.)
                - Expected columns: vix_close (or CLOSE), vix_high, vix_low with DatetimeIndex
                - If None: VIX features will be zeros/defaults (backward compatible)
            **kwargs: Additional arguments (reserved for future use)

        df should have columns: spy_open, spy_high, spy_low, spy_close, spy_volume,
                                tsla_open, tsla_high, tsla_low, tsla_close, tsla_volume

        Returns tuple of (features_df, continuation_df):
        - features_df: DataFrame with feature columns (180 non-channel when using mmaps,
          14,502 total when not using mmaps). Use get_feature_dim() for model input size.
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
        - 15 VIX features (v3.20): vix_level, percentile_20d/252d, change_1d/5d, regime, tsla/spy_corr,
          momentum_10d, ma_ratio, high_low_range, trend_20d, above_20, above_30, spike
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

        # ═══════════════════════════════════════════════════════════════
        # UNIFIED CACHE VALIDATION (v3.15)
        # ═══════════════════════════════════════════════════════════════

        # Determine unified cache directory for ALL cached data
        if shard_storage_path:
            unified_cache_dir = Path(shard_storage_path)
        else:
            unified_cache_dir = Path('data/feature_cache')
        unified_cache_dir.mkdir(exist_ok=True, parents=True)

        # Generate unified cache key (version + date range + length + horizon + VIX/Events timestamps)
        # v4.4: Include VIX and Events file timestamps to detect stale data
        vix_suffix = ""
        if vix_data is not None:
            try:
                vix_csv_path = Path(config.DATA_DIR) / "VIX_History.csv"
                if vix_csv_path.exists():
                    vix_mtime = int(vix_csv_path.stat().st_mtime)
                    vix_suffix = f"_vix{vix_mtime}"
            except Exception:
                pass  # If VIX file not accessible, skip suffix

        events_suffix = ""
        if events_handler is not None:
            try:
                events_csv_path = Path(config.TSLA_EVENTS_FILE) if hasattr(config, 'TSLA_EVENTS_FILE') else None
                if events_csv_path and events_csv_path.exists():
                    events_mtime = int(events_csv_path.stat().st_mtime)
                    events_suffix = f"_ev{events_mtime}"
            except Exception:
                pass  # If events file not accessible, skip suffix

        cache_key = f"{FEATURE_VERSION}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}_{len(df)}{vix_suffix}{events_suffix}_h24"

        # Upfront cache validation
        print(f"\n📂 Cache Location: {unified_cache_dir}")
        print(f"📂 Cache Validation ({FEATURE_VERSION}, {df.index[0].date()} to {df.index[-1].date()}, {len(df):,} bars, horizon=24):")

        # Check 1: Channel shards (mmap metadata) or legacy pickle
        import json
        channel_cache_valid = False
        channel_cache_type = None
        validated_meta_path = None

        # Check for sharded cache first (new method)
        mmap_meta_files = list(unified_cache_dir.glob(f"features_mmap_meta_{FEATURE_VERSION}_*.json"))
        if mmap_meta_files:
            meta_file = mmap_meta_files[0]
            try:
                meta = json.load(open(meta_file))
                cache_base_dir = meta_file.parent  # Base for resolving relative paths

                # Helper to resolve paths (handles both relative and legacy absolute paths)
                def resolve_shard_path(p):
                    path = Path(p)
                    if path.is_absolute():
                        return path  # Legacy absolute path
                    return cache_base_dir / path  # New relative path

                # Validate all shard files exist
                all_shards_exist = all(
                    resolve_shard_path(c['path']).exists() for c in meta['chunk_info']
                )
                if all_shards_exist:
                    total_gb = sum(c['rows'] * c['cols'] * (8 if 'float64' in meta['dtype'] else 4) for c in meta['chunk_info']) / 1e9
                    print(f"   ✓ Channel shards: Valid ({len(meta['chunk_info'])} shards, {meta['total_rows']:,} rows, {total_gb:.1f} GB, {meta['dtype']})")
                    channel_cache_valid = True
                    channel_cache_type = 'mmap'
                    self._mmap_meta_path = str(meta_file)
                    validated_meta_path = meta_file
                else:
                    print(f"   ⚠️  Channel shards: Metadata found but shard files missing - will extract")
            except Exception as e:
                print(f"   ⚠️  Channel shards: Validation failed ({type(e).__name__}) - will extract")

        # Fallback: Check for legacy pickle cache
        if not channel_cache_valid:
            legacy_cache = unified_cache_dir / f'rolling_channels_{cache_key}.pkl'
            if legacy_cache.exists() and use_cache:
                # Will be validated later in _extract_channel_features
                print(f"   ℹ️  Channel cache: Found legacy pickle (will validate on load)")
                channel_cache_type = 'legacy'
            else:
                print(f"   ❌ Channel cache: Not found - will extract (~5-7 min)")

        # Check 2: Continuation labels (hierarchical per-TF format, v4.3+)
        # Labels are stored per-timeframe: continuation_labels_{tf}_v{version}_{dates}.pkl
        cont_cache_valid = False
        if continuation and use_cache:
            # Build cache suffix matching the format used in extract_features() and generate_hierarchical_continuation_labels()
            # Uses df.index dates formatted as YYYYMMDD
            # v5.0: FEATURE_VERSION already starts with 'v', don't add another
            hierarchical_cache_suffix = f"{FEATURE_VERSION}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}"

            # Check if all per-TF label files exist
            found_tfs = []
            missing_tfs = []
            for tf in HIERARCHICAL_TIMEFRAMES:
                tf_label_path = unified_cache_dir / f"continuation_labels_{tf}_{hierarchical_cache_suffix}.pkl"
                if tf_label_path.exists():
                    found_tfs.append(tf)
                else:
                    missing_tfs.append(tf)

            if len(found_tfs) == len(HIERARCHICAL_TIMEFRAMES):
                # All TFs cached - validate one file to ensure format is correct
                sample_path = unified_cache_dir / f"continuation_labels_5min_{hierarchical_cache_suffix}.pkl"
                try:
                    test_load = pd.read_pickle(sample_path)
                    if len(test_load) > 0 and 'duration_bars' in test_load.columns:
                        print(f"   ✓ Continuation labels (hierarchical): Valid ({len(HIERARCHICAL_TIMEFRAMES)} TFs cached)")
                        cont_cache_valid = True
                    else:
                        print(f"   ⚠️  Continuation labels: Invalid format - will regenerate")
                except Exception as e:
                    print(f"   ⚠️  Continuation labels: Corrupted ({type(e).__name__}) - will regenerate")
            elif len(found_tfs) > 0:
                print(f"   ⚠️  Continuation labels: Partial ({len(found_tfs)}/{len(HIERARCHICAL_TIMEFRAMES)} TFs) - will regenerate missing")
            else:
                print(f"   ❌ Continuation labels: Not found - will generate (~1 hour)")

        # Check 3: Non-channel features (Price, RSI, Correlation, Cycle, Volume, Time, Breakdown)
        non_channel_cache_path = unified_cache_dir / f"non_channel_features_{cache_key}.pkl"
        non_channel_cache_valid = False
        if non_channel_cache_path.exists() and use_cache:
            try:
                test_load = pd.read_pickle(non_channel_cache_path)
                if len(test_load) > 0 and len(test_load.columns) > 0:
                    print(f"   ✓ Non-channel features: Valid ({len(test_load.columns)} cols, {len(test_load):,} rows)")
                    non_channel_cache_valid = True
                else:
                    print(f"   ⚠️  Non-channel features: Empty - will regenerate")
            except Exception as e:
                print(f"   ⚠️  Non-channel features: Corrupted ({type(e).__name__}) - will regenerate")
                non_channel_cache_valid = False
        else:
            print(f"   ❌ Non-channel features: Not found - will extract (~10-30 sec)")

        # Invalidation: If ANY cache fails and use_cache=True, should we invalidate all?
        # For now, keep independent (channel cache can work without continuation labels)

        # Summary
        print()
        all_caches_valid = channel_cache_valid and non_channel_cache_valid and (cont_cache_valid or not continuation)
        if all_caches_valid:
            print(f"   🚀 All caches valid - skipping extraction entirely!")
        elif channel_cache_valid and (cont_cache_valid or not continuation):
            print(f"   ⚡ Channel + labels cached - will only recompute non-channel features (~10-30 sec)")
        else:
            needed = []
            if not channel_cache_valid:
                needed.append("channels (~5-7 min)")
            if not non_channel_cache_valid:
                needed.append("non-channel (~10-30 sec)")
            if continuation and not cont_cache_valid:
                needed.append("continuation labels (~1 hour)")
            if needed:
                print(f"   ⏱️  Will regenerate: {', '.join(needed)}")
        print()

        # Store cache paths for later use
        self._unified_cache_dir = unified_cache_dir
        self._cache_key = cache_key
        # Note: continuation labels now use hierarchical per-TF format (directory-based), not single file
        self._non_channel_cache_path = non_channel_cache_path  # Store for saving/loading
        # Short-circuit channel extraction when mmap cache is valid and use_cache is True
        skip_channel_calc = channel_cache_valid and use_cache and validated_meta_path is not None
        # Short-circuit ALL extraction when all caches valid
        skip_all_extraction = skip_channel_calc and non_channel_cache_valid and use_cache

        # PASS 1: Extract base features (or load from cache if all valid)
        if skip_all_extraction:
            # Load non-channel features from cache - skip ALL extraction!
            print("   📂 Loading non-channel features from cache...")
            features_df = pd.read_pickle(non_channel_cache_path)
            print(f"   ✓ Loaded {len(features_df.columns)} non-channel features from cache (saved ~10-30 sec!)")
        else:
            # Normal extraction path
            print("   Extracting base features...")
            with tqdm(total=8, desc="   Feature extraction", leave=True, ncols=100, ascii=True, mininterval=0.5) as pbar:
                price_df = self._extract_price_features(df, use_gpu=use_gpu_resolved)
                pbar.update(1)

                # Chunked extraction logic
                # Note: skip_channel_calc=True when mmap cache is valid, so this only runs on first extraction
                if use_chunking and not skip_channel_calc:
                    print(f"      Using chunked extraction ({chunk_size_years}-year chunks)")
                    chunk_result = self._extract_channel_features_chunked(
                        df,
                        multi_res_data=multi_res_data,
                        use_gpu=use_gpu_resolved,
                        chunk_size_years=chunk_size_years,
                        shard_storage_path=shard_storage_path
                    )

                    # Chunked extraction returns mmap metadata (not DataFrame!)
                    # v3.18: Also extracts monthly/3month separately
                    self._mmap_meta_path = chunk_result['mmap_meta_path']
                    monthly_3month_df = chunk_result.get('monthly_3month_features')  # Hybrid processing
                    channel_df = None  # Will be loaded as mmap in dataset
                elif not skip_channel_calc:
                    # Normal extraction (all at once) or load from cache
                    channel_df = self._extract_channel_features(
                        df,
                        multi_res_data=multi_res_data,
                        use_cache=use_cache,
                        use_gpu=use_gpu_resolved,
                        cache_suffix=cache_suffix
                    )
                    monthly_3month_df = None  # Not using hybrid mode (all TFs processed together)
                else:
                    # Skip channel calc entirely; rely on cached mmap
                    channel_df = None
                    monthly_3month_df = None
                pbar.update(1)

                rsi_df = self._extract_rsi_features(df, multi_res_data=multi_res_data)
                pbar.update(1)

                correlation_df = self._extract_correlation_features(df, use_gpu=use_gpu_resolved)
                pbar.update(1)

                cycle_df = self._extract_cycle_features(df)
                pbar.update(1)

                volume_df = self._extract_volume_features(df)
                pbar.update(1)

                time_df = self._extract_time_features(df)
                pbar.update(1)

                # v3.20: VIX features for volatility regime detection
                vix_df = self._extract_vix_features(df, vix_data)
                pbar.update(1)

            # Concat base features FIRST (skip channel_df if using mmap)
            if channel_df is None and hasattr(self, '_mmap_meta_path'):
                # Mmap mode - channel features will be loaded separately in dataset
                # v3.19: Monthly/3month now in separate shard (not in RAM concat)
                # v3.20: Added VIX features
                concat_list = [price_df, rsi_df, correlation_df, cycle_df, volume_df, time_df, vix_df]
                # Monthly/3month removed - they're in monthly_3month_shard.npy, loaded as mmap
                # Optimize: copy=False avoids unnecessary data copies
                base_features_df = pd.concat(concat_list, axis=1, copy=False)
                # Free intermediate DataFrames (small in mmap mode, but good hygiene)
                del price_df, rsi_df, correlation_df, cycle_df, volume_df, time_df, vix_df
            else:
                # Normal mode - include channel features
                # Optimize: copy=False avoids unnecessary data copies
                base_features_df = pd.concat([
                    price_df,
                    channel_df,
                    rsi_df,
                    correlation_df,
                    cycle_df,
                    volume_df,
                    time_df,
                    vix_df
                ], axis=1, copy=False)
                # CRITICAL: Free intermediate DataFrames IMMEDIATELY to prevent OOM
                # channel_df alone is ~90GB; without this, peak RAM hits ~280GB during concat
                import gc
                del price_df, channel_df, rsi_df, correlation_df, cycle_df, volume_df, time_df, vix_df
                gc.collect()
                print(f"   🧹 Freed intermediate feature DataFrames (~90GB)")

            # PASS 2: Extract breakdown features (needs base features + optional events)
            # v5.3.3: Only calculate legacy 1-min breakdown if native TF generation will be skipped
            # (live mode, legacy mode, or when explicitly disabled)
            if skip_native_tf_generation:
                # Legacy/live mode - need breakdown at 1-min resolution now
                with tqdm(total=1, desc="   Breakdown features", leave=False, ncols=100, ascii=True, mininterval=0.5) as pbar:
                    breakdown_df = self._extract_breakdown_features(base_features_df, df, events_handler)
                    pbar.update(1)

                # Final concat (non-channel features only if using mmap)
                # Optimize: copy=False avoids unnecessary data copies
                features_df = pd.concat([base_features_df, breakdown_df], axis=1, copy=False)
                # Free concat intermediates to reclaim ~92GB before continuation labels
                import gc
                del base_features_df, breakdown_df
                gc.collect()
            else:
                # Native TF mode - breakdown will be calculated at native resolution later
                # in _precompute_timeframe_sequences() or generate_native_tf_from_chunks()
                features_df = base_features_df  # No breakdown yet (avoids duplicates)
                # Free base_features_df reference (features_df now owns the data)
                import gc
                del base_features_df
                gc.collect()

        # Generate continuation labels if requested
        # v4.3: Now generates hierarchical per-TF labels using channel-structure break detection
        # v5.2: Also generates transition labels for multi-phase compositor
        continuation_labels_dir = None
        if continuation:
            cache_dir = self._unified_cache_dir if hasattr(self, '_unified_cache_dir') else Path('data/feature_cache')
            # v5.0 fix: FEATURE_VERSION already starts with 'v', don't add another
            cache_suffix = f"{FEATURE_VERSION}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}"

            # Check if all per-TF label files exist (cache check)
            # v5.2: Also check for transition labels
            all_continuation_cached = True
            all_transition_cached = True
            if use_cache:
                for tf in HIERARCHICAL_TIMEFRAMES:
                    # Check continuation labels
                    tf_label_path = cache_dir / f"continuation_labels_{tf}_{cache_suffix}.pkl"
                    if not tf_label_path.exists():
                        all_continuation_cached = False
                    # Check transition labels (v5.2)
                    tf_transition_path = cache_dir / f"transition_labels_{tf}_{cache_suffix}.pkl"
                    if not tf_transition_path.exists():
                        all_transition_cached = False
            else:
                all_continuation_cached = False
                all_transition_cached = False

            if all_continuation_cached:
                print(f"   📂 Found cached hierarchical continuation labels ({len(HIERARCHICAL_TIMEFRAMES)} TFs)")
                continuation_labels_dir = cache_dir
            else:
                # Generate fresh hierarchical continuation labels
                saved_files = self.generate_hierarchical_continuation_labels(
                    df=df,
                    timeframes=HIERARCHICAL_TIMEFRAMES,
                    output_dir=cache_dir,
                    cache_suffix=cache_suffix
                )
                if saved_files:
                    continuation_labels_dir = cache_dir
                    print(f"   ✓ Hierarchical continuation labels saved to: {cache_dir}")
                    # Mark that we generated fresh continuation labels → need fresh transition labels
                    all_transition_cached = False

            # v5.2: Generate transition labels (uses continuation labels)
            if continuation_labels_dir:
                if all_transition_cached:
                    print(f"   📂 Found cached transition labels ({len(HIERARCHICAL_TIMEFRAMES)} TFs)")
                else:
                    print(f"\n   🔄 Generating transition labels for v5.2 multi-phase compositor...")
                    transition_files = self.generate_transition_labels(
                        continuation_labels_dir=cache_dir,
                        output_dir=cache_dir,
                        cache_suffix=cache_suffix
                    )
                    if transition_files:
                        print(f"   ✓ Transition labels saved to: {cache_dir}")

        # Fill NaNs (only if we extracted - cached data already has NaNs filled)
        if not skip_all_extraction:
            features_df = features_df.bfill().fillna(0)

            # Save non-channel features to cache for next time (only when using mmap mode)
            if hasattr(self, '_mmap_meta_path') and hasattr(self, '_non_channel_cache_path'):
                print(f"   💾 Caching non-channel features to: {self._non_channel_cache_path.name}")
                self._non_channel_cache_path.parent.mkdir(exist_ok=True, parents=True)
                features_df.to_pickle(self._non_channel_cache_path)
                print(f"   ✓ Non-channel cache saved ({len(features_df.columns)} cols, {len(features_df):,} rows)")

        # If using mmap shards, return metadata alongside features
        if hasattr(self, '_mmap_meta_path'):
            print(f"   ✓ {len(features_df.columns)} non-channel features + mmap channel shards ready")
            # v4.3: Return continuation_labels_dir (Path) instead of continuation_df (DataFrame)
            return (features_df, continuation_labels_dir, self._mmap_meta_path)
        else:
            # v4.1: Pre-compute timeframe-specific sequences for hierarchical model
            # This creates separate .npy files for each timeframe with native resolution
            # v5.1: Skip in live mode to avoid creating new tf_meta files
            if not skip_native_tf_generation:
                cache_dir = self._unified_cache_dir
                cache_key = self._cache_key if hasattr(self, '_cache_key') else FEATURE_VERSION
                # v5.3.3: Pass raw_df and events_handler for native TF breakdown calculation
                self._precompute_timeframe_sequences(
                    features_df,
                    cache_dir,
                    cache_key,
                    raw_df=df,
                    events_handler=events_handler
                )

            print(f"   ✓ {len(features_df.columns)} features ready")
            # v4.3: Return continuation_labels_dir (Path) instead of continuation_df (DataFrame)
            return features_df, continuation_labels_dir

    def _precompute_timeframe_sequences(
        self,
        features_df: pd.DataFrame,
        cache_dir: Path,
        cache_key: str,
        raw_df: pd.DataFrame = None,      # v5.3.3: For event lookups in breakdown
        events_handler = None              # v5.3.3: For event features in breakdown
    ) -> None:
        """
        Pre-compute resampled feature sequences for each timeframe.

        This enables the hierarchical model to receive native timeframe resolution:
        - 5min layer sees 5-min bars (not 1-min bars)
        - 1h layer sees hourly bars
        - etc.

        v5.3.3: Now calculates breakdown features AFTER resampling to native TF.
        This ensures train-test consistency with live predictions.

        Args:
            features_df: Full features DataFrame at 1-min resolution
            cache_dir: Directory to save timeframe-specific .npy files
            cache_key: Cache key for file naming
            raw_df: Original OHLCV DataFrame (for event timestamp lookups)
            events_handler: Optional event handler for breakdown features
        """
        import json

        print(f"\n   🔄 Pre-computing timeframe sequences for hierarchical model...")

        # Identify shared columns (not timeframe-specific)
        # These are columns that don't contain _5min_, _15min_, etc.
        all_cols = list(features_df.columns)
        shared_cols = []
        tf_specific_cols = {tf: [] for tf in HIERARCHICAL_TIMEFRAMES}

        for col in all_cols:
            is_tf_specific = False
            for tf in HIERARCHICAL_TIMEFRAMES:
                # Match patterns like _5min_, _15min_, _1h_, _daily_, etc.
                if f'_{tf}_' in col:
                    tf_specific_cols[tf].append(col)
                    is_tf_specific = True
                    break
            if not is_tf_specific:
                shared_cols.append(col)

        print(f"   📊 Found {len(shared_cols)} shared columns, {sum(len(v) for v in tf_specific_cols.values())} timeframe-specific")

        # Metadata to save
        meta = {
            'feature_version': FEATURE_VERSION,
            'cache_key': cache_key,
            'sequence_lengths': TIMEFRAME_SEQUENCE_LENGTHS,
            'shared_columns': shared_cols,
            'timeframe_columns': {},
            'timeframe_shapes': {},
            'total_rows_1min': len(features_df),
        }

        # v5.3.3: Two-pass processing for native TF breakdown features
        # Pass 1: Calculate breakdown for ALL TFs at their native resolutions
        print(f"   📊 Pass 1/2: Calculating breakdown features at native TF resolutions...")
        all_tf_resampled = {}  # Store resampled DataFrames
        all_tf_breakdown = {}  # Store breakdown features per TF

        for tf in tqdm(HIERARCHICAL_TIMEFRAMES, desc="   Calc breakdown per TF", leave=False, ncols=100, ascii=True):
            tf_cols = shared_cols + tf_specific_cols[tf]
            tf_features = features_df[tf_cols].copy()

            # Resample to native TF resolution
            tf_rule = TIMEFRAME_RESAMPLE_RULES[tf]
            resampled = tf_features.resample(tf_rule).last().dropna()

            # Calculate breakdown at THIS TF's native resolution
            breakdown_native = self._calculate_breakdown_at_native_tf(
                resampled,
                tf=tf,
                raw_df=raw_df,
                events_handler=events_handler
            )

            # Store for Pass 2
            all_tf_resampled[tf] = resampled
            all_tf_breakdown[tf] = breakdown_native

        # Pass 2: Add cross-TF breakdown features and save
        print(f"   💾 Pass 2/2: Adding cross-TF features and saving...")

        for tf in tqdm(HIERARCHICAL_TIMEFRAMES, desc="   Saving TF sequences", leave=False, ncols=100, ascii=True):
            # Get this TF's base features
            resampled = all_tf_resampled[tf]

            # Add breakdown from ALL TFs (resampled to match this TF's resolution)
            for other_tf, other_breakdown in all_tf_breakdown.items():
                # Resample other TF's breakdown to match current TF's index
                # Use forward-fill (ffill) to broadcast coarser→finer (e.g., daily→5min)
                if len(other_breakdown) > 0:
                    breakdown_aligned = other_breakdown.reindex(resampled.index, method='ffill')
                    # Concat horizontally (add columns)
                    resampled = pd.concat([resampled, breakdown_aligned], axis=1, copy=False)

            # Update metadata with final column list (includes cross-TF breakdown)
            meta['timeframe_columns'][tf] = list(resampled.columns)

            # Save as .npy for memory-mapped loading
            output_path = cache_dir / f"tf_sequence_{tf}_{cache_key}.npy"
            np.save(output_path, resampled.values.astype(np.float32))

            # Save timestamps separately for index conversion
            ts_path = cache_dir / f"tf_timestamps_{tf}_{cache_key}.npy"
            timestamps_ns = resampled.index.view(np.int64)
            np.save(ts_path, timestamps_ns)

            meta['timeframe_shapes'][tf] = list(resampled.shape)

            # Log progress
            seq_len = TIMEFRAME_SEQUENCE_LENGTHS[tf]
            real_time = {
                '5min': '~17 hours', '15min': '~25 hours', '30min': '~40 hours',
                '1h': '1 week', '2h': '1 week', '3h': '1 week', '4h': '1 week',
                'daily': '30 days', 'weekly': '20 weeks', 'monthly': '12 months', '3month': '24 months'
            }
            print(f"      {tf}: {resampled.shape[0]:,} bars × {resampled.shape[1]} features (seq_len={seq_len}, {real_time.get(tf, '?')})")

        # Save metadata
        meta_path = cache_dir / f"tf_meta_{cache_key}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"   ✓ Saved {len(HIERARCHICAL_TIMEFRAMES)} timeframe sequences to {cache_dir}")
        print(f"   📄 Metadata: {meta_path.name}")

    def generate_native_tf_from_chunks(
        self,
        chunks_meta_path: Path,
        output_cache_dir: Path,
        streaming: bool = True
    ) -> None:
        """
        Priority 3 (Option A): Generate native timeframe sequences from existing mmap chunks.

        This enables the two-machine workflow:
        1. Machine A (low RAM): Extract features with --use-chunking → generates mmap shards
        2. Machine B: Call this method → generates native TF sequences from chunks
        3. Machine B: Train with --native-timeframes → uses pre-computed sequences

        Args:
            chunks_meta_path: Path to mmap_meta JSON file from chunked extraction
            output_cache_dir: Directory to save native TF .npy files
            streaming: If True (default), process one chunk at a time per timeframe.
                      Peak RAM ~5-8GB. If False, load all chunks into RAM (~50GB)
                      for slightly faster processing.

        Workflow:
        ```bash
        # Machine A (low RAM)
        python train_hierarchical.py --train_start_year 2015 --train_end_year 2023 \\
          --use-chunking --feature_workers 1
        # Creates: data/feature_cache/features_mmap_meta_*.json + mmap shards

        # Transfer chunks to Machine B ...

        # Machine B (high RAM)
        python -c "
          from src.ml.features import TradingFeatureExtractor
          extractor = TradingFeatureExtractor()
          extractor.generate_native_tf_from_chunks(
              chunks_meta_path='data/feature_cache/features_mmap_meta_v3.20_vix_20150101_20231230_1234567_h24.json',
              output_cache_dir='data/feature_cache'
          )
        "
        # Creates: tf_sequence_*.npy, tf_timestamps_*.npy, tf_meta_*.json

        # Now train with native TF mode
        python train_hierarchical.py --native-timeframes --tf-meta data/feature_cache/tf_meta_v3.20_vix_*.json
        ```

        Implementation approach:
        1. Load mmap_meta to understand chunk structure
        2. Memory-map all feature chunks (don't load into RAM)
        3. For each timeframe:
           a. Create output .npy file
           b. Stream through chunks sequentially
           c. Resample on-the-fly using pandas
           d. Append resampled data to output file
        4. Create tf_meta JSON with mapping info

        This avoids loading all 49GB chunks into RAM at once.
        """
        import json
        import pandas as pd
        from pathlib import Path

        chunks_meta_path = Path(chunks_meta_path)
        output_cache_dir = Path(output_cache_dir)
        cache_dir = chunks_meta_path.parent

        print(f"\n   🔄 Generating native timeframe sequences from chunks...")
        print(f"   📄 Loading metadata from: {chunks_meta_path.name}")

        # Load mmap metadata
        with open(chunks_meta_path) as f:
            mmap_meta = json.load(f)

        chunk_info = mmap_meta['chunk_info']
        dtype = np.dtype(mmap_meta['dtype'])
        total_rows = mmap_meta['total_rows']
        monthly_shard_info = mmap_meta.get('monthly_3month_shard')

        print(f"   📊 Found {len(chunk_info)} chunks, {total_rows:,} total rows")
        if monthly_shard_info:
            print(f"   📊 Found monthly/3month shard: {monthly_shard_info['rows']:,} rows × {monthly_shard_info['cols']} cols")
        print(f"   💾 Mode: {'Streaming (low RAM ~5-8GB)' if streaming else 'Full load (~50GB RAM)'}")

        # Load feature column names from metadata (v4.1: required for TF identification)
        feature_columns = mmap_meta.get('feature_columns')
        if not feature_columns:
            raise ValueError(
                "Missing 'feature_columns' in mmap_meta. "
                "Re-run chunked extraction with the latest code to generate column names."
            )

        # Branch based on streaming mode
        if streaming:
            self._generate_native_tf_streaming(
                chunk_info, cache_dir, output_cache_dir, mmap_meta,
                monthly_shard_info, feature_columns
            )
        else:
            self._generate_native_tf_full_load(
                chunk_info, cache_dir, output_cache_dir, mmap_meta,
                monthly_shard_info, feature_columns
            )

    def _generate_native_tf_full_load(
        self,
        chunk_info: list,
        cache_dir: Path,
        output_cache_dir: Path,
        mmap_meta: dict,
        monthly_shard_info: dict,
        feature_columns: list
    ) -> None:
        """
        Full load implementation: load all chunks into RAM then resample.
        Peak RAM: ~50GB for full dataset.
        Faster than streaming but requires high RAM.
        """
        import json
        import pandas as pd

        # Step 1: Load all chunks and timestamps to build full feature DataFrame
        print(f"   ⚙️  Loading chunks into memory...")

        all_features = []
        all_indices = []

        for i, chunk in enumerate(chunk_info):
            chunk_path = cache_dir / chunk['path']
            index_path = cache_dir / chunk['index_path']

            if not chunk_path.exists() or not index_path.exists():
                raise FileNotFoundError(f"Chunk files missing: {chunk_path}")

            # Memory-map the feature array
            chunk_array = np.load(str(chunk_path), mmap_mode='r').astype(np.float32)
            index_array = np.load(str(index_path), mmap_mode='r')

            all_features.append(chunk_array)
            all_indices.append(index_array)

            if (i + 1) % 5 == 0 or i == len(chunk_info) - 1:
                print(f"       Loaded {i+1}/{len(chunk_info)} chunks ({sum(len(idx) for idx in all_indices):,} rows)")

        # Concatenate all features and indices
        full_features = np.vstack(all_features)
        full_indices = np.concatenate(all_indices)

        # feature_columns is passed as parameter (already validated in main method)

        # Step 1b: Load monthly/3month shard if present
        if monthly_shard_info:
            monthly_path = cache_dir / monthly_shard_info['path']
            if monthly_path.exists():
                print(f"   📂 Loading monthly/3month shard...")
                monthly_array = np.load(str(monthly_path), mmap_mode='r').astype(np.float32)

                # Get monthly column names
                monthly_columns = monthly_shard_info.get('columns')
                if not monthly_columns:
                    # Generate column names based on known structure (fallback)
                    monthly_columns = [f'monthly_3month_col_{i}' for i in range(monthly_array.shape[1])]
                    print(f"   ⚠️  Monthly column names not in metadata, using generic names")

                # Check row count alignment
                chunk_rows = full_features.shape[0]
                monthly_rows = monthly_array.shape[0]
                if chunk_rows != monthly_rows:
                    print(f"\n" + "=" * 70)
                    print(f"   ❌ CACHE ROW COUNT MISMATCH DETECTED!")
                    print(f"=" * 70)
                    print(f"   Chunks total:    {chunk_rows:,} rows")
                    print(f"   Monthly shard:   {monthly_rows:,} rows")
                    print(f"   Difference:      {abs(monthly_rows - chunk_rows):,} rows")
                    print(f"")
                    print(f"   This means your cache shards were created with different data ranges.")
                    print(f"   FIX: Delete cache and regenerate:")
                    print(f"        rm -rf data/feature_cache/features_mmap_*")
                    print(f"        rm -rf data/feature_cache/monthly_3month_*")
                    print(f"=" * 70 + "\n")
                    raise ValueError(
                        f"Cache row count mismatch: chunks={chunk_rows:,} vs monthly={monthly_rows:,}. "
                        f"Delete cache and regenerate."
                    )

                # Concatenate monthly features to main features
                full_features = np.hstack([full_features, monthly_array[:]])
                feature_columns = feature_columns + monthly_columns
                print(f"   ✓ Added monthly/3month: {monthly_array.shape[1]} columns")
            else:
                print(f"   ⚠️  Monthly shard not found: {monthly_path}")

        # Create DataFrame with timestamps as index and proper column names
        df = pd.DataFrame(full_features, columns=feature_columns)
        df.index = pd.to_datetime(full_indices, unit='ns')

        print(f"   ✓ Loaded {len(df):,} rows × {len(df.columns)} channel features")

        # Load and merge non-channel features (contains tsla_close, spy_close, etc.)
        non_channel_path = cache_dir / f"non_channel_features_{mmap_meta['cache_key']}.pkl"
        if non_channel_path.exists():
            print(f"   📂 Loading non-channel features...")
            non_channel_df = pd.read_pickle(non_channel_path)
            # Ensure same index format
            if not isinstance(non_channel_df.index, pd.DatetimeIndex):
                non_channel_df.index = pd.to_datetime(non_channel_df.index)
            # Align indices and merge (non-channel first so tsla_close is accessible)
            common_idx = df.index.intersection(non_channel_df.index)
            if len(common_idx) > 0:
                df = pd.concat([non_channel_df.loc[common_idx], df.loc[common_idx]], axis=1)
                print(f"   ✓ Merged {len(non_channel_df.columns)} non-channel features ({len(df):,} aligned rows)")
            else:
                print(f"   ⚠️  No overlapping timestamps between channel and non-channel features")
        else:
            print(f"   ⚠️  No non-channel features found: {non_channel_path.name}")

        # Step 2: Identify shared vs timeframe-specific columns (same logic as _precompute_timeframe_sequences)
        all_cols = list(df.columns)
        shared_cols = []
        tf_specific_cols = {tf: [] for tf in HIERARCHICAL_TIMEFRAMES}

        for col in all_cols:
            is_tf_specific = False
            for tf in HIERARCHICAL_TIMEFRAMES:
                if f'_{tf}_' in col:  # Now col is a string (column name), not integer
                    tf_specific_cols[tf].append(col)
                    is_tf_specific = True
                    break
            if not is_tf_specific:
                shared_cols.append(col)

        print(f"   📊 Found {len(shared_cols)} shared columns, {sum(len(v) for v in tf_specific_cols.values())} timeframe-specific")

        # Metadata for output
        meta = {
            'feature_version': FEATURE_VERSION,
            'cache_key': mmap_meta.get('cache_key', 'from_chunks'),
            'sequence_lengths': TIMEFRAME_SEQUENCE_LENGTHS,
            'shared_columns': shared_cols,
            'timeframe_columns': {},
            'timeframe_shapes': {},
            'total_rows_1min': len(df),
        }

        # Step 3: v5.3.3 Two-pass for cross-TF breakdown features
        # Pass 1: Calculate breakdown for each TF at native resolution
        print(f"   📊 Pass 1/2: Calculating breakdown at native TF resolutions...")
        all_tf_resampled = {}  # Store resampled DataFrames
        all_tf_breakdown = {}  # Store breakdown features per TF

        for tf in tqdm(HIERARCHICAL_TIMEFRAMES, desc="   Calc breakdown per TF", leave=False, ncols=100, ascii=True):
            tf_cols = shared_cols + tf_specific_cols[tf]

            # Select columns for this timeframe and resample
            tf_features = df[tf_cols].copy()
            tf_rule = TIMEFRAME_RESAMPLE_RULES[tf]

            # Use .last() to get value at end of each bar
            resampled = tf_features.resample(tf_rule).last().dropna()

            # Calculate breakdown at THIS TF's native resolution
            breakdown_tf = self._calculate_breakdown_at_native_tf(
                resampled,
                tf=tf,
                raw_df=None,          # Not available in chunked mode
                events_handler=None   # Not available in chunked mode
            )

            # Store for Pass 2
            all_tf_resampled[tf] = resampled
            all_tf_breakdown[tf] = breakdown_tf

        # Pass 2: Add cross-TF breakdown features and save
        print(f"   💾 Pass 2/2: Adding cross-TF features and saving...")

        for tf in tqdm(HIERARCHICAL_TIMEFRAMES, desc="   Saving TF sequences", leave=False, ncols=100, ascii=True):
            # Get this TF's base features
            resampled = all_tf_resampled[tf]

            # Add breakdown from ALL TFs (resampled to match this TF's resolution)
            for other_tf, other_breakdown in all_tf_breakdown.items():
                # Resample other TF's breakdown to match current TF's index
                # Use forward-fill (ffill) to broadcast coarser→finer (e.g., daily→5min)
                if len(other_breakdown) > 0:
                    breakdown_aligned = other_breakdown.reindex(resampled.index, method='ffill')
                    # Concat horizontally (add columns)
                    resampled = pd.concat([resampled, breakdown_aligned], axis=1, copy=False)

            # Remove duplicate columns (can happen from same-TF concat)
            resampled = resampled.loc[:, ~resampled.columns.duplicated(keep='first')]

            # Update metadata with final column list (includes cross-TF breakdown)
            meta['timeframe_columns'][tf] = list(resampled.columns)

            # Save as .npy file (now includes cross-TF breakdown)
            output_path = output_cache_dir / f"tf_sequence_{tf}_{meta['cache_key']}.npy"
            np.save(output_path, resampled.values.astype(np.float32))

            # Save timestamps separately
            ts_path = output_cache_dir / f"tf_timestamps_{tf}_{meta['cache_key']}.npy"
            timestamps_ns = resampled.index.view(np.int64)
            np.save(ts_path, timestamps_ns)

            meta['timeframe_shapes'][tf] = list(resampled.shape)

            # Log progress
            seq_len = TIMEFRAME_SEQUENCE_LENGTHS[tf]
            real_time = {
                '5min': '~17 hours', '15min': '~25 hours', '30min': '~40 hours',
                '1h': '1 week', '2h': '1 week', '3h': '1 week', '4h': '1 week',
                'daily': '30 days', 'weekly': '20 weeks', 'monthly': '12 months', '3month': '24 months'
            }
            print(f"      {tf}: {resampled.shape[0]:,} bars × {resampled.shape[1]} features (seq_len={seq_len}, {real_time.get(tf, '?')})")

        # Save metadata
        meta_path = output_cache_dir / f"tf_meta_{meta['cache_key']}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"   ✓ Generated {len(HIERARCHICAL_TIMEFRAMES)} timeframe sequences to {output_cache_dir}")
        print(f"   📄 Metadata: {meta_path.name}")

    def _generate_native_tf_streaming(
        self,
        chunk_info: list,
        cache_dir: Path,
        output_cache_dir: Path,
        mmap_meta: dict,
        monthly_shard_info: dict,
        feature_columns: list
    ) -> None:
        """
        Streaming implementation: process one timeframe at a time, one chunk at a time.
        Peak RAM: ~5-8GB (one chunk's TF columns + accumulated resampled results).
        """
        import json
        import gc
        import pandas as pd

        # Get column indices for shared vs TF-specific
        shared_col_indices = []
        tf_col_indices = {tf: [] for tf in HIERARCHICAL_TIMEFRAMES}

        for i, col in enumerate(feature_columns):
            is_tf_specific = False
            for tf in HIERARCHICAL_TIMEFRAMES:
                if f'_{tf}_' in col:
                    tf_col_indices[tf].append(i)
                    is_tf_specific = True
                    break
            if not is_tf_specific:
                shared_col_indices.append(i)

        print(f"   📊 Found {len(shared_col_indices)} shared columns, {sum(len(v) for v in tf_col_indices.values())} timeframe-specific")

        # Load monthly shard info for column indices (if present)
        monthly_array = None
        monthly_columns = []
        if monthly_shard_info:
            monthly_path = cache_dir / monthly_shard_info['path']
            if monthly_path.exists():
                monthly_array = np.load(str(monthly_path), mmap_mode='r')
                monthly_columns = monthly_shard_info.get('columns', [])
                print(f"   📂 Monthly shard: {monthly_array.shape[1]} columns (memory-mapped)")

                # Check row count alignment with chunks
                chunk_total_rows = mmap_meta.get('total_rows', 0)
                monthly_rows = monthly_array.shape[0]
                if chunk_total_rows > 0 and chunk_total_rows != monthly_rows:
                    print(f"\n" + "=" * 70)
                    print(f"   ❌ CACHE ROW COUNT MISMATCH DETECTED!")
                    print(f"=" * 70)
                    print(f"   Chunks total:    {chunk_total_rows:,} rows")
                    print(f"   Monthly shard:   {monthly_rows:,} rows")
                    print(f"   Difference:      {abs(monthly_rows - chunk_total_rows):,} rows")
                    print(f"")
                    print(f"   This means your cache shards were created with different data ranges.")
                    print(f"   FIX: Delete cache and regenerate:")
                    print(f"        rm -rf data/feature_cache/features_mmap_*")
                    print(f"        rm -rf data/feature_cache/monthly_3month_*")
                    print(f"=" * 70 + "\n")
                    raise ValueError(
                        f"Cache row count mismatch: chunks={chunk_total_rows:,} vs monthly={monthly_rows:,}. "
                        f"Delete cache and regenerate."
                    )

        # Load non-channel features (contains tsla_close, spy_close, etc.)
        non_channel_path = cache_dir / f"non_channel_features_{mmap_meta['cache_key']}.pkl"
        non_channel_df = None
        non_channel_columns = []
        if non_channel_path.exists():
            non_channel_df = pd.read_pickle(non_channel_path)
            if not isinstance(non_channel_df.index, pd.DatetimeIndex):
                non_channel_df.index = pd.to_datetime(non_channel_df.index)
            non_channel_columns = list(non_channel_df.columns)
            print(f"   📂 Non-channel features: {len(non_channel_columns)} columns")
        else:
            print(f"   ⚠️  No non-channel features found: {non_channel_path.name}")

        # Metadata for output
        meta = {
            'feature_version': FEATURE_VERSION,
            'cache_key': mmap_meta.get('cache_key', 'from_chunks'),
            'sequence_lengths': TIMEFRAME_SEQUENCE_LENGTHS,
            'shared_columns': non_channel_columns + [feature_columns[i] for i in shared_col_indices],
            'timeframe_columns': {},
            'timeframe_shapes': {},
            'total_rows_1min': mmap_meta['total_rows'],
        }

        print(f"   🔄 Streaming resample (one TF at a time, one chunk at a time)...")

        # v5.3.3: Two-pass processing for cross-TF breakdown features
        # Pass 1: Calculate breakdown for each TF at native resolution, store in memory
        # Pass 2: Add all TF breakdowns to each file (reindexed to match)
        all_tf_breakdown = {}  # Store breakdown DataFrames (small, ~400MB total)
        all_tf_resampled_base = {}  # Store base features temporarily

        # Pass 1: Process each timeframe and calculate its breakdown
        print(f"   📊 Pass 1/2: Calculating breakdown at native TF resolutions...")
        for tf in tqdm(HIERARCHICAL_TIMEFRAMES, desc="   Timeframes", leave=True, ncols=100, ascii=True):
            # Columns for this TF: non-channel + shared channel + TF-specific channel
            tf_indices = shared_col_indices + tf_col_indices[tf]
            tf_col_names = non_channel_columns + [feature_columns[i] for i in tf_indices]

            # Add monthly columns for monthly/3month TFs
            if tf in ['monthly', '3month'] and monthly_columns:
                monthly_tf_cols = [mc for mc in monthly_columns if f'_{tf}_' in mc]
                tf_col_names.extend(monthly_tf_cols)

            meta['timeframe_columns'][tf] = tf_col_names

            resampled_chunks = []
            cumulative_row_offset = 0  # Track position in monthly array across chunks

            # Stream through each chunk
            for chunk in chunk_info:
                chunk_path = cache_dir / chunk['path']
                index_path = cache_dir / chunk['index_path']

                # Memory-map chunk (minimal RAM - just the file mapping)
                chunk_array = np.load(str(chunk_path), mmap_mode='r')
                index_array = np.load(str(index_path), mmap_mode='r')

                # Extract only this TF's columns (copies only ~1200 cols, not 14000)
                tf_chunk = chunk_array[:, tf_indices].astype(np.float32)

                # Add monthly data for monthly/3month TFs
                if tf in ['monthly', '3month'] and monthly_array is not None:
                    # Find monthly column indices for this TF
                    monthly_tf_indices = [
                        i for i, mc in enumerate(monthly_columns) if f'_{tf}_' in mc
                    ]
                    if monthly_tf_indices:
                        # Get corresponding rows from monthly shard using cumulative offset
                        chunk_start = cumulative_row_offset
                        chunk_end = chunk_start + len(tf_chunk)
                        monthly_tf_data = monthly_array[chunk_start:chunk_end, monthly_tf_indices].astype(np.float32)
                        if len(monthly_tf_data) == len(tf_chunk):
                            tf_chunk = np.hstack([tf_chunk, monthly_tf_data])

                # Update cumulative offset for next chunk
                cumulative_row_offset += len(chunk_array)

                # Create DataFrame with proper column names and timestamps
                # Start with channel features only (tf_col_names includes non-channel at front)
                channel_col_names = tf_col_names[len(non_channel_columns):]
                df = pd.DataFrame(tf_chunk, columns=channel_col_names[:tf_chunk.shape[1]])
                df.index = pd.to_datetime(index_array, unit='ns')

                # Merge non-channel features for this chunk
                if non_channel_df is not None and len(non_channel_columns) > 0:
                    # Slice non-channel DataFrame to match chunk's timestamp range
                    nc_chunk = non_channel_df.loc[df.index[0]:df.index[-1]]
                    if len(nc_chunk) == len(df):
                        # Prepend non-channel columns (so tsla_close comes first)
                        df = pd.concat([nc_chunk, df], axis=1)

                # Resample this chunk
                tf_rule = TIMEFRAME_RESAMPLE_RULES[tf]
                resampled = df.resample(tf_rule).last().dropna()

                if len(resampled) > 0:
                    resampled_chunks.append(resampled)

                # Free memory immediately
                del chunk_array, index_array, tf_chunk, df, resampled

            # Concatenate all resampled chunks for this TF
            if resampled_chunks:
                final_df = pd.concat(resampled_chunks, axis=0)
                # Handle overlapping timestamps at chunk boundaries
                final_df = final_df[~final_df.index.duplicated(keep='last')]
                final_df = final_df.sort_index()

                # v5.3.3: Calculate breakdown at native TF resolution (after resampling)
                # Note: events_handler not available in chunked mode (set to None)
                breakdown_tf = self._calculate_breakdown_at_native_tf(
                    final_df,
                    tf=tf,
                    raw_df=None,          # Not available in chunked mode
                    events_handler=None   # Not available in chunked mode
                )

                # Store breakdown for cross-TF broadcast (Pass 2)
                all_tf_breakdown[tf] = breakdown_tf

                # Save base features to temp file (will add cross-TF breakdown in Pass 2)
                temp_path = output_cache_dir / f"tf_sequence_{tf}_{meta['cache_key']}_temp.npy"
                np.save(temp_path, final_df.values.astype(np.float32))

                # Save timestamps (final location - doesn't change in Pass 2)
                ts_path = output_cache_dir / f"tf_timestamps_{tf}_{meta['cache_key']}.npy"
                np.save(ts_path, final_df.index.view(np.int64))

                # Store column names and index for Pass 2
                all_tf_resampled_base[tf] = {
                    'columns': list(final_df.columns),
                    'index': final_df.index.copy(),
                    'shape': final_df.shape
                }

                seq_len = TIMEFRAME_SEQUENCE_LENGTHS[tf]
                print(f"      {tf}: {final_df.shape[0]:,} bars × {final_df.shape[1]} base features (seq_len={seq_len})")

                del final_df, resampled_chunks

            # Force garbage collection after each TF
            gc.collect()

        # Pass 2: Add cross-TF breakdown features to each file
        print(f"\n   💾 Pass 2/2: Adding cross-TF breakdown features...")
        for tf in tqdm(HIERARCHICAL_TIMEFRAMES, desc="   Adding cross-TF", leave=False, ncols=100, ascii=True):
            if tf not in all_tf_resampled_base:
                continue

            base_info = all_tf_resampled_base[tf]
            temp_path = output_cache_dir / f"tf_sequence_{tf}_{meta['cache_key']}_temp.npy"

            # Load base features
            base_array = np.load(str(temp_path))
            base_df = pd.DataFrame(base_array, columns=base_info['columns'], index=base_info['index'])

            # Add breakdown from ALL TFs (reindexed to match this TF's resolution)
            for other_tf, other_breakdown in all_tf_breakdown.items():
                if len(other_breakdown) > 0:
                    # Reindex other TF's breakdown to match current TF's index
                    # Use forward-fill (ffill) to broadcast coarser→finer (e.g., daily→5min)
                    breakdown_aligned = other_breakdown.reindex(base_df.index, method='ffill')
                    # Concat horizontally (add columns)
                    base_df = pd.concat([base_df, breakdown_aligned], axis=1, copy=False)

            # Remove duplicate columns (from current TF's own breakdown already added)
            base_df = base_df.loc[:, ~base_df.columns.duplicated(keep='first')]

            # Save final file with all cross-TF breakdown
            output_path = output_cache_dir / f"tf_sequence_{tf}_{meta['cache_key']}.npy"
            np.save(output_path, base_df.values.astype(np.float32))

            # Update metadata with final column count
            meta['timeframe_columns'][tf] = list(base_df.columns)
            meta['timeframe_shapes'][tf] = list(base_df.shape)

            seq_len = TIMEFRAME_SEQUENCE_LENGTHS[tf]
            print(f"      {tf}: {base_df.shape[0]:,} bars × {base_df.shape[1]} features (with cross-TF breakdown)")

            # Clean up temp file
            temp_path.unlink()
            del base_array, base_df

            gc.collect()

        # Clean up memory
        del all_tf_breakdown, all_tf_resampled_base
        gc.collect()

        # Save metadata
        meta_path = output_cache_dir / f"tf_meta_{meta['cache_key']}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"   ✓ Generated {len(HIERARCHICAL_TIMEFRAMES)} timeframe sequences (streaming)")
        print(f"   📄 Metadata: {meta_path.name}")

    def _extract_price_features(self, df: pd.DataFrame, use_gpu: bool = False) -> pd.DataFrame:
        """Extract basic price features. Returns DataFrame with 12 columns (v3.8: +2 normalized prices).

        Args:
            df: OHLCV DataFrame
            use_gpu: If True, use GPU-accelerated rolling statistics (CUDA only)
        """
        price_features = {}

        # GPU acceleration check - use_gpu flag comes from extract_features()
        # which resolves it based on GPU_ROLLING_AVAILABLE and user device selection
        if use_gpu and GPU_ROLLING_AVAILABLE:
            gpu_roller = CUDARollingStats(device='cuda')

        for symbol in ['spy', 'tsla']:
            close_col = f'{symbol}_close'
            price_features[close_col] = df[close_col]

            # Returns (needed for volatility)
            returns = df[close_col].pct_change()
            price_features[f'{symbol}_returns'] = returns
            price_features[f'{symbol}_log_returns'] = np.log(df[close_col] / df[close_col].shift(1))

            if use_gpu and GPU_ROLLING_AVAILABLE:
                # GPU path: batch compute all rolling operations
                close_data = df[close_col].values
                returns_data = returns.values

                # Rolling min/max for normalization
                rolling_results = gpu_roller.rolling_stats(
                    close_data, windows=[252], stats=['min', 'max']
                )
                rolling_min = pd.Series(rolling_results['min_252'], index=df.index)
                rolling_max = pd.Series(rolling_results['max_252'], index=df.index)

                # Volatility (std of returns)
                vol_results = gpu_roller.rolling_stats(
                    returns_data, windows=[10, 50], stats=['std']
                )
                price_features[f'{symbol}_volatility_10'] = vol_results['std_10']
                price_features[f'{symbol}_volatility_50'] = vol_results['std_50']
            else:
                # CPU path: original pandas implementation
                rolling_min = df[close_col].rolling(window=252, min_periods=20).min()
                rolling_max = df[close_col].rolling(window=252, min_periods=20).max()

                # Volatility
                price_features[f'{symbol}_volatility_10'] = returns.rolling(10).std()
                price_features[f'{symbol}_volatility_50'] = returns.rolling(50).std()

            # Normalized price (common to both paths)
            price_range = rolling_max - rolling_min
            price_features[f'{symbol}_close_norm'] = ((df[close_col] - rolling_min) / price_range).fillna(0.5)

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
            # Use unified cache directory if set, otherwise default
            if hasattr(self, '_unified_cache_dir'):
                cache_dir = self._unified_cache_dir
            else:
                cache_dir = Path('data/feature_cache')
                cache_dir.mkdir(exist_ok=True)

            # Use unified cache key if available (includes horizon), otherwise generate
            if hasattr(self, '_cache_key'):
                cache_key = self._cache_key
            else:
                # Create cache key from version + data range + horizon (version ensures cache invalidation when logic changes)
                cache_key = f"{FEATURE_VERSION}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}_{len(df)}_h24"

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

                        # Validate loaded data (v3.15: Dynamic column count)
                        num_cols = len(result.columns)

                        # Calculate expected based on multi-window system
                        num_windows = len(config.CHANNEL_WINDOW_SIZES)  # 21 windows
                        features_per_window = 28  # OHLC slopes, r-squared, position, etc.
                        timeframes = 11  # 5min, 15min, ..., monthly, 3month
                        stocks = 2  # TSLA, SPY
                        expected_cols = num_windows * features_per_window * timeframes * stocks  # = 12,936

                        # Allow some tolerance (±10%) for version differences
                        if num_cols < expected_cols * 0.9 or num_cols > expected_cols * 1.1:
                            print(f"   ⚠️  Cache has {num_cols} columns (expected ~{expected_cols})")
                            print(f"   ⚠️  Reason: Wrong version or corrupted - regenerating...")
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
        # v3.18: Skip monthly/3month when chunking (processed separately on full dataset)
        all_timeframes = {
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

        # Filter out monthly/3month if called from chunked extraction
        # (They're processed separately on full dataset to avoid insufficient data)
        if cache_suffix and 'chunk' in str(cache_suffix):
            # Called from chunking - skip long TFs
            timeframes = {k: v for k, v in all_timeframes.items() if k not in ['monthly', '3month']}
            print(f"   ℹ️  Skipping monthly/3month (processed separately on full dataset)")
        else:
            timeframes = all_timeframes

        # Process both TSLA and SPY
        total_calcs = len(timeframes) * 2  # timeframes × 2 stocks

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

        # v3.20: GPU rolling stats (price/correlation) can run alongside parallel channel extraction
        # GPU rolling stats are for _extract_price_features() and _extract_correlation_features()
        # Channel extraction is CPU-bound linear regression - parallel CPU is still fastest
        use_parallel = parallel_enabled and not is_live_mode

        # Notify user of processing mode and time estimate
        if not use_parallel:
            reasons = []
            if is_live_mode:
                reasons.append("live mode (stability)")
            if not parallel_enabled:
                reasons.append("disabled in config")

            print(f"   ℹ️  Sequential processing: {', '.join(reasons)}")
            print(f"   ⏱️  Using multi-window OHLC channels ({len(config.CHANNEL_WINDOW_SIZES)} windows per timeframe)")
        else:
            requested = config.MAX_PARALLEL_WORKERS if hasattr(config, 'MAX_PARALLEL_WORKERS') else 0
            cores_to_use = get_safe_worker_count(requested if requested > 0 else None)
            gpu_note = " + GPU rolling stats" if use_gpu else ""
            print(f"   🚀 Parallel processing: using {cores_to_use} of {n_cores} available cores{gpu_note}")
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

            # FIX 2: Clean up intermediate data after DataFrame creation
            del all_channel_data
            del tasks, tsla_ohlcv, spy_ohlcv, timestamps
            gc.collect()

        else:
            # Sequential processing with multi-window (for GPU mode or live mode)
            # v3.19: Try RICH for better progress display
            try:
                from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
                from rich.console import Console

                console = Console()
                progress_ctx = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("({task.completed}/{task.total})"),
                    TimeRemainingColumn(),
                    console=console,
                    expand=True,
                    refresh_per_second=10
                )
                calc_progress = progress_ctx.__enter__()

                # Create task for each TF (status board - shows sequential queue)
                tf_tasks = {}
                for sym in ['tsla', 'spy']:
                    for tf in timeframes.keys():
                        task_name = f"{sym}_{tf}"
                        tf_tasks[task_name] = calc_progress.add_task(
                            f"[dim]{task_name}",  # Dim initially (waiting)
                            total=100,
                            visible=True
                        )

                using_rich = True
            except:
                # Fallback to tqdm
                calc_progress = tqdm(total=total_calcs, desc="   Sequential multi-window channels", ncols=100, leave=False, ascii=True, mininterval=0.5)
                tf_tasks = None
                using_rich = False
                progress_ctx = None

            channel_features = {}  # Initialize

            for symbol in ['tsla', 'spy']:
                for tf_name, tf_rule in timeframes.items():
                    task_name = f"{symbol}_{tf_name}"

                    # Update status: Mark current as active
                    if using_rich and tf_tasks:
                        calc_progress.update(
                            tf_tasks[task_name],
                            description=f"[cyan]⠹ {task_name}",
                            completed=0
                        )
                    # Get data
                    if is_live_mode:
                        # Use native yfinance intervals to avoid resampling gaps
                        if tf_name == '5min':
                            source_data = multi_res_data.get('5min', multi_res_data.get('1min'))
                        elif tf_name == '15min':
                            source_data = multi_res_data.get('15min', multi_res_data.get('1min'))
                        elif tf_name == '30min':
                            source_data = multi_res_data.get('30min', multi_res_data.get('1min'))
                        elif tf_name in ['1h', '2h', '3h', '4h']:
                            source_data = multi_res_data['1hour']
                        elif tf_name == 'daily':
                            source_data = multi_res_data['daily']
                        elif tf_name == 'weekly':
                            source_data = multi_res_data.get('weekly', multi_res_data.get('daily'))
                        elif tf_name in ['monthly', '3month']:
                            source_data = multi_res_data.get('monthly', multi_res_data.get('daily'))
                        else:
                            source_data = multi_res_data['daily']
                        # Optimize: chain column selection with rename to avoid intermediate copy
                        symbol_df = source_data[[c for c in source_data.columns if c.startswith(f'{symbol}_')]].rename(columns=lambda c: c.replace(f'{symbol}_', ''))
                    else:
                        # Optimize: chain column selection with rename to avoid intermediate copy
                        symbol_df = df[[c for c in df.columns if c.startswith(f'{symbol}_')]].rename(columns=lambda c: c.replace(f'{symbol}_', ''))

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
                                                'complete_cycles', 'complete_cycles_0_5pct', 'complete_cycles_1_0pct', 'complete_cycles_3_0pct',
                                                'is_bull', 'is_bear', 'is_sideways',
                                                'quality_score', 'is_valid', 'insufficient_data', 'duration',
                                                'projected_high', 'projected_low']:  # v5.0: Channel projections
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
                                channel_features[f'{w_prefix}_complete_cycles'][indices] = channel.complete_cycles
                                channel_features[f'{w_prefix}_complete_cycles_0_5pct'][indices] = channel.complete_cycles_0_5pct
                                channel_features[f'{w_prefix}_complete_cycles_1_0pct'][indices] = channel.complete_cycles_1_0pct
                                channel_features[f'{w_prefix}_complete_cycles_3_0pct'][indices] = channel.complete_cycles_3_0pct
                                channel_features[f'{w_prefix}_is_bull'][indices] = float(slope_pct > 0.1)
                                channel_features[f'{w_prefix}_is_bear'][indices] = float(slope_pct < -0.1)
                                channel_features[f'{w_prefix}_is_sideways'][indices] = float(abs(slope_pct) <= 0.1)
                                channel_features[f'{w_prefix}_quality_score'][indices] = channel.quality_score
                                channel_features[f'{w_prefix}_is_valid'][indices] = channel.is_valid
                                channel_features[f'{w_prefix}_insufficient_data'][indices] = channel.insufficient_data
                                channel_features[f'{w_prefix}_duration'][indices] = channel.actual_duration

                                # v5.0: Store channel projections (geometric predictions)
                                projected_high_pct = (channel.predicted_high - current_price) / current_price * 100 if current_price > 0 else 0.0
                                projected_low_pct = (channel.predicted_low - current_price) / current_price * 100 if current_price > 0 else 0.0
                                channel_features[f'{w_prefix}_projected_high'][indices] = projected_high_pct
                                channel_features[f'{w_prefix}_projected_low'][indices] = projected_low_pct

                    # Update progress: Mark current as complete
                    if using_rich and tf_tasks:
                        calc_progress.update(
                            tf_tasks[task_name],
                            completed=100,
                            description=f"[green]✓ {task_name}"
                        )
                    else:
                        calc_progress.update(1)

                    # CRITICAL: Clear memory after each timeframe
                    del all_windows
                    if 'resampled' in locals():
                        del resampled
                    if 'symbol_df' in locals():
                        del symbol_df
                    import gc
                    gc.collect()

            # Close progress (RICH context manager or tqdm)
            if using_rich and progress_ctx:
                progress_ctx.__exit__(None, None, None)
            elif not using_rich:
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

    def _extract_monthly_3month_features(self, df: pd.DataFrame, use_cache: bool = True,
                                         shard_storage_path: str = None) -> Optional[pd.DataFrame]:
        """
        Extract monthly/3month features on FULL dataset (v3.18 - Hybrid processing).

        These timeframes are too long for per-chunk processing (18 months → only 18 monthly bars).
        But they're tiny in memory (~500 KB for 10 years), so process on full dataset once.

        Memory: 108 monthly × 31 metrics × 21 windows × 2 symbols = ~141K values = 565 KB
        Cached after first run for instant loading (~1 sec vs 2-3 min calculation)!

        Returns:
            DataFrame with monthly/3month channel features, or None if insufficient data
        """
        if len(df) < 365 * 390:  # Less than 1 year of 1-min data
            print("   ⚠️  Insufficient data for monthly/3month features (need 1+ year)")
            return None

        # Check cache first
        if use_cache:
            if shard_storage_path:
                cache_dir = Path(shard_storage_path)
            elif hasattr(self, '_unified_cache_dir'):
                cache_dir = self._unified_cache_dir
            else:
                cache_dir = Path('data/feature_cache')

            cache_dir.mkdir(exist_ok=True, parents=True)

            # Cache key includes version and data range
            cache_key = f"{FEATURE_VERSION}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}_{len(df)}"
            cache_file = cache_dir / f'monthly_3month_features_{cache_key}.pkl'

            if cache_file.exists():
                try:
                    print(f"\n   📂 Loading cached monthly/3month features...")
                    cached_df = pd.read_pickle(cache_file)
                    print(f"   ✓ Loaded {len(cached_df.columns)} monthly/3month features from cache (saved ~2-3 min!)")
                    return cached_df
                except Exception as e:
                    print(f"   ⚠️  Cache load failed: {e}, regenerating...")
                    if cache_file.exists():
                        cache_file.unlink()

        print("\n   📊 Pre-processing monthly/3month on full dataset (hybrid mode)...")

        long_timeframes = {'monthly': '1ME', '3month': '3ME'}
        all_features = {}

        # Same metrics as regular channels
        metrics = [
            'position', 'upper_dist', 'lower_dist',
            'close_slope', 'high_slope', 'low_slope',
            'close_slope_pct', 'high_slope_pct', 'low_slope_pct',
            'close_r_squared', 'high_r_squared', 'low_r_squared', 'r_squared_avg',
            'channel_width_pct', 'slope_convergence', 'stability',
            'ping_pongs', 'ping_pongs_0_5pct', 'ping_pongs_1_0pct', 'ping_pongs_3_0pct',
            'complete_cycles', 'complete_cycles_0_5pct', 'complete_cycles_1_0pct', 'complete_cycles_3_0pct',
            'is_bull', 'is_bear', 'is_sideways',
            'quality_score', 'is_valid', 'insufficient_data', 'duration'
        ]

        for symbol in ['tsla', 'spy']:
            for tf_name, tf_rule in long_timeframes.items():
                # Extract symbol data
                symbol_df = df[[c for c in df.columns if c.startswith(f'{symbol}_')]].copy()
                symbol_df.columns = [c.replace(f'{symbol}_', '') for c in symbol_df.columns]

                # Resample to target timeframe
                resampled = symbol_df.resample(tf_rule).agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'
                }).dropna()

                print(f"     {symbol}_{tf_name}: {len(resampled)} bars (full dataset, memory: ~{len(resampled) * 31 * 21 * 4 / 1e3:.0f} KB)")

                if len(resampled) < 10:
                    print(f"       ⚠️  Very few bars ({len(resampled)}), channels will have low quality")

                # Calculate multi-window rolling channels
                all_windows = self.channel_calc.calculate_multi_window_rolling(resampled, tf_name)

                # Map to original 1-min timestamps (broadcast each TF bar to all 1-min bars it spans)
                for window, channels_list in all_windows.items():
                    w_prefix = f'{symbol}_channel_{tf_name}_w{window}'

                    # Initialize arrays
                    for metric in metrics:
                        all_features[f'{w_prefix}_{metric}'] = np.zeros(len(df), dtype=config.NUMPY_DTYPE)

                    # Broadcast to 1-min bars
                    for i, channel in enumerate(channels_list):
                        if channel is None:
                            continue

                        timestamp = resampled.index[i]
                        if i < len(resampled) - 1:
                            next_timestamp = resampled.index[i + 1]
                            mask = (df.index >= timestamp) & (df.index < next_timestamp)
                        else:
                            mask = df.index >= timestamp

                        indices = np.where(mask)[0]
                        current_price = resampled['close'].iloc[i]
                        position_data = self.channel_calc.get_channel_position(current_price, channel)

                        # Assign all metrics (matching regular extraction)
                        all_features[f'{w_prefix}_position'][indices] = position_data['position']
                        all_features[f'{w_prefix}_upper_dist'][indices] = position_data['distance_to_upper_pct']
                        all_features[f'{w_prefix}_lower_dist'][indices] = position_data['distance_to_lower_pct']
                        all_features[f'{w_prefix}_close_slope'][indices] = channel.close_slope
                        all_features[f'{w_prefix}_high_slope'][indices] = channel.high_slope
                        all_features[f'{w_prefix}_low_slope'][indices] = channel.low_slope
                        slope_pct = (channel.close_slope / current_price) * 100 if current_price > 0 else 0.0
                        all_features[f'{w_prefix}_close_slope_pct'][indices] = slope_pct
                        high_slope_pct = (channel.high_slope / current_price) * 100 if current_price > 0 else 0.0
                        all_features[f'{w_prefix}_high_slope_pct'][indices] = high_slope_pct
                        low_slope_pct = (channel.low_slope / current_price) * 100 if current_price > 0 else 0.0
                        all_features[f'{w_prefix}_low_slope_pct'][indices] = low_slope_pct
                        all_features[f'{w_prefix}_close_r_squared'][indices] = channel.close_r_squared
                        all_features[f'{w_prefix}_high_r_squared'][indices] = channel.high_r_squared
                        all_features[f'{w_prefix}_low_r_squared'][indices] = channel.low_r_squared
                        all_features[f'{w_prefix}_r_squared_avg'][indices] = channel.r_squared
                        all_features[f'{w_prefix}_channel_width_pct'][indices] = channel.channel_width_pct
                        all_features[f'{w_prefix}_slope_convergence'][indices] = channel.slope_convergence
                        all_features[f'{w_prefix}_stability'][indices] = channel.stability_score
                        all_features[f'{w_prefix}_ping_pongs'][indices] = channel.ping_pongs
                        all_features[f'{w_prefix}_ping_pongs_0_5pct'][indices] = channel.ping_pongs_0_5pct
                        all_features[f'{w_prefix}_ping_pongs_1_0pct'][indices] = channel.ping_pongs_1_0pct
                        all_features[f'{w_prefix}_ping_pongs_3_0pct'][indices] = channel.ping_pongs_3_0pct
                        all_features[f'{w_prefix}_complete_cycles'][indices] = channel.complete_cycles
                        all_features[f'{w_prefix}_complete_cycles_0_5pct'][indices] = channel.complete_cycles_0_5pct
                        all_features[f'{w_prefix}_complete_cycles_1_0pct'][indices] = channel.complete_cycles_1_0pct
                        all_features[f'{w_prefix}_complete_cycles_3_0pct'][indices] = channel.complete_cycles_3_0pct
                        all_features[f'{w_prefix}_is_bull'][indices] = float(slope_pct > 0.1)
                        all_features[f'{w_prefix}_is_bear'][indices] = float(slope_pct < -0.1)
                        all_features[f'{w_prefix}_is_sideways'][indices] = float(abs(slope_pct) <= 0.1)
                        all_features[f'{w_prefix}_quality_score'][indices] = channel.quality_score
                        all_features[f'{w_prefix}_is_valid'][indices] = channel.is_valid
                        all_features[f'{w_prefix}_insufficient_data'][indices] = channel.insufficient_data
                        all_features[f'{w_prefix}_duration'][indices] = channel.actual_duration

                # FIX 1: Clean up intermediate data after each timeframe
                del symbol_df, resampled, all_windows
                gc.collect()

        result_df = pd.DataFrame(all_features, index=df.index)

        # FIX 2: Free the massive dict (~15.6 GB) now that DataFrame is created
        del all_features
        gc.collect()

        print(f"   ✓ Monthly/3month features: {len(result_df.columns)} columns, memory: {result_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

        # Save to cache for next time
        if use_cache and 'cache_file' in locals():
            try:
                print(f"   💾 Caching monthly/3month features to: {cache_file.name}")
                result_df.to_pickle(cache_file)
                print(f"   ✓ Cache saved (will load instantly next time!)")
            except Exception as e:
                print(f"   ⚠️  Could not save cache: {e}")

        return result_df

    def _extract_channel_features_chunked(
        self,
        df: pd.DataFrame,
        multi_res_data: dict = None,
        use_gpu: bool = False,
        chunk_size_years: int = 1,
        shard_storage_path: str = None
    ) -> dict:
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

        # v3.19: Pre-process monthly/3month on full dataset → SAVE TO SEPARATE SHARD (zero RAM!)
        # This solves: 1) Insufficient data in chunks, 2) RAM explosion from DataFrame

        # Use persistent directory (respects user's shard_storage_path choice)
        if shard_storage_path:
            monthly_shard_dir = Path(shard_storage_path) / "monthly_shards"
        else:
            monthly_shard_dir = Path('data/feature_cache') / "monthly_shards"
        monthly_shard_dir.mkdir(exist_ok=True, parents=True)

        # Use cache key for uniqueness + version for invalidation
        if hasattr(self, '_cache_key'):
            cache_suffix = self._cache_key
        else:
            cache_suffix = f"{FEATURE_VERSION}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}_{len(df)}_h24"

        monthly_shard_path = monthly_shard_dir / f"monthly_3month_shard_{cache_suffix}.npy"
        monthly_shard_info = None

        # Base cache dir for relative paths (monthly_shard_dir is {base}/monthly_shards)
        base_cache_dir = monthly_shard_dir.parent

        if monthly_shard_path.exists():
            # Load existing shard
            print(f"   📂 Loading cached monthly/3month shard: {monthly_shard_path.name}")
            monthly_array = np.load(monthly_shard_path, mmap_mode='r')
            monthly_shard_info = {
                'path': str(monthly_shard_path.relative_to(base_cache_dir)),  # Relative path for portability
                'rows': monthly_array.shape[0],
                'cols': monthly_array.shape[1],
                'type': 'monthly_3month'
            }
            print(f"   ✓ Loaded monthly/3month: {monthly_array.shape[0]:,} rows × {monthly_array.shape[1]} cols")
            del monthly_array  # Release mmap reference
        else:
            # Calculate and save to shard
            print("\n   🔄 Calculating monthly/3month on full dataset (108 bars, high quality)...")
            monthly_3month_features = self._extract_monthly_3month_features(
                df,
                use_cache=False,  # Don't pickle cache, we're sharding now
                shard_storage_path=None  # Not using pickle cache anymore
            )

            if monthly_3month_features is not None and len(monthly_3month_features) > 0:
                # FIX 3: Avoid holding DataFrame + numpy array simultaneously (~31 GB peak)
                # Save shape info before conversion
                monthly_rows = len(monthly_3month_features)
                monthly_cols = monthly_3month_features.shape[1]
                monthly_columns = list(monthly_3month_features.columns)  # v4.1: Save column names

                # Convert to numpy and delete DataFrame IMMEDIATELY
                monthly_array = monthly_3month_features.values.astype(config.NUMPY_DTYPE)
                del monthly_3month_features  # Free ~15.6 GB before np.save
                gc.collect()

                # Now save (only monthly_array in memory, not both)
                np.save(monthly_shard_path, monthly_array)

                monthly_shard_info = {
                    'path': str(monthly_shard_path.relative_to(base_cache_dir)),  # Relative path for portability
                    'rows': monthly_rows,
                    'cols': monthly_cols,
                    'columns': monthly_columns,  # v4.1: Column names for native TF generation
                    'type': 'monthly_3month'
                }

                print(f"   ✓ Saved monthly/3month shard: {monthly_array.shape} ({monthly_array.nbytes / 1e9:.2f} GB on disk)")

                # Final cleanup
                del monthly_array
                gc.collect()
                print(f"   ✓ Monthly/3month data freed from RAM")
            else:
                print(f"   ⚠️  No monthly/3month features generated")

        # Calculate chunk boundaries
        start_date = df.index[0]
        end_date = df.index[-1]
        total_years = (end_date - start_date).days / 365.25

        print(f"     Total period: {start_date.date()} to {end_date.date()} ({total_years:.1f} years)")

        # Create chunk date ranges
        # Use YS (Year Start) but prepend actual start_date to avoid losing partial first year
        # Bug fix: pd.date_range with freq='1YS' skips to next Jan 1st, dropping data before that
        chunk_starts = pd.date_range(start=start_date, end=end_date, freq=f'{chunk_size_years}YS')
        # Prepend start_date if it's before the first YS boundary (e.g., 2015-01-02 < 2016-01-01)
        if len(chunk_starts) == 0 or chunk_starts[0] > start_date:
            chunk_starts = pd.DatetimeIndex([start_date]).append(chunk_starts)
        if chunk_starts[-1] < end_date:
            chunk_starts = chunk_starts.append(pd.DatetimeIndex([end_date]))

        overlap_months = config.CHUNK_OVERLAP_MONTHS

        # Create temp directory for sharded .npy files (memory-mapped, zero RAM spike)
        import time
        import os
        import json

        # Use custom path or default
        if shard_storage_path:
            cache_dir = Path(shard_storage_path)
        else:
            cache_dir = Path('data/feature_cache')

        cache_dir.mkdir(exist_ok=True, parents=True)

        timestamp = int(time.time())
        temp_dir = cache_dir / f"mmap_chunks_{timestamp}"
        temp_dir.mkdir(exist_ok=False)  # Fail if somehow exists

        print(f"     Using sharded memory-mapped storage: {temp_dir.name}")
        print(f"     Precision: {config.NUMPY_DTYPE} (from config)")

        chunk_info = []  # Will store metadata for each chunk
        feature_columns = None  # Will be set from first chunk

        for i in range(len(chunk_starts) - 1):
            chunk_start = chunk_starts[i]
            chunk_end = chunk_starts[i + 1]

            # Add overlap for lookback (except for first chunk)
            if i > 0:
                chunk_start_with_overlap = chunk_start - pd.DateOffset(months=overlap_months)
            else:
                chunk_start_with_overlap = chunk_start

            # Extract chunk
            # Use <= for last chunk to include the final row, < for others to avoid overlap
            is_last_chunk = (i == len(chunk_starts) - 2)
            if is_last_chunk:
                chunk_df = df[(df.index >= chunk_start_with_overlap) & (df.index <= chunk_end)].copy()
            else:
                chunk_df = df[(df.index >= chunk_start_with_overlap) & (df.index < chunk_end)].copy()
            chunk_multi_res = None
            if multi_res_data:
                if is_last_chunk:
                    chunk_multi_res = {
                        tf: mdf[(mdf.index >= chunk_start_with_overlap) & (mdf.index <= chunk_end)].copy()
                        for tf, mdf in multi_res_data.items()
                    }
                else:
                    chunk_multi_res = {
                        tf: mdf[(mdf.index >= chunk_start_with_overlap) & (mdf.index < chunk_end)].copy()
                        for tf, mdf in multi_res_data.items()
                    }

            print(f"\n     Chunk {i+1}/{len(chunk_starts)-1}: {chunk_start.date()} to {chunk_end.date()}")
            print(f"       Bars: {len(chunk_df):,} (including {overlap_months}mo overlap)")

            # Process chunk (no cache for individual chunks)
            # v3.18: Pass cache_suffix='chunk' to signal skipping monthly/3month
            chunk_features = self._extract_channel_features(
                chunk_df,
                multi_res_data=chunk_multi_res,
                use_cache=False,  # Don't cache individual chunks
                use_gpu=use_gpu,
                cache_suffix='chunk_skip_long_tfs'  # Signals to skip monthly/3month
            )

            # FIX 1: Free chunk_df and chunk_multi_res immediately (no longer needed)
            del chunk_df
            if chunk_multi_res:
                del chunk_multi_res
            gc.collect()

            # Remove overlap from results (keep only the actual chunk period)
            chunk_features = chunk_features[chunk_features.index >= chunk_start]

            print(f"       Result: {len(chunk_features):,} bars after trimming overlap")
            print(f"       Memory: ~{chunk_features.memory_usage(deep=True).sum() / 1e6:.1f} MB")

            # Save as memory-mapped .npy shard (respects dtype from config!)
            chunk_path = temp_dir / f"chunk_{i:04d}.npy"
            index_path = temp_dir / f"chunk_{i:04d}_index.npy"

            # FIX 3: Save index first, then convert DataFrame to numpy and free DataFrame immediately
            # This avoids holding both DataFrame (~5GB) and numpy array (~5GB) simultaneously
            index_values = chunk_features.index.values.copy()  # Small - just timestamps
            chunk_shape = chunk_features.shape  # Save shape before deleting

            # Capture column names from first chunk (all chunks have same columns)
            if feature_columns is None:
                feature_columns = list(chunk_features.columns)

            chunk_array = chunk_features.values.astype(config.NUMPY_DTYPE)
            del chunk_features  # Free DataFrame immediately (~5GB freed)
            gc.collect()

            print(f"       Saving shard {i} as .npy...")
            np.save(chunk_path, chunk_array)
            np.save(index_path, index_values)

            # Store metadata (use relative paths for portability across machines)
            chunk_info.append({
                'path': str(chunk_path.relative_to(cache_dir)),
                'index_path': str(index_path.relative_to(cache_dir)),
                'rows': len(chunk_array),
                'cols': chunk_array.shape[1],
                'start_date': str(chunk_start.date()),
                'end_date': str(chunk_end.date())
            })

            print(f"       ✓ Shard saved: {chunk_array.nbytes / 1e6:.1f} MB on disk")

            # Final cleanup - only chunk_array remains
            del chunk_array, index_values
            gc.collect()

        # Save metadata for memory-mapped loading (NO RAM spike!)
        total_rows = sum(c['rows'] for c in chunk_info)
        num_features = chunk_info[0]['cols'] if chunk_info else 0

        # Use unified cache key for metadata file (includes horizon)
        if hasattr(self, '_cache_key'):
            cache_suffix = self._cache_key
        else:
            cache_suffix = f"{FEATURE_VERSION}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}_{len(df)}_h24"

        meta_path = cache_dir / f"features_mmap_meta_{cache_suffix}.json"

        metadata = {
            'chunk_info': chunk_info,
            'monthly_3month_shard': monthly_shard_info,  # v3.19: Separate shard for monthly/3month
            'feature_columns': feature_columns,  # v4.1: Column names for native TF generation
            'num_features': num_features + (monthly_shard_info['cols'] if monthly_shard_info else 0),
            'dtype': np.dtype(config.NUMPY_DTYPE).name,
            'total_rows': total_rows,
            'version': FEATURE_VERSION,
            'temp_dir': str(temp_dir),
            'cache_key': cache_suffix  # v4.1: Add cache key for native TF generation
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n     ✅ Sharded extraction complete:")
        print(f"        Shards: {len(chunk_info)}")
        print(f"        Total rows: {total_rows:,}")
        print(f"        Features: {num_features:,}")
        print(f"        Dtype: {config.NUMPY_DTYPE}")
        print(f"        Metadata: {meta_path.name}")
        print(f"        Disk usage: ~{sum(c['rows'] * c['cols'] * (8 if config.NUMPY_DTYPE == np.float64 else 4) for c in chunk_info) / 1e9:.1f} GB")

        gc.collect()

        # Return metadata path instead of DataFrame (zero RAM spike!)
        # v3.19: Monthly/3month now in separate shard (not in RAM!)
        return {
            'mmap_meta_path': str(meta_path),
            'type': 'mmap_sharded',
            'monthly_3month_features': None  # Not in RAM - saved to separate shard
        }

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
                f'{prefix}_complete_cycles': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_complete_cycles_0_5pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_complete_cycles_1_0pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_complete_cycles_3_0pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
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

                    # v3.17: Find best channel regardless of quality (no filtering)
                    # Even "bad" channels are stored - model learns when to ignore them
                    channel = channel_calc.find_best_channel_any_quality(
                        available_window,
                        timeframe=tf_name,
                        max_lookback=dynamic_lookback
                    )

                    # Only skip if truly no data (very rare)
                    if channel is None:
                        continue  # Insufficient data, not quality issue

                    current_price = resampled['close'].iloc[i]
                    position_data = channel_calc.get_channel_position(current_price, channel)

                    # Get the actual window used
                    actual_window = resampled.iloc[i-channel.actual_duration:i]

                    # Calculate multi-threshold ping-pongs and complete cycles
                    window_prices = actual_window['close'].values
                    multi_pp = channel_calc._detect_ping_pongs_multi_threshold(
                        window_prices,
                        channel.upper_line,
                        channel.lower_line,
                        thresholds=[0.005, 0.01, 0.02, 0.03]
                    )

                    # v3.17: Complete cycles (full round-trips)
                    multi_cycles = channel_calc._detect_complete_cycles_multi_threshold(
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

                        # v3.17: Multi-threshold complete cycles
                        results[f'{prefix}_complete_cycles_0_5pct'][idx] = multi_cycles[0.005]
                        results[f'{prefix}_complete_cycles_1_0pct'][idx] = multi_cycles[0.01]
                        results[f'{prefix}_complete_cycles'][idx] = multi_cycles[0.02]  # Default 2%
                        results[f'{prefix}_complete_cycles_3_0pct'][idx] = multi_cycles[0.03]

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
                f'{prefix}_complete_cycles': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_complete_cycles_0_5pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_complete_cycles_1_0pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
                f'{prefix}_complete_cycles_3_0pct': np.zeros(n, dtype=config.NUMPY_DTYPE),
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
            'complete_cycles': np.zeros(num_original_rows),  # v3.17: Complete round-trips at 2%
            'complete_cycles_0_5pct': np.zeros(num_original_rows),
            'complete_cycles_1_0pct': np.zeros(num_original_rows),
            'complete_cycles_3_0pct': np.zeros(num_original_rows),
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

                # v3.17: Find best channel regardless of quality (no filtering)
                # Even "bad" channels are stored - model learns when to ignore them
                channel = self.channel_calc.find_best_channel_any_quality(
                    available_window,
                    timeframe=tf_name,
                    max_lookback=dynamic_lookback
                )

                # Only skip if truly no data (very rare)
                if channel is None:
                    # Insufficient data (window too large for available data)
                    continue

                current_price = resampled_df['close'].iloc[i]
                position_data = self.channel_calc.get_channel_position(current_price, channel)

                # Get the actual window used for this channel
                actual_window = resampled_df.iloc[i-channel.actual_duration:i]

                # Calculate multi-threshold ping-pongs and complete cycles
                window_prices = actual_window['close'].values
                multi_pp = self.channel_calc._detect_ping_pongs_multi_threshold(
                    window_prices,
                    channel.upper_line,
                    channel.lower_line,
                    thresholds=[0.005, 0.01, 0.02, 0.03]
                )

                # v3.17: Complete cycles (full round-trips)
                multi_cycles = self.channel_calc._detect_complete_cycles_multi_threshold(
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
                results['complete_cycles'][mask] = channel.complete_cycles  # v3.17: 2% threshold
                results['complete_cycles_0_5pct'][mask] = multi_cycles[0.005]
                results['complete_cycles_1_0pct'][mask] = multi_cycles[0.01]
                results['complete_cycles_3_0pct'][mask] = multi_cycles[0.03]
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
        # v3.17: All arrays use config.NUMPY_DTYPE for float32/float64 toggleability
        results = {
            'position': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),
            'upper_dist': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),
            'lower_dist': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),
            'slope': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),  # Raw slope ($/bar)
            'slope_pct': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),  # Normalized slope (% per bar)
            'stability': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),
            'ping_pongs': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),  # Default 2% threshold
            'ping_pongs_0_5pct': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),  # 0.5% threshold
            'ping_pongs_1_0pct': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),  # 1.0% threshold
            'ping_pongs_3_0pct': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),  # 3.0% threshold
            'complete_cycles': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),  # v3.17: Complete round-trips at 2%
            'complete_cycles_0_5pct': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),
            'complete_cycles_1_0pct': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),
            'complete_cycles_3_0pct': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),
            'r_squared': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),
            'is_bull': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),  # Uptrending channel
            'is_bear': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),  # Downtrending channel
            'is_sideways': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE),  # Ranging channel
            'duration': np.zeros(num_original_rows, dtype=config.NUMPY_DTYPE)  # v3.11: Channel duration (fixed lookback for GPU)
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
                    # HYBRID DATA SELECTION: Use native yfinance intervals for live mode
                    if is_live_mode:
                        # Use native yfinance intervals to avoid resampling gaps
                        if tf_name == '5min':
                            source_data = multi_res_data.get('5min', multi_res_data.get('1min'))
                        elif tf_name == '15min':
                            source_data = multi_res_data.get('15min', multi_res_data.get('1min'))
                        elif tf_name == '30min':
                            source_data = multi_res_data.get('30min', multi_res_data.get('1min'))
                        elif tf_name in ['1h', '2h', '3h', '4h']:
                            source_data = multi_res_data['1hour']
                        elif tf_name == 'daily':
                            source_data = multi_res_data['daily']
                        elif tf_name == 'weekly':
                            source_data = multi_res_data.get('weekly', multi_res_data.get('daily'))
                        elif tf_name in ['monthly', '3month']:
                            source_data = multi_res_data.get('monthly', multi_res_data.get('daily'))
                        else:
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

    def _extract_correlation_features(self, df: pd.DataFrame, use_gpu: bool = False) -> pd.DataFrame:
        """Extract SPY-TSLA correlation and divergence features. Returns DataFrame with 5 columns.

        Args:
            df: OHLCV DataFrame
            use_gpu: If True, use GPU-accelerated rolling correlation (CUDA only)
        """
        spy_returns = df['spy_close'].pct_change()
        tsla_returns = df['tsla_close'].pct_change()

        # GPU acceleration check - use_gpu flag comes from extract_features()
        # which resolves it based on GPU_ROLLING_AVAILABLE and user device selection
        if use_gpu and GPU_ROLLING_AVAILABLE:
            # GPU path: compute correlations on CUDA
            gpu_roller = CUDARollingStats(device='cuda')
            corr_results = gpu_roller.rolling_correlation(
                spy_returns.values, tsla_returns.values,
                windows=[10, 50, 200]
            )

            correlation_features = {
                'correlation_10': corr_results['corr_10'],
                'correlation_50': corr_results['corr_50'],
                'correlation_200': corr_results['corr_200'],
                'divergence': (((spy_returns > 0) & (tsla_returns < 0)) |
                              ((spy_returns < 0) & (tsla_returns > 0))).astype(float),
                'divergence_magnitude': abs(spy_returns - tsla_returns)
            }
        else:
            # CPU path: original pandas implementation
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

    def _extract_vix_features(self, df: pd.DataFrame, vix_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract VIX-based features for volatility regime detection (v3.20).

        Args:
            df: Main DataFrame with TSLA/SPY data (must have tsla_close, spy_close)
            vix_data: Optional VIX DataFrame. If None, returns zeros for all VIX features.
                     Expected columns: vix_close (or CLOSE), vix_high, vix_low
                     Index should be DatetimeIndex aligned with df

        Returns:
            DataFrame with ~15 VIX features. If vix_data is None or empty,
            returns zeros (backward compatible with models trained without VIX).

        Features (15 total):
            - vix_level: Current VIX close (normalized by typical range 10-40)
            - vix_percentile_20d: 20-day percentile rank
            - vix_percentile_252d: 252-day (yearly) percentile rank
            - vix_change_1d: 1-day percentage change
            - vix_change_5d: 5-day percentage change
            - vix_regime: Categorical (0=low <15, 1=normal 15-20, 2=elevated 20-30, 3=extreme >30)
            - vix_tsla_corr_20d: 20-day rolling correlation (VIX changes vs TSLA returns)
            - vix_spy_corr_20d: 20-day rolling correlation (VIX changes vs SPY returns)
            - vix_momentum_10d: 10-day rate of change
            - vix_ma_ratio: Ratio of VIX to 20-day MA (spike detection)
            - vix_high_low_range: Daily high-low range normalized
            - vix_trend_20d: 20-day slope of VIX (linear regression)
            - vix_above_20: Binary flag if VIX > 20
            - vix_above_30: Binary flag if VIX > 30
            - vix_spike: Binary flag for >15% single-day spike
        """
        num_rows = len(df)
        vix_features = {}

        # Check if we have VIX data
        has_vix = vix_data is not None and len(vix_data) > 0

        if has_vix:
            # Normalize column names (handle both raw CSV and processed formats)
            vix_df = vix_data.copy()
            if 'CLOSE' in vix_df.columns:
                vix_df = vix_df.rename(columns={'CLOSE': 'vix_close', 'HIGH': 'vix_high', 'LOW': 'vix_low', 'OPEN': 'vix_open'})

            # Align VIX data with main df index
            # VIX is daily, so we forward-fill to match intraday data
            vix_aligned = vix_df.reindex(df.index, method='ffill')

            vix_close = vix_aligned['vix_close'] if 'vix_close' in vix_aligned.columns else pd.Series(np.nan, index=df.index)
            vix_high = vix_aligned['vix_high'] if 'vix_high' in vix_aligned.columns else vix_close
            vix_low = vix_aligned['vix_low'] if 'vix_low' in vix_aligned.columns else vix_close

            # Fill any remaining NaNs (at start before VIX data begins)
            vix_close = vix_close.bfill().fillna(20.0)  # Default to 20 (normal level)
            vix_high = vix_high.bfill().fillna(20.0)
            vix_low = vix_low.bfill().fillna(20.0)

            # 1. VIX level (normalized: divide by 40 to get 0-1 range for typical VIX)
            vix_features['vix_level'] = (vix_close / 40.0).clip(0, 2.5).values

            # 2. VIX percentile ranks
            def rolling_percentile(series, window):
                """Calculate rolling percentile rank (0-1)"""
                return series.rolling(window, min_periods=max(1, window // 4)).apply(
                    lambda x: (x.values[-1] - x.min()) / (x.max() - x.min() + 1e-8) if len(x) > 1 else 0.5,
                    raw=False
                )

            vix_features['vix_percentile_20d'] = rolling_percentile(vix_close, 20).fillna(0.5).values
            vix_features['vix_percentile_252d'] = rolling_percentile(vix_close, 252).fillna(0.5).values

            # 3. VIX changes
            vix_pct_change = vix_close.pct_change()
            vix_features['vix_change_1d'] = vix_pct_change.fillna(0).clip(-0.5, 0.5).values
            vix_features['vix_change_5d'] = vix_close.pct_change(5).fillna(0).clip(-1.0, 1.0).values

            # 4. VIX regime (categorical: 0=low, 1=normal, 2=elevated, 3=extreme)
            regime = np.zeros(num_rows)
            regime[vix_close.values < 15] = 0  # Low volatility
            regime[(vix_close.values >= 15) & (vix_close.values < 20)] = 1  # Normal
            regime[(vix_close.values >= 20) & (vix_close.values < 30)] = 2  # Elevated
            regime[vix_close.values >= 30] = 3  # Extreme
            vix_features['vix_regime'] = regime / 3.0  # Normalize to 0-1

            # 5. Correlations with TSLA and SPY
            tsla_returns = df['tsla_close'].pct_change().fillna(0)
            spy_returns = df['spy_close'].pct_change().fillna(0)

            # Rolling correlation (VIX typically moves inverse to stocks)
            vix_features['vix_tsla_corr_20d'] = vix_pct_change.rolling(20, min_periods=5).corr(tsla_returns).fillna(0).values
            vix_features['vix_spy_corr_20d'] = vix_pct_change.rolling(20, min_periods=5).corr(spy_returns).fillna(0).values

            # 6. VIX momentum (rate of change)
            vix_features['vix_momentum_10d'] = (vix_close / vix_close.shift(10) - 1).fillna(0).clip(-1, 1).values

            # 7. VIX MA ratio (spike detection)
            vix_ma_20 = vix_close.rolling(20, min_periods=5).mean()
            vix_features['vix_ma_ratio'] = (vix_close / vix_ma_20).fillna(1.0).clip(0.5, 2.0).values

            # 8. VIX high-low range (normalized)
            vix_range = (vix_high - vix_low) / vix_close
            vix_features['vix_high_low_range'] = vix_range.fillna(0).clip(0, 0.5).values

            # 9. VIX trend (20-day slope using linear regression)
            def rolling_slope(series, window):
                """Calculate rolling linear regression slope"""
                def slope(y):
                    if len(y) < 2:
                        return 0
                    x = np.arange(len(y))
                    try:
                        return np.polyfit(x, y, 1)[0]
                    except:
                        return 0
                return series.rolling(window, min_periods=max(5, window // 4)).apply(slope, raw=True)

            vix_slope = rolling_slope(vix_close, 20).fillna(0)
            # Normalize slope (typical daily VIX change is <1 point)
            vix_features['vix_trend_20d'] = (vix_slope / 2.0).clip(-1, 1).values

            # 10. Binary flags
            vix_features['vix_above_20'] = (vix_close > 20).astype(float).values
            vix_features['vix_above_30'] = (vix_close > 30).astype(float).values

            # 11. Spike detection (>15% single-day increase)
            vix_features['vix_spike'] = (vix_pct_change > 0.15).astype(float).values

        else:
            # No VIX data - return zeros for backward compatibility
            vix_features['vix_level'] = np.zeros(num_rows)
            vix_features['vix_percentile_20d'] = np.full(num_rows, 0.5)
            vix_features['vix_percentile_252d'] = np.full(num_rows, 0.5)
            vix_features['vix_change_1d'] = np.zeros(num_rows)
            vix_features['vix_change_5d'] = np.zeros(num_rows)
            vix_features['vix_regime'] = np.full(num_rows, 1.0 / 3.0)  # Default to "normal" regime
            vix_features['vix_tsla_corr_20d'] = np.zeros(num_rows)
            vix_features['vix_spy_corr_20d'] = np.zeros(num_rows)
            vix_features['vix_momentum_10d'] = np.zeros(num_rows)
            vix_features['vix_ma_ratio'] = np.ones(num_rows)
            vix_features['vix_high_low_range'] = np.zeros(num_rows)
            vix_features['vix_trend_20d'] = np.zeros(num_rows)
            vix_features['vix_above_20'] = np.zeros(num_rows)
            vix_features['vix_above_30'] = np.zeros(num_rows)
            vix_features['vix_spike'] = np.zeros(num_rows)

        return pd.DataFrame(vix_features, index=df.index)

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

        # v5.3.2: For native TF mode, we need to load w50 window stability/position from mmap chunks
        # In native TF mode, channel features are in chunks, not in features_df (which only has shared cols)
        # We'll load them from the cache directory if available
        # IMPORTANT: Track added columns so we can remove them after calculation (don't save to cache!)
        _temp_w50_cols_added = []

        if hasattr(self, '_unified_cache_dir'):
            cache_dir = self._unified_cache_dir

            # Try to find and load mmap chunks to get w50 stability/position
            mmap_meta_files = list(cache_dir.glob('features_mmap_meta_*.json'))

            if mmap_meta_files and len(mmap_meta_files) > 0:
                import json

                # Load the mmap metadata
                with open(mmap_meta_files[0], 'r') as f:
                    mmap_meta = json.load(f)

                chunk_info = mmap_meta.get('chunk_info', [])
                # FIX: Use correct key name 'feature_columns' (not 'columns')
                columns = mmap_meta.get('feature_columns', [])

                # Find w50 stability and position columns
                w50_cols = {}
                w50_indices = {}
                for i, col in enumerate(columns):
                    if '_w50_stability' in col or '_w50_position' in col:
                        w50_cols[col] = i
                        w50_indices[i] = col

                if w50_cols:
                    # Load just the w50 columns from mmap chunks
                    channel_data = {col: [] for col in w50_cols.keys()}

                    for chunk in chunk_info:
                        chunk_path = cache_dir / chunk['path']
                        if chunk_path.exists():
                            # Memory-map this chunk
                            chunk_mmap = np.load(chunk_path, mmap_mode='r')

                            # Extract w50 columns
                            for col_name, col_idx in w50_cols.items():
                                channel_data[col_name].append(chunk_mmap[:, col_idx])

                    # Concatenate all chunks and add to features_df temporarily
                    for col in w50_cols.keys():
                        if channel_data[col]:
                            features_df[col] = np.concatenate(channel_data[col])
                            _temp_w50_cols_added.append(col)  # Track for removal later

                # Also load monthly/3month w50 columns from monthly shard
                monthly_shard_info = mmap_meta.get('monthly_3month_shard', {})
                if monthly_shard_info and monthly_shard_info.get('columns'):
                    monthly_cols = monthly_shard_info['columns']
                    monthly_path_rel = monthly_shard_info.get('path')

                    if monthly_path_rel:
                        monthly_path = cache_dir / monthly_path_rel

                        # Find monthly/3month w50 stability and position columns
                        monthly_w50_cols = {}
                        for i, col in enumerate(monthly_cols):
                            # Only get monthly and 3month w50 stability/position
                            if ('_monthly_w50_stability' in col or '_monthly_w50_position' in col or
                                '_3month_w50_stability' in col or '_3month_w50_position' in col):
                                monthly_w50_cols[col] = i

                        if monthly_w50_cols and monthly_path.exists():
                            # Load the monthly shard
                            monthly_mmap = np.load(monthly_path, mmap_mode='r')

                            # Extract w50 columns and add to features_df
                            for col_name, col_idx in monthly_w50_cols.items():
                                features_df[col_name] = monthly_mmap[:, col_idx]
                                _temp_w50_cols_added.append(col_name)

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

        # 3. Channel duration vs historical average
        # v5.3.2: Expanded to ALL 11 TFs with adaptive rolling windows
        # Use stability score as proxy for channel duration

        # Adaptive window sizes based on data availability (maximized where possible)
        # CRITICAL: These are in 1-MIN bars (features_df is 1-min resolution!)
        # To get N bars at TF resolution, multiply by minutes per bar
        adaptive_windows = {
            '5min': 7500,     # 1500 5-min bars × 5 min/bar = 7500 1-min bars (~19 trading days)
            '15min': 6000,    # 400 15-min bars × 15 min/bar = 6000 1-min bars (~15 trading days)
            '30min': 9000,    # 300 30-min bars × 30 min/bar = 9000 1-min bars (~23 trading days)
            '1h': 12000,      # 200 1-hour bars × 60 min/bar = 12000 1-min bars (~31 trading days)
            '2h': 12000,      # 100 2-hour bars × 120 min/bar = 12000 1-min bars (~31 trading days)
            '3h': 14400,      # 80 3-hour bars × 180 min/bar = 14400 1-min bars (~37 trading days)
            '4h': 14400,      # 60 4-hour bars × 240 min/bar = 14400 1-min bars (~37 trading days)
            'daily': 39000,   # 100 daily bars × 390 min/bar = 39000 1-min bars (100 trading days)
            'weekly': 39000,  # 20 weekly bars × 1950 min/bar = 39000 1-min bars (100 trading days)
            'monthly': 128700, # 15 monthly bars × 8580 min/bar = 128700 1-min bars (330 trading days)
            '3month': 206160, # 8 quarters × 25770 min/bar = 206160 1-min bars (528 trading days)
        }

        for tf_name in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            # v5.3.2: In native TF mode, stability has window suffix (e.g., w50)
            # Try both column name formats
            stability_col = f'tsla_channel_{tf_name}_stability'  # Non-native mode
            stability_col_w50 = f'tsla_channel_{tf_name}_w50_stability'  # Native TF mode

            if stability_col in features_df.columns:
                stability = features_df[stability_col]
            elif stability_col_w50 in features_df.columns:
                # Native TF mode - use w50 window as representative
                stability = features_df[stability_col_w50]
            else:
                # Column not found - fill with default
                breakdown_features[f'tsla_channel_duration_ratio_{tf_name}'] = np.ones(num_rows)
                continue

            # Duration ratio = current stability vs rolling average (adaptive window per TF)
            window = adaptive_windows[tf_name]
            avg_stability = stability.rolling(window, min_periods=window//2).mean()
            # v5.3.2 fix: Use larger epsilon (0.01) to prevent extremes from near-zero averages
            # 1e-8 was too small, causing 1.7M extremes when avg_stability was tiny
            duration_ratio = (stability / (avg_stability + 0.01)).fillna(1.0)
            breakdown_features[f'tsla_channel_duration_ratio_{tf_name}'] = duration_ratio.values

        # 4. SPY-TSLA channel alignment
        # v5.3.2: Expanded to ALL 11 TFs for comprehensive break prediction
        for tf_name in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            # v5.3.2: In native TF mode, position has window suffix (e.g., w50)
            # Try both column name formats
            tsla_pos_col = f'tsla_channel_{tf_name}_position'
            spy_pos_col = f'spy_channel_{tf_name}_position'
            tsla_pos_col_w50 = f'tsla_channel_{tf_name}_w50_position'
            spy_pos_col_w50 = f'spy_channel_{tf_name}_w50_position'

            # Try to find position columns (both formats)
            if tsla_pos_col in features_df.columns and spy_pos_col in features_df.columns:
                tsla_pos = features_df[tsla_pos_col]
                spy_pos = features_df[spy_pos_col]
            elif tsla_pos_col_w50 in features_df.columns and spy_pos_col_w50 in features_df.columns:
                # Native TF mode - use w50 window
                tsla_pos = features_df[tsla_pos_col_w50]
                spy_pos = features_df[spy_pos_col_w50]
            else:
                # Columns not found - fill with default
                breakdown_features[f'channel_alignment_spy_tsla_{tf_name}'] = np.zeros(num_rows)
                continue

            # Convert to -1 to +1 range and calculate alignment
            tsla_pos = tsla_pos * 2 - 1  # Convert 0-1 to -1 to +1
            spy_pos = spy_pos * 2 - 1

            # Alignment: both at top (+1) or both at bottom (-1) = high alignment
            alignment = tsla_pos * spy_pos  # -1 to +1
            breakdown_features[f'channel_alignment_spy_tsla_{tf_name}'] = alignment.values

        # 5. Time in channel features (11 timeframes × 2 stocks)
        # Use channel stability as proxy (higher stability = longer time in channel)
        for tf_name in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            for symbol in ['tsla', 'spy']:
                # v5.3.2: Try both column formats (with/without window suffix)
                stability_col = f'{symbol}_channel_{tf_name}_stability'
                stability_col_w50 = f'{symbol}_channel_{tf_name}_w50_stability'

                if stability_col in features_df.columns:
                    stability = features_df[stability_col]
                elif stability_col_w50 in features_df.columns:
                    # Native TF mode - use w50 window
                    stability = features_df[stability_col_w50]
                else:
                    breakdown_features[f'{symbol}_time_in_channel_{tf_name}'] = np.zeros(num_rows)
                    continue

                # Normalize stability to represent "time in channel" score
                time_in_channel = np.clip(stability * 100, 0, 100)  # 0-100 scale (np.clip returns ndarray)
                breakdown_features[f'{symbol}_time_in_channel_{tf_name}'] = np.asarray(time_in_channel)

        # 6. Enhanced normalized channel position (11 timeframes × 2 stocks)
        for tf_name in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            for symbol in ['tsla', 'spy']:
                # v5.3.2: Try both column formats (with/without window suffix)
                pos_col = f'{symbol}_channel_{tf_name}_position'
                pos_col_w50 = f'{symbol}_channel_{tf_name}_w50_position'

                if pos_col in features_df.columns:
                    position = features_df[pos_col]
                elif pos_col_w50 in features_df.columns:
                    # Native TF mode - use w50 window
                    position = features_df[pos_col_w50]
                else:
                    breakdown_features[f'{symbol}_channel_position_norm_{tf_name}'] = np.zeros(num_rows)
                    continue

                # Convert 0-1 position to -1 to +1 (bottom to top)
                position_norm = position * 2 - 1
                breakdown_features[f'{symbol}_channel_position_norm_{tf_name}'] = position_norm.values

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
        # v4.2: Optimized - compute at daily level then forward-fill to 1-min
        # Events only change daily, so checking every minute was wasteful (1.7M → ~3,400 lookups)
        if events_handler is not None:
            import config as cfg

            # Step 1: Get unique dates (vectorized) - ~3,400 dates for 8 years of data
            date_strings = raw_df.index.strftime('%Y-%m-%d')
            unique_dates = sorted(set(date_strings))

            # Step 2: Pre-compute events by date (fast - only ~3,400 lookups instead of 1.7M)
            daily_features = {}
            for date_str in tqdm(unique_dates, desc="      Event features", leave=False, ncols=100, ascii=True):
                try:
                    events = events_handler.get_events_for_date(date_str, lookback_days=cfg.EVENT_LOOKBACK_DAYS)

                    feat = {
                        'is_earnings_week': 0.0,
                        'days_until_earnings': 0.0,
                        'days_until_fomc': 0.0,
                        'is_high_impact_event': 0.0
                    }

                    if events:
                        # Find closest earnings event
                        earnings_events = [e for e in events if e['event_type'] in ['earnings', 'delivery']]
                        if earnings_events:
                            # v4.2 bugfix: correct key is 'days_until_event' not 'days_until'
                            closest = min(earnings_events, key=lambda e: abs(e['days_until_event']))
                            feat['days_until_earnings'] = closest['days_until_event']
                            feat['is_earnings_week'] = float(abs(closest['days_until_event']) <= cfg.EVENT_LOOKBACK_DAYS)

                        # Find closest FOMC event
                        fomc_events = [e for e in events if e['event_type'] == 'fomc']
                        if fomc_events:
                            closest = min(fomc_events, key=lambda e: abs(e['days_until_event']))
                            feat['days_until_fomc'] = closest['days_until_event']

                        # High impact = earnings/FOMC within 3 days
                        high_impact = [e for e in events
                                      if e['event_type'] in ['earnings', 'fomc', 'delivery']
                                      and abs(e['days_until_event']) <= 3]
                        feat['is_high_impact_event'] = float(len(high_impact) > 0)

                    daily_features[date_str] = feat
                except Exception:
                    daily_features[date_str] = {
                        'is_earnings_week': 0.0, 'days_until_earnings': 0.0,
                        'days_until_fomc': 0.0, 'is_high_impact_event': 0.0
                    }

            # Step 3: Map daily values to all 1-min bars (vectorized forward-fill)
            is_earnings_week = np.array([daily_features.get(d, {}).get('is_earnings_week', 0.0) for d in date_strings])
            days_until_earnings = np.array([daily_features.get(d, {}).get('days_until_earnings', 0.0) for d in date_strings])
            days_until_fomc = np.array([daily_features.get(d, {}).get('days_until_fomc', 0.0) for d in date_strings])
            is_high_impact_event = np.array([daily_features.get(d, {}).get('is_high_impact_event', 0.0) for d in date_strings])

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
        # v5.3.3: Expanded to all 11 TFs, count varies (60-95 depending on features present)
        # Legacy mode: ~72-95 features (all 11 TFs at 1-min)
        # Just log count, don't validate (count varies by TF availability)
        print(f"   ✓ Breakdown features: {num_breakdown} (legacy 1-min calculation)")

        # FIX: Remove temporary w50 columns from features_df to prevent saving to cache
        # These were only needed for breakdown calculation, not for final saved features
        if _temp_w50_cols_added:
            for col in _temp_w50_cols_added:
                if col in features_df.columns:
                    features_df.drop(columns=[col], inplace=True)

        return pd.DataFrame(breakdown_features, index=raw_df.index)

    def _calculate_breakdown_at_native_tf(
        self,
        resampled_df: pd.DataFrame,  # Already at native TF resolution
        tf: str,                     # Which timeframe ('5min', '1h', '3month', etc.)
        raw_df: pd.DataFrame = None, # Original 1-min OHLCV for event lookups
        events_handler = None
    ) -> pd.DataFrame:
        """
        Calculate breakdown features at NATIVE timeframe resolution (v5.3.3).

        This refactored version calculates breakdown AFTER resampling to native TF,
        ensuring train-test consistency. Training and live use same calculation.

        Key differences from _extract_breakdown_features:
        - Operates on NATIVE TF bars (e.g., 1500 5-min bars, not 7500 1-min bars)
        - Calculates only for THIS TF (not all 11 TFs)
        - Simpler, more semantically correct

        Args:
            resampled_df: Features already resampled to native TF
            tf: Timeframe name ('5min', '15min', ..., '3month')
            raw_df: Original OHLCV data (for event timestamp lookups)
            events_handler: Optional event handler

        Returns:
            DataFrame with breakdown features for THIS TF at native resolution
        """
        breakdown_features = {}
        num_rows = len(resampled_df)

        # Get adaptive window for THIS TF (in native TF bars, not 1-min!)
        window = config.ADAPTIVE_WINDOW_BARS_NATIVE.get(tf, 100)

        # 1. Duration ratio: current stability vs rolling historical average
        stability_col = f'tsla_channel_{tf}_w50_stability'
        if stability_col in resampled_df.columns:
            stability = resampled_df[stability_col]
            avg_stability = stability.rolling(window, min_periods=window//2).mean()
            duration_ratio = (stability / (avg_stability + 0.01)).fillna(1.0)
            breakdown_features[f'tsla_channel_duration_ratio_{tf}'] = np.asarray(duration_ratio)
        else:
            breakdown_features[f'tsla_channel_duration_ratio_{tf}'] = np.ones(num_rows)

        # 2. SPY-TSLA channel alignment
        tsla_pos_col = f'tsla_channel_{tf}_w50_position'
        spy_pos_col = f'spy_channel_{tf}_w50_position'
        if tsla_pos_col in resampled_df.columns and spy_pos_col in resampled_df.columns:
            tsla_pos = resampled_df[tsla_pos_col] * 2 - 1  # Convert 0-1 to -1 to +1
            spy_pos = resampled_df[spy_pos_col] * 2 - 1
            alignment = tsla_pos * spy_pos  # -1 to +1 (both aligned)
            breakdown_features[f'channel_alignment_spy_tsla_{tf}'] = np.asarray(alignment)
        else:
            breakdown_features[f'channel_alignment_spy_tsla_{tf}'] = np.zeros(num_rows)

        # 3. Time in channel (stability-based proxy)
        for symbol in ['tsla', 'spy']:
            stability_col = f'{symbol}_channel_{tf}_w50_stability'
            if stability_col in resampled_df.columns:
                stability = resampled_df[stability_col]
                time_in_channel = np.clip(stability * 100, 0, 100)  # 0-100 scale (np.clip returns ndarray)
                breakdown_features[f'{symbol}_time_in_channel_{tf}'] = np.asarray(time_in_channel)
            else:
                breakdown_features[f'{symbol}_time_in_channel_{tf}'] = np.zeros(num_rows)

        # 4. Normalized channel position (-1 to +1)
        for symbol in ['tsla', 'spy']:
            pos_col = f'{symbol}_channel_{tf}_w50_position'
            if pos_col in resampled_df.columns:
                position_norm = resampled_df[pos_col] * 2 - 1  # Convert 0-1 to -1 to +1
                breakdown_features[f'{symbol}_channel_position_norm_{tf}'] = np.asarray(position_norm)
            else:
                breakdown_features[f'{symbol}_channel_position_norm_{tf}'] = np.zeros(num_rows)

        # 5. Volume surge (if volume data available at this resolution)
        if 'tsla_volume' in resampled_df.columns and len(resampled_df) >= 20:
            volume = resampled_df['tsla_volume']
            recent_vol = volume.rolling(5, min_periods=1).mean()
            historical_vol = volume.rolling(20, min_periods=5).mean().shift(5)
            volume_surge = ((recent_vol - historical_vol) / (historical_vol + 1e-8)).fillna(0)
            breakdown_features['tsla_volume_surge'] = np.asarray(volume_surge)
        else:
            breakdown_features['tsla_volume_surge'] = np.zeros(num_rows)

        # 6. RSI divergence (for this TF only)
        rsi_col = f'tsla_rsi_{tf}'
        pos_col = f'tsla_channel_{tf}_w50_position'
        if rsi_col in resampled_df.columns and pos_col in resampled_df.columns:
            rsi_normalized = resampled_df[rsi_col] / 100.0  # 0-1 range
            channel_pos = resampled_df[pos_col]  # Already 0-1
            divergence = rsi_normalized - channel_pos  # -1 to +1
            breakdown_features[f'tsla_rsi_divergence_{tf}'] = np.asarray(divergence)
        else:
            breakdown_features[f'tsla_rsi_divergence_{tf}'] = np.zeros(num_rows)

        # 7. Day of week flags (use resampled index)
        breakdown_features['is_monday'] = (resampled_df.index.dayofweek == 0).astype(float)
        breakdown_features['is_tuesday'] = (resampled_df.index.dayofweek == 1).astype(float)
        breakdown_features['is_wednesday'] = (resampled_df.index.dayofweek == 2).astype(float)
        breakdown_features['is_thursday'] = (resampled_df.index.dayofweek == 3).astype(float)
        breakdown_features['is_friday'] = (resampled_df.index.dayofweek == 4).astype(float)

        # 8. Market timing flags (intraday only)
        if hasattr(resampled_df.index, 'hour') and len(resampled_df) > 0:
            hours = resampled_df.index.hour
            breakdown_features['is_first_hour'] = ((hours >= 9) & (hours < 11)).astype(float)
            breakdown_features['is_last_hour'] = ((hours >= 15) & (hours < 16)).astype(float)
        else:
            # Daily/weekly/monthly don't have hours
            breakdown_features['is_first_hour'] = np.zeros(num_rows)
            breakdown_features['is_last_hour'] = np.zeros(num_rows)

        # 9. Volatility regime flag
        if 'tsla_close' in resampled_df.columns and len(resampled_df) >= 50:
            tsla_close = resampled_df['tsla_close']
            current_vol = tsla_close.pct_change().rolling(10, min_periods=1).std()
            historical_vol = current_vol.rolling(50, min_periods=10).mean()
            is_volatile = (current_vol > 1.5 * historical_vol).fillna(False).astype(float)
            breakdown_features['is_volatile_now'] = np.asarray(is_volatile)
        else:
            breakdown_features['is_volatile_now'] = np.zeros(num_rows)

        # 10. In-channel binary flags (for THIS TF)
        for symbol in ['tsla', 'spy']:
            stability_col = f'{symbol}_channel_{tf}_w50_stability'
            if stability_col in resampled_df.columns:
                stability = resampled_df[stability_col]
                # In channel if stability > 5 (simplified threshold)
                in_channel = (stability > 5.0).astype(float)
                breakdown_features[f'{symbol}_in_channel_{tf}'] = np.asarray(in_channel)
            else:
                breakdown_features[f'{symbol}_in_channel_{tf}'] = np.zeros(num_rows)

        # 11. Event features (if handler provided and raw_df available)
        if events_handler and raw_df is not None:
            # For each resampled timestamp, lookup events
            # This is approximate - uses nearest timestamp matching
            try:
                for ts in resampled_df.index[:10]:  # Sample to test
                    # Find nearest in raw_df
                    if ts in raw_df.index:
                        # Could lookup event features here
                        pass
                # For now, set to zeros (events not critical, only 4 cols)
                breakdown_features['is_earnings_week'] = np.zeros(num_rows)
                breakdown_features['days_until_earnings'] = np.zeros(num_rows)
                breakdown_features['days_until_fomc'] = np.zeros(num_rows)
                breakdown_features['is_high_impact_event'] = np.zeros(num_rows)
            except:
                breakdown_features['is_earnings_week'] = np.zeros(num_rows)
                breakdown_features['days_until_earnings'] = np.zeros(num_rows)
                breakdown_features['days_until_fomc'] = np.zeros(num_rows)
                breakdown_features['is_high_impact_event'] = np.zeros(num_rows)
        else:
            breakdown_features['is_earnings_week'] = np.zeros(num_rows)
            breakdown_features['days_until_earnings'] = np.zeros(num_rows)
            breakdown_features['days_until_fomc'] = np.zeros(num_rows)
            breakdown_features['is_high_impact_event'] = np.zeros(num_rows)

        return pd.DataFrame(breakdown_features, index=resampled_df.index)

    def test_continuation_labels(self, df: pd.DataFrame) -> None:
        """Test continuation label generation on sample data."""
        print("Testing continuation label generation...")

        # Sample timestamps
        timestamps = df.index[:10]  # First 10 timestamps

        # Generate labels
        labels_df = self.generate_continuation_labels(df, timestamps, prediction_horizon=24, mode=config.CONTINUATION_MODE)
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

    def _get_channel_for_continuation_cached(
        self,
        ohlc_df: pd.DataFrame,
        timeframe: str,
        max_lookback: int,
        timestamp: pd.Timestamp
    ):
        """
        Get channel with 5-minute bucket caching for continuation labels (v3.19).

        Channels don't change significantly minute-to-minute, so cache per 5-min bucket.
        Typical reuse: 5-10 timestamps per bucket (10:00, 10:01, 10:02, 10:03, 10:04).
        Expected speedup: 10-20× vs recalculating every timestamp.

        Args:
            ohlc_df: OHLC DataFrame for channel calculation
            timeframe: '1h' or '4h'
            max_lookback: Maximum bars to look back
            timestamp: Current timestamp (for bucket key)

        Returns:
            ChannelData or None
        """
        # Create cache key: Round to 5-minute bucket
        bucket_minutes = timestamp.minute - (timestamp.minute % 5)
        bucket_ts = timestamp.replace(minute=bucket_minutes, second=0, microsecond=0)
        cache_key = (bucket_ts, timeframe, len(ohlc_df))

        # Check cache
        if cache_key in self._channel_continuation_cache:
            self._cache_hits += 1
            return self._channel_continuation_cache[cache_key]

        # Cache miss - calculate channel
        self._cache_misses += 1

        channel = self.channel_calc.find_best_channel_any_quality(
            ohlc_df,
            timeframe=timeframe,
            max_lookback=max_lookback
        )

        # Store in cache with simple size limit (LRU-style eviction)
        if len(self._channel_continuation_cache) > 1000:
            # Evict oldest entry (first in dict)
            oldest_key = next(iter(self._channel_continuation_cache))
            del self._channel_continuation_cache[oldest_key]

        self._channel_continuation_cache[cache_key] = channel

        return channel

    def generate_continuation_labels(self, df: pd.DataFrame, timestamps: list,
                                               prediction_horizon: int = 24, mode: str = None, debug: bool = False) -> pd.DataFrame:
        """
        Generate continuation prediction labels using multi-timeframe analysis.

        This is an optimized implementation using pre-resampling and parallelization.
        Legacy unoptimized version preserved in deprecated/continuation_labels_legacy.py

        Implementation optimizations:
        1. Pre-resample entire dataframe once (5-8x speedup)
        2. Parallelize timestamp processing (2-4x speedup)
        3. Batch future price windows (2x speedup)

        Total speedup: 20-60x faster while maintaining 100% identical mathematical results.
        Respects config.NUMPY_DTYPE for precision control (float32/float64).

        Args:
            df: Full OHLC DataFrame (1-min bars)
            timestamps: List of timestamps to process
            prediction_horizon: Base horizon for 'simple' mode, min horizon for 'adaptive'
            mode: 'simple' (fixed) or 'adaptive' (variable 24-48). If None, uses config.CONTINUATION_MODE
            debug: Enable debug logging

        Modes:
            - simple: Fixed prediction_horizon (default 24 bars = 24 minutes)
            - adaptive: Variable horizon (24-48 bars = 24-48 minutes) based on RSI/slope confidence
                       Neutral RSI (stable) → high conf → long horizon (continuation)
                       Extreme RSI (break likely) → low conf → short horizon (predict change soon)
                       Includes both continuation AND break labels for balanced training

        Returns:
            DataFrame with continuation labels
        """
        from joblib import Parallel, delayed
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)

        # Try importing RICH for better progress display
        try:
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
            from rich.console import Console
            RICH_AVAILABLE = True
        except ImportError:
            RICH_AVAILABLE = False

        # Determine mode (use config default if not specified)
        if mode is None:
            mode = config.CONTINUATION_MODE

        # OPTIMIZATION 1: Pre-resample entire dataframe once
        print("   Pre-resampling data for optimization...")
        one_h_full = df.resample('1h').agg({
            'tsla_open': 'first',
            'tsla_high': 'max',
            'tsla_low': 'min',
            'tsla_close': 'last'
        }).dropna()

        four_h_full = df.resample('4h').agg({
            'tsla_open': 'first',
            'tsla_high': 'max',
            'tsla_low': 'min',
            'tsla_close': 'last'
        }).dropna()

        # Rename columns immediately (same as original)
        one_h_full.columns = [c.replace('tsla_', '') for c in one_h_full.columns]
        four_h_full.columns = [c.replace('tsla_', '') for c in four_h_full.columns]

        # Cast to configured dtype for precision control
        one_h_full = one_h_full.astype(config.NUMPY_DTYPE)
        four_h_full = four_h_full.astype(config.NUMPY_DTYPE)

        # OPTIMIZATION 2: Pre-compute future price windows
        if 'adaptive' in mode:  # Matches 'adaptive_labels' and 'adaptive_full'
            max_horizon = config.ADAPTIVE_MAX_HORIZON
            print(f"   Pre-computing future price windows (adaptive mode, max={max_horizon} bars)...")
        else:
            max_horizon = prediction_horizon  # 24 bars default
            print("   Pre-computing future price windows...")

        future_windows = []
        for i in range(len(df)):
            end = min(i + max_horizon, len(df) - 1)
            future_windows.append(df.iloc[i:end+1]['tsla_close'].values.astype(config.NUMPY_DTYPE))

        # Calculate bar conversions
        bars_1h = config.CONTINUATION_LOOKBACK_1H // 60  # Convert 1-min to 1h bars
        bars_4h = config.CONTINUATION_LOOKBACK_4H // 240  # Convert 1-min to 4h bars

        def reconstruct_with_partial(resampled_full, raw_df, ts, freq):
            """
            Reconstruct resampled data with partial bin for unaligned timestamp.
            This ensures 100% identical results to original resampling.
            """
            if freq == '1h':
                is_aligned = ts.minute == 0 and ts.second == 0
                lookback_bars = bars_1h
            else:  # 4h
                is_aligned = (ts.hour % 4 == 0) and ts.minute == 0 and ts.second == 0
                lookback_bars = bars_4h

            if is_aligned:
                # Timestamp is aligned, can use pre-resampled directly
                try:
                    idx = resampled_full.index.get_indexer([ts], method='ffill')[0]
                    if idx < 0:
                        return None
                    start_idx = max(0, idx - lookback_bars + 1)
                    return resampled_full.iloc[start_idx:idx+1].copy()
                except:
                    return None
            else:
                # Unaligned - need to reconstruct partial bin
                # Get the floor timestamp for this frequency
                if freq == '1h':
                    ts_floor = ts.floor('H')
                else:  # 4h
                    ts_floor = ts.floor('4H')

                # Get complete bins before current period
                try:
                    # Find where the floored timestamp would be
                    floor_idx = resampled_full.index.get_indexer([ts_floor], method='ffill')[0]
                    if floor_idx <= 0:
                        # Not enough history
                        return None

                    # Get previous complete bins
                    start_idx = max(0, floor_idx - lookback_bars + 1)
                    complete_bins = resampled_full.iloc[start_idx:floor_idx].copy()

                    # Reconstruct the partial bin
                    partial_data = raw_df.loc[ts_floor:ts]
                    if len(partial_data) == 0:
                        return None

                    partial_bin = pd.DataFrame({
                        'open': [partial_data.iloc[0]['tsla_open']],
                        'high': [partial_data['tsla_high'].max()],
                        'low': [partial_data['tsla_low'].min()],
                        'close': [partial_data.iloc[-1]['tsla_close']]
                    }, index=[ts_floor], dtype=config.NUMPY_DTYPE)

                    # Combine complete bins with partial bin
                    return pd.concat([complete_bins, partial_bin])
                except Exception:
                    return None

        def process_single_timestamp(ts_idx_tuple):
            """Process a single timestamp. Designed for parallel execution."""
            ts, idx = ts_idx_tuple

            try:
                current_price = df.loc[ts, 'tsla_close']

                # Reconstruct resampled data with partial bins if needed (Solution C)
                one_h_ohlc = reconstruct_with_partial(one_h_full, df, ts, '1h')
                four_h_ohlc = reconstruct_with_partial(four_h_full, df, ts, '4h')

                if one_h_ohlc is None or four_h_ohlc is None:
                    return None

                if len(one_h_ohlc) < 3 or len(four_h_ohlc) < 2:
                    return None

                # Calculate RSI (must be done on reconstructed data, not pre-cached)
                rsi_1h = self.rsi_calc.get_rsi_data(one_h_ohlc).value or 50.0
                rsi_4h = self.rsi_calc.get_rsi_data(four_h_ohlc).value or 50.0

                # Fit channels - v3.19: Cached channels (5-min buckets, 10-20× speedup)
                # "Bad" channels signal that THIS timeframe is unreliable right now
                # Model learns when to switch timeframes based on quality scores
                channel_1h = self._get_channel_for_continuation_cached(
                    one_h_ohlc,
                    timeframe='1h',
                    max_lookback=min(60, max(5, len(one_h_ohlc)-2)),
                    timestamp=ts
                )

                channel_4h = self._get_channel_for_continuation_cached(
                    four_h_ohlc,
                    timeframe='4h',
                    max_lookback=min(120, max(10, len(four_h_ohlc)-2)),
                    timestamp=ts
                )

                # Only skip if truly no data (very rare)
                if channel_1h is None or channel_4h is None:
                    return None  # Insufficient data, not quality issue

                # Get slopes
                slope_1h = channel_1h.slope if channel_1h else 0.0
                slope_4h = channel_4h.slope if channel_4h else 0.0

                # Apply scoring logic (identical to original)
                score = 0

                if rsi_1h < 40:
                    score += 1

                if rsi_4h < 40:
                    score += 1

                slope_1h_direction = 1 if slope_1h > 0.0001 else (-1 if slope_1h < -0.0001 else 0)
                slope_4h_direction = 1 if slope_4h > 0.0001 else (-1 if slope_4h < -0.0001 else 0)

                if slope_1h_direction == slope_4h_direction and slope_1h_direction != 0:
                    score += 1
                elif slope_1h_direction != slope_4h_direction and slope_1h_direction != 0 and slope_4h_direction != 0:
                    score -= 1

                if rsi_4h > 70:
                    score -= 1

                # Calculate adaptive horizon if in adaptive mode
                if 'adaptive' in mode:  # Matches 'adaptive_labels' and 'adaptive_full'
                    # Calculate confidence components (using configured dtype)
                    rsi_conf_1h = np.array(1.0 - abs(rsi_1h - 50) / 50, dtype=config.NUMPY_DTYPE)
                    rsi_conf_4h = np.array(1.0 - abs(rsi_4h - 50) / 50, dtype=config.NUMPY_DTYPE)

                    # Use already calculated slope directions for alignment
                    slope_alignment = np.array(1.0 if (slope_1h_direction == slope_4h_direction and slope_1h_direction != 0) else 0.0,
                                               dtype=config.NUMPY_DTYPE)

                    # RSI confidence: High for neutral (stable continuation, long horizon)
                    #                 Low for extreme (likely break/reversal, short horizon)
                    # This lets model predict near-term when breaks expected, far ahead when stable
                    # Combined confidence score (0-1 range)
                    conf_score = float((rsi_conf_1h + rsi_conf_4h + slope_alignment) / 3.0)

                    # Adaptive horizon: 24-48 bars based on confidence
                    adaptive_horizon = int(config.ADAPTIVE_MIN_HORIZON +
                                          (config.ADAPTIVE_MAX_HORIZON - config.ADAPTIVE_MIN_HORIZON) * conf_score)

                    # Use adaptive slice of pre-computed window
                    future_prices = future_windows[idx][:adaptive_horizon]
                else:
                    # Simple mode - use fixed horizon
                    adaptive_horizon = prediction_horizon
                    conf_score = None
                    future_prices = future_windows[idx][:prediction_horizon]

                if len(future_prices) < 2:
                    return None

                # Calculate metrics for upside breaks
                future_high = np.max(future_prices)
                future_low = np.min(future_prices)
                max_gain = (future_high - current_price) / current_price * 100

                # Upside break threshold (+2%)
                break_threshold = current_price * 1.02
                break_indices = np.where(future_prices >= break_threshold)[0]

                # Downside break threshold (-2%)
                downside_threshold = current_price * 0.98
                downside_break_indices = np.where(future_prices <= downside_threshold)[0]

                # Determine if this is a break or continuation
                # Low score OR no upside break within horizon = treat as break/range
                is_break = (score <= 1) or (len(break_indices) == 0)

                if is_break:
                    # Label as break - include with downside metrics
                    continues = False

                    if len(downside_break_indices) > 0:
                        # Found downside break
                        break_idx = downside_break_indices[0]
                        actual_duration_hours = break_idx * (1/60)  # 1-min bars to hours
                        max_gain = (future_prices[break_idx] - current_price) / current_price * 100  # Negative
                        label = f"breaks down in {actual_duration_hours:.1f}h, {max_gain:.1f}%"
                    else:
                        # No clear break, ranging/choppy
                        actual_duration_hours = len(future_prices) * (1/60)  # 1-min bars to hours
                        max_gain = (future_high - current_price) / current_price * 100  # Small gain
                        label = f"ranges {actual_duration_hours:.1f}h, {max_gain:.1f}%"

                    # Force short horizon and low confidence for breaks
                    if 'adaptive' in mode:  # Matches 'adaptive_labels' and 'adaptive_full'
                        adaptive_horizon = config.ADAPTIVE_MIN_HORIZON  # 24 bars (short)
                        conf_score = 0.0  # Low confidence
                else:
                    # Continuation - upside break detected
                    continues = True
                    break_idx = break_indices[0]
                    actual_duration_hours = break_idx * (1/60)  # 1-min bars to hours
                    label = f"continues {actual_duration_hours:.1f}h, +{max_gain:.1f}%"

                confidence = min(max(abs(score) * 0.2, 0.1), 0.9)

                return {
                    'timestamp': ts,
                    'label': label,
                    'continues': float(continues),
                    'duration_hours': actual_duration_hours,
                    'projected_gain': max_gain,  # Can be negative for breaks
                    'confidence': confidence,
                    'score': score,
                    'rsi_1h': rsi_1h,
                    'rsi_4h': rsi_4h,
                    'slope_1h': slope_1h,
                    'slope_4h': slope_4h,
                    # v3.17: Channel quality signals for timeframe switching
                    'channel_1h_cycles': channel_1h.complete_cycles if channel_1h else 0,
                    'channel_4h_cycles': channel_4h.complete_cycles if channel_4h else 0,
                    'channel_1h_r_squared': channel_1h.r_squared if channel_1h else 0.0,
                    'channel_4h_r_squared': channel_4h.r_squared if channel_4h else 0.0,
                    'channel_1h_valid': channel_1h.is_valid if channel_1h else 0.0,
                    'channel_4h_valid': channel_4h.is_valid if channel_4h else 0.0,
                    # Adaptive mode fields (None for simple mode)
                    'adaptive_horizon': adaptive_horizon,
                    'conf_score': conf_score
                }

            except Exception as e:
                if debug:
                    print(f"Error processing {ts}: {e}")
                return None

        # OPTIMIZATION 3: Parallel processing with RICH progress display
        # Create tuples of (timestamp, index) for processing
        ts_idx_pairs = [(ts, df.index.get_loc(ts)) for ts in timestamps]

        if RICH_AVAILABLE:
            # Use RICH for beautiful progress display (matches channel extraction UI)
            print(f"   🎨 Processing {len(timestamps):,} timestamps with RICH progress...")

            console = Console()
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed:,}/{task.total:,})"),
                TimeRemainingColumn(),
                console=console,
                expand=True,
                refresh_per_second=10
            ) as progress:

                task_id = progress.add_task(
                    "[cyan]Continuation labels (parallel threading)",
                    total=len(ts_idx_pairs)
                )

                # Thread pool executor for parallel processing
                # v3.19: Use dict-of-lists for efficient DataFrame construction (50-300× faster, 80% less memory)
                label_data = defaultdict(list)
                completed_count = 0

                with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    # Submit all tasks
                    futures = {executor.submit(process_single_timestamp, ts_idx): i
                              for i, ts_idx in enumerate(ts_idx_pairs)}

                    # Collect results as they complete with progress updates
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            if result is not None:
                                # Append to dict-of-lists (efficient column-wise construction)
                                for key, value in result.items():
                                    label_data[key].append(value)
                                completed_count += 1
                        except Exception as e:
                            # Thread failed - log error and continue with other timestamps
                            if debug:
                                print(f"\n   ⚠️  Timestamp failed: {e}")
                        progress.update(task_id, advance=1)

        else:
            # Fallback to tqdm if RICH not available
            print(f"   Processing {len(timestamps):,} timestamps in parallel...")
            from tqdm import tqdm

            # Use threading backend for better compatibility on macOS
            results = Parallel(n_jobs=-1, backend='threading', verbose=0)(
                delayed(process_single_timestamp)(ts_idx)
                for ts_idx in tqdm(ts_idx_pairs, desc="   Continuation labels",
                                  unit="timestamps", ncols=100, leave=False, ascii=True)
            )

            # v3.19: Convert list-of-dicts to dict-of-lists (efficient DataFrame construction)
            label_data = defaultdict(list)
            for result in results:
                if result is not None:
                    for key, value in result.items():
                        label_data[key].append(value)

        if debug:
            label_count = len(label_data['timestamp']) if 'timestamp' in label_data else 0
            print(f"   Generated {label_count} labels out of {len(timestamps)} timestamps")

        # v3.19: Log channel cache performance
        if self._cache_hits + self._cache_misses > 0:
            total = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total * 100 if total > 0 else 0
            speedup_estimate = total / self._cache_misses if self._cache_misses > 0 else 1.0

            print(f"\n   📊 Channel Cache Performance:")
            print(f"      Cache hits: {self._cache_hits:,} ({hit_rate:.1f}%)")
            print(f"      Cache misses: {self._cache_misses:,}")
            print(f"      Estimated speedup: {speedup_estimate:.1f}× vs no cache")

        # v3.19: Efficient DataFrame construction from dict-of-lists
        print("   🔍 DEBUG: Building DataFrame from dict-of-lists (optimized)...")
        df_result = pd.DataFrame(label_data)
        print(f"   🔍 DEBUG: DataFrame built: {len(df_result)} rows × {len(df_result.columns)} columns")
        return df_result

    def generate_hierarchical_continuation_labels(
        self,
        df: pd.DataFrame,
        timeframes: list = None,
        output_dir: Path = None,
        cache_suffix: str = None
    ) -> Dict[str, Path]:
        """
        Generate continuation labels for all hierarchical timeframes (v4.3).

        Uses CHANNEL STRUCTURE to detect breaks (not arbitrary thresholds).
        The channel's ±2σ deviation lines already reflect VIX/volatility/market conditions.

        Key design decisions:
        1. Break = price closes outside channel's ±2σ bounds (data-driven, not hardcoded %)
        2. Prediction horizon = unlimited (scan until break OR end of data)
        3. Each TF gets its own labels at native resolution
        4. Model learns WHEN breaks happen from labels, learns WHY from features

        Args:
            df: 1-minute OHLC data with columns: tsla_open, tsla_high, tsla_low, tsla_close
            timeframes: List of timeframes (defaults to HIERARCHICAL_TIMEFRAMES)
            output_dir: Where to save per-TF label files
            cache_suffix: Cache key suffix for file naming

        Returns:
            Dict mapping timeframe -> file path of saved labels
        """
        from scipy import stats

        if timeframes is None:
            timeframes = HIERARCHICAL_TIMEFRAMES
        if output_dir is None:
            output_dir = Path('data/feature_cache')
        if cache_suffix is None:
            # v5.0: FEATURE_VERSION already starts with 'v', don't add another
            cache_suffix = f"{FEATURE_VERSION}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}

        print(f"\n   🔄 Generating hierarchical continuation labels for {len(timeframes)} timeframes...")
        print(f"      Break detection: Channel structure (±2σ lines) - no hardcoded thresholds")
        print(f"      Prediction horizon: Unlimited (scan until break)")

        # Process each timeframe
        for tf in timeframes:
            print(f"\n      Processing {tf} timeframe...")

            # Resample to this TF's native resolution
            tf_rule = TIMEFRAME_RESAMPLE_RULES[tf]
            tf_data = df.resample(tf_rule).agg({
                'tsla_open': 'first',
                'tsla_high': 'max',
                'tsla_low': 'min',
                'tsla_close': 'last'
            }).dropna()

            print(f"         Resampled to {len(tf_data):,} {tf} bars")

            # Get lookback for this TF
            lookback_bars = TIMEFRAME_SEQUENCE_LENGTHS[tf]

            if len(tf_data) < lookback_bars + 10:
                print(f"         ⚠️  Not enough data for {tf} (need {lookback_bars}+ bars, have {len(tf_data)})")
                continue

            # Generate labels
            tf_labels = []

            for i in tqdm(range(lookback_bars, len(tf_data) - 1),
                         desc=f"         {tf} labels", leave=False, ncols=100, ascii=True):
                # Lookback window for channel fitting
                lookback = tf_data.iloc[i - lookback_bars:i]

                # Fit linear regression channel on lookback
                try:
                    x = np.arange(lookback_bars)
                    closes = lookback['tsla_close'].values
                    highs = lookback['tsla_high'].values
                    lows = lookback['tsla_low'].values

                    # Use scipy for linear regression
                    slope, intercept, r_value, _, _ = stats.linregress(x, closes)
                    r_squared = r_value ** 2

                    # Calculate residuals and std dev
                    predicted = slope * x + intercept
                    residuals = closes - predicted
                    residual_std = np.std(residuals)

                    if residual_std < 1e-10:
                        # Degenerate channel (flat line)
                        continue

                    # Calculate channel metrics
                    upper_line = predicted + (2.0 * residual_std)
                    lower_line = predicted - (2.0 * residual_std)

                    # Simple cycle counting: alternating touches of upper/lower bands
                    touches_upper = closes >= (predicted + 1.5 * residual_std)
                    touches_lower = closes <= (predicted - 1.5 * residual_std)

                    complete_cycles = 0
                    last_touch = None
                    for j in range(len(closes)):
                        if touches_upper[j] and last_touch != 'upper':
                            if last_touch == 'lower':
                                complete_cycles += 1
                            last_touch = 'upper'
                        elif touches_lower[j] and last_touch != 'lower':
                            if last_touch == 'upper':
                                complete_cycles += 1
                            last_touch = 'lower'

                    is_valid = complete_cycles >= 2 and r_squared > 0.5

                except Exception:
                    continue

                # Get current price and timestamp
                current_price = tf_data.iloc[i]['tsla_close']
                current_timestamp = tf_data.index[i]

                # Future window (unlimited - scan until break OR end of data)
                future = tf_data.iloc[i:]

                # Detect continuation vs break using CHANNEL STRUCTURE
                break_idx, max_gain = self._detect_channel_break_structure(
                    slope=slope,
                    intercept=intercept,
                    residual_std=residual_std,
                    lookback_bars=lookback_bars,
                    future_prices=future,
                    current_price=current_price
                )

                # Calculate duration (in bars)
                duration_bars = break_idx if break_idx is not None else (len(future) - 1)

                # Calculate confidence based on channel quality
                confidence = self._calculate_continuation_confidence(
                    r_squared=r_squared,
                    cycles=complete_cycles,
                    is_valid=is_valid
                )

                tf_labels.append({
                    'timestamp': current_timestamp,
                    'duration_bars': float(duration_bars),
                    'max_gain_pct': float(max_gain),
                    'confidence': float(confidence),
                    'channel_cycles': int(complete_cycles),
                    'channel_r_squared': float(r_squared),
                    'channel_valid': int(is_valid),
                    'channel_slope': float(slope),
                    'channel_width': float(residual_std * 4)  # Full width (±2σ)
                })

            if len(tf_labels) == 0:
                print(f"         ⚠️  No valid labels generated for {tf}")
                continue

            # Convert to DataFrame
            labels_df = pd.DataFrame(tf_labels)
            labels_df.set_index('timestamp', inplace=True)

            # Save to file
            output_path = output_dir / f"continuation_labels_{tf}_{cache_suffix}.pkl"
            labels_df.to_pickle(output_path)

            saved_files[tf] = output_path

            # Stats
            avg_duration = labels_df['duration_bars'].mean()
            avg_confidence = labels_df['confidence'].mean()
            print(f"         ✓ Saved {len(labels_df):,} labels | avg duration: {avg_duration:.1f} bars | avg conf: {avg_confidence:.2f}")

        print(f"\n   ✓ Generated {len(saved_files)}/{len(timeframes)} timeframe continuation labels")
        return saved_files

    def _detect_channel_break_structure(
        self,
        slope: float,
        intercept: float,
        residual_std: float,
        lookback_bars: int,
        future_prices: pd.DataFrame,
        current_price: float
    ) -> Tuple[int, float]:
        """
        Detect when a channel breaks by scanning forward.
        Uses CHANNEL STRUCTURE (±2σ deviation lines) not arbitrary thresholds.

        Args:
            slope: Channel slope from linear regression
            intercept: Channel intercept
            residual_std: Standard deviation of residuals (defines channel width)
            lookback_bars: Number of bars in the lookback window
            future_prices: DataFrame of future OHLC bars
            current_price: Price at prediction point

        Returns:
            break_idx: Index of break (or None if no break)
            max_gain: Maximum % gain before break
        """
        max_gain = 0.0

        for i in range(len(future_prices)):
            row = future_prices.iloc[i]
            close = row['tsla_close']
            high = row['tsla_high']

            # Track maximum gain
            gain_pct = (high - current_price) / current_price * 100
            max_gain = max(max_gain, abs(gain_pct))

            # Project channel forward to this bar
            # At bar i (relative to prediction point), we're at x = lookback_bars + i
            x_current = lookback_bars + i
            center_at_x = slope * x_current + intercept
            upper_at_x = center_at_x + (2.0 * residual_std)
            lower_at_x = center_at_x - (2.0 * residual_std)

            # Method 1: Price CLOSES outside channel bounds (primary break detection)
            if close > upper_at_x or close < lower_at_x:
                return i, max_gain

        return None, max_gain

    def _calculate_continuation_confidence(
        self,
        r_squared: float,
        cycles: int,
        is_valid: bool
    ) -> float:
        """
        Calculate confidence score based on channel quality.

        Returns: Float 0.0-1.0
        """
        if not is_valid:
            return 0.1

        # Base on r_squared (0.0-1.0) → contributes 0-0.7
        conf = r_squared * 0.7

        # Bonus for complete cycles (up to +0.3)
        # 5+ cycles = maximum bonus
        cycle_bonus = min(cycles / 5.0, 1.0) * 0.3

        return min(conf + cycle_bonus, 0.99)

    def generate_transition_labels(
        self,
        continuation_labels_dir: Path,
        output_dir: Path = None,
        cache_suffix: str = None
    ) -> Dict[str, Path]:
        """
        Generate transition labels for v5.2 Multi-Phase Compositor.

        USES existing hierarchical continuation labels (duration_bars, channel_slope).
        ADDS transition classification: what happens AFTER each channel breaks?

        Transition Types:
            CONTINUE (0): Same channel extends (didn't break within lookahead)
            SWITCH_TF (1): Different TF's channel takes over (has higher quality)
            REVERSE (2): Same TF, opposite direction (bull→bear or vice versa)
            SIDEWAYS (3): Same TF, enters consolidation

        Args:
            continuation_labels_dir: Directory with existing continuation_labels_{tf}_*.pkl files
            output_dir: Where to save transition labels (defaults to same as input)
            cache_suffix: Cache suffix for file naming (if None, extracts from continuation labels)

        Returns:
            Dict mapping timeframe -> transition labels file path
        """
        from pathlib import Path

        # Constants
        TRANSITION_CONTINUE = 0
        TRANSITION_SWITCH_TF = 1
        TRANSITION_REVERSE = 2
        TRANSITION_SIDEWAYS = 3

        SLOPE_THRESHOLD = 0.0005  # Below this absolute value = sideways

        def slope_to_direction(slope: float) -> int:
            """Convert slope to direction: 0=BULL, 1=BEAR, 2=SIDEWAYS"""
            if slope > SLOPE_THRESHOLD:
                return 0  # BULL
            elif slope < -SLOPE_THRESHOLD:
                return 1  # BEAR
            else:
                return 2  # SIDEWAYS

        continuation_labels_dir = Path(continuation_labels_dir)
        if output_dir is None:
            output_dir = continuation_labels_dir
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        print(f"      Input: {continuation_labels_dir}")
        print(f"      Output: {output_dir}")

        # Load all TF continuation labels for cross-TF quality comparison
        all_tf_labels = {}
        for tf in HIERARCHICAL_TIMEFRAMES:
            cont_files = list(continuation_labels_dir.glob(f"continuation_labels_{tf}_*.pkl"))
            if cont_files:
                all_tf_labels[tf] = pd.read_pickle(cont_files[0])
                print(f"      Loaded {tf}: {len(all_tf_labels[tf]):,} rows")

        if len(all_tf_labels) == 0:
            print("      ⚠️  No continuation labels found!")
            return {}

        # Process each timeframe
        for current_tf in HIERARCHICAL_TIMEFRAMES:
            if current_tf not in all_tf_labels:
                continue

            cont_labels = all_tf_labels[current_tf]
            transition_labels = []

            print(f"\n      Processing {current_tf}...")

            for i in tqdm(range(len(cont_labels) - 20), desc=f"         {current_tf}", leave=False, ncols=100):
                row = cont_labels.iloc[i]
                timestamp = row.name if hasattr(row, 'name') else cont_labels.index[i]
                duration = int(row['duration_bars'])
                current_slope = row['channel_slope']
                current_r_squared = row.get('channel_r_squared', 0.5)
                current_direction = slope_to_direction(current_slope)

                # Look at what happened after the channel ended
                future_idx = i + max(1, duration) + 5  # Look 5 bars after break

                if future_idx >= len(cont_labels):
                    # Not enough future data
                    transition_labels.append({
                        'timestamp': timestamp,
                        'duration_bars': duration,
                        'transition_type': TRANSITION_CONTINUE,
                        'switch_to_tf': None,
                        'current_direction': current_direction,
                        'new_direction': current_direction,
                        'new_slope': current_slope,
                        'current_r_squared': current_r_squared,
                    })
                    continue

                future_row = cont_labels.iloc[future_idx]
                future_slope = future_row['channel_slope']
                future_direction = slope_to_direction(future_slope)
                future_r_squared = future_row.get('channel_r_squared', 0.5)

                # Determine transition type
                transition_type = TRANSITION_CONTINUE
                switch_to_tf = None

                # Check 1: Did direction change in same TF?
                if current_direction != future_direction:
                    if future_direction == 2:  # SIDEWAYS
                        transition_type = TRANSITION_SIDEWAYS
                    else:
                        transition_type = TRANSITION_REVERSE

                # Check 2: Did a different TF take over (higher quality)?
                # Only check if we didn't already detect a direction change
                if transition_type == TRANSITION_CONTINUE:
                    # Find TF with highest quality at the break point
                    best_tf = current_tf
                    best_quality = future_r_squared

                    for other_tf, other_labels in all_tf_labels.items():
                        if other_tf == current_tf:
                            continue

                        # Find the row in other TF closest to our timestamp
                        try:
                            # Get the index in the other TF's labels
                            other_idx = other_labels.index.get_indexer(
                                [timestamp], method='nearest'
                            )[0]

                            if other_idx >= 0 and other_idx < len(other_labels):
                                other_quality = other_labels.iloc[other_idx].get('channel_r_squared', 0)
                                if other_quality > best_quality + 0.1:  # 0.1 threshold
                                    best_quality = other_quality
                                    best_tf = other_tf
                        except Exception:
                            continue

                    if best_tf != current_tf:
                        transition_type = TRANSITION_SWITCH_TF
                        switch_to_tf = best_tf

                transition_labels.append({
                    'timestamp': timestamp,
                    'duration_bars': duration,
                    'transition_type': transition_type,
                    'switch_to_tf': switch_to_tf,
                    'current_direction': current_direction,
                    'new_direction': future_direction,
                    'new_slope': future_slope,
                    'current_r_squared': current_r_squared,
                })

            if len(transition_labels) == 0:
                print(f"         ⚠️  No transition labels generated for {current_tf}")
                continue

            # Convert to DataFrame and save
            labels_df = pd.DataFrame(transition_labels)
            labels_df.set_index('timestamp', inplace=True)

            # Use provided cache_suffix, or extract from existing continuation labels
            effective_suffix = cache_suffix
            if effective_suffix is None:
                cont_files = list(continuation_labels_dir.glob(f"continuation_labels_{current_tf}_*.pkl"))
                if cont_files:
                    # Extract suffix from existing file name
                    existing_name = cont_files[0].stem
                    effective_suffix = existing_name.replace(f"continuation_labels_{current_tf}_", "")
                else:
                    effective_suffix = "v5.2"

            output_path = output_dir / f"transition_labels_{current_tf}_{effective_suffix}.pkl"
            labels_df.to_pickle(output_path)
            saved_files[current_tf] = output_path

            # Stats
            type_counts = labels_df['transition_type'].value_counts().to_dict()
            type_names = {0: 'CONTINUE', 1: 'SWITCH_TF', 2: 'REVERSE', 3: 'SIDEWAYS'}
            print(f"         ✓ Saved {len(labels_df):,} labels to {output_path.name}")
            for t_type, count in sorted(type_counts.items()):
                pct = 100 * count / len(labels_df)
                print(f"           {type_names.get(t_type, t_type)}: {count:,} ({pct:.1f}%)")

        print(f"\n   ✓ Generated transition labels for {len(saved_files)}/{len(HIERARCHICAL_TIMEFRAMES)} timeframes")
        return saved_files

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
